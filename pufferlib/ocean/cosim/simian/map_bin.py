"""Simian road graph (sent by simian-adapter) -> map-only PufferDrive bin.

Road graph schema, all coordinates in the Simian map frame [m]:
    lanes:          [{id, type, points: [[x, y, z]], widths: [w], speed_limit_mps, successors: [id]}]
                    type is a PufferDrive lane type (constants.h, 0-9)
    edges:          [{points: [[x, y, z]]}]
    traffic_lights: [{id, lane_id, stop_line: [[x, y, z], [x, y, z]]}], one entry per (light, controlled lane)

The bin frame is the map frame translated by `origin` (float32 bins lose cm precision at UTM magnitudes).
"""

import heapq
from pathlib import Path

import numpy as np

import data_utils.mirror_map_bin as mbin
from pufferlib.ocean.cosim.nuplan_bridge import LIGHT_STATE_GREEN, ROAD_EDGE_BOUNDARY, TRAFFIC_TYPE_LIGHT
from pufferlib.ocean.cosim.nuplan_bridge import _polyline_heading

MIN_POLYLINE_POINTS = 2
UNREACHABLE_LANE_DISTANCE = -1.0  # C treats negative/non-finite lane-graph distances as unreachable
NO_TRAFFIC_PHASE = -1


class SimianMap:
    """Bin-frame lookup tables of a written map: lane index by Simian lane id, light ids per traffic element."""

    def __init__(self, lane_idx_by_id, element_light_ids, origin):
        self.lane_idx_by_id = lane_idx_by_id
        self.element_light_ids = element_light_ids
        self.origin = origin

    @property
    def num_traffic(self):
        return len(self.element_light_ids)

    def to_bin(self, x, y):
        return x - self.origin[0], y - self.origin[1]

    def from_bin(self, x, y):
        return x + self.origin[0], y + self.origin[1]

    def heading_to_bin(self, heading_rad):
        return heading_rad

    def heading_from_bin(self, heading_rad):
        return heading_rad

    def velocity_to_bin(self, vx, vy):
        return vx, vy


def _points_bin(points, origin):
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return pts - (origin[0], origin[1], 0.0)


def _road(road_idx, road_type, pts):
    return {
        "id": road_idx,
        "type": road_type,
        "S": len(pts),
        "x": tuple(pts[:, 0].astype(np.float32).tolist()),
        "y": tuple(pts[:, 1].astype(np.float32).tolist()),
        "z": tuple(pts[:, 2].astype(np.float32).tolist()),
        "headings": tuple(_polyline_heading(pts[:, :2]).tolist()),
    }


def _lane_graph(lane_roads):
    """distances[i, g]: shortest drive from the start of lane i to the start of lane g (lane lengths summed)."""
    lane_count = len(lane_roads)
    distances = np.full((lane_count, lane_count), UNREACHABLE_LANE_DISTANCE, dtype=np.float64)
    for source_idx in range(lane_count):
        settled = distances[source_idx]
        frontier = [(0.0, source_idx)]
        for _ in range(lane_count * lane_count + 1):
            if not frontier:
                break
            distance_m, lane_idx = heapq.heappop(frontier)
            if settled[lane_idx] >= 0.0:
                continue
            settled[lane_idx] = distance_m
            next_distance_m = distance_m + lane_roads[lane_idx]["length"]
            for exit_idx in lane_roads[lane_idx]["exit_lanes"]:
                if settled[exit_idx] < 0.0:
                    heapq.heappush(frontier, (next_distance_m, exit_idx))
    return {
        "n": lane_count,
        "lane_ids": tuple(range(lane_count)),
        "distances": tuple(distances.astype(np.float32).reshape(-1).tolist()),
    }


def write_simian_bin(road_graph, origin, out_path: Path, light_state_steps):
    """Write the bin and return its SimianMap. light_state_steps sizes each light's state schedule,
    which set_traffic_light_states indexes by env tick."""
    lanes = [lane for lane in road_graph["lanes"] if len(lane["points"]) >= MIN_POLYLINE_POINTS]
    if not lanes:
        raise ValueError("road graph has no lanes with at least two points")
    lane_idx_by_id = {lane["id"]: idx for idx, lane in enumerate(lanes)}
    predecessors = {lane["id"]: [] for lane in lanes}
    for lane in lanes:
        for successor_id in lane["successors"]:
            if successor_id in predecessors:
                predecessors[successor_id].append(lane_idx_by_id[lane["id"]])

    roads = []
    for lane_idx, lane in enumerate(lanes):
        pts = _points_bin(lane["points"], origin)
        widths = np.asarray(lane["widths"], dtype=np.float32)
        if len(widths) != len(pts):
            raise ValueError(f"lane {lane['id']}: {len(widths)} widths for {len(pts)} points")
        cum = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(pts[:, :2], axis=0).T))])
        road = _road(lane_idx, int(lane["type"]), pts)
        road["entry_lanes"] = tuple(sorted(predecessors[lane["id"]]))
        road["exit_lanes"] = tuple(sorted(lane_idx_by_id[s] for s in lane["successors"] if s in lane_idx_by_id))
        road["speed_limit"] = float(lane["speed_limit_mps"])
        road["length"] = float(cum[-1])
        road["cum_lengths"] = tuple(cum.astype(np.float32).tolist())
        road["widths"] = tuple(widths.tolist())
        roads.append(road)
    lane_graph = _lane_graph(roads)

    for edge in road_graph["edges"]:
        if len(edge["points"]) < MIN_POLYLINE_POINTS:
            continue
        roads.append(_road(len(roads), ROAD_EDGE_BOUNDARY, _points_bin(edge["points"], origin)))

    traffic, element_light_ids = [], []
    for light in road_graph["traffic_lights"]:
        lane_idx = lane_idx_by_id.get(light["lane_id"])
        if lane_idx is None:
            continue
        (x1, y1, z1), (x2, y2, z2) = _points_bin(light["stop_line"], origin)
        lane_heading = roads[lane_idx]["headings"][0]
        traffic.append(
            {
                "id": len(traffic),
                "type": TRAFFIC_TYPE_LIGHT,
                "stop_line": (x1, y1, z1, x2, y2, z2),
                "heading": lane_heading,
                "states": (LIGHT_STATE_GREEN,) * light_state_steps,
                "controlled_lanes": (lane_idx,),
                "junction_id": NO_TRAFFIC_PHASE,
                "phase_idx": NO_TRAFFIC_PHASE,
            }
        )
        element_light_ids.append(light["id"])

    data = {
        "agents": [],
        "roads": roads,
        "traffic": traffic,
        "objects": [],
        "lane_graph": lane_graph,
        "scenario_id": b"simian".ljust(128, b"\0"),
        "dataset_name": b"simian".ljust(32, b"\0"),
        "log_length": 0,
        "log_dt": 0.0,
        "objects_of_interest": (),
        "tracks_to_predict": (),
        "has_phase_section": True,  # the loader reads TLPHASE1 before LANEWID1, so widths need it present
        "has_width_section": True,
        "has_zone_section": False,
        "has_scripted_goal_section": False,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mbin.write_bin(data, out_path)
    return SimianMap(lane_idx_by_id, element_light_ids, origin)
