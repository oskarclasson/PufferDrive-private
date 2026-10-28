"""Turn a map-editor spec (JSON, local metric frame) into a map-only PufferDrive .bin shaped like the CARLA bins.

Spec schema (all coordinates in metres, x east / y north of the spec origin):
    lanes:    [{id, points: [[x, y]], width_m, speed_limit_kph, successors: [lane id]}]
    edges:    [{points: [[x, y]]}]
    controls: [{type: "light" | "stop" | "yield", lane: lane id, line: [[x, y], [x, y]]}]
"""

import heapq
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "data_utils"))
from mirror_map_bin import write_bin  # noqa: E402

LANE_SURFACE_STREET = 2
ROAD_EDGE_BOUNDARY = 21
CONTROL_TYPES = {"light": 1, "stop": 2, "yield": 3}
NO_TRAFFIC_PHASE = -1
UNREACHABLE_LANE_DISTANCE = -1.0  # C treats negative lane-graph distances as unreachable
SCENARIO_ID_BYTES = 128
DATASET_NAME_BYTES = 32
DATASET_NAME = b"map_editor"

MAX_LANE_COUNT = 2000
MAX_EDGE_COUNT = 4000
MAX_CONTROL_COUNT = 500
MAX_POLYLINE_POINTS = 5000
MAX_SUCCESSOR_COUNT = 16
MAX_ABS_COORD_M = 20000.0
MIN_LANE_LENGTH_M = 1.0
MIN_LANE_WIDTH_M = 1.5
MAX_LANE_WIDTH_M = 10.0
MIN_SPEED_LIMIT_KPH = 5.0
MAX_SPEED_LIMIT_KPH = 130.0
MIN_STOP_LINE_LENGTH_M = 0.5

DENSE_SPACING_M = 0.25
SIMPLIFY_TOLERANCE_M = 0.05
MAX_SEGMENT_M = 9.5  # CARLA bins keep segments under ~10 m, the obs segment-length normaliser
KPH_PER_MPS = 3.6


class MapSpecError(ValueError):
    pass


def _finite(value, name, low=-math.inf, high=math.inf):
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise MapSpecError(f"{name} must be a finite number")
    if not low <= value <= high:
        raise MapSpecError(f"{name}={value} outside [{low}, {high}]")
    return float(value)


def _list(value, name, min_count, max_count):
    if not isinstance(value, list) or not min_count <= len(value) <= max_count:
        raise MapSpecError(f"{name} must be a list of {min_count}..{max_count} entries")
    return value


def _polyline(value, name, min_points):
    points = _list(value, name, min_points, MAX_POLYLINE_POINTS)
    xy = []
    for point_idx, point in enumerate(points):
        if not isinstance(point, list) or len(point) != 2:
            raise MapSpecError(f"{name}[{point_idx}] must be [x, y]")
        xy.append([_finite(coord, f"{name}[{point_idx}]", -MAX_ABS_COORD_M, MAX_ABS_COORD_M) for coord in point])
    return np.asarray(xy, dtype=np.float64)


def _is_lane_id(value, lane_idx_by_id):
    return isinstance(value, int) and not isinstance(value, bool) and value in lane_idx_by_id


def validate_spec(spec):
    """Reject malformed, out-of-range or dangling specs before anything is written."""
    if not isinstance(spec, dict):
        raise MapSpecError("map must be a JSON object")
    lanes_in = _list(spec.get("lanes"), "lanes", 1, MAX_LANE_COUNT)
    edges_in = _list(spec.get("edges", []), "edges", 0, MAX_EDGE_COUNT)
    controls_in = _list(spec.get("controls", []), "controls", 0, MAX_CONTROL_COUNT)

    lanes, lane_idx_by_id = [], {}
    for lane_idx, entry in enumerate(lanes_in):
        name = f"lanes[{lane_idx}]"
        if not isinstance(entry, dict):
            raise MapSpecError(f"{name} must be an object")
        lane_id = entry.get("id")
        if not isinstance(lane_id, int) or isinstance(lane_id, bool) or lane_id in lane_idx_by_id:
            raise MapSpecError(f"{name}.id must be a unique integer")
        lane_idx_by_id[lane_id] = lane_idx
        points = _polyline(entry.get("points"), f"{name}.points", 2)
        if np.hypot(*np.diff(points, axis=0).T).sum() < MIN_LANE_LENGTH_M:
            raise MapSpecError(f"{name} is shorter than {MIN_LANE_LENGTH_M} m")
        lanes.append(
            {
                "id": lane_id,
                "points": points,
                "width_m": _finite(entry.get("width_m"), f"{name}.width_m", MIN_LANE_WIDTH_M, MAX_LANE_WIDTH_M),
                "speed_limit_mps": _finite(
                    entry.get("speed_limit_kph"), f"{name}.speed_limit_kph", MIN_SPEED_LIMIT_KPH, MAX_SPEED_LIMIT_KPH
                )
                / KPH_PER_MPS,
                "successors": _list(entry.get("successors", []), f"{name}.successors", 0, MAX_SUCCESSOR_COUNT),
            }
        )
    for lane in lanes:
        if any(not _is_lane_id(succ, lane_idx_by_id) or succ == lane["id"] for succ in lane["successors"]):
            raise MapSpecError(f"lane {lane['id']} has an unknown or self successor {lane['successors']}")
        if len(set(lane["successors"])) != len(lane["successors"]):
            raise MapSpecError(f"lane {lane['id']} lists a successor twice")
        lane["successor_idxs"] = sorted(lane_idx_by_id[succ] for succ in lane["successors"])

    edges = []
    for edge_idx, entry in enumerate(edges_in):
        if not isinstance(entry, dict):
            raise MapSpecError(f"edges[{edge_idx}] must be an object")
        edges.append(_polyline(entry.get("points"), f"edges[{edge_idx}].points", 2))

    controls = []
    for control_idx, entry in enumerate(controls_in):
        name = f"controls[{control_idx}]"
        if not isinstance(entry, dict) or entry.get("type") not in CONTROL_TYPES:
            raise MapSpecError(f"{name}.type must be one of {sorted(CONTROL_TYPES)}")
        if not _is_lane_id(entry.get("lane"), lane_idx_by_id):
            raise MapSpecError(f"{name}.lane references an unknown lane")
        line = _polyline(entry.get("line"), f"{name}.line", 2)
        if len(line) != 2 or np.hypot(*(line[1] - line[0])) < MIN_STOP_LINE_LENGTH_M:
            raise MapSpecError(f"{name}.line must be two points at least {MIN_STOP_LINE_LENGTH_M} m apart")
        controls.append({"type": CONTROL_TYPES[entry["type"]], "lane_idx": lane_idx_by_id[entry["lane"]], "line": line})
    return lanes, edges, controls


def resample_polyline(points):
    """Dense resample, then greedy chord simplification so straights get long segments and curves short ones."""
    segment_lengths = np.hypot(*np.diff(points, axis=0).T)
    keep = np.concatenate([[True], segment_lengths > 1e-6])
    points = points[keep]
    cum = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(points, axis=0).T))])
    dense_count = max(2, int(math.ceil(cum[-1] / DENSE_SPACING_M)) + 1)
    dense_s = np.linspace(0.0, cum[-1], dense_count)
    dense = np.stack([np.interp(dense_s, cum, points[:, 0]), np.interp(dense_s, cum, points[:, 1])], axis=1)

    kept_idxs = [0]
    anchor_idx = 0
    for _ in range(dense_count):
        if anchor_idx == dense_count - 1:
            break
        end_idx = anchor_idx + 1
        for candidate_idx in range(anchor_idx + 2, dense_count):
            if dense_s[candidate_idx] - dense_s[anchor_idx] > MAX_SEGMENT_M:
                break
            chord = dense[candidate_idx] - dense[anchor_idx]
            chord_length = np.hypot(*chord)
            offsets = dense[anchor_idx + 1 : candidate_idx] - dense[anchor_idx]
            deviation = np.abs(offsets[:, 0] * chord[1] - offsets[:, 1] * chord[0]) / chord_length
            if deviation.max() > SIMPLIFY_TOLERANCE_M:
                break
            end_idx = candidate_idx
        kept_idxs.append(end_idx)
        anchor_idx = end_idx
    return dense[kept_idxs]


def _headings(xy):
    deltas = np.diff(xy, axis=0)
    headings = np.arctan2(deltas[:, 1], deltas[:, 0])
    return tuple(np.append(headings, headings[-1]).astype(np.float32).tolist())


def _road(road_idx, road_type, xy):
    point_count = len(xy)
    return {
        "id": road_idx,
        "type": road_type,
        "S": point_count,
        "x": tuple(xy[:, 0].astype(np.float32).tolist()),
        "y": tuple(xy[:, 1].astype(np.float32).tolist()),
        "z": (0.0,) * point_count,
        "headings": _headings(xy),
    }


def _lane_graph(lane_roads):
    """distances[i, g]: shortest drive from the start of lane i to the start of lane g."""
    lane_count = len(lane_roads)
    distances = np.full((lane_count, lane_count), UNREACHABLE_LANE_DISTANCE, dtype=np.float64)
    for source_idx in range(lane_count):
        settled = distances[source_idx]
        frontier = [(0.0, source_idx)]
        for _ in range(lane_count * MAX_SUCCESSOR_COUNT + 1):
            if not frontier:
                break
            distance_m, lane_idx = heapq.heappop(frontier)
            if settled[lane_idx] >= 0.0:
                continue
            settled[lane_idx] = distance_m
            for exit_idx in lane_roads[lane_idx]["exit_lanes"]:
                if settled[exit_idx] < 0.0:
                    heapq.heappush(frontier, (distance_m + lane_roads[lane_idx]["length"], exit_idx))
    return {
        "n": lane_count,
        "lane_ids": tuple(range(lane_count)),
        "distances": tuple(distances.astype(np.float32).reshape(-1).tolist()),
    }


def _speed_zones(lanes):
    """Connected lanes sharing a speed limit form one zone, so speed-limit randomisation moves them together."""
    zone_parent = list(range(len(lanes)))

    def root(lane_idx):
        for _ in range(len(lanes)):
            if zone_parent[lane_idx] == lane_idx:
                break
            lane_idx = zone_parent[lane_idx]
        return lane_idx

    for lane_idx, lane in enumerate(lanes):
        for succ_idx in lane["successor_idxs"]:
            if lanes[succ_idx]["speed_limit_mps"] == lane["speed_limit_mps"]:
                zone_parent[root(succ_idx)] = root(lane_idx)
    zone_idx_by_root = {}
    return [zone_idx_by_root.setdefault(root(lane_idx), len(zone_idx_by_root)) for lane_idx in range(len(lanes))]


def build_map_bin(spec, out_path, map_name):
    """Validate `spec`, write the map .bin to out_path and return a summary of what was written."""
    lanes, edges, controls = validate_spec(spec)
    predecessor_idxs = [[] for _ in lanes]
    for lane_idx, lane in enumerate(lanes):
        for succ_idx in lane["successor_idxs"]:
            predecessor_idxs[succ_idx].append(lane_idx)

    roads = []
    for lane_idx, (lane, zone_idx) in enumerate(zip(lanes, _speed_zones(lanes))):
        xy = resample_polyline(lane["points"])
        cum = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])
        road = _road(lane_idx, LANE_SURFACE_STREET, xy)
        road["entry_lanes"] = tuple(sorted(predecessor_idxs[lane_idx]))
        road["exit_lanes"] = tuple(lane["successor_idxs"])
        road["speed_limit"] = lane["speed_limit_mps"]
        road["length"] = float(cum[-1])
        road["cum_lengths"] = tuple(cum.astype(np.float32).tolist())
        road["widths"] = (lane["width_m"],) * len(xy)
        road["speed_zone_idx"] = zone_idx
        roads.append(road)
    lane_graph = _lane_graph(roads)
    for edge_points in edges:
        roads.append(_road(len(roads), ROAD_EDGE_BOUNDARY, resample_polyline(edge_points)))

    traffic = []
    for control in controls:
        (x1, y1), (x2, y2) = control["line"]
        traffic.append(
            {
                "id": len(traffic),
                "type": control["type"],
                "stop_line": (x1, y1, 0.0, x2, y2, 0.0),
                "heading": roads[control["lane_idx"]]["headings"][0],
                "states": (),
                "controlled_lanes": (control["lane_idx"],),
                "junction_id": NO_TRAFFIC_PHASE,
                "phase_idx": NO_TRAFFIC_PHASE,
            }
        )

    data = {
        "agents": [],
        "roads": roads,
        "traffic": traffic,
        "objects": [],
        "lane_graph": lane_graph,
        "scenario_id": map_name.encode()[:SCENARIO_ID_BYTES].ljust(SCENARIO_ID_BYTES, b"\0"),
        "dataset_name": DATASET_NAME.ljust(DATASET_NAME_BYTES, b"\0"),
        "log_length": 0,
        "log_dt": 0.0,
        "objects_of_interest": (),
        "tracks_to_predict": (),
        "has_phase_section": True,
        "has_width_section": True,
        "has_zone_section": True,
        "has_scripted_goal_section": False,
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_bin(data, out_path)
    return {
        "lanes": len(lanes),
        "edges": len(edges),
        "controls": len(traffic),
        "lane_points": sum(road["S"] for road in roads[: len(lanes)]),
    }
