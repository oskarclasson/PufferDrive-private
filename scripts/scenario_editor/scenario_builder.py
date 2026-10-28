"""Turn a scenario-editor spec (JSON) plus a base map .bin into a scenario .bin the sim can evaluate.

The base map keeps its roads, lane graph and traffic controls. Its logged agents are replaced by the
authored actors (policy-controlled, following their scripted goals in order) and static obstacles.
"""

import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "data_utils"))
from mirror_map_bin import read_bin, write_bin  # noqa: E402

VEHICLE_TYPE = 1
TRAFFIC_LIGHT_TYPE = 1
LIGHT_STATE_RED = 1
LIGHT_STATE_YELLOW = 2
LIGHT_STATE_GREEN = 3
DRIVABLE_LANE_TYPES = (1, 2)
LANE_TYPE_MAX = 9
ROAD_LINE_TYPE_MAX = 19
ROAD_EDGE_TYPE_MAX = 29
DEFAULT_LANE_WIDTH_M = 3.7
SCENARIO_ID_BYTES = 128
SIM_DT_SECONDS = 0.1
MAX_DURATION_SECONDS = 300.0
MAX_ACTOR_COUNT = 150
MAX_OBSTACLE_COUNT = 150  # actors + obstacles stay within max_agents_per_env
MAX_GOALS_PER_ACTOR = 256
MAX_SPEED_MPS = 40.0
MAX_VEHICLE_DIMENSION_M = 30.0
MIN_VEHICLE_DIMENSION_M = 0.1
MAX_LIGHT_PHASE_SECONDS = 300.0
ROUTE_LANE_MAX_HEADING_DIFF_RAD = math.pi / 2
LIGHT_MODES = ("log", "red", "green", "cycle_green", "cycle_red")


class ScenarioSpecError(ValueError):
    pass


def _finite(value, name, low=-math.inf, high=math.inf):
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ScenarioSpecError(f"{name} must be a finite number")
    if not low <= value <= high:
        raise ScenarioSpecError(f"{name}={value} outside [{low}, {high}]")
    return float(value)


def _validate_box(entry, name):
    if not isinstance(entry, dict):
        raise ScenarioSpecError(f"{name} must be an object")
    return {
        "x": _finite(entry.get("x"), f"{name}.x"),
        "y": _finite(entry.get("y"), f"{name}.y"),
        "heading": _finite(entry.get("heading"), f"{name}.heading", -4 * math.pi, 4 * math.pi),
        "length": _finite(entry.get("length"), f"{name}.length", MIN_VEHICLE_DIMENSION_M, MAX_VEHICLE_DIMENSION_M),
        "width": _finite(entry.get("width"), f"{name}.width", MIN_VEHICLE_DIMENSION_M, MAX_VEHICLE_DIMENSION_M),
    }


def validate_spec(spec):
    """Reject malformed or out-of-range specs before anything is written."""
    if not isinstance(spec, dict):
        raise ScenarioSpecError("scenario must be a JSON object")
    duration_seconds = _finite(spec.get("duration_s"), "duration_s", SIM_DT_SECONDS, MAX_DURATION_SECONDS)
    actors_in = spec.get("actors", [])
    obstacles_in = spec.get("obstacles", [])
    if not isinstance(actors_in, list) or not 1 <= len(actors_in) <= MAX_ACTOR_COUNT:
        raise ScenarioSpecError(f"need between 1 and {MAX_ACTOR_COUNT} actors")
    if not isinstance(obstacles_in, list) or len(obstacles_in) > MAX_OBSTACLE_COUNT:
        raise ScenarioSpecError(f"at most {MAX_OBSTACLE_COUNT} obstacles")
    actors = []
    for actor_idx, entry in enumerate(actors_in):
        actor = _validate_box(entry, f"actors[{actor_idx}]")
        actor["speed"] = _finite(entry.get("speed", 0.0), f"actors[{actor_idx}].speed", 0.0, MAX_SPEED_MPS)
        goals = entry.get("goals")
        if not isinstance(goals, list) or not 1 <= len(goals) <= MAX_GOALS_PER_ACTOR:
            raise ScenarioSpecError(f"actors[{actor_idx}] needs between 1 and {MAX_GOALS_PER_ACTOR} goals")
        actor["goals"] = []
        for goal_idx, goal in enumerate(goals):
            if not isinstance(goal, list) or len(goal) != 2:
                raise ScenarioSpecError(f"actors[{actor_idx}].goals[{goal_idx}] must be [x, y]")
            name = f"actors[{actor_idx}].goals[{goal_idx}]"
            actor["goals"].append((_finite(goal[0], name), _finite(goal[1], name)))
        actors.append(actor)
    obstacles = [_validate_box(entry, f"obstacles[{idx}]") for idx, entry in enumerate(obstacles_in)]
    lights_in = spec.get("lights", {})
    if not isinstance(lights_in, dict):
        raise ScenarioSpecError("lights must be an object of {traffic_idx: mode}")
    lights = {}
    for key, mode in lights_in.items():
        if not str(key).isdigit() or mode not in LIGHT_MODES:
            raise ScenarioSpecError(f"lights[{key}]={mode!r}: mode must be one of {LIGHT_MODES}")
        lights[int(key)] = mode
    cycle_in = spec.get("light_cycle", {})
    if not isinstance(cycle_in, dict):
        raise ScenarioSpecError("light_cycle must be an object")
    cycle = {
        phase: _finite(cycle_in.get(phase, default), f"light_cycle.{phase}", 0.0, MAX_LIGHT_PHASE_SECONDS)
        for phase, default in (("green_s", 15.0), ("yellow_s", 3.0), ("red_s", 15.0))
    }
    if cycle["green_s"] + cycle["yellow_s"] + cycle["red_s"] <= 0.0:
        raise ScenarioSpecError("light_cycle phases must not all be zero")
    return {
        "duration_s": duration_seconds,
        "actors": actors,
        "obstacles": obstacles,
        "lights": lights,
        "light_cycle": cycle,
    }


class LaneIndex:
    """Flat arrays of every drivable lane vertex for nearest-lane queries."""

    def __init__(self, roads):
        xs, ys, zs, headings, lane_idxs = [], [], [], [], []
        for road in roads:
            if road["type"] not in DRIVABLE_LANE_TYPES or road["S"] == 0:
                continue
            xs.extend(road["x"])
            ys.extend(road["y"])
            zs.extend(road["z"])
            headings.extend(road["headings"])
            lane_idxs.extend([road["id"]] * road["S"])
        if not xs:
            raise ScenarioSpecError("base map has no drivable lanes")
        self.x = np.asarray(xs)
        self.y = np.asarray(ys)
        self.z = np.asarray(zs)
        self.heading = np.asarray(headings)
        self.lane_idx = np.asarray(lane_idxs)

    def nearest(self, x, y, heading=None):
        dist_sq = (self.x - x) ** 2 + (self.y - y) ** 2
        if heading is not None:
            heading_diff = np.abs(np.angle(np.exp(1j * (self.heading - heading))))
            dist_sq = np.where(heading_diff < ROUTE_LANE_MAX_HEADING_DIFF_RAD, dist_sq, np.inf)
            if not np.isfinite(dist_sq).any():
                dist_sq = (self.x - x) ** 2 + (self.y - y) ** 2
        vertex_idx = int(np.argmin(dist_sq))
        return int(self.lane_idx[vertex_idx]), float(self.z[vertex_idx])


def _constant_track(agent_id, box, z, speed_mps, step_count):
    cols = {
        "x": (box["x"],) * step_count,
        "y": (box["y"],) * step_count,
        "z": (z,) * step_count,
        "h": (box["heading"],) * step_count,
        "vx": (speed_mps * math.cos(box["heading"]),) * step_count,
        "vy": (speed_mps * math.sin(box["heading"]),) * step_count,
        "len": (box["length"],) * step_count,
        "wid": (box["width"],) * step_count,
        "hgt": (1.5,) * step_count,
        "valid": (1,) * step_count,
    }
    return {"id": agent_id, "type": VEHICLE_TYPE, "T": step_count, "cols": cols, "mark_as_expert": 0}


def _light_states(base_states, mode, cycle, step_count):
    if mode == "red":
        return (LIGHT_STATE_RED,) * step_count
    if mode == "green":
        return (LIGHT_STATE_GREEN,) * step_count
    if mode == "log" and not base_states:
        mode = "cycle_green"  # drawn and CARLA maps carry no light log; dark lights would never stop anyone
    if mode == "log":
        held = list(base_states[:step_count])
        return tuple(held + [held[-1]] * (step_count - len(held)))
    phases = [(LIGHT_STATE_GREEN, cycle["green_s"]), (LIGHT_STATE_YELLOW, cycle["yellow_s"]), (LIGHT_STATE_RED, cycle["red_s"])]
    if mode == "cycle_red":
        phases = phases[2:] + phases[:2]
    pattern = []
    for state, seconds in phases:
        pattern += [state] * int(round(seconds / SIM_DT_SECONDS))
    return tuple(pattern[step % len(pattern)] for step in range(step_count))


def build_scenario_bin(base_bin_path, spec, out_path, scenario_name):
    """Validate `spec`, write the scenario .bin to out_path and return the simulated step count."""
    clean = validate_spec(spec)
    data = read_bin(Path(base_bin_path))
    lanes = LaneIndex(data["roads"])
    scenario_steps = int(round(clean["duration_s"] / SIM_DT_SECONDS))
    step_count = scenario_steps + 1

    agents = []
    for actor in clean["actors"]:
        start_lane_idx, start_z = lanes.nearest(actor["x"], actor["y"], actor["heading"])
        agent = _constant_track(len(agents), actor, start_z, actor["speed"], step_count)
        scripted_goals = tuple((gx, gy, lanes.nearest(gx, gy)[1]) for gx, gy in actor["goals"])
        agent.update(route=(start_lane_idx,), route_gt_len=1, goal=scripted_goals[-1], scripted_goals=scripted_goals)
        agents.append(agent)
    for obstacle in clean["obstacles"]:
        obstacle_z = lanes.nearest(obstacle["x"], obstacle["y"])[1]
        agent = _constant_track(len(agents), obstacle, obstacle_z, 0.0, step_count)
        agent.update(route=(), route_gt_len=0, goal=(obstacle["x"], obstacle["y"], obstacle_z), scripted_goals=())
        agents.append(agent)
    data["agents"] = agents
    data["objects_of_interest"] = ()
    data["tracks_to_predict"] = ()

    for traffic in data["traffic"]:
        if traffic["type"] != TRAFFIC_LIGHT_TYPE:
            continue
        mode = clean["lights"].get(traffic["id"], "log")
        traffic["states"] = _light_states(traffic["states"], mode, clean["light_cycle"], step_count)

    # The scripted-goal section is only read after the three earlier optional sections, so all are written.
    if not data["has_phase_section"]:
        for traffic in data["traffic"]:
            traffic["junction_id"], traffic["phase_idx"] = -1, -1
    if not data["has_width_section"]:
        for road in data["roads"]:
            if 0 <= road["type"] <= LANE_TYPE_MAX:
                road["widths"] = (DEFAULT_LANE_WIDTH_M,) * road["S"]
    if not data["has_zone_section"]:
        for road in data["roads"]:
            if 0 <= road["type"] <= LANE_TYPE_MAX:
                road["speed_zone_idx"] = -1
    data["has_phase_section"] = data["has_width_section"] = data["has_zone_section"] = True
    data["has_scripted_goal_section"] = True
    data["scenario_id"] = scenario_name.encode()[:SCENARIO_ID_BYTES].ljust(SCENARIO_ID_BYTES, b"\0")
    data["log_length"] = step_count
    data["log_dt"] = SIM_DT_SECONDS  # map-only bases (CARLA, drawn maps) carry log_dt 0

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_bin(data, out_path)
    return scenario_steps


def export_map_geometry(base_bin_path):
    """Road geometry, traffic controls and the logged agents at t=0, as JSON-ready lists for the editor."""
    data = read_bin(Path(base_bin_path))

    def rounded(values):
        return [round(v, 2) for v in values]

    lanes, lines, edges = [], [], []
    for road in data["roads"]:
        polyline = {"id": road["id"], "type": road["type"], "x": rounded(road["x"]), "y": rounded(road["y"])}
        if road["type"] <= LANE_TYPE_MAX:
            polyline["h"] = [round(v, 3) for v in road["headings"]]
            polyline["drivable"] = road["type"] in DRIVABLE_LANE_TYPES
            lanes.append(polyline)
        elif road["type"] <= ROAD_LINE_TYPE_MAX:
            lines.append(polyline)
        elif road["type"] <= ROAD_EDGE_TYPE_MAX:
            edges.append(polyline)
    traffic = [
        {
            "id": t["id"],
            "type": t["type"],
            "line": rounded((t["stop_line"][0], t["stop_line"][1], t["stop_line"][3], t["stop_line"][4])),
            "state0": t["states"][0] if t["states"] else 0,
            "states": list(t["states"]),
        }
        for t in data["traffic"]
    ]
    logged_agents = []
    for agent in data["agents"]:
        cols = agent["cols"]
        if agent["T"] == 0 or not cols["valid"][0]:
            continue
        logged_agents.append(
            {
                "type": agent["type"],
                "x": round(cols["x"][0], 2),
                "y": round(cols["y"][0], 2),
                "heading": round(cols["h"][0], 3),
                "length": round(cols["len"][0], 2),
                "width": round(cols["wid"][0], 2),
                "speed": round(math.hypot(cols["vx"][0], cols["vy"][0]), 2),
            }
        )
    return {
        "lanes": lanes,
        "lines": lines,
        "edges": edges,
        "traffic": traffic,
        "logged_agents": logged_agents,
        "log_dt": data["log_dt"],
    }
