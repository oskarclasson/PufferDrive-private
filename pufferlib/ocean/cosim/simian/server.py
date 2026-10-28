"""PufferDrive policy server for simian-adapter (Simian / ADP co-simulation).

The adapter owns the Simian loop and the vehicle controller; this server owns the observation and the
policy. POST /reset builds a map-only bin from the adapter's road graph, or loads a map-editor bin given with
--drawn-map, and a shadow Drive env with the
ego in slot 0 and Simian's ground-truth actors streamed into the co-sim partner slots. Every POST /step
syncs ego, actors, lights and the route goal window, runs the policy once and integrates the jerk
dynamics one dt; the adapter tracks the returned acceleration, speed and curvature with the tire model.

All JSON coordinates are in the Simian map frame [m], headings CCW from +x [rad], velocities in that
frame [m/s], and the ego position is the bounding-box center.
"""

import argparse
import json
import math
import tempfile
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch
import yaml

import pufferlib.ocean.torch as drive_torch
import pufferlib.pytorch
import pufferlib.spaces
from pufferlib.ocean.cosim.arch import checkpoint_config_path, shadow_env_kwargs
from pufferlib.ocean.cosim.goals import RouteGoalWindow, route_goals_from_xy
from pufferlib.ocean.cosim.nuplan_bridge import DEFAULT_ARCH, FAR_AWAY, LIGHT_STATE_GREEN, goals_along
from pufferlib.ocean.cosim.simian.debug import POOL_NAMES, decode_observation
from pufferlib.ocean.cosim.simian.drawn_map import DrawnMap
from pufferlib.ocean.cosim.simian.map_bin import write_simian_bin
from pufferlib.ocean.drive import binding
from pufferlib.ocean.drive.drive import Drive

EGO_SLOT = 0
EGO_OBS_SPEED_IDX = 0
EGO_OBS_STEERING_IDX = 3
EGO_OBS_ACCEL_LONG_IDX = 4
EGO_OBS_STEERING_NORM_RAD = 0.667  # STEERING_LIMIT in constants.h
WHEELBASE_LENGTH_RATIO = 0.6  # agent->wheelbase = 0.6 * sim_length in drive.h
SCENARIO_LENGTH_MARGIN_STEPS = 2
SPAWN_PROBE_TOLERANCE_M = 0.5
DEFAULT_PORT = 8710
DEFAULT_MAX_STEPS = 6000
DEFAULT_MAX_PARTNERS = 63
DEFAULT_GOAL_SPACING_M = 30.0
COMPILE_PREFIXES = ("module.", "_orig_mod.")


def clean_policy_state_dict(state_dict):
    def clean(key):
        while key.startswith(COMPILE_PREFIXES):
            key = key.split(".", 1)[1]
        return key

    return {clean(k): v for k, v in state_dict.items()}


class PolicySession:
    """One Simian scenario: shadow env, policy and route goal window."""

    def __init__(self, request, args, work_dir):
        checkpoint, device = args.checkpoint, args.device
        self.device = device
        self.max_partners = int(args.max_partners)
        ego = request["ego"]
        self.ego_length_m = float(args.ego_length_m or ego["length"])
        self.ego_width_m = float(args.ego_width_m or ego["width"])
        scenario_length = int(args.max_steps) + SCENARIO_LENGTH_MARGIN_STEPS
        if args.drawn_map:
            bin_path = Path(args.drawn_map)
            self.map = DrawnMap(
                args.drawn_map_spec, request["utm_zone"], request["utm_north"], float(ego["x"]), float(ego["y"])
            )
        else:
            bin_path = Path(work_dir) / "simian_map.bin"
            origin = (round(float(ego["x"])), round(float(ego["y"])))
            self.map = write_simian_bin(request["road_graph"], origin, bin_path, scenario_length + 1)

        cfg = yaml.safe_load(open(checkpoint_config_path(checkpoint)))
        overrides = {
            "map_dir": str(bin_path),
            "num_maps": 1,
            "num_agents": 1,
            "min_agents_per_env": 1,
            "cosim_partner_slots": self.max_partners,
            "goal_source": "external",
            "goal_reach_requires_speed": True,
            "scenario_length": scenario_length,
            "resample_frequency": 0,
            "termination_mode": 0,
            "eval_mode": False,
            "collision_behavior": "ignore",
            "offroad_behavior": "ignore",
            "traffic_light_behavior": "ignore",
        }
        if request.get("dt"):
            overrides["dt"] = float(request["dt"])
        self.arch = shadow_env_kwargs(cfg, defaults=DEFAULT_ARCH, overrides=overrides)
        self.env = Drive(**self.arch)
        self.env.reset()
        self.dt = float(self.arch["dt"])

        ex, ey = self.map.to_bin(float(ego["x"]), float(ego["y"]))
        self.ego_stopped_s = 0.0
        self.partner_stopped_s = {}  # Simian actor id -> seconds at standstill
        self._set_ego(ego, ex, ey, accel_long=0.0)
        readback = self.env.get_global_agent_state()
        if (
            abs(float(readback["x"][0]) - ex) > SPAWN_PROBE_TOLERANCE_M
            or abs(float(readback["y"][0]) - ey) > SPAWN_PROBE_TOLERANCE_M
        ):
            raise RuntimeError("shadow env slot 0 is not the ego: gigaflow spawning failed on the Simian map bin")
        self.env.set_agent_sizes(
            np.array([EGO_SLOT], np.int32),
            np.array([self.ego_length_m], np.float32),
            np.array([self.ego_width_m], np.float32),
        )

        cfg["policy"].setdefault("action_type", cfg["env"]["action_type"])
        policy = getattr(drive_torch, cfg.get("policy_name", "Drive"))(self.env, **cfg["policy"])
        policy.load_state_dict(clean_policy_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)))
        self.policy = policy.to(device).eval()
        self.env_continuous = isinstance(self.env.single_action_space, pufferlib.spaces.Box)
        self.action_selection = (
            pufferlib.pytorch.ACTION_SELECT_MEAN
            if self.env_continuous and not self.policy.is_continuous
            else pufferlib.pytorch.ACTION_SELECT_MODE
        )

        route_utm = np.asarray(request["route"], dtype=np.float64).reshape(-1, 3)
        route = np.stack(self.map.to_bin(route_utm[:, 0], route_utm[:, 1]), axis=1)
        start_idx = int(np.argmin(np.hypot(route[:, 0] - ex, route[:, 1] - ey)))
        goals = goals_along(route[start_idx:], args.goal_spacing_m)
        if len(goals) == 0:
            raise ValueError("route has no points ahead of the ego")
        self.goal_window = RouteGoalWindow(self.env, route_goals_from_xy(goals, origin_xy=(ex, ey)))
        self.goal_count = len(goals)
        self.lane_count = len(self.map.lane_idx_by_id)
        self.scenario_length = scenario_length

    def _set_ego(self, ego, ex, ey, accel_long):
        vx, vy = self.map.velocity_to_bin(float(ego["vx"]), float(ego["vy"]))
        self.env.set_agent_states(
            np.array([EGO_SLOT], np.int32),
            np.array([ex], np.float32),
            np.array([ey], np.float32),
            np.array([float(ego["z"])], np.float32),
            np.array([self.map.heading_to_bin(float(ego["heading"]))], np.float32),
            np.array([vx], np.float32),
            np.array([vy], np.float32),
            np.array([float(ego["yaw_rate"])], np.float32),
            np.array([accel_long], np.float32),
            np.array([self.ego_stopped_s], np.float32),
        )

    def _update_stopped_times(self, ego, actors):
        # stopped time follows the Simian speeds and actor ids: c_step only accumulates it for the shadow ego slot
        threshold_mps, dt = float(binding.AGENT_STOPPED_SPEED_THRESHOLD), self.dt
        ego_stopped = math.hypot(float(ego["vx"]), float(ego["vy"])) < threshold_mps
        self.ego_stopped_s = self.ego_stopped_s + dt if ego_stopped else 0.0
        stopped_by_id = {}
        for actor in actors:
            actor_stopped = math.hypot(actor["vx"], actor["vy"]) < threshold_mps
            stopped_by_id[actor["id"]] = self.partner_stopped_s.get(actor["id"], 0.0) + dt if actor_stopped else 0.0
        self.partner_stopped_s = stopped_by_id

    def _set_partners(self, ego_x, ego_y, actors):
        actors = sorted(actors, key=lambda a: (a["x"] - ego_x) ** 2 + (a["y"] - ego_y) ** 2)[: self.max_partners]
        if actors:
            idx = np.arange(1, 1 + len(actors), dtype=np.int32)
            bx, by = self.map.to_bin(np.array([a["x"] for a in actors]), np.array([a["y"] for a in actors]))
            bvx, bvy = self.map.velocity_to_bin(
                np.array([a["vx"] for a in actors]), np.array([a["vy"] for a in actors])
            )
            zero = np.zeros(len(actors), np.float32)
            self.env.set_agent_states(
                idx,
                bx.astype(np.float32),
                by.astype(np.float32),
                np.array([a["z"] for a in actors], np.float32),
                self.map.heading_to_bin(np.array([a["heading"] for a in actors], np.float32)),
                bvx.astype(np.float32),
                bvy.astype(np.float32),
                zero,
                zero,
                np.array([self.partner_stopped_s[a["id"]] for a in actors], np.float32),
            )
            self.env.set_agent_sizes(
                idx,
                np.array([a["length"] for a in actors], np.float32),
                np.array([a["width"] for a in actors], np.float32),
            )
        surplus = np.arange(1 + len(actors), 1 + self.max_partners, dtype=np.int32)
        if len(surplus):
            far = np.full(len(surplus), FAR_AWAY, np.float32)
            zero = np.zeros_like(far)
            self.env.set_agent_states(surplus, far, far, far, zero, zero, zero, zero, zero)

    def _set_lights(self, light_states):
        if not self.map.num_traffic:
            return
        states = np.array(
            [int(light_states.get(light_id, LIGHT_STATE_GREEN)) for light_id in self.map.element_light_ids], np.int32
        )
        self.env.set_traffic_light_states(states)

    def _debug(self, obs_tensor, logits, value):
        """What the policy saw and chose this step: the decoded ego-frame observation, the goal window in the map
        frame and the action distribution."""
        with torch.no_grad():
            pool_counts = self.policy.pool_slot_counts(obs_tensor)
        pool = {name: pool_counts[name][EGO_SLOT].cpu().numpy() for name in POOL_NAMES if name in pool_counts}
        probs = logits.softmax(-1)[EGO_SLOT].float().cpu()
        consumed = self.goal_window.consumed_count
        goals = []
        for i, goal in enumerate(self.goal_window.window):
            x, y = self.map.from_bin(float(goal[0]), float(goal[1]))
            goals.append({"x": float(x), "y": float(y), "consumed": i < consumed})
        return {
            "observation": decode_observation(self.env, obs_tensor[EGO_SLOT].cpu().numpy(), pool),
            "goals": goals,
            "action_probs": probs.tolist(),
            "entropy": float(-(probs * probs.clamp_min(1e-12).log()).sum()),
            "value": float(value.reshape(-1)[EGO_SLOT]),
        }

    def step(self, request):
        if self.env.tick >= self.scenario_length - 1:
            raise RuntimeError(f"shadow env reached scenario_length {self.scenario_length}; raise --max-steps")
        ego = request["ego"]
        ex, ey = self.map.to_bin(float(ego["x"]), float(ego["y"]))
        self._update_stopped_times(ego, request["actors"])
        self._set_ego(ego, ex, ey, accel_long=float(ego["accel_long"]))
        self._set_partners(float(ego["x"]), float(ego["y"]), request["actors"])
        self._set_lights(request["traffic_light_states"])
        self.goal_window.sync(ex, ey, self.map.heading_to_bin(float(ego["heading"])))
        obs = np.asarray(self.env.recompute_observations())
        start_speed_mps = float(obs[EGO_SLOT][EGO_OBS_SPEED_IDX]) * float(self.env.obs_norm_speed_mps)

        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs).to(self.device)
            logits, value = self.policy.forward_eval(obs_tensor)
            action, _, _, cont_action = pufferlib.pytorch.sample_logits(
                logits, action_selection=self.action_selection, env_continuous=self.env_continuous, policy=self.policy
            )
        env_action = cont_action if self.env_continuous and cont_action is not None else action
        act = env_action.cpu().numpy().reshape(self.env.num_agents, -1)
        if not self.env_continuous:
            act = act.astype(np.int32)
        debug = self._debug(obs_tensor, logits, value)
        self.env.step(act)

        ego_obs = np.asarray(self.env.observations)[EGO_SLOT]
        steering_angle = float(ego_obs[EGO_OBS_STEERING_IDX]) * EGO_OBS_STEERING_NORM_RAD
        current_goal_idx, goal_count = self.env.get_agent_goal_progress(EGO_SLOT)
        return {
            "dt": self.dt,
            "start_speed_mps": start_speed_mps,
            "speed_mps": float(ego_obs[EGO_OBS_SPEED_IDX]) * float(self.env.obs_norm_speed_mps),
            "accel_long_mps2": float(ego_obs[EGO_OBS_ACCEL_LONG_IDX]) * float(binding.ACCEL_LONG_NORM),
            "steering_angle_rad": steering_angle,
            "curvature_inv_m": math.tan(steering_angle) / (WHEELBASE_LENGTH_RATIO * self.ego_length_m),
            "route_done": bool(self.goal_window.exhausted),
            "goal_index": int(self.goal_window.window_start + min(current_goal_idx, goal_count)),
            "goal_count": self.goal_count,
            "debug": debug,
        }


class PolicyServer(ThreadingHTTPServer):
    def __init__(self, address, args):
        super().__init__(address, PolicyHandler)
        self.args = args
        self.session = None
        self.lock = threading.Lock()
        self.work_dir = tempfile.mkdtemp(prefix="pufferdrive_simian_")


class PolicyHandler(BaseHTTPRequestHandler):
    def _reply(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path != "/healthz":
            self._reply(404, {"error": f"unknown path {self.path}"})
            return
        self._reply(200, {"status": "ok"})

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        server = self.server
        try:
            with server.lock:
                if self.path == "/reset":
                    server.session = PolicySession(request, server.args, server.work_dir)
                    session = server.session
                    self._reply(
                        200,
                        {
                            "dt": session.dt,
                            "lanes": session.lane_count,
                            "traffic_elements": session.map.num_traffic,
                            "goals": session.goal_count,
                        },
                    )
                elif self.path == "/step":
                    if server.session is None:
                        raise RuntimeError("POST /reset before /step")
                    self._reply(200, server.session.step(request))
                else:
                    self._reply(404, {"error": f"unknown path {self.path}"})
        except Exception as exc:
            traceback.print_exc()
            self._reply(500, {"error": f"{type(exc).__name__}: {exc}"})

    def log_message(self, format, *args):
        return


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="final_model.pt with config.yaml next to it")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS, help="policy steps per scenario")
    parser.add_argument("--max-partners", type=int, default=DEFAULT_MAX_PARTNERS, help="streamed actor slots")
    parser.add_argument("--goal-spacing-m", type=float, default=DEFAULT_GOAL_SPACING_M)
    parser.add_argument("--ego-length-m", type=float, help="ego length seen by the policy, default the Simian one")
    parser.add_argument("--ego-width-m", type=float, help="ego width seen by the policy, default the Simian one")
    parser.add_argument("--drawn-map", help="map-editor .bin used instead of the adapter's road graph")
    parser.add_argument("--drawn-map-spec", help="map-editor spec .json of --drawn-map, for its lat/lon origin")
    args = parser.parse_args()
    if bool(args.drawn_map) != bool(args.drawn_map_spec):
        parser.error("--drawn-map and --drawn-map-spec go together")
    if any(size is not None and not size > 0 for size in (args.ego_length_m, args.ego_width_m)):
        parser.error("--ego-length-m and --ego-width-m must be positive")
    torch.set_num_threads(1)
    server = PolicyServer((args.host, args.port), args)
    print(f"[simian_policy] serving {args.checkpoint} on {args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
