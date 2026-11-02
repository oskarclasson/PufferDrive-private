"""PufferDrive policy server for simian-adapter (Simian / ADP co-simulation).

The adapter owns the Simian loop and the vehicle controller; this server owns the observation and the
policy. POST /reset builds a map-only bin from the adapter's road graph, or loads a map-editor bin given with
--drawn-map, and a shadow Drive env with the
ego in slot 0 and Simian's ground-truth actors streamed into the co-sim partner slots. Every POST /step
syncs ego, actors, lights and the route goal window, runs the policy once and integrates the jerk
dynamics one dt; the adapter tracks the returned acceleration, speed and curvature with the tire model.
With --rollout-horizon-s every step also rolls the policy ahead, the actors held at constant velocity, those slower
than --min-moving-speed-mps standing still, and returns the rollout and the path of every tracked actor in it in the
step debug; GET /reference serves the rollout too. With --actor-prediction joint the nearest moving tracked actors
are instead driven by the policy too, each to the goals goal inference finds likeliest for it, so the ego rollout
and the actors' predicted paths come from one joint rollout; the others, and those with a negative id, untracked
obstacles, are held as without it. In the full stack AD is the client instead of the adapter: it resets and
steps the policy with its own ego state and tracked obstacles, without a road graph, so --drawn-map is required,
and the rollout is the reference its trajectory planner tracks.

All JSON coordinates are in the Simian map frame [m], headings CCW from +x [rad], velocities in that
frame [m/s], and the ego position is the bounding-box center.
"""

import argparse
import json
import math
import tempfile
import threading
import time
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
from pufferlib.ocean.cosim.joint_prediction import GOAL_COLUMNS, GoalInference
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
DEFAULT_ROLLOUT_HORIZON_S = 0.0
ACTOR_PREDICTION_CONSTANT_VELOCITY = "constant-velocity"
ACTOR_PREDICTION_JOINT = "joint"
DEFAULT_JOINT_PREDICTION_ACTORS = 8
DEFAULT_MIN_MOVING_SPEED_MPS = 0.5
# The eval's goal_inference_* defaults in puffer_drive.yaml.
DEFAULT_GOAL_INFERENCE_HYPOTHESES = 8
DEFAULT_GOAL_INFERENCE_GOAL_SPACING_M = 10.0
DEFAULT_GOAL_INFERENCE_UPDATE_S = 1.0
DEFAULT_GOAL_INFERENCE_BETA = 10.0
DEFAULT_GOAL_INFERENCE_FORGETTING = 0.01
STEP_LOG_INTERVAL_STEPS = 10
REFERENCE_MISS_LOG_INTERVAL_S = 5.0
NS_PER_S = 1e9
COMPILE_PREFIXES = ("module.", "_orig_mod.")
MAX_PARTNER_LENGTH_M = 7.0  # largest agent spawn_agent in drive.h trains with
MAX_PARTNER_WIDTH_M = 3.0


def log(message):
    print(f"[simian_policy] {message}", flush=True)


def split_actor(actor):
    """The actor as the fewest equal pieces no longer than MAX_PARTNER_LENGTH_M and no wider than
    MAX_PARTNER_WIDTH_M, each moving like it; the actor itself when it fits."""
    length_count = math.ceil(float(actor["length"]) / MAX_PARTNER_LENGTH_M)
    width_count = math.ceil(float(actor["width"]) / MAX_PARTNER_WIDTH_M)
    if length_count <= 1 and width_count <= 1:
        return [actor]
    piece_length_m = float(actor["length"]) / length_count
    piece_width_m = float(actor["width"]) / width_count
    cos_heading, sin_heading = math.cos(float(actor["heading"])), math.sin(float(actor["heading"]))
    pieces = []
    for length_idx in range(length_count):
        along_m = (length_idx + 0.5) * piece_length_m - 0.5 * float(actor["length"])
        for width_idx in range(width_count):
            across_m = (width_idx + 0.5) * piece_width_m - 0.5 * float(actor["width"])
            pieces.append(
                {
                    **actor,
                    "x": float(actor["x"]) + along_m * cos_heading - across_m * sin_heading,
                    "y": float(actor["y"]) + along_m * sin_heading + across_m * cos_heading,
                    "length": piece_length_m,
                    "width": piece_width_m,
                }
            )
    return pieces


def rollout_summary(points):
    """Length [m] and duration [s] of the rollout until the policy reverses, and its end speed [m/s]."""
    length_m, forward_end = 0.0, points[0]
    for prev, point in zip(points, points[1:]):
        if point["speed_mps"] < 0:
            break
        length_m += math.hypot(point["x"] - prev["x"], point["y"] - prev["y"])
        forward_end = point
    return length_m, forward_end["t"], points[-1]["speed_mps"]


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
        joint = args.actor_prediction == ACTOR_PREDICTION_JOINT
        self.actor_slot_count = int(args.joint_prediction_actors) if joint else 0
        self.min_moving_speed_mps = float(args.min_moving_speed_mps)
        self.static_slot_start = 1 + self.actor_slot_count
        if args.drawn_map:
            bin_path = Path(args.drawn_map)
            self.map = DrawnMap(
                args.drawn_map_spec, request["utm_zone"], request["utm_north"], float(ego["x"]), float(ego["y"])
            )
        elif "road_graph" not in request:
            raise ValueError("reset without a road graph needs the server started with --drawn-map")
        else:
            bin_path = Path(work_dir) / "simian_map.bin"
            origin = (round(float(ego["x"])), round(float(ego["y"])))
            self.map = write_simian_bin(request["road_graph"], origin, bin_path, scenario_length + 1)

        cfg = yaml.safe_load(open(checkpoint_config_path(checkpoint)))
        overrides = {
            "map_dir": str(bin_path),
            "num_maps": 1,
            "num_agents": 1 + self.actor_slot_count,
            "min_agents_per_env": 1 + self.actor_slot_count,
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
        if int(self.env.agent_offsets[-1]) != 1 + self.actor_slot_count:
            raise RuntimeError(
                f"gigaflow spawned {int(self.env.agent_offsets[-1])} of the {1 + self.actor_slot_count} policy agents "
                "on the map bin; lower --joint-prediction-actors"
            )

        ex, ey = self.map.to_bin(float(ego["x"]), float(ego["y"]))
        self.ego_stopped_s = 0.0
        self.partner_stopped_s = {}  # Simian actor id -> seconds at standstill
        self.partners = None
        self.static_actors = []
        self.predicted = None
        self._set_ego(ego, ex, ey, accel_long=0.0)
        readback = self.env.get_global_agent_state()
        if (
            abs(float(readback["x"][0]) - ex) > SPAWN_PROBE_TOLERANCE_M
            or abs(float(readback["y"][0]) - ey) > SPAWN_PROBE_TOLERANCE_M
        ):
            raise RuntimeError("shadow env slot 0 is not the ego: gigaflow spawning failed on the Simian map bin")
        sim_xy = np.zeros((1 + self.actor_slot_count, 2), np.float32)
        self.env.get_active_agent_xy(sim_xy)
        self.world_minus_sim_xy = np.array([readback["x"][EGO_SLOT], readback["y"][EGO_SLOT]]) - sim_xy[EGO_SLOT]
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
        self.rollout_steps = int(round(float(args.rollout_horizon_s) / self.dt))
        self.reference = None
        self.goal_inference = None
        if joint:
            self.goal_inference = GoalInference(
                bin_path,
                num_goals=self.env.num_goals,
                hypothesis_count=args.goal_inference_hypotheses,
                goal_spacing_m=args.goal_inference_goal_spacing_m,
                update_steps=max(1, round(args.goal_inference_update_s / self.dt)),
                beta=args.goal_inference_beta,
                forgetting=args.goal_inference_forgetting,
                gamma=cfg["train"]["gamma"],
                goal_radius_m=self.env.goal_radius,
                reward_goal=self.env.reward_goal,
                horizon_s=args.rollout_horizon_s,
            )

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

    def _partner_states(self, actors, first_slot):
        bx, by = self.map.to_bin(np.array([a["x"] for a in actors]), np.array([a["y"] for a in actors]))
        bvx, bvy = self.map.velocity_to_bin(np.array([a["vx"] for a in actors]), np.array([a["vy"] for a in actors]))
        return {
            "ids": [a["id"] for a in actors],
            "idx": np.arange(first_slot, first_slot + len(actors), dtype=np.int32),
            "x": bx,
            "y": by,
            "z": np.array([a["z"] for a in actors], np.float32),
            "heading": self.map.heading_to_bin(np.array([a["heading"] for a in actors], np.float32)),
            "vx": bvx,
            "vy": bvy,
            "moving": np.array([self._moving(a) for a in actors]),
            "stopped_s": np.array([self.partner_stopped_s[a["id"]] for a in actors], np.float32),
        }

    def _moving(self, actor):
        """Whether the actor moves in the rollout: slower than --min-moving-speed-mps it stands still there."""
        return math.hypot(actor["vx"], actor["vy"]) >= self.min_moving_speed_mps

    def _set_partners(self, actors, predicted_ids):
        """Write the joint-predicted actors into the policy actor slots and the rest, split into pieces the policy
        knows the size of, into the static partner slots, nearest first, and park the slots left over."""
        predicted = [a for a in actors if a["id"] in predicted_ids]
        static = [a for a in actors if a["id"] not in predicted_ids]
        pieces = [piece for actor in static for piece in split_actor(actor)][: self.max_partners]
        self.predicted = self._partner_states(predicted, 1) if predicted else None
        self.partners = self._partner_states(pieces, self.static_slot_start) if pieces else None
        self.static_actors = static
        for partners, slot_actors in ((self.predicted, predicted), (self.partners, pieces)):
            if partners is None:
                continue
            self._place(partners, elapsed_s=0.0)
            self.env.set_agent_sizes(
                partners["idx"],
                np.array([a["length"] for a in slot_actors], np.float32),
                np.array([a["width"] for a in slot_actors], np.float32),
            )
        surplus = np.concatenate(
            [
                np.arange(1 + len(predicted), self.static_slot_start, dtype=np.int32),
                np.arange(
                    self.static_slot_start + len(pieces), self.static_slot_start + self.max_partners, dtype=np.int32
                ),
            ]
        )
        if len(surplus):
            far = np.full(len(surplus), FAR_AWAY, np.float32)
            zero = np.zeros_like(far)
            self.env.set_agent_states(surplus, far, far, far, zero, zero, zero, zero, zero)

    def _place_partners(self, elapsed_s):
        """Write the static partners into their slots, extrapolated elapsed_s ahead at constant velocity."""
        if self.partners is not None:
            self._place(self.partners, elapsed_s)

    def _place(self, partners, elapsed_s):
        # the policy sees the actors as they are now, and in the rollout the slow ones as standing still
        still = ~partners["moving"] & (elapsed_s > 0)
        vx = np.where(still, 0.0, partners["vx"])
        vy = np.where(still, 0.0, partners["vy"])
        stopped = np.hypot(vx, vy) < float(binding.AGENT_STOPPED_SPEED_THRESHOLD)
        zero = np.zeros(len(partners["idx"]), np.float32)
        self.env.set_agent_states(
            partners["idx"],
            (partners["x"] + vx * elapsed_s).astype(np.float32),
            (partners["y"] + vy * elapsed_s).astype(np.float32),
            partners["z"],
            partners["heading"],
            vx.astype(np.float32),
            vy.astype(np.float32),
            zero,
            zero,
            (partners["stopped_s"] + stopped * elapsed_s).astype(np.float32),
        )

    def _advance_goal_inference(self, actors):
        """Advance goal inference for the nearest moving tracked actors no larger than the policy was trained with
        until the actor slots are full; returns the ids of those with a way ahead, which the joint rollout drives."""
        predicted_ids, advanced_ids = [], []
        for actor in actors:
            if len(predicted_ids) == self.actor_slot_count:
                break
            if actor["id"] < 0 or not self._moving(actor) or len(split_actor(actor)) > 1:
                continue
            bx, by = self.map.to_bin(float(actor["x"]), float(actor["y"]))
            heading = self.map.heading_to_bin(float(actor["heading"]))
            advanced_ids.append(actor["id"])
            if self.goal_inference.advance(actor["id"], float(bx), float(by), float(heading)):
                predicted_ids.append(actor["id"])
        self.goal_inference.retain(advanced_ids)
        return set(predicted_ids)

    def _update_goal_beliefs(self):
        """Update the predicted actors' beliefs with the critic values of their goal windows."""
        if self.predicted is None:
            return
        inference = self.goal_inference
        agent_count = 1 + self.actor_slot_count
        windows = np.zeros((agent_count, inference.hypothesis_count, inference.num_goals, GOAL_COLUMNS), np.float32)
        counts = np.zeros((agent_count, inference.hypothesis_count), np.int32)
        for row, actor_id in zip(self.predicted["idx"], self.predicted["ids"]):
            windows[row], counts[row] = inference.windows(actor_id)
            windows[row, ..., :2] -= self.world_minus_sim_xy
        obs = np.zeros((*counts.shape, self.env.observations.shape[1]), np.float32)
        self.env.goal_window_observations(windows, counts, obs)
        values = np.zeros(counts.shape, np.float32)
        valid = counts > 0
        if valid.any():
            with torch.no_grad():
                _, value = self.policy.forward_eval(torch.as_tensor(obs[valid]).to(self.device))
            values[valid] = value.reshape(-1).float().cpu().numpy()
        for row, actor_id in zip(self.predicted["idx"], self.predicted["ids"]):
            inference.update(actor_id, values[row])

    def _set_lights(self, light_states):
        if not self.map.num_traffic:
            return
        states = np.array(
            [int(light_states.get(light_id, LIGHT_STATE_GREEN)) for light_id in self.map.element_light_ids], np.int32
        )
        self.env.set_traffic_light_states(states)

    def _infer(self, obs):
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
        return obs_tensor, act, logits, value

    def _ego_motion(self):
        """Speed [m/s], acceleration [m/s^2], steering angle [rad] and curvature [1/m] of the shadow ego."""
        ego_obs = np.asarray(self.env.observations)[EGO_SLOT]
        steering_angle = float(ego_obs[EGO_OBS_STEERING_IDX]) * EGO_OBS_STEERING_NORM_RAD
        return (
            float(ego_obs[EGO_OBS_SPEED_IDX]) * float(self.env.obs_norm_speed_mps),
            float(ego_obs[EGO_OBS_ACCEL_LONG_IDX]) * float(binding.ACCEL_LONG_NORM),
            steering_angle,
            math.tan(steering_angle) / (WHEELBASE_LENGTH_RATIO * self.ego_length_m),
        )

    def _rollout_point(self, time_s):
        state = self.env.get_global_agent_state()
        x, y = self.map.from_bin(float(state["x"][EGO_SLOT]), float(state["y"][EGO_SLOT]))
        speed_mps, accel_mps2, _, curvature_inv_m = self._ego_motion()
        return {
            "t": time_s,
            "x": float(x),
            "y": float(y),
            "heading": float(self.map.heading_from_bin(float(state["heading"][EGO_SLOT]))),
            "speed_mps": speed_mps,
            "accel_long_mps2": accel_mps2,
            "curvature_inv_m": curvature_inv_m,
        }

    def _actor_point(self, state, slot, time_s):
        x, y = self.map.from_bin(float(state["x"][slot]), float(state["y"][slot]))
        return {
            "t": time_s,
            "x": float(x),
            "y": float(y),
            "heading": float(self.map.heading_from_bin(float(state["heading"][slot]))),
        }

    def _constant_velocity_rollout(self, actor, times_s):
        """The path of a static partner in the rollout, at constant velocity or standing still like _place_partners
        moves it."""
        vx, vy = (float(actor["vx"]), float(actor["vy"])) if self._moving(actor) else (0.0, 0.0)
        points = [
            {
                "t": time_s,
                "x": float(actor["x"]) + vx * time_s,
                "y": float(actor["y"]) + vy * time_s,
                "heading": float(actor["heading"]),
            }
            for time_s in times_s
        ]
        return {"id": actor["id"], "points": points}

    def _rollout(self, first_action):
        """The ego poses of the policy rolled ahead from now, the Simian actors held at constant velocity."""
        points = [self._rollout_point(0.0)]
        action = first_action
        self.env.predict_begin()
        try:
            for step_idx in range(self.rollout_steps):
                if step_idx:
                    action = self._infer(self.env.observations)[1]
                self.env.predict_step(action)
                # c_step holds static slots in place, so the streamed actors are moved here
                self._place_partners(elapsed_s=(step_idx + 1) * self.dt)
                self.env.recompute_observations()
                points.append(self._rollout_point((step_idx + 1) * self.dt))
        finally:
            self.env.predict_end()
        return points

    def _joint_rollout(self):
        """The ego poses of the policy rolled ahead from now together with the predicted actors, each driving to its
        likeliest inferred goals, the other actors held at constant velocity; and the predicted actors' paths with the
        goals they drive to."""
        agent_count = 1 + self.actor_slot_count
        goal_lists = np.zeros((agent_count, binding.MAX_GOALS, GOAL_COLUMNS), np.float32)
        list_counts = np.zeros(agent_count, np.int32)
        predicted = [] if self.predicted is None else list(zip(self.predicted["idx"], self.predicted["ids"]))
        actor_goals = {}
        for row, actor_id in predicted:
            goal_list = self.goal_inference.likeliest_goal_list(actor_id)
            goals_x, goals_y = self.map.from_bin(goal_list[:, 0].astype(np.float64), goal_list[:, 1].astype(np.float64))
            actor_goals[actor_id] = [{"x": float(x), "y": float(y)} for x, y in zip(goals_x, goals_y)]
            goal_list[:, :2] -= self.world_minus_sim_xy
            goal_lists[row, : len(goal_list)] = goal_list
            list_counts[row] = len(goal_list)
        state = self.env.get_global_agent_state()
        points = [self._rollout_point(0.0)]
        actor_paths = {actor_id: [self._actor_point(state, row, 0.0)] for row, actor_id in predicted}
        self.env.predict_begin()
        try:
            self.env.predict_set_goals(goal_lists, list_counts)
            for step_idx in range(self.rollout_steps):
                self.env.predict_step(self._infer(self.env.observations)[1])
                # c_step holds static slots in place, so the streamed actors are moved here
                self._place_partners(elapsed_s=(step_idx + 1) * self.dt)
                self.env.recompute_observations()
                time_s = (step_idx + 1) * self.dt
                points.append(self._rollout_point(time_s))
                state = self.env.get_global_agent_state()
                for row, actor_id in predicted:
                    actor_paths[actor_id].append(self._actor_point(state, row, time_s))
        finally:
            self.env.predict_end()
        return points, [
            {"id": actor_id, "points": path, "goals": actor_goals[actor_id]} for actor_id, path in actor_paths.items()
        ]

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
        actors = sorted(
            request["actors"], key=lambda a: (a["x"] - float(ego["x"])) ** 2 + (a["y"] - float(ego["y"])) ** 2
        )[: self.max_partners]
        predicted_ids = set() if self.goal_inference is None else self._advance_goal_inference(actors)
        self._set_partners(actors, predicted_ids)
        self._set_lights(request["traffic_light_states"])
        self.goal_window.sync(ex, ey, self.map.heading_to_bin(float(ego["heading"])))
        obs = np.asarray(self.env.recompute_observations())
        if self.goal_inference is not None:
            self._update_goal_beliefs()
        start_speed_mps = float(obs[EGO_SLOT][EGO_OBS_SPEED_IDX]) * float(self.env.obs_norm_speed_mps)

        obs_tensor, act, logits, value = self._infer(obs)
        debug = self._debug(obs_tensor, logits, value)
        if self.rollout_steps:
            if self.goal_inference is None:
                points, actor_rollouts = self._rollout(act), []
            else:
                points, actor_rollouts = self._joint_rollout()
            times_s = [point["t"] for point in points]
            actor_rollouts += [
                self._constant_velocity_rollout(actor, times_s) for actor in self.static_actors if actor["id"] >= 0
            ]
            self.reference = {"time_ns": int(request["time_ns"]), "dt": self.dt, "points": points}
            debug["rollout"] = points
            debug["actor_rollouts"] = actor_rollouts
        self.env.step(act)

        speed_mps, accel_mps2, steering_angle, curvature_inv_m = self._ego_motion()
        current_goal_idx, goal_count = self.env.get_agent_goal_progress(EGO_SLOT)
        return {
            "dt": self.dt,
            "start_speed_mps": start_speed_mps,
            "speed_mps": speed_mps,
            "accel_long_mps2": accel_mps2,
            "steering_angle_rad": steering_angle,
            "curvature_inv_m": curvature_inv_m,
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
        self.step_count = 0
        self.reference_miss_count = 0
        self.reference_miss_logged_s = -math.inf
        self.reference_served_count = 0


class PolicyHandler(BaseHTTPRequestHandler):
    def _reply(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/healthz":
            self._reply(200, {"status": "ok"})
            return
        if self.path != "/reference":
            self._reply(404, {"error": f"unknown path {self.path}"})
            return
        # not under the lock, which a step holds while AD reads within its planning tick
        server = self.server
        session = server.session
        reference = None if session is None else session.reference
        if reference is None:
            self._reference_miss(session)
            self._reply(404, {"error": "no rollout yet: POST /reset and /step with --rollout-horizon-s"})
            return
        server.reference_served_count += 1
        self._reply(200, reference)

    def _reference_miss(self, session):
        server = self.server
        server.reference_miss_count += 1
        now_s = time.monotonic()
        if now_s - server.reference_miss_logged_s < REFERENCE_MISS_LOG_INTERVAL_S:
            return
        server.reference_miss_logged_s = now_s
        if session is None:
            reason = "no client has reset the policy"
        elif not session.rollout_steps:
            reason = "--rollout-horizon-s is 0"
        else:
            reason = "no client has stepped the policy since its reset"
        log(f"GET /reference without a rollout ({server.reference_miss_count} so far): {reason}")

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        server = self.server
        try:
            with server.lock:
                if self.path == "/reset":
                    start_s = time.monotonic()
                    server.session = PolicySession(request, server.args, server.work_dir)
                    session = server.session
                    server.step_count = 0
                    ego = request["ego"]
                    log(
                        f"reset in {time.monotonic() - start_s:.1f} s: ego at ({ego['x']:.1f}, {ego['y']:.1f}) "
                        f"heading {ego['heading']:+.2f} rad, {'drawn' if server.args.drawn_map else 'Simian'} map "
                        f"with {session.lane_count} lanes, {session.goal_count} goals, dt {session.dt} s, "
                        f"rollout {session.rollout_steps} steps, {server.args.actor_prediction} actor prediction"
                    )
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
                    start_s = time.monotonic()
                    response = server.session.step(request)
                    self._reply(200, response)
                    self._log_step(request, response, time.monotonic() - start_s)
                else:
                    self._reply(404, {"error": f"unknown path {self.path}"})
        except Exception as exc:
            traceback.print_exc()
            self._reply(500, {"error": f"{type(exc).__name__}: {exc}"})

    def _log_step(self, request, response, duration_s):
        server = self.server
        server.step_count += 1
        if (server.step_count - 1) % STEP_LOG_INTERVAL_STEPS:
            return
        ego = request["ego"]
        message = (
            f"step {server.step_count} in {1000 * duration_s:.0f} ms: "
            f"speed {math.hypot(ego['vx'], ego['vy']):.2f} -> {response['speed_mps']:.2f} m/s, "
            f"accel {response['accel_long_mps2']:+.2f} m/s^2, curvature {response['curvature_inv_m']:+.4f} 1/m, "
            f"{len(request['actors'])} actors, goal {response['goal_index']}/{response['goal_count']}"
        )
        reference = server.session.reference
        if reference is not None:
            length_m, forward_s, end_speed_mps = rollout_summary(reference["points"])
            message += (
                f", rollout at {reference['time_ns'] / NS_PER_S:.1f} s: {length_m:.1f} m forward over {forward_s:.1f} s, "
                f"end speed {end_speed_mps:+.2f} m/s, served {server.reference_served_count} times"
            )
        log(message)

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
    parser.add_argument(
        "--rollout-horizon-s",
        type=float,
        default=DEFAULT_ROLLOUT_HORIZON_S,
        help="seconds the policy is rolled ahead every step for GET /reference, 0 disables",
    )
    parser.add_argument(
        "--actor-prediction",
        choices=(ACTOR_PREDICTION_CONSTANT_VELOCITY, ACTOR_PREDICTION_JOINT),
        default=ACTOR_PREDICTION_CONSTANT_VELOCITY,
        help="how the rollout moves the actors: at constant velocity, or driven by the policy to inferred goals",
    )
    parser.add_argument(
        "--joint-prediction-actors",
        type=int,
        default=DEFAULT_JOINT_PREDICTION_ACTORS,
        help="nearest tracked actors the joint rollout drives",
    )
    parser.add_argument(
        "--min-moving-speed-mps",
        type=float,
        default=DEFAULT_MIN_MOVING_SPEED_MPS,
        help="estimated speed from which an actor moves in the rollout, and a tracked one can be driven by the policy",
    )
    parser.add_argument("--goal-inference-hypotheses", type=int, default=DEFAULT_GOAL_INFERENCE_HYPOTHESES)
    parser.add_argument("--goal-inference-goal-spacing-m", type=float, default=DEFAULT_GOAL_INFERENCE_GOAL_SPACING_M)
    parser.add_argument("--goal-inference-update-s", type=float, default=DEFAULT_GOAL_INFERENCE_UPDATE_S)
    parser.add_argument("--goal-inference-beta", type=float, default=DEFAULT_GOAL_INFERENCE_BETA)
    parser.add_argument("--goal-inference-forgetting", type=float, default=DEFAULT_GOAL_INFERENCE_FORGETTING)
    args = parser.parse_args()
    if bool(args.drawn_map) != bool(args.drawn_map_spec):
        parser.error("--drawn-map and --drawn-map-spec go together")
    if any(size is not None and not size > 0 for size in (args.ego_length_m, args.ego_width_m)):
        parser.error("--ego-length-m and --ego-width-m must be positive")
    if not args.rollout_horizon_s >= 0:
        parser.error("--rollout-horizon-s must not be negative")
    if args.actor_prediction == ACTOR_PREDICTION_JOINT and not args.rollout_horizon_s > 0:
        parser.error("--actor-prediction joint needs --rollout-horizon-s")
    if not 0 < args.joint_prediction_actors <= args.max_partners:
        parser.error("--joint-prediction-actors must be in 1..--max-partners")
    if not args.min_moving_speed_mps >= 0:
        parser.error("--min-moving-speed-mps must not be negative")
    if args.goal_inference_hypotheses < 1:
        parser.error("--goal-inference-hypotheses must be positive")
    if not (
        args.goal_inference_goal_spacing_m > 0 and args.goal_inference_update_s > 0 and args.goal_inference_beta >= 0
    ):
        parser.error("--goal-inference-goal-spacing-m and -update-s must be positive and -beta not negative")
    if not 0 <= args.goal_inference_forgetting <= 1:
        parser.error("--goal-inference-forgetting must be in [0, 1]")
    torch.set_num_threads(1)
    server = PolicyServer((args.host, args.port), args)
    log(
        f"serving {args.checkpoint} on {args.host}:{args.port}, rollout {args.rollout_horizon_s} s "
        f"{'enabled' if args.rollout_horizon_s else 'disabled'}, {args.actor_prediction} actor prediction"
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
