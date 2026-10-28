"""Decode one ego observation for visualization in simian-adapter's Foxglove topics, like pufferlib.viz.

Slots are in the ego frame of the observation (x forward, y left, meters, radians), only the valid ones per the
valid-count block at the end of the observation, each with its encoder max-pool wins.
"""

import math

import numpy as np

from pufferlib.ocean.drive import binding

VALID_COUNT_FEATURES = 4
STEERING_LIMIT_RAD = 0.667  # STEERING_LIMIT in constants.h
ACCEL_LAT_LIMIT_MPS2 = 4.0  # ACCEL_LAT_LIMIT[1] in constants.h
LANE_DISTANCE_NORMALIZATION_M = 4.0  # LANE_DISTANCE_NORMALIZATION in constants.h
MAX_STOPPED_SECONDS = 60.0  # MAX_STOPPED_SECONDS in constants.h
LANE_CURVATURE_NORM_INV_M = 0.2  # LANE_CURVATURE_NORM in constants.h
CONDITIONING_LABELS = (
    "goal_radius",
    "goal_speed",
    "collision",
    "offroad",
    "comfort",
    "lane_align",
    "vel_align",
    "lane_center",
    "center_bias",
    "velocity",
    "reverse",
    "stop_line",
    "timestep",
    "overspeed",
    "c_throttle",
    "c_steer",
    "c_acc",
    "c_speed",
)
POOL_NAMES = ("pool_partner", "pool_lane", "pool_boundary", "pool_traffic")


def _ego_features(env):
    """(name, scale, unit) of each ego feature in write_ego_obs order (drive.h)."""
    lane_angle = ("lane_heading_error", math.pi, "rad") if env.obs_lane_heading_signed else ("lane_angle_cos", 1.0, "")
    return (
        ("speed", env.obs_norm_speed_mps, "m/s"),
        ("width", env.obs_norm_veh_width_m, "m"),
        ("length", env.obs_norm_veh_length_m, "m"),
        ("steering_angle", STEERING_LIMIT_RAD, "rad"),
        ("accel_long", float(binding.ACCEL_LONG_NORM), "m/s2"),
        ("accel_lat", ACCEL_LAT_LIMIT_MPS2, "m/s2"),
        ("lane_dist", LANE_DISTANCE_NORMALIZATION_M, "m"),
        lane_angle,
        ("lane_speed_limit", env.obs_norm_speed_mps, "m/s"),
        ("seconds_stopped", MAX_STOPPED_SECONDS, "s"),
        ("lane_curvature", LANE_CURVATURE_NORM_INV_M, "1/m"),
    )


def _blocks(env, obs):
    sizes = [
        ("ego", env.ego_features),
        ("conditioning", env.num_reward_coefs),
        ("goals", env.goal_dim),
        ("partners", env.obs_slots_partners_n * env.partner_features),
        ("lanes", env.obs_slots_lane_kept * env.lane_features),
        ("boundaries", env.obs_slots_boundary_kept * env.boundary_features),
        ("traffic", env.obs_slots_traffic_controls_n * env.traffic_control_features),
        ("counts", VALID_COUNT_FEATURES),
    ]
    blocks, offset = {}, 0
    for name, size in sizes:
        blocks[name] = obs[offset : offset + size]
        offset += size
    if offset != len(obs):
        raise ValueError(f"observation has {len(obs)} values, layout expects {offset}")
    return blocks


def _road_slots(rows, count, pool, env, has_lane_features):
    xy_norm = env.obs_norm_xy_offset_m
    slots = []
    for slot_idx, row in enumerate(rows[:count]):
        mx, my = row[0] * xy_norm, row[1] * xy_norm
        half_length = row[3] * env.obs_norm_road_seg_length_m
        slot = {
            "x1": mx - row[4] * half_length,
            "y1": my - row[5] * half_length,
            "x2": mx + row[4] * half_length,
            "y2": my + row[5] * half_length,
            "pool": int(pool[slot_idx]),
        }
        if has_lane_features:
            slot.update(width=row[6] * env.obs_norm_road_seg_width_m, gps_abs=row[7], gps_rel=row[8])
        slots.append(slot)
    return slots


def decode_observation(env, obs, pool):
    """obs: one agent's observation row; pool: {POOL_NAMES: per-slot max-pool win counts} of that agent."""
    obs = np.asarray(obs, dtype=np.float64)
    blocks = _blocks(env, obs)
    lane_count, boundary_count, partner_count, traffic_count = blocks["counts"].astype(int)
    xy_norm = env.obs_norm_xy_offset_m

    ego = [
        {"name": name, "normalized": float(value), "value": float(value) * scale, "unit": unit}
        for (name, scale, unit), value in zip(_ego_features(env), blocks["ego"])
    ]
    conditioning = [
        {"name": name, "normalized": float(value)} for name, value in zip(CONDITIONING_LABELS, blocks["conditioning"])
    ]
    goals = [
        {
            "x": float(row[0]) * env.obs_norm_goal_offset_m,
            "y": float(row[1]) * env.obs_norm_goal_offset_m,
            "z": float(row[2]) * env.obs_norm_z_m,
            "zero": bool(not row.any()),
        }
        for row in blocks["goals"].reshape(env.num_goals, -1)
    ]

    partners = []
    partner_rows = blocks["partners"].reshape(env.obs_slots_partners_n, env.partner_features)
    for slot_idx, row in enumerate(partner_rows[:partner_count]):
        partner = {
            "x": row[0] * xy_norm,
            "y": row[1] * xy_norm,
            "heading": math.atan2(row[6], row[5]),
            "length": row[3] * env.obs_norm_veh_length_m,
            "width": row[4] * env.obs_norm_veh_width_m,
            "speed_mps": row[7] * env.obs_norm_speed_mps,
            "seconds_stopped": row[8] * MAX_STOPPED_SECONDS,
            "pool": int(pool["pool_partner"][slot_idx]),
        }
        if env.partner_features > binding.PARTNER_FEATURES:
            partner.update(rel_vx=row[9] * env.obs_norm_speed_mps, rel_vy=row[10] * env.obs_norm_speed_mps)
        partners.append(partner)

    traffic_controls = []
    traffic_rows = blocks["traffic"].reshape(env.obs_slots_traffic_controls_n, env.traffic_control_features)
    for slot_idx, row in enumerate(traffic_rows[:traffic_count]):
        traffic_controls.append(
            {
                "x1": row[0] * xy_norm,
                "y1": row[1] * xy_norm,
                "x2": row[2] * xy_norm,
                "y2": row[3] * xy_norm,
                "type": int(round(row[5])),
                "state": int(round(row[6])),
                "pool": int(pool["pool_traffic"][slot_idx]),
            }
        )

    lanes = blocks["lanes"].reshape(env.obs_slots_lane_kept, env.lane_features)
    boundaries = blocks["boundaries"].reshape(env.obs_slots_boundary_kept, env.boundary_features)
    return {
        "ego": ego,
        "conditioning": conditioning,
        "goals": goals,
        "partners": partners,
        "lanes": _road_slots(lanes, lane_count, pool["pool_lane"], env, has_lane_features=True),
        "boundaries": _road_slots(boundaries, boundary_count, pool["pool_boundary"], env, has_lane_features=False),
        "traffic_controls": traffic_controls,
        "counts": {
            "lanes": int(lane_count),
            "boundaries": int(boundary_count),
            "partners": int(partner_count),
            "traffic_controls": int(traffic_count),
        },
        "raw": obs.astype(np.float32).tolist(),
    }
