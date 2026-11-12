"""GoalInference with the action likelihood: an actor's belief moves to the route whose mean policy jerk matches the
jerk it drives, estimated from its speed and heading."""

import numpy as np
import pytest

from pufferlib.ocean.cosim import joint_prediction as jp

DT_S = 0.3
ACCEL_MPS2 = 1.0  # steady, so the jerk the actor drives is zero
STEADY = np.array([0.0, 0.0], np.float32)
BRAKING = np.array([-4.0, 0.0], np.float32)


class _TwoBranchGraph:
    """Two ways ahead along x, far enough that no goal is reached."""

    def __init__(self, _roads):
        pass

    def branch_goals(self, x, y, heading, spacing_m, goal_count, max_branches):
        return [
            [(x + 100.0 + spacing_m * k, y + offset, 0.0, lane) for k in range(goal_count)]
            for lane, offset in enumerate((0.0, 20.0))
        ][:max_branches]


@pytest.fixture
def inference(monkeypatch):
    from data_utils import mirror_map_bin

    monkeypatch.setattr(mirror_map_bin, "read_bin", lambda _path: {"roads": None})
    monkeypatch.setattr(jp.drive_route_graph, "RouteGraph", _TwoBranchGraph)
    return jp.GoalInference(
        "unused.bin",
        num_goals=3,
        hypothesis_count=2,
        goal_spacing_m=20.0,
        update_steps=100,
        beta=10.0,
        forgetting=0.01,
        gamma=0.99,
        goal_radius_m=2.0,
        reward_goal=1.0,
        horizon_s=6.0,
        likelihood=jp.LIKELIHOOD_ACTION,
        jerk_std=(0.3, 0.3),
    )


def drive(inference, step_count, start_s=0.0):
    """Drive actor 1 east at ACCEL_MPS2 for step_count steps, the policy steady under route 0 and braking under 1."""
    for step in range(step_count):
        time_s = start_s + step * DT_S
        speed_mps = 10.0 + ACCEL_MPS2 * time_s
        assert inference.advance(1, 10.0 * time_s + 0.5 * ACCEL_MPS2 * time_s**2, 0.0, 0.0, speed_mps, time_s)
        inference.update(1, np.zeros(2, np.float32), np.stack([STEADY, BRAKING]))


def test_the_belief_moves_to_the_route_whose_mean_jerk_matches(inference):
    drive(inference, 4)

    belief = inference.actors[1].belief
    assert belief[0] > 0.95
    assert belief.sum() == pytest.approx(1.0)


def test_the_belief_needs_two_accelerations_before_scoring(inference):
    drive(inference, 2)

    np.testing.assert_allclose(inference.actors[1].belief, [0.5, 0.5])


def test_an_actor_seen_again_at_the_same_time_fails(inference):
    drive(inference, 1)

    with pytest.raises(ValueError):
        drive(inference, 1)
