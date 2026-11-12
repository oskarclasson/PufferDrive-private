import math
from types import SimpleNamespace

import pytest

from pufferlib.ocean.cosim.simian.server import NS_PER_S, PolicySession

START_NS = 100 * int(NS_PER_S)


def rollout_point(t, x, heading, speed_mps):
    return {
        "t": t,
        "x": x,
        "y": 0.0,
        "heading": heading,
        "speed_mps": speed_mps,
        "accel_long_mps2": speed_mps / 10,
        "curvature_inv_m": speed_mps / 100,
    }


def session(points):
    return SimpleNamespace(reference={"time_ns": START_NS, "dt": 0.3, "points": points})


def test_open_loop_anchor_is_the_previous_rollout_at_the_step_time():
    points = [rollout_point(0.0, 0.0, 0.0, 2.0), rollout_point(0.3, 0.6, 0.3, 4.0)]

    x, y, heading, speed_mps, accel_mps2, curvature_inv_m = PolicySession._open_loop_anchor(
        session(points), START_NS + int(0.2 * NS_PER_S)
    )

    assert x == pytest.approx(0.4)
    assert y == pytest.approx(0.0)
    assert heading == pytest.approx(0.2)
    assert speed_mps == pytest.approx(10 / 3)
    assert accel_mps2 == pytest.approx(1 / 3)
    assert curvature_inv_m == pytest.approx(1 / 30)


def test_open_loop_anchor_turns_the_shorter_way_round():
    points = [rollout_point(0.0, 0.0, 3.0, 1.0), rollout_point(0.3, 0.3, -3.0, 1.0)]

    heading = PolicySession._open_loop_anchor(session(points), START_NS + int(0.15 * NS_PER_S))[2]

    assert heading == pytest.approx(3.0 + 0.5 * (2 * math.pi - 6.0))


def test_open_loop_anchor_without_a_previous_rollout_is_none():
    assert PolicySession._open_loop_anchor(SimpleNamespace(reference=None), START_NS) is None


def test_open_loop_anchor_beyond_the_previous_rollout_fails():
    points = [rollout_point(0.0, 0.0, 0.0, 1.0), rollout_point(0.3, 0.3, 0.0, 1.0)]

    with pytest.raises(ValueError):
        PolicySession._open_loop_anchor(session(points), START_NS + int(0.4 * NS_PER_S))
