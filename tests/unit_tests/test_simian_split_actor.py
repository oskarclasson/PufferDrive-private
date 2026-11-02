"""split_actor: an actor larger than the agents the policy trains with becomes the fewest equal pieces that fit, tiling
its box and moving like it; one that fits is passed through."""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pufferlib.ocean.cosim.simian.server import MAX_PARTNER_LENGTH_M, MAX_PARTNER_WIDTH_M, split_actor


def actor(length, width, heading=0.0):
    return {
        "id": 3,
        "x": 10.0,
        "y": 20.0,
        "z": 0.0,
        "heading": heading,
        "vx": 1.0,
        "vy": 0.5,
        "length": length,
        "width": width,
    }


def test_an_actor_that_fits_is_kept():
    small = actor(MAX_PARTNER_LENGTH_M, MAX_PARTNER_WIDTH_M)
    assert split_actor(small) == [small]


def test_a_long_actor_is_split_along_its_heading():
    pieces = split_actor(actor(16.0, 2.5, heading=math.pi / 2))

    assert len(pieces) == 3
    for piece in pieces:
        assert piece["length"] == pytest.approx(16.0 / 3)
        assert piece["width"] == pytest.approx(2.5)
        assert piece["x"] == pytest.approx(10.0)
        assert (piece["id"], piece["vx"], piece["vy"], piece["heading"]) == (3, 1.0, 0.5, math.pi / 2)
    assert sorted(piece["y"] for piece in pieces) == pytest.approx([20.0 - 16.0 / 3, 20.0, 20.0 + 16.0 / 3])


def test_a_large_actor_is_tiled_both_ways():
    pieces = split_actor(actor(10.0, 4.0))

    assert len(pieces) == 4
    assert {(round(p["x"], 6), round(p["y"], 6)) for p in pieces} == {
        (7.5, 19.0),
        (7.5, 21.0),
        (12.5, 19.0),
        (12.5, 21.0),
    }
    assert all(p["length"] == pytest.approx(5.0) and p["width"] == pytest.approx(2.0) for p in pieces)
