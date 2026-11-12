import re
from pathlib import Path

from pufferlib.ocean.cosim.simian.server import ROUTE_ONLY_REWARDS

BINDING_PATH = Path(__file__).resolve().parents[2] / "pufferlib" / "ocean" / "drive" / "binding.c"
ROUTE_ONLY_PATTERN = re.compile(r'"(\w+) requires goal_source=route"')


def test_server_disables_every_route_only_reward():
    route_only = set(ROUTE_ONLY_PATTERN.findall(BINDING_PATH.read_text()))
    assert route_only
    assert route_only == set(ROUTE_ONLY_REWARDS)
