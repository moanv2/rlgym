"""Lock the rl_constants ↔ rlgym_sim equivalence.

custom.py was migrated to source field/physics constants from
``rlbot.utils.rl_constants`` instead of ``rlgym_sim.utils.common_values``. That
migration is only safe if the values are identical — this test guards against
silent drift (e.g. a typo'd goal-back Y that would skew shot/aim rewards).
"""
from __future__ import annotations

import pytest

pytest.importorskip("rlgym_sim")

import rlgym_sim.utils.common_values as cv

from rlbot.utils import rl_constants as rc

# Every name custom.py imports from rl_constants must equal rlgym_sim's value.
_SHARED_NAMES = [
    "BACK_WALL_Y",
    "BALL_MAX_SPEED",
    "BLUE_GOAL_BACK",
    "ORANGE_GOAL_BACK",
    "BLUE_TEAM",
    "ORANGE_TEAM",
    "CAR_MAX_SPEED",
    "CEILING_Z",
]


@pytest.mark.parametrize("name", _SHARED_NAMES)
def test_rl_constants_match_rlgym_sim(name):
    ours = getattr(rc, name)
    theirs = getattr(cv, name)
    if isinstance(theirs, (list, tuple)):
        assert tuple(ours) == pytest.approx(tuple(theirs))
    else:
        assert ours == pytest.approx(theirs)
