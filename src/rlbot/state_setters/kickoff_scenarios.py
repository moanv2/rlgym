"""Kickoff state setters using the canonical 5 Rocket League kickoff spots.

The 5 standard kickoff positions are encoded in rl_constants.py. rlgym_sim's
DefaultState also produces standard kickoffs, but this version exposes more
control:

  - RandomKickoffSetter   — pick one of 5 kickoff spots uniformly
  - FixedKickoffSetter    — always start from a specific kickoff index (for
                            scenario-focused training)

Combine with rlgym_sim's RandomState (general exploration) via
rlbot.state_setters.weighted_sample_setter.WeightedSampleSetter to give the
bot a mix of structured kickoff training and broad state coverage.
"""
from __future__ import annotations

import numpy as np
from rlgym_sim.utils.state_setters import StateSetter
from rlgym_sim.utils.state_setters.wrappers import StateWrapper

from rlbot.utils.rl_constants import (
    BALL_HEIGHT_AT_REST,
    KICKOFF_BOOST,
    KICKOFF_POSITIONS_BLUE,
    KICKOFF_POSITIONS_ORANGE,
    OCTANE_HEIGHT_AT_REST,
)


def _set_kickoff_positions(state: StateWrapper, kickoff_index: int) -> None:
    """Place ball at center and cars at the kickoff_index-th standard spot."""
    # Ball at center
    state.ball.set_pos(0.0, 0.0, BALL_HEIGHT_AT_REST)
    state.ball.set_lin_vel(0.0, 0.0, 0.0)
    state.ball.set_ang_vel(0.0, 0.0, 0.0)

    for car in state.cars:
        if car.team_num == 0:  # blue
            x, y, yaw = KICKOFF_POSITIONS_BLUE[kickoff_index]
        else:  # orange
            x, y, yaw = KICKOFF_POSITIONS_ORANGE[kickoff_index]
        car.set_pos(x, y, OCTANE_HEIGHT_AT_REST)
        car.set_rot(0.0, yaw, 0.0)
        car.set_lin_vel(0.0, 0.0, 0.0)
        car.set_ang_vel(0.0, 0.0, 0.0)
        car.boost = KICKOFF_BOOST


class RandomKickoffSetter(StateSetter):
    """Spawns at one of the 5 standard kickoff positions, chosen uniformly.

    Both teams start at the mirrored kickoff position so the episode is a
    proper symmetrical kickoff. Boost is set to the canonical 33-unit
    starting amount.
    """

    def reset(self, state: StateWrapper) -> None:
        idx = int(np.random.randint(0, 5))
        _set_kickoff_positions(state, idx)


class FixedKickoffSetter(StateSetter):
    """Always spawn from a specific kickoff index (0–4).

    Use for scenario-focused training: e.g., FixedKickoffSetter(4) hammers
    only the far-back-center kickoff if you want the bot to specialize on it.

    Indices map to:
        0 → right corner
        1 → left corner
        2 → back right
        3 → back left
        4 → far back center
    """

    def __init__(self, kickoff_index: int):
        if not 0 <= kickoff_index < 5:
            raise ValueError(f"kickoff_index must be 0..4, got {kickoff_index}")
        self.kickoff_index = kickoff_index

    def reset(self, state: StateWrapper) -> None:
        _set_kickoff_positions(state, self.kickoff_index)
