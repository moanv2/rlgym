"""Build the list of terminal conditions from config.

Tick-time conversions:
    physics fps = 120, default tick_skip = 8, so 1 step ≈ 1/15 s
    timeout_seconds * 120 = timeout_ticks
"""
from __future__ import annotations

from typing import Any


def build_terminal_conditions(config: dict[str, Any], tick_skip: int) -> list:
    from rlgym_sim.utils.terminal_conditions.common_conditions import (
        GoalScoredCondition,
        NoTouchTimeoutCondition,
        TimeoutCondition,
    )

    fps = 120
    conds = []

    if config.get("goal_scored", True):
        conds.append(GoalScoredCondition())

    no_touch_s = config.get("no_touch_timeout_seconds")
    if no_touch_s is not None:
        ticks = int(no_touch_s * fps / tick_skip)
        conds.append(NoTouchTimeoutCondition(ticks))

    timeout_s = config.get("timeout_seconds")
    if timeout_s is not None:
        ticks = int(timeout_s * fps / tick_skip)
        conds.append(TimeoutCondition(ticks))

    if not conds:
        raise ValueError("terminal: at least one condition must be enabled")

    return conds
