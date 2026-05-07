"""Build a CombinedReward from a YAML reward config.

Config schema (excerpt):
    rewards:
      team_spirit: 0.0
      opp_scale: 1.0
      components:
        - name: velocity_player_to_ball
          weight: 0.05
        - name: velocity_ball_to_goal
          weight: 0.5
        - name: event
          weight: 10.0
          kwargs:
            goal: 1.0
            concede: -1.0
            demo: 0.1
"""
from __future__ import annotations

from typing import Any

from rlbot.rewards.registry import REWARDS
from rlbot.rewards.zero_sum import ZeroSumReward


def build_reward(config: dict[str, Any]):
    """Build the top-level reward function used by the env."""
    from rlgym_sim.utils.reward_functions import CombinedReward

    components = config.get("components", [])
    if not components:
        raise ValueError("rewards.components must be a non-empty list")

    fns = []
    weights = []
    for spec in components:
        name = spec["name"]
        weight = float(spec.get("weight", 1.0))
        kwargs = spec.get("kwargs", {}) or {}
        cls = REWARDS.get(name)
        fns.append(cls(**kwargs))
        weights.append(weight)

    combined = CombinedReward(reward_functions=tuple(fns), reward_weights=tuple(weights))

    if config.get("zero_sum", True):
        return ZeroSumReward(
            combined,
            team_spirit=float(config.get("team_spirit", 0.0)),
            opp_scale=float(config.get("opp_scale", 1.0)),
        )
    return combined
