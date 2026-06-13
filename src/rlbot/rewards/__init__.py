"""Reward registry. Add new rewards by registering them with @REWARDS.register('name')."""
from rlbot.rewards.builder import build_reward
from rlbot.rewards.registry import REWARDS
from rlbot.rewards.zero_sum import ZeroSumReward

# Importing these modules triggers their @REWARDS.register decorators.
from rlbot.rewards import builtin  # noqa: F401
from rlbot.rewards import advanced  # noqa: F401  (aerial_touch, supersonic, ball_away_from_own_goal, save_boost, align_ball_goal)

__all__ = ["REWARDS", "ZeroSumReward", "build_reward"]
