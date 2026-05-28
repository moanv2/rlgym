"""Reward registry. Add new rewards by registering them with @REWARDS.register('name')."""
# Importing these modules triggers their @REWARDS.register decorators.
from rlbot.rewards import (
    builtin,  # noqa: F401
    custom,  # noqa: F401
)
from rlbot.rewards.builder import build_reward
from rlbot.rewards.registry import REWARDS
from rlbot.rewards.zero_sum import ZeroSumReward

__all__ = ["REWARDS", "ZeroSumReward", "build_reward"]
