"""Kinesthetic Reward Combination (KRC) — from Lucy-SKG (arXiv:2305.15801).

Lucy-SKG beat the reigning champion bots (Necto, Nexto) and did so ~5x more
sample-efficiently. One of its core contributions is KRC: instead of combining
reward components with a weighted SUM, it combines them with a signed GEOMETRIC
MEAN:

    R = sign * (|r_1| * |r_2| * ... * |r_n|) ** (1/n)
    where sign = +1 if ALL r_i > 0, else -1

Why this beats a weighted sum:
  * A weighted sum lets the agent farm one easy component and ignore the rest.
  * The geometric mean only pays out meaningfully when the agent does MULTIPLE
    sub-skills well *at the same time* (e.g. being near the ball AND driving it
    toward the goal). It rewards compound skills, not cheese.
  * It is magnitude-robust: a tiny component drags the product down, so you don't
    get the "one reward 50x bigger than another" balancing problem of weighted sums.

Usage (Lucy-SKG's "mixed approach"): wrap a few related components in a KRC group
to form a compound skill, then linearly combine that group with simple components
in the outer CombinedReward. See configs/experiments/exp_008_krc.yaml.
"""
from __future__ import annotations

import numpy as np
from rlgym_sim.utils.reward_functions import RewardFunction


class KRCReward(RewardFunction):
    """Combines sub-reward functions via a signed geometric mean (Lucy-SKG KRC)."""

    def __init__(self, reward_functions):
        super().__init__()
        self.reward_functions = list(reward_functions)
        if not self.reward_functions:
            raise ValueError("KRCReward needs at least one sub-reward function")

    def reset(self, initial_state):
        for r in self.reward_functions:
            r.reset(initial_state)

    def pre_step(self, state):
        # Delegate so stateful sub-rewards still get their per-step hook.
        for r in self.reward_functions:
            r.pre_step(state)

    def _combine(self, values) -> float:
        n = len(values)
        # Signed geometric mean. If any component is <= 0, the whole thing goes
        # negative (you must do ALL sub-skills well). If any is exactly 0, the
        # product is 0 -> reward 0 (no compound skill achieved this step).
        prod = 1.0
        all_positive = True
        for v in values:
            prod *= abs(v)
            if v <= 0.0:
                all_positive = False
        if prod == 0.0:
            return 0.0
        magnitude = prod ** (1.0 / n)
        return magnitude if all_positive else -magnitude

    def get_reward(self, player, state, previous_action: np.ndarray) -> float:
        vals = [r.get_reward(player, state, previous_action) for r in self.reward_functions]
        return self._combine(vals)

    def get_final_reward(self, player, state, previous_action: np.ndarray) -> float:
        vals = [r.get_final_reward(player, state, previous_action) for r in self.reward_functions]
        return self._combine(vals)
