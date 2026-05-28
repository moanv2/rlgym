"""Local stand-in for rlgym-tools' ``WeightedSampleSetter``.

rlgym-tools v2 reorganized into ``rlgym_tools.rocket_league.state_mutators``, and the
old ``rlgym_tools.extra_state_setters.weighted_sample_setter`` import path no longer
exists in the installed package. This is a small, dependency-free replacement: each
``reset`` picks one child setter to delegate to, sampled with the given weights.

We deliberately keep the YAML name ``weighted_sample`` for backward compatibility so
exp_007's existing config continues to work without an edit.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from rlgym_sim.utils.state_setters import StateSetter
from rlgym_sim.utils.state_setters.wrappers import StateWrapper


class WeightedRandomSetter(StateSetter):
    """Delegates each reset to exactly one of its children, sampled by weight."""

    def __init__(self, state_setters: Sequence[StateSetter], weights: Sequence[float]) -> None:
        if len(state_setters) == 0:
            raise ValueError("WeightedRandomSetter needs at least one child setter")
        if len(state_setters) != len(weights):
            raise ValueError("state_setters and weights must have the same length")
        w = np.asarray(weights, dtype=float)
        if np.any(w < 0) or w.sum() <= 0:
            raise ValueError("weights must be non-negative and have a positive sum")
        self.state_setters: tuple[StateSetter, ...] = tuple(state_setters)
        # Normalized once at construction; no re-normalization per call.
        self._probs: np.ndarray = w / w.sum()

    def reset(self, state_wrapper: StateWrapper) -> None:
        idx = int(np.random.choice(len(self.state_setters), p=self._probs))
        self.state_setters[idx].reset(state_wrapper)
