"""Vendored WeightedSampleSetter.

rlgym-tools v2 removed ``rlgym_tools.extra_state_setters``, so we vendor this small
wrapper (mirroring the original rlgym-tools API) to keep the ``weighted_sample``
state setter working. Each episode it picks ONE of its child state setters at
random, weighted by ``weights``, and delegates the reset to it.

This matches how LookupAction is vendored at ``rlbot/actions/lookup_action.py`` to
decouple from the rlgym-tools v2 API.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from rlgym_sim.utils.state_setters import StateSetter, StateWrapper


class WeightedSampleSetter(StateSetter):
    """Randomly delegates each reset to one of ``state_setters``, by ``weights``."""

    def __init__(self, state_setters: Sequence[StateSetter], weights: Sequence[float]) -> None:
        super().__init__()
        if len(state_setters) == 0:
            raise ValueError("WeightedSampleSetter needs at least one state setter")
        if len(state_setters) != len(weights):
            raise ValueError("state_setters and weights must have the same length")

        self.state_setters = list(state_setters)
        w = np.asarray(weights, dtype=np.float64)
        total = w.sum()
        if total <= 0:
            raise ValueError("WeightedSampleSetter weights must sum to a positive value")
        self.probs = w / total

    def reset(self, state_wrapper: StateWrapper) -> None:
        idx = int(np.random.choice(len(self.state_setters), p=self.probs))
        self.state_setters[idx].reset(state_wrapper)
