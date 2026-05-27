"""WeightedSampleSetter ported to rlgym_sim imports (rlgym-tools 1.x used `rlgym`, not `rlgym_sim`)."""
from random import choices
from typing import Sequence, Tuple, Union

from rlgym_sim.utils.state_setters import StateSetter, StateWrapper


class WeightedSampleSetter(StateSetter):
    """Samples StateSetters randomly according to their weights."""

    def __init__(self, state_setters: Sequence[StateSetter], weights: Sequence[float]):
        super().__init__()
        self.state_setters = state_setters
        self.weights = weights
        assert len(state_setters) == len(weights), (
            f"Length of state_setters should match weights, "
            f"got {len(state_setters)} and {len(weights)}."
        )

    def reset(self, state_wrapper: StateWrapper):
        choices(self.state_setters, weights=self.weights)[0].reset(state_wrapper)
