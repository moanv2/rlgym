"""Pick the initial-state setter from config.

For early training, RandomState with airborne cars + random ball velocity gets the
bot exposed to a much wider distribution of states than kickoff alone.
"""
from __future__ import annotations

from typing import Any


def build_state_setter(config: dict[str, Any]):
    name = config.get("name", "random")

    if name == "random":
        from rlgym_sim.utils.state_setters import RandomState

        return RandomState(
            ball_rand_speed=bool(config.get("ball_rand_speed", True)),
            cars_rand_speed=bool(config.get("cars_rand_speed", True)),
            cars_on_ground=bool(config.get("cars_on_ground", False)),
        )

    if name == "default":
        from rlgym_sim.utils.state_setters import DefaultState

        return DefaultState()

    if name == "weighted_sample":
        # Mix multiple state setters with weights — useful for curriculum.
        from rlbot.state_setters.weighted_sample_setter import WeightedSampleSetter
        components = config["components"]  # list of {name, weight, kwargs}
        setters = [build_state_setter(c) for c in components]
        weights = [float(c.get("weight", 1.0)) for c in components]
        return WeightedSampleSetter(state_setters=setters, weights=weights)

    if name == "kickoff":
        from rlbot.state_setters.curriculum import KickoffState
        return KickoffState()

    if name == "defensive":
        from rlbot.state_setters.curriculum import DefensiveState
        return DefensiveState()

    if name == "aerial":
        from rlbot.state_setters.curriculum import AerialState
        return AerialState()

    raise ValueError(f"Unknown state setter: {name!r}")
