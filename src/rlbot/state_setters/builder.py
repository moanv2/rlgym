"""Pick the initial-state setter from config.

For early training, RandomState with airborne cars + random ball velocity gets the
bot exposed to a much wider distribution of states than kickoff alone. For curriculum
stages where the bot is clumsy in a specific situation, mix ``near_ball`` in via
``weighted_sample`` to give it repeated reps in exactly that situation.
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

    if name == "near_ball":
        # Targeted low-speed near-ball drill — see near_ball.py docstring.
        from rlbot.state_setters.near_ball import NearBallState

        # Forward every config key except the dispatch-only ones as kwargs.
        kwargs = {k: v for k, v in config.items() if k not in {"name", "weight"}}
        return NearBallState(**kwargs)

    if name == "shooting":
        # Targeted long-range shooting drill — see shooting.py docstring.
        from rlbot.state_setters.shooting import ShootingState

        # Forward every config key except the dispatch-only ones as kwargs.
        kwargs = {k: v for k, v in config.items() if k not in {"name", "weight"}}
        return ShootingState(**kwargs)

    if name == "weighted_sample":
        # Mix multiple state setters with weights — useful for curriculum.
        # We use a local replacement for rlgym-tools' WeightedSampleSetter because
        # rlgym-tools v2 moved that path away from ``extra_state_setters``.
        from rlbot.state_setters.weighted_random import WeightedRandomSetter

        components = config["components"]  # list of {name, weight, ...kwargs}
        setters = [build_state_setter(c) for c in components]
        weights = [float(c.get("weight", 1.0)) for c in components]
        return WeightedRandomSetter(state_setters=setters, weights=weights)

    raise ValueError(f"Unknown state setter: {name!r}")
