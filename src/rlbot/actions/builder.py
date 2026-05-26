"""Action parser selection. Most bots use LookupAction (fully discrete) — recommended."""
from __future__ import annotations

from typing import Any


def build_action_parser(config: dict[str, Any]):
    name = config.get("name", "lookup")

    if name == "lookup":
        from rlbot.actions.lookup_act import LookupAction
        return LookupAction()

    if name == "discrete":
        # rlgym_sim's DiscreteAction is actually MultiDiscrete — slower to train.
        from rlgym_sim.utils.action_parsers import DiscreteAction

        return DiscreteAction()

    if name == "continuous":
        from rlgym_sim.utils.action_parsers import ContinuousAction

        return ContinuousAction()

    raise ValueError(f"Unknown action parser: {name!r}")
