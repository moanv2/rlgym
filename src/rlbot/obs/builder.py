"""Pick an obs builder by name. Keep the API stable across experiments — changing obs
mid-training invalidates the policy network."""
from __future__ import annotations

from typing import Any


def build_obs(config: dict[str, Any]):
    name = config.get("name", "default")

    if name == "default":
        from rlgym_sim.utils.obs_builders import DefaultObs

        return DefaultObs()

    if name == "advanced":
        # Custom rlgym_sim-compatible AdvancedObs (relative pos/vel to ball &
        # opponent). We ship our own because rlgym_tools 2.6.4 has no AdvancedObs
        # for the rlgym_sim API — its RelativeDefaultObs is RLGym 2.0 API and is
        # not a rlgym_sim ObsBuilder subclass, so it crashes in rlgym_sim.make().
        from rlbot.obs.advanced_obs import AdvancedObs

        return AdvancedObs()

    raise ValueError(f"Unknown obs builder: {name!r}")
