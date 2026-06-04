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
        # rlgym_sim ships its own AdvancedObs (relative positions/velocities). The old
        # rlgym_tools.extra_obs path was removed in rlgym-tools v2, so import it from
        # rlgym_sim directly — same source as DefaultObs above.
        from rlgym_sim.utils.obs_builders import AdvancedObs

        return AdvancedObs()

    raise ValueError(f"Unknown obs builder: {name!r}")
