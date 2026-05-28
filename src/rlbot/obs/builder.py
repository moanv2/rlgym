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
        # More feature-rich obs (relative ball/teammate/opponent positions, etc.).
        # rlgym_sim ships this directly — same builder that rlgym-tools v1 exposed
        # under extra_obs.advanced_obs (v2 dropped that path; see actions/lookup_action.py).
        from rlgym_sim.utils.obs_builders import AdvancedObs

        return AdvancedObs()

    raise ValueError(f"Unknown obs builder: {name!r}")
