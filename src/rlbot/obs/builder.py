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
        # rlgym_tools.advanced_obs is a more feature-rich obs (relative positions, etc.)
        try:
            from rlgym_tools.extra_obs.advanced_obs import AdvancedObs

            return AdvancedObs()
        except ImportError as e:
            raise ImportError(
                "obs.name='advanced' requires rlgym-tools. pip install -r requirements.txt"
            ) from e

    raise ValueError(f"Unknown obs builder: {name!r}")
