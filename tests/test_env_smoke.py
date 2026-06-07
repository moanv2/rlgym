"""End-to-end smoke test: build the env from the default config, reset it once.

Marked `rocketsim` because it requires the rlgym_sim install + collision_meshes/.
CI runs without these by default; flip the marker selection to enable.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.rocketsim
@pytest.mark.slow
def test_env_builds_and_resets(default_config_path: Path) -> None:
    pytest.importorskip("rlgym_sim")
    pytest.importorskip("rlgym_tools")

    from rlbot.env import make_env_builder
    from rlbot.utils.config import load_config

    cfg = load_config(default_config_path)
    builder = make_env_builder(cfg.env, cfg.to_dict())
    env = builder()
    obs = env.reset()
    assert obs is not None
