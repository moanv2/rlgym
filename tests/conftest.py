"""Shared pytest fixtures."""
from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def configs_dir(repo_root: Path) -> Path:
    return repo_root / "configs"


@pytest.fixture
def default_config_path(configs_dir: Path) -> Path:
    return configs_dir / "default.yaml"
