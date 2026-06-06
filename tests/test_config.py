"""Configs are the single source of truth — keep them parseable + sane."""

from __future__ import annotations

from pathlib import Path

from rlbot.utils.config import Config, load_config


def test_default_config_loads(default_config_path: Path) -> None:
    cfg = load_config(default_config_path)
    assert isinstance(cfg, Config)
    assert cfg.experiment_name == "default"
    assert cfg.seed == 42
    assert cfg.env["team_size"] == 1
    assert cfg.action["name"] == "lookup"


def test_all_experiment_configs_load(configs_dir: Path) -> None:
    exp_dir = configs_dir / "experiments"
    exp_files = sorted(exp_dir.glob("*.yaml"))
    assert exp_files, "no experiment configs found"
    for path in exp_files:
        cfg = load_config(path)
        assert cfg.experiment_name, f"{path.name} missing experiment_name"
        assert cfg.experiment_name != "default", f"{path.name} did not override experiment_name"


def test_extends_merges_deeply(configs_dir: Path) -> None:
    cfg = load_config(configs_dir / "experiments" / "exp_001_baseline.yaml")
    # baseline overrides timestep_limit, but not learner.arch which stays from default
    assert cfg.learner["timestep_limit"] == 10_000_000
    assert cfg.learner["ppo_batch_size"] == 50_000


def test_reward_components_have_required_fields(default_config_path: Path) -> None:
    cfg = load_config(default_config_path)
    components = cfg.rewards["components"]
    assert components, "default config must define at least one reward"
    for c in components:
        assert "name" in c
        assert "weight" in c
