"""Config loader. One YAML defines an experiment end-to-end."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

CONFIG_ROOT = Path(__file__).resolve().parents[3] / "configs"


@dataclass
class Config:
    """Typed view over the YAML config. Extra keys land in `extras`."""

    experiment_name: str
    seed: int
    env: dict[str, Any]
    obs: dict[str, Any]
    action: dict[str, Any]
    rewards: dict[str, Any]
    state_setter: dict[str, Any]
    terminal: dict[str, Any]
    learner: dict[str, Any]
    logging: dict[str, Any]
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Config:
        known = {
            "experiment_name",
            "seed",
            "env",
            "obs",
            "action",
            "rewards",
            "state_setter",
            "terminal",
            "learner",
            "logging",
        }
        extras = {k: v for k, v in data.items() if k not in known}
        return cls(
            experiment_name=data["experiment_name"],
            seed=int(data.get("seed", 0)),
            env=data.get("env", {}),
            obs=data.get("obs", {}),
            action=data.get("action", {}),
            rewards=data.get("rewards", {}),
            state_setter=data.get("state_setter", {}),
            terminal=data.get("terminal", {}),
            learner=data.get("learner", {}),
            logging=data.get("logging", {}),
            extras=extras,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "seed": self.seed,
            "env": self.env,
            "obs": self.obs,
            "action": self.action,
            "rewards": self.rewards,
            "state_setter": self.state_setter,
            "terminal": self.terminal,
            "learner": self.learner,
            "logging": self.logging,
            **self.extras,
        }


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path) -> Config:
    """Load a YAML config. Supports `extends: <relative path>` for inheritance."""
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f) or {}

    parent = data.pop("extends", None)
    if parent:
        parent_path = (path.parent / parent).resolve()
        parent_cfg = load_config(parent_path).to_dict()
        data = _deep_merge(parent_cfg, data)

    return Config.from_dict(data)
