"""Builds the rlgym_sim env factory expected by rlgym-ppo's Learner.

The Learner spawns worker processes and PICKLES the returned callable to ship it
to each one, so the callable must be picklable. A nested closure is NOT picklable
(Python can't pickle local functions), so we return a module-level callable class
that stores only plain config values and builds the env lazily in each worker.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from rlbot.actions import build_action_parser
from rlbot.obs import build_obs
from rlbot.rewards import build_reward
from rlbot.state_setters import build_state_setter
from rlbot.terminal import build_terminal_conditions


class _EnvBuilder:
    """Picklable zero-arg env factory.

    Stores only plain dicts/scalars (loaded from YAML) so an instance pickles
    cleanly and can be sent to rlgym-ppo's worker processes. The rlgym_sim env is
    constructed lazily inside ``__call__``, which runs in each worker process.
    """

    def __init__(self, env_cfg: dict[str, Any], full_cfg: dict[str, Any]) -> None:
        self.team_size = int(env_cfg.get("team_size", 1))
        self.spawn_opponents = bool(env_cfg.get("spawn_opponents", True))
        self.tick_skip = int(env_cfg.get("tick_skip", 8))
        self.obs_cfg = full_cfg["obs"]
        self.action_cfg = full_cfg["action"]
        self.reward_cfg = full_cfg["rewards"]
        self.state_cfg = full_cfg["state_setter"]
        self.term_cfg = full_cfg["terminal"]
        self.sb3_metrics = bool(full_cfg.get("logging", {}).get("sb3_metrics", False))

    def __call__(self):
        import rlgym_sim

        env = rlgym_sim.make(
            tick_skip=self.tick_skip,
            team_size=self.team_size,
            spawn_opponents=self.spawn_opponents,
            terminal_conditions=build_terminal_conditions(self.term_cfg, self.tick_skip),
            reward_fn=build_reward(self.reward_cfg),
            obs_builder=build_obs(self.obs_cfg),
            state_setter=build_state_setter(self.state_cfg),
            action_parser=build_action_parser(self.action_cfg),
        )

        # Optional rlgym-tools wrappers (e.g. SB3 logging) — opt-in, left as a hook.
        return env


def make_env_builder(env_cfg: dict[str, Any], full_cfg: dict[str, Any]) -> Callable[[], Any]:
    """Returns a *picklable* zero-arg callable that constructs an rlgym_sim env."""
    return _EnvBuilder(env_cfg, full_cfg)
