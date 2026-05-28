"""Builds the rlgym_sim env factory expected by rlgym-ppo's Learner.

The Learner spawns multiple processes and pickles the returned callable,
so it must be a module-level class (closures are not picklable).
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
    def __init__(
        self,
        team_size: int,
        spawn_opponents: bool,
        tick_skip: int,
        obs_cfg: dict,
        action_cfg: dict,
        reward_cfg: dict,
        state_cfg: dict,
        term_cfg: dict,
    ) -> None:
        self.team_size = team_size
        self.spawn_opponents = spawn_opponents
        self.tick_skip = tick_skip
        self.obs_cfg = obs_cfg
        self.action_cfg = action_cfg
        self.reward_cfg = reward_cfg
        self.state_cfg = state_cfg
        self.term_cfg = term_cfg

    def __call__(self) -> Any:
        import rlgym_sim

        return rlgym_sim.make(
            tick_skip=self.tick_skip,
            team_size=self.team_size,
            spawn_opponents=self.spawn_opponents,
            terminal_conditions=build_terminal_conditions(self.term_cfg, self.tick_skip),
            reward_fn=build_reward(self.reward_cfg),
            obs_builder=build_obs(self.obs_cfg),
            state_setter=build_state_setter(self.state_cfg),
            action_parser=build_action_parser(self.action_cfg),
        )


def make_env_builder(env_cfg: dict[str, Any], full_cfg: dict[str, Any]) -> Callable[[], Any]:
    """Returns a zero-arg callable that constructs an rlgym_sim environment."""
    return _EnvBuilder(
        team_size=int(env_cfg.get("team_size", 1)),
        spawn_opponents=bool(env_cfg.get("spawn_opponents", True)),
        tick_skip=int(env_cfg.get("tick_skip", 8)),
        obs_cfg=full_cfg["obs"],
        action_cfg=full_cfg["action"],
        reward_cfg=full_cfg["rewards"],
        state_cfg=full_cfg["state_setter"],
        term_cfg=full_cfg["terminal"],
    )
