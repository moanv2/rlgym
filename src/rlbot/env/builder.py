"""Builds the rlgym_sim env factory expected by rlgym-ppo's Learner.

The Learner spawns multiple processes and calls the returned function in each one,
so it must be picklable — keep state inside the closure.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from rlbot.actions import build_action_parser
from rlbot.obs import build_obs
from rlbot.rewards import build_reward
from rlbot.state_setters import build_state_setter
from rlbot.terminal import build_terminal_conditions


def make_env_builder(env_cfg: dict[str, Any], full_cfg: dict[str, Any]) -> Callable[[], Any]:
    """Returns a zero-arg callable that constructs an rlgym_sim environment."""
    team_size = int(env_cfg.get("team_size", 1))
    spawn_opponents = bool(env_cfg.get("spawn_opponents", True))
    tick_skip = int(env_cfg.get("tick_skip", 8))

    obs_cfg = full_cfg["obs"]
    action_cfg = full_cfg["action"]
    reward_cfg = full_cfg["rewards"]
    state_cfg = full_cfg["state_setter"]
    term_cfg = full_cfg["terminal"]

    def _build():
        import rlgym_sim

        env = rlgym_sim.make(
            tick_skip=tick_skip,
            team_size=team_size,
            spawn_opponents=spawn_opponents,
            terminal_conditions=build_terminal_conditions(term_cfg, tick_skip),
            reward_fn=build_reward(reward_cfg),
            obs_builder=build_obs(obs_cfg),
            state_setter=build_state_setter(state_cfg),
            action_parser=build_action_parser(action_cfg),
        )

        # Optional rlgym-tools wrappers (e.g. SB3 logging)
        if full_cfg.get("logging", {}).get("sb3_metrics", False):
            try:
                from rlgym_tools.sb3_utils.sb3_log_reward import SB3CombinedLogReward  # noqa

                # Wrap if available — left as opt-in.
            except ImportError:
                pass

        return env

    return _build
