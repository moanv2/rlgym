"""Frozen-opponent env wrapper for best-response / opponent-pool training.

Wraps a 1v1 rlgym_sim env so team-0 (the AGENT) is trained by rlgym-ppo while team-1
(the OPPONENT) is driven by a FROZEN policy sampled from a pool (PFSP-weighted). It
presents a 1-AGENT view (one obs / one reward) — rlgym-ppo infers n_agents=1 from the
obs shape (batched_agent.py:80,127,131), so this trains a single learner against a
fixed opponent ("best response") instead of symmetric self-play.

`SparEnvBuilder` is module-level + picklable so it ships to the rollout worker processes,
which each load the frozen policies themselves (don't pickle torch weights across procs).
Activated by a top-level `opponent_pool:` list in the training config (see train.py).
"""
from __future__ import annotations

import numpy as np
import torch

from rlbot.actions.lookup_action import LookupAction
from rlbot.env import make_env_builder
from rlbot.evaluation.evaluate import _load_policy, _resolve_checkpoint


def _make_obs(name):
    from rlgym_sim.utils.obs_builders import AdvancedObs, DefaultObs
    return AdvancedObs() if name == "advanced" else DefaultObs()


class FrozenOpponent:
    """A frozen policy + its own obs builder, driving the opponent car. Plays stochastically
    (deterministic=False) so the learner sees varied opponent behavior."""

    def __init__(self, label, policy_path, obs_name, obs_dim, n_actions, weight=1.0):
        self.label = label
        self.obs_name = obs_name
        self.weight = float(weight)
        self.policy = _load_policy(_resolve_checkpoint(policy_path), obs_dim, n_actions, "cpu")
        self.obs_builder = _make_obs(obs_name)
        self._lut = LookupAction().make_lookup_table()
        self._prev = np.zeros(8, dtype=np.float32)

    def reset(self, state):
        self.obs_builder.reset(state)
        self._prev = np.zeros(8, dtype=np.float32)

    def act(self, player, state, underlying_obs):
        # advanced-obs opponents reuse the env's obs; a default-obs opponent builds its own
        ob = underlying_obs if self.obs_name == "advanced" else self.obs_builder.build_obs(player, state, self._prev)
        with torch.no_grad():
            idx = int(self.policy.get_action(np.asarray(ob, dtype=np.float32), deterministic=False)[0])
        self._prev = self._lut[idx]
        return idx


class SparEnv:
    """1-agent view of a 1v1 env; the opponent car is driven by a PFSP-sampled frozen policy."""

    def __init__(self, pool, full_cfg):
        env_cfg = dict(full_cfg["env"]); env_cfg["team_size"] = 1; env_cfg["spawn_opponents"] = True
        self._env = make_env_builder(env_cfg, full_cfg)()
        self.action_space = self._env.action_space
        self.observation_space = getattr(self._env, "observation_space", None)
        self._pool = pool
        self._opp = None
        self._state = None
        self._agent_i = 0
        self._opp_i = 1
        self._last_obs = None

    def _sample_opponent(self):
        w = np.array([max(o.weight, 1e-6) for o in self._pool], dtype=np.float64)
        return self._pool[int(np.random.choice(len(self._pool), p=w / w.sum()))]

    def _indices(self):
        pl = self._state.players
        self._agent_i = next(i for i, p in enumerate(pl) if p.team_num == 0)
        self._opp_i = next(i for i, p in enumerate(pl) if p.team_num == 1)

    def reset(self, return_info=False):
        obs, info = self._env.reset(return_info=True)
        obs = obs if isinstance(obs, list) else [obs]
        self._state = info["state"]
        self._indices()
        self._opp = self._sample_opponent()
        self._opp.reset(self._state)
        self._last_obs = obs
        ao = obs[self._agent_i]
        return (ao, info) if return_info else ao

    def step(self, agent_action):
        a = int(np.asarray(agent_action).flatten()[0])
        opp_idx = self._opp.act(self._state.players[self._opp_i], self._state, self._last_obs[self._opp_i])
        acts = [[0], [0]]
        acts[self._agent_i] = [a]
        acts[self._opp_i] = [opp_idx]
        obs, rews, done, info = self._env.step(np.array(acts))
        obs = obs if isinstance(obs, list) else [obs]
        rews = rews if isinstance(rews, (list, np.ndarray)) else [rews, rews]
        self._state = info["state"]
        self._last_obs = obs
        return obs[self._agent_i], rews[self._agent_i], done, info

    def close(self):
        if hasattr(self._env, "close"):
            self._env.close()

    def render(self, *args, **kwargs):
        pass


class SparEnvBuilder:
    """Picklable env builder for rlgym-ppo's rollout workers. Loads the frozen pool in-worker."""

    def __init__(self, pool_spec, full_cfg, n_actions=90):
        self.pool_spec = pool_spec
        self.full_cfg = full_cfg
        self.n_actions = int(n_actions)

    def __call__(self):
        pool = [
            FrozenOpponent(s["label"], s["policy"], s["obs"], int(s["dim"]), self.n_actions, s.get("weight", 1.0))
            for s in self.pool_spec
        ]
        return SparEnv(pool, self.full_cfg)
