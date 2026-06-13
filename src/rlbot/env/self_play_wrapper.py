"""Self-play env wrapper: substitute orange-team actions with a frozen past policy.

This is the wrapper that turns symmetric self-play (both cars driven by the
*current* policy via shared weights) into Fictitious Self-Play (orange car
driven by a randomly-sampled *past* policy). Past-self diversity is the
single biggest gap between our bot and Diego's / Marian's, both of which
use stock rlgym-ppo (current-only self-play).

Caveats / known trade-offs:
  * rlgym-ppo records the action sampled by the *training* policy in its
    rollout buffer, then env.step() executes our overridden orange action.
    For the orange transitions this introduces a mild off-policy artifact.
    PPO clipping is forgiving here, and the net effect — opponent diversity —
    outweighs the noise in practice. If you need strictly on-policy training
    you'd have to fork rlgym-ppo to filter orange transitions out of the
    buffer, which is a much bigger change.
  * Inference for the opponent runs inside each rollout worker process, so
    each worker holds its own torch policy. Memory cost = n_proc * policy
    params; for arch='small' (3x256) that's ~600KB per worker — negligible.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np


class _FrozenPolicy:
    """Loads a rlgym-ppo DiscreteFF checkpoint and exposes a numpy-in,
    numpy-out act() method. Constructed lazily inside worker processes so
    torch is only imported when self-play is actually enabled.

    obs_size, n_actions, and the hidden layer sizes are all inferred from
    the saved state dict's tensor shapes. This means a snapshot can be
    swapped in even if the current training run has a different arch — as
    long as the obs shape matches the env."""

    def __init__(self, ckpt_dir: Path, device: str = "cpu"):
        import torch
        from rlgym_ppo.ppo.discrete_policy import DiscreteFF

        self._torch = torch
        state = torch.load(ckpt_dir / "PPO_POLICY.pt", map_location=device)
        # State dict keys are `model.0.weight`, `model.0.bias`, `model.2.weight`, ...
        # Linear layer weight shape is (out, in); first layer's in == obs_size,
        # last layer's out == n_actions. Intermediate outs == hidden sizes.
        linear_outs: list[int] = []
        linear_ins: list[int] = []
        for k, v in state.items():
            if k.endswith(".weight") and v.ndim == 2:
                linear_outs.append(v.shape[0])
                linear_ins.append(v.shape[1])
        if not linear_ins:
            raise RuntimeError(f"No linear layers found in {ckpt_dir}/PPO_POLICY.pt")
        obs_size = linear_ins[0]
        n_actions = linear_outs[-1]
        layer_sizes = linear_outs[:-1]

        self.policy = DiscreteFF(obs_size, n_actions, layer_sizes, device)
        self.policy.load_state_dict(state)
        self.policy.eval()
        self.device = device
        self.ckpt_name = ckpt_dir.name
        self.obs_size = obs_size

    def act(self, obs: np.ndarray) -> int:
        with self._torch.no_grad():
            obs_t = self._torch.as_tensor(obs, dtype=self._torch.float32,
                                          device=self.device).unsqueeze(0)
            action, _logp = self.policy.get_action(obs_t, deterministic=False)
        # get_action returns a (1,) tensor for a single env; return the scalar id
        return int(action.squeeze().cpu().item())


class SelfPlayWrapper:
    """Wraps an rlgym_sim Gym env. Player 0 (blue) acts normally. Player 1
    (orange) has its action overridden with a frozen past-self policy
    sampled from the opponent pool at every reset.

    With probability `latest_prob`, the wrapper passes orange's action
    through unchanged — i.e. orange is driven by whatever policy the
    Learner is currently training. This keeps symmetric self-play in the
    mix as a baseline alongside the past-self matchups.
    """

    def __init__(self, env, pool_dir: str | Path, latest_prob: float = 0.4,
                 device: str = "cpu", seed: int | None = None):
        self.env = env
        self.pool_dir = Path(pool_dir)
        self.latest_prob = float(latest_prob)
        self.device = device
        self._rng = random.Random(seed)
        self._opponent: _FrozenPolicy | None = None
        self._last_orange_obs: np.ndarray | None = None

    def _list_snapshots(self) -> list[Path]:
        if not self.pool_dir.exists():
            return []
        return [p for p in self.pool_dir.iterdir()
                if p.is_dir() and (p / "PPO_POLICY.pt").exists()]

    def _sample_opponent(self) -> _FrozenPolicy | None:
        snaps = self._list_snapshots()
        if not snaps or self._rng.random() < self.latest_prob:
            return None  # fall back to current-policy self-play
        choice = self._rng.choice(snaps)
        try:
            return _FrozenPolicy(choice, self.device)
        except Exception:
            # If a checkpoint fails to load (corruption, arch mismatch),
            # silently fall back rather than crashing the worker.
            return None

    def reset(self) -> Any:
        obs = self.env.reset()
        self._opponent = self._sample_opponent()
        # rlgym_sim returns a list of per-agent observations
        if isinstance(obs, (list, tuple)) and len(obs) >= 2:
            self._last_orange_obs = np.asarray(obs[1], dtype=np.float32)
        return obs

    def step(self, actions):
        if self._opponent is not None and self._last_orange_obs is not None:
            actions = list(actions)
            opp_id = self._opponent.act(self._last_orange_obs)
            # Match the shape/dtype of the action the Learner sent for orange.
            # rlgym_ppo's DiscreteFF.get_action() yields shape (1,) numpy
            # arrays — substituting a bare Python int produces an
            # inhomogeneous list that LookupAction.parse_actions() can't
            # np.asarray() cleanly. Wrap in the original action's container.
            orig = actions[1]
            if isinstance(orig, np.ndarray):
                actions[1] = np.asarray([opp_id], dtype=orig.dtype).reshape(orig.shape)
            else:
                actions[1] = np.asarray([opp_id], dtype=np.int64)
        obs, reward, done, info = self.env.step(actions)
        if isinstance(obs, (list, tuple)) and len(obs) >= 2:
            self._last_orange_obs = np.asarray(obs[1], dtype=np.float32)
        return obs, reward, done, info

    # rlgym_sim Gym envs expose more methods — pass everything else through
    def __getattr__(self, name):
        return getattr(self.env, name)
