"""Load an rlgym-ppo DiscreteFF policy from a checkpoint, arch inferred from weights.

Mirrors the proven loader in diego-bots/papaya_1v1_viewer.py: the network shape is
read straight from PPO_POLICY.pt, so any width/obs combination loads without config.
torch / rlgym_ppo are imported lazily so this module is import-safe for tests.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def load_policy(ckpt_dir: Path, expected_obs_dim: int, device: str = "cpu"):
    """Build a DiscreteFF matching the checkpoint's weights and load them.

    `expected_obs_dim` is the dim this bot will actually be fed (its own training
    obs); a mismatch raises loudly instead of silently feeding a wrong-size vector.
    """
    import torch
    from rlgym_ppo.ppo.discrete_policy import DiscreteFF

    sd = torch.load(ckpt_dir / "PPO_POLICY.pt", map_location=device, weights_only=True)
    weight_keys = [k for k in sd if k.endswith("weight")]
    if not weight_keys:
        raise ValueError(f"No linear weights found in {ckpt_dir / 'PPO_POLICY.pt'}")
    in_dim = int(sd[weight_keys[0]].shape[1])
    out_dim = int(sd[weight_keys[-1]].shape[0])
    hidden = tuple(int(sd[k].shape[0]) for k in weight_keys[:-1])

    if in_dim != expected_obs_dim:
        raise ValueError(
            f"obs mismatch for {ckpt_dir}: checkpoint expects input dim {in_dim} but the "
            f"env feeds this car {expected_obs_dim}-dim obs."
        )

    policy = DiscreteFF(expected_obs_dim, out_dim, hidden, device)
    policy.load_state_dict(sd)
    policy.eval()
    return policy


def action_to_int(action) -> int:
    """Normalise whatever get_action() returned into a plain Python int index."""
    import torch

    if isinstance(action, np.ndarray):
        return int(action.flat[0])
    if isinstance(action, torch.Tensor):
        return int(action.item())
    return int(action)
