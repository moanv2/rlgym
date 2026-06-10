"""Vendored copy of rlgym_ppo's DiscreteFF actor network.

Copied verbatim (architecture-wise) from
``rlgym_ppo/ppo/discrete_policy.py`` so the deployment env does NOT need the
full rlgym_ppo / rlgym_sim / rocketsim stack just to run a forward pass. The
layer layout MUST stay byte-identical to the training class, otherwise
``load_state_dict`` on ``PPO_POLICY.pt`` will fail or silently load garbage:

    Linear(in, 512) -> ReLU -> Linear(512,512) -> ReLU
    -> Linear(512,512) -> ReLU -> Linear(512, 90) -> Softmax

Only the inference paths (`get_output`, `get_action`) are kept; the training
backprop helper is dropped.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class DiscreteFF(nn.Module):
    def __init__(self, input_shape, n_actions, layer_sizes, device):
        super().__init__()
        self.device = device

        assert len(layer_sizes) != 0, "AT LEAST ONE LAYER MUST BE SPECIFIED"
        layers = [nn.Linear(input_shape, layer_sizes[0]), nn.ReLU()]
        prev_size = layer_sizes[0]
        for size in layer_sizes[1:]:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            prev_size = size

        layers.append(nn.Linear(layer_sizes[-1], n_actions))
        layers.append(nn.Softmax(dim=-1))
        self.model = nn.Sequential(*layers).to(self.device)

        self.n_actions = n_actions

    def get_output(self, obs):
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(
                np.asarray(obs), dtype=torch.float32, device=self.device
            )
        return self.model(obs)

    def get_action(self, obs, deterministic: bool = False) -> int:
        """Return the chosen action index in ``0..n_actions-1``.

        ``deterministic=True`` takes the argmax (greedy, less twitchy — best for
        playing a human). ``False`` samples from the categorical distribution,
        matching how the policy explored during training.
        """
        probs = self.get_output(obs).view(-1, self.n_actions)
        probs = torch.clamp(probs, min=1e-11, max=1)

        if deterministic:
            return int(probs.detach().cpu().numpy().argmax())

        action = torch.multinomial(probs, 1, True)
        return int(action.flatten().cpu().item())
