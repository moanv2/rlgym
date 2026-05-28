#!/usr/bin/env python
"""Watch a trained checkpoint play in the rlviser visualizer.

Rebuilds the training env from the experiment config, loads a PPO_POLICY.pt
checkpoint, and plays episodes in real time. Both cars in a 1v1 share the same
policy (self-play), exactly as during training.

Requires rlviser_py installed and the rlviser binary running first; see
https://github.com/VirxEC/rlviser. Without the binary the policy still runs,
it just won't display.

    python scripts/visualize.py \
        --checkpoint checkpoints/exp_001_baseline-<ts>/10000192 \
        --config configs/experiments/exp_001_baseline.yaml
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from rlbot.env import make_env_builder
from rlbot.models.architectures import get_layer_sizes
from rlbot.utils.config import load_config
from rlbot.utils.logging import get_logger

REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve_policy_path(checkpoint: Path) -> Path:
    """Accept either a step folder (containing PPO_POLICY.pt) or the .pt file itself."""
    return checkpoint / "PPO_POLICY.pt" if checkpoint.is_dir() else checkpoint


def main() -> None:
    p = argparse.ArgumentParser(description="Watch a checkpoint play in rlviser.")
    p.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Checkpoint step folder (with PPO_POLICY.pt) or the .pt file directly",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/experiments/exp_001_baseline.yaml",
        help="Experiment YAML the checkpoint was trained with (defines arch/obs/action/env)",
    )
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument(
        "--deterministic",
        action="store_true",
        help="Pick the most likely action instead of sampling (cleaner but less varied play)",
    )
    args = p.parse_args()

    log = get_logger("rlbot.visualize")

    policy_path = _resolve_policy_path(args.checkpoint)
    if not policy_path.exists():
        raise FileNotFoundError(f"No policy weights at {policy_path}")

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = make_env_builder(cfg.env, cfg.to_dict())()
    obs = env.reset()

    obs_size = int(np.asarray(obs).shape[-1])
    n_actions = int(env.action_space.n)
    layer_sizes = get_layer_sizes(cfg.learner.get("arch", "small"))

    from rlgym_ppo.ppo import DiscreteFF

    policy = DiscreteFF(obs_size, n_actions, layer_sizes, device)
    policy.load_state_dict(torch.load(policy_path, map_location=device))
    policy.eval()
    log.info(
        f"Loaded {policy_path} — obs={obs_size}, actions={n_actions}, "
        f"arch={layer_sizes}, device={device}"
    )

    tick_skip = int(cfg.env.get("tick_skip", 8))
    step_dt = tick_skip / 120.0  # game-seconds advanced per env step

    try:
        import rlviser_py as rlviser
    except ImportError:
        rlviser = None
        log.warning("rlviser_py not installed — running headless (policy runs, no visuals).")

    try:
        for ep in range(args.episodes):
            obs = env.reset()
            done = False
            ep_reward = 0.0
            while not done:
                obs_arr = np.asarray(obs, dtype=np.float32)
                single = obs_arr.ndim == 1
                if single:
                    obs_arr = obs_arr[None, :]

                with torch.no_grad():
                    probs = policy.get_output(obs_arr).view(-1, n_actions)
                    if args.deterministic:
                        chosen = probs.argmax(dim=-1)
                    else:
                        chosen = torch.multinomial(probs, 1).flatten()
                actions = chosen.cpu().numpy()
                if single:
                    actions = actions[0]

                obs, reward, done, _ = env.step(actions)
                ep_reward += float(np.mean(reward))

                if rlviser is not None:
                    env.render()
                    while rlviser.get_game_paused():
                        time.sleep(0.05)
                    speed = max(rlviser.get_game_speed(), 1e-3)
                    time.sleep(step_dt / speed)

            log.info(f"Episode {ep + 1}/{args.episodes} — mean reward {ep_reward:.2f}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
