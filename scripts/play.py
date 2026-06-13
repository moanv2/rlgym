"""Watch a trained checkpoint play in the rlviser visualizer.

Loads the same YAML config used for training so the policy arch, env, and
action parser all match what the weights expect. No hardcoded values.

Prerequisites (one-time install):
  1. pip install rlviser-py
  2. Download rlviser binary from https://github.com/VirxEC/rlviser/releases
     extract it, double-click rlviser.exe — leave the empty arena window open
  3. Then run:  python scripts/play.py --config configs/experiments/exp_006_flagship.yaml

Press Ctrl+C in this terminal to stop. The play session does not save new
checkpoints, so your training data is safe.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from rlbot.env import make_env_builder
from rlbot.models.architectures import get_layer_sizes
from rlbot.utils.config import load_config
from rlbot.utils.logging import get_logger

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_ROOT = REPO_ROOT / "checkpoints"


def find_latest_checkpoint(experiment_name: str) -> str:
    run_dir = CHECKPOINT_ROOT / experiment_name
    if not run_dir.exists():
        raise SystemExit(f"No checkpoint folder at {run_dir}. Train first.")
    timesteps = [p for p in run_dir.iterdir()
                 if p.is_dir() and p.name.isdigit() and (p / "PPO_POLICY.pt").exists()]
    if not timesteps:
        raise SystemExit(f"No saved checkpoints inside {run_dir}.")
    latest = max(timesteps, key=lambda p: int(p.name))
    return str(latest)


def main() -> None:
    p = argparse.ArgumentParser(description="Watch a trained checkpoint play in rlviser.")
    p.add_argument("--config", required=True, help="Path to experiment YAML (same one used for training)")
    p.add_argument("--checkpoint", default=None,
                   help="Specific checkpoint path; defaults to latest in checkpoints/<experiment_name>/")
    p.add_argument("--render-delay", type=float, default=0.006,
                   help="Seconds between rendered steps (smaller=faster playback). 0.006 ~= real time at tick_skip=8.")
    args = p.parse_args()

    log = get_logger("rlbot.play")
    cfg = load_config(args.config)
    log.info(f"Loaded config: {cfg.experiment_name}")

    # Disable self-play during visualization — we want to watch the current
    # policy play against itself, not against past snapshots (clearer signal).
    if cfg.extras.get("self_play"):
        cfg.extras["self_play"]["enabled"] = False
        log.info("Self-play disabled for visualization (current policy on both sides).")

    load_folder = args.checkpoint or find_latest_checkpoint(cfg.experiment_name)
    log.info(f"Loading checkpoint: {load_folder}")

    env_builder = make_env_builder(cfg.env, cfg.to_dict())

    from rlgym_ppo import Learner

    L = cfg.learner
    arch = get_layer_sizes(L.get("arch", "small"))

    learner = Learner(
        env_builder,
        n_proc=1,                 # one worker is plenty for visualization
        min_inference_size=1,
        metrics_logger=None,
        ppo_batch_size=int(L.get("ppo_batch_size", 50_000)),
        ts_per_iteration=int(L.get("ts_per_iteration", 50_000)),
        exp_buffer_size=int(L.get("exp_buffer_size", 150_000)),
        ppo_minibatch_size=int(L.get("ppo_minibatch_size", 50_000)),
        ppo_ent_coef=float(L.get("ppo_ent_coef", 0.01)),
        ppo_epochs=int(L.get("ppo_epochs", 2)),
        ppo_clip_range=float(L.get("ppo_clip_range", 0.2)),
        policy_lr=float(L.get("policy_lr", 3e-4)),
        critic_lr=float(L.get("critic_lr", 3e-4)),
        gae_lambda=float(L.get("gae_lambda", 0.95)),
        standardize_returns=bool(L.get("standardize_returns", True)),
        standardize_obs=bool(L.get("standardize_obs", False)),
        # Do not save new checkpoints while watching — keep training data safe.
        save_every_ts=10**12,
        timestep_limit=10**12,
        log_to_wandb=False,
        checkpoint_load_folder=load_folder,
        checkpoints_save_folder=str(REPO_ROOT / "checkpoints" / "_play_session_unused"),
        add_unix_timestamp=False,
        policy_layer_sizes=arch,
        critic_layer_sizes=arch,
        render=True,
        render_delay=args.render_delay,
    )
    log.info("Press Ctrl+C to stop. Make sure rlviser.exe is running in another window.")
    learner.learn()


if __name__ == "__main__":
    main()
