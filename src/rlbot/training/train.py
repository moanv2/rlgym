"""Training entrypoint.

    python -m rlbot.training.train --config configs/experiments/exp_001_baseline.yaml

Resumes automatically if `checkpoints/<experiment_name>/latest` exists.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from rlbot.env import make_env_builder
from rlbot.models.architectures import get_layer_sizes
from rlbot.utils.config import Config, load_config
from rlbot.utils.logging import get_logger
from rlbot.utils.seeding import seed_everything

REPO_ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT_ROOT = REPO_ROOT / "checkpoints"
LOG_ROOT = REPO_ROOT / "logs"


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def _snapshot_run_metadata(cfg: Config, run_dir: Path) -> None:
    """Save a frozen copy of the config + git SHA + timestamp next to the checkpoints.
    Reproducibility floor: never train without this."""
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "experiment_name": cfg.experiment_name,
        "started_at": datetime.utcnow().isoformat() + "Z",
        "git_sha": _git_sha(),
        "config": cfg.to_dict(),
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def train(cfg: Config) -> None:
    log = get_logger("rlbot.train", log_file=LOG_ROOT / f"{cfg.experiment_name}.log")
    seed_everything(cfg.seed)

    run_dir = CHECKPOINT_ROOT / cfg.experiment_name
    run_dir.mkdir(parents=True, exist_ok=True)
    # Save metadata to logs/, not checkpoints/ — rlgym_ppo expects only digit-named subdirs in checkpoint folder
    _snapshot_run_metadata(cfg, LOG_ROOT)

    env_builder = make_env_builder(cfg.env, cfg.to_dict())

    # --- rlgym-ppo Learner ---
    from rlgym_ppo import Learner

    L = cfg.learner
    arch = get_layer_sizes(L.get("arch", "small"))

    log_cfg = cfg.logging
    wandb_enabled = bool(log_cfg.get("wandb", True))
    wandb_entity = log_cfg.get("wandb_entity")

    # rlgym_ppo's Learner doesn't expose `wandb_entity` directly. If a team
    # entity is set in the config, we pre-init the wandb run ourselves and
    # hand it to the Learner via the `wandb_run` kwarg.
    wandb_run = None
    if wandb_enabled and wandb_entity:
        import wandb

        wandb_run = wandb.init(
            entity=wandb_entity,
            project=log_cfg.get("wandb_project", "rlgym-finalproject"),
            group=log_cfg.get("wandb_group", cfg.experiment_name),
            name=log_cfg.get("wandb_run") or None,
            config=cfg.to_dict(),
            reinit=True,
        )
        log.info(f"wandb run initialized at entity='{wandb_entity}' project='{wandb_run.project}'")

    learner = Learner(
        env_builder,
        n_proc=int(L.get("n_proc", 8)),
        min_inference_size=int(L.get("min_inference_size", 80)),
        metrics_logger=None,
        ppo_batch_size=int(L.get("ppo_batch_size", 50_000)),
        ts_per_iteration=int(L.get("ts_per_iteration", 50_000)),
        exp_buffer_size=int(L.get("exp_buffer_size", 150_000)),
        ppo_minibatch_size=int(L.get("ppo_minibatch_size", 50_000)),
        ppo_ent_coef=float(L.get("ppo_ent_coef", 0.01)),
        ppo_epochs=int(L.get("ppo_epochs", 2)),
        standardize_returns=bool(L.get("standardize_returns", True)),
        standardize_obs=bool(L.get("standardize_obs", False)),
        save_every_ts=int(L.get("save_every_ts", 1_000_000)),
        timestep_limit=int(L.get("timestep_limit", 1_000_000_000)),
        log_to_wandb=wandb_enabled,
        wandb_run=wandb_run,
        wandb_project_name=log_cfg.get("wandb_project", "rlgym-finalproject"),
        wandb_group_name=log_cfg.get("wandb_group", cfg.experiment_name),
        wandb_run_name=log_cfg.get("wandb_run", None),
        checkpoints_save_folder=str(run_dir),
        add_unix_timestamp=False,  # disable timestamp so auto-resume works correctly
        policy_layer_sizes=arch,
        critic_layer_sizes=arch,
        render=False,
    )
    log.info(f"[bold green]Starting training[/]: {cfg.experiment_name}  arch={arch}  "
             f"timestep_limit={L.get('timestep_limit'):,}")
    learner.learn()


def main() -> None:
    p = argparse.ArgumentParser(description="Train an RL Rocket League bot.")
    p.add_argument("--config", required=True, help="Path to experiment YAML")
    p.add_argument("--dry-run", action="store_true", help="Build env once and exit (smoke test)")
    args = p.parse_args()

    cfg = load_config(args.config)

    if args.dry_run:
        log = get_logger("rlbot.train")
        log.info(f"[yellow]Dry run[/] — building env for '{cfg.experiment_name}'")
        env_builder = make_env_builder(cfg.env, cfg.to_dict())
        env = env_builder()
        env.reset()
        log.info("[green]Env built and reset successfully[/]")
        return

    train(cfg)


if __name__ == "__main__":
    main()
