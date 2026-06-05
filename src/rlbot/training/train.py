"""Training entrypoint.

    python -m rlbot.training.train --config configs/experiments/exp_001_baseline.yaml

Resumes automatically if `checkpoints/<experiment_name>/latest` exists.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

from rlbot.env import make_env_builder
from rlbot.models.architectures import get_layer_sizes
from rlbot.utils.config import Config, load_config
from rlbot.utils.logging import get_logger
from rlbot.utils.power import keep_awake
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


def _numbered_checkpoints(run_dir: Path) -> list[Path]:
    """Numbered checkpoint sub-folders (one per saved cumulative-timestep) inside a run.

    Only valid when the run was trained with ``add_unix_timestamp: false`` (the
    curriculum default), which keeps checkpoints directly under
    ``checkpoints/<experiment_name>/<cumulative_ts>/``.
    """
    if not run_dir.is_dir():
        return []
    return [d for d in run_dir.iterdir() if d.is_dir() and d.name.isdigit()]


def _resolve_init_checkpoint(init_from: str) -> Path | None:
    """Resolve the latest checkpoint folder of a previous stage, for warm-starting.

    `init_from` is a prior experiment_name. Returns the highest-timestep checkpoint
    sub-folder (the path the Learner's `checkpoint_load_folder` expects), or None if
    that stage has no checkpoints yet.
    """
    numbered = _numbered_checkpoints(CHECKPOINT_ROOT / init_from)
    if not numbered:
        return None
    return max(numbered, key=lambda d: int(d.name))


def _snapshot_run_metadata(cfg: Config) -> Path:
    """Save a frozen copy of the config + git SHA + timestamp for reproducibility.

    Written to logs/ — NOT the checkpoint folder. rlgym_ppo's checkpoint saver
    ``int()``-parses every entry in ``checkpoints_save_folder`` to find old
    checkpoints, so any stray file there crashes it when ``add_unix_timestamp`` is
    false (which the curriculum stages use). Reproducibility floor: never train
    without this.
    """
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    meta_path = LOG_ROOT / f"{cfg.experiment_name}.run_metadata.json"
    meta = {
        "experiment_name": cfg.experiment_name,
        "started_at": datetime.utcnow().isoformat() + "Z",
        "git_sha": _git_sha(),
        "config": cfg.to_dict(),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta_path


def _install_eta_reporter(timestep_limit: int, log, window: int = 10) -> None:
    """Wrap rlgym_ppo's per-iteration reporter to add rolling step-time avg and ETA.

    The Learner calls ``rlgym_ppo.util.reporting.report_metrics`` once per iteration
    with the cumulative-timestep and wall-time fields already filled in. We intercept
    that call, keep a rolling window of (cumulative_ts, wall_time) samples, derive an
    average steps/sec + estimated time-to-``timestep_limit``, push the values into the
    loggable dict so wandb picks them up, and emit a one-line console summary.
    """
    import time as _time
    from collections import deque
    from datetime import datetime, timedelta

    from rlgym_ppo.util import reporting

    orig_report = reporting.report_metrics
    history: deque[tuple[int, float]] = deque(maxlen=max(2, window))

    def wrapped(loggable_metrics, debug_metrics, wandb_run=None):
        now = _time.time()
        cum_ts = int(loggable_metrics.get("Cumulative Timesteps", 0))
        iter_time = float(loggable_metrics.get("Total Iteration Time", 0.0))
        history.append((cum_ts, now))

        if len(history) >= 2:
            ts_old, t_old = history[0]
            dt = max(now - t_old, 1e-9)
            d_ts = max(cum_ts - ts_old, 0)
            avg_sps = d_ts / dt
            avg_iter_time = dt / (len(history) - 1)
        else:
            avg_sps = float(loggable_metrics.get("Overall Steps per Second", 0.0))
            avg_iter_time = iter_time

        remaining = max(timestep_limit - cum_ts, 0)
        eta_s = remaining / avg_sps if avg_sps > 0 else float("inf")
        progress_pct = (100.0 * cum_ts / timestep_limit) if timestep_limit else 0.0

        loggable_metrics["Avg Iteration Time (rolling)"] = float(avg_iter_time)
        loggable_metrics["Avg Steps per Second (rolling)"] = float(avg_sps)
        loggable_metrics["Progress Percent"] = float(progress_pct)
        loggable_metrics["ETA Seconds"] = 0.0 if eta_s == float("inf") else float(eta_s)

        if eta_s == float("inf"):
            eta_str, finish_str = "inf", "n/a"
        else:
            eta_str = str(timedelta(seconds=int(eta_s)))
            finish_str = (datetime.now() + timedelta(seconds=int(eta_s))).strftime("%Y-%m-%d %H:%M:%S")

        # Let rlgym_ppo emit its own iteration block first, then print ours last so
        # it stays at the bottom of the console (and goes to the log file via `log`).
        result = orig_report(loggable_metrics, debug_metrics, wandb_run=wandb_run)

        msg = (
            f"Progress {cum_ts:,}/{timestep_limit:,} ({progress_pct:.2f}%)  "
            f"avg iter={avg_iter_time:.2f}s ({int(avg_sps):,} sps over last {len(history)} iters)  "
            f"ETA={eta_str}  finish~{finish_str}"
        )
        # `print` so it always lands on stdout next to rlgym_ppo's own prints;
        # `log.info` so it also goes to the run's log file.
        print(f">>> [ETA] {msg}", flush=True)
        log.info(f"[cyan]{msg}[/]")
        return result

    reporting.report_metrics = wrapped


def _install_kbhit_guard() -> None:
    """Make rlgym_ppo's keyboard poller crash-proof on Windows.

    The learn loop polls the console each iteration for the ``p``/``c``/``q``
    controls. On Windows ``KBHit.getch()`` does ``msvcrt.getch().decode('utf-8')``,
    which kills the entire run when any *extended* key is pressed (arrows, Home/End,
    Page Up/Down, Insert/Delete, F-keys): ``msvcrt`` returns a 0x00/0xe0 prefix byte
    that is an incomplete UTF-8 sequence. We replace it with a version that consumes
    the prefix, ignores undecodable input, and can never raise — so only ``p``/``c``/
    ``q`` do anything and every other keystroke is a silent no-op.
    """
    import os

    if os.name != "nt":
        return

    import msvcrt
    from rlgym_ppo.util.kbhit import KBHit

    def _safe_getch(self) -> str:
        try:
            ch = msvcrt.getch()
            if ch in (b"\x00", b"\xe0"):  # extended-key prefix — drop the scancode too
                msvcrt.getch()
                return ""
            return ch.decode("utf-8", errors="ignore")
        except Exception:
            return ""

    KBHit.getch = _safe_getch


def train(cfg: Config, keep_system_awake: bool = True) -> None:
    log = get_logger("rlbot.train", log_file=LOG_ROOT / f"{cfg.experiment_name}.log")
    seed_everything(cfg.seed)

    run_dir = CHECKPOINT_ROOT / cfg.experiment_name
    meta_path = _snapshot_run_metadata(cfg)
    log.info(f"Run metadata snapshot: {meta_path}")

    env_builder = make_env_builder(cfg.env, cfg.to_dict())

    # --- rlgym-ppo Learner ---
    from rlgym_ppo import Learner

    _install_kbhit_guard()

    L = cfg.learner
    arch = get_layer_sizes(L.get("arch", "small"))

    # --- checkpoint loading: resume this run, or warm-start from a previous stage ---
    # Default "latest" makes the Learner auto-resume its own most-recent checkpoint.
    # A curriculum stage sets `learner.init_from: <previous_experiment_name>` to inherit
    # the prior stage's policy/critic. We only warm-start when this run has no
    # checkpoints of its own yet, so re-launching mid-stage resumes instead.
    add_unix_ts = bool(L.get("add_unix_timestamp", True))
    init_from = L.get("init_from")
    checkpoint_load_folder = "latest"
    warm_starting = False
    if init_from:
        if _numbered_checkpoints(run_dir):
            log.info(f"Found existing checkpoints for '{cfg.experiment_name}' — resuming "
                     f"(ignoring init_from='{init_from}').")
        else:
            resolved = _resolve_init_checkpoint(init_from)
            if resolved is None:
                raise FileNotFoundError(
                    f"learner.init_from='{init_from}' but no checkpoint was found under "
                    f"{CHECKPOINT_ROOT / init_from}. Train that stage first "
                    f"(and ensure it used add_unix_timestamp: false)."
                )
            checkpoint_load_folder = str(resolved)
            warm_starting = True
            log.info(f"[bold cyan]Warm-starting[/] '{cfg.experiment_name}' from "
                     f"'{init_from}' checkpoint: {resolved.name}")

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
        policy_lr=float(L.get("policy_lr", 3e-4)),
        critic_lr=float(L.get("critic_lr", 3e-4)),
        standardize_returns=bool(L.get("standardize_returns", True)),
        standardize_obs=bool(L.get("standardize_obs", False)),
        save_every_ts=int(L.get("save_every_ts", 1_000_000)),
        timestep_limit=int(L.get("timestep_limit", 1_000_000_000)),
        log_to_wandb=wandb_enabled,
        # When warm-starting from a *different* stage, don't resume that stage's wandb
        # run — this stage gets its own run. On a same-run resume, let it reload.
        load_wandb=not warm_starting,
        wandb_run=wandb_run,
        wandb_project_name=log_cfg.get("wandb_project", "rlgym-finalproject"),
        wandb_group_name=log_cfg.get("wandb_group", cfg.experiment_name),
        wandb_run_name=log_cfg.get("wandb_run", None),
        checkpoints_save_folder=str(run_dir),
        add_unix_timestamp=add_unix_ts,
        checkpoint_load_folder=checkpoint_load_folder,
        n_checkpoints_to_keep=int(L.get("n_checkpoints_to_keep", 5)),
        policy_layer_sizes=arch,
        critic_layer_sizes=arch,
        render=False,
    )
    log.info(f"[bold green]Starting training[/]: {cfg.experiment_name}  arch={arch}  "
             f"policy_lr={L.get('policy_lr', 3e-4)}  "
             f"timestep_limit={L.get('timestep_limit'):,}")

    _install_eta_reporter(
        timestep_limit=int(L.get("timestep_limit", 1_000_000_000)),
        log=log,
        window=int(log_cfg.get("eta_window", 10)),
    )

    # Long runs span days — keep the machine from sleeping out from under us.
    if keep_system_awake:
        with keep_awake(reason=f"rlbot {cfg.experiment_name}"):
            learner.learn()
    else:
        learner.learn()


def main() -> None:
    p = argparse.ArgumentParser(description="Train an RL Rocket League bot.")
    p.add_argument("--config", required=True, help="Path to experiment YAML")
    p.add_argument("--dry-run", action="store_true", help="Build env once and exit (smoke test)")
    p.add_argument("--no-keep-awake", action="store_true",
                   help="Don't inhibit system sleep during training (default: keep awake).")
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

    train(cfg, keep_system_awake=not args.no_keep_awake)


if __name__ == "__main__":
    main()
