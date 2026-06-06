"""Sequential multi-stage training runner (the "leave it on for a week" script).

Chains the committed experiments into ONE unattended curriculum so you can start
it and walk away. Each stage runs to its timestep target, then the next begins.

    Stage 1  exp_001_baseline   small  (256x3) / default  obs / basics  rewards ->  10M  (sanity check)
    Stage 2  exp_002_advanced   medium (512x3) / advanced obs / basics  rewards ->  50M  (learn to hit the ball)
    Stage 3  exp_003_long_run   medium (512x3) / advanced obs / basics  rewards -> 250M  (long self-play)
    Stage 4  exp_003_long_run   medium (512x3) / advanced obs / OFFENSE rewards -> 600M  (reward SHIFT, same policy)

Why some stages start fresh and some resume
--------------------------------------------
A policy network's INPUT size is fixed by the obs builder and its HIDDEN sizes by
`arch`. Stage 1 (default obs, small) and Stage 2 (advanced obs, medium) therefore
cannot share weights - each trains a fresh network. That is expected and correct.

Stage 3 and Stage 4 use the SAME obs + arch + experiment_name, so Stage 4 *resumes
Stage 3's checkpoint* and only swaps the reward weights (basics -> offense). That is
the "Stage 1 -> Stage 2 reward shift" curriculum from docs/roadmap_45_days.md, done
WITHOUT throwing the trained policy away.

How it runs each stage
----------------------
For each stage it (1) fully resolves the base YAML's `extends` chain, (2) deep-merges
the stage overrides on top, (3) writes the resolved config to
`checkpoints/_stage_configs/<label>.yaml` (gitignored), and (4) shells out to the
normal entrypoint:  `python -m rlbot.training.train --config <that file>`.
Running each stage in its own process keeps rollout workers cleanly isolated and is
identical to how you'd launch a single run by hand.

Resuming
--------
Safe to Ctrl+C and re-run. A finished stage writes a `.done` marker under
`checkpoints/` and is skipped next time; an interrupted stage is auto-resumed by
rlgym-ppo from its latest checkpoint. Nothing here edits the committed configs.

Usage
-----
    python scripts/train_stages.py                # run the whole curriculum
    python scripts/train_stages.py --from 3       # start at stage 3
    python scripts/train_stages.py --only 1       # run just stage 1
    python scripts/train_stages.py --dry-run      # build each stage's env once, no training
    python scripts/train_stages.py --force        # ignore .done markers and re-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))  # so it runs even before `pip install -e .`

from rlbot.utils.config import load_config  # noqa: E402  (after sys.path tweak)

CONFIGS = REPO_ROOT / "configs"
CHECKPOINTS = REPO_ROOT / "checkpoints"
LOGS = REPO_ROOT / "logs"
STAGE_CFG_DIR = CHECKPOINTS / "_stage_configs"

# ---------------------------------------------------------------------------
# Tunables -- edit THESE, not the committed YAMLs.
# ---------------------------------------------------------------------------
# Rollout worker processes. Your Ryzen 9 3900X has 12 cores / 24 threads. Start at
# 20, leaving headroom for the learner + GPU inference thread + OS. Watch the
# "Collected Steps per Second" line in the first 2-3 minutes and try 16 / 20 / 24 --
# pick whichever gives the highest SPS. More workers is NOT always faster.
N_PROC = 20

# Long-run network. The committed configs use "medium" (512x3). The team agreed on
# "1024x3" in chat -- that is arch "large". Set this to "large" to match them.
ARCH_LONG = "large"  # "medium" (committed, faster) | "large" (team's 1024x3)


def _rewards(stage_file: str) -> dict:
    """Load configs/reward_weights/<stage_file> as a rewards dict."""
    with (CONFIGS / "reward_weights" / stage_file).open(encoding="utf-8") as f:
        return yaml.safe_load(f)


# Each stage = a base experiment YAML + in-memory overrides. No committed file is touched.
STAGES = [
    {
        "label": "stage1_baseline_sanity",
        "base": "experiments/exp_001_baseline.yaml",
        "overrides": {
            "learner": {"n_proc": N_PROC, "arch": "small", "timestep_limit": 10_000_000},
            "rewards": _rewards("stage_1_basics.yaml"),
            "logging": {"wandb_group": "exp_001_baseline", "wandb_run": "stage1_baseline"},
        },
    },
    {
        "label": "stage2_advanced_obs",
        "base": "experiments/exp_002_advanced_obs.yaml",
        "overrides": {
            "learner": {"n_proc": N_PROC, "arch": "medium", "timestep_limit": 50_000_000},
            "rewards": _rewards("stage_1_basics.yaml"),
            "logging": {"wandb_group": "exp_002_advanced_obs", "wandb_run": "stage2_advobs"},
        },
    },
    {
        "label": "stage3_longrun_basics",
        "base": "experiments/exp_003_long_run.yaml",
        "overrides": {
            "learner": {"n_proc": N_PROC, "arch": ARCH_LONG, "timestep_limit": 250_000_000},
            "rewards": _rewards("stage_1_basics.yaml"),
            "logging": {"wandb_group": "exp_003_long_run", "wandb_run": "stage3_basics"},
        },
    },
    {
        # SAME experiment_name as stage 3 -> resumes that checkpoint; only the reward
        # weights change (basics -> offense). This is the curriculum reward shift.
        "label": "stage4_longrun_offense",
        "base": "experiments/exp_003_long_run.yaml",
        "overrides": {
            "learner": {"n_proc": N_PROC, "arch": ARCH_LONG, "timestep_limit": 600_000_000},
            "rewards": _rewards("stage_2_offense.yaml"),
            "logging": {"wandb_group": "exp_003_long_run", "wandb_run": "stage4_offense"},
        },
    },
]


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into `base` (lists are replaced wholesale)."""
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _log(msg: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    LOGS.mkdir(exist_ok=True)
    with (LOGS / "train_stages.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _write_stage_config(stage: dict) -> Path:
    """Resolve base `extends` chain, merge overrides, write a self-contained YAML."""
    base = load_config(CONFIGS / stage["base"]).to_dict()
    merged = _deep_merge(base, stage["overrides"])
    STAGE_CFG_DIR.mkdir(parents=True, exist_ok=True)
    out = STAGE_CFG_DIR / f"{stage['label']}.yaml"
    out.write_text(yaml.safe_dump(merged, sort_keys=False), encoding="utf-8")
    return out


def _done_marker(label: str) -> Path:
    return CHECKPOINTS / f"_{label}.done"


def main() -> None:
    p = argparse.ArgumentParser(description="Run the full training curriculum end to end.")
    p.add_argument(
        "--from", dest="start", type=int, default=1, help="1-based stage number to start from (default 1)."
    )
    p.add_argument("--only", type=int, default=None, help="Run only this one stage number.")
    p.add_argument(
        "--dry-run", action="store_true", help="Build each stage's env once and exit (no training)."
    )
    p.add_argument("--force", action="store_true", help="Ignore .done markers and re-run stages.")
    args = p.parse_args()

    _log(
        f"Curriculum start | N_PROC={N_PROC} ARCH_LONG={ARCH_LONG} stages={len(STAGES)} "
        f"from={args.start} only={args.only} dry_run={args.dry_run}"
    )

    for i, stage in enumerate(STAGES, start=1):
        if args.only is not None and i != args.only:
            continue
        if args.only is None and i < args.start:
            continue

        label = stage["label"]
        marker = _done_marker(label)
        if marker.exists() and not args.force and not args.dry_run:
            _log(f"Stage {i} '{label}' already complete (marker found) -- skipping.")
            continue

        cfg_path = _write_stage_config(stage)
        merged = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        L = merged.get("learner", {})
        _log(
            f"=== Stage {i}/{len(STAGES)}: {label} | exp={merged.get('experiment_name')} "
            f"arch={L.get('arch')} obs={merged.get('obs', {}).get('name')} "
            f"n_proc={L.get('n_proc')} limit={int(L.get('timestep_limit', 0)):,} ==="
        )

        cmd = [sys.executable, "-m", "rlbot.training.train", "--config", str(cfg_path)]
        if args.dry_run:
            cmd.append("--dry-run")

        rc = subprocess.run(cmd, cwd=str(REPO_ROOT)).returncode
        if rc != 0:
            _log(
                f"Stage {i} '{label}' exited with code {rc}. Stopping. "
                f"Fix the issue and re-run -- finished stages are skipped, this one resumes."
            )
            sys.exit(rc)

        if not args.dry_run:
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(f"done {datetime.now().isoformat()}\n", encoding="utf-8")
        _log(f"Stage {i} '{label}' complete.")

    _log("Curriculum finished (all selected stages done).")


if __name__ == "__main__":
    main()
