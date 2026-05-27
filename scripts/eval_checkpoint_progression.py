"""Evaluate every saved checkpoint in an experiment against a fixed
reference opponent, logging per-checkpoint win rate / goal metrics to wandb.

Designed to produce the chart you actually need under ZeroSumReward training:
`blue_win_rate vs cumulative_timesteps` — a curve that goes up as the bot
learns, instead of the raw Policy Reward which is forced to ~0 in symmetric
self-play.

See docs/eval_dashboard_plan.md for the full design rationale.

Usage:

    python scripts/eval_checkpoint_progression.py \
        --experiment nexto_plus_kickoff \
        --reference "diego-bots/checkpoints/nexto_rewards/nexto_rewards-1779876636941376400/130506086" \
        --episodes-per-checkpoint 30 \
        --subsample-every 2

Or with the 'latest:<exp>' shorthand to pin to the latest of another experiment:

    python scripts/eval_checkpoint_progression.py \
        --experiment nexto_plus_kickoff \
        --reference latest:nexto_rewards \
        --episodes-per-checkpoint 30
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np  # noqa: F401  # imported by helpers below
import torch

from rlbot.evaluation.evaluate import (
    _action_to_int,
    _build_eval_env,
    _load_policy,
    _resolve_checkpoint_path,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
HISTORY_DIR = REPO_ROOT / "history_and_summary"


# ────────────────────────────────────────────────────────────────────────────
# Checkpoint discovery
# ────────────────────────────────────────────────────────────────────────────
def find_checkpoints(experiment: str) -> list[tuple[int, Path]]:
    """Find all valid checkpoint folders for the given experiment.

    Tries both layouts the project produces:
        diego-bots/checkpoints/<experiment>/<session>/<timestep>/
        checkpoints/<experiment>/<timestep>/

    A 'valid' checkpoint has a BOOK_KEEPING_VARS.json inside. Returns a list
    of (cumulative_timesteps_int, path) sorted ascending by timestep.
    """
    candidates: list[tuple[int, Path]] = []

    # Layout A: diego-bots/checkpoints/<exp>/<session>/<ts>/
    diego_path = REPO_ROOT / "diego-bots" / "checkpoints" / experiment
    if diego_path.exists():
        for session_dir in diego_path.iterdir():
            if not session_dir.is_dir():
                continue
            for ts_dir in session_dir.iterdir():
                if (
                    ts_dir.is_dir()
                    and ts_dir.name.isdigit()
                    and (ts_dir / "BOOK_KEEPING_VARS.json").exists()
                ):
                    candidates.append((int(ts_dir.name), ts_dir))

    # Layout B: checkpoints/<exp>/<ts>/  (Marian's YAML-driven training layout)
    standard_path = REPO_ROOT / "checkpoints" / experiment
    if standard_path.exists():
        for ts_dir in standard_path.iterdir():
            if (
                ts_dir.is_dir()
                and ts_dir.name.isdigit()
                and (ts_dir / "BOOK_KEEPING_VARS.json").exists()
            ):
                candidates.append((int(ts_dir.name), ts_dir))

    # Dedupe by timestep (in case both layouts contain the same one) and sort
    seen: dict[int, Path] = {}
    for ts, path in candidates:
        # Prefer the diego-bots layout if duplicates exist (it is the latest session)
        if ts not in seen:
            seen[ts] = path
    return sorted(seen.items(), key=lambda kv: kv[0])


# ────────────────────────────────────────────────────────────────────────────
# Per-episode and per-checkpoint evaluation
# ────────────────────────────────────────────────────────────────────────────
def _play_one_episode(env, blue_policy, orange_policy, deterministic: bool) -> tuple[int, float]:
    """Play one full episode, return (result, simulated_seconds).

    result encoding (from rlgym_sim):
        > 0  blue net-scored
        < 0  orange net-scored
        = 0  draw / timeout
    """
    obs_list = env.reset()
    blue_obs, orange_obs = obs_list[0], obs_list[1]
    done = False
    info: dict = {}
    n_steps = 0

    while not done:
        with torch.no_grad():
            b_act, _ = blue_policy.get_action(blue_obs, deterministic=deterministic)
            o_act, _ = orange_policy.get_action(orange_obs, deterministic=deterministic)
        obs_list, _r, done, info = env.step([_action_to_int(b_act), _action_to_int(o_act)])
        blue_obs, orange_obs = obs_list[0], obs_list[1]
        n_steps += 1

    # tick_skip=8 in _build_eval_env, 120 physics ticks/sec → 1 step ≈ 1/15 s
    episode_seconds = n_steps * 8.0 / 120.0
    return int(info.get("result", 0)), episode_seconds


def eval_one_checkpoint(
    env,
    blue_policy,
    orange_policy,
    episodes: int,
    deterministic: bool,
) -> dict:
    """Aggregate N episodes into per-checkpoint metrics."""
    blue_wins = 0
    orange_wins = 0
    draws = 0
    total_seconds = 0.0
    total_goal_diff = 0
    blue_goals = 0
    orange_goals = 0

    for _ in range(episodes):
        result, ep_sec = _play_one_episode(env, blue_policy, orange_policy, deterministic)
        total_seconds += ep_sec
        total_goal_diff += result
        if result > 0:
            blue_wins += 1
            blue_goals += result
        elif result < 0:
            orange_wins += 1
            orange_goals += -result
        else:
            draws += 1

    avg_ep_sec = total_seconds / episodes if episodes else 0.0
    win_rate = blue_wins / episodes if episodes else 0.0
    minutes = total_seconds / 60.0 if total_seconds > 0 else 1.0  # avoid /0
    goals_scored_per_min = blue_goals / minutes
    goals_conceded_per_min = orange_goals / minutes
    goal_diff_per_min = total_goal_diff / minutes

    return {
        "blue_wins": blue_wins,
        "orange_wins": orange_wins,
        "draws": draws,
        "blue_win_rate": round(win_rate, 4),
        "goal_differential": total_goal_diff,
        "blue_goals_total": blue_goals,
        "orange_goals_total": orange_goals,
        "avg_episode_seconds": round(avg_ep_sec, 2),
        "goals_scored_per_min": round(goals_scored_per_min, 3),
        "goals_conceded_per_min": round(goals_conceded_per_min, 3),
        "goal_diff_per_min": round(goal_diff_per_min, 3),
    }


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--experiment", required=True,
                   help="Experiment folder name (e.g. nexto_plus_kickoff)")
    p.add_argument("--reference", required=True,
                   help="Reference (orange) checkpoint path or 'latest:<exp>'")
    p.add_argument("--episodes-per-checkpoint", type=int, default=30,
                   help="Number of episodes to play per checkpoint (default: 30)")
    p.add_argument("--device", default="cpu",
                   help="Torch device ('cpu' is fine for eval; default: cpu)")
    p.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True,
                   help="Greedy argmax (default: True). Use --no-deterministic for stochastic.")
    p.add_argument("--subsample-every", type=int, default=1,
                   help="Eval every Nth checkpoint (default: 1 = all)")
    p.add_argument("--wandb-project", default="rlgym-finalproject",
                   help="wandb project to log to (default: rlgym-finalproject)")
    p.add_argument("--wandb-run-name", default=None,
                   help="Override wandb run name (default: auto)")
    p.add_argument("--no-wandb", action="store_true",
                   help="Skip wandb logging entirely")
    args = p.parse_args()

    # ── 1. Resolve reference, find checkpoints ─────────────────────────────
    reference_path = _resolve_checkpoint_path(args.reference)
    print(f"Reference (orange): {reference_path}")

    checkpoints = find_checkpoints(args.experiment)
    if not checkpoints:
        print(
            f"ERROR: no checkpoints found for experiment '{args.experiment}'. "
            f"Looked under diego-bots/checkpoints/ and checkpoints/.",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"Found {len(checkpoints)} checkpoint(s) for {args.experiment}")

    if args.subsample_every > 1:
        checkpoints = checkpoints[::args.subsample_every]
        print(f"Subsampled to {len(checkpoints)} checkpoints (every {args.subsample_every}th)")

    # ── 2. wandb setup ─────────────────────────────────────────────────────
    wandb_run = None
    if not args.no_wandb:
        import wandb

        ref_short = reference_path.name  # numeric timestep suffix
        default_name = (
            f"eval_progression_{args.experiment}_vs_{ref_short}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M')}"
        )
        run_name = args.wandb_run_name or default_name

        wandb_run = wandb.init(
            project=args.wandb_project,
            name=run_name,
            tags=["eval_progression", args.experiment],
            config={
                "experiment": args.experiment,
                "reference_path": str(reference_path),
                "reference_cumulative_timesteps": int(ref_short) if ref_short.isdigit() else None,
                "episodes_per_checkpoint": args.episodes_per_checkpoint,
                "deterministic": args.deterministic,
                "subsample_every": args.subsample_every,
                "num_checkpoints_evaluated": len(checkpoints),
            },
        )
        # Make wandb plot every metric against cumulative_timesteps on the X axis.
        wandb.define_metric("cumulative_timesteps")
        wandb.define_metric("*", step_metric="cumulative_timesteps")
        print(f"wandb run: {wandb_run.url}")

    # ── 3. Load reference, build env, prep output ─────────────────────────
    print("Loading reference (orange) policy ...")
    orange_policy = _load_policy(reference_path, args.device)
    print("Building eval env (1v1, DefaultState kickoff) ...")
    env = _build_eval_env()

    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    json_path = HISTORY_DIR / (
        f"eval_progression_{args.experiment}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )

    all_results: list[dict] = []
    started_at = datetime.now(timezone.utc)

    # ── 4. Evaluate each checkpoint ────────────────────────────────────────
    header = f"\n{'#':>3} | {'cum_ts':>13} | {'win_rate':>9} | {'W/L/D':>10} | {'goal_diff':>9} | {'avg_ep_s':>9} | {'eval_s':>7}"
    print(header)
    print("-" * len(header))

    try:
        for idx, (cum_ts, ckpt_path) in enumerate(checkpoints):
            t_start = time.time()
            try:
                blue_policy = _load_policy(ckpt_path, args.device)
                metrics = eval_one_checkpoint(
                    env,
                    blue_policy,
                    orange_policy,
                    args.episodes_per_checkpoint,
                    args.deterministic,
                )
            except Exception as exc:
                print(f"{idx + 1:>3} | {cum_ts:>13,} | ERROR loading/evaluating: {exc}")
                continue

            elapsed = time.time() - t_start

            row = {
                "checkpoint_index": idx,
                "cumulative_timesteps": cum_ts,
                "checkpoint_path": str(ckpt_path),
                "eval_seconds": round(elapsed, 1),
                **metrics,
            }
            all_results.append(row)

            print(
                f"{idx + 1:>3} | {cum_ts:>13,} | {metrics['blue_win_rate']:>9.3f} | "
                f"{metrics['blue_wins']}/{metrics['orange_wins']}/{metrics['draws']:<6} | "
                f"{metrics['goal_differential']:>+9d} | {metrics['avg_episode_seconds']:>9.1f} | "
                f"{elapsed:>7.1f}"
            )

            if wandb_run is not None:
                wandb.log({
                    "cumulative_timesteps": cum_ts,
                    "blue_win_rate": metrics["blue_win_rate"],
                    "blue_wins": metrics["blue_wins"],
                    "orange_wins": metrics["orange_wins"],
                    "draws": metrics["draws"],
                    "goal_differential": metrics["goal_differential"],
                    "blue_goals_total": metrics["blue_goals_total"],
                    "orange_goals_total": metrics["orange_goals_total"],
                    "avg_episode_seconds": metrics["avg_episode_seconds"],
                    "goals_scored_per_min": metrics["goals_scored_per_min"],
                    "goals_conceded_per_min": metrics["goals_conceded_per_min"],
                    "goal_diff_per_min": metrics["goal_diff_per_min"],
                    "eval_seconds": elapsed,
                })

            # Incremental JSON write — survive Ctrl+C and crashes
            with json_path.open("w", encoding="utf-8") as f:
                json.dump({
                    "experiment": args.experiment,
                    "reference_path": str(reference_path),
                    "reference_cumulative_timesteps": (
                        int(reference_path.name) if reference_path.name.isdigit() else None
                    ),
                    "started_at": started_at.isoformat(),
                    "deterministic": args.deterministic,
                    "episodes_per_checkpoint": args.episodes_per_checkpoint,
                    "subsample_every": args.subsample_every,
                    "results": all_results,
                }, f, indent=2, default=str)

    except KeyboardInterrupt:
        print("\n[interrupted] partial results saved to JSON.")
    finally:
        try:
            env.close()
        except Exception:
            pass
        if wandb_run is not None:
            wandb.finish()

    # ── 5. Final summary ───────────────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print(f"Evaluated {len(all_results)} checkpoint(s) for '{args.experiment}'.")
    print(f"Results JSON: {json_path}")
    if wandb_run is not None:
        print(f"Wandb URL:    {wandb_run.url}")

    if all_results:
        first = all_results[0]
        last = all_results[-1]
        print(f"\nProgression summary:")
        print(f"  First eval  @ {first['cumulative_timesteps']:>13,} ts  →  win_rate {first['blue_win_rate']:.3f}")
        print(f"  Last eval   @ {last['cumulative_timesteps']:>13,} ts  →  win_rate {last['blue_win_rate']:.3f}")
        delta = last["blue_win_rate"] - first["blue_win_rate"]
        sign = "+" if delta >= 0 else ""
        print(f"  Δ win_rate                                 :  {sign}{delta:.3f}")


if __name__ == "__main__":
    main()
