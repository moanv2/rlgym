"""AdvancedObs (107-dim) checkpoint-progression eval.

The AdvancedObs counterpart of scripts/eval_checkpoint_progression.py. That
script imports DefaultObs/89-dim helpers from rlbot.evaluation.evaluate, so it
CANNOT evaluate papaya (AdvancedObs, 107-dim) -- every checkpoint would fail to
load with a size mismatch. This script reproduces the same progression curve
(blue_win_rate vs cumulative_timesteps, logged to wandb) but with a 107-dim
AdvancedObs env and a loader that infers each net's arch from its own weights.

Intended use: measure papaya's win rate over training against a FIXED teammate
reference bot (Martin / Nachi). Both papaya and the reference must be 107-dim
AdvancedObs bots -- the loader validates this and errors clearly otherwise.

The teammate's PPO_POLICY.pt is gitignored on their branch, so you need their
checkpoint folder locally (have them share it). --reference points at it.

Usage:

    python scripts/eval_progression_advobs.py ^
        --experiment papaya_1024 ^
        --reference path\to\martin\checkpoints\<exp>\<timestep> ^
        --episodes-per-checkpoint 30 ^
        --subsample-every 2

--reference also accepts any parent folder (latest checkpoint beneath it is
auto-selected).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
HISTORY_DIR = REPO_ROOT / "history_and_summary"

# AdvancedObs 1v1 spec.
OBS_DIM_1V1 = 107
N_ACTIONS = 90


# ----------------------------------------------------------------------------
# AdvancedObs env + 107-dim loader (the parts that differ from the 89-dim script)
# ----------------------------------------------------------------------------
def build_eval_env():
    """1v1 AdvancedObs env, DefaultState kickoff, goal/timeout termination."""
    import rlgym_sim
    from rlgym_sim.utils.obs_builders import AdvancedObs
    from rlgym_sim.utils.reward_functions import DefaultReward
    from rlgym_sim.utils.state_setters import DefaultState
    from rlgym_sim.utils.terminal_conditions.common_conditions import (
        GoalScoredCondition,
        TimeoutCondition,
    )

    from rlbot.actions.lookup_action import LookupAction

    return rlgym_sim.make(
        tick_skip=8,
        team_size=1,
        spawn_opponents=True,
        obs_builder=AdvancedObs(),
        action_parser=LookupAction(),
        reward_fn=DefaultReward(),
        state_setter=DefaultState(),
        terminal_conditions=[GoalScoredCondition(), TimeoutCondition(3000)],
    )


def load_policy(ckpt_dir: Path, device: str = "cpu", label: str = "policy"):
    """Build a DiscreteFF from the checkpoint's own weights (arch inferred) and
    load them. Validates input dim == 107 so a DefaultObs (89) checkpoint fails
    loudly instead of silently."""
    from rlgym_ppo.ppo.discrete_policy import DiscreteFF

    sd = torch.load(ckpt_dir / "PPO_POLICY.pt", map_location=device, weights_only=True)
    weight_keys = [k for k in sd if k.endswith("weight")]
    in_dim = int(sd[weight_keys[0]].shape[1])
    out_dim = int(sd[weight_keys[-1]].shape[0])
    hidden = tuple(int(sd[k].shape[0]) for k in weight_keys[:-1])

    if in_dim != OBS_DIM_1V1:
        raise ValueError(
            f"{label} obs mismatch at {ckpt_dir}: checkpoint expects input dim "
            f"{in_dim}, this eval feeds AdvancedObs ({OBS_DIM_1V1}). A DefaultObs "
            f"(89) bot cannot be evaluated here."
        )
    if out_dim != N_ACTIONS:
        raise ValueError(f"{label} action mismatch at {ckpt_dir}: {out_dim} != {N_ACTIONS}")

    policy = DiscreteFF(OBS_DIM_1V1, N_ACTIONS, hidden, device)
    policy.load_state_dict(sd)
    policy.eval()
    return policy


def resolve_checkpoint(path_str: str) -> Path:
    """Return a timestep folder with PPO_POLICY.pt, auto-selecting the latest
    checkpoint beneath the path if it isn't a leaf checkpoint."""
    p = Path(path_str)
    if not p.is_absolute():
        p = REPO_ROOT / p
    if not p.exists():
        raise SystemExit(f"Reference path does not exist: {p}")
    if (p / "PPO_POLICY.pt").is_file():
        return p
    candidates = [
        d for d in p.rglob("*")
        if d.is_dir() and d.name.isdigit() and (d / "PPO_POLICY.pt").is_file()
    ]
    if not candidates:
        raise SystemExit(f"No checkpoint with PPO_POLICY.pt found under {p}")
    return max(candidates, key=lambda d: int(d.name))


def _action_to_int(action) -> int:
    if isinstance(action, np.ndarray):
        return int(action.flat[0])
    if isinstance(action, torch.Tensor):
        return int(action.item())
    return int(action)


# ----------------------------------------------------------------------------
# Checkpoint discovery (papaya layout: diego-bots/checkpoints/<exp>/<session>/<ts>/)
# ----------------------------------------------------------------------------
def find_checkpoints(experiment: str) -> list[tuple[int, Path]]:
    """All valid checkpoint folders for the experiment, sorted by timestep.
    A valid checkpoint has both PPO_POLICY.pt and BOOK_KEEPING_VARS.json."""
    candidates: list[tuple[int, Path]] = []

    diego_path = REPO_ROOT / "diego-bots" / "checkpoints" / experiment
    if diego_path.exists():
        for session_dir in diego_path.iterdir():
            if not session_dir.is_dir():
                continue
            for ts_dir in session_dir.iterdir():
                if (
                    ts_dir.is_dir()
                    and ts_dir.name.isdigit()
                    and (ts_dir / "PPO_POLICY.pt").is_file()
                ):
                    candidates.append((int(ts_dir.name), ts_dir))

    standard_path = REPO_ROOT / "checkpoints" / experiment
    if standard_path.exists():
        for ts_dir in standard_path.iterdir():
            if (
                ts_dir.is_dir()
                and ts_dir.name.isdigit()
                and (ts_dir / "PPO_POLICY.pt").is_file()
            ):
                candidates.append((int(ts_dir.name), ts_dir))

    seen: dict[int, Path] = {}
    for ts, path in candidates:
        if ts not in seen:
            seen[ts] = path
    return sorted(seen.items(), key=lambda kv: kv[0])


# ----------------------------------------------------------------------------
# Episode + checkpoint evaluation (identical metrics to the 89-dim script)
# ----------------------------------------------------------------------------
def _play_one_episode(env, blue_policy, orange_policy, deterministic: bool) -> tuple[int, float]:
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
    return int(info.get("result", 0)), n_steps * 8.0 / 120.0


def eval_one_checkpoint(env, blue_policy, orange_policy, episodes: int, deterministic: bool) -> dict:
    blue_wins = orange_wins = draws = 0
    total_seconds = 0.0
    total_goal_diff = blue_goals = orange_goals = 0
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
    minutes = total_seconds / 60.0 if total_seconds > 0 else 1.0
    return {
        "blue_wins": blue_wins,
        "orange_wins": orange_wins,
        "draws": draws,
        "blue_win_rate": round(blue_wins / episodes if episodes else 0.0, 4),
        "goal_differential": total_goal_diff,
        "blue_goals_total": blue_goals,
        "orange_goals_total": orange_goals,
        "avg_episode_seconds": round(total_seconds / episodes if episodes else 0.0, 2),
        "goals_scored_per_min": round(blue_goals / minutes, 3),
        "goals_conceded_per_min": round(orange_goals / minutes, 3),
        "goal_diff_per_min": round(total_goal_diff / minutes, 3),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment", default="papaya_1024", help="Experiment folder (default: papaya_1024)")
    p.add_argument("--reference", required=True,
                   help="Fixed orange opponent: teammate (Martin/Nachi) checkpoint or parent folder. Must be AdvancedObs (107).")
    p.add_argument("--episodes-per-checkpoint", type=int, default=30)
    p.add_argument("--device", default="cpu")
    p.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--subsample-every", type=int, default=1)
    p.add_argument("--wandb-project", default="rlgym-finalproject")
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--no-wandb", action="store_true")
    args = p.parse_args()

    reference_path = resolve_checkpoint(args.reference)
    print(f"Reference (orange): {reference_path}")

    checkpoints = find_checkpoints(args.experiment)
    if not checkpoints:
        print(f"ERROR: no checkpoints found for experiment '{args.experiment}'.", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(checkpoints)} checkpoint(s) for {args.experiment}")
    if args.subsample_every > 1:
        checkpoints = checkpoints[::args.subsample_every]
        print(f"Subsampled to {len(checkpoints)} checkpoints (every {args.subsample_every}th)")

    wandb_run = None
    if not args.no_wandb:
        import wandb

        ref_short = reference_path.name
        default_name = (
            f"eval_progression_{args.experiment}_vs_{ref_short}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M')}"
        )
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or default_name,
            tags=["eval_progression", "advanced_obs", args.experiment],
            config={
                "experiment": args.experiment,
                "reference_path": str(reference_path),
                "obs": "advanced_107",
                "episodes_per_checkpoint": args.episodes_per_checkpoint,
                "deterministic": args.deterministic,
                "subsample_every": args.subsample_every,
                "num_checkpoints_evaluated": len(checkpoints),
            },
        )
        wandb.define_metric("cumulative_timesteps")
        wandb.define_metric("*", step_metric="cumulative_timesteps")
        print(f"wandb run: {wandb_run.url}")

    print("Loading reference (orange) policy ...")
    try:
        orange_policy = load_policy(reference_path, args.device, label="reference")
    except ValueError as exc:
        raise SystemExit(f"ERROR: bad reference checkpoint -- {exc}")
    print("Building AdvancedObs eval env (1v1, DefaultState kickoff) ...")
    env = build_eval_env()

    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    json_path = HISTORY_DIR / f"eval_progression_advobs_{args.experiment}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    all_results: list[dict] = []
    started_at = datetime.now(timezone.utc)
    header = f"\n{'#':>3} | {'cum_ts':>13} | {'win_rate':>9} | {'W/L/D':>10} | {'goal_diff':>9} | {'avg_ep_s':>9} | {'eval_s':>7}"
    print(header)
    print("-" * len(header))

    try:
        for idx, (cum_ts, ckpt_path) in enumerate(checkpoints):
            t_start = time.time()
            try:
                blue_policy = load_policy(ckpt_path, args.device, label="checkpoint")
                metrics = eval_one_checkpoint(
                    env, blue_policy, orange_policy, args.episodes_per_checkpoint, args.deterministic
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
                f"{metrics['goal_differential']:>+9d} | {metrics['avg_episode_seconds']:>9.1f} | {elapsed:>7.1f}"
            )
            if wandb_run is not None:
                wandb.log({"cumulative_timesteps": cum_ts, **{k: v for k, v in metrics.items()}, "eval_seconds": elapsed})
            with json_path.open("w", encoding="utf-8") as f:
                json.dump({
                    "experiment": args.experiment,
                    "reference_path": str(reference_path),
                    "obs": "advanced_107",
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

    print(f"\nEvaluated {len(all_results)} checkpoint(s) for '{args.experiment}'.")
    print(f"Results JSON: {json_path}")
    if all_results:
        first, last = all_results[0], all_results[-1]
        print(f"  First @ {first['cumulative_timesteps']:>13,} ts -> win_rate {first['blue_win_rate']:.3f}")
        print(f"  Last  @ {last['cumulative_timesteps']:>13,} ts -> win_rate {last['blue_win_rate']:.3f}")


if __name__ == "__main__":
    main()
