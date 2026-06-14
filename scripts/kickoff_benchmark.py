"""Kickoff benchmark -- the attributable metric for the v7 fast-kickoff package.

Plays every canonical kickoff spawn (0-4) in BOTH color assignments against a
fixed opponent, deterministic argmax on both sides, and measures:

  - time_to_first_touch : seconds from kickoff to the first ball contact
  - first_toucher       : papaya / opponent / none (15s cap)
  - ball_adv_3s         : ball's Y position (in papaya's attacking direction)
                          3 seconds after the first touch -- did winning the
                          touch actually win territory?
  - outcome             : goal result within the 15s window (papaya/opponent/none)

Because both policies are deterministic and the 5 spawns are exact, this is a
complete enumeration of the kickoff matchup (10 distinct playouts), not a
sample. Run BEFORE a kickoff-targeted change for the baseline and AFTER for
the delta.

Usage (rlbot310, repo root):
    python scripts/kickoff_benchmark.py --label pre_v7
    python scripts/kickoff_benchmark.py --opponent <ckpt-or-parent> --label post_v7
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
HISTORY_DIR = REPO_ROOT / "history_and_summary"

OBS_DIM = 107
N_ACTIONS = 90
TICK_SKIP = 8
SECONDS_PER_STEP = TICK_SKIP / 120.0
EPISODE_CAP_S = 15.0
POST_TOUCH_WINDOW_S = 3.0

DEFAULT_PAPAYA = "diego-bots/checkpoints/papaya_1024"
DEFAULT_OPPONENT = "martin-bots/checkpoints/CHAMPION_2.1B_recipeD_advanced1024"


def resolve_checkpoint(path_str: str) -> Path:
    p = Path(path_str)
    if not p.is_absolute():
        p = REPO_ROOT / p
    if not p.exists():
        raise SystemExit(f"Checkpoint path does not exist: {p}")
    if (p / "PPO_POLICY.pt").is_file():
        return p
    cands = [
        d for d in p.rglob("*")
        if d.is_dir() and d.name.isdigit() and (d / "PPO_POLICY.pt").is_file()
    ]
    if not cands:
        raise SystemExit(f"No checkpoint with PPO_POLICY.pt under {p}")
    return max(cands, key=lambda d: int(d.name))


def load_policy(ckpt_dir: Path, label: str):
    from rlgym_ppo.ppo.discrete_policy import DiscreteFF

    sd = torch.load(ckpt_dir / "PPO_POLICY.pt", map_location="cpu", weights_only=True)
    wk = [k for k in sd if k.endswith("weight")]
    in_dim = int(sd[wk[0]].shape[1])
    if in_dim != OBS_DIM:
        raise SystemExit(f"{label}: expects obs dim {in_dim}, benchmark feeds {OBS_DIM} (AdvancedObs)")
    hidden = tuple(int(sd[k].shape[0]) for k in wk[:-1])
    pol = DiscreteFF(OBS_DIM, N_ACTIONS, hidden, "cpu")
    pol.load_state_dict(sd)
    pol.eval()
    return pol


def build_env(spawn_idx: int):
    import rlgym_sim
    from rlgym_sim.utils.obs_builders import AdvancedObs
    from rlgym_sim.utils.reward_functions import DefaultReward
    from rlgym_sim.utils.terminal_conditions.common_conditions import (
        GoalScoredCondition,
        TimeoutCondition,
    )

    from rlbot.actions.lookup_action import LookupAction
    from rlbot.state_setters.kickoff_scenarios import FixedKickoffSetter

    cap_steps = int(EPISODE_CAP_S / SECONDS_PER_STEP)
    return rlgym_sim.make(
        tick_skip=TICK_SKIP,
        team_size=1,
        spawn_opponents=True,
        obs_builder=AdvancedObs(),
        action_parser=LookupAction(),
        reward_fn=DefaultReward(),
        state_setter=FixedKickoffSetter(spawn_idx),
        terminal_conditions=[GoalScoredCondition(), TimeoutCondition(cap_steps)],
    )


def _act(policy, obs) -> int:
    with torch.no_grad():
        a, _ = policy.get_action(obs, deterministic=True)
    if isinstance(a, np.ndarray):
        return int(a.flat[0])
    if isinstance(a, torch.Tensor):
        return int(a.item())
    return int(a)


def play_kickoff(spawn_idx: int, papaya, opponent, papaya_is_blue: bool) -> dict:
    env = build_env(spawn_idx)
    obs_list = env.reset()
    blue_pol, orange_pol = (papaya, opponent) if papaya_is_blue else (opponent, papaya)
    blue_obs, orange_obs = obs_list[0], obs_list[1]

    first_touch_step = None
    first_toucher = None          # 'papaya' | 'opponent'
    ball_adv_3s = None
    done = False
    info: dict = {}
    step = 0

    while not done:
        b = _act(blue_pol, blue_obs)
        o = _act(orange_pol, orange_obs)
        obs_list, _r, done, info = env.step([b, o])
        blue_obs, orange_obs = obs_list[0], obs_list[1]
        step += 1

        state = getattr(env, "_prev_state", None)
        if state is not None:
            if first_touch_step is None:
                for p in state.players:
                    if p.ball_touched:
                        first_touch_step = step
                        toucher_is_blue = int(p.team_num) == 0
                        first_toucher = (
                            "papaya" if toucher_is_blue == papaya_is_blue else "opponent"
                        )
                        break
            elif ball_adv_3s is None and step - first_touch_step >= int(
                POST_TOUCH_WINDOW_S / SECONDS_PER_STEP
            ):
                ball_y = float(state.ball.position[1])
                ball_adv_3s = ball_y if papaya_is_blue else -ball_y

    # If the episode ended (goal) before the 3s window elapsed, use the goal sign.
    result = int(info.get("result", 0))  # >0 blue scored, <0 orange
    if result != 0:
        papaya_scored = (result > 0) == papaya_is_blue
        outcome = "papaya" if papaya_scored else "opponent"
    else:
        outcome = "none"

    env.close()
    return {
        "spawn_idx": spawn_idx,
        "papaya_color": "blue" if papaya_is_blue else "orange",
        "time_to_first_touch_s": (
            round(first_touch_step * SECONDS_PER_STEP, 3) if first_touch_step else None
        ),
        "first_toucher": first_toucher or "none",
        "ball_adv_3s_uu": round(ball_adv_3s, 1) if ball_adv_3s is not None else None,
        "outcome_15s": outcome,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--papaya", default=DEFAULT_PAPAYA)
    ap.add_argument("--opponent", default=DEFAULT_OPPONENT)
    ap.add_argument("--label", default="run", help="Tag for the output JSON (e.g. pre_v7 / post_v7)")
    args = ap.parse_args()

    papaya_ckpt = resolve_checkpoint(args.papaya)
    opp_ckpt = resolve_checkpoint(args.opponent)
    print(f"PAPAYA   {papaya_ckpt}")
    print(f"OPPONENT {opp_ckpt}")

    papaya = load_policy(papaya_ckpt, "papaya")
    opponent = load_policy(opp_ckpt, "opponent")

    spawn_names = ["right corner", "left corner", "back right", "back left", "far back center"]
    rows = []
    for spawn_idx in range(5):
        for papaya_is_blue in (True, False):
            rows.append(play_kickoff(spawn_idx, papaya, opponent, papaya_is_blue))

    print(f"\n{'spawn':<16} {'color':<7} {'1st touch':<10} {'t (s)':<7} {'ball_adv_3s':<12} {'outcome':<9}")
    print("-" * 65)
    for r in rows:
        print(
            f"{spawn_names[r['spawn_idx']]:<16} {r['papaya_color']:<7} "
            f"{r['first_toucher']:<10} {str(r['time_to_first_touch_s']):<7} "
            f"{str(r['ball_adv_3s_uu']):<12} {r['outcome_15s']:<9}"
        )

    n = len(rows)
    papaya_first = sum(1 for r in rows if r["first_toucher"] == "papaya")
    papaya_touch_times = [r["time_to_first_touch_s"] for r in rows if r["first_toucher"] == "papaya" and r["time_to_first_touch_s"]]
    opp_touch_times = [r["time_to_first_touch_s"] for r in rows if r["first_toucher"] == "opponent" and r["time_to_first_touch_s"]]
    adv_vals = [r["ball_adv_3s_uu"] for r in rows if r["ball_adv_3s_uu"] is not None]
    goals = {"papaya": 0, "opponent": 0, "none": 0}
    for r in rows:
        goals[r["outcome_15s"]] += 1

    summary = {
        "first_possession_rate": round(papaya_first / n, 3),
        "mean_touch_time_when_papaya_first_s": round(float(np.mean(papaya_touch_times)), 3) if papaya_touch_times else None,
        "mean_touch_time_when_opponent_first_s": round(float(np.mean(opp_touch_times)), 3) if opp_touch_times else None,
        "mean_ball_adv_3s_uu": round(float(np.mean(adv_vals)), 1) if adv_vals else None,
        "outcomes_15s": goals,
    }
    print("\nSummary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    out = HISTORY_DIR / f"kickoff_benchmark_{args.label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps({
        "label": args.label,
        "papaya": str(papaya_ckpt),
        "opponent": str(opp_ckpt),
        "rows": rows,
        "summary": summary,
    }, indent=2), encoding="utf-8")
    print(f"\nJSON: {out}")


if __name__ == "__main__":
    main()
