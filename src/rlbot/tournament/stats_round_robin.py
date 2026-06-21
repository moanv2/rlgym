"""Instrumented round-robin for the presentation analytics.

Plays a full round-robin (every bot vs every other, sides swapped each game) in a
chosen mode, and instruments EVERY step of EVERY game to collect rich per-bot
stats beyond win/loss:

  - goals for / against (each goal-terminated game = 1 goal to the winner)
  - saves (defensive clears -- heuristic, labelled as such)
  - demos inflicted (opponent.is_demoed 0->1 transitions)
  - % time airborne, % time dribbling, % time supersonic
  - possession % (share of time as the last bot to touch the ball)
  - average boost held
  - the ordered win/loss sequence per bot (for the win-rate convergence chart)
  - aggregate goal margin per matchup (for biggest-win / narrowest-loss)

Stochastic mode is the default and the right choice for averages: every game is
distinct, so N games is N independent samples (deterministic + a 5-kickoff state
setter would just repeat ~5 playouts). Writes one JSON consumed by
`scripts/tournament_charts.py`.

Usage (rlbot310, repo root):
    python -m rlbot.tournament.stats_round_robin --games 30 --mode stochastic
    #   5 bots -> 10 pairings x 30 = 300 games
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
HISTORY_DIR = REPO_ROOT / "history_and_summary"

# Full team, one bot per person (Martin = his strongest, 8.23B). Nachi included
# now that he pushed the real binary. Same field as the final ranking.
FIELD = [
    {"owner": "diego",  "name": "Diego — papaya v7",    "path": "diego-bots/checkpoints/papaya_1024"},
    {"owner": "martin", "name": "Martin — 9B",          "path": "martin-bots/checkpoints/CHAMPION_9.0B_advanced1024"},
    {"owner": "nachi",  "name": "Nachi — 2.9B",         "path": "teammates/nachi"},
    {"owner": "marco",  "name": "Marco — 2.0B",         "path": "teammates/marco"},
    {"owner": "marian", "name": "Marian — 1.35B",       "path": "checkpoints/marian_iterations/1349081288"},
]

SUPERSONIC = 2200.0
BACK_WALL_Y = 5120.0
# dribble pose: ball horizontally over the car and above the roof
DRIBBLE_HORIZ = 170.0
DRIBBLE_MIN_H = 110.0
DRIBBLE_MAX_H = 400.0


def _fresh() -> dict:
    return {"steps": 0, "air": 0, "ss": 0, "boost_sum": 0.0, "dribble": 0,
            "touches": 0, "saves": 0, "demos": 0, "poss": 0}


def play_game_instrumented(env, blue_pol, orange_pol, deterministic: bool):
    """Play one goal-terminated game; return (result, blue_stats, orange_stats).

    result: +1 blue scored, -1 orange scored, 0 timeout draw. Stats are per team
    (0=blue, 1=orange) accumulated over every step.
    """
    import torch

    from .policy_io import action_to_int

    obs = env.reset()
    blue_obs, orange_obs = obs[0], obs[1]
    stats = {0: _fresh(), 1: _fresh()}
    prev_demoed = {0: False, 1: False}
    last_toucher = None
    done = False
    info: dict = {}

    while not done:
        with torch.no_grad():
            b_act, _ = blue_pol.get_action(blue_obs, deterministic=deterministic)
            o_act, _ = orange_pol.get_action(orange_obs, deterministic=deterministic)
        obs, _r, done, info = env.step([action_to_int(b_act), action_to_int(o_act)])
        blue_obs, orange_obs = obs[0], obs[1]

        st = env._prev_state
        ball = st.ball.position
        bvel = st.ball.linear_velocity
        toucher = None
        for p in st.players:
            t = int(p.team_num)
            s = stats[t]
            s["steps"] += 1
            spd = float(np.linalg.norm(p.car_data.linear_velocity))
            if not p.on_ground:
                s["air"] += 1
            if spd >= SUPERSONIC:
                s["ss"] += 1
            s["boost_sum"] += float(p.boost_amount)
            cx, cy, cz = p.car_data.position
            horiz = ((ball[0] - cx) ** 2 + (ball[1] - cy) ** 2) ** 0.5
            height = ball[2] - cz
            if horiz < DRIBBLE_HORIZ and DRIBBLE_MIN_H < height < DRIBBLE_MAX_H:
                s["dribble"] += 1
            if p.ball_touched:
                s["touches"] += 1
                toucher = t
                # save (clear) heuristic: touched in own defensive half and the
                # ball is now moving away from own goal (cleared it).
                own_goal_sign = -1.0 if t == 0 else 1.0   # blue defends -Y, orange +Y
                ball_on_own_side = (float(ball[1]) * own_goal_sign) > 0
                cleared = (float(bvel[1]) * own_goal_sign) < -200.0
                if ball_on_own_side and cleared:
                    s["saves"] += 1
            # demo: this car just became demoed -> credit a demo to the other team
            if p.is_demoed and not prev_demoed[t]:
                stats[1 - t]["demos"] += 1
            prev_demoed[t] = p.is_demoed

        if toucher is not None:
            last_toucher = toucher
        if last_toucher is not None:
            stats[last_toucher]["poss"] += 1

    return int(info.get("result", 0)), stats[0], stats[1]


def main() -> None:
    from .obs import make_env as build_env
    from .policy_io import load_policy
    from .roster import build_roster

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=int, default=30, help="games per pairing (sides swap each game)")
    ap.add_argument("--mode", choices=["deterministic", "stochastic"], default="stochastic")
    args = ap.parse_args()
    deterministic = args.mode == "deterministic"

    print("Resolving roster:")
    bots = build_roster(FIELD, verbose=True)
    if len(bots) < 2:
        raise SystemExit("need >=2 bots")

    policies = {b.owner: load_policy(b.checkpoint, b.obs_dim) for b in bots}
    env_cache: dict[tuple[int, int], object] = {}

    def env_for(bd, od):
        if (bd, od) not in env_cache:
            env_cache[(bd, od)] = build_env(bd, od)
        return env_cache[(bd, od)]

    # per-bot aggregate accumulators
    agg = {b.owner: {"name": b.name, "obs": b.obs_dim, "steps_total": b.timesteps,
                     "games": 0, "wins": 0, "losses": 0, "draws": 0,
                     "saves": 0, "demos": 0, "air": 0, "dribble": 0, "ss": 0,
                     "poss": 0, "boost_sum": 0.0, "play_steps": 0,
                     "seq": []}  # ordered 1/0/0.5 outcomes for convergence
                for b in bots}
    matchups = []
    total_games = args.games * (len(bots) * (len(bots) - 1) // 2)
    progress = HISTORY_DIR / f"_progress_{args.mode}.json"
    done_pairs = set()
    if progress.exists():
        _p = json.loads(progress.read_text(encoding="utf-8"))
        agg = _p["agg"]; matchups = _p["matchups"]; done_pairs = {tuple(k) for k in _p["done_pairs"]}
        print(f"  [resume] {len(done_pairs)} pairings already done -> skipping them")
    print(f"\n>>> {args.mode.upper()} round-robin: {len(bots)} bots, "
          f"{len(bots)*(len(bots)-1)//2} pairings x {args.games} = {total_games} games\n")

    g_done = 0
    t_start = time.time()
    for a, b in itertools.combinations(bots, 2):
        if tuple(sorted([a.owner, b.owner])) in done_pairs:
            continue
        a_goals = b_goals = 0
        for g in range(args.games):
            a_is_blue = (g % 2 == 0)
            if a_is_blue:
                env = env_for(a.obs_dim, b.obs_dim)
                result, bs_blue, bs_orange = play_game_instrumented(env, policies[a.owner], policies[b.owner], deterministic)
                a_stats, b_stats = bs_blue, bs_orange
                a_scored = result > 0
                b_scored = result < 0
            else:
                env = env_for(b.obs_dim, a.obs_dim)
                result, bs_blue, bs_orange = play_game_instrumented(env, policies[b.owner], policies[a.owner], deterministic)
                b_stats, a_stats = bs_blue, bs_orange
                a_scored = result < 0
                b_scored = result > 0

            for owner, stt, scored, conceded in (
                (a.owner, a_stats, a_scored, b_scored),
                (b.owner, b_stats, b_scored, a_scored),
            ):
                ag = agg[owner]
                ag["games"] += 1
                ag["saves"] += stt["saves"]; ag["demos"] += stt["demos"]
                ag["air"] += stt["air"]; ag["dribble"] += stt["dribble"]; ag["ss"] += stt["ss"]
                ag["poss"] += stt["poss"]; ag["boost_sum"] += stt["boost_sum"]; ag["play_steps"] += stt["steps"]
                if scored:
                    ag["wins"] += 1; ag["seq"].append(1.0)
                elif conceded:
                    ag["losses"] += 1; ag["seq"].append(0.0)
                else:
                    ag["draws"] += 1; ag["seq"].append(0.5)
            a_goals += int(a_scored); b_goals += int(b_scored)
            g_done += 1

        matchups.append({"a": a.owner, "b": b.owner, "a_goals": a_goals, "b_goals": b_goals,
                         "margin": a_goals - b_goals})
        print(f"  {a.owner:<8} vs {b.owner:<8}  goals {a_goals}-{b_goals}  "
              f"({g_done}/{total_games} games, {time.time()-t_start:.0f}s)")
        done_pairs.add(tuple(sorted([a.owner, b.owner])))
        progress.write_text(json.dumps({"agg": agg, "matchups": matchups,
            "done_pairs": [list(k) for k in done_pairs]}), encoding="utf-8")

    for env in env_cache.values():
        env.close()

    # finalize per-bot derived metrics
    per_bot = {}
    for owner, ag in agg.items():
        gp = max(1, ag["games"]); ps = max(1, ag["play_steps"])
        per_bot[owner] = {
            "name": ag["name"], "obs_dim": ag["obs"], "train_steps": ag["steps_total"],
            "games": ag["games"], "wins": ag["wins"], "losses": ag["losses"], "draws": ag["draws"],
            "win_rate": round(ag["wins"] / gp, 4),
            "goals_for": ag["wins"], "goals_against": ag["losses"], "goal_diff": ag["wins"] - ag["losses"],
            "avg_goals": round(ag["wins"] / gp, 4),
            "avg_saves": round(ag["saves"] / gp, 4),
            "avg_demos": round(ag["demos"] / gp, 4),
            "air_pct": round(100 * ag["air"] / ps, 2),
            "dribble_pct": round(100 * ag["dribble"] / ps, 2),
            "supersonic_pct": round(100 * ag["ss"] / ps, 2),
            "possession_pct": round(100 * ag["poss"] / ps, 2),
            "avg_boost": round(ag["boost_sum"] / ps, 2),
            "seq": ag["seq"],
        }

    out = {
        "mode": args.mode, "games_per_pairing": args.games, "total_games": total_games,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "per_bot": per_bot, "matchups": matchups,
    }
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    path = HISTORY_DIR / f"tournament_stats_{args.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    if progress.exists():
        progress.unlink()
    print(f"\nWrote stats JSON: {path}")
    print(f"Total wall time: {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
