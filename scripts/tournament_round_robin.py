"""Round-robin ranking of the team bots, in BOTH deterministic and stochastic
modes, for the presentation analysis.

The branch's built-in `rlbot.tournament.run` is single-elimination + deterministic
only. A bracket gives a coarse, draw-dependent 1..N order with no comparable
per-bot SCORE. For "rank the players + their score, deterministic vs stochastic so
we can compare," a round-robin is the right shape: every bot plays every other,
points accumulate, and the same number is produced in both modes.

Reuses the scaffolding's cross-obs match runner (`tournament.match.play_match`)
and checkpoint resolution (`tournament.roster`), so 89-dim DefaultObs and 107-dim
AdvancedObs bots play each other correctly (each car gets its own obs).

Scoring per mode:
  - Match = best-of-`games` (sides swapped each game). Match winner gets 3 points.
    (play_match always decides a winner via goal-diff -> stochastic sudden-death
    -> seed, so there are no match draws.)
  - Game-level: total games won/lost/drawn, game win-rate, aggregate goal diff.
  - Rank by: match points -> aggregate goal diff -> games won.

Usage (rlbot310, repo root):
    python scripts/tournament_round_robin.py --games 10 --mode both
    python scripts/tournament_round_robin.py --games 2 --mode deterministic   # pilot
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HISTORY_DIR = REPO_ROOT / "history_and_summary"

# The field. diego + marian + marco are one-per-person; Martin appears twice
# (2.1B and 8.23B) so the analysis can see the training-scale effect. Nachi is
# excluded: he committed Git-LFS *pointers* but never pushed the actual binary
# to LFS storage, so his weights aren't retrievable — add him once he re-pushes.
FOUR = [
    {"owner": "diego",  "name": "Diego — papaya 3.5B",  "path": "diego-bots/checkpoints/papaya_1024"},
    {"owner": "martin", "name": "Martin — champ 8.23B", "path": "martin-bots/checkpoints/CHAMPION_8.23B_advanced1024"},
    {"owner": "nachi",  "name": "Nachi — 2.9B",         "path": "teammates/nachi"},
    {"owner": "marco",  "name": "Marco — 2.0B",         "path": "teammates/marco"},
    {"owner": "marian", "name": "Marian — 1.35B",       "path": "checkpoints/marian_iterations/1349081288"},
]


def build_four():
    from rlbot.tournament.roster import build_roster
    bots = build_roster(FOUR, verbose=True)
    if len(bots) != len(FOUR):
        missing = {e["owner"] for e in FOUR} - {b.owner for b in bots}
        raise SystemExit(f"Could not resolve all 4 bots; missing: {missing}")
    return bots


def run_round_robin(bots, games: int, deterministic: bool):
    from rlbot.tournament.match import play_match

    table = {
        b.owner: {
            "bot": b, "owner": b.owner, "name": b.name,
            "steps": b.timesteps, "obs": b.obs_dim,
            "mw": 0, "ml": 0, "gw": 0, "gl": 0, "gd": 0, "gdiff": 0, "pts": 0,
        }
        for b in bots
    }
    matches = []
    mode = "DETERMINISTIC" if deterministic else "STOCHASTIC"
    for a, b in itertools.combinations(bots, 2):
        t0 = time.time()
        out = play_match(a, b, games=games, deterministic=deterministic)
        dt = time.time() - t0
        ta, tb = table[a.owner], table[b.owner]
        # game tallies (out is from a's perspective)
        ta["gw"] += out.a_wins; ta["gl"] += out.b_wins; ta["gd"] += out.draws; ta["gdiff"] += out.a_goal_diff
        tb["gw"] += out.b_wins; tb["gl"] += out.a_wins; tb["gd"] += out.draws; tb["gdiff"] -= out.a_goal_diff
        # match points
        if out.winner_owner == a.owner:
            ta["mw"] += 1; tb["ml"] += 1; ta["pts"] += 3
        else:
            tb["mw"] += 1; ta["ml"] += 1; tb["pts"] += 3
        print(f"  [{mode}] {a.owner:<12} vs {b.owner:<12}  "
              f"{out.a_wins}-{out.b_wins}" + (f" ({out.draws}D)" if out.draws else "")
              + f"  winner={out.winner_owner:<12} by {out.decided_by:<11} ({dt:.0f}s)")
        matches.append(out.to_detail() | {"winner": out.winner_owner})

    ranked = sorted(table.values(), key=lambda r: (r["pts"], r["gdiff"], r["gw"]), reverse=True)
    return ranked, matches


def print_table(title, ranked, games):
    print(f"\n================  {title}  ================")
    print(f"{'#':<3}{'bot':<22}{'match W-L':<11}{'games W-L-D':<14}{'win%':<8}{'goaldiff':<10}{'pts':<5}")
    print("-" * 73)
    for i, r in enumerate(ranked, 1):
        gp = r["gw"] + r["gl"] + r["gd"]
        wr = (r["gw"] / gp) if gp else 0.0
        match_rec = f"{r['mw']}-{r['ml']}"
        game_rec = f"{r['gw']}-{r['gl']}-{r['gd']}"
        winpct = f"{wr:.1%}"
        gdiff = f"{r['gdiff']:+d}"
        print(f"{i:<3}{r['name']:<22}{match_rec:<11}{game_rec:<14}{winpct:<8}{gdiff:<10}{r['pts']:<5}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=int, default=10, help="games per pairing (sides swap each game)")
    ap.add_argument("--mode", choices=["both", "deterministic", "stochastic"], default="both")
    args = ap.parse_args()

    print("Resolving roster:")
    bots = build_four()

    modes = (["deterministic", "stochastic"] if args.mode == "both" else [args.mode])
    results = {"games_per_pairing": args.games, "bots": [
        {"owner": b.owner, "name": b.name, "obs_dim": b.obs_dim,
         "hidden": list(b.hidden_sizes), "steps": b.timesteps, "checkpoint": str(b.checkpoint)}
        for b in bots], "modes": {}}

    for m in modes:
        det = (m == "deterministic")
        print(f"\n>>> Running {m.upper()} round-robin ({args.games} games/pairing, "
              f"{len(bots)*(len(bots)-1)//2} pairings)...")
        ranked, matches = run_round_robin(bots, args.games, det)
        print_table(f"{m.upper()} RANKING", ranked, args.games)
        results["modes"][m] = {
            "ranking": [
                {"rank": i, "owner": r["owner"], "name": r["name"],
                 "match_w": r["mw"], "match_l": r["ml"],
                 "games_w": r["gw"], "games_l": r["gl"], "games_d": r["gd"],
                 "game_win_rate": round(r["gw"] / max(1, r["gw"] + r["gl"] + r["gd"]), 3),
                 "goal_diff": r["gdiff"], "points": r["pts"]}
                for i, r in enumerate(ranked, 1)
            ],
            "matches": matches,
        }

    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    out = HISTORY_DIR / f"tournament_round_robin_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nJSON: {out}")


if __name__ == "__main__":
    main()
