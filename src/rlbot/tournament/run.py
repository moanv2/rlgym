"""Run the bot tournament: discover the roster, play a seeded single-elim bracket,
write the ranking to JSON.

    conda activate rlbot310
    python -m rlbot.tournament.run                     # best-of-5 deterministic, all present bots
    python -m rlbot.tournament.run --games 3           # shorter matches
    python -m rlbot.tournament.run --out results.json  # custom output path

Headless (no rlviser needed) -- runs at full sim speed. Use rlbot.tournament.record
afterwards to render the matches for video.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .bracket import run_bracket
from .match import make_play_match_fn
from .roster import REPO_ROOT, build_roster

DEFAULT_OUT = REPO_ROOT / "history_and_summary" / "tournament_results.json"


def _print_ranking(result: dict) -> None:
    print("\n" + "=" * 52)
    print(f"  TOURNAMENT RESULT — champion: {result['champion'].upper()}")
    print("=" * 52)
    for row in result["ranking"]:
        print(f"  {row['position']}.  {row['name']}  (seed {row['seed']})")
    print("=" * 52)


def main() -> None:
    p = argparse.ArgumentParser(description="Run the bot-vs-bot single-elim tournament.")
    p.add_argument("--games", type=int, default=5, help="Games per match (best-of-N). Default 5.")
    p.add_argument("--stochastic", action="store_true",
                   help="Sample actions instead of argmax (more variety, weaker for some bots).")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help=f"Where to write results JSON. Default: {DEFAULT_OUT}")
    args = p.parse_args()

    print("Discovering roster...")
    bots = build_roster()
    if len(bots) < 2:
        raise SystemExit(
            f"\nOnly {len(bots)} bot(s) present — need at least 2 to run a tournament.\n"
            "Download the missing teammates first:  python -m rlbot.tournament.download"
        )
    print(f"\n{len(bots)} bots in the bracket. Seeding by timesteps (strongest first)...")

    play = make_play_match_fn(games=args.games, deterministic=not args.stochastic)
    result = run_bracket(bots, play)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    _print_ranking(result)
    print(f"\nResults written to {args.out}")


if __name__ == "__main__":
    main()
