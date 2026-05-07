"""Bot-vs-bot evaluation. Runs N episodes between two checkpoints, reports win rate.

    python -m rlbot.evaluation.evaluate \
        --blue checkpoints/exp_002_curriculum/latest \
        --orange checkpoints/exp_001_baseline/latest \
        --episodes 100

Used by CI smoke tests (with episodes=1) and by you to confirm a new run is
actually stronger before adopting it as the new reference bot.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from rlbot.utils.logging import get_logger


def evaluate(blue_path: str, orange_path: str, episodes: int, deterministic: bool) -> dict:
    """Returns a metrics dict. Stub — fill in once we have the policy-loading code wired
    against rlgym-ppo's checkpoint format and a 1v1 env."""
    log = get_logger("rlbot.evaluate")

    log.info(f"Evaluating [cyan]{blue_path}[/] vs [magenta]{orange_path}[/] for {episodes} eps")
    log.warning("evaluate(): stub — full implementation is part of the eval pipeline milestone.")

    return {
        "blue": str(Path(blue_path)),
        "orange": str(Path(orange_path)),
        "episodes": episodes,
        "deterministic": deterministic,
        "blue_wins": 0,
        "orange_wins": 0,
        "draws": episodes,
        "blue_win_rate": 0.0,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--blue", required=True, help="Path to blue policy checkpoint")
    p.add_argument("--orange", required=True, help="Path to orange policy checkpoint")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--deterministic", action="store_true")
    args = p.parse_args()

    result = evaluate(args.blue, args.orange, args.episodes, args.deterministic)
    log = get_logger("rlbot.evaluate")
    log.info(f"Result: blue_win_rate={result['blue_win_rate']:.3f} "
             f"(W/L/D = {result['blue_wins']}/{result['orange_wins']}/{result['draws']})")


if __name__ == "__main__":
    main()
