"""Render a bot's bracket match(es) in rlviser for ~60s of presentation video,
optionally screen-capturing the rlviser window with ffmpeg.

    conda activate rlbot310
    # 1) open rlviser.exe (an empty arena window appears)
    # 2) record one person's bracket footage (auto-captures if ffmpeg is present):
    python -m rlbot.tournament.record --owner diego --capture
    # every person, one file each under videos/:
    python -m rlbot.tournament.record --all --capture

How it picks the matchup:
    With a results JSON present (default: history_and_summary/tournament_results.json)
    it replays the actual bracket games that bot played (deepest match first). With
    no results, it falls back to <owner> vs the next-strongest present bot.

Footage is real-time; games are looped until --seconds of video is reached. If
ffmpeg isn't installed the exact gdigrab command is printed so you can capture the
rlviser window with OBS instead.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

from .match import play_match
from .roster import REPO_ROOT, Bot, build_roster

DEFAULT_RESULTS = REPO_ROOT / "history_and_summary" / "tournament_results.json"
VIDEO_DIR = REPO_ROOT / "videos"


def _opponents_from_results(owner: str, results: dict) -> list[str]:
    """Owners that `owner` faced in the bracket, deepest round first (best footage)."""
    matches: list[dict] = []
    for rnd in results.get("rounds", []):
        matches.extend(m for m in rnd["matches"] if m.get("type") == "match")
    if results.get("third_place"):
        matches.append(results["third_place"])
    depth = {"final": 5, "third-place": 4, "semifinal": 3, "quarterfinal": 2, "round-of-16": 1}
    faced: list[tuple[int, str]] = []
    for m in matches:
        if owner in (m.get("winner"), m.get("loser")):
            opp = m["loser"] if m["winner"] == owner else m["winner"]
            faced.append((depth.get(m["round"], 0), opp))
    faced.sort(key=lambda t: -t[0])
    return [opp for _, opp in faced]


def _matchups(owner: str, by_owner: dict[str, Bot], results: dict | None) -> list[tuple[Bot, Bot]]:
    me = by_owner[owner]
    opp_owners: list[str] = []
    if results:
        opp_owners = [o for o in _opponents_from_results(owner, results) if o in by_owner]
    if not opp_owners:  # fallback: next strongest present bot
        others = sorted((b for o, b in by_owner.items() if o != owner),
                        key=lambda b: -b.timesteps)
        if not others:
            raise SystemExit(f"No opponent available for {owner}.")
        opp_owners = [others[0].owner]
    return [(me, by_owner[o]) for o in opp_owners]


def _ffmpeg_cmd(out_path: Path, seconds: int, window_title: str) -> list[str]:
    return [
        "ffmpeg", "-y", "-f", "gdigrab", "-framerate", "60",
        "-i", f"title={window_title}", "-t", str(seconds),
        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(out_path),
    ]


def record_owner(owner: str, by_owner: dict[str, Bot], results: dict | None, *,
                 seconds: int, deterministic: bool, capture: bool, window_title: str) -> None:
    if owner not in by_owner:
        print(f"[skip] {owner}: no checkpoint present.")
        return

    matchups = _matchups(owner, by_owner, results)
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    out_path = VIDEO_DIR / f"{owner}_bracket.mp4"

    print("\n" + "#" * 60)
    print(f"#  RECORDING {owner.upper()}  ->  {out_path.name}")
    print(f"#  matchup(s): {', '.join(f'{a.owner} vs {b.owner}' for a, b in matchups)}")
    print("#  Make sure rlviser.exe is open and visible.")
    print("#" * 60)

    proc: subprocess.Popen | None = None
    if capture:
        if shutil.which("ffmpeg"):
            proc = subprocess.Popen(_ffmpeg_cmd(out_path, seconds + 3, window_title))
            time.sleep(1.0)  # let ffmpeg attach to the window
        else:
            print("\n[ffmpeg not found] Capture this window manually (e.g. OBS), or run:")
            print("  " + " ".join(_ffmpeg_cmd(out_path, seconds + 3, window_title)) + "\n")

    start = time.time()
    try:
        # Loop the matchup(s) until we have enough footage. Short matches (a goal
        # ends an episode) are replayed as needed to fill `seconds`.
        while time.time() - start < seconds:
            for a, b in matchups:
                if time.time() - start >= seconds:
                    break
                play_match(a, b, games=5, deterministic=deterministic, render=True)
    except KeyboardInterrupt:
        print("Interrupted — stopping recording.")
    finally:
        if proc is not None:
            proc.wait(timeout=10)
            print(f"Saved {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Render bracket footage for the presentation.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--owner", help="Record this single owner (diego/martin/marian/nachi/marco).")
    g.add_argument("--all", action="store_true", help="Record every present bot, one file each.")
    p.add_argument("--results", type=Path, default=DEFAULT_RESULTS,
                   help="Tournament results JSON to pick matchups from (optional).")
    p.add_argument("--seconds", type=int, default=60, help="Target footage length per bot.")
    p.add_argument("--stochastic", action="store_true", help="Sample actions instead of argmax.")
    p.add_argument("--capture", action="store_true",
                   help="Auto screen-capture the rlviser window with ffmpeg -> videos/<owner>.mp4")
    p.add_argument("--window-title", default="RLViser",
                   help="rlviser window title for ffmpeg gdigrab (default: RLViser).")
    args = p.parse_args()

    bots = build_roster()
    by_owner = {b.owner: b for b in bots}
    if not by_owner:
        raise SystemExit("No checkpoints present — nothing to record. Run the download script.")

    results = None
    if args.results.is_file():
        results = json.loads(args.results.read_text())
        print(f"Using matchups from {args.results}")
    else:
        print(f"No results JSON at {args.results} — falling back to next-strongest opponent.")

    owners = list(by_owner) if args.all else [args.owner]
    for owner in owners:
        record_owner(owner, by_owner, results, seconds=args.seconds,
                     deterministic=not args.stochastic, capture=args.capture,
                     window_title=args.window_title)


if __name__ == "__main__":
    main()
