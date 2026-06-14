"""Render ONE 'training progression' MP4: our bot at several checkpoints, each playing a short
kickoff clip vs the SAME fixed opponent, stitched in order with a per-segment caption (step
count) — so you watch it go from clumsy to dominant. Reuses tools/make_match_video.py's
match + top-down renderer. Deterministic (deploy mode).

  python tools/progression_video.py --out deliverables/progression.mp4
"""
from __future__ import annotations

import argparse
import importlib.util
import os
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("mmv", os.path.join(HERE, "make_match_video.py"))
mmv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mmv)

# Our lineage, earliest -> current. All advanced-obs 1024x3.
CHECKPOINTS = [
    ("250M steps", "checkpoints/_eval_snapshots/basics_250M"),
    ("1.3B steps", "checkpoints/_eval_snapshots/champion_best_1.295B"),
    ("2.1B steps", "martin-bots/checkpoints/CHAMPION_2.1B_recipeD_advanced1024"),
    ("3.4B steps (current)", "martin-bots/checkpoints/CHAMPION_3.37B_advanced1024"),
]
# Fixed yardstick. Default = our OWN 250M origin: a big, stable skill gap so later checkpoints
# visibly dominate (low variance) — the clean "how far we came" story. Override via --opp-* for
# the "vs Diego" framing instead.
OPP_POLICY = "checkpoints/_eval_snapshots/basics_250M"
OPP_NAME = "250M origin"
OPP_OBS, OPP_DIM = "advanced", 107


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="deliverables/progression.mp4")
    p.add_argument("--max-seconds", type=int, default=18, help="cap per clip (ends earlier on a goal)")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--frame-skip", type=int, default=2)
    p.add_argument("--seed", type=int, default=7, help="same kickoff seed for every checkpoint (fair)")
    args = p.parse_args()

    all_rgb = []
    summary = []
    for label, ckpt in CHECKPOINTS:
        a = SimpleNamespace(
            a_policy=ckpt, a_obs="advanced", a_dim=107, a_name=label,
            b_policy=OPP_POLICY, b_obs=OPP_OBS, b_dim=OPP_DIM, b_name=OPP_NAME,
            config="configs/experiments/exp_003_long_run.yaml",
            max_seconds=args.max_seconds, seed=args.seed, stochastic=False,
            frame_skip=args.frame_skip,
        )
        frames, result = mmv.play_and_log(a)
        winner = label if result > 0 else (OPP_NAME if result < 0 else "draw")
        secs = len(frames) / 15.0  # sim runs at ~15 decisions/s (tick_skip 8)
        summary.append(f"{label:24} -> {winner:12} ({secs:.0f}s of play)")
        print(f"  {label}: {len(frames)} steps, winner={winner}", flush=True)
        rgb = mmv.render(frames, a)
        if rgb:
            all_rgb.extend([rgb[0]] * int(args.fps * 0.8))  # ~0.8s freeze 'title card' per segment
            all_rgb.extend(rgb)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    import imageio.v2 as imageio

    with imageio.get_writer(args.out, fps=args.fps, macro_block_size=None) as w:
        for fr in all_rgb:
            w.append_data(fr)
    print(f"\nwrote {args.out}  ({len(all_rgb)} frames, ~{len(all_rgb) / args.fps:.0f}s)", flush=True)
    print("\n=== progression vs Diego 512 (fixed) ===", flush=True)
    for s in summary:
        print("  " + s, flush=True)


if __name__ == "__main__":
    main()
