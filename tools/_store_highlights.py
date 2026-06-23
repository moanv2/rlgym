"""Clip EVERY goal (and notable aerial) from a capture into a persistent review library.

Unlike _clip_reel.py (which builds the top-N reel), this saves ALL good highlights as individual,
descriptively-named files into Downloads\MARTIN_10B_HIGHLIGHTS_REVIEW so Martin can review and pick.
Accumulates across capture runs (each run gets a unique tag so nothing is overwritten).

Usage (from repo root):  python tools/_store_highlights.py <run_tag>
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg

FF = imageio_ffmpeg.get_ffmpeg_exe()
OUT = Path("highlights_3d")
RAW = OUT / "_raw_session.mp4"
REVIEW = Path(r"C:\Users\Lasca\Downloads\MARTIN_10B_HIGHLIGHTS_REVIEW")
LEAD, TRAIL = 6.5, 2.5
SCALE = "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=black"
RUN_TAG = sys.argv[1] if len(sys.argv) > 1 else "run"


def label(g):
    sc = g.get("scorer", "MARTIN 10B")
    cz = g.get("champ_z", 0)
    if g.get("goal"):
        if g.get("aerial"):
            return f"{sc}  AERIAL GOAL  {cz}uu"
        return f"{sc}  GOAL  {g.get('ball_spd', 0)} speed"
    return f"MARTIN 10B  AERIAL  {cz}uu"


def fname(i, g):
    sc = g.get("scorer", "MARTIN 10B").replace(" ", "")
    kind = ("AERIALGOAL" if g.get("aerial") else "GOAL") if g.get("goal") else "AERIAL"
    return f"{i:02d}_{sc}_{kind}_{g.get('champ_z', 0)}uu_{g.get('ball_spd', 0)}spd_sc{g.get('score', 0)}_{RUN_TAG}.mp4"


def vdur(path):
    import imageio.v2 as imageio
    return imageio.get_reader(str(path)).get_meta_data().get("duration", 0)


def main():
    REVIEW.mkdir(parents=True, exist_ok=True)
    t0 = float((OUT / "_capture_t0.txt").read_text().strip().replace(",", "."))
    dur = vdur(RAW)
    goals = [json.loads(l) for l in (OUT / "_goals.jsonl").read_text().splitlines() if l.strip()]
    for g in goals:
        g["vt"] = g["t"] - t0
    usable = [g for g in goals if LEAD <= g["vt"] <= dur - TRAIL]
    # goals first (best score), then notable aerials; gently de-prioritise camera-whip
    def rank(g):
        return g["score"] - 0.02 * min(g.get("shake", 0), 25)
    real_goals = sorted([g for g in usable if g.get("goal")], key=lambda g: -rank(g))
    aerials = sorted([g for g in usable if not g.get("goal")], key=lambda g: -rank(g))
    ordered = real_goals + aerials
    print(f"recording {dur:.0f}s, {len(real_goals)} goals + {len(aerials)} aerials usable -> storing all "
          f"in {REVIEW}")

    stored = []
    for i, g in enumerate(ordered, 1):
        start = max(0, g["vt"] - LEAD)
        cap = label(g).replace("'", "")
        out = REVIEW / fname(i, g)
        vf = (f"{SCALE},drawtext=text='{cap}':fontcolor=white:fontsize=40:box=1:"
              f"boxcolor=black@0.5:boxborderw=14:x=(w-text_w)/2:y=h-100:enable='lte(t,3.5)',"
              f"drawtext=text='MARTIN 10B':fontcolor=0xF5C451:fontsize=30:x=46:y=46")
        subprocess.run([FF, "-y", "-ss", f"{start:.2f}", "-i", str(RAW), "-t", f"{LEAD + TRAIL:.2f}",
                        "-vf", vf, "-c:v", "h264_nvenc", "-preset", "p4", "-cq", "21",
                        "-pix_fmt", "yuv420p", "-an", str(out)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        if out.exists():
            stored.append(out.name)
            print(f"  stored {out.name}  (@{g['vt']:.0f}s in raw)")
    print(f"\nDONE -> {len(stored)} clips in {REVIEW}")


if __name__ == "__main__":
    main()
