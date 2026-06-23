"""Auto-clip the coolest goals out of the synced rlviser capture into a high-fps reel.

Reads the capture start time (_capture_t0.txt) and the goal log (_goals.jsonl, each goal has a
wall-clock time + coolness), computes each goal's position in the recording, clips the top ones
with a caption, and concats them into a reel (high + efficient tiers). GPU-encoded (NVENC).
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
LEAD, TRAIL = 6.5, 2.5   # TRAIL covers the post-goal follow-through (sim keeps playing ~2.5s)
TOP = int(sys.argv[1]) if len(sys.argv) > 1 else 5
# preserve the source aspect (no stretch/zoom): fit inside 1920x1080, letterbox the rest
SCALE = "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=black"


def label(g):
    sc = g.get("scorer", "MARTIN 10B")
    cz = g.get("champ_z", 0)
    if g.get("goal"):
        if g.get("aerial"):
            return f"{sc}  AERIAL GOAL  {cz}uu"
        return f"{sc}  GOAL  {g.get('ball_spd', 0)} speed"
    return f"MARTIN 10B  AERIAL  {cz}uu"


def vdur(path):
    import imageio.v2 as imageio
    return imageio.get_reader(str(path)).get_meta_data().get("duration", 0)


def main():
    t0 = float(Path(OUT / "_capture_t0.txt").read_text().strip().replace(",", "."))
    dur = vdur(RAW)
    goals = [json.loads(l) for l in (OUT / "_goals.jsonl").read_text().splitlines() if l.strip()]
    for g in goals:
        g["vt"] = g["t"] - t0           # position of the goal within the recording
    usable = [g for g in goals if LEAD <= g["vt"] <= dur - TRAIL]
    # GOALS first (Martin wants top scores). Rank by coolness but gently penalise camera-whip
    # (shake = yaw-rate reversals of the followed car) so the smoother cool goals float to the top.
    def rank(g):
        return g["score"] - 0.02 * min(g.get("shake", 0), 25)
    real_goals = sorted([g for g in usable if g.get("goal")], key=lambda g: -rank(g))
    aerials = sorted([g for g in usable if not g.get("goal")], key=lambda g: -rank(g))
    top = (real_goals + aerials)[:TOP]
    print(f"recording {dur:.0f}s, {len(goals)} highlights logged, "
          f"{len(real_goals)} goals + {len(aerials)} aerials inside the recording, "
          f"clipping top {len(top)} (goals first)")
    if not top:
        print("no usable highlights (sync window). vt:", [round(g['vt'], 1) for g in goals])
        return

    clips = []
    for i, g in enumerate(top, 1):
        start = max(0, g["vt"] - LEAD)
        cap = label(g).replace("'", "")
        clip = OUT / f"goal_{i:02d}.mp4"
        vf = (f"{SCALE},drawtext=text='{cap}':fontcolor=white:fontsize=40:box=1:"
              f"boxcolor=black@0.5:boxborderw=14:x=(w-text_w)/2:y=h-100:enable='lte(t,3.5)',"
              f"drawtext=text='MARTIN 10B':fontcolor=0xF5C451:fontsize=30:x=46:y=46")
        subprocess.run([FF, "-y", "-ss", f"{start:.2f}", "-i", str(RAW), "-t", f"{LEAD+TRAIL:.2f}",
                        "-vf", vf, "-c:v", "h264_nvenc", "-preset", "p4", "-cq", "21",
                        "-pix_fmt", "yuv420p", "-an", str(clip)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        if clip.exists():
            clips.append(clip)
            print(f"  goal {i}: {cap}  (@{g['vt']:.0f}s)")

    if not clips:
        print("clip extraction failed")
        return
    title = OUT / "_title.mp4"
    subprocess.run([FF, "-y", "-f", "lavfi", "-i", "color=c=0x060a14:s=1920x1080:d=2.2:r=60",
                    "-vf", ("drawtext=text='MARTIN 10B CHAMPION':fontcolor=0xF5C451:fontsize=72:"
                            "x=(w-text_w)/2:y=(h-text_h)/2-40,drawtext=text='top goals vs Nachi 2.9B':"
                            "fontcolor=0x3FA9FF:fontsize=38:x=(w-text_w)/2:y=(h-text_h)/2+60"),
                    "-c:v", "h264_nvenc", "-preset", "p4", "-cq", "21", "-pix_fmt", "yuv420p", str(title)],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    lst = OUT / "_concat.txt"
    lst.write_text("file '_title.mp4'\n" + "\n".join(f"file '{c.name}'" for c in clips) + "\n")
    hi = OUT / "RLVISER_HIGHLIGHTS_HIGH.mp4"
    subprocess.run([FF, "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
                    "-c:v", "h264_nvenc", "-preset", "p2", "-cq", "20", "-pix_fmt", "yuv420p", str(hi)],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    lo = OUT / "RLVISER_HIGHLIGHTS_EFFICIENT.mp4"
    subprocess.run([FF, "-y", "-i", str(hi), "-vf", "scale=1280:720", "-c:v", "libx264",
                    "-crf", "26", "-pix_fmt", "yuv420p", str(lo)],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    print(f"\nDONE -> {hi}  ({hi.stat().st_size/1e6:.0f} MB)" if hi.exists() else "reel concat failed")


if __name__ == "__main__":
    main()
