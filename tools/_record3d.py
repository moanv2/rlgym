"""Auto-record cinematic 3D highlights from rlviser, fully hands-off after the camera is set.

Flow:
  1. start an ffmpeg ddagrab capture of the rlviser game region (1080p, high fps, high quality)
  2. play champion (10B) vs Nachi in rlviser (kickoff or --aerial), logging each champion goal
     with a wall-clock time + a coolness score (shot speed / aerial height / distance)
  3. when the session ends, auto-clip the coolest goals out of the capture (synced by wall-clock),
     add a title card and per-goal captions, concat into a reel
  4. export TWO tiers: high (1080p, source fps) and efficient (720p, 30fps, small file)
  optional: --music <file> to mux a soundtrack.

ONE manual step first: click the rlviser window, press Esc then 9 (Director) or Esc then 1.

Run from repo root, PYTHONPATH=src:
  python tools/_record3d.py --mode aerial --session 150 --top 6
  python tools/_record3d.py --mode kickoff --session 180 --top 6 --music song.mp3
"""
from __future__ import annotations

import argparse
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import imageio_ffmpeg

sys.path.insert(0, "tools")
from papaya_1v1_viewer import (  # noqa: E402
    AerialState, DEFAULT_BLUE, _action_to_int, build_env, load_policy, resolve_checkpoint,
)

FF = imageio_ffmpeg.get_ffmpeg_exe()
OUT = Path("highlights_3d")
RAW = OUT / "_raw_capture.mp4"
CROP = "1920:1080:64:32"          # game region inside the maximized rlviser window
BACK = 5120.0


def snap(state):
    b = state.ball
    cars = [(float(p.car_data.position[0]), float(p.car_data.position[1]), float(p.car_data.position[2]),
             int(p.team_num), bool(p.ball_touched)) for p in state.players]
    return (float(b.position[0]), float(b.position[1]), float(b.position[2]),
            float(b.linear_velocity[0]), float(b.linear_velocity[1]), float(b.linear_velocity[2]), cars)


def coolness(traj):
    tail = traj[-14:]
    spd = max((math.sqrt(s[3] ** 2 + s[4] ** 2 + s[5] ** 2) for s in tail), default=0.0)
    air = max((s[2] for s in traj[-45:]), default=0.0)
    dist = 0.0
    for s in reversed(traj):
        ch = [c for c in s[6] if c[4] and c[3] == 0]
        if ch:
            dist = math.hypot(ch[0][0], ch[0][1] - BACK)
            break
    return 0.45 * (spd / 6000) + 0.30 * (air / 2000) + 0.25 * (dist / 7000), spd, air, dist


def label(spd, air, dist):
    if air > 700:
        return f"AERIAL GOAL  -  {air:.0f}uu up, {spd:.0f} speed"
    if dist > 4500:
        return f"LONG RANGE  -  from {dist:.0f}uu, {spd:.0f} speed"
    if spd > 2400:
        return f"ROCKET  -  {spd:.0f} ball speed"
    return f"GOAL  -  {spd:.0f} speed"


def run(args):
    OUT.mkdir(exist_ok=True)
    blue, _ = load_policy(resolve_checkpoint(args.blue))
    orange, _ = load_policy(resolve_checkpoint(args.orange))
    env = build_env(AerialState() if args.mode == "aerial" else None)

    # 1) start the screen capture (cropped to the rlviser game region)
    cap_cmd = [FF, "-y", "-filter_complex",
               f"ddagrab=output_idx=0:framerate={args.fps},hwdownload,format=bgra,crop={CROP}",
               "-t", str(args.session + 6), "-c:v", "libx264", "-preset", "veryfast",
               "-crf", "16", "-pix_fmt", "yuv420p", str(RAW)]
    print(">>> starting capture...")
    cap = subprocess.Popen(cap_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(args.warmup)            # let ddagrab stabilize
    t0 = time.time()                   # video time 0 ~= now + (already captured `warmup`s)

    # 2) play + log goals
    goals = []
    print(f">>> playing {args.mode} games for {args.session}s (camera should be Director/follow)...")
    while time.time() - t0 < args.session:
        obs = env.reset()
        traj = []
        done = False
        info = {}
        while not done and time.time() - t0 < args.session:
            with torch.no_grad():
                ba, _ = blue.get_action(obs[0], deterministic=False)
                oa, _ = orange.get_action(obs[1], deterministic=False)
            obs, _r, done, info = env.step([_action_to_int(ba), _action_to_int(oa)])
            traj.append(snap(env._prev_state))
            env.render()
            time.sleep(args.step_delay)
        if int(info.get("result", 0)) > 0 and len(traj) > 8:
            vt = time.time() - t0 + args.warmup          # goal time within the captured video
            sc, spd, air, dist = coolness(traj)
            goals.append({"vt": vt, "score": sc, "spd": spd, "air": air, "dist": dist})
            print(f"  GOAL @video {vt:5.1f}s  spd={spd:4.0f} air={air:4.0f} dist={dist:4.0f} score={sc:.3f}")
    env.close()
    print(">>> waiting for capture to finalize...")
    cap.wait(timeout=30)

    if not goals:
        print("No champion goals captured. Try a longer --session.")
        return
    goals.sort(key=lambda d: -d["score"])
    top = goals[: args.top]

    # 3) clip each cool goal (goal sits at `lead`s into the clip), with a caption
    clips = []
    for i, g in enumerate(top, 1):
        start = max(0.0, g["vt"] - args.lead)
        dur = args.lead + args.trail
        cap_txt = label(g["spd"], g["air"], g["dist"]).replace("'", "")
        clip = OUT / f"clip_{i:02d}.mp4"
        draw = (f"drawtext=text='{cap_txt}':fontcolor=white:fontsize=34:box=1:boxcolor=black@0.45:"
                f"boxborderw=12:x=(w-text_w)/2:y=h-90:enable='lte(t,3)',"
                f"drawtext=text='MARTIN 10B':fontcolor=0xF5C451:fontsize=26:x=40:y=40")
        subprocess.run([FF, "-y", "-ss", f"{start:.2f}", "-i", str(RAW), "-t", f"{dur:.2f}",
                        "-vf", draw, "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                        "-pix_fmt", "yuv420p", "-an", str(clip)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        if clip.exists():
            clips.append(clip)
            print(f"  clip {i}: {cap_txt}")

    if not clips:
        print("clip extraction failed.")
        return

    # 4) title card + concat -> reel
    title = OUT / "_title.mp4"
    subprocess.run([FF, "-y", "-f", "lavfi", "-i", f"color=c=0x060a14:s=1920x1080:d=2.5:r={args.fps}",
                    "-vf", ("drawtext=text='MARTIN 10B CHAMPION':fontcolor=0xF5C451:fontsize=70:"
                            "x=(w-text_w)/2:y=(h-text_h)/2-40,"
                            "drawtext=text='top goals vs Nachi 2.9B':fontcolor=0x3FA9FF:fontsize=36:"
                            "x=(w-text_w)/2:y=(h-text_h)/2+60"),
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
                    str(title)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    listf = OUT / "_concat.txt"
    listf.write_text("file '_title.mp4'\n" + "\n".join(f"file '{c.name}'" for c in clips) + "\n")
    reel_hi = OUT / "highlights_3d_HIGH.mp4"
    cat = [FF, "-y", "-f", "concat", "-safe", "0", "-i", str(listf)]
    if args.music and Path(args.music).is_file():
        cat += ["-i", args.music, "-c:v", "libx264", "-preset", "veryfast", "-crf", "16",
                "-c:a", "aac", "-b:a", "192k", "-shortest", "-pix_fmt", "yuv420p", str(reel_hi)]
    else:
        cat += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p", str(reel_hi)]
    subprocess.run(cat, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)

    # 5) efficient tier
    reel_lo = OUT / "highlights_3d_EFFICIENT.mp4"
    subprocess.run([FF, "-y", "-i", str(reel_hi), "-vf", "scale=1280:720", "-r", "30",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "28", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "128k", str(reel_lo)],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)

    print("\nDONE.")
    for f in (reel_hi, reel_lo):
        if f.exists():
            print(f"  {f}  ({f.stat().st_size/1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["kickoff", "aerial"], default="aerial")
    ap.add_argument("--blue", default=DEFAULT_BLUE)
    ap.add_argument("--orange", default=r"C:\Users\Lasca\rlgym_tourney_wt\teammates\nachi")
    ap.add_argument("--session", type=int, default=150, help="seconds of live play to capture")
    ap.add_argument("--top", type=int, default=6)
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--lead", type=float, default=11.0, help="seconds of buildup before the goal")
    ap.add_argument("--trail", type=float, default=3.5, help="seconds after the goal")
    ap.add_argument("--warmup", type=float, default=2.0)
    ap.add_argument("--step-delay", type=float, default=0.008)
    ap.add_argument("--music", default=None)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
