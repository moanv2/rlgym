"""Cinematic 3D highlight renderer (no rlviser, full camera control).

Mines the coolest champion goals (vs Nachi) headless, then renders each with a perspective
3D camera: a high, wide, smoothly-tracking broadcast view (not a dizzy ball-chase), slowed
playback, ground shadows, ball trail, slow-mo on the finish. Outputs clips + a reel in two
quality tiers. Fully deterministic to produce (renders from saved trajectories, no capture).

Run from repo root, PYTHONPATH=src:
  python tools/_cinematic3d.py --games 60 --top 6
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "tools")

SIDE, BACK, CEIL = 4096.0, 5120.0, 2044.0
GOAL_HALF, GOAL_H = 893.0, 642.0
BALL_R = 92.75
OUT = Path("highlights_3d")

# ---------- camera / projection ----------
FOV = math.radians(52)


def look_at(cam, tgt, up=(0, 0, 1)):
    f = np.array(tgt) - np.array(cam)
    f = f / (np.linalg.norm(f) + 1e-9)
    up = np.array(up, float)
    r = np.cross(f, up)
    r = r / (np.linalg.norm(r) + 1e-9)
    u = np.cross(r, f)
    return np.array(cam, float), r, u, f


def project(P, cam, r, u, f, W, H):
    rel = np.asarray(P, float) - cam
    z = rel @ f
    if z <= 1.0:
        return None
    x = rel @ r
    y = rel @ u
    foc = (H / 2) / math.tan(FOV / 2)
    return np.array([W / 2 + foc * x / z, H / 2 - foc * y / z]), z


# ---------- mining ----------
def snap(state):
    b = state.ball
    cars = []
    for p in state.players:
        cd = p.car_data
        try:
            fwd = cd.forward(); fx, fy = float(fwd[0]), float(fwd[1])
        except Exception:
            fx, fy = 1.0, 0.0
        cars.append((float(cd.position[0]), float(cd.position[1]), float(cd.position[2]),
                     fx, fy, int(p.team_num), bool(p.ball_touched)))
    return (float(b.position[0]), float(b.position[1]), float(b.position[2]),
            float(b.linear_velocity[0]), float(b.linear_velocity[1]), float(b.linear_velocity[2]), cars)


def coolness(traj):
    tail = traj[-14:]
    spd = max((math.sqrt(s[3] ** 2 + s[4] ** 2 + s[5] ** 2) for s in tail), default=0.0)
    air = max((s[2] for s in traj[-45:]), default=0.0)
    dist = 0.0
    for s in reversed(traj):
        ch = [c for c in s[6] if c[6] and c[5] == 0]
        if ch:
            dist = math.hypot(ch[0][0], ch[0][1] - BACK); break
    return 0.45 * (spd / 6000) + 0.30 * (air / 2000) + 0.25 * (dist / 7000), spd, air, dist


def mine(games):
    import torch
    from papaya_1v1_viewer import DEFAULT_BLUE, _action_to_int, build_env, load_policy, resolve_checkpoint
    blue, _ = load_policy(resolve_checkpoint(DEFAULT_BLUE))
    orange, _ = load_policy(resolve_checkpoint(r"C:\Users\Lasca\rlgym_tourney_wt\teammates\nachi"))
    env = build_env()
    goals = []
    for g in range(games):
        obs = env.reset(); traj = []; done = False; info = {}
        while not done:
            with torch.no_grad():
                ba, _ = blue.get_action(obs[0], deterministic=False)
                oa, _ = orange.get_action(obs[1], deterministic=False)
            obs, _r, done, info = env.step([_action_to_int(ba), _action_to_int(oa)])
            traj.append(snap(env._prev_state))
        if int(info.get("result", 0)) > 0 and len(traj) > 10:
            sc, spd, air, dist = coolness(traj)
            goals.append({"score": sc, "spd": spd, "air": air, "dist": dist, "traj": traj})
            print(f"  game {g+1}/{games}: GOAL spd={spd:4.0f} air={air:4.0f} dist={dist:4.0f}")
        else:
            print(f"  game {g+1}/{games}: -")
    env.close()
    goals.sort(key=lambda d: -d["score"])
    return goals


# ---------- interpolation (slow + smooth) ----------
def smooth_traj(traj, sub):
    out = []
    for i in range(len(traj) - 1):
        a, b = traj[i], traj[i + 1]
        for k in range(sub):
            t = k / sub
            ball = tuple(a[j] * (1 - t) + b[j] * t for j in range(6))
            cars = [(ca[0] * (1 - t) + cb[0] * t, ca[1] * (1 - t) + cb[1] * t, ca[2] * (1 - t) + cb[2] * t,
                     ca[3], ca[4], ca[5], ca[6]) for ca, cb in zip(a[6], b[6])]
            out.append(ball + (cars,))
    out.append(traj[-1])
    return out


def car_box(x, y, z, fx, fy):
    ang = math.atan2(fy, fx); c, s = math.cos(ang), math.sin(ang)
    L, W, H = 230, 150, 85
    corners = []
    for sx in (-L / 2, L / 2):
        for sy in (-W / 2, W / 2):
            for sz in (0, H):
                corners.append((x + sx * c - sy * s, y + sx * s + sy * c, max(17, z) - 17 + sz))
    return corners  # 8 corners


def render_goal(goal, idx, reel_writer, fps, lead_label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Circle, Ellipse
    import imageio.v2 as imageio

    base = goal["traj"]
    base = base[-min(len(base), 230):]                 # last ~15s of sim
    frames = smooth_traj(base, sub=3)                  # 3x = smooth + slower
    frames = frames + [frames[-1]] * int(fps * 1.6)    # linger / slow-mo on goal

    W, H = 1920, 1080
    FOC = (H / 2) / math.tan(FOV / 2)
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    fig.patch.set_facecolor("#070b16")
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_facecolor("#070b16")
    clip = OUT / f"cine_{idx:02d}.mp4"
    cw = imageio.get_writer(clip, fps=fps, codec="libx264", quality=9,
                            macro_block_size=None, ffmpeg_log_level="error")
    bsx = bsy = None
    trail = []
    n = len(frames)
    for fi, s in enumerate(frames):
        ax.clear(); ax.set_facecolor("#070b16"); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
        ax.add_patch(plt.Rectangle((0, 0), W, H, facecolor="#070b16", zorder=-10))
        bx, by, bz = s[0], s[1], s[2]
        bsx = bx if bsx is None else bsx * 0.90 + bx * 0.10      # smoothed target
        bsy = by if bsy is None else bsy * 0.90 + by * 0.10
        # side broadcast cam: fixed on the +x touchline, elevated, gently pans along y.
        # Always sees the whole pitch, aerials read as height, never dizzy.
        cam = (SIDE + 3200, bsy * 0.30, 1750)
        tgt = (-200, bsy * 0.52, 360)
        C, r, u, f = look_at(cam, tgt)

        def pr(P):
            return project(P, C, r, u, f, W, H)

        # pitch
        corners = [pr((-SIDE, -BACK, 0)), pr((SIDE, -BACK, 0)), pr((SIDE, BACK, 0)), pr((-SIDE, BACK, 0))]
        if all(corners):
            ax.add_patch(Polygon([c[0] for c in corners], closed=True, facecolor="#0b1426",
                                 edgecolor="#21304d", lw=2, zorder=0))
        # field lines
        for seg in [((-SIDE, 0, 0), (SIDE, 0, 0))]:
            p0, p1 = pr(seg[0]), pr(seg[1])
            if p0 and p1:
                ax.plot([p0[0][0], p1[0][0]], [p0[0][1], p1[0][1]], color="#22324f", lw=1.4, zorder=1)
        circ = [pr((900 * math.cos(a), 900 * math.sin(a), 0)) for a in np.linspace(0, 2 * math.pi, 40)]
        circ = [c for c in circ if c]
        if len(circ) > 5:
            ax.plot([c[0][0] for c in circ], [c[0][1] for c in circ], color="#22324f", lw=1.2, zorder=1)
        # goals
        for gy, col in ((BACK, "#F5C451"), (-BACK, "#3FA9FF")):
            gp = [pr((-GOAL_HALF, gy, 0)), pr((-GOAL_HALF, gy, GOAL_H)), pr((GOAL_HALF, gy, GOAL_H)), pr((GOAL_HALF, gy, 0))]
            if all(gp):
                ax.add_patch(Polygon([p[0] for p in gp], closed=True, fill=False, edgecolor=col, lw=2.5, zorder=2))

        # collect drawables with depth
        draw = []
        # ball shadow + ball
        sh = pr((bx, by, 0))
        if sh:
            draw.append((sh[1] + 1e6, "shadow", sh[0], FOC * 100 / sh[1]))
        pb = pr((bx, by, bz + BALL_R))
        if pb:
            rad = max(6, FOC * BALL_R * 1.35 / pb[1])
            draw.append((pb[1], "ball", pb[0], rad))
            trail.append((bx, by, bz))
        # cars
        for cc in s[6]:
            col = "#F5C451" if cc[5] == 0 else "#3FA9FF"
            box = car_box(cc[0], cc[1], cc[2], cc[3], cc[4])
            proj = [pr(p) for p in box]
            if all(proj):
                zc = np.mean([p[1] for p in proj])
                draw.append((zc, "car", proj, col))
            csh = pr((cc[0], cc[1], 0))
            if csh:
                draw.append((csh[1] + 1e6, "shadow", csh[0], FOC * 120 / csh[1]))
        # trail (under everything moving)
        for j, (tx, ty, tz) in enumerate(trail[-26:]):
            tp = pr((tx, ty, tz + BALL_R))
            if tp:
                a = (j + 1) / 26
                ax.plot(tp[0][0], tp[0][1], "o", ms=2 + 7 * a, color="#9fe8ff",
                        alpha=0.04 + 0.18 * a, zorder=3)
        # paint far->near
        for item in sorted(draw, key=lambda d: -d[0]):
            kind = item[1]
            if kind == "shadow":
                ax.add_patch(Ellipse(item[2], item[3] * 2.2, max(2, item[3]) * 0.7, facecolor="black", alpha=0.20, zorder=2))
            elif kind == "ball":
                ax.add_patch(Circle(item[2], item[3], facecolor="white", edgecolor="#9fe8ff", lw=1.4, zorder=6))
            elif kind == "car":
                proj, col = item[2], item[3]
                top = [proj[i][0] for i in (1, 3, 7, 5)]
                ax.add_patch(Polygon(top, closed=True, facecolor=col, edgecolor="white", lw=0.5, alpha=0.96, zorder=5))
                side = [proj[i][0] for i in (0, 1, 5, 4)]
                ax.add_patch(Polygon(side, closed=True, facecolor=col, edgecolor="none", alpha=0.7, zorder=5))

        # goal flash + caption on the tail
        if fi > n - int(fps * 1.6):
            k = (fi - (n - int(fps * 1.6))) / (fps * 1.6)
            ax.add_patch(plt.Rectangle((0, 0), W, H, facecolor="#F5C451", alpha=0.05 + 0.10 * k, zorder=8))
            ax.text(W / 2, H / 2, "GOAL", color="#F5C451", fontsize=78, fontweight="bold",
                    ha="center", va="center", family="monospace", zorder=9, alpha=0.92)
        ax.text(54, 54, "MARTIN 10B", color="#F5C451", fontsize=24, fontweight="bold", family="monospace", zorder=9)
        if fi < fps * 3:
            ax.text(54, H - 60, lead_label, color="white", fontsize=26, family="monospace",
                    bbox=dict(boxstyle="round", fc="#000000aa", ec="none"), zorder=9)

        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(H, W, 4)[..., :3]
        cw.append_data(buf)
        if reel_writer is not None:
            reel_writer.append_data(buf)
    cw.close(); plt.close(fig)
    return clip


def label(spd, air, dist):
    if air > 700:
        return f"AERIAL  {air:.0f}uu up"
    if dist > 4500:
        return f"LONG RANGE  {dist:.0f}uu"
    if spd > 2400:
        return f"ROCKET  {spd:.0f} speed"
    return f"GOAL  {spd:.0f} speed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=60)
    ap.add_argument("--top", type=int, default=6)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    print(f">>> mining {args.games} games for cool goals...")
    goals = mine(args.games)
    print(f">>> {len(goals)} goals, rendering top {min(args.top, len(goals))} cinematically")
    if not goals:
        return
    import imageio.v2 as imageio
    reel = imageio.get_writer(OUT / "cinematic_reel.mp4", fps=args.fps, codec="libx264",
                              quality=9, macro_block_size=None, ffmpeg_log_level="error")
    for i, g in enumerate(goals[:args.top], 1):
        lab = label(g["spd"], g["air"], g["dist"])
        print(f"  rendering {i}: {lab}")
        render_goal(g, i, reel, args.fps, lab)
    reel.close()
    # efficient tier
    import subprocess, imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ff, "-y", "-i", str(OUT / "cinematic_reel.mp4"), "-vf", "scale=1280:720",
                    "-c:v", "libx264", "-crf", "28", "-pix_fmt", "yuv420p", str(OUT / "cinematic_reel_EFFICIENT.mp4")],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    print(f"\nDONE -> {OUT/'cinematic_reel.mp4'}")


if __name__ == "__main__":
    main()
