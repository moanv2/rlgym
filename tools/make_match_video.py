"""Play ONE 1v1 match between two bots and render it to a top-down MP4 (self-contained).

Handles mixed obs builders (cross-obs), forces CPU, deterministic by default. Uses imageio's
bundled ffmpeg (no system ffmpeg / PATH needed); falls back to an animated GIF if mp4 fails.

  python tools/make_match_video.py \
    --a-policy <path/to/checkpoint_dir> --a-obs advanced --a-dim 107 --a-name botA \
    --b-policy <path/to/checkpoint_dir> --b-obs default --b-dim 89 --b-name botB \
    --out match.mp4
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.cuda.is_available = lambda: False

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402

from rlbot.actions.lookup_action import LookupAction  # noqa: E402
from rlbot.env import make_env_builder  # noqa: E402
from rlbot.evaluation.evaluate import _load_policy, _resolve_checkpoint  # noqa: E402
from rlbot.utils.config import load_config  # noqa: E402

SIDE = 4096.0  # SIDE_WALL_X
BACK = 5120.0  # BACK_WALL_Y
GOAL_HALF = 893.0  # half goal width


def make_obs(name):
    from rlgym_sim.utils.obs_builders import AdvancedObs, DefaultObs

    return AdvancedObs() if name == "advanced" else DefaultObs()


def _heading(car_data):
    """2D forward direction (x, y) of a car, robust to the rlgym_sim API."""
    try:
        f = car_data.forward()
        return float(f[0]), float(f[1])
    except Exception:
        try:
            y = car_data.yaw()
            return float(np.cos(y)), float(np.sin(y))
        except Exception:
            return 1.0, 0.0


def play_and_log(a):
    full = load_config(a.config).to_dict()
    full["state_setter"] = {"name": "default"}  # a real kickoff -> watchable point
    full["terminal"]["timeout_seconds"] = int(a.max_seconds)
    env_cfg = dict(full["env"])
    env_cfg["team_size"] = 1
    env_cfg["spawn_opponents"] = True
    env = make_env_builder(env_cfg, full)()
    n_actions = int(env.action_space.n)
    lut = LookupAction().make_lookup_table()
    pol_a = _load_policy(_resolve_checkpoint(a.a_policy), a.a_dim, n_actions, "cpu")
    pol_b = _load_policy(_resolve_checkpoint(a.b_policy), a.b_dim, n_actions, "cpu")
    obs_a, obs_b = make_obs(a.a_obs), make_obs(a.b_obs)
    if a.seed is not None:
        np.random.seed(a.seed)
        torch.manual_seed(a.seed)

    _, info = env.reset(return_info=True)
    state = info["state"]
    obs_a.reset(state)
    obs_b.reset(state)
    prev = {0: np.zeros(8, dtype=np.float32), 1: np.zeros(8, dtype=np.float32)}
    frames = []
    done, result = False, 0.0
    while not done:
        acts = []
        for pl in state.players:
            if pl.team_num == 0:
                ob = obs_a.build_obs(pl, state, prev[0])
                pol = pol_a
            else:
                ob = obs_b.build_obs(pl, state, prev[1])
                pol = pol_b
            with torch.no_grad():
                idx = int(pol.get_action(np.asarray(ob, dtype=np.float32), deterministic=not a.stochastic)[0])
            acts.append([idx])
            prev[pl.team_num] = lut[idx]
        _, _, done, info = env.step(np.array(acts))
        state = info["state"]
        result = info["result"]
        cars = []
        for pl in state.players:
            p = pl.car_data.position
            hx, hy = _heading(pl.car_data)
            cars.append((float(p[0]), float(p[1]), hx, hy, int(pl.team_num)))
        frames.append((tuple(float(x) for x in state.ball.position[:2]), cars))
    return frames, result


def _draw_field(ax):
    ax.set_xlim(-SIDE - 200, SIDE + 200)
    ax.set_ylim(-BACK - 300, BACK + 300)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(Rectangle((-SIDE, -BACK), 2 * SIDE, 2 * BACK, fill=True, color="#3a7d44", zorder=0))
    ax.add_patch(Rectangle((-SIDE, -BACK), 2 * SIDE, 2 * BACK, fill=False, ec="white", lw=2, zorder=1))
    ax.plot([-SIDE, SIDE], [0, 0], color="white", lw=1.2, zorder=1)
    ax.add_patch(Circle((0, 0), 900, fill=False, ec="white", lw=1.2, zorder=1))
    # goals (blue defends -BACK, orange defends +BACK)
    ax.add_patch(Rectangle((-GOAL_HALF, -BACK - 120), 2 * GOAL_HALF, 120, color="#4aa3ff", zorder=1))
    ax.add_patch(Rectangle((-GOAL_HALF, BACK), 2 * GOAL_HALF, 120, color="#ff8c42", zorder=1))


def render(frames, a):
    fig, ax = plt.subplots(figsize=(6.0, 7.4))
    fig.subplots_adjust(left=0, right=1, top=0.95, bottom=0)
    colors = {0: "#1f6fff", 1: "#ff7a1a"}
    title = f"{a.a_name} (blue)  vs  {a.b_name} (orange)"
    frames_rgb = []
    step = max(1, a.frame_skip)
    for fi in range(0, len(frames), step):
        ball_xy, cars = frames[fi]
        ax.clear()
        _draw_field(ax)
        ax.set_title(title, color="white", fontsize=12, pad=6)
        ax.add_patch(Circle(ball_xy, 120, color="white", ec="black", zorder=4))
        for cx, cy, hx, hy, team in cars:
            n = (hx * hx + hy * hy) ** 0.5 or 1.0
            ax.arrow(
                cx,
                cy,
                hx / n * 380,
                hy / n * 380,
                width=120,
                head_width=300,
                length_includes_head=True,
                color=colors[team],
                ec="black",
                zorder=5,
            )
        fig.patch.set_facecolor("#11141a")
        fig.canvas.draw()
        frames_rgb.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)
    return frames_rgb


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a-policy", required=True)
    p.add_argument("--a-obs", default="advanced")
    p.add_argument("--a-dim", type=int, default=107)
    p.add_argument("--a-name", default="A")
    p.add_argument("--b-policy", required=True)
    p.add_argument("--b-obs", default="advanced")
    p.add_argument("--b-dim", type=int, default=107)
    p.add_argument("--b-name", default="B")
    p.add_argument("--config", default="configs/experiments/exp_003_long_run.yaml")
    p.add_argument("--out", default="deliverables/match.mp4")
    p.add_argument("--max-seconds", type=int, default=30)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--frame-skip", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stochastic", action="store_true")
    a = p.parse_args()

    frames, result = play_and_log(a)
    winner = a.a_name if result > 0 else (a.b_name if result < 0 else "draw")
    print(f"match: {len(frames)} steps, result={result} ({winner})", flush=True)
    rgb = render(frames, a)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    try:
        import imageio.v2 as imageio

        with imageio.get_writer(a.out, fps=a.fps, macro_block_size=None) as w:
            for fr in rgb:
                w.append_data(fr)
        print(f"wrote {a.out}  ({len(rgb)} frames @ {a.fps}fps, winner={winner})", flush=True)
    except Exception as e:
        gif = os.path.splitext(a.out)[0] + ".gif"
        import imageio.v2 as imageio

        imageio.mimsave(gif, rgb, fps=a.fps)
        print(f"mp4 failed ({e}); wrote GIF fallback {gif}", flush=True)


if __name__ == "__main__":
    main()
