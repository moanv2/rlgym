"""Mine the coolest goals (seed-reproducible), then replay ONLY those goals in rlviser (3D).

Two passes, same env/policies/seeding so a cool goal found headless replays identically in 3D:
  1) mine   : play N seeded games headless, score champion goals, write the best seeds to JSON.
  2) replay : re-run just those seeds with rlviser rendering (real time) -> a 3D highlight reel.
              record the rlviser window with Game Bar (Win+Alt+R) or pass --capture for ffmpeg.

Run from repo root, PYTHONPATH=src:
    python tools/_highlight_3d.py mine   --orange <opp> --games 60 --top 6
    python tools/_highlight_3d.py replay --orange <opp>            # rlviser opens, plays the 6 goals
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

# reuse the proven viewer (same env, same loaders -> reproducible across mine/replay)
from papaya_1v1_viewer import (  # noqa: E402  (tools/ is on sys.path when run from repo root)
    DEFAULT_BLUE,
    _action_to_int,
    build_env,
    load_policy,
    resolve_checkpoint,
)

BACK = 5120.0
SEEDS_FILE = Path("highlights_3d/cool_seeds.json")


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


def play_seeded(blue, orange, env, seed, *, render, step_delay):
    np.random.seed(seed)
    torch.manual_seed(seed)
    obs = env.reset()
    traj = []
    done = False
    info = {}
    while not done:
        with torch.no_grad():
            ba, _ = blue.get_action(obs[0], deterministic=False)
            oa, _ = orange.get_action(obs[1], deterministic=False)
        obs, _r, done, info = env.step([_action_to_int(ba), _action_to_int(oa)])
        traj.append(snap(env._prev_state))
        if render:
            env.render()
            time.sleep(step_delay)
    return int(info.get("result", 0)), traj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["mine", "replay", "verify"])
    ap.add_argument("--blue", default=DEFAULT_BLUE)
    ap.add_argument("--orange", required=True)
    ap.add_argument("--games", type=int, default=60)
    ap.add_argument("--top", type=int, default=6)
    ap.add_argument("--base-seed", type=int, default=1000)
    ap.add_argument("--step-delay", type=float, default=0.012)
    args = ap.parse_args()

    blue, _ = load_policy(resolve_checkpoint(args.blue))
    orange, _ = load_policy(resolve_checkpoint(args.orange))
    env = build_env()
    SEEDS_FILE.parent.mkdir(exist_ok=True)

    if args.mode == "mine":
        print(f">>> mining {args.games} seeded games...")
        found = []
        for i in range(args.games):
            seed = args.base_seed + i
            res, traj = play_seeded(blue, orange, env, seed, render=False, step_delay=0)
            if res > 0 and len(traj) > 10:
                sc, spd, air, dist = coolness(traj)
                found.append({"seed": seed, "score": round(sc, 4), "spd": round(spd), "air": round(air), "dist": round(dist)})
                print(f"  seed {seed}: GOAL  spd={spd:5.0f} air={air:5.0f} dist={dist:5.0f} score={sc:.3f}")
        found.sort(key=lambda d: -d["score"])
        top = found[: args.top]
        SEEDS_FILE.write_text(json.dumps(top, indent=2))
        print(f"\n{len(found)} champion goals. Wrote top {len(top)} seeds -> {SEEDS_FILE}")
        for d in top:
            print(f"  seed {d['seed']}  score {d['score']}  (spd {d['spd']}, air {d['air']}, dist {d['dist']})")

    elif args.mode == "verify":
        # reproducibility check: replay the top seed headless, confirm same coolness
        top = json.loads(SEEDS_FILE.read_text())
        d = top[0]
        res, traj = play_seeded(blue, orange, env, d["seed"], render=False, step_delay=0)
        sc, spd, air, dist = coolness(traj)
        ok = (res > 0) and abs(sc - d["score"]) < 1e-3
        print(f"seed {d['seed']}: stored score {d['score']}, replayed {sc:.4f} (spd {spd:.0f})  -> "
              f"{'REPRODUCIBLE' if ok else 'MISMATCH'}")

    else:  # replay (rlviser)
        top = json.loads(SEEDS_FILE.read_text())
        print(f">>> replaying {len(top)} cool goals in rlviser (real time). Record with Game Bar (Win+Alt+R).")
        print("    rlviser opens on the first goal. Press 9 in rlviser for the cinematic Director camera.")
        for n, d in enumerate(top, 1):
            print(f"  [{n}/{len(top)}] seed {d['seed']}  (spd {d['spd']}, air {d['air']}, dist {d['dist']})")
            play_seeded(blue, orange, env, d["seed"], render=True, step_delay=args.step_delay)
            time.sleep(0.6)
        print("done.")
    env.close()


if __name__ == "__main__":
    main()
