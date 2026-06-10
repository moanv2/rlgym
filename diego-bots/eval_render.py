"""Watch two trained policies play 1v1 inside rlviser.

Thin wrapper over Marian's headless evaluate.py that adds a per-step
rlviser render call and a real-time throttle. Re-uses the loader/env
helpers from src/rlbot/evaluation/evaluate.py so we never drift from
Marian's checkpoint-loading logic.

Prereqs:
  - rlviser.exe is running in the background (double-click rlviser.exe in repo root)
  - rlviser_py is installed in the rlbot310 env (it is — see simple_bot_play.py)
  - You're on a branch that has both Diego's diego-bots/ and Marian's src/rlbot/
    (after the merge of marian/setup-fixes into diego, you are)

Run:
  conda activate rlbot310
  python diego-bots/eval_render.py \
      --blue   diego-bots/checkpoints/nexto_rewards/<run>/<timestep> \
      --orange checkpoints/exp_001_baseline/<timestep>
"""
from __future__ import annotations

import argparse
import time

import torch

# Re-use Marian's helpers so we don't drift from his loader logic. These are
# private (underscore) by convention but it's fine to depend on them inside
# the same repo — if their signature changes, we change with it.
from rlbot.evaluation.evaluate import (
    _action_to_int,
    _build_eval_env,
    _load_policy,
    _resolve_checkpoint_path,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--blue", required=True, help="Blue policy checkpoint folder (a numeric timestep dir).")
    p.add_argument(
        "--orange",
        required=True,
        help="Orange policy checkpoint folder, or 'latest:<experiment_name>' to auto-resolve.",
    )
    p.add_argument("--episodes", type=int, default=5, help="How many full games to play before exiting.")
    p.add_argument(
        "--deterministic",
        action="store_true",
        help="Greedy argmax instead of sampling — looks less twitchy on the visualizer.",
    )
    p.add_argument(
        "--step-delay",
        type=float,
        default=0.006,
        help="Seconds to sleep per env.step. 0.006 ≈ real time at tick_skip=8. "
        "Bump higher (e.g. 0.01) if the visualization runs too fast to follow.",
    )
    args = p.parse_args()

    blue_ckpt = _resolve_checkpoint_path(args.blue)
    orange_ckpt = _resolve_checkpoint_path(args.orange)
    print(f"BLUE   {blue_ckpt}")
    print(f"ORANGE {orange_ckpt}")

    # Load both policies on CPU — inference is cheap, GPU would just contend
    # with whatever else is running on the box.
    blue_policy = _load_policy(blue_ckpt, device="cpu")
    orange_policy = _load_policy(orange_ckpt, device="cpu")

    env = _build_eval_env()
    print("Make sure rlviser.exe is open. Press Ctrl+C to stop.")
    try:
        for ep in range(1, args.episodes + 1):
            obs_list = env.reset()
            blue_obs, orange_obs = obs_list[0], obs_list[1]
            done = False
            while not done:
                with torch.no_grad():
                    b_act, _ = blue_policy.get_action(blue_obs, deterministic=args.deterministic)
                    o_act, _ = orange_policy.get_action(orange_obs, deterministic=args.deterministic)
                obs_list, _r, done, info = env.step(
                    [_action_to_int(b_act), _action_to_int(o_act)]
                )
                blue_obs, orange_obs = obs_list[0], obs_list[1]

                env.render()  # push current state to rlviser
                time.sleep(args.step_delay)  # throttle to ~real time

            result = int(info.get("result", 0))
            outcome = "BLUE" if result > 0 else "ORANGE" if result < 0 else "DRAW"
            print(f"Ep {ep}/{args.episodes}  {outcome}  (delta={result:+d})")
    finally:
        env.close()


if __name__ == "__main__":
    main()
