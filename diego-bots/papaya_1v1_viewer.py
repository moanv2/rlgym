"""Watch papaya 1v1 a teammate's bot (Martin / Nachi) inside rlviser.

This is the AdvancedObs head-to-head viewer. It is separate from
diego-bots/eval_render.py, which is wired to Marian's DefaultObs (89-dim)
loader and CANNOT load AdvancedObs (107-dim) bots.

Why a fresh loader:
  - Each policy's hidden-layer sizes are inferred from its OWN saved weights, so
    papaya (1024x3) can play a teammate of any width without you specifying arch.
  - The obs dimension the checkpoint expects is validated against the env
    (AdvancedObs = 107). If you point it at a DefaultObs bot (89-dim, e.g.
    Marian's current branch) it raises a clear error instead of feeding the
    policy a mismatched vector and silently playing nonsense.

Compatibility (verified): papaya's custom AdvancedObs and the rlgym_sim built-in
AdvancedObs that Martin/Nachi use are numerically identical (max diff ~1e-7), so
all three read the same 107-dim vector. Everyone uses LookupAction (90 discrete),
team_size=1, tick_skip=8.

Prereqs:
  - rlviser.exe is running (double-click it; an empty arena window opens).
  - rlviser_py + rlgym_sim installed in rlbot310 (they are).
  - Both checkpoints are AdvancedObs bots (papaya + a teammate's pushed checkpoint).

Run (from repo root, rlbot310 active):
  conda activate rlbot310
  # papaya (auto: latest papaya_1024 checkpoint) vs a teammate checkpoint:
  python diego-bots/papaya_1v1_viewer.py --orange path/to/martin/<run>/<timestep>
  # or be explicit about both:
  python diego-bots/papaya_1v1_viewer.py ^
      --blue   diego-bots/checkpoints/papaya_1024 ^
      --orange path/to/nachi/checkpoints/<exp>/<timestep> ^
      --episodes 10 --deterministic

--blue / --orange accept either a numeric timestep folder (one containing
PPO_POLICY.pt) OR any parent folder -- the latest timestep checkpoint beneath it
is auto-selected.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

# AdvancedObs 1v1 = 107 dims; LookupAction = 90 discrete actions. These are the
# fixed specs every AdvancedObs bot on the team trains against.
OBS_DIM_1V1 = 107
N_ACTIONS = 90

# Default location of papaya's checkpoints (the "blue"/home bot).
DEFAULT_BLUE = "diego-bots/checkpoints/papaya_1024"


def resolve_checkpoint(path_str: str) -> Path:
    """Return a timestep folder containing PPO_POLICY.pt.

    Accepts either that folder directly, or any ancestor folder -- in which case
    the highest-numbered (latest cumulative timestep) checkpoint beneath it that
    actually has weights is selected. Works across papaya's nested layout
    (<exp>/<exp>-<unix>/<timestep>/) and the teammates' pipeline layout
    (<exp>/<timestep>/).
    """
    p = Path(path_str)
    if not p.exists():
        raise SystemExit(f"Checkpoint path does not exist: {p}")
    if (p / "PPO_POLICY.pt").is_file():
        return p

    candidates = [
        d for d in p.rglob("*")
        if d.is_dir() and d.name.isdigit() and (d / "PPO_POLICY.pt").is_file()
    ]
    if not candidates:
        raise SystemExit(
            f"No checkpoint with PPO_POLICY.pt found under {p}. "
            "Has this bot saved a checkpoint yet?"
        )
    latest = max(candidates, key=lambda d: int(d.name))
    return latest


def load_policy(ckpt_dir: Path, device: str = "cpu"):
    """Build a DiscreteFF matching the checkpoint and load its weights.

    Hidden-layer sizes and the input/action dims are read straight from the
    saved state dict, so we never need to know the bot's arch in advance. The
    input dim is validated against AdvancedObs (107) to catch obs mismatches
    (e.g. a DefaultObs 89-dim bot) before they cause silent garbage play.
    """
    from rlgym_ppo.ppo.discrete_policy import DiscreteFF

    weights_path = ckpt_dir / "PPO_POLICY.pt"
    sd = torch.load(weights_path, map_location=device)

    weight_keys = [k for k in sd if k.endswith("weight")]
    if not weight_keys:
        raise SystemExit(f"No linear weights found in {weights_path}")

    in_dim = int(sd[weight_keys[0]].shape[1])
    out_dim = int(sd[weight_keys[-1]].shape[0])
    hidden_sizes = tuple(int(sd[k].shape[0]) for k in weight_keys[:-1])

    if in_dim != OBS_DIM_1V1:
        raise SystemExit(
            f"OBS MISMATCH for {ckpt_dir}\n"
            f"  checkpoint expects input dim {in_dim}, but this viewer feeds "
            f"AdvancedObs ({OBS_DIM_1V1}).\n"
            f"  This is almost certainly a DefaultObs (89-dim) bot -- it cannot "
            f"play AdvancedObs bots. Use a 107-dim AdvancedObs checkpoint."
        )
    if out_dim != N_ACTIONS:
        raise SystemExit(
            f"ACTION MISMATCH for {ckpt_dir}: checkpoint has {out_dim} actions, "
            f"expected {N_ACTIONS} (LookupAction)."
        )

    policy = DiscreteFF(OBS_DIM_1V1, N_ACTIONS, hidden_sizes, device)
    policy.load_state_dict(sd)
    policy.eval()
    return policy, hidden_sizes


def build_env():
    """Minimal 1v1 AdvancedObs env starting from a clean kickoff each episode."""
    import rlgym_sim
    from rlgym_sim.utils.obs_builders import AdvancedObs
    from rlgym_sim.utils.reward_functions import DefaultReward
    from rlgym_sim.utils.state_setters import DefaultState
    from rlgym_sim.utils.terminal_conditions.common_conditions import (
        GoalScoredCondition,
        TimeoutCondition,
    )

    from rlbot.actions.lookup_action import LookupAction

    # ~200 simulated seconds cap so a scoreless episode still ends.
    timeout_steps = 3000

    return rlgym_sim.make(
        tick_skip=8,
        team_size=1,
        spawn_opponents=True,
        obs_builder=AdvancedObs(),
        action_parser=LookupAction(),
        reward_fn=DefaultReward(),
        state_setter=DefaultState(),
        terminal_conditions=[GoalScoredCondition(), TimeoutCondition(timeout_steps)],
    )


def _action_to_int(action) -> int:
    if isinstance(action, np.ndarray):
        return int(action.flat[0])
    if isinstance(action, torch.Tensor):
        return int(action.item())
    return int(action)


def run_match(
    blue_ckpt: Path,
    orange_ckpt: Path,
    episodes: int,
    deterministic: bool,
    step_delay: float,
    render: bool = True,
) -> dict:
    """Play `episodes` games of blue vs orange. Returns a W/L/D summary dict.

    `render=False` skips the rlviser calls -- used by the smoke test so it can run
    headless without the rlviser binary open.
    """
    blue_policy, blue_arch = load_policy(blue_ckpt)
    orange_policy, orange_arch = load_policy(orange_ckpt)
    print(f"BLUE   {blue_ckpt}  arch={blue_arch}")
    print(f"ORANGE {orange_ckpt}  arch={orange_arch}")

    env = build_env()
    blue_wins = orange_wins = draws = 0
    if render:
        print("Make sure rlviser.exe is open. Press Ctrl+C to stop.")
    try:
        for ep in range(1, episodes + 1):
            obs_list = env.reset()
            blue_obs, orange_obs = obs_list[0], obs_list[1]
            done = False
            info: dict = {}
            while not done:
                with torch.no_grad():
                    b_act, _ = blue_policy.get_action(blue_obs, deterministic=deterministic)
                    o_act, _ = orange_policy.get_action(orange_obs, deterministic=deterministic)
                obs_list, _r, done, info = env.step(
                    [_action_to_int(b_act), _action_to_int(o_act)]
                )
                blue_obs, orange_obs = obs_list[0], obs_list[1]
                if render:
                    env.render()
                    time.sleep(step_delay)

            result = int(info.get("result", 0))
            if result > 0:
                blue_wins += 1
                outcome = "BLUE"
            elif result < 0:
                orange_wins += 1
                outcome = "ORANGE"
            else:
                draws += 1
                outcome = "DRAW"
            print(f"Ep {ep}/{episodes}  {outcome}  (delta={result:+d})")
    finally:
        env.close()

    print(
        f"\nFinal: BLUE {blue_wins}W / ORANGE {orange_wins}W / {draws}D "
        f"over {episodes} games  ->  blue_win_rate={blue_wins / max(episodes,1):.1%}"
    )
    return {"blue_wins": blue_wins, "orange_wins": orange_wins, "draws": draws}


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--blue",
        default=DEFAULT_BLUE,
        help=f"Blue (home) checkpoint or parent folder. Default: latest under {DEFAULT_BLUE}",
    )
    p.add_argument(
        "--orange",
        required=True,
        help="Orange (opponent) checkpoint or parent folder -- e.g. Martin's or Nachi's.",
    )
    p.add_argument("--episodes", type=int, default=5, help="Number of games to play.")
    p.add_argument(
        "--deterministic",
        action="store_true",
        help="Greedy argmax instead of sampling -- looks less twitchy.",
    )
    p.add_argument(
        "--step-delay",
        type=float,
        default=0.006,
        help="Seconds to sleep per env.step. 0.006 ~= real time at tick_skip=8; bump to 0.01 if too fast.",
    )
    args = p.parse_args()

    blue_ckpt = resolve_checkpoint(args.blue)
    orange_ckpt = resolve_checkpoint(args.orange)
    run_match(
        blue_ckpt,
        orange_ckpt,
        episodes=args.episodes,
        deterministic=args.deterministic,
        step_delay=args.step_delay,
        render=True,
    )


if __name__ == "__main__":
    main()
