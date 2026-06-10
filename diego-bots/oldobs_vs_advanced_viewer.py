"""Watch a DefaultObs bot 1v1 an AdvancedObs bot in rlviser (mixed-obs match).

Specifically built to pit papaya_1024_oldobs (DefaultObs, 89-dim) against
papaya_1024_318M (AdvancedObs, 107-dim) so you can see whether the AdvancedObs
upgrade actually plays better.

The hard problem this solves:
  A normal rlgym_sim env has ONE obs builder feeding both cars, so a 89-dim bot
  and a 107-dim bot can't share it -- one net always gets a wrong-size input and
  crashes. MixedObs fixes that by returning each car its OWN training obs:
      blue  (team 0) -> DefaultObs  (89)   == papaya_1024_oldobs
      orange(team 1) -> AdvancedObs (107)  == papaya_1024_318M
  Each policy consumes only its own car's obs, so the two dims never need to
  match each other. This is also the FAIR comparison: each bot sees the world
  exactly as it was trained to.

Prereqs:
  - rlviser.exe is running (double-click it; an arena window opens).
  - rlbot310 env (rlgym_sim + rlviser_py + torch).

Run (from repo root):
  conda activate rlbot310
  # defaults already point blue->archived oldobs, orange->latest papaya_1024:
  python diego-bots/oldobs_vs_advanced_viewer.py --episodes 10 --deterministic
  # or override either side:
  python diego-bots/oldobs_vs_advanced_viewer.py ^
      --blue   diego-bots/checkpoints/_archive/papaya_1024_DEFAULTOBS_287M ^
      --orange diego-bots/checkpoints/papaya_1024 ^
      --episodes 10 --deterministic

BLUE must be the DefaultObs (89-dim) bot, ORANGE the AdvancedObs (107-dim) bot.
The loader validates this and errors clearly if you swap them.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

# Per-side obs dims (1v1) and the shared discrete action space.
BLUE_OBS_DIM = 89    # DefaultObs  -> papaya_1024_oldobs
ORANGE_OBS_DIM = 107  # AdvancedObs -> papaya_1024_318M
N_ACTIONS = 90

DEFAULT_BLUE = "diego-bots/checkpoints/_archive/papaya_1024_DEFAULTOBS_287M"
DEFAULT_ORANGE = "diego-bots/checkpoints/papaya_1024"


def resolve_checkpoint(path_str: str) -> Path:
    """Return a timestep folder containing PPO_POLICY.pt, auto-selecting the
    latest checkpoint beneath the path if it isn't a leaf checkpoint itself."""
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
        raise SystemExit(f"No checkpoint with PPO_POLICY.pt found under {p}")
    return max(candidates, key=lambda d: int(d.name))


def load_policy(ckpt_dir: Path, expected_obs_dim: int, side: str, device: str = "cpu"):
    """Build a DiscreteFF matching the checkpoint's weights and load them.

    Validates the checkpoint's input dim equals expected_obs_dim for this side,
    so a swapped --blue/--orange fails loudly instead of crashing mid-match.
    """
    from rlgym_ppo.ppo.discrete_policy import DiscreteFF

    sd = torch.load(ckpt_dir / "PPO_POLICY.pt", map_location=device, weights_only=True)
    weight_keys = [k for k in sd if k.endswith("weight")]
    in_dim = int(sd[weight_keys[0]].shape[1])
    out_dim = int(sd[weight_keys[-1]].shape[0])
    hidden = tuple(int(sd[k].shape[0]) for k in weight_keys[:-1])

    if in_dim != expected_obs_dim:
        raise SystemExit(
            f"{side.upper()} obs mismatch: {ckpt_dir}\n"
            f"  this checkpoint expects input dim {in_dim}, but the {side} car is "
            f"fed {expected_obs_dim}-dim obs.\n"
            f"  BLUE must be the DefaultObs (89) bot, ORANGE the AdvancedObs (107) "
            f"bot. Did you swap --blue/--orange?"
        )
    if out_dim != N_ACTIONS:
        raise SystemExit(f"{side.upper()} action mismatch: {out_dim} != {N_ACTIONS}")

    policy = DiscreteFF(expected_obs_dim, N_ACTIONS, hidden, device)
    policy.load_state_dict(sd)
    policy.eval()
    return policy, hidden


def build_env():
    """1v1 env with a MIXED obs builder: DefaultObs for blue, AdvancedObs for orange."""
    import rlgym_sim
    from rlgym_sim.utils.obs_builders import AdvancedObs, DefaultObs, ObsBuilder
    from rlgym_sim.utils.reward_functions import DefaultReward
    from rlgym_sim.utils.state_setters import DefaultState
    from rlgym_sim.utils.terminal_conditions.common_conditions import (
        GoalScoredCondition,
        TimeoutCondition,
    )

    from rlbot.actions.lookup_action import LookupAction

    class MixedObs(ObsBuilder):
        """Each car gets its own training obs: team 0 -> DefaultObs (89),
        team 1 -> AdvancedObs (107). Returns different-length vectors per player,
        which is fine -- each policy only ever consumes its own car's obs."""

        def __init__(self):
            super().__init__()
            self._default = DefaultObs()
            self._advanced = AdvancedObs()

        def reset(self, initial_state):
            self._default.reset(initial_state)
            self._advanced.reset(initial_state)

        def build_obs(self, player, state, previous_action):
            if player.team_num == 0:  # blue
                return self._default.build_obs(player, state, previous_action)
            return self._advanced.build_obs(player, state, previous_action)

    return rlgym_sim.make(
        tick_skip=8,
        team_size=1,
        spawn_opponents=True,
        obs_builder=MixedObs(),
        action_parser=LookupAction(),
        reward_fn=DefaultReward(),
        state_setter=DefaultState(),
        terminal_conditions=[GoalScoredCondition(), TimeoutCondition(3000)],
    )


def _action_to_int(action) -> int:
    if isinstance(action, np.ndarray):
        return int(action.flat[0])
    if isinstance(action, torch.Tensor):
        return int(action.item())
    return int(action)


def run_match(blue_ckpt, orange_ckpt, episodes, deterministic, step_delay, render=True):
    blue_policy, blue_arch = load_policy(blue_ckpt, BLUE_OBS_DIM, "blue")
    orange_policy, orange_arch = load_policy(orange_ckpt, ORANGE_OBS_DIM, "orange")
    print(f"BLUE   (DefaultObs 89)  {blue_ckpt}  arch={blue_arch}")
    print(f"ORANGE (AdvancedObs 107) {orange_ckpt}  arch={orange_arch}")

    env = build_env()
    blue_wins = orange_wins = draws = 0
    if render:
        print("Make sure rlviser.exe is open. Press Ctrl+C to stop.")
    try:
        for ep in range(1, episodes + 1):
            obs_list = env.reset()
            blue_obs, orange_obs = obs_list[0], obs_list[1]
            done = False
            info = {}
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
                outcome = "BLUE (oldobs)"
            elif result < 0:
                orange_wins += 1
                outcome = "ORANGE (318M adv)"
            else:
                draws += 1
                outcome = "DRAW"
            print(f"Ep {ep}/{episodes}  {outcome}  (delta={result:+d})")
    finally:
        env.close()

    print(
        f"\nFinal: BLUE(oldobs) {blue_wins}W / ORANGE(318M adv) {orange_wins}W / "
        f"{draws}D over {episodes} games  ->  orange_win_rate="
        f"{orange_wins / max(episodes,1):.1%}"
    )
    return {"blue_wins": blue_wins, "orange_wins": orange_wins, "draws": draws}


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--blue", default=DEFAULT_BLUE, help="DefaultObs (89) bot = papaya_1024_oldobs.")
    p.add_argument("--orange", default=DEFAULT_ORANGE, help="AdvancedObs (107) bot = papaya_1024_318M.")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--deterministic", action="store_true", help="Greedy argmax (less twitchy).")
    p.add_argument("--step-delay", type=float, default=0.006, help="Per-step sleep; 0.006 ~= real time.")
    args = p.parse_args()

    blue_ckpt = resolve_checkpoint(args.blue)
    orange_ckpt = resolve_checkpoint(args.orange)
    run_match(blue_ckpt, orange_ckpt, args.episodes, args.deterministic, args.step_delay, render=True)


if __name__ == "__main__":
    main()
