"""Real bot-vs-bot evaluation. Runs N episodes between two checkpoints, reports win rate.

    # Compare oldest vs newest checkpoint (from different timestep dirs)
    python -m rlbot.evaluation.evaluate \
        --blue  checkpoints/exp_003_long_run/736294572 \
        --orange checkpoints/exp_003_long_run/756195776 \
        --episodes 20

    # Use --latest to automatically pick the highest-numbered checkpoint
    python -m rlbot.evaluation.evaluate \
        --blue  checkpoints/exp_003_long_run/736294572 \
        --orange latest:exp_003_long_run \
        --episodes 20

    # Pass --deterministic for greedy (no random) action selection during eval
    python -m rlbot.evaluation.evaluate \
        --blue  checkpoints/exp_003_long_run/736294572 \
        --orange checkpoints/exp_003_long_run/756195776 \
        --episodes 50 --deterministic

Can run in a separate terminal while training is running — reads checkpoints, does not
write anything, does not affect the training process.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from rlbot.utils.logging import get_logger

# ── Constants computed from DefaultObs + LookupAction for 1v1 ──────────────────
# ball(pos 3 + linvel 3 + angvel 3) + prev_action 8 + pads 34
#   + self_player(pos 3 + fwd 3 + up 3 + linvel 3 + angvel 3 + misc 4)
#   + opp_player (same 19)
# Total = 51 + 19 + 19 = 89
_OBS_SIZE_1V1 = 89
# LookupAction with default bins=(-1,0,1)×5 produces 90 discrete actions
_N_ACTIONS = 90


def _resolve_checkpoint_path(path_str: str) -> Path:
    """Resolve 'latest:exp_name' shortcut or return the path as-is.

    Examples
    --------
    'checkpoints/exp_003_long_run/756195776'  →  that exact dir (relative to repo root)
    'latest:exp_003_long_run'                 →  highest-numbered subdir in checkpoints/exp_003_long_run/
    """
    repo_root = Path(__file__).resolve().parents[3]

    if path_str.startswith("latest:"):
        exp_name = path_str[len("latest:"):]
        ckpt_root = repo_root / "checkpoints" / exp_name
        subdirs = sorted(
            (d for d in ckpt_root.iterdir() if d.is_dir() and d.name.isdigit()),
            key=lambda d: int(d.name),
        )
        if not subdirs:
            raise FileNotFoundError(f"No checkpoint subdirs found under {ckpt_root}")
        return subdirs[-1]

    p = Path(path_str)
    # If relative, resolve against repo root (works regardless of where user runs from)
    if not p.is_absolute():
        p = repo_root / p
    return p


def _load_policy(checkpoint_path: Path, device: str = "cpu"):
    """Load a DiscreteFF policy from an rlgym_ppo checkpoint folder.

    Reads layer sizes from BOOK_KEEPING_VARS.json, builds the network,
    loads PPO_POLICY.pt weights.
    """
    from rlgym_ppo.ppo.discrete_policy import DiscreteFF

    book_path = checkpoint_path / "BOOK_KEEPING_VARS.json"
    if not book_path.exists():
        raise FileNotFoundError(f"BOOK_KEEPING_VARS.json not found in {checkpoint_path}")

    book = json.loads(book_path.read_text())
    layer_sizes = tuple(book["wandb_config"]["policy_layer_sizes"])

    policy = DiscreteFF(
        input_shape=_OBS_SIZE_1V1,
        n_actions=_N_ACTIONS,
        layer_sizes=layer_sizes,
        device=device,
    )

    weights_path = checkpoint_path / "PPO_POLICY.pt"
    state_dict = torch.load(weights_path, map_location=device)
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy


def _build_eval_env():
    """Build a minimal 1v1 rlgym_sim env for evaluation.

    Uses DefaultState (kickoff) so every episode starts from a clean kickoff.
    Terminates on goal scored OR after ~40 simulated seconds (3000 steps × tick_skip 8 = 24 000 ticks ÷ 120 Hz ≈ 200 s).
    """
    import rlgym_sim
    from rlgym_sim.utils.obs_builders import DefaultObs
    from rlgym_sim.utils.reward_functions import DefaultReward
    from rlgym_sim.utils.state_setters import DefaultState
    from rlgym_sim.utils.terminal_conditions.common_conditions import (
        GoalScoredCondition,
        TimeoutCondition,
    )

    from rlbot.actions.lookup_act import LookupAction

    # ~200 simulated seconds max per episode to avoid infinite no-touch episodes
    timeout_steps = 3000

    env = rlgym_sim.make(
        tick_skip=8,
        team_size=1,
        spawn_opponents=True,
        obs_builder=DefaultObs(),
        action_parser=LookupAction(),
        reward_fn=DefaultReward(),
        terminal_conditions=[GoalScoredCondition(), TimeoutCondition(timeout_steps)],
        state_setter=DefaultState(),
    )
    return env


def _action_to_int(action) -> int:
    """Normalise whatever get_action() returned into a plain Python int."""
    if isinstance(action, np.ndarray):
        return int(action.flat[0])
    if isinstance(action, torch.Tensor):
        return int(action.item())
    return int(action)


def evaluate(
    blue_path: str,
    orange_path: str,
    episodes: int,
    deterministic: bool,
    device: str = "cpu",
) -> dict:
    """Run N episodes of blue vs orange, return win/loss/draw metrics.

    Parameters
    ----------
    blue_path:    Path to checkpoint dir (or 'latest:exp_name' shorthand)
    orange_path:  Same for the orange policy
    episodes:     How many full episodes to simulate
    deterministic: If True, both policies pick the highest-prob action (greedy)
    device:       Torch device, 'cpu' is fine for eval

    Returns
    -------
    dict with keys: blue_wins, orange_wins, draws, blue_win_rate
    """
    log = get_logger("rlbot.evaluate")

    blue_ckpt = _resolve_checkpoint_path(blue_path)
    orange_ckpt = _resolve_checkpoint_path(orange_path)

    log.info(f"[cyan]Blue[/]   checkpoint: {blue_ckpt}")
    log.info(f"[magenta]Orange[/] checkpoint: {orange_ckpt}")

    blue_policy = _load_policy(blue_ckpt, device)
    orange_policy = _load_policy(orange_ckpt, device)
    log.info("Policies loaded.")

    log.info("Building eval env (1v1, DefaultState kickoff)...")
    env = _build_eval_env()
    log.info("Env ready. Starting episodes...")

    blue_wins = 0
    orange_wins = 0
    draws = 0

    try:
        for ep in range(1, episodes + 1):
            obs_list = env.reset()
            blue_obs: np.ndarray = obs_list[0]
            orange_obs: np.ndarray = obs_list[1]
            done = False
            info: dict = {}

            while not done:
                with torch.no_grad():
                    b_act, _ = blue_policy.get_action(blue_obs, deterministic=deterministic)
                    o_act, _ = orange_policy.get_action(orange_obs, deterministic=deterministic)

                b_idx = _action_to_int(b_act)
                o_idx = _action_to_int(o_act)

                obs_list, _rewards, done, info = env.step([b_idx, o_idx])
                blue_obs = obs_list[0]
                orange_obs = obs_list[1]

            # info['result'] = blue_score - orange_score - initial_score_delta
            # Positive  → blue scored net this episode
            # Negative  → orange scored net this episode
            # Zero      → timeout / no goals
            result: int = info.get("result", 0)

            if result > 0:
                blue_wins += 1
                outcome = "[cyan]BLUE[/] wins"
            elif result < 0:
                orange_wins += 1
                outcome = "[magenta]ORANGE[/] wins"
            else:
                draws += 1
                outcome = "draw (timeout)"

            log.info(f"Ep {ep:>3}/{episodes}  {outcome}  (score delta={int(result):+d})")

    finally:
        env.close()

    blue_win_rate = blue_wins / episodes

    log.info(
        f"\nFinal: [cyan]Blue[/] {blue_wins}W / [magenta]Orange[/] {orange_wins}W / {draws}D "
        f"over {episodes} episodes  →  blue_win_rate={blue_win_rate:.1%}"
    )

    return {
        "blue": str(blue_ckpt),
        "orange": str(orange_ckpt),
        "episodes": episodes,
        "deterministic": deterministic,
        "blue_wins": blue_wins,
        "orange_wins": orange_wins,
        "draws": draws,
        "blue_win_rate": blue_win_rate,
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="Evaluate two rlgym-ppo checkpoints head-to-head."
    )
    p.add_argument(
        "--blue",
        required=True,
        help="Path to blue checkpoint dir, e.g. checkpoints/exp_003_long_run/736294572",
    )
    p.add_argument(
        "--orange",
        required=True,
        help="Path to orange checkpoint dir, or 'latest:exp_name' to auto-pick newest",
    )
    p.add_argument("--episodes", type=int, default=20, help="Number of episodes to run")
    p.add_argument(
        "--deterministic",
        action="store_true",
        help="Use greedy (argmax) action selection instead of sampling",
    )
    args = p.parse_args()

    result = evaluate(args.blue, args.orange, args.episodes, args.deterministic)
    log = get_logger("rlbot.evaluate")
    log.info(
        f"Result: blue_win_rate={result['blue_win_rate']:.3f} "
        f"(W/L/D = {result['blue_wins']}/{result['orange_wins']}/{result['draws']})"
    )


if __name__ == "__main__":
    main()
