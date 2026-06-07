"""Bot-vs-bot evaluation. Runs N episodes between two checkpoints, reports win rate.

    python -m rlbot.evaluation.evaluate \
        --blue   checkpoints/exp_003_long_run-<id>/<step> \
        --orange checkpoints/exp_002_advanced_obs-<id>/<step> \
        --episodes 50

Both checkpoints must share the same observation builder (the eval env produces one
obs per car and feeds it to both policies). Episodes start from kickoff and end on a
goal (a win for the scoring side) or a timeout (a draw). Each network's hidden-layer
sizes are inferred from its `PPO_POLICY.pt`, so blue and orange may differ in arch.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from rlbot.env import make_env_builder
from rlbot.utils.config import load_config
from rlbot.utils.logging import get_logger

# exp_003 supplies advanced obs + lookup action — what the 1024x3 bots trained on.
DEFAULT_EVAL_CONFIG = "configs/experiments/exp_003_long_run.yaml"


def _resolve_checkpoint(path: str) -> Path:
    """Resolve a path to the folder that actually contains PPO_POLICY.pt."""
    p = Path(path)
    if (p / "PPO_POLICY.pt").exists():
        return p
    base = p.parent if p.name == "latest" else p
    steps = sorted(
        (d for d in base.glob("*") if d.is_dir() and d.name.isdigit()),
        key=lambda d: int(d.name),
    )
    if steps and (steps[-1] / "PPO_POLICY.pt").exists():
        return steps[-1]
    raise FileNotFoundError(f"No PPO_POLICY.pt found at or under {path}")


def _load_policy(ckpt_dir: Path, obs_dim: int, n_actions: int, device: str):
    """Rebuild a DiscreteFF policy, inferring hidden sizes from the saved weights."""
    from rlgym_ppo.ppo import DiscreteFF

    sd = torch.load(ckpt_dir / "PPO_POLICY.pt", map_location=device, weights_only=True)
    weight_keys = sorted((k for k in sd if k.endswith(".weight")), key=lambda k: int(k.split(".")[1]))
    layer_sizes = tuple(int(sd[k].shape[0]) for k in weight_keys[:-1])  # all but the action head
    policy = DiscreteFF(obs_dim, n_actions, layer_sizes, device)
    policy.load_state_dict(sd)
    policy.eval()
    return policy


def _build_eval_env(config_path: str, max_seconds: int):
    """1v1 eval env: kickoff start, goal-or-timeout end."""
    full = load_config(config_path).to_dict()
    full["state_setter"] = {"name": "default"}  # kickoff every episode
    full["terminal"]["timeout_seconds"] = int(max_seconds)
    env_cfg = dict(full["env"])
    env_cfg["team_size"] = 1
    env_cfg["spawn_opponents"] = True
    return make_env_builder(env_cfg, full)()


def _act(policy, obs, deterministic: bool) -> int:
    with torch.no_grad():
        return int(policy.get_action(np.asarray(obs, dtype=np.float32), deterministic=deterministic)[0])


def evaluate(
    blue_path: str,
    orange_path: str,
    episodes: int,
    deterministic: bool,
    config_path: str = DEFAULT_EVAL_CONFIG,
    max_seconds: int = 60,
) -> dict:
    """Play blue vs orange for `episodes` games. Returns a metrics dict."""
    log = get_logger("rlbot.evaluate")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = _build_eval_env(config_path, max_seconds)
    first = env.reset()
    first = first if isinstance(first, list) else [first]
    obs_dim = int(np.asarray(first[0], dtype=np.float32).shape[0])
    n_actions = int(env.action_space.n)

    blue = _load_policy(_resolve_checkpoint(blue_path), obs_dim, n_actions, device)
    orange = _load_policy(_resolve_checkpoint(orange_path), obs_dim, n_actions, device)
    log.info(
        f"Eval: [cyan]{blue_path}[/] (blue) vs [magenta]{orange_path}[/] (orange) — "
        f"{episodes} eps, deterministic={deterministic}, device={device}"
    )

    blue_wins = orange_wins = draws = 0
    for ep in range(episodes):
        obs, info = env.reset(return_info=True)
        obs = obs if isinstance(obs, list) else [obs]
        teams = [int(p.team_num) for p in info["state"].players]
        done, result = False, 0.0
        while not done:
            acts = [[_act(blue if teams[i] == 0 else orange, o, deterministic)] for i, o in enumerate(obs)]
            obs, _, done, info = env.step(np.array(acts))
            obs = obs if isinstance(obs, list) else [obs]
            result = info["result"]
        if result > 0:
            blue_wins += 1
        elif result < 0:
            orange_wins += 1
        else:
            draws += 1
        if (ep + 1) % 10 == 0 or ep + 1 == episodes:
            log.info(f"  {ep + 1}/{episodes}  blue {blue_wins} / orange {orange_wins} / draw {draws}")

    decisive = blue_wins + orange_wins
    return {
        "blue": str(blue_path),
        "orange": str(orange_path),
        "episodes": episodes,
        "deterministic": deterministic,
        "blue_wins": blue_wins,
        "orange_wins": orange_wins,
        "draws": draws,
        "blue_win_rate": round(blue_wins / episodes, 3),
        "blue_win_rate_decisive": round(blue_wins / decisive, 3) if decisive else None,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--blue", required=True, help="Path to blue policy checkpoint")
    p.add_argument("--orange", required=True, help="Path to orange policy checkpoint")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--config", default=DEFAULT_EVAL_CONFIG, help="Env config (obs/action source)")
    p.add_argument("--max-seconds", type=int, default=60, help="Per-episode timeout (game seconds)")
    args = p.parse_args()

    result = evaluate(
        args.blue, args.orange, args.episodes, args.deterministic, args.config, args.max_seconds
    )
    log = get_logger("rlbot.evaluate")
    log.info(
        f"Result: blue_win_rate={result['blue_win_rate']:.3f}  "
        f"(W/L/D = {result['blue_wins']}/{result['orange_wins']}/{result['draws']}, "
        f"decisive blue rate={result['blue_win_rate_decisive']})"
    )


if __name__ == "__main__":
    main()
