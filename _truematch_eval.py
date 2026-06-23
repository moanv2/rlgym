"""TRUE head-to-head: our DefaultObs(89) bot vs a rival AdvancedObs(107) bot.
Both use LookupAction(90) -> one action parser. Per-player obs via a wrapper
(rlgym_sim returns per-player obs as a list, so 89 vs 107 sizes are fine).

Arch + obs-size are INFERRED from each policy's state dict (don't trust bookkeeping).
"""
import sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

from rlgym_ppo.ppo.discrete_policy import DiscreteFF
from rlgym_sim.utils.obs_builders import DefaultObs, ObsBuilder
from rlbot.obs.advanced_obs import AdvancedObs           # Diego's 107-dim obs
from rlbot.actions.lookup_action import LookupAction


def load_policy(ckpt_dir, device="cpu"):
    ckpt = Path(ckpt_dir)
    if not ckpt.is_absolute():
        ckpt = REPO / ckpt
    sd = torch.load(ckpt / "PPO_POLICY.pt", map_location=device, weights_only=True)
    wkeys = sorted((k for k in sd if k.endswith(".weight")),
                   key=lambda k: int(k.split(".")[1]))
    weights = [sd[k] for k in wkeys]
    obs_size = int(weights[0].shape[1])
    n_actions = int(weights[-1].shape[0])
    layers = tuple(int(w.shape[0]) for w in weights[:-1])
    pol = DiscreteFF(obs_size, n_actions, layers, device)
    pol.load_state_dict(sd)
    pol.eval()
    return pol, obs_size


class AsymObs(ObsBuilder):
    def __init__(self, blue, orange):
        super().__init__()
        self.blue, self.orange = blue, orange
    def reset(self, state):
        self.blue.reset(state); self.orange.reset(state)
    def build_obs(self, player, state, prev):
        return (self.blue if player.team_num == 0 else self.orange).build_obs(player, state, prev)
    def get_obs_space(self):
        return None


def blue_obs_for(obs_size):
    """Pick our bot's obs builder from its detected input size."""
    if obs_size == 107:
        from rlbot.obs.default_plus_obs import DefaultPlusObs
        return DefaultPlusObs()
    return DefaultObs()      # 89-dim original 2B bot


def build_env(blue_obs, orange_obs):
    import rlgym_sim
    from rlgym_sim.utils.reward_functions import DefaultReward
    from rlgym_sim.utils.state_setters import DefaultState
    from rlgym_sim.utils.terminal_conditions.common_conditions import (
        GoalScoredCondition, TimeoutCondition)
    return rlgym_sim.make(
        tick_skip=8, team_size=1, spawn_opponents=True,
        obs_builder=AsymObs(blue_obs, orange_obs),
        action_parser=LookupAction(),
        reward_fn=DefaultReward(),
        terminal_conditions=[GoalScoredCondition(), TimeoutCondition(3000)],
        state_setter=DefaultState(),
    )


def argmax(pol, obs):
    probs = pol.get_output(obs).view(-1, pol.n_actions)
    return int(probs.detach().cpu().numpy().argmax())


def run(our_ckpt, opp_ckpt, episodes, orange_obs):
    our, our_n = load_policy(our_ckpt)
    opp, opp_n = load_policy(opp_ckpt)
    print(f"  [our obs={our_n}, opp obs={opp_n}]", flush=True)
    env = build_env(blue_obs_for(our_n), orange_obs)
    w = l = d = 0
    try:
        for _ in range(episodes):
            obs = env.reset(); done = False; info = {}
            while not done:
                with torch.no_grad():
                    b = argmax(our, obs[0])     # blue = us (89)
                    o = argmax(opp, obs[1])     # orange = rival (107)
                obs, _, done, info = env.step([b, o])
            r = info.get("result", 0)
            w += r > 0; l += r < 0; d += r == 0
    finally:
        env.close()
    return w, l, d


def make_orange_obs(kind):
    if kind == "diego_custom":
        from rlbot.obs.advanced_obs import AdvancedObs as A   # Diego's custom 107
        return A()
    from rlgym_sim.utils.obs_builders import AdvancedObs as A  # stock rlgym_sim (Martin/Nachi)
    return A()

OPPONENTS = {
    "diego":  ("diego-bots/checkpoints/MILESTONE_2.85B_papaya_1024_v7", "diego_custom", "Diego papaya 2.85B"),
    "martin": ("martin-bots/checkpoints/CHAMPION_8.23B_advanced1024",   "stock_adv",    "Martin 8.23B"),
    "nachi":  ("checkpoints/shared/2896166208",                          "stock_adv",    "Nachi 2.9B"),
}

if __name__ == "__main__":
    name = sys.argv[1]
    EP = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    OUR = sys.argv[3] if len(sys.argv) > 3 else "checkpoints/exp_007_large/1999210400"
    ckpt, kind, label = OPPONENTS[name]
    print(f"=== Marco 2.0B (blue) vs {label} (orange), deterministic, {EP} eps ===", flush=True)
    w, l, d = run(OUR, ckpt, EP, make_orange_obs(kind))
    print(f"\nMarco vs {label}: win%={w/EP:.0%}   (W/L/D = {w}-{l}-{d})", flush=True)
