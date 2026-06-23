"""Counter-training vs the FIELD: fine-tune our expanded (107-obs) bot to beat the
actual tournament rivals, rotating opponents so it doesn't overfit one.

Single-agent env:
  * blue  = OUR policy (DefaultPlusObs 107) — the one being trained
  * orange = a FROZEN rival, re-picked each reset from {Diego, Nachi, Martin}
The Learner only ever sees/trains blue. Because the rivals are FIXED (exactly the
tournament bots), exploiting their weaknesses TRANSFERS to the tournament — unlike
self-play, this sidesteps the non-transitivity that regressed us past 2B.

Run:
    python _counter_train_field.py --smoke           # build env, rotate opponents, step
    python _counter_train_field.py <resume_ckpt>     # counter-train, resume from 107 ckpt
"""
from __future__ import annotations

import sys
import random
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

# (name, checkpoint dir, orange-obs kind, rotation weight)
# Focus the two WINNABLE rivals (Diego, Nachi) equally; light Martin exposure (he's ~unbeatable).
OPPONENTS = [
    ("nachi",  str(REPO / "checkpoints" / "shared" / "2896166208"),                          "stock_adv",    0.40),
    ("diego",  str(REPO / "diego-bots" / "checkpoints" / "MILESTONE_2.85B_papaya_1024_v7"), "diego_custom", 0.40),
    ("martin", str(REPO / "martin-bots" / "checkpoints" / "CHAMPION_8.23B_advanced1024"),     "stock_adv",    0.20),
]
SAVE_DIR = str(REPO / "checkpoints" / "exp014_lowlr_from95M")

# PLAIN aggressive scoring reward — the exact stack that built the 2B bot. No defensive
# shaping (the anti-Nachi turtling killed our Diego edge). Just: score more vs the fixed rivals.
REWARD_CFG = {
    "zero_sum": True, "team_spirit": 0.0, "opp_scale": 1.0,
    "components": [
        {"name": "touch_ball", "weight": 1.0},
        {"name": "velocity_player_to_ball", "weight": 0.3},
        {"name": "face_ball", "weight": 0.1},
        {"name": "velocity_ball_to_goal", "weight": 0.8},
        {"name": "event", "weight": 15.0,
         "kwargs": {"goal": 1.0, "concede": -1.0, "shot": 0.2, "save": 0.3, "demo": 0.1}},
    ],
}
STATE_CFG = {"name": "weighted_sample", "components": [
    {"name": "default", "weight": 0.5},
    {"name": "random", "weight": 0.5, "ball_rand_speed": True,
     "cars_rand_speed": True, "cars_on_ground": False},
]}


def make_orange_obs(kind):
    if kind == "diego_custom":
        from rlbot.obs.advanced_obs import AdvancedObs as A
        return A()
    from rlgym_sim.utils.obs_builders import AdvancedObs as A
    return A()


class _Frozen:
    """A frozen rival policy: numpy obs in, action int out (lazy torch per worker)."""
    def __init__(self, ckpt: str):
        import torch
        torch.set_num_threads(1)   # 20 worker procs each run this -> avoid CPU oversubscription
        from rlgym_ppo.ppo.discrete_policy import DiscreteFF
        self._t = torch
        sd = torch.load(Path(ckpt) / "PPO_POLICY.pt", map_location="cpu", weights_only=True)
        wk = sorted((k for k in sd if k.endswith(".weight")), key=lambda k: int(k.split(".")[1]))
        ws = [sd[k] for k in wk]
        self.policy = DiscreteFF(int(ws[0].shape[1]), int(ws[-1].shape[0]),
                                 tuple(int(w.shape[0]) for w in ws[:-1]), "cpu")
        self.policy.load_state_dict(sd); self.policy.eval()

    def act(self, obs) -> int:
        with self._t.no_grad():
            a, _ = self.policy.get_action(
                self._t.as_tensor(obs, dtype=self._t.float32).unsqueeze(0), deterministic=False)
        return int(np.asarray(a).flatten()[0])


from rlgym_sim.utils.obs_builders import ObsBuilder


class _AsymObs(ObsBuilder):
    """blue -> DefaultPlusObs(107); orange -> current rival's obs (rotates via .idx)."""
    def __init__(self, blue, oranges):
        super().__init__()
        self.blue = blue
        self.oranges = oranges
        self.idx = 0
    def reset(self, state):
        self.blue.reset(state)
        self.oranges[self.idx].reset(state)
    def build_obs(self, player, state, prev):
        if player.team_num == 0:
            return self.blue.build_obs(player, state, prev)
        return self.oranges[self.idx].build_obs(player, state, prev)
    def get_obs_space(self):
        return None


class _CounterEnv:
    """Presents ONLY blue to the Learner; drives orange with a rotating frozen rival."""
    def __init__(self, env, asym, opp_ckpts, weights=None, seed=0):
        self.env = env; self.asym = asym
        self.opp_ckpts = opp_ckpts
        self.weights = weights
        self._frozen = {}                 # idx -> _Frozen (lazy, cached)
        self._orange_obs = None
        self._cur = 0
        self._rng = random.Random(seed)
    @property
    def observation_space(self):
        from gym.spaces import Box
        return Box(low=-np.inf, high=np.inf, shape=(107,), dtype=np.float32)
    def reset(self, *a, **k):
        if self.weights:
            self._cur = self._rng.choices(range(len(self.opp_ckpts)), weights=self.weights, k=1)[0]
        else:
            self._cur = self._rng.randrange(len(self.opp_ckpts))
        self.asym.idx = self._cur
        if self._cur not in self._frozen:
            self._frozen[self._cur] = _Frozen(self.opp_ckpts[self._cur])
        obs = self.env.reset(*a, **k)
        self._orange_obs = np.asarray(obs[1], dtype=np.float32)
        return [obs[0]]
    def step(self, actions):
        blue = actions[0] if isinstance(actions, (list, tuple, np.ndarray)) else actions
        blue = np.asarray(blue, dtype=np.int64).reshape(-1)[:1]
        d = self._frozen[self._cur].act(self._orange_obs)
        obs, rew, done, info = self.env.step([blue, np.asarray([d], dtype=np.int64)])
        self._orange_obs = np.asarray(obs[1], dtype=np.float32)
        r0 = rew[0] if isinstance(rew, (list, tuple, np.ndarray)) else rew
        return [obs[0]], [r0], done, info
    def __getattr__(self, n):
        return getattr(self.env, n)


class FieldBuilder:
    """Picklable env factory: builds the rotating counter-env per worker."""
    def __init__(self, opponents, reward_cfg, state_cfg, seed=0):
        self.opponents = opponents
        self.reward_cfg = reward_cfg
        self.state_cfg = state_cfg
        self.seed = seed
    def __call__(self):
        import rlgym_sim
        from rlbot.obs.default_plus_obs import DefaultPlusObs
        from rlbot.rewards import build_reward
        from rlbot.state_setters import build_state_setter
        from rlbot.actions.lookup_action import LookupAction
        from rlgym_sim.utils.terminal_conditions.common_conditions import (
            GoalScoredCondition, NoTouchTimeoutCondition)
        oranges = [make_orange_obs(kind) for (_, _, kind, _) in self.opponents]
        asym = _AsymObs(DefaultPlusObs(), oranges)
        env = rlgym_sim.make(
            tick_skip=8, team_size=1, spawn_opponents=True,
            obs_builder=asym,
            reward_fn=build_reward(self.reward_cfg),
            state_setter=build_state_setter(self.state_cfg),
            terminal_conditions=[GoalScoredCondition(), NoTouchTimeoutCondition(int(10 * 120 / 8))],
            action_parser=LookupAction(),
        )
        return _CounterEnv(env, asym, [c for (_, c, _, _) in self.opponents],
                           weights=[w for (_, _, _, w) in self.opponents], seed=self.seed)


def smoke():
    print("[smoke] building rotating counter-env...", flush=True)
    b = FieldBuilder(OPPONENTS, REWARD_CFG, STATE_CFG)
    env = b()
    seen = {}
    for ep in range(6):
        obs = env.reset()
        nm = OPPONENTS[env._cur][0]
        seen[nm] = seen.get(nm, 0) + 1
        assert np.asarray(obs[0]).shape[0] == 107, f"blue obs not 107: {np.asarray(obs[0]).shape}"
        for _ in range(30):
            o, r, done, info = env.step([np.asarray([random.randint(0, 89)], dtype=np.int64)])
            if done:
                break
        print(f"[smoke] ep{ep}: opp={nm}  blue_obs={np.asarray(obs[0]).shape[0]}  last_rew={r[0]:.3f}", flush=True)
    env.close()
    print(f"[smoke] OK — rotation seen={seen}, blue stays 107, frozen rivals drive orange.", flush=True)


def train(resume_ckpt, timestep_limit=400_000_000):
    from rlgym_ppo import Learner
    builder = FieldBuilder(OPPONENTS, REWARD_CFG, STATE_CFG)
    learner = Learner(
        builder,
        n_proc=20, min_inference_size=80, metrics_logger=None,
        ppo_batch_size=100_000, ts_per_iteration=100_000,
        exp_buffer_size=300_000, ppo_minibatch_size=50_000,
        ppo_ent_coef=0.01, ppo_epochs=3, ppo_clip_range=0.2,
        policy_lr=3.0e-5, critic_lr=3.0e-5, gae_lambda=0.95,   # lowered 1e-4->3e-5: finer steps near the 95M peak (avoid overshoot)
        standardize_returns=True, standardize_obs=False,
        save_every_ts=5_000_000, n_checkpoints_to_keep=200,
        timestep_limit=timestep_limit,
        log_to_wandb=False,
        checkpoints_save_folder=SAVE_DIR,
        checkpoint_load_folder=resume_ckpt,       # resume from a 107 expansion checkpoint
        add_unix_timestamp=False,
        policy_layer_sizes=(1024, 1024, 1024),
        critic_layer_sizes=(1024, 1024, 1024),
        render=False,
    )
    print(f"[field] counter-training vs {[o[0] for o in OPPONENTS]} from {resume_ckpt}", flush=True)
    learner.learn()


if __name__ == "__main__":
    if "--smoke" in sys.argv:
        smoke()
    else:
        ckpt = sys.argv[1]
        train(ckpt)
