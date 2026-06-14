"""Imitation Learning core engine (Behavioural Cloning + DAgger).

This is the shared, tested engine behind the four project notebooks. It implements the
graded "Imitation Learning" half of the rubric on top of our PPO expert.

ENVIRONMENT DEVIATION (approved by Jaume): we run the IL pipeline in 1v1 Rocket League
(rlgym_sim) instead of Walker2D / Ant. The *structure* of the assignment is unchanged:

  - expert policy pi_E  = our champion PPO bot (a fixed BLACK BOX: we only call it to act
    or to label states, never to read its training recipe).
  - student policy pi_theta = a from-scratch MLP trained to imitate the expert.
  - Behavioural Cloning  = supervised learning on expert (obs, action) demonstrations.
  - DAgger               = roll out the student, let the expert relabel the states the
    student actually visits, aggregate, retrain. Beats BC under covariate shift
    (regret O(eps*T) vs BC's O(eps*T^2)).

DISCRETE ACTIONS: the action space is the 90-way LookupAction, so imitation is a
CLASSIFICATION problem (cross-entropy over 90 classes), NOT regression. We never use MSE.

REUSE TRICK: BCStudent mirrors rlgym-ppo's DiscreteFF layer layout exactly (a stack of
Linear+ReLU with a final Linear logits head). Its state_dict therefore loads straight into
DiscreteFF, so a trained student exported with `save_student_checkpoint` plugs into the
repo's existing evaluate.py / tournament.py / make_match_video.py with zero new harness.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Make `src/` importable whether we run from a notebook, a script, or pytest.
_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rlbot.env import make_env_builder  # noqa: E402
from rlbot.evaluation.evaluate import _load_policy, _resolve_checkpoint  # noqa: E402
from rlbot.utils.config import load_config  # noqa: E402

# ----------------------------------------------------------------------------- constants
OBS_DIM = 107          # AdvancedObs
N_ACTIONS = 90         # LookupAction
EXPERT_CKPT = str(_HERE.parent / "martin-bots" / "checkpoints" / "CHAMPION_3.37B_advanced1024")
EVAL_CONFIG = str(_HERE.parent / "configs" / "experiments" / "exp_003_long_run.yaml")
DATA_DIR = str(_HERE / "data")
ARTIFACT_DIR = str(_HERE / "artifacts")
REPO_ROOT = str(_HERE.parent)


def pick_device(prefer_gpu: bool = True) -> str:
    return "cuda" if (prefer_gpu and torch.cuda.is_available()) else "cpu"


def use_repo_cwd() -> str:
    """RocketSim looks for ./collision_meshes/ in the current working directory. Chdir to the
    repo root (where it lives) so envs build no matter where the kernel/script started -
    notebooks run from imitation/notebooks/, scripts may run from anywhere."""
    os.chdir(REPO_ROOT)
    return REPO_ROOT


# ============================================================================= expert
class ExpertPolicy:
    """Thin black-box wrapper around the champion DiscreteFF.

    We only ever call `.act` (the oracle's chosen action) and `.action_probs` (its full
    distribution, used for optional soft-label diagnostics). We never inspect its weights
    or training recipe, exactly as DAgger treats its expert.
    """

    def __init__(self, ckpt_path: str = EXPERT_CKPT, device: str = "cpu"):
        self.device = device
        self.policy = _load_policy(_resolve_checkpoint(Path(ckpt_path)), OBS_DIM, N_ACTIONS, device)
        self.policy.eval()

    @torch.no_grad()
    def act(self, obs, deterministic: bool = True) -> int:
        return int(self.policy.get_action(np.asarray(obs, dtype=np.float32), deterministic=deterministic)[0])

    @torch.no_grad()
    def act_batch(self, obs_batch: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Label a whole batch of observations in one forward pass (utility; the per-step rollout
        loop in rollout_student_relabel labels one state at a time, so this is for offline relabelling)."""
        probs = self.policy.get_output(np.asarray(obs_batch, dtype=np.float32)).view(-1, N_ACTIONS)
        if deterministic:
            return probs.argmax(dim=-1).cpu().numpy().astype(np.int64)
        return torch.multinomial(torch.clamp(probs, 1e-11, 1), 1).flatten().cpu().numpy().astype(np.int64)

    @torch.no_grad()
    def action_probs(self, obs) -> np.ndarray:
        return self.policy.get_output(np.asarray(obs, dtype=np.float32)).view(-1, N_ACTIONS).cpu().numpy()[0]


# ============================================================================= student
class BCStudent(nn.Module):
    """From-scratch MLP classifier: AdvancedObs(107) -> logits over 90 LookupActions.

    Layer layout mirrors DiscreteFF (Linear+ReLU stack + final Linear) but stops at the
    LOGITS (no softmax) so we can train with nn.CrossEntropyLoss. Because the softmax in
    DiscreteFF carries no parameters, this student's state_dict loads 1:1 into a DiscreteFF
    for evaluation with the existing harness.
    """

    def __init__(self, hidden_sizes=(256, 256), obs_dim: int = OBS_DIM, n_actions: int = N_ACTIONS):
        super().__init__()
        self.hidden_sizes = tuple(hidden_sizes)
        layers: list[nn.Module] = [nn.Linear(obs_dim, hidden_sizes[0]), nn.ReLU()]
        prev = hidden_sizes[0]
        for size in hidden_sizes[1:]:
            layers += [nn.Linear(prev, size), nn.ReLU()]
            prev = size
        layers.append(nn.Linear(prev, n_actions))  # logits head (no softmax -> CE loss)
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    @torch.no_grad()
    def act(self, obs, deterministic: bool = True) -> int:
        logits = self.forward(torch.as_tensor(np.asarray(obs, dtype=np.float32),
                                               device=next(self.parameters()).device)).view(-1, N_ACTIONS)
        if deterministic:
            return int(logits.argmax(dim=-1).item())
        probs = torch.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, 1).item())


def save_student_checkpoint(student: BCStudent, out_dir: str) -> str:
    """Write PPO_POLICY.pt so the repo's evaluate/tournament/video tooling can load the student."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Save the BCStudent state_dict: its Sequential is named `self.model`, so keys are
    # `model.0.weight`, `model.2.weight`, ... -- identical to DiscreteFF, which lets
    # evaluate._load_policy parse layer indices from key.split(".")[1] and load it directly.
    cpu_sd = {k: v.detach().cpu() for k, v in student.state_dict().items()}
    torch.save(cpu_sd, out / "PPO_POLICY.pt")
    (out / "META.json").write_text(json.dumps(
        {"kind": "bc_student", "hidden_sizes": list(student.hidden_sizes),
         "obs_dim": OBS_DIM, "n_actions": N_ACTIONS, "obs": "AdvancedObs", "action": "LookupAction"}, indent=2))
    return str(out)


# ============================================================================= env
def build_env(config_path: str = EVAL_CONFIG, max_seconds: int = 60, randomize: bool = True):
    """1v1 env. randomize=True keeps the config's weighted_sample setter (70% random states,
    good demonstration coverage); randomize=False forces a clean kickoff (fair eval / video)."""
    full = load_config(config_path).to_dict()
    if not randomize:
        full["state_setter"] = {"name": "default"}
    full["terminal"]["timeout_seconds"] = int(max_seconds)
    env_cfg = dict(full["env"])
    env_cfg["team_size"] = 1
    env_cfg["spawn_opponents"] = True
    return make_env_builder(env_cfg, full)()


def _as_list(x):
    return x if isinstance(x, list) else [x]


# ============================================================================= data collection
@dataclass
class DemoBuffer:
    """Aggregating dataset of (obs, expert_action) pairs. Grows across DAgger rounds.

    `episode_ids[i]` is the episode each frame came from. This is essential for an honest
    train/val split: frames are collected sequentially and at tick_skip=8 (~15 Hz) consecutive
    frames are near-identical, so a *random* frame split leaks near-duplicate neighbours across
    train and val and inflates accuracy. Always split by whole EPISODES (see split_by_episode).
    """
    observations: np.ndarray = field(default_factory=lambda: np.zeros((0, OBS_DIM), np.float32))
    actions: np.ndarray = field(default_factory=lambda: np.zeros((0,), np.int64))
    episode_ids: np.ndarray = field(default_factory=lambda: np.zeros((0,), np.int64))
    episode_returns: list = field(default_factory=list)
    episode_lengths: list = field(default_factory=list)

    def add(self, obs, acts, ep_id: int = -1):
        obs = np.asarray(obs, np.float32).reshape(-1, OBS_DIM)
        acts = np.asarray(acts, np.int64).reshape(-1)
        self.observations = np.concatenate([self.observations, obs], axis=0)
        self.actions = np.concatenate([self.actions, acts], axis=0)
        self.episode_ids = np.concatenate([self.episode_ids, np.full(len(acts), ep_id, np.int64)])

    def __len__(self):
        return len(self.actions)

    def save(self, out_dir: str):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "observations.npy", self.observations)
        np.save(out / "actions.npy", self.actions)
        np.save(out / "episode_ids.npy", self.episode_ids)
        np.save(out / "episode_returns.npy", np.asarray(self.episode_returns, np.float32))
        np.save(out / "episode_lengths.npy", np.asarray(self.episode_lengths, np.int64))
        return str(out)

    @classmethod
    def load(cls, out_dir: str) -> "DemoBuffer":
        out = Path(out_dir)
        b = cls()
        b.observations = np.load(out / "observations.npy")
        b.actions = np.load(out / "actions.npy")
        ep_path = out / "episode_ids.npy"
        b.episode_ids = np.load(ep_path) if ep_path.exists() else np.zeros(len(b.actions), np.int64)
        b.episode_returns = list(np.load(out / "episode_returns.npy"))
        b.episode_lengths = list(np.load(out / "episode_lengths.npy"))
        return b


def split_by_episode(buf: "DemoBuffer", val_frac: float = 0.2, seed: int = 0):
    """Hold out whole EPISODES for validation (no within-episode frame leakage).

    Returns (train_idx, val_idx) into the buffer's frames. Use a single such split as a fixed
    yardstick across an ablation so every model is judged on the same unseen episodes."""
    eps = np.unique(buf.episode_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(eps)
    n_val = max(1, int(len(eps) * val_frac))
    val_eps = set(eps[:n_val].tolist())
    is_val = np.array([e in val_eps for e in buf.episode_ids])
    return np.where(~is_val)[0], np.where(is_val)[0]


def collect_expert_demos(expert: ExpertPolicy, n_episodes: int, *, max_seconds: int = 60,
                         randomize: bool = True, seed: int | None = 0, env=None,
                         progress=None) -> DemoBuffer:
    """Roll out the expert and record (obs, expert_action) demonstrations.

    Both cars are driven by the champion. The recorded label is ALWAYS the deterministic
    (argmax) expert action for that observation. To diversify the states the expert faces,
    the orange car *executes* stochastic samples while still being recorded with its argmax
    label; the blue car executes argmax (a clean expert trajectory). Combined with the
    randomized start states this gives broad, on-expert-distribution coverage.
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    own_env = env is None
    if own_env:
        env = build_env(max_seconds=max_seconds, randomize=randomize)
    buf = DemoBuffer()
    try:
        for ep in range(n_episodes):
            obs, info = env.reset(return_info=True)
            obs = _as_list(obs)
            teams = [int(p.team_num) for p in info["state"].players]
            done, ret, steps = False, 0.0, 0
            while not done:
                acts = []
                for i, o in enumerate(obs):
                    argmax_a = expert.act(o, deterministic=True)
                    buf.add(o, argmax_a, ep_id=ep)                # record argmax label + episode id for every car
                    exec_a = argmax_a if teams[i] == 0 else expert.act(o, deterministic=False)
                    acts.append([exec_a])
                obs, _, done, info = env.step(np.array(acts))
                obs = _as_list(obs)
                ret = info.get("result", ret)
                steps += 1
            buf.episode_returns.append(float(ret))
            buf.episode_lengths.append(int(steps))
            if progress:
                progress(ep + 1, n_episodes, len(buf))
    finally:
        if own_env and hasattr(env, "close"):
            env.close()
    return buf


# ============================================================================= BC training
def train_bc(student: BCStudent, X: np.ndarray, y: np.ndarray, *, epochs: int = 40,
             batch_size: int = 512, lr: float = 1e-3, val_frac: float = 0.1,
             val_data: tuple | None = None,
             weight_decay: float = 0.0, device: str | None = None, seed: int = 0,
             verbose: bool = False) -> dict:
    """Supervised cross-entropy training of the student on (obs, expert_action) pairs.

    Returns a history dict (train_loss, val_loss, val_acc, val_top3 per epoch) plus the
    final metrics. Validation = action-agreement on held-out EXPERT states.

    Pass `val_data=(Xval, yval)` to evaluate on an EXTERNAL held-out set (the honest, leak-free
    way: hold out whole episodes via split_by_episode and train on all of X, y). If omitted,
    falls back to an internal RANDOM frame split (`val_frac`) -- convenient for quick checks but
    it leaks correlated neighbouring frames, so it OVERESTIMATES accuracy on sequential data.
    """
    device = device or pick_device()
    Xt = torch.as_tensor(X, dtype=torch.float32)
    yt = torch.as_tensor(y, dtype=torch.long)
    if val_data is not None:
        Xtr, ytr = Xt.to(device), yt.to(device)
        Xval = torch.as_tensor(val_data[0], dtype=torch.float32).to(device)
        yval = torch.as_tensor(val_data[1], dtype=torch.long).to(device)
        n_tr, n_val = len(ytr), len(yval)
    else:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(y))
        n_val = max(1, int(len(y) * val_frac))
        val_idx, tr_idx = idx[:n_val], idx[n_val:]
        Xtr, ytr = Xt[tr_idx].to(device), yt[tr_idx].to(device)
        Xval, yval = Xt[val_idx].to(device), yt[val_idx].to(device)
        n_tr = len(tr_idx)

    student = student.to(device)
    opt = torch.optim.Adam(student.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    hist = {"train_loss": [], "val_loss": [], "val_acc": [], "val_top3": []}

    for ep in range(epochs):
        student.train()
        perm = torch.randperm(len(ytr), device=device)
        running = 0.0
        for b in range(0, len(ytr), batch_size):
            bi = perm[b:b + batch_size]
            opt.zero_grad()
            logits = student(Xtr[bi])
            loss = loss_fn(logits, ytr[bi])
            loss.backward()
            opt.step()
            running += loss.item() * len(bi)
        student.eval()
        with torch.no_grad():
            vl = student(Xval)
            vloss = loss_fn(vl, yval).item()
            top1 = (vl.argmax(dim=-1) == yval).float().mean().item()
            top3 = (vl.topk(3, dim=-1).indices == yval.unsqueeze(1)).any(dim=1).float().mean().item()
        hist["train_loss"].append(running / max(1, len(ytr)))
        hist["val_loss"].append(vloss)
        hist["val_acc"].append(top1)
        hist["val_top3"].append(top3)
        if verbose and (ep % max(1, epochs // 10) == 0 or ep == epochs - 1):
            print(f"  epoch {ep + 1:3d}/{epochs}  train {hist['train_loss'][-1]:.3f}  "
                  f"val {vloss:.3f}  acc {top1:.3f}  top3 {top3:.3f}", flush=True)
    student.to("cpu")
    return {"history": hist, "val_acc": hist["val_acc"][-1], "val_top3": hist["val_top3"][-1],
            "val_loss": hist["val_loss"][-1], "n_train": int(n_tr), "n_val": int(n_val)}


# ============================================================================= DAgger
def rollout_student_relabel(student: BCStudent, expert: ExpertPolicy, n_episodes: int, *,
                            beta: float = 0.0, max_seconds: int = 60, randomize: bool = True,
                            opponent: ExpertPolicy | None = None, seed: int | None = None, env=None):
    """One DAgger data-gathering pass.

    The STUDENT controls the blue car (mixed with the expert at probability `beta` for the
    classic DAgger beta-mixing; beta=0 = pure student control). At every blue state we ask
    the EXPERT for the correct label. The orange car is driven by `opponent` (default: the
    expert) so the student faces a strong, consistent adversary.

    Returns (new_obs, new_labels, agreement, mean_return): the relabelled dataset to
    aggregate, plus the student/expert action-agreement on STUDENT-visited states (the
    covariate-shift metric DAgger is meant to fix) and the mean episode result.
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    opponent = opponent or expert
    own_env = env is None
    if own_env:
        env = build_env(max_seconds=max_seconds, randomize=randomize)
    obs_list, lbl_list = [], []
    agree_hits, agree_tot = 0, 0
    returns = []
    try:
        for _ in range(n_episodes):
            obs, info = env.reset(return_info=True)
            obs = _as_list(obs)
            teams = [int(p.team_num) for p in info["state"].players]
            done, ret = False, 0.0
            while not done:
                acts = []
                for i, o in enumerate(obs):
                    if teams[i] == 0:  # blue = student under DAgger
                        exp_a = expert.act(o, deterministic=True)
                        obs_list.append(np.asarray(o, np.float32))
                        lbl_list.append(exp_a)
                        stu_a = student.act(o, deterministic=True)
                        agree_hits += int(stu_a == exp_a)
                        agree_tot += 1
                        # beta-mixing: occasionally defer to the expert early on
                        take = exp_a if (beta > 0 and np.random.rand() < beta) else stu_a
                        acts.append([take])
                    else:              # orange = fixed strong opponent
                        acts.append([opponent.act(o, deterministic=True)])
                obs, _, done, info = env.step(np.array(acts))
                obs = _as_list(obs)
                ret = info.get("result", ret)
            returns.append(float(ret))
    finally:
        if own_env and hasattr(env, "close"):
            env.close()
    new_obs = np.asarray(obs_list, np.float32).reshape(-1, OBS_DIM) if obs_list else np.zeros((0, OBS_DIM), np.float32)
    new_lbl = np.asarray(lbl_list, np.int64)
    agreement = agree_hits / max(1, agree_tot)
    return new_obs, new_lbl, agreement, float(np.mean(returns) if returns else 0.0)


def evaluate_student_winrate(student: BCStudent, opponent_ckpt: str, *, episodes: int = 30,
                             deterministic: bool = True, tmp_dir: str | None = None) -> dict:
    """Export the student to a PPO_POLICY.pt and reuse the repo's evaluate harness to play it
    head-to-head vs a fixed opponent checkpoint. Returns the evaluate() metrics dict."""
    from rlbot.evaluation.evaluate import evaluate
    tmp_dir = tmp_dir or str(_HERE / "artifacts" / "_tmp_student_eval")
    save_student_checkpoint(student, tmp_dir)
    return evaluate(tmp_dir, opponent_ckpt, episodes, deterministic, EVAL_CONFIG)


# ============================================================================= smoke test
def smoke_test():
    """Tiny end-to-end check: load expert, collect 1 ep, fit BC briefly, run 1 DAgger pass."""
    print("[smoke] loading expert...", flush=True)
    expert = ExpertPolicy(device="cpu")
    print("[smoke] collecting 1 demo episode...", flush=True)
    buf = collect_expert_demos(expert, 1, max_seconds=8, seed=0)
    print(f"[smoke] collected {len(buf)} (obs,action) pairs over {len(buf.episode_lengths)} ep", flush=True)
    assert buf.observations.shape[1] == OBS_DIM and len(buf) > 0
    student = BCStudent(hidden_sizes=(128, 128))
    res = train_bc(student, buf.observations, buf.actions, epochs=5, batch_size=256, verbose=True)
    print(f"[smoke] BC val_acc={res['val_acc']:.3f}", flush=True)
    print("[smoke] 1 DAgger relabel pass...", flush=True)
    no, nl, agree, mret = rollout_student_relabel(student, expert, 1, max_seconds=8, seed=1)
    print(f"[smoke] dagger pass: +{len(nl)} labels, agreement={agree:.3f}, mean_return={mret:.2f}", flush=True)
    print("[smoke] OK", flush=True)


if __name__ == "__main__":
    smoke_test()
