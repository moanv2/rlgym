"""OPTIONAL cross-check: Behavioural Cloning via the `imitation` library.

This trains BC on the SAME expert demonstrations (imitation/data/*.npy) that the from-scratch
student in il_core.py uses, but with the standard HumanCompatibleAI `imitation` library on a
Stable-Baselines3 policy. It demonstrates that our hand-written BC matches an established
implementation - it is NOT the primary graded artifact (that is the from-scratch version).

RUN IT IN A SEPARATE VENV (see requirements-imitation.txt). Do NOT install `imitation` /
stable-baselines3 / gymnasium into the rl-group-project training env: gymnasium clashes with
the legacy `gym` that rlgym_sim needs, and breaking the env interrupts live PPO training.

    python -m venv .venv-il && . .venv-il/Scripts/activate
    pip install -r imitation/requirements-imitation.txt
    python imitation/bc_imitation_lib.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent / "data"
OBS_DIM, N_ACTIONS = 107, 90


def _require_libs():
    try:
        import gymnasium  # noqa: F401
        import stable_baselines3  # noqa: F401
        from imitation.algorithms import bc  # noqa: F401
        from imitation.data.types import Transitions  # noqa: F401
    except Exception as e:  # pragma: no cover - environment guard
        raise SystemExit(
            "Missing imitation-library deps. Install them in a SEPARATE venv:\n"
            "    pip install -r imitation/requirements-imitation.txt\n"
            f"(import error: {e})"
        )


def main(n_epochs: int = 30, val_frac: float = 0.15, seed: int = 0):
    _require_libs()
    from gymnasium.spaces import Box, Discrete
    from imitation.algorithms import bc
    from imitation.data.types import Transitions

    obs = np.load(DATA_DIR / "observations.npy").astype(np.float32)
    acts = np.load(DATA_DIR / "actions.npy").astype(np.int64)
    n = len(acts)
    print(f"loaded {n} demonstrations from {DATA_DIR}")

    # held-out split for an apples-to-apples accuracy comparison with the from-scratch BC
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = max(1, int(n * val_frac))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    obs_space = Box(low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)
    act_space = Discrete(N_ACTIONS)

    # BC only consumes (obs, acts); next_obs/dones/infos are required by the dataclass but
    # unused by the loss, so we supply valid placeholders.
    tr = Transitions(
        obs=obs[tr_idx],
        acts=acts[tr_idx],
        infos=np.array([{} for _ in tr_idx]),
        next_obs=obs[tr_idx],
        dones=np.zeros(len(tr_idx), dtype=bool),
    )

    trainer = bc.BC(
        observation_space=obs_space,
        action_space=act_space,
        demonstrations=tr,
        rng=np.random.default_rng(seed),
    )
    print(f"training imitation.bc.BC for {n_epochs} epochs...")
    trainer.train(n_epochs=n_epochs)

    preds, _ = trainer.policy.predict(obs[val_idx], deterministic=True)
    acc = float((np.asarray(preds).reshape(-1) == acts[val_idx]).mean())
    print(f"\nimitation-library BC  held-out top-1 accuracy: {acc:.3f}")
    print("(compare with the from-scratch BC in notebook 03_behavioural_cloning.ipynb)")
    return acc


if __name__ == "__main__":
    main()
