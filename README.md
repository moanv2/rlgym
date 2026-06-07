# rlgym

> Final Project for Reinforcement Learning @ IE School of Science and Technology

A modular, MLOps-disciplined Rocket League bot trained with **RLGym-PPO + rlgym_sim**, built for a 1v1 final-class showdown.

> **Deadline:** 2026-06-21 (final presentation, 40% of grade) — see [docs/roadmap_45_days.md](docs/roadmap_45_days.md).

---

## Quickstart

```bash
# 1. Install runtime deps (see docs/setup.md for CUDA/RocketSim assets)
pip install -e ".[dev]"

# 2. Dump collision meshes (required by rlgym_sim) — see docs/setup.md
#    Output goes to ./collision_meshes/

# 3. Train the baseline
python -m rlbot.training.train --config configs/experiments/exp_001_baseline.yaml

# 4. Watch the bot in the visualizer
python scripts/visualize.py --checkpoint checkpoints/exp_001_baseline/latest

# 5. Evaluate vs another checkpoint
python scripts/evaluate.py \
  --blue checkpoints/exp_001_baseline/latest \
  --orange checkpoints/exp_000_random/latest \
  --episodes 100

# 6. Run the team tournament — round-robin + Elo across everyone's bots, even when
#    they use different obs builders / net sizes. See docs/TOURNAMENT.md to submit a bot.
python scripts/tournament.py --manifest configs/tournament_bots.yaml --games 30
```

## Layout

```
rlgym/
├── configs/              # YAML configs — every run is reproducible from one
├── src/rlbot/            # importable package (pip install -e .)
│   ├── env/              # rlgym_sim env builder
│   ├── obs/              # observation builders
│   ├── actions/          # action parsers (LookupAction, etc.)
│   ├── rewards/          # reward functions, ZeroSumReward wrapper, combos
│   ├── state_setters/    # curriculum/random initial states
│   ├── terminal/         # episode end conditions
│   ├── models/           # policy network customization
│   ├── training/         # PPO training entrypoint + callbacks
│   ├── evaluation/       # bot-vs-bot eval, win-rate metrics
│   ├── deployment/       # RLBot-compatible export
│   └── utils/            # config loader, seeding, logging
├── scripts/              # thin CLI wrappers
├── tests/                # pytest unit + smoke tests
├── checkpoints/          # gitignored — model snapshots
├── logs/                 # gitignored — local logs
├── data/                 # gitignored
├── docs/                 # setup, architecture, training guide, 45-day plan
└── .github/workflows/    # CI (lint + tests)
```

See [docs/architecture.md](docs/architecture.md) for the design rationale.

## MLOps principles applied

| Principle           | How                                                                   |
|---------------------|-----------------------------------------------------------------------|
| Reproducibility     | Pinned deps, every run defined by one YAML, deterministic seeds       |
| Modularity          | Reward / obs / action / state-setter components are independent       |
| Versioning          | Checkpoints saved with their config + git SHA + wandb run id          |
| Experiment tracking | wandb integration; every iteration's metrics are logged               |
| Testing             | Unit tests on rewards, configs, env build; CI runs on every push      |
| CI/CD               | GitHub Actions: lint (ruff), type-check (mypy), tests (pytest)        |
| Observability       | wandb dashboards + structured local logs                              |

## Useful commands

```bash
make install       # editable install with dev extras
make test          # run pytest
make lint          # ruff + mypy
make format        # ruff format
make train EXP=exp_001_baseline
make eval BLUE=... ORANGE=...
make clean         # wipe __pycache__ etc.
```

## Status

Day 1 of 45. Scaffolding complete; baseline experiment defined; first training run pending.
