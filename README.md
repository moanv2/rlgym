# rlgym — a 1v1 Rocket League bot

> Final Project for Reinforcement Learning @ IE School of Science and Technology

A modular, MLOps-disciplined Rocket League bot trained with **RLGym-PPO + rlgym_sim**,
built for a 1v1 final-class showdown. The codebase covers the full lifecycle: a
reproducible PPO training stack, headless bot-vs-bot evaluation, a live `rlviser`
visualizer, and a **single-elimination tournament** harness that ranks the team's
bots and records match videos. 

The flagship bot, **papaya_1024**, is a 1024×3 network trained on AdvancedObs to
~3.5B steps. Earlier milestones (a 512×3 DefaultObs line up to ~1.3B) are preserved
for comparison. See the [training journey](#the-bots-training-journey) below.

---

## Setup (Windows 11 / Linux)

Everything runs in one Python **3.10 or 3.11** environment (rlgym-ppo doesn't support
3.12+ cleanly). A conda env is the smoothest path on Windows:

```bash
# 1. Create and activate the environment
conda create -n rlbot310 python=3.10 -y
conda activate rlbot310

# 2. Install PyTorch FIRST (pick the right wheel for your machine)
#    NVIDIA GPU (CUDA 11.8):
pip install torch --index-url https://download.pytorch.org/whl/cu118
#    CPU-only (training ~10x slower, but fine for eval/visualizing):
#    pip install torch --index-url https://download.pytorch.org/whl/cpu

# 3. Install the project + its RL stack
pip install -e ".[dev]"          # the rlbot package + dev tools (pytest, ruff, mypy)
pip install -r requirements.txt  # pulls rocketsim, rlgym_sim, rlgym-ppo, rlgym-tools from git

# 4. Visualizer (to watch bots play)
pip install rlviser-py
#    then download the rlviser binary from https://github.com/VirxEC/rlviser/releases
#    and keep rlviser.exe in the repo root.
```

**Collision meshes (required by rlgym_sim).** Rocket League's arena collision data
can't be redistributed, so dump it once from a local RL install using
[RLArenaCollisionDumper](https://github.com/ZealanL/RLArenaCollisionDumper/releases/tag/v1.0.0)
and drop the resulting `collision_meshes/` folder in the repo root. It's gitignored —
each developer dumps their own.

Full first-time walkthrough: **[docs/setup.md](docs/setup.md)**.

Verify the install:

```bash
make test-fast      # unit + smoke tests (skips slow/rocketsim/gpu-marked)
```

---

## Quickstart

```bash
conda activate rlbot310

# Train the baseline experiment
python -m rlbot.training.train --config configs/experiments/exp_001_baseline.yaml

# Watch a checkpoint play (open rlviser.exe first)
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

> **Checkpoints are gitignored** (`*.pt`), so a fresh clone has none — you produce them
> by training, or copy a teammate's checkpoint folder (`PPO_POLICY.pt` +
> `BOOK_KEEPING_VARS.json`) into place.

---

## Training

Every run is defined by a single YAML in `configs/experiments/`, so it's fully
reproducible (pinned deps + seed + git SHA + wandb run id are recorded with each
checkpoint). Three reference experiments ship in the repo:

| Config | What it is |
|--------|-----------|
| `exp_001_baseline.yaml`     | Minimal reward, small net — the starting point |
| `exp_002_advanced_obs.yaml` | AdvancedObs (107-dim) observation upgrade |
| `exp_003_long_run.yaml`     | Long training run config |

```bash
python -m rlbot.training.train --config configs/experiments/exp_003_long_run.yaml
# or:  make train EXP=exp_003_long_run
```

The training stack is fully modular — reward functions, observation builders, action
parsers, state setters, and terminal conditions are independent components under
`src/rlbot/`, composed by the env builder. See **[docs/architecture.md](docs/architecture.md)**.

---

## Evaluating & watching bots

```bash
# Headless win-rate eval (no visualizer needed). `latest:<exp>` auto-picks newest ckpt.
python -m rlbot.evaluation.evaluate --blue <ckptA> --orange <ckptB> --episodes 100 --deterministic

# Live 1v1 in rlviser (real-time). Open rlviser.exe first.
python scripts/visualize.py --checkpoint <ckpt>
```

Result signal: a match ends on a goal or timeout, and `info["result"]` is the signed
goal differential (`+` blue, `−` orange, `0` draw) — the same metric the tournament
uses to score games.

---

## Tournament & match videos

The `rlbot.tournament` package runs a **seeded single-elimination bracket** across the
team's bots, ranks them 1..N, and records a ~60s rlviser clip of each bot's bracket
match for the presentation. It auto-detects each checkpoint's architecture and
observation size from its weights, so bots trained on **DefaultObs (89-dim)** and
**AdvancedObs (107-dim)** can play in the same match (each car is fed the obs it was
trained on).

```bash
conda activate rlbot310

# 0. (optional) fetch teammates' checkpoints listed in the manifest
python -m rlbot.tournament.download --list      # show status
python -m rlbot.tournament.download             # fetch all with a source set

# 1. Run the bracket (headless) -> writes ranking JSON + prints a 1..N table
python -m rlbot.tournament.run                  # best-of-5, deterministic
#    -> history_and_summary/tournament_results.json

# 2. Record each bot's bracket match (open rlviser.exe first)
python -m rlbot.tournament.record --all --capture
#    -> videos/<owner>_bracket.mp4   (uses ffmpeg if present, else prints the OBS command)
```

The roster lives in `src/rlbot/tournament/roster.py`; teammate checkpoint sources go in
`src/rlbot/tournament/manifest.json`. Missing teammates are skipped until downloaded.
Details: **[src/rlbot/tournament/README.md](src/rlbot/tournament/README.md)**.

---

## Project layout

```
rlgym/
├── configs/experiments/    # one YAML per run — reproducible training
├── src/rlbot/              # importable package (pip install -e .)
│   ├── env/                # rlgym_sim env builder
│   ├── obs/                # observation builders (DefaultObs, AdvancedObs)
│   ├── actions/            # action parsers (LookupAction, 90 discrete)
│   ├── rewards/            # reward fns, ZeroSumReward wrapper, Nexto-style combos
│   ├── state_setters/      # kickoff / random / curriculum initial states
│   ├── terminal/           # episode end conditions (goal, timeout, kickoff-stall)
│   ├── models/             # policy network customization
│   ├── training/           # PPO training entrypoint + callbacks
│   ├── evaluation/         # headless bot-vs-bot eval, win-rate metrics
│   ├── deployment/         # RLBot-compatible export
│   ├── tournament/         # single-elim bracket, ranking, video capture, downloads
│   └── utils/              # config loader, seeding, logging
├── scripts/               # thin CLI wrappers (visualize, evaluate, export, ...)
├── tests/                 # pytest unit + smoke tests
├── teammates/             # teammate checkpoints (gitignored weights)
├── videos/                # recorded match clips (gitignored)
├── checkpoints/           # gitignored — model snapshots
├── docs/                  # setup, architecture, training guide
└── rlviser.exe            # visualizer binary (gitignored; download separately)
```

---

## The bots (training journey)

The project iterated through several reward/architecture generations — a useful story
for the writeup:

1. **baseline** — tiny net, minimal reward; plateaued early (entropy ceiling).
2. **nexto_rewards** — 10-component Nexto-style reward; broke the plateau.
3. **nexto_plus_kickoff_512** — 512×3 net, DefaultObs, dedicated kickoff reward;
   reached ~1.18B steps and was the prior champion.
4. **papaya_1024** — 1024×3 net on **AdvancedObs (107-dim)**; reward tuned across
   v4→v7 (fixed boost-dumping/overcommitting, optimizer tuning, a fast-kickoff suite).
   Trained to ~3.5B steps — the current flagship.

Trained weights aren't committed (`*.pt` is gitignored); they're produced by training
or shared directly between teammates.

---

## MLOps principles applied

| Principle           | How                                                                   |
|---------------------|-----------------------------------------------------------------------|
| Reproducibility     | Pinned deps, every run defined by one YAML, deterministic seeds        |
| Modularity          | Reward / obs / action / state-setter components are independent        |
| Versioning          | Checkpoints saved with their config + git SHA + wandb run id           |
| Experiment tracking | wandb integration; every iteration's metrics are logged                |
| Testing             | Unit tests on rewards, configs, env build, tournament logic; CI on push|
| CI/CD               | GitHub Actions: lint (ruff), type-check (mypy), tests (pytest)         |

---

## Useful commands

```bash
make install       # editable install with dev extras
make test          # full pytest suite
make test-fast     # skip slow/rocketsim/gpu-marked tests
make lint          # ruff + mypy
make format        # ruff format + autofix
make train EXP=exp_003_long_run
make eval BLUE=<ckpt> ORANGE=<ckpt>
make clean         # wipe caches/build artifacts
```
