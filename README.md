# rlgym, a 1v1 Rocket League bot

> Final project for Reinforcement Learning at IE School of Science and Technology.

## 1. Objective

Train a Rocket League bot to play 1v1 using PPO, which stands for Proximal Policy Optimization. The agent learns entirely from self play inside a headless physics simulator. It sees the game as a vector of numbers, picks one of 90 discrete controller actions many times per second, and gets better by playing billions of simulated steps against copies of itself. This was a team project, so each person trained their own bot and the strongest ones met in a final tournament.

## 2. Setup

This is the full walkthrough. It covers installing everything on a clean Windows 11 machine, running one of the bots, and what to do when rlviser is not installed.

### What you need first

A Windows 11 machine and Miniconda or Anaconda. An NVIDIA GPU makes training much faster but is optional. Everything runs in one Python 3.10 environment, because the rlgym_ppo trainer does not support Python 3.12 and newer cleanly.

### Fresh Windows install, step by step

1. Install Miniconda from the official website and open the Anaconda Prompt.
2. Clone the repository and move into it.

```bash
git clone https://github.com/moanv2/rlgym.git
cd rlgym
```

3. Create and activate the environment.

```bash
conda create -n rlbot310 python=3.10 -y
conda activate rlbot310
```

4. Install PyTorch first, choosing the build for your machine.

```bash
# NVIDIA GPU with CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118
# CPU only, which is slower for training but fine for watching and evaluating
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

5. Install the project and the reinforcement learning stack.

```bash
pip install -e ".[dev]"
pip install -r requirements.txt
```

6. Add the arena collision meshes. Rocket League collision data cannot be shared, so each person dumps it once from a local Rocket League install using RLArenaCollisionDumper, then drops the resulting collision_meshes folder into the repository root.
7. Check that the install works.

```bash
make test-fast
```

### Run one of the bots

To watch a trained bot play, open rlviser and run the visualizer on a checkpoint folder.

```bash
conda activate rlbot310
python scripts/visualize.py --checkpoint checkpoints/<experiment>/latest
```

A checkpoint is a folder that holds `PPO_POLICY.pt` and `BOOK_KEEPING_VARS.json`. Trained weights are never committed, so a fresh clone has none. You make them by training, or you copy a teammate checkpoint folder into place. Each person shares their own.

### If rlviser is not installed

rlviser is the live visualizer and you only need it to watch a bot play. If it is missing you have two choices.

First choice, install it. Install the Python bindings, then download the binary.

```bash
pip install rlviser-py
```

Download the rlviser executable from the VirxEC rlviser releases page and keep `rlviser.exe` in the repository root. Open it before you run the visualizer. The full guide lives in `rlvisor_claude_setup.md`.

Second choice, skip the visualizer entirely. You can evaluate two bots fully headless and read the win rate in the terminal.

```bash
python -m rlbot.evaluation.evaluate --blue <checkpointA> --orange <checkpointB> --episodes 100 --deterministic
```

## 3. Project layout

```
rlgym/
  configs/experiments/   one YAML file per run for reproducible training
  src/rlbot/             the importable package (pip install -e .)
    env/                 the rlgym_sim environment builder
    obs/                 observation builders (DefaultObs, AdvancedObs)
    actions/             action parsers (LookupAction, 90 discrete moves)
    rewards/             reward functions and the ZeroSumReward wrapper
    state_setters/       where each episode starts (kickoff, random, drills)
    terminal/            episode end conditions (goal, timeout, kickoff stall)
    models/              policy and value network configuration
    training/            the PPO training entry point
    evaluation/          headless bot versus bot win rate evaluation
    tournament/          ranking bracket, cross obs matches, video capture
    utils/               config loader, seeding, logging
  scripts/               command line wrappers (visualize, evaluate)
  tests/                 pytest unit and smoke tests
  teammates/             teammate checkpoints (weights are gitignored)
  videos/                recorded match clips
  checkpoints/           model snapshots (gitignored)
  docs/                  setup, architecture, and training notes
  README.md              this file
```

## 4. The bots of different people

Five people trained bots, each on their own branch. Architecture and observation size are read straight from the saved weights, so bots trained on DefaultObs (89 numbers) and AdvancedObs (107 numbers) can still play each other in the same match.

| Person | Bot | Observation | Network | Steps | Branch |
|---|---|---|---|---|---|
| Diego | papaya_1024 | AdvancedObs 107 | 1024x3 | about 3.5B | `diego` |
| Martin | champion | AdvancedObs 107 | 1024x3 | about 9B | `martin` |
| Nachi | nachi | AdvancedObs 107 | 1024x3 | about 2.9B | `nachi` |
| Marco | exp_007 | DefaultObs 89 | 1024x3 | about 2.0B | `marco` |
| Marian | marian | DefaultObs 89 | 512x3 | about 1.35B | `marian/setup-fixes` |

Each person commits only the policy weights and the book keeping file for their strongest checkpoint. To face a teammate, pull their branch or copy their checkpoint folder into the matching path under the teammates folder.

## 5. Tournament and match videos

The rlbot.tournament package plays the team bots against each other and ranks them. It reads each checkpoint network and observation size from its weights automatically, so DefaultObs and AdvancedObs bots play in one match with each car fed the observation it trained on. Sides swap every game and the kickoff is randomized, so the games stay fair.

The final ranking across 3000 games:

| Rank | Bot | Win rate |
|---|---|---|
| 1 | Martin | 73% |
| 2 | Nachi | 63% |
| 3 | Diego (papaya) | 56% |
| 4 | Marco | 35% |
| 5 | Marian | 23% |

Run the tournament and produce the ranking.

```bash
conda activate rlbot310
# seeded single elimination bracket, best of 5, headless
python -m rlbot.tournament.run
```

Record match videos. Open rlviser first, then capture each bot bracket match to an mp4.

```bash
python -m rlbot.tournament.record --all --capture
```

The roster lives in `src/rlbot/tournament/roster.py` and teammate checkpoint sources go in `src/rlbot/tournament/manifest.json`. Missing teammates are skipped until their checkpoints are present. Videos are written to the videos folder.

## 6. Training

Training uses PPO through the rlgym_ppo library. Many parallel worker processes each run their own copy of the simulator and generate games, while one central learner gathers the experience and updates two networks. The actor picks the action and the critic scores the state. The bot improves by repeating this loop across billions of steps.

Every run is defined by one YAML file in the configs folder, so it is fully reproducible. The seed, the pinned dependencies, the git commit, and the wandb run id are stored with every checkpoint.

```bash
conda activate rlbot310
python -m rlbot.training.train --config configs/experiments/exp_003_long_run.yaml
```

Reference experiments that ship in the repository:

| Config | What it is |
|---|---|
| exp_001_baseline.yaml | A minimal reward and a small network, the starting point |
| exp_002_advanced_obs.yaml | The AdvancedObs 107 observation upgrade |
| exp_003_long_run.yaml | A long training run |

The flagship bot papaya_1024 was trained on the diego branch using a self contained training script that keeps every setting inline and resumes from the latest checkpoint automatically. Check out the diego branch to run it. Training logs to wandb and runs until you stop it, saving a checkpoint on a regular interval so you can stop and resume at any time.
