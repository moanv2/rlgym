# Setup

End-to-end first-time install on Windows 11.

## 1. Python + tooling

1. Install Python **3.10 or 3.11** (rlgym-ppo doesn't support 3.12+ cleanly yet).
   Make sure "Add Python to PATH" is checked.
2. Install Git: https://git-scm.com/downloads

## 2. PyTorch

If you have an NVIDIA GPU:

1. Install [CUDA 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive).
2. Install the CUDA-matched PyTorch wheel:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   ```

CPU-only fallback (training will be ~10x slower):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## 3. Project deps

From the `rlgym/` folder:

```bash
pip install -e ".[dev]"
pip install -r requirements.txt
```

`-r requirements.txt` is what pulls `rocketsim`, `rlgym_sim`, `rlgym-ppo`, and
`rlgym-tools` from git.

## 4. Collision meshes (required by rlgym_sim)

`rlgym_sim` needs Rocket League's arena collision data, which can't be redistributed.
Dump it from a local Rocket League install:

1. Download the **RLArenaCollisionDumper** release from
   https://github.com/ZealanL/RLArenaCollisionDumper/releases/tag/v1.0.0
2. Follow its README — it produces a `collision_meshes/` folder.
3. Move that folder into the `rlgym/` project root (next to `pyproject.toml`).

`collision_meshes/` is gitignored, so each developer dumps their own.

## 5. wandb (optional but recommended)

```bash
make wandb-login
```

If you skip this, set `logging.wandb: false` in your experiment config.

## 6. Smoke test

```bash
make test-fast
python -m rlbot.training.train --config configs/experiments/exp_001_baseline.yaml --dry-run
```

Both should pass before you commit to a long training run.

## 7. Visualizer (later)

The `rlviser` visualizer lets you watch the bot play. Install it when you reach
[docs/roadmap_45_days.md](roadmap_45_days.md) week 2:

```bash
pip install rlviser-py
```

Then download the visualizer binary from https://github.com/VirxEC/rlviser/releases
and run it before launching `scripts/visualize.py`.
