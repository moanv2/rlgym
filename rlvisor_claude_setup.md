# rlvisor Claude Setup — Teammate Onboarding Runbook

Hand this whole file to your Claude Code session along with the repo cloned. Claude can then walk you through the exact same setup Diego built, end to end, on your machine. Following every step produces a working environment plus a 500k step "simple bot" trained and visualized locally (Windows) or trained only (Mac).

**This is not optional reading. Skipping a step or substituting a version will break reproducibility and your training results will not be comparable to Diego's.**

---

## Step 0 — Create your own branch FIRST

Before installing anything, branch off `main` so your work is isolated and Diego can tell who did what. Use your first name lowercase as the branch name.

```bash
git checkout main
git pull
git checkout -b <yourname>     # e.g.  git checkout -b jorge
```

Replace `<yourname>` with your actual first name in lowercase: `jorge`, `maria`, `pablo`, etc. **Do not work on `main` directly.** Push at the end with `git push -u origin <yourname>`.

---

## Step 1 — System prerequisites

You need: Python (via conda), Git, and the ability to install C++ extensions.

### Windows (2 of us)
- Install Miniconda: https://docs.conda.io/projects/miniconda/en/latest/
- Install Git: https://git-scm.com/downloads
- Recommended: an NVIDIA GPU with a recent CUDA driver. Training is ~10x faster on GPU. No NVIDIA GPU means CPU fallback, which works but takes hours instead of minutes.

### Mac (1 of us)
- Install Miniconda for Apple Silicon (M1/M2/M3) or Intel: https://docs.conda.io/projects/miniconda/en/latest/
- Install Git via Xcode Command Line Tools:
  ```bash
  xcode-select --install
  ```
- **Important Mac limitations** (read before you commit time):
  - Rocket League has no Mac version, so you cannot dump `collision_meshes/` locally. Diego shares it via Google Drive (see Step 2).
  - There is no Mac build of `rlviser` (the visualizer). You can train, but you cannot watch the bot play locally. You will rely on Diego's Windows machine for visualization, or look at wandb training curves.
  - No NVIDIA GPU on Mac. Training uses CPU or Apple's MPS (slower than CUDA but functional). For long runs, use Google Colab's free T4 GPU instead.

---

## Step 2 — Get `collision_meshes/` from Diego

`collision_meshes/` is a ~10 MB folder of arena collision geometry dumped from Rocket League. It is gitignored on purpose because it is derived from a Psyonix asset and cannot be redistributed publicly.

1. Ask Diego for the Google Drive link to `collision_meshes.zip`.
2. Download and unzip.
3. Place the resulting `collision_meshes/` folder in the project root, next to `pyproject.toml`:

```
rlgym/
├── collision_meshes/        ← here
│   └── soccar/
│       ├── mesh_0.cmf
│       ├── mesh_1.cmf
│       ├── ...
│       └── mesh_15.cmf
├── pyproject.toml
└── src/
```

Verify:
```bash
ls collision_meshes/soccar/   # should list 16 .cmf files
```

**Never `git add` this folder.** It is in `.gitignore` already.

---

## Step 3 — Create the conda env (must match Diego's exactly)

Diego uses Python 3.10.x inside an env named `rlbot310`. Use the same env name and Python major version. The rlgym ecosystem is fragile around Python versions — **do not substitute 3.11, 3.12, or 3.13**. The pyproject pins `<3.12`. `rocketsim` does not have a Python 3.13 wheel.

```bash
conda create -n rlbot310 python=3.10 -y
conda activate rlbot310
python -m pip install --upgrade pip
```

Verify:
```bash
python --version    # must print Python 3.10.x
```

Activate this env every time you work on the project. If your terminal does not show `(rlbot310)` in the prompt, you are in the wrong env.

---

## Step 4 — Install PyTorch with the correct wheel

Pick exactly ONE of the following based on your hardware.

### Windows with NVIDIA GPU (recommended)
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```
This installs `torch 2.5.1+cu121` matching Diego's setup. Works with any NVIDIA driver supporting CUDA 12.x.

### Windows without NVIDIA GPU (CPU fallback)
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Mac Intel or Apple Silicon
```bash
pip install torch
```
The default Mac wheel from PyPI supports MPS on Apple Silicon and CPU on Intel. No `--index-url` needed.

### Verify (Windows NVIDIA only)
```bash
python -c "import torch; print('cuda:', torch.cuda.is_available(), 'device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

Expected: `cuda: True device: NVIDIA GeForce RTX <something>`.

If `cuda: False`, training will fall back to CPU and run much slower. Fine for the simple_bot tutorial; expect longer runtimes.

---

## Step 5 — Install the RL stack

From the project root:

```bash
pip install -r requirements.txt
pip install -e . --no-deps
pip install pytest pytest-cov ruff mypy
```

`-e .` installs our `rlbot` package in editable mode so any code change is picked up immediately. `--no-deps` is important because we already installed deps from `requirements.txt`.

### Mac caveat for rocketsim

`rocketsim==2.2.1` ships prebuilt wheels for Windows and Linux x86_64. Mac wheels are inconsistent. If pip fails with "Could not find a version that satisfies the requirement rocketsim":

1. **Easiest path**: use Google Colab for training (Linux x86_64, wheel works out of the box).
2. **Harder path**: build from source. Requires Xcode + CMake. Not recommended for a class project deadline.
3. **Compromise**: run the env build dry-run on Mac to validate code, do real training on Colab, share checkpoints via Drive.

Verify imports (works on all OSes if rocketsim installed):
```bash
python -c "import torch, rlgym_sim, RocketSim, rlgym_ppo, rlgym_tools; print('all imports ok')"
```

**Note the casing**: the PyPI package is named `rocketsim` but the import name is `RocketSim` (capital R, S). Our code never imports it directly — it goes through `rlgym_sim` — so this is only relevant for sanity checking.

---

## Step 6 — Install rlviser (visualization)

**WINDOWS ONLY.** Mac users skip this step entirely.

There is no Mac build of `rlviser` in the official GitHub release page. If you want to see the bot play and you are on Mac, ask a Windows teammate.

### 6a — Install the Python bindings
```bash
pip install rlviser-py
```

This installs `rlviser-py 0.6.13` (latest as of writing).

### 6b — Download the COMPATIBLE binary (the version trap)

**Critical**: the wire protocol between `rlviser-py` and the `rlviser.exe` binary changed between binary versions `v0.8.x` and `v0.9.x`. `rlviser-py 0.6.13` only works with `v0.8.x` binaries. Using the latest binary release (`v0.9.1` at writing) crashes with `memory allocation of 72057594037927944 bytes failed`. This cost us debugging time and is the entire reason this onboarding doc exists.

**Do this** (PowerShell on Windows):
```powershell
curl -L -o rlviser.exe https://github.com/VirxEC/rlviser/releases/download/v0.8.7/rlviser.exe
```

If `curl` is not available, paste this URL into a browser and save the file as `rlviser.exe` in the project root:
```
https://github.com/VirxEC/rlviser/releases/download/v0.8.7/rlviser.exe
```

Place the file in the project root, next to `pyproject.toml`. `rlviser-py` auto-launches it on the first render call. Do not download anything from the `v0.9.x` releases page.

`rlviser.exe` is gitignored — never commit it.

---

## Step 7 — Verify everything works

### Run the test suite
```bash
pytest -v
```

Expected: **13 tests pass** in a few seconds. The `rocketsim` marked test confirms your `collision_meshes/` is wired up. If that one test fails with a missing-mesh error, recheck Step 2.

### Dry-run the env builder
```bash
python -m rlbot.training.train --config configs/experiments/exp_001_baseline.yaml --dry-run
```

Expected output:
```
Dry run — building env for 'exp_001_baseline'
Env built and reset successfully
```

If both pass, your environment is ready.

---

## Step 8 — Pull the simple_bot scripts from Diego's branch

The educational scripts live in `diego-bots/` on Diego's branch. Copy them into your branch so you have them locally:

```bash
git fetch origin
git checkout origin/diego -- diego-bots/
```

This pulls just the `diego-bots/` folder into your working tree. Commit it onto your own branch so it survives:

```bash
git add diego-bots/
git commit -m "chore: import diego-bots reference scripts onto my branch"
```

Optional: rename to `<yourname>-bots/` if you want to fork and experiment without overlapping Diego's path.

---

## Step 9 — Train your first simple bot

```bash
python diego-bots/simple_bot.py
```

Expected runtime to hit the 500k timestep cap:

| Setup | Approximate time |
|---|---|
| Windows + RTX 4070 class GPU | 10 to 20 min |
| Windows + older NVIDIA GPU (1060, 2060) | 30 to 60 min |
| Windows CPU only | 1 to 3 hours |
| Mac CPU / MPS | 1 to 3 hours |
| Google Colab T4 | 15 to 25 min |

### What to watch in the console

Every iteration prints a report. Key numbers:

- **Policy Reward**: average reward per episode. Should trend upward over iterations. This is the bot getting smarter.
- **Policy Entropy**: starts near `ln(90) ≈ 4.50` (max for 90 discrete actions) and drops over time as the bot commits to specific behavior. Crashing to zero fast is bad.
- **Mean KL Divergence**: should stay small (under 0.05). Spikes mean the policy is moving too aggressively per update.
- **SB3 Clip Fraction**: healthy range is 0.05 to 0.20. Near zero means nothing is updating.
- **Collected Steps per Second**: throughput. Hardware dependent.

Stop any time with Ctrl+C. The Learner saves before exiting. Checkpoints land in `diego-bots/checkpoints/simple_bot-<timestamp>/`.

### What 500k steps looks like

Bad. Entropy at the end is still ~4.48 (near max), KL divergence ~0.001 (almost no movement). The policy is essentially random. That is **expected** and is the educational point: it is your "before" baseline. Future longer runs (10M+ steps) will visibly differ.

---

## Step 10 — Watch the bot play (Windows only)

After at least one checkpoint has saved:

```bash
python diego-bots/simple_bot_play.py
```

The rlviser window auto-launches and shows two cars playing in a Soccar arena. Press Ctrl+C in the terminal (not the rlviser window) to stop. The window closes itself when the Python process exits.

### Useful rlviser controls
- **Game speed slider** → drop to 0.5 or 0.25 to see car behavior clearly. Bots act at 15 Hz, which is fast.
- **Ball cam toggle** → follows ball perspective.
- **Mouse drag** → rotate free camera. Wheel zooms.

Mac users: this step does not work. Either ask a Windows teammate to play your checkpoint, or wait until we wire up wandb video logging.

---

## Step 11 — Push your work to GitHub

When you have something worth sharing:

```bash
git status                                       # sanity check
git add <paths-you-want-to-commit>
git commit -m "feat(<yourname>): describe your change"
git push -u origin <yourname>                    # first push: sets upstream
# subsequent pushes are just: git push
```

If you want your changes merged into `main`, open a Pull Request on GitHub from your branch. Diego will review.

---

## Mac path summary

| Activity | Mac status |
|---|---|
| Clone repo, read code, write code, push commits | Works |
| Run tests | Works |
| Build env via `--dry-run` | Works if rocketsim wheel installs |
| Train short runs locally | Works on CPU or MPS (slow) |
| Train long runs locally | Painful — use Google Colab instead |
| Visualize with rlviser | Does not work — ask a Windows teammate |
| Dump `collision_meshes/` | Does not work — get from Diego via Drive |
| Contribute custom rewards, obs builders, configs | Works |

If you are the Mac teammate, your most productive contributions are:
1. Writing custom reward functions and obs builders (pure Python, no GPU needed).
2. Designing experiment configs (`configs/experiments/exp_NNN_*.yaml`).
3. Reviewing pull requests.
4. Running long Colab training jobs and sharing checkpoints.

---

## Common errors and fixes

### `memory allocation of 72057594037927944 bytes failed`
You downloaded the wrong `rlviser` binary. The latest release (`v0.9.x`) is incompatible with `rlviser-py 0.6.13`. Replace with `v0.8.7`. See Step 6b.

### `ModuleNotFoundError: No module named 'rocketsim'`
The PyPI package installs as `RocketSim` (capital R, S). You probably tried to `import rocketsim`. Our code never does this — go through `rlgym_sim` instead. If you really need it directly, `import RocketSim`.

### `Package 'rlbot' requires a different Python: 3.13.12 not in '<3.12,>=3.10'`
Your conda env is on the wrong Python. Run `python --version` — if it's not 3.10.x, recreate the env from Step 3 with `python=3.10` explicitly.

### `ModuleNotFoundError: No module named 'rlgym_tools.extra_action_parsers'`
`rlgym-tools` v2 moved this module. We vendored `LookupAction` at `src/rlbot/actions/lookup_action.py` to bypass this. Use that import path. If you are seeing this error, you are probably running stale code — pull `main` again.

### `ERROR: Could not find a version that satisfies the requirement rocketsim==2.2.4`
The pin in `requirements.txt` is `2.2.1`, not `2.2.4`. If you see this error, you are on an old branch — pull `main`.

### `FileNotFoundError: ...PPO_POLICY.pt` when running `simple_bot_play.py`
The play script needs the path to point at a specific timestep subfolder (e.g. `.../500028/`), not the parent run folder. The `find_latest_checkpoint()` helper in `simple_bot_play.py` handles this. If you got an old copy of the script that uses `find_latest_run_folder()`, pull the latest `diego-bots/` from Diego's branch.

### `Failed to launch RLViser (./rlviser.exe): The system cannot find the file specified`
`rlviser.exe` is not in the project root. See Step 6b. The Python side auto-launches the binary from the current working directory; if the binary isn't there, this error fires. Always run Python commands from the `rlgym/` root.

---

## Reference — Diego's exact environment

For diff'ing if anything goes sideways:

| Component | Version |
|---|---|
| OS | Windows 11 |
| GPU | NVIDIA GeForce RTX 4070 Laptop (8 GB VRAM) |
| Driver | 566.07, CUDA runtime 12.7 |
| Conda env name | `rlbot310` |
| Python | 3.10.20 |
| torch | 2.5.1+cu121 |
| numpy | 1.26.4 |
| rocketsim | 2.2.1 |
| rlgym-sim | 1.2.6 |
| rlgym-ppo | 1.3.13 |
| rlgym-tools | 2.6.4 |
| rlgym | 2.0.1 (pulled in transitively, not used directly) |
| rlviser-py | 0.6.13 |
| rlviser.exe | v0.8.7 |
| wandb | 0.17.5 |
| pytest | 8.3.2 |

If your `pip list` for any of these differs, your training results may diverge from Diego's. Match the column above and re-run.

---

## End goal

After completing every step, you should be able to:

1. Train the simple bot on your machine and produce checkpoints.
2. (Windows) Visualize the bot playing locally.
3. Push your own branch with your name on it to GitHub.
4. Iterate on rewards, configs, and architectures using the modular framework already in `src/rlbot/`.

Once this works end to end, you are at the same starting line Diego was on day 4 of the 45-day plan. From here, real experimentation begins — see `docs/roadmap_45_days.md` for the week-by-week plan.
