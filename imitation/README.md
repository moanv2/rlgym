# Imitation Learning: Behavioural Cloning + DAgger

The **Imitation Learning** half of the group project (the PPO half is the trained Rocket
League bot in the rest of the repo). With Jaume's approval we run the standard IL pipeline
in **1v1 Rocket League** (`rlgym_sim`) instead of Walker2D / Ant. The structure of the
assignment is unchanged: a PPO **expert**, a cloned **student**, **Behavioural Cloning**,
**DAgger**, ablations, an expert-vs-student **video**, and the **research questions**.

## What the expert is

Our champion 1v1 PPO bot (`martin-bots/checkpoints/CHAMPION_3.37B_advanced1024`, ~3.4B
steps). It is treated as a **black box**: we only call it to choose an action or to label a
state, never reading its weights or recipe - exactly how DAgger treats its expert oracle.

- **Observation:** `AdvancedObs`, 107-dim.
- **Action:** `LookupAction`, **90 discrete** controller presets.
- Because actions are discrete, imitation is a **classification** problem with
  **cross-entropy** loss over 90 classes - not MSE regression. This is the central
  modelling decision.

## Layout

```
imitation/
  il_core.py            # the engine: expert wrapper, data collection, from-scratch BC, DAgger
  _build_notebooks.py   # regenerates the notebooks below (review cell sources as Python)
  notebooks/
    01_expert_overview.ipynb       # the PPO expert + problem framing
    02_data_collection.ipynb       # roll the expert out -> (obs, action) dataset + analysis
    03_behavioural_cloning.ipynb   # from-scratch BC + dataset-size & capacity ablations
    04_dagger.ipynb                # DAgger loop + BC-vs-DAgger + expert-vs-student video
  bc_imitation_lib.py   # OPTIONAL cross-check using the `imitation` library (separate env)
  requirements-imitation.txt       # deps for that optional cross-check only
  data/                 # generated demonstrations (.npy)        [gitignored]
  artifacts/            # trained students + plots + video       [gitignored]
```

## How to run

Use the project conda env (`rl-group-project`). Run the notebooks **in order** (02 -> 03 ->
04; 01 is standalone). They import `il_core` and auto-`chdir` to the repo root so RocketSim
finds `collision_meshes/`.

> **Fresh clone:** `data/` and `artifacts/` are gitignored (large/regenerable), so a clean
> checkout has neither. You **must execute notebook 02 first** to regenerate `data/*.npy`
> before 03/04 will run (they start with `DemoBuffer.load(DATA_DIR)`). If you only want to
> *read* the results, the committed notebooks already carry executed outputs (plots + numbers).

```bash
# regenerate notebooks from source (optional)
python imitation/_build_notebooks.py

# execute a notebook headless
cd imitation/notebooks
python -m nbconvert --to notebook --execute --inplace 02_data_collection.ipynb \
    --ExecutePreprocessor.timeout=3600 --ExecutePreprocessor.kernel_name=python3
```

The committed notebooks already contain executed outputs (plots + numbers). The dataset and
DAgger rollout/eval counts are set **modest** so the pipeline runs alongside live PPO
training; bump the `CONFIG` cells (e.g. `N_EPISODES`, `DAGGER_ITERS`, `WINRATE_EPISODES`)
for tighter final numbers when the CPU is free.

## The reuse trick (why the student plugs into the existing harness)

`BCStudent` mirrors `rlgym-ppo`'s `DiscreteFF` layer layout exactly (Linear+ReLU stack, a
final **logits** head; the only difference is the softmax, which carries no parameters). So
a trained student exported with `save_student_checkpoint` writes a `PPO_POLICY.pt` that the
repo's existing `src/rlbot/evaluation/evaluate.py`, `scripts/tournament.py`, and
`tools/make_match_video.py` load **unchanged**. That is how we measure student win-rate and
render the expert-vs-student match without any new evaluation code.

## Results summary (from the executed notebooks)

All accuracies are **leak-free** (validation holds out whole EPISODES, a random frame split
would scatter near-identical 15 Hz neighbours across train/val and inflate the numbers), and
the ablations and DAgger curve are reported as **mean +/- std over repeats**.

- **Dataset:** 60 kickoff-game episodes -> ~26.5k `(obs, action)` pairs, all 90 actions used,
  heavy class imbalance (most common action ~13%, majority-class baseline top-1 = 0.13).
- **Behavioural Cloning (episode-held-out):** **top-1 = 0.35, top-3 = 0.57** (~2.7x the
  majority baseline; still <50%, i.e. the clone disagrees with the expert on most states).
  Accuracy scales with data (0.29 -> 0.36 across the size ablation, tight error bars over 3
  seeds) and is flat across network width (0.35 to 0.36, error bars overlap, so capacity barely
  matters in this regime).
- **Covariate shift, made concrete:** a 5k-pair BC clone agrees with the expert ~0.31 per
  action on held-out expert states but only ~0.20 on its *own* visited states (gap ~+0.11), and
  wins ~0% even against a much smaller 250M-step bot - action errors compound over a ~200-step
  episode (high per-action accuracy does not imply task skill).
- **DAgger:** labelling the student's own visited states raises agreement on that distribution
  from ~0.15 to **~0.69** (mean +/- std over 3 rollouts per round) across 6 rounds, closing the
  covariate-shift gap - the **O(eps*T)** vs **O(eps*T^2)** story made concrete. The per-round
  metric is noisy but the net rise is large and clear. The expert-vs-student video is a close
  game. (Exact curve in `04_dagger.ipynb`.)

## How this covers the IL requirements

| Requirement | Where |
|---|---|
| Expert policy (PPO) | champion bot, loaded in `01` / used everywhere |
| Expert demonstrations | `02_data_collection.ipynb` -> `data/*.npy` |
| Behavioural Cloning (from scratch) | `il_core.BCStudent` + `train_bc`, `03_behavioural_cloning.ipynb` |
| DAgger | `il_core.rollout_student_relabel` + loop in `04_dagger.ipynb` |
| Ablations | dataset size + network capacity in `03`; BC-vs-DAgger in `04` |
| Evaluation | held-out action-agreement + head-to-head win-rate (reused harness) |
| Expert-vs-student video | `artifacts/expert_vs_student.mp4` from `04` |
| Research questions | discussion cells in `03` and `04` (classification vs regression; covariate shift O(eps*T^2) vs DAgger O(eps*T); data/capacity scaling) |
| Imitation library cross-check | `bc_imitation_lib.py` (optional, separate env) |

## Note on the `imitation` library

The graded, primary implementation is **from scratch** (it shows we understand the
algorithms). `bc_imitation_lib.py` is an optional cross-check that runs BC via the
`imitation` library. We deliberately do **not** install `imitation` / Stable-Baselines3 /
gymnasium into the training env: they depend on `gymnasium`, which clashes with the legacy
`gym` that `rlgym_sim` needs, and breaking the env would interrupt live training. Run that
cross-check in a separate virtualenv from `requirements-imitation.txt`.
