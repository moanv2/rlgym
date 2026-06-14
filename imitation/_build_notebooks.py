"""Generate the four project notebooks from source (so they stay valid + reproducible).

Run:  python imitation/_build_notebooks.py
Then execute them with nbconvert (see imitation/README.md).

Keeping the notebooks under version control as generated artifacts means we can regenerate
them deterministically and review the cell sources as plain Python here, rather than diffing
raw .ipynb JSON.
"""
from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

NB_DIR = Path(__file__).resolve().parent / "notebooks"
NB_DIR.mkdir(parents=True, exist_ok=True)


def build(cells, path):
    nb = new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    })
    nbformat.write(nb, str(path))
    print("wrote", path)


# preamble code shared by notebooks that use the engine
PREAMBLE = """import sys, os
sys.path.insert(0, os.path.abspath('..'))   # make imitation/ importable -> il_core (cwd = notebooks/)
import il_core
il_core.use_repo_cwd()                       # chdir to repo root so RocketSim finds ./collision_meshes/
import numpy as np
import torch
import matplotlib.pyplot as plt
from il_core import (ExpertPolicy, BCStudent, DemoBuffer, collect_expert_demos,
                     train_bc, rollout_student_relabel, evaluate_student_winrate,
                     save_student_checkpoint, build_env, OBS_DIM, N_ACTIONS,
                     EXPERT_CKPT, DATA_DIR, ARTIFACT_DIR, REPO_ROOT)
os.makedirs(DATA_DIR, exist_ok=True); os.makedirs(ARTIFACT_DIR, exist_ok=True)
print('engine loaded | OBS_DIM', OBS_DIM, '| N_ACTIONS', N_ACTIONS)"""


# ============================================================ 01 expert overview
def nb_01():
    return [
        new_markdown_cell(
            "# 01 - The PPO Expert\n"
            "### Reinforcement Learning & Autonomous Systems - Group Project (Imitation Learning half)\n\n"
            "**Project framing.** The assignment is *PPO + Imitation Learning* (Behavioural Cloning and DAgger). "
            "With Jaume's approval we run the **same pipeline in 1v1 Rocket League** (`rlgym_sim` + `rlgym-ppo`) "
            "instead of Walker2D / Ant. The structure of the rubric is unchanged:\n\n"
            "| Rubric element | Our realisation |\n"
            "|---|---|\n"
            "| PPO expert | Our champion 1v1 bot, trained with PPO self-play (~3.4B steps) |\n"
            "| Expert demonstrations | `(observation, action)` pairs rolled out from the champion |\n"
            "| Behavioural Cloning | A from-scratch MLP trained by supervised learning (notebook 03) |\n"
            "| DAgger | Student rollout -> expert relabelling -> aggregate -> retrain (notebook 04) |\n"
            "| Ablations | Dataset size + network capacity (notebook 03) |\n"
            "| Expert vs student video | Rendered head-to-head match (notebook 04) |\n\n"
            "**Why this is a classification problem.** The action space is the **90-way discrete `LookupAction`** "
            "(each index = a controller vector). So cloning the expert is a *classification* task with "
            "**cross-entropy** loss over 90 classes - **not** MSE regression. This is the single most important "
            "modelling decision in the whole pipeline.\n\n"
            "**The expert is a black box.** We only ever *call* the champion to choose an action or to label a "
            "state. We never read its weights or training recipe - exactly how DAgger treats its expert oracle."
        ),
        new_code_cell(PREAMBLE),
        new_markdown_cell(
            "## Load the expert and inspect it\n"
            "The champion is a `DiscreteFF` policy: `AdvancedObs` (107-dim) in, a softmax over 90 actions out. "
            "We wrap it in `ExpertPolicy`, exposing only `.act()` (chosen action) and `.action_probs()` "
            "(full distribution, for diagnostics)."
        ),
        new_code_cell(
            "expert = ExpertPolicy(device='cpu')\n"
            "print('Expert loaded from:', EXPERT_CKPT)\n"
            "print('Hidden layers (inferred from weights):',\n"
            "      [tuple(m.weight.shape) for m in expert.policy.model if hasattr(m, 'weight')])"
        ),
        new_markdown_cell(
            "### The expert acting on a real game state\n"
            "We reset a 1v1 env, take the blue car's observation, and show the expert's chosen action plus its "
            "top-5 action probabilities. A peaked distribution means a confident expert - good news for cloning."
        ),
        new_code_cell(
            "env = build_env(max_seconds=10, randomize=False)\n"
            "obs, info = env.reset(return_info=True)\n"
            "obs0 = (obs if isinstance(obs, list) else [obs])[0]\n"
            "a = expert.act(obs0, deterministic=True)\n"
            "probs = expert.action_probs(obs0)\n"
            "top5 = np.argsort(probs)[::-1][:5]\n"
            "print('obs shape       :', np.asarray(obs0).shape)\n"
            "print('chosen action   :', a)\n"
            "print('top-5 actions   :', list(top5))\n"
            "print('their probs     :', [round(float(probs[i]), 3) for i in top5])\n"
            "env.close()\n\n"
            "fig, ax = plt.subplots(figsize=(8, 3))\n"
            "ax.bar(range(N_ACTIONS), probs)\n"
            "ax.set_title('Expert action distribution on one state'); ax.set_xlabel('action index'); ax.set_ylabel('prob')\n"
            "plt.tight_layout(); plt.show()"
        ),
        new_markdown_cell(
            "**Takeaway.** The expert is loaded and behaves deterministically (argmax) as deployed. "
            "Notebook **02** turns it into a demonstration dataset; **03** clones it with BC; **04** improves the "
            "clone with DAgger and renders an expert-vs-student match."
        ),
    ]


# ============================================================ 02 data collection
def nb_02():
    return [
        new_markdown_cell(
            "# 02 - Collecting Expert Demonstrations\n\n"
            "Behavioural Cloning needs a dataset `D = {(obs, expert_action)}`. We roll the champion out in the 1v1 "
            "env and record, for **every** car at **every** step, the observation and the expert's **deterministic "
            "(argmax)** action - the target we want the student to reproduce.\n\n"
            "**We collect from real games (kickoff starts).** Every episode begins at a normal kickoff and plays "
            "out; the orange car *executes* stochastic samples (while still being recorded with its argmax label) "
            "so games diverge and we get varied, realistic trajectories. This **narrow, on-task distribution** is "
            "deliberate: it is exactly where Behavioural Cloning's *covariate shift* bites (the student drifts off "
            "the expert's trajectories at deployment) and therefore where DAgger earns its keep in notebook 04. "
            "(Collecting from wildly randomized states instead would broaden the demo distribution and *mask* the "
            "covariate-shift problem.) This is the dataset notebook 03 trains on."
        ),
        new_code_cell(PREAMBLE),
        new_markdown_cell(
            "## Config\n"
            "`N_EPISODES` is deliberately modest so this runs quickly and barely disturbs the live PPO training "
            "sharing the CPU. Scale it up (e.g. 200+) for the final submission numbers when the CPU is free."
        ),
        new_code_cell(
            "N_EPISODES = 60      # scaled up for tighter numbers (bump to 200+ if CPU is fully free)\n"
            "MAX_SECONDS = 20     # per-episode cap (game seconds)\n"
            "SEED = 0"
        ),
        new_code_cell(
            "expert = ExpertPolicy(device='cpu')\n"
            "def _p(ep, tot, n): print(f'  episode {ep}/{tot}  |  {n} pairs so far', flush=True)\n"
            "buf = collect_expert_demos(expert, N_EPISODES, max_seconds=MAX_SECONDS,\n"
            "                           randomize=False, seed=SEED, progress=_p)  # kickoff games\n"
            "buf.save(DATA_DIR)\n"
            "print('\\nsaved', len(buf), 'demonstrations to', DATA_DIR)"
        ),
        new_markdown_cell("## Dataset analysis\nSize, action usage, episode returns/lengths, and observation stats."),
        new_code_cell(
            "X, y = buf.observations, buf.actions\n"
            "print('observations:', X.shape, '| actions:', y.shape)\n"
            "print('distinct actions used:', len(np.unique(y)), '/', N_ACTIONS)\n"
            "print('episodes:', len(buf.episode_lengths),\n"
            "      '| mean length:', round(float(np.mean(buf.episode_lengths)), 1),\n"
            "      '| mean result:', round(float(np.mean(buf.episode_returns)), 3))"
        ),
        new_code_cell(
            "fig, ax = plt.subplots(1, 3, figsize=(15, 3.5))\n"
            "counts = np.bincount(y, minlength=N_ACTIONS)\n"
            "ax[0].bar(range(N_ACTIONS), counts); ax[0].set_title('Action frequency (class balance)')\n"
            "ax[0].set_xlabel('action index'); ax[0].set_ylabel('count')\n"
            "ax[1].hist(buf.episode_lengths, bins=15, color='#3a7d44'); ax[1].set_title('Episode length (steps)')\n"
            "ax[2].hist(buf.episode_returns, bins=[-1.5,-0.5,0.5,1.5], color='#4aa3ff'); ax[2].set_title('Episode result (-1/0/+1)')\n"
            "plt.tight_layout(); plt.show()"
        ),
        new_code_cell(
            "# class imbalance summary -> motivates why top-1 accuracy alone can mislead\n"
            "frac = counts / counts.sum()\n"
            "order = np.argsort(frac)[::-1]\n"
            "print('most common action %4d used %.1f%% of the time' % (order[0], 100*frac[order[0]]))\n"
            "print('top-5 actions cover %.1f%% of all demonstrations' % (100*frac[order[:5]].sum()))\n"
            "print('a majority-class baseline would score top-1 acc =', round(float(frac.max()), 3))"
        ),
        new_markdown_cell(
            "**Research note (covered in 03).** The action distribution is highly imbalanced - a handful of "
            "controls dominate. That sets the bar: a trivial 'always predict the most common action' baseline "
            "already gets the accuracy printed above, so the BC student must beat *that* to show it learned the "
            "expert's state-dependent behaviour, not just the marginal."
        ),
    ]


# ============================================================ 03 behavioural cloning
def nb_03():
    return [
        new_markdown_cell(
            "# 03 - Behavioural Cloning (from scratch)\n\n"
            "We clone the expert with a **from-scratch PyTorch MLP** trained by supervised learning. Because the "
            "action space is the discrete 90-way `LookupAction`, this is **multi-class classification** with "
            "**cross-entropy** loss - not regression.\n\n"
            "`BCStudent` is an MLP `107 -> hidden -> 90 logits`. Its layer layout mirrors `rlgym-ppo`'s "
            "`DiscreteFF` exactly (Linear+ReLU stack, logits head), so a trained student exports to a "
            "`PPO_POLICY.pt` that the repo's existing `evaluate.py` / `tournament.py` / `make_match_video.py` load "
            "with no changes - which is how we measure win-rate and render video later."
        ),
        new_code_cell(PREAMBLE),
        new_markdown_cell(
            "### Honest train/val split: hold out whole EPISODES\n"
            "Frames are collected sequentially and at ~15 Hz consecutive frames are near-identical with the same "
            "label. A *random* frame split would scatter near-duplicate neighbours across train and val and "
            "**inflate** accuracy (it measures interpolation between adjacent frames, not generalization). So we "
            "hold out **whole episodes** (`split_by_episode`) and use that one fixed val set everywhere below."
        ),
        new_code_cell(
            "from il_core import split_by_episode\n"
            "buf = DemoBuffer.load(DATA_DIR)\n"
            "tr_idx, val_idx = split_by_episode(buf, val_frac=0.2, seed=0)\n"
            "Xtr, ytr = buf.observations[tr_idx], buf.actions[tr_idx]\n"
            "Xval, yval = buf.observations[val_idx], buf.actions[val_idx]\n"
            "n_val_eps = len(set(buf.episode_ids[val_idx].tolist()))\n"
            "print(f'train {len(ytr)} frames | val {len(yval)} frames from {n_val_eps} held-out episodes')\n"
            "device = il_core.pick_device(); print('train device:', device)"
        ),
        new_markdown_cell("## Train the baseline BC student\nCross-entropy on the training episodes; accuracy on the held-out episodes."),
        new_code_cell(
            "student = BCStudent(hidden_sizes=(256, 256))\n"
            "res = train_bc(student, Xtr, ytr, val_data=(Xval, yval), epochs=80, batch_size=512, lr=1e-3, verbose=True)\n"
            "print('\\nfinal (episode-held-out)  top-1:', round(res['val_acc'], 3), '| top-3:', round(res['val_top3'], 3))\n"
            "save_student_checkpoint(student, os.path.join(ARTIFACT_DIR, 'bc_student'))"
        ),
        new_code_cell(
            "h = res['history']\n"
            "fig, ax = plt.subplots(1, 2, figsize=(12, 4))\n"
            "ax[0].plot(h['train_loss'], label='train'); ax[0].plot(h['val_loss'], label='val (held-out episodes)')\n"
            "ax[0].set_title('Cross-entropy loss'); ax[0].set_xlabel('epoch'); ax[0].legend()\n"
            "ax[1].plot(h['val_acc'], label='top-1'); ax[1].plot(h['val_top3'], label='top-3')\n"
            "ax[1].set_title('Held-out action-agreement'); ax[1].set_xlabel('epoch'); ax[1].legend()\n"
            "plt.tight_layout(); plt.show()"
        ),
        new_markdown_cell(
            "## Ablation 1 - dataset size\n"
            "**Research question:** how does BC accuracy scale with the number of demonstrations? We retrain on "
            "increasing fractions of the **training** frames and evaluate every model on the **same fixed held-out "
            "episodes** - so the only thing changing is training-set size. More expert data should help (up to a "
            "point) - the empirical face of BC's sample complexity."
        ),
        new_code_cell(
            "fractions = [0.1, 0.25, 0.5, 0.75, 1.0]\n"
            "SEEDS = [0, 1, 2]   # repeat each point to get error bars\n"
            "sizes, accs_mean, accs_std = [], [], []\n"
            "for f in fractions:\n"
            "    m = max(64, int(len(ytr) * f)); accs_f = []\n"
            "    for sd in SEEDS:\n"
            "        rng = np.random.default_rng(sd)\n"
            "        sub = rng.permutation(len(ytr))[:m]\n"
            "        torch.manual_seed(sd)\n"
            "        s = BCStudent(hidden_sizes=(256, 256))\n"
            "        r = train_bc(s, Xtr[sub], ytr[sub], val_data=(Xval, yval), epochs=60, batch_size=512, lr=1e-3, seed=sd)\n"
            "        accs_f.append(r['val_acc'])\n"
            "    sizes.append(m); accs_mean.append(np.mean(accs_f)); accs_std.append(np.std(accs_f))\n"
            "    print(f'  {m:6d} train frames -> held-out acc {np.mean(accs_f):.3f} +/- {np.std(accs_f):.3f}')\n"
            "fig, ax = plt.subplots(figsize=(7, 4))\n"
            "ax.errorbar(sizes, accs_mean, yerr=accs_std, fmt='o-', capsize=4)\n"
            "ax.set_xlabel('# training frames'); ax.set_ylabel('held-out top-1 accuracy')\n"
            "ax.set_title('BC accuracy vs dataset size (mean +/- std over 3 seeds, fixed held-out episodes)')\n"
            "plt.tight_layout(); plt.show()"
        ),
        new_markdown_cell(
            "## Ablation 2 - network capacity\n"
            "**Research question:** does a wider student clone the expert better? We hold **depth fixed at 3 layers** "
            "and vary only the **width**, evaluating on the same fixed held-out episodes. (Single seed, so read small "
            "differences as within-noise rather than a strong capacity law.)"
        ),
        new_code_cell(
            "archs = {'narrow (128x3)': (128, 128, 128), 'mid (256x3)': (256, 256, 256), 'wide (512x3)': (512, 512, 512)}\n"
            "SEEDS = [0, 1, 2]\n"
            "names, cap_mean, cap_std = [], [], []\n"
            "for name, hs in archs.items():\n"
            "    accs_a = []\n"
            "    for sd in SEEDS:\n"
            "        torch.manual_seed(sd)\n"
            "        s = BCStudent(hidden_sizes=hs)\n"
            "        r = train_bc(s, Xtr, ytr, val_data=(Xval, yval), epochs=60, batch_size=512, lr=1e-3, seed=sd)\n"
            "        accs_a.append(r['val_acc'])\n"
            "    names.append(name); cap_mean.append(np.mean(accs_a)); cap_std.append(np.std(accs_a))\n"
            "    print(f'  {name:14s} -> held-out acc {np.mean(accs_a):.3f} +/- {np.std(accs_a):.3f}')\n"
            "fig, ax = plt.subplots(figsize=(7, 4))\n"
            "ax.bar(names, cap_mean, yerr=cap_std, capsize=4, color='#1f6fff'); ax.set_ylabel('held-out top-1 accuracy')\n"
            "ax.set_title('BC accuracy vs width (depth 3, mean +/- std over 3 seeds)'); plt.tight_layout(); plt.show()"
        ),
        new_markdown_cell(
            "## Discussion\n"
            "- **Classification, not regression.** Cross-entropy over 90 discrete actions is the correct objective; "
            "MSE on action indices would be meaningless (action 89 is not '89x' action 1).\n"
            "- **Leak-free evaluation.** We split by whole episodes; a naive random frame split inflates accuracy on "
            "this 15 Hz sequential data because adjacent near-duplicate frames leak across train and val.\n"
            "- **The covariate-shift problem.** BC only ever sees *expert* states. At deployment the student makes "
            "small errors, drifts into states the expert never visited, and has no idea what to do there - errors "
            "compound as **O(eps*T^2)** over a horizon T. Even a respectable held-out accuracy does **not** "
            "guarantee good play (notebook 04 shows the clone winning ~0% despite this). That gap is exactly what "
            "**DAgger** fixes.\n"
            "- **Scaling.** Accuracy rises with more demonstrations; widening the net helps only marginally here "
            "(single seed - treat small gaps as noise). Both matter less than fixing the distribution mismatch."
        ),
    ]


# ============================================================ 04 dagger
def nb_04():
    return [
        new_markdown_cell(
            "# 04 - DAgger (Dataset Aggregation)\n\n"
            "BC fails under **covariate shift**: it is trained on expert states but deployed on its own. The "
            "student makes small errors, drifts into states the expert never demonstrated, and - having never been "
            "told what to do there - makes bigger errors. Errors compound as **O(eps*T^2)** over a horizon T.\n\n"
            "**DAgger** fixes this by training on the student's *own* state distribution:\n\n"
            "1. Roll the **student** out in the env.\n"
            "2. Ask the **expert** for the correct action at every state the student visited.\n"
            "3. **Aggregate** those `(state, expert_action)` pairs into the dataset.\n"
            "4. **Retrain** the student on the aggregated dataset. Repeat.\n\n"
            "This bounds regret at **O(eps*T)**. Our headline metric is **expert-agreement on "
            "STUDENT-visited states** - the direct measure of covariate shift. For pure BC it is far *below* its "
            "agreement on expert states (the gap = covariate shift); DAgger should close that gap.\n\n"
            "We use the **narrow kickoff-game demonstrations from notebook 02**: this realistic on-task "
            "distribution is exactly where covariate shift bites, so the effect is clean and large."
        ),
        new_code_cell(PREAMBLE),
        new_markdown_cell("## Config"),
        new_code_cell(
            "SEED_PAIRS = 5000        # small BC seed from the kickoff demos -> pronounced covariate shift\n"
            "DAGGER_ITERS = 6         # aggregation rounds\n"
            "RELABEL_EPISODES = 8     # student rollouts per measurement\n"
            "N_MEAS = 3               # independent rollouts per round -> agreement error bars\n"
            "WINRATE_EPISODES = 20    # head-to-head task check vs a weak fixed opponent\n"
            "WEAK_OPP = os.path.join(REPO_ROOT, 'checkpoints', '_eval_snapshots', 'basics_250M')  # advanced-obs\n"
            "ARCH = (256, 256)"
        ),
        new_code_cell(
            "from il_core import split_by_episode\n"
            "expert = ExpertPolicy(device='cpu')\n"
            "buf = DemoBuffer.load(DATA_DIR)\n"
            "# hold out whole episodes for a leak-free expert-state reference; seed BC from train episodes only\n"
            "tr_idx, val_idx = split_by_episode(buf, val_frac=0.2, seed=0)\n"
            "Xval, yval = buf.observations[val_idx], buf.actions[val_idx]\n"
            "rng = np.random.default_rng(0)\n"
            "sub = rng.permutation(tr_idx)[:SEED_PAIRS]\n"
            "Xs, ys = buf.observations[sub].copy(), buf.actions[sub].copy()\n"
            "print('BC seed:', len(ys), 'pairs (from train episodes) |', len(yval), 'held-out val frames')"
        ),
        new_markdown_cell(
            "## BC baseline and the covariate-shift gap\n"
            "Train a clean BC student on the seed demos, then measure two agreements: on held-out **expert** "
            "states (whole held-out episodes - leak-free) and on the student's **own** rollout states. The "
            "difference is the covariate shift DAgger must fix."
        ),
        new_code_cell(
            "student = BCStudent(hidden_sizes=ARCH)\n"
            "bc_res = train_bc(student, Xs, ys, val_data=(Xval, yval), epochs=80, batch_size=512, lr=1e-3)\n"
            "expert_state_acc = bc_res['val_acc']\n"
            "_, _, bc_student_agree, _ = rollout_student_relabel(student, expert, RELABEL_EPISODES, randomize=False, seed=500)\n"
            "wr_bc = evaluate_student_winrate(student, WEAK_OPP, episodes=WINRATE_EPISODES)\n"
            "print(f'BC  agreement on EXPERT states : {expert_state_acc:.3f}')\n"
            "print(f'BC  agreement on STUDENT states: {bc_student_agree:.3f}')\n"
            "print(f'==> covariate-shift GAP        : {expert_state_acc - bc_student_agree:+.3f}')\n"
            "print(f'BC  win-rate vs weak bot       : {wr_bc[\"blue_win_rate\"]:.3f}  (per-action accuracy does not imply task skill)')"
        ),
        new_markdown_cell(
            "## DAgger loop\n"
            "Each round: roll the current student out (kickoff games), record the expert's label at every visited "
            "state, aggregate, and **retrain from scratch** on the growing dataset. We log the student's agreement "
            "on its own visited states each round."
        ),
        new_code_cell(
            "Xa, ya = Xs.copy(), ys.copy()\n"
            "agree_mean, agree_std, size_hist = [], [], []\n"
            "for it in range(DAGGER_ITERS + 1):\n"
            "    # N_MEAS independent rollouts: aggregate all their labels, report agreement mean +/- std\n"
            "    obs_b, lbl_b, ags = [], [], []\n"
            "    for m in range(N_MEAS):\n"
            "        nobs, nlbl, ag, _ = rollout_student_relabel(student, expert, RELABEL_EPISODES, randomize=False, seed=600 + it*10 + m)\n"
            "        obs_b.append(nobs); lbl_b.append(nlbl); ags.append(ag)\n"
            "    agree_mean.append(float(np.mean(ags))); agree_std.append(float(np.std(ags))); size_hist.append(len(ya))\n"
            "    print(f'iter {it}: |D|={len(ya):6d} | agreement {np.mean(ags):.3f} +/- {np.std(ags):.3f}')\n"
            "    if it < DAGGER_ITERS:\n"
            "        Xa = np.concatenate([Xa] + obs_b, axis=0); ya = np.concatenate([ya] + lbl_b, axis=0)\n"
            "        student = BCStudent(hidden_sizes=ARCH)\n"
            "        train_bc(student, Xa, ya, epochs=80, batch_size=512, lr=1e-3, val_frac=0.15)"
        ),
        new_code_cell(
            "wr_dagger = evaluate_student_winrate(student, WEAK_OPP, episodes=WINRATE_EPISODES)\n"
            "save_student_checkpoint(student, os.path.join(ARTIFACT_DIR, 'dagger_student'))\n"
            "print(f'BC     student-state agreement: {agree_mean[0]:.3f} +/- {agree_std[0]:.3f}  | win-rate vs weak bot: {wr_bc[\"blue_win_rate\"]:.3f}')\n"
            "print(f'DAgger student-state agreement: {agree_mean[-1]:.3f} +/- {agree_std[-1]:.3f}  | win-rate vs weak bot: {wr_dagger[\"blue_win_rate\"]:.3f}')"
        ),
        new_code_cell(
            "fig, ax = plt.subplots(1, 2, figsize=(13, 4))\n"
            "ax[0].errorbar(range(len(agree_mean)), agree_mean, yerr=agree_std, fmt='o-', capsize=4, label='student-visited states')\n"
            "ax[0].axhline(expert_state_acc, ls='--', color='gray', label='BC agreement on expert states')\n"
            "ax[0].set_title('Expert agreement on student-visited states (mean +/- std, 3 rollouts/round)'); ax[0].set_xlabel('DAgger iteration')\n"
            "ax[0].set_ylabel('expert agreement'); ax[0].legend()\n"
            "ax[1].bar(['BC seed', 'DAgger final'], [agree_mean[0], agree_mean[-1]], yerr=[agree_std[0], agree_std[-1]], capsize=4, color=['#888', '#1f6fff'])\n"
            "ax[1].set_title('Agreement on student-visited states'); ax[1].set_ylabel('agreement')\n"
            "plt.tight_layout(); plt.show()"
        ),
        new_markdown_cell(
            "## Expert vs student video\n"
            "Render a head-to-head match: champion (blue) vs our DAgger student (orange), reusing the repo's "
            "tested `make_match_video.py`. Both use AdvancedObs (107-dim), so no cross-obs adapter is needed."
        ),
        new_code_cell(
            "import subprocess, sys\n"
            "student_dir = os.path.join(ARTIFACT_DIR, 'dagger_student')\n"
            "out_mp4 = os.path.join(ARTIFACT_DIR, 'expert_vs_student.mp4')\n"
            "tool = os.path.join(REPO_ROOT, 'tools', 'make_match_video.py')\n"
            "cmd = [sys.executable, tool,\n"
            "       '--a-policy', EXPERT_CKPT, '--a-obs', 'advanced', '--a-dim', '107', '--a-name', 'Champion(expert)',\n"
            "       '--b-policy', student_dir, '--b-obs', 'advanced', '--b-dim', '107', '--b-name', 'DAgger student',\n"
            "       '--out', out_mp4, '--max-seconds', '20']\n"
            "print('rendering video...', flush=True)\n"
            "r = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)  # cwd has collision_meshes/\n"
            "print(r.stdout[-500:]); print(r.stderr[-500:] if r.returncode else 'video OK -> ' + out_mp4)"
        ),
        new_markdown_cell(
            "## Conclusion\n"
            "- **The covariate-shift gap is real and large.** Pure BC agrees with the expert far more on held-out "
            "expert states than on its own visited states (the gap printed above). High per-action accuracy does "
            "**not** imply task skill: this BC clone wins ~0% even against a weak bot, because action errors "
            "compound over a ~200-step episode.\n"
            "- **DAgger closes the gap.** Labelling the states the student actually drifts into raises its agreement "
            "on its own distribution across rounds (the per-round estimate is noisy - it dips early before rising - "
            "but the net increase from first to last round is clear and large). This is the **O(eps*T)** vs "
            "**O(eps*T^2)** story made concrete.\n"
            "- **Distribution matters.** This clean result depends on demonstrating from a *narrow, on-task* "
            "distribution (kickoff games). Demonstrating from wildly randomized states would broaden the demo "
            "coverage and mask the covariate shift - a useful caveat about when DAgger helps most.\n"
            "- **Cost.** DAgger needs the expert available *during* training (to relabel); BC needs only a fixed "
            "dataset. That is the practical trade-off.\n"
            "- **Honesty on scale.** Counts are modest so the pipeline runs alongside live PPO training; bump the "
            "`CONFIG` constants for tighter final numbers when the CPU is free."
        ),
    ]


if __name__ == "__main__":
    build(nb_01(), NB_DIR / "01_expert_overview.ipynb")
    build(nb_02(), NB_DIR / "02_data_collection.ipynb")
    build(nb_03(), NB_DIR / "03_behavioural_cloning.ipynb")
    build(nb_04(), NB_DIR / "04_dagger.ipynb")
    print("all notebooks generated in", NB_DIR)
