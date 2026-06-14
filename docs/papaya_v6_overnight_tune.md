# papaya v6 — the overnight optimizer tune (2026-06-11)

The last code change before pure-training mode. Rewards are **untouched** (still v5);
this pass changes only how hard the PPO optimizer works per batch of gameplay, plus
checkpoint safety for unattended runs. Everything below was verified against the
installed `rlgym_ppo` source (file:line refs included) and stress-tested by an
adversarial review that **rejected half of the original proposal** — the rejected
parts and why are documented at the bottom, because they're the most instructive bit.

---

## 1. TL;DR — what changed

| Parameter | Before | After | One-line why |
|---|---|---|---|
| `ppo_epochs` | 2 | **3** | The ONE real learning lever: 6 → 9 optimizer steps/iteration. KL & clip fraction ran 3-5x below the healthy band all run — updates were too timid. |
| `ppo_ent_coef` | 0.01 | **0.005** | Entropy pinned at ~4.0/4.5 for 1.3B steps — the policy never committed. Halve the bonus, let it sharpen. |
| `policy_lr` / `critic_lr` | 3e-4 (implicit defaults) | **3e-4 (explicit)** | Same values, now pinned in the script and recorded in run history. Deliberately NOT cut — see §6. |
| `timestep_limit` | 2B | **5B** | papaya is at ~1.35B and gains ~300M+/night. At 2B the run would have **silently stopped itself** within a night or two. |
| `save_every_ts` | 100k | **1M** | 100k = a ~54MB checkpoint write every ~10 s ≈ ~200GB of SSD writes per night. 1M = every ~100 s; a crash loses ≤2 min. |
| `n_checkpoints_to_keep` | 5 (default) | **50** | With 5, the morning rollback window was the last ~minutes only. 50 × 1M = a ~50M-step (~1.4 h) window (~2.7GB disk). |
| Rollback archive | — | `diego-bots/checkpoints/_archive/papaya_1024_PRE_V6_1.346B/` | Full pre-v6 checkpoint (policy + critic + both Adam states) copied out of the rotation's reach. |

Unchanged: all v5 rewards, obs (AdvancedObs 107), action parser, state-setter mix,
`ppo_batch_size`/`ts_per_iteration` 100k, `ppo_minibatch_size` 100k, `exp_buffer_size`
300k, `gamma` 0.99, `lambda` 0.95, `clip_range` 0.2, net 1024x3, `n_proc` 16.

Run command is unchanged: `python diego-bots/papaya_1024.py` (auto-resumes from the
latest checkpoint; all v6 parameters take effect on resume — verified, see §3).

---

## 2. Reading the dashboard correctly (why "make value update magnitude go up" is the wrong goal)

**What the metric actually is.** "Value Function Update Magnitude" is computed in
`rlgym_ppo/ppo/ppo_learner.py` (lines 114-116, 216-220): it snapshots every critic
parameter before the update, again after all epochs, and reports the **L2 norm of the
parameter delta**: `‖θ_before − θ_after‖₂`. "Policy Update Magnitude" is the identical
construction for the policy net. These are *how far the weights moved this iteration*
— not learning progress, not value accuracy, not reward.

**Why it plateaued at ~1.0.** With `standardize_returns=True`, the critic's regression
targets live on a normalized scale, and with a constant learning rate + constant
gradient-clip (0.5) + Adam, the per-iteration parameter displacement settles into a
characteristic magnitude. The plateau means the optimizer reached its steady operating
point — expected, not pathological. The slight late decline most likely means the
critic *fits its targets better* (smaller errors → smaller gradients), which is mildly
**good** news. The sharp dips in the chart line up with session restarts (the first
iterations after a resume train on a partially refilled experience buffer — benign).

**The actual pathology is elsewhere on your dashboard:**

| Signal | Value (entire run) | Healthy band | Meaning |
|---|---|---|---|
| Mean KL Divergence | ~0.0024-0.0032 | ~0.008-0.02 | Policy moves 3-5x less per iteration than typical PPO. |
| SB3 Clip Fraction | ~0.022-0.031 | ~0.05-0.15 | Almost no sample ever hits the clip boundary — updates never even approach PPO's safety rail. |
| Policy Entropy | pinned ~4.0 (max ln 90 ≈ 4.5) | should decline over training | The policy stayed near-maximally random for 1.3B steps; it never committed to preferences. |

Conclusion: papaya doesn't learn slowly because the value function "stopped" — it
learns slowly because **every update is tiny and the policy is kept artificially
diffuse**. v6 attacks exactly those two things. After v6, the value-update magnitude
will likely rise somewhat (more steps per iteration = more displacement) — but treat
that as a side effect, not the success metric. The success metrics are in §5.

---

## 3. How the optimizer actually works here (verified in source)

Three findings from `rlgym_ppo` (env `rlbot310`, `site-packages/rlgym_ppo/`) that
shaped — and corrected — this tune:

**(a) Gradient steps per iteration = `ppo_epochs × floor(exp_buffer_size / ppo_batch_size)`.**
The epoch loop (`ppo_learner.py:119`) calls `exp.get_all_batches_shuffled(batch_size)`
(`:121`), which walks the **entire 300k experience buffer** in shuffled 100k batches
(`experience_buffer.py:89-102`) — so each epoch = 3 batches = 3 optimizer steps.
Current config: 2 epochs × 3 = **6 steps/iteration** (confirmed live: the console's
Cumulative Model Updates counter increments by 6 per iteration). v6: 3 × 3 = **9**.

**(b) `ppo_minibatch_size` is gradient ACCUMULATION, not extra steps.** The minibatch
loop (`ppo_learner.py:134`) scales each slice's loss by `minibatch_size/batch_size`
(`:175-177`), accumulates `.backward()` calls, and performs **one** `optimizer.step()`
per 100k batch (`:192-193`). Lowering it changes VRAM usage, not learning. This killed
one of the originally proposed changes (§6).

**(c) Every v6 parameter takes effect on resume.** `ppo_ent_coef`, `ppo_epochs`,
`ppo_minibatch_size` are plain constructor attributes never touched by checkpoint
loading (`ppo_learner.py:255-271` loads only the four state dicts). Learning rates are
*re-applied on top of* the loaded Adam state: `learner.py:192` passes the constructor's
`policy_lr`/`critic_lr` into `load()` (`:446`), which calls `update_learning_rate`
(`:544-546` → `:205-216`) and overwrites `param_group['lr']` in both loaded optimizers.
So the constructor values always win — which is also why v6 pins them explicitly: what
stands in the script is what runs, and the run-history JSONs now record it.

Also verified: the critic uses plain MSE (no value clipping) with grad-norm clip 0.5
(`ppo_learner.py:60, 176, 187-190`), and both Adam moment states load on resume — the
optimizer continues smoothly, no warm-up shock.

---

## 4. Why each change, in one paragraph each

**`ppo_epochs` 2 → 3.** The dashboard says updates are too small (KL ~0.003 vs healthy
0.008-0.02). The cleanest way to apply more learning per collected sample without
touching step *size* is one more pass over the buffer: 6 → 9 optimizer steps per
iteration, a bounded 1.5x increase. Three independent safety mechanisms bound the
worst case: PPO ratio clipping (0.2), gradient-norm clipping (0.5), and the fact that
third-epoch samples that drift outside the clip contribute zero gradient. Projected
landing zone: KL ~0.004-0.007, clip fraction ~0.04-0.10 — still on the conservative
side of healthy. Cost: the learn phase grows ~1.2-1.6 s on a ~10 s iteration, so
overnight collection drops from ~330-370M to ~295-330M env steps — an acceptable trade
for 50% more optimization per sample.

**`ppo_ent_coef` 0.01 → 0.005.** The entropy bonus pays the policy to stay random;
at 0.01 it has held entropy at ~4.0/4.5 for the whole run while the bot's *deterministic*
play is demonstrably stronger than its sampled play (papaya's own evals, and the rival
bot's README says the same about theirs). Halving the bonus shifts the equilibrium
toward commitment. Honest expectation-setting: many of LookupAction's 90 rows are
near-duplicates in most states (e.g. differing only in pitch/yaw mid-drive), and
identical-consequence actions get identical gradients — that creates an entropy
**floor** no coefficient can push below. So expect a slow drift to ~3.7-3.95 overnight,
not a cliff. If entropy is still ~4.02 in the morning, the pin is structural: do NOT
chase it with 0.0025 — it costs nothing in matches (we deploy argmax) and the night
still counts as an epochs-only experiment.

**LRs pinned at 3e-4 (not cut).** See §6 — the original proposal cut `policy_lr` to
2e-4 and review showed that would roughly cancel the epochs change. With KL projected
to land at most ~0.007, there is no instability case for a cut. Passing them explicitly
also future-proofs the script: rlgym_ppo silently applies constructor LRs over the
checkpoint's saved LRs on every resume, so an implicit default here is a hidden
parameter — now it's visible and logged.

**`timestep_limit` 2B → 5B.** `learn()` exits when cumulative steps hit the limit.
At ~1.35B + ~300M/night, 2B would have ended a session *silently mid-night* within two
nights — the process exits cleanly, no error, and you lose the remaining hours. 5B is
"effectively until Ctrl+C" for the rest of the project.

**`save_every_ts` 100k → 1M and `n_checkpoints_to_keep` 5 → 50.** Two sides of the
same unattended-run problem. At ~10k steps/sec, 100k = a full ~54MB checkpoint write
every ~10 seconds — roughly 200GB of SSD writes per night, plus save time inside the
iteration loop, for no benefit since rotation kept only the last 5 (a ~50-second
rollback window!). v6 saves every ~100 seconds and keeps 50, giving a ~50M-step
(~1.4 hour) rollback window for ~2.7GB of disk. The full pre-v6 checkpoint (including
both optimizer states) is archived outside the rotation's reach at
`diego-bots/checkpoints/_archive/papaya_1024_PRE_V6_1.346B/`, so worst case is always
recoverable. Trade-off accepted: a mid-run Ctrl+C now loses up to ~2 minutes of
training instead of ~10 seconds.

---

## 5. Overnight expectations + morning checklist

### At launch (60 seconds, before you walk away)
1. Console prints `[resume] loading checkpoint: ...` with a ~1.35B timestep folder.
2. Console prints **`New policy learning rate: 0.0003` TWICE** — the second line is
   actually the critic's LR but a copy-paste bug in the library prints it with the
   "policy" label (`learner.py:210` and `:216`). Two identical lines = both LRs
   applied correctly; don't be confused by the wording.
3. From the 3rd iteration on (buffer full again), **Cumulative Model Updates should
   increment by 9 per iteration** (was 6) — proof `ppo_epochs=3` is live.
4. wandb run appears as `papaya_1024_1.3B` (or similar) in the usual project.

> **Never switch resume to the library's built-in `checkpoint_load_folder="latest"`
> mode** — that code path is broken in this rlgym_ppo version (`learner.py:480-482`
> slices the wrong string and silently skips every session folder). papaya's own
> `find_latest_checkpoint()` passes an explicit folder, which is the only safe path.

### Expected chart behavior (healthy night)
| Metric | Baseline | Expected with v6 | Dial back if | Roll back if |
|---|---|---|---|---|
| Mean KL Divergence | 0.0024-0.0032 | **0.004-0.007** | median > 0.012 → `ppo_epochs=2` | sustained > 0.03 AND play degraded |
| SB3 Clip Fraction | 0.022-0.031 | **0.04-0.10** | sustained > 0.15 → fewer epochs or lower lr | sustained > 0.25 |
| Policy Entropy | ~4.025 | **3.7-4.0, slow drift down** | < 3.5 → ent_coef back to 0.0075 | < 3.0 (restore a checkpoint where it was > 3.5) |
| Value Function Loss | ~0.030 | **0.02-0.05** (dip then settle) | sustained > 0.06 → critic_lr 1.5e-4 | > 0.15 rising, or NaN |
| Value Update Magnitude | 0.66-0.96 | **may rise to ~1.0-1.4** — fine | — | > 3.0 sustained 30+ min |
| Policy Update Magnitude | 0.6-0.9 | **up to ~1.2** | — | > 2.5 sustained 30+ min |
| Overall Steps/sec | 9.7-10.7k | **8.5-9.5k** (epochs cost) | < 7k → check for crash-restart loops* | — |

*Each crash-restart creates a new `papaya_1024-<unix_ts>/` session folder — more than
2-3 new folders overnight means sim instability, not optimizer trouble.

### The decisive test (morning)
Charts are proxies. The real question is play strength — run the fixed-opponent eval:

```powershell
# 20+ deterministic games vs the same rival you went 6-4 against:
python diego-bots/papaya_1v1_viewer.py --orange martin-bots/checkpoints/CHAMPION_2.1B_recipeD_advanced1024 --episodes 20 --deterministic
```

Decision rule: the 6-4 baseline was a 10-game sample (95% CI roughly 30-85%), so demand
a clear signal — **win rate < 45% over 20 games AND red metrics above → roll back**;
otherwise keep the night's weights even if a chart looks merely odd. For the full
training curve, `scripts/eval_progression_advobs.py --experiment papaya_1024
--reference martin-bots/checkpoints/CHAMPION_2.1B_recipeD_advanced1024` plots win rate
across every checkpoint.

### Rollback procedure (if needed)
1. Stop training (Ctrl+C).
2. Option A — partial: pick a mid-night checkpoint from the 50 retained
   (`diego-bots/checkpoints/papaya_1024/<newest session>/<timestep>/`), delete the
   newer timestep folders in that session, restart (auto-resume picks the survivor).
3. Option B — full: copy the contents of
   `diego-bots/checkpoints/_archive/papaya_1024_PRE_V6_1.346B/` into a fresh
   `diego-bots/checkpoints/papaya_1024/papaya_1024-manual_restore/1346169600/` folder,
   delete (or move away) the overnight session folder, restart.
4. If rolling back, also revert the parameter(s) implicated by the watch rules — not
   necessarily all of v6.

---

## 6. What the review rejected from the original proposal (and why it matters)

The first draft of v6 was: minibatch 100k→25k, epochs 2→3, ent_coef 0.01→0.005,
**policy_lr 3e-4→2e-4**. Two parts died under adversarial review against the actual
source code:

**Rejected: `ppo_minibatch_size` 100k → 25k ("4x more gradient steps").** Wrong model
of the library. In this rlgym_ppo version minibatches are *gradient accumulation
slices* — losses are scaled by `minibatch/batch` and summed into ONE optimizer step
per batch (`ppo_learner.py:131-193`). The change would have altered nothing about
learning, added 4x more CPU→GPU transfers, and — worse — we would have credited it
with whatever happened overnight. Dropped entirely; the 8GB GPU has run 100k slices
for 1.34B steps without OOM, so the VRAM-insurance argument doesn't pay its overhead.

**Rejected: `policy_lr` 3e-4 → 2e-4.** This was sized to "compensate a 6x gradient
step increase" — but the true increase is 1.5x (6→9, because the buffer already
provided 3 batches/epoch, not 1). Cutting the LR by a third while raising steps by
half nets out to roughly **zero change in per-iteration policy movement**, while still
paying the ~9% throughput cost of the extra epoch: a package that silently does
nothing, discovered after a wasted night. The corrected package keeps 3e-4; the
projected KL (~0.004-0.007) remains under half the upper healthy bound.

The general lesson recorded here for the project report: *hyperparameters act through
the implementation, not through their names* — every knob was traced to the exact
loop that consumes it before being changed.

---

## 7. Deliberately NOT changed (and the next levers, in order)

| Not changed | Why |
|---|---|
| Rewards (v5) | Changed 500M steps ago; changing two variable families at once makes the night unattributable. |
| `gamma` 0.99 / `lambda` 0.95 | Changing the discount re-defines the value target mid-run — the critic would spend the night re-fitting a moved goalpost. A gamma bump (0.995) is a defensible *future* experiment, not an unattended one. |
| Reward annealing | Reward functions live in worker processes with no access to the global timestep — a clean implementation needs plumbing, not a pre-bedtime patch. |
| `exp_buffer_size` 300k | Interacts with the steps-per-iteration formula (§3a); shrinking it would *reduce* optimizer steps and fight the epochs change. |
| Opponent-pool / past-self training | The biggest structural upgrade left (self-play never punishes overcommits), but it's a training-loop feature, not a parameter flip. |

**If the morning is green** (KL < 0.008, clip < 0.12, entropy > 3.6, value loss < 0.05,
eval ≥ baseline): the next single-variable step the following night is `ppo_epochs=4`
**or** `policy_lr=3.5e-4` — walking KL stepwise toward ~0.01, one watched night at a
time. If entropy proved structural (unmoved at ~4.02), leave `ent_coef` alone and stop
spending nights on it.
