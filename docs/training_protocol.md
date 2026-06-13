# Training Protocol — Rules & Locked Parameters

Purpose: enforce the discipline that produces the best possible bot, and never
repeat the mistakes that cost us on earlier runs. Every rule maps to a real
lesson learned. **Do not deviate without an explicit, recorded reason.**

---

## 0. The lessons that created these rules

| What went wrong | The rule it created |
|---|---|
| FSP self-play wrapper corrupted ~30% of training data (off-policy action mismatch). Win rate plateaued at 12%; clean data doubled it to 26%. | **R1: clean data only** |
| `n_proc: 8` on a 24-core CPU — used 1/3 of the machine, ~1,500 steps/sec. | **R2: max the CPU** |
| `ppo_epochs: 6` (generic course guidance) vs RLGym domain standard 2-3. | **P: locked params** |
| `touch_ball: 0.1` — ~50x too small; barely rewarded the core skill. | **P + R3** |
| Over-engineered curriculum (kickoff/defensive/aerial) top bots don't use. | **R3: simple > clever** |
| A 30-episode eval read 8-10 and scared us; 100 episodes showed the truth. | **E1: 100-ep minimum** |
| Single-point extrapolation gave a wrong "1.7B for 50%" estimate. | **E2: milestone curve** |
| Trusted the noisy W&B reward over head-to-head eval. | **E5: eval is truth** |

---

## 1. LOCKED PARAMETERS (evidence-based; change only with cause)

Source: ZealanL RLGym-PPO-Guide + our own Option A result.

```yaml
learner:
  arch: large              # 1024 — match Diego's capacity; fill it with the long run
  n_proc: 20               # i9-14900HX = 24 cores; leave ~4 for OS+learner
  ppo_epochs: 3            # RLGym domain standard (2-3), NOT generic 3-10
  policy_lr: 2.0e-4        # start; anneal per §4
  critic_lr: 2.0e-4
  ppo_batch_size: 100_000  # MUST equal ts_per_iteration
  ts_per_iteration: 100_000
  ppo_minibatch_size: 50_000
  exp_buffer_size: 300_000 # ts x 3
  ppo_ent_coef: 0.01
  ppo_clip_range: 0.2
  gae_lambda: 0.95
  n_checkpoints_to_keep: 200
  timestep_limit: 2_000_000_000
self_play:
  enabled: false           # NON-NEGOTIABLE — clean symmetric self-play only
rewards:                   # touch-dominant early, simple & general (Nexto)
  touch_ball: 5.0          # the core-skill bootstrap; never below velocity-to-ball
  velocity_player_to_ball: 0.5
  face_ball: 0.1
  velocity_ball_to_goal: 0.5
  event(goal): 10.0        # moderate, NOT massive
state_setter: default 0.5 / random 0.5   # no heavy curriculum
obs: default (89-dim)      # changing obs = fresh run; don't mid-stream
action: lookup             # discrete 90 — proven
```

---

## 2. TRAINING DISCIPLINE RULES

- **R1 — Clean data only.** Never enable the FSP self-play wrapper. It mislabels
  ~30% of transitions. Stock symmetric self-play (both cars = current policy) only.
- **R2 — Max the CPU.** `n_proc: 20`. On the FIRST check of any run, verify
  throughput ≥ 4,500 steps/sec. If it's ~1,500, n_proc didn't apply — stop & fix.
- **R3 — Simple > clever.** No new reward terms, curriculum scenarios, or wrapper
  features mid-project without an eval proving they help. Nexto hit GC1 on simple rewards.
- **R4 — Don't change arch/obs mid-run.** Both force a fresh start. Decide once, commit.
- **R5 — Minimize interruptions.** Pause Windows Update (4 weeks). Lid-close = "Do nothing".
  Each stop costs a post-resume wobble + lost in-flight rollout.
- **R6 — One variable at a time.** When testing a change, change ONE thing so the
  eval delta is attributable (how Option A isolated the corruption).

---

## 3. EVALUATION DISCIPLINE

- **E1 — 100 episodes minimum.** 30 episodes has ±15% error and lied to us. Never decide on <100.
- **E2 — Milestone curve, not points.** Eval at fixed milestones (250M, 500M, 750M,
  1B, 1.5B, 2B) to see the trajectory. Single points mislead.
- **E3 — Eval vs real opponents, deterministic.** vs `marian_1900M` always; vs Diego's
  checkpoint once obtained. Use `--deterministic` (it's our best-foot-forward; stochastic
  was worse). Record W/L/D every time.
- **E4 — Preserve milestone checkpoints.** Copy each milestone to `checkpoints/<name>_<TS>`
  so the curve + fallbacks survive the keep=200 eviction window.
- **E5 — Eval is ground truth.** Trust head-to-head win rate over W&B Policy Reward
  (which is a noisy zero-sum artifact and lies in self-play).

---

## 4. LR ANNEALING SCHEDULE (manual, via stop/resume)

| Milestone | policy_lr / critic_lr | Trigger |
|---|---|---|
| 0 → ~300M | 2.0e-4 | start |
| ~300M | 1.0e-4 | bot scores reliably / eval win-rate climbing |
| ~700M+ | 0.8e-4 | refining mechanics |

---

## 5. INTERVENTION TRIGGERS (when to ACT vs leave alone)

Act only if one fires; otherwise **leave it alone and let it cook**:

- **KL divergence > 0.02 sustained** → drop ppo_epochs 3→2 OR LR one step.
- **Clip fraction > 0.10 sustained (3+ iters)** → drop LR one step.
- **Eval win rate DROPS between consecutive milestones** → regression; stop & investigate.
- **Throughput < 3,000 steps/sec sustained** → check n_proc / background CPU load.
- Everything else (reward swings, single-iter spikes, entropy drift within ±0.2) = noise. Ignore.

---

## 6. STOP / DONE CONDITIONS

- **Primary goal: beat Diego head-to-head (≥55% over 100 eps).** Once achieved, optionally
  keep going for margin, or stop and bank it.
- **Plateau: no eval win-rate improvement across 2 consecutive milestones (~500M of training)**
  → diminishing returns; stop, the bot is as good as this config gets.
- **Hard cap: 2B timesteps** → final eval, then done regardless.
- **Deadline guard: by June 18**, stop training and lock the final checkpoint — leave 3 days
  for eval, visualization, and the presentation.

---

## 7. THE ONE-LINE PHILOSOPHY

**Clean data, max compute utilization, simple rewards, proven hyperparams, measure
with 100-ep evals at milestones — and stop touching it between triggers.** Discipline
beats cleverness. We learned that the hard way.

---

## 8. HARD CONDITIONS — quality gates to build a beast

These are non-negotiable bars. "Make it a beast" = hit these, not add complexity.

### 8.1 Primary objective (the "beast" bar)
- **Beat Diego ≥ 60%** in a 100-ep deterministic head-to-head. (55% = competitive; 60%+ = clearly better.) Do not declare success below this.
- **Stretch vs Marian: ≥ 40%.** Clean data already hit 26% from a *corrupted* base; a
  fresh clean 1B run should plausibly clear 40%. If it does, the "Marian is unbeatable"
  conclusion is overturned.

### 8.2 Milestone gates (enforce at every eval point)
At 250M / 500M / 750M / 1B / 1.5B, eval vs Marian (and Diego once we have his checkpoint):
- **Win rate MUST strictly increase vs the previous milestone.**
- **2 consecutive flat/declining milestones = hard plateau.** Do NOT just keep grinding —
  diagnose first: (a) is LR annealed for this stage? (b) is the reward stage right (§8.3)?
  (c) is the bot passive (raise concede penalty)? Fix the cause, then continue.
- **Any milestone where win rate DROPS = stop immediately, investigate** (regression =
  something broke; never train through a regression).

### 8.3 Reward staging ladder (the disciplined way to get stronger)
Rewards must EVOLVE as the bot improves — touch-dominant early, scoring-dominant late.
Swap at milestones (stop, edit config, resume):

| Stage | Trigger | touch | velocity_to_ball | velocity_ball_to_goal | event(goal) |
|---|---|---|---|---|---|
| Bootstrap | 0–~150M | 5.0 | 0.5 | 0.5 | 10 |
| Scoring | bot hits ball reliably (~150–500M) | 1.0 | 0.3 | 0.8 | 15 |
| Mechanics | bot scores reliably (~500M+) | 0.3 | 0.1 | 1.0 | 20 |

Rationale (RLGym Guide + Nexto): bootstrap the core skill, then shift the gradient toward
actually winning. Keep it this simple — do NOT add aerial/dribble/positioning terms unless
a one-variable eval proves they help (R6). Most good behavior emerges from simple rewards.

### 8.4 Compute discipline (feed the beast)
- **Max CPU at all times** (n_proc=20). Throughput floor for large arch: **≥ 4,500/s
  steady-state** (warmup iterations exempt). Below that sustained → investigate.
- **Train long: 1B minimum, 2B target.** A large net under ~1B is under-filled (§ arch rule).
- **Uptime is a weapon.** Every hour down is timesteps Diego gains. Pause Windows Update,
  lid=Do-Nothing, keep plugged in + ventilated (thermal throttle = lost throughput).

### 8.5 The deadline ladder (work backward from June 21)
- **By June 18:** lock the final checkpoint. No more training.
- **June 18–20:** final 100-ep evals vs Diego + Marian, visualization, presentation.
- **June 21:** submit. The bot is whatever it is by June 18 — protect that buffer.

### 8.6 What "hard" does NOT mean
- ❌ More reward terms / curriculum scenarios (over-engineering cost us once already)
- ❌ Bigger arch than large (xlarge can't be filled in time — under-trained = worse)
  - **Recorded decision (2026-06-03):** explicitly evaluated xlarge `(2048,2048,1024,1024)`.
    The ZealanL RLGym-PPO-Guide *does* recommend exactly that config — BUT for training to
    convergence over weeks/months on a desktop GPU with no deadline. We are deadline-bound
    (15 days, laptop 4060), which flips the regime: when time is fixed, **fill-rate beats
    ceiling.** Projection at our ~2,500/s end-to-end throughput: large reaches ~2–2.5B
    (well-filled, strong); xlarge reaches only ~1.2–1.5B (under-filled = weaker than a
    filled large net). Diego is also on large, so we win by OUT-TRAINING on shared arch
    (clean data, KRC, reward ladder, uptime) rather than a capacity gamble. **Verdict: large.**
- ❌ Re-enabling FSP self-play (corrupts data)
- ❌ Tinkering between triggers (noise ≠ signal)
Hard = disciplined and relentless, not complicated.
