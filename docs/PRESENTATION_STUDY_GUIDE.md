# papaya_1024 — Final Presentation Study Guide

Everything needed to understand and defend the bot end-to-end. Read top to
bottom once, then drill the "Tough questions" section (§13) until the answers
are reflexes. Every number here is pulled from the current code, not memory.

## Contents
1. The 30-second pitch
2. The full pipeline (state → action)
3. The algorithm: PPO
4. What rlgym-PPO actually is
5. Why PPO and not something else
6. PPO pros & cons
7. ALL hyperparameters (table + meaning + why we changed them)
8. The reward function (ALL components)
9. The training curriculum (state setters + terminal conditions)
10. The version history (v1 → v7)
11. Evaluation metrics (how we measure strength)
12. wandb training metrics (how we read the dashboard)
13. Tough questions a professor will ask (with answers)
14. What I learned

---

## 1. The 30-second pitch

papaya_1024 is a 1v1 Rocket League bot trained with **Proximal Policy
Optimization (PPO)** via **self-play** in a headless physics simulator
(`rlgym_sim` / RocketSim). It observes the game as a **107-number vector**
(AdvancedObs), feeds it through a **1024×1024×1024 fully-connected neural
network**, and outputs **one of 90 discrete controller actions** (LookupAction)
15 times per second. It learned entirely from scratch by playing ~**2.85
billion** simulated timesteps against copies of itself, guided by a shaped
**reward function**. As of the final it beats a teammate's 2.1B-step "champion"
bot ~**66%** of games (stochastic, 50-game sample).

---

## 2. The full pipeline (state → action)

This is an **actor-critic** setup: **two separate networks** (the actor/policy
and the critic/value net). They are *not* one network with two heads — in
rlgym-PPO they are physically distinct nets with their own weights
(`PPO_POLICY.pt` and `PPO_VALUE_NET.pt`) and their own layer configs
(`policy_layer_sizes`, `critic_layer_sizes`). What they share is the **input**:
the same 107-dim obs is fed to both. The pipeline therefore **forks at the obs**.

Both networks start from the **same observation**, then do completely different
things. Here they are separately.

Every 8 physics ticks (tick_skip=8 → the bot acts at 120/8 = **15 Hz**):

**Diagram A — the ACTOR / policy net (the bot playing).** Runs in both training
and deployment:

```
Game state (RocketSim)
   │  positions, velocities, orientations of both cars + ball, boost pads
   ▼
AdvancedObs builder  ──►  107-dim observation vector (normalized)
   ▼
ACTOR / policy net : 107 → 1024 → 1024 → 1024 → 90   (ReLU hidden, Softmax out)
   ▼  probability over 90 actions
Sample (training) / argmax (deployment)  ──►  action index 0..89
   ▼
LookupAction parser  ──►  8-dim controller [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]
   ▼
RocketSim advances 8 ticks  ──►  next state, reward
```

**Diagram B — the CRITIC / value net (the coach grading each move).** Runs
during **training only**; discarded at deployment:

1. **Same input:** take the same 107-dim observation the actor saw.
2. **Forward pass:** `107 → 1024 → 1024 → 1024 → 1` (ReLU hidden layers, **no
   Softmax** on the output).
3. **Output V(s):** a single number = "how good is this state for me" (the
   expected future reward from here).
4. **Compute the advantage:** `advantage = actual return − V(s)` → was the action
   better or worse than the critic expected?
5. **PPO update:** nudge the **actor** toward actions with **positive** advantage
   (and away from negative ones); separately, train the critic to predict V(s)
   more accurately next time.

**The two networks, side by side:**

| | Actor (policy) | Critic (value) |
|---|---|---|
| Weights file | `PPO_POLICY.pt` | `PPO_VALUE_NET.pt` |
| Output | 90 numbers (action probabilities, via Softmax) | 1 number (state value V(s), no Softmax) |
| Job | *choose what to do* | *judge how good the state is* |
| Touches the controller? | yes | never |
| At deployment | **loads and plays** | **discarded** |

The critic exists only to compute the **advantage** (was the action better or
worse than the critic expected?), which is the signal PPO uses to update the
actor. Once training is done you only want the bot to *act*, so deployment loads
the actor alone.

Key spec facts (memorize these — they're the compatibility contract):
- **Obs:** AdvancedObs, **107-dim** for 1v1 (DefaultObs would be 89 — the wall
  we hit repeatedly).
- **Action:** LookupAction, **90 discrete** actions (a curated table of common
  controller combinations; fully-discrete heads train far more stably than
  continuous ones).
- **Net:** 1024×3 MLP, separate actor and critic, ReLU activations, Adam.
- **tick_skip 8** (15 Hz), **team_size 1**, **self-play** (the bot trains
  against a copy of its own current policy).

---

## 3. The algorithm: PPO

### What it is
PPO is a **policy-gradient, actor-critic, on-policy** reinforcement learning
algorithm (Schulman et al., 2017). "Policy gradient" = we directly adjust the
parameters of the policy network to make good actions more likely. "Actor-
critic" = a second network (critic) estimates state value to reduce the
variance of the gradient. "On-policy" = we learn only from data collected by the
*current* policy; there is no long-term replay buffer of old games.

### The loop (one iteration)
1. **Rollout:** run the current policy in many parallel game copies, collect a
   batch of (state, action, reward, value) transitions.
2. **Advantage estimation (GAE):** compute, for each step, how much better the
   action turned out than the critic predicted, using **Generalized Advantage
   Estimation** with discount **γ=0.99** and **λ=0.95**. (γ = how far ahead we
   care; λ = bias/variance trade-off in the advantage estimate.)
3. **Policy update (the PPO trick):** maximize the **clipped surrogate
   objective**
   ```
   L = E[ min( rₜ·Aₜ ,  clip(rₜ, 1−ε, 1+ε)·Aₜ ) ]
   ```
   where `rₜ = π_new(a|s) / π_old(a|s)` is the probability ratio and **ε=0.2**
   (clip_range). The `min`+`clip` is the entire point of PPO: it **stops the
   policy from moving too far in one update** even if the advantage says "do way
   more of this." That's what makes PPO stable where vanilla policy gradient
   blows up.
4. **Value update:** train the critic to predict the GAE return targets with
   **plain MSE loss** (no value clipping in our library).
5. **Entropy bonus:** add `+ent_coef · entropy(π)` to the objective to keep the
   policy exploring (don't collapse to one action too early).
6. Repeat. "Improvement" comes from collecting *fresh* data with an
   *ever-smarter* policy, not from re-watching old games.

### The one PPO equation to be able to explain
The clipped surrogate (step 3). If a professor asks "what makes it *Proximal*?"
— the clip keeps the new policy in a trust region *proximal* to the old one, so
each update is a small, safe step.

---

## 4. What rlgym-PPO actually is

Three distinct pieces people conflate — be precise:
- **RLGym / rlgym_sim** = the *environment*. `rlgym_sim` is a headless Python
  wrapper around **RocketSim**, a C++ reimplementation of Rocket League's
  physics. It's what lets us simulate millions of steps per minute with no game
  client. It defines the `obs_builder`, `action_parser`, `reward_fn`,
  `state_setter`, `terminal_conditions` API.
- **rlgym-ppo** = the *trainer* (Matthew Allen's library). It is a specific,
  production-grade PPO implementation built for rlgym. What it adds on top of
  "PPO the algorithm":
  - **Massively parallel rollouts:** spawns `n_proc` worker *processes* (we use
    18), each running its own RocketSim env, all feeding experience to one
    central **Learner**. This is why training is CPU-bound and scales with
    cores — RocketSim is fast, the bottleneck is collecting games.
  - **Batched GPU inference:** the Learner batches action requests across all
    workers (`min_inference_size`) so the GPU does one big forward pass instead
    of 18 tiny ones.
  - The **PPO update loop**, checkpointing, and wandb logging.
- **RLBot v5** = a *third* thing entirely — the framework that runs the trained
  policy in the *actual* Rocket League game (`rl-bot/` in our repo). Used only
  for human-vs-bot play, not training.

**How the update math actually works in this library** (I verified this in the
source — it's a common exam trap): gradient steps per iteration =
`ppo_epochs × floor(exp_buffer_size / ppo_batch_size)` = `3 × floor(300k/100k)`
= **9 optimizer steps per iteration**. `ppo_minibatch_size` is *gradient
accumulation* (splits a batch to fit VRAM, one optimizer step per batch), **not**
extra updates. Knowing this saved us from a self-cancelling hyperparameter
change (see §13, "the LR mistake").

---

## 5. Why PPO and not something else

This is the #1 "defend your design" question. Crisp comparisons:

| Alternative | Why not for this problem |
|---|---|
| **DQN / value-based** | Our action space is discrete (90), so DQN is *technically* possible — but DQN is off-policy and notoriously unstable in **self-play** (the opponent is non-stationary, the target keeps moving) and over **long episodes** with sparse goals. PPO's clipped on-policy updates are far more robust here. |
| **A2C / vanilla policy gradient** | Same family as PPO but no trust region — large updates destabilize training. PPO is "A2C that can't shoot itself in the foot." |
| **SAC / DDPG / TD3** | Off-policy *continuous*-control algorithms. Our controls are discretized via LookupAction (discrete trains more stably for RL bots), and these methods are fragile under self-play non-stationarity. |
| **Evolutionary strategies (ES/GA)** | Black-box, extremely sample-hungry. At billions of steps on one consumer GPU, gradient-based PPO is far more sample-efficient. |
| **Imitation / Behavior Cloning** | Needs a dataset of expert play and **caps the bot at the demonstrator's skill** — it can't discover superhuman mechanics. We wanted *emergent* skill from self-play, which only RL gives. |
| **Offline RL** | No fixed dataset exists; we generate unlimited fresh data cheaply in sim, so online PPO is the natural fit. |

**The clincher answer:** PPO is the *de facto standard* for Rocket League RL —
every strong community bot (Nexto, Seer, Element) is PPO-based — because it's
the best-understood, most stable algorithm for **massively-parallel, long-
horizon, self-play** control with a consumer GPU. We didn't pick it by
default; it's the right tool for exactly this shape of problem.

---

## 6. PPO pros & cons

**Pros**
- **Stable & robust:** the clip makes it forgiving of hyperparameters and
  imperfect tuning — critical for a 45-day project with no compute budget for
  giant sweeps.
- **Parallelizes beautifully:** scales with CPU cores via independent rollout
  workers.
- **Handles self-play:** on-policy updates track a moving opponent better than
  off-policy methods.
- **Simple, well-understood, battle-tested** in this exact domain.
- **Sample-efficient *for an on-policy method*** (the small experience-buffer
  reuse helps).

**Cons**
- **On-policy = data-hungry in absolute terms:** each batch is used for a
  handful of updates then largely discarded. Off-policy methods reuse data more,
  but trade away stability.
- **Sensitive to reward shaping:** a bad reward → confident wrong behavior. We
  felt this directly (boost-dumping, overcommitting) — see §8/§10.
- **Can plateau / cycle in self-play:** the policy can forget how to beat older
  styles ("strategy cycling"). Mitigated by long training and diverse spawns.
- **No convergence guarantee:** it's an empirical optimizer; "is it better?" is
  answered by evaluation, not theory.
- **Local optima:** entropy bonus fights this, but the policy can still settle
  into a comfortable suboptimal style.

---

## 7. ALL hyperparameters (current values + meaning + why we changed them)

Source of truth: the `Learner(...)` call in `diego-bots/papaya_1024.py`.

### Network & environment
| Param | Value | Meaning | History |
|---|---|---|---|
| `policy_layer_sizes` | **(1024,1024,1024)** | Actor MLP hidden sizes | 256→512→**1024** across experiments (bigger net = higher skill ceiling, needs more data). Fixed once papaya started — can't change without orphaning the checkpoint. |
| `critic_layer_sizes` | (1024,1024,1024) | Value net hidden sizes | matches actor |
| obs | AdvancedObs **107** | input vector | DefaultObs(89) → AdvancedObs(107) early in papaya; richer relative features |
| action | LookupAction **90** | discrete controller table | unchanged all project |
| `tick_skip` | 8 | act every 8 ticks (15 Hz) | community standard; never changed (would orphan net) |

### PPO core
| Param | Value | Meaning | History |
|---|---|---|---|
| `ppo_batch_size` | **100_000** | timesteps collected per update iteration | 50k→100k (smoother gradient, longer rollouts) |
| `ts_per_iteration` | 100_000 | env steps per iteration (= batch, full on-policy) | tied to batch |
| `exp_buffer_size` | **300_000** | rolling buffer of recent experience (= 3× batch) | a little controlled data reuse; gives 3 batches/epoch |
| `ppo_minibatch_size` | 100_000 | VRAM slicing (gradient **accumulation**, not extra steps) | full-batch |
| `ppo_epochs` | **3** | passes over the buffer per iteration → **9 grad steps/iter** | **v6: 2→3**. KL & clip fraction were 3-5× below the healthy band → updates too timid. +1 epoch = 50% more learning per sample. |
| `ppo_ent_coef` | **0.005** | entropy-bonus weight (exploration vs commitment) | 0.01 (pinned entropy at 4.0) → **v6: 0.005** (entropy fell to 3.53, play improved). Briefly tried 0.0075, **reverted** — the chart proved 0.005 is the strong regime. |
| `policy_lr` | **3e-4** | actor Adam learning rate | unchanged; made *explicit* in v6 (the library re-applies it on resume, so it was a hidden parameter) |
| `critic_lr` | 3e-4 | critic Adam learning rate | unchanged |
| `clip_range` (ε) | 0.2 | PPO trust-region clip | library default, unchanged |
| `gae_gamma` (γ) | 0.99 | reward discount (how far ahead we care) | default; ~0.99 ≈ a few seconds of foresight at 15 Hz |
| `gae_lambda` (λ) | 0.95 | GAE bias/variance knob | default |
| grad-norm clip | 0.5 | caps gradient size (extra stability) | library default |
| `standardize_returns` | True | normalize advantage targets | almost always helps |
| `standardize_obs` | False | obs already roughly scaled by AdvancedObs | off |

### Infrastructure (throughput / safety, NOT learning)
| Param | Value | Meaning | History |
|---|---|---|---|
| `n_proc` | **18** | parallel rollout worker processes | 14→16→18; ~1.5× physical cores |
| `min_inference_size` | 180 | min queued steps before a GPU forward pass | batches inference for GPU efficiency |
| `save_every_ts` | **1_000_000** | checkpoint cadence | **v6: 100k→1M** (100k = ~54MB write every 10s ≈ 200GB/night of SSD wear) |
| `n_checkpoints_to_keep` | **50** | rollback window | **v6: 5→50** (default 5 = ~50s rollback window overnight; 50 = ~1.4h) |
| `timestep_limit` | **5_000_000_000** | when `learn()` auto-stops | **v6: 2B→5B** (at 2B the run would *silently* stop itself mid-grind) |

**The hyperparameter philosophy to state out loud:** we treated tuning as
**single-variable experiments** — change one knob, train, measure against a
fixed benchmark, keep or revert. We never changed two learning-relevant knobs at
once, because then a result can't be attributed. (We even reverted an entropy
change when the data didn't support it.)

---

## 8. The reward function (ALL components)

The bot optimizes a **single scalar reward per step**, built as:

```
ZeroSumReward( CombinedReward( nexto_base + 11 custom components ) )
```

### The wrapper: ZeroSumReward (team_spirit=0, opp_scale=1)
**This is conceptually the most important piece.** It subtracts the opponent's
reward from yours: `my_reward − opponent_reward`. Without it, self-play
converges on *cooperative* behavior (both bots farm shaping rewards together)
that collapses against a real adversary. Zero-sum makes the game **competitive**:
what's good for me is bad for you. It's why the bot learned to actually *win*,
not just touch the ball a lot.

### The Nexto-style base (10 components) — `nexto_style.py`
A faithful re-implementation of the *shape* of the Nexto bot's reward. Event
rewards dominate; continuous signals give dense gradient. Internal weights:

| Component | Weight | What it rewards |
|---|---|---|
| VelocityPlayerToBall | **0.3** (v5 ↓ from 0.6) | moving toward the ball (lowered to curb overcommitting) |
| LiuDistancePlayerToBall | **0.3** (v5 ↓ from 0.7) | being close to the ball (exponential, peaks at contact) |
| VelocityBallToGoal | 2.0 | hitting the ball toward the opponent net (strongest offensive signal) |
| LiuDistanceBallToGoal | 1.0 | ball geometrically near the opponent net |
| AlignBallGoal (def+off) | 0.4 | positioning between ball and the right goal (shadow defense) |
| BallYCoordinate | 0.5 | ball in the opponent's half (territory) |
| FaceBall | 0.3 | orientation toward the ball |
| TouchBall | 5.0 | continuous bonus while in contact (rare → high weight) |
| SaveBoost | **0.3** (v5 ↑ from 0.05) | keeping boost in the tank |
| **EventReward** | **12.0** | the objectives: goal **+10**, concede **−10**, shot +1.5, save +3, touch +0.05, demo +0.5, boost_pickup +0.6 |

**Why events dominate (12×):** they are sparse but represent the *actual game
objective*. The continuous terms are scaffolding that gives the policy gradient
when no goal is happening; the event reward is the truth.

### The custom mechanic stack (11 components) — `custom_rl.py`
Added on top to teach specific skills the base learns slowly. Weights from the
current `CombinedReward`:

| Component | Weight | What it rewards / why |
|---|---|---|
| SupersonicReward | **0.03** (v5 ↓ from 0.25) | brief bonus at supersonic — slashed because it was making the bot **dump all its boost** for raw speed |
| AerialBallReward | 0.6 | being airborne near a high ball |
| AerialTouchReward | **2.0** (v5 ↑ from 1.5) | actually *touching* the ball in the air — the real aerial signal |
| BigBoostProximityReward | 0.8 | grabbing a big pad when low on boost AND the ball is far (smart economy, not chasing pads off a play) |
| BackboardDefenseReward | **0.7** (v5 ↑ from 0.45) | staying goal-side of the ball in your half — **anti-overcommit** |
| BallAwayFromOwnGoalReward | 0.6 | ball moving away from your own net (anti-own-goal under pressure) |
| DribbleToGoalReward | 0.20 | carrying the ball *toward* the enemy net (it used to dribble into walls) |
| **KickoffReward v2** | **0.5** (v7) | **time-decaying first touch**: pays once per kickoff, `1−t/4s` (early ≫ late) + a term for sending the ball to the opponent half. The v7 centerpiece. |
| RecoveryReward | 0.15 | orienting upright + toward motion while airborne *outside* an aerial — stop tumbling after hits, land control-ready |
| FlickReward | 1.0 | launching the ball up+forward off a dribble — the 1v1 finishing mechanic |
| BoostReserveReward | 0.4 (v5 new) | holding boost when *not* in an active play, so there's boost to recover/save after a mistake |
| ~~MaintainSpeedReward~~ | **removed v5** | was the "always floor it" boost-dumping driver |

**The reward-design lesson to articulate:** more components is not better. We
peaked at ~14 and several were *fighting each other* (Supersonic said "burn
boost," SaveBoost said "keep it"). v5's fix was to *reduce conflict*, not add
rewards. Strong bots (Nexto, the teammate's champion) use **leaner**,
event-dominated rewards — that's a known tension in our design and an honest
talking point.

---

## 9. The training curriculum (state setters + terminal conditions)

The bot doesn't just play full matches — we control *where episodes start* and
*when they end* to manufacture practice.

### State setters (where episodes spawn) — weights sum to 1.0
- **RandomState 0.35** — random positions/velocities; broad coverage.
- **RandomKickoffSetter 0.25** — the 5 canonical kickoff spawns (pairs with
  KickoffReward).
- **AerialSetupState 0.25** — ball high + grounded cars with boost, so aerial
  rewards actually fire (they're near-dead under random spawns).
- **DribbleSetupState 0.15** — a car carrying the ball toward the net, so flick
  practice happens.

**Why this matters:** a reward for aerials is useless if the bot is rarely in an
aerial situation. The state setters and the mechanic rewards are a **pair** —
the spawn creates the situation, the reward shapes the response.

### Terminal conditions (when an episode ends)
- **GoalScoredCondition** — a goal ends the point.
- **NoTouchTimeoutCondition (10s)** — discard idle states where nobody touches
  the ball.
- **KickoffStallCondition (4s)** — **v7**: ends kickoffs that never resolve (ball
  still at center after 4s) so kickoff drills recycle ~2.5× faster. It *arms
  only on a kickoff spawn and disarms the instant the ball leaves center*, so it
  can never cut normal mid-game play (we unit-tested this).

---

## 10. The version history (v1 → v7)

The bot's evolution is the story of the project — each version is a hypothesis
tested against evaluation.

| Version | Net / Obs | The change & why |
|---|---|---|
| baseline → nexto_rewards | 256×3, DefaultObs | first working PPO loop; ported a Nexto-style reward; reached ~130M |
| nexto_plus_kickoff_512 | 512×3, DefaultObs | bigger net + kickoff drills; the **1.18B** bot (prior best) |
| **papaya v4** | **1024×3, AdvancedObs** | fresh start: large net + richer obs + **RecoveryReward, FlickReward** + aerial/dribble drill states |
| **papaya v5** | same | watched it play: it **dumped boost** and **overcommitted**. Fix = *reduce conflicting rewards*: removed MaintainSpeed, slashed Supersonic, raised SaveBoost + added BoostReserve, lowered ball-chase, raised BackboardDefense + AerialTouch. The "lean toward the goal signal" pass. |
| **papaya v6** | same | **optimizer tune** (rewards frozen): KL/clip were far below the healthy band and entropy pinned at 4.0 → `ppo_epochs 2→3`, `ent_coef 0.01→0.005`, explicit LRs, and checkpoint-retention fixes for safe overnight runs. Entropy dropped to 3.53, play improved. |
| **papaya v7** | same | **fast-kickoff package** (the deterministic eval showed we won only ~2 of 5 kickoff lines): `KickoffReward v2` time-decaying first-touch + `KickoffStallCondition` for faster kickoff reps. |

**Critical engineering constraint that shaped all of this:** obs and
architecture are *fixed at birth*. A 1024×3 net can't load 512×3 weights, and a
107-dim obs net can't load an 89-dim checkpoint (`load_state_dict` is strict).
So changing obs or net width means **starting from scratch** — which is exactly
why v4 abandoned the 1.18B DefaultObs bot. Rewards, hyperparameters, and state
setters *can* change on a resume (they don't touch the network shape), which is
why v5/v6/v7 all continued the same checkpoint.

---

## 11. Evaluation metrics (how we measure strength)

Training loss tells you almost nothing about *strength* in self-play (see §12).
Real strength is measured by **playing a fixed opponent**:

- **Win rate vs a fixed reference bot** (Martin's 2.1B champion is ours). Headline number.
- **Deterministic vs stochastic** — a hard-won distinction:
  - **Deterministic (argmax)** + fixed kickoffs = the ~5 kickoff lines repeated.
    Coarse: a single line flipping swings it 5%. Ours hovers 45-50% — it's a
    *kickoff-line* check, not overall strength.
  - **Stochastic (sampling)** over 50+ games = every game differs → real signal.
    Ours is **66%**. **This is the benchmark.**
- **Kickoff benchmark** (`scripts/kickoff_benchmark.py`, the v7 attributable
  metric): time-to-first-touch, first-possession %, territory 3s after kickoff,
  per spawn. Baseline before v7: **20% first-possession** vs the champion.
- **Confidence intervals matter:** 6-4 (60%) and 9-11 (45%) on 10-20 games are
  *statistically indistinguishable* (p≈0.5). We only trust ≥50-game samples and
  always state the band. This kept us from chasing noise.

---

## 12. wandb training metrics (how we read the dashboard)

Exact code-level definitions (verified in `rlgym_ppo` source — be ready to
explain each):

| Metric | What it literally is | How to read it |
|---|---|---|
| **Policy Reward** | mean reward/episode this iteration | In **zero-sum self-play it sits at ~0 by construction** (your gain = opponent's loss). Flat-at-0 is *expected*, **not** "not learning." A classic trap. |
| **Policy Entropy** | randomness of the policy (max = ln(90) ≈ 4.5) | Falls as the bot commits. `ent_coef` sets its **equilibrium** (0.01→~4.0, 0.005→~3.53). Sudden crash to 0 = bad. Ours settled at 3.53 — committed but still exploring. |
| **Mean KL Divergence** | how far the policy moved this update (k3 estimator) | Healthy band ~0.008-0.02. Ours ran ~0.003 (too timid → v6 raised it via epochs). Blowing up = unstable; drop epochs/LR. |
| **SB3 Clip Fraction** | fraction of samples hitting the PPO clip | Healthy ~0.05-0.15. Ours ~0.03-0.08. Near 0 = nothing updating; near 0.5 = updates too aggressive. |
| **Policy Update Magnitude** | L2 norm of the actor's weight change this iteration | A *displacement* measure, not progress. Steady ~0.8-1.4. |
| **Value Function Update Magnitude** | same, for the critic | Plateau ~1.0 = critic at steady state (good). "Make it go up" is the wrong goal — it's a symptom, not a target. |
| **Value Function Loss** | critic MSE to GAE returns | Should be low & stable (~0.03). Rising/NaN = critic diverging. |
| **Collected / Overall Steps per Second** | throughput | Hardware/throttling diagnostic, not learning. Watch for overnight decay (heat-soak). |
| **Cumulative Model Updates** | total optimizer steps | Increments by **9 per iteration** (= our epochs×batches) — we used this to *prove* a hyperparameter change took effect. |

**The meta-point for the panel:** in symmetric self-play, the dashboard tells
you about *training health* (is it stable? is it updating?), not *skill*. Skill
is measured only by §11's evaluations. Confusing the two is the most common
mistake — and we explicitly didn't.

---

## 13. Tough questions a professor will ask (with answers)

**Q: Why PPO and not DQN/SAC/etc.?**
A: §5. Discrete-but-stable, on-policy suits self-play's non-stationarity, scales
on a consumer GPU, and it's the proven standard for this exact domain. DQN is
unstable in self-play; SAC/DDPG are continuous-control; BC caps at human skill.

**Q: Your Policy Reward is flat at zero — is it even learning?**
A: Yes — that's *expected* in zero-sum self-play (my reward = −opponent's, so
the mean is ~0). Learning is measured by win rate vs a fixed opponent (66%
stochastic), not by self-play reward.

**Q: You have ~20 reward components. Isn't that over-engineered?**
A: Honest answer: partly, yes. We peaked higher and found components *fighting*
(Supersonic vs SaveBoost). v5 was a deliberate *reduction* of conflict. Leaner,
event-dominated rewards (like the strong reference bots) are arguably better —
it's a known tension in our design and the main thing we'd do differently.

**Q: How do you know it actually improved, vs random noise?**
A: Fixed-opponent evaluation with confidence intervals. We treat <50-game
results as noise (6-4 and 9-11 are statistically identical) and only act on
50+-game stochastic evals plus the attributable kickoff benchmark.

**Q: Why did you change obs/net mid-project and "lose" a 1.18B bot?**
A: `load_state_dict` is strict — a 1024×3/107-dim net physically cannot load a
512×3/89-dim checkpoint. Upgrading capacity (bigger net) and information (richer
obs) required a fresh start; we judged the higher ceiling worth the restart, and
the result (papaya beats the old line) validated it.

**Q: Walk me through one PPO update.**
A: §3 steps 1-6. Rollout → GAE advantages (γ=0.99, λ=0.95) → clipped surrogate
(ε=0.2) policy step → MSE critic step → entropy bonus → repeat. The clip is what
keeps each step "proximal" and stable.

**Q: Is your bot one network or two? Do the actor and critic share weights?**
A: **Two separate networks** — actor (policy) and critic (value), each 1024×3
with its own weights (`PPO_POLICY.pt`, `PPO_VALUE_NET.pt`). They are *not* a
shared trunk with two heads; the only thing they share is the input obs. The
actor outputs 90 action probabilities and plays the game; the critic outputs a
single state-value used only to compute advantages during training, and is
discarded at deployment.

**Q: What was your biggest tuning insight?**
A: The dashboard said our updates were *too timid* — KL and clip fraction sat
3-5× below the healthy band, and entropy was pinned. We raised effective update
size (epochs 2→3) and let the policy commit (ent_coef 0.01→0.005). We also
caught a self-cancelling change in review: a proposed LR cut was sized to
"compensate a 6× step increase" that was actually only 1.5× (because the
experience buffer already provides 3 batches/epoch) — cutting LR would have
neutralized the whole change. We verified the library's actual math before
committing. **The LR mistake** is a great story to tell.

**Q: Deterministic vs stochastic — which is "the real bot"?**
A: We deploy **deterministic (argmax)** — it's stronger and less twitchy. But we
*evaluate* with stochastic-50 because deterministic + fixed kickoffs only tests
~5 scripted lines. The in-game match sits between (argmax play, but tick jitter
breaks perfect determinism), which is why kickoff lines still matter — hence v7.

**Q: What's your single biggest weakness?**
A: Kickoffs — deterministic eval showed we won only ~2 of 5 lines (20%
first-possession). v7 directly targets it with a time-decaying first-touch
reward. Honestly stating the weakness *and* the fix is stronger than claiming
perfection.

**Q: How is this reproducible / engineered?**
A: Versioned experiments (each config snapshotted to a run-history JSON),
checkpoint archival before risky changes, single-variable tuning, fixed-opponent
benchmarks, and a v6/v7 design doc per change. wandb tracks every run.

---

## 14. What I learned

- **Self-play reward ≈ 0 is normal.** Strength lives in evaluation, not loss
  curves. Separating "training health" metrics from "skill" metrics is the whole
  game.
- **Architecture and obs are immutable contracts.** Strict weight loading means
  a 1v1 bot's obs/action/net shape is decided on day one; changing it = restart.
  This wall appeared *constantly* (papaya vs old line, papaya vs DefaultObs
  teammates).
- **Reward shaping is a double-edged sword.** Dense shaping gave fast early
  learning but caused confident bad habits (boost-dumping, overcommitting), and
  too many components *conflict*. The fix is usually to *remove* signal, not add.
  Pinned entropy at 4.0 was the fingerprint of a conflicted reward landscape.
- **Hyperparameters act through the implementation, not their names.** Gradient
  steps = epochs×(buffer/batch); minibatch is accumulation, not extra steps. We
  almost made a no-op change by trusting the *name* of a knob over its *code*.
- **Statistics over vibes.** A 6-4 and a 9-11 feel different but are identical;
  acting on small samples wastes training days. Confidence intervals before
  decisions.
- **Single-variable discipline.** Change one learning-relevant knob at a time, or
  results aren't attributable. We even reverted an entropy change when its own
  chart argued against it.
- **MLOps pays off.** Versioned configs, run-history JSONs, checkpoint archives,
  fixed-opponent benchmarks, and design docs per change are what let us reason
  about a 2.85-billion-step training run instead of guessing.
- **Lean beats baroque.** The strongest reference bot used 4 reward components
  and beat its rivals; our 14 didn't make us 3× better. The hardest skill in RL
  isn't adding capability — it's resisting the urge to over-shape.
