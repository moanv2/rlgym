# papaya_1024 — Presentation Source (Deck + Appendix)

Source file for slide generation. Merges the **code-verified spec**
(`PRESENTATION_STUDY_GUIDE.md`) with the **class-alignment map**
(`papaya_1024_class_alignment.md`, Manero S2–S12). Every claim is traceable to
both the live code and a course slide.

**Notation:** `S9 s40` = Session 9, slide 40. ✅ taught · ⚠️ named/implied · ❌ bridge given.

**Your two presentation parts (the deck below):**
- **PART A — RLGym, the environment & the MDP.** Introduce the tooling and the
  agent's world: rlgym / rlgym_sim / rlgym-ppo, env, observation, actions,
  states, MDP. (4 slides, ~2 min)
- **PART B — papaya, my bot.** What it is, how I built it, results, the loss.
  (4 slides, ~2 min)
- **PART 2 = the appendix** — backup slides + Q&A ammo; surface on a probe.

Each deck slide has on-slide bullets, a `SAY` speaker note (timed), a `VISUAL`
hint, and course-slide refs (`S9 s40`). Whole talk ≈ 4 min.

---
---

# PART 1 — THE DECK

# ░ PART A — RLGym, the environment & the MDP ░

## A1 — "Three tools, three jobs: rlgym · rlgym_sim · rlgym-ppo"

**On slide**
- **rlgym** — the **environment API**: defines what the agent sees (obs), can do
  (actions), is rewarded for (reward), and where episodes start (states)
- **rlgym_sim** — the **physics**: a headless C++ reimplementation of Rocket
  League (**RocketSim**) → millions of steps/min, no game client
- **rlgym-ppo** — the **trainer**: many parallel env workers feeding one PPO
  learner *(S9 s42)*
- *(RLBot v5 = separate — runs the finished bot in the real game)*

**SAY (~30s):** "Three tools people conflate. RLGym is the environment interface
— it defines what the agent sees, the actions it can take, the reward, and where
episodes start. rlgym_sim is the physics: a headless C++ reimplementation of
Rocket League called RocketSim, which lets us simulate millions of steps a minute
with no game running. And rlgym-ppo is the trainer — it spins up many parallel
copies of the environment and runs the PPO learning loop. A fourth thing, RLBot
v5, is separate: that's what runs the finished bot in the actual game."

**VISUAL:** 3 stacked boxes (env API → physics sim → trainer) + a side chip "RLBot v5 = real game".

---

## A2 — "Underneath, it's a Markov Decision Process"

**On slide — the MDP 5-tuple *(S2)*** mapped to Rocket League:
- **S** (state) = the **107-dim observation**
- **A** (actions) = **90 discrete** controller actions (LookupAction)
- **P** (transition) = **RocketSim** physics, `P(s'|s,a)`
- **R** (reward) = my **reward function** (one scalar per step)
- **γ** (discount) = **0.99** (≈ a few seconds of foresight at 15 Hz)
- **Goal:** learn a policy **π(a|s)** that maximizes expected discounted reward

**SAY (~30s):** "Under the Rocket League costume it's a textbook Markov Decision
Process — the Session 2 five-tuple. The state is the 107-number observation; the
action space is 90 discrete controller moves; the transition function is
RocketSim's physics; the reward is my reward function, one number per step; and
gamma, the discount, is 0.99 — about a few seconds of foresight. The agent's
entire job is to learn a policy — a mapping from state to action — that maximizes
expected future reward."

**VISUAL:** S/A/P/R/γ as five labelled tiles, each pointing to its Rocket League part.

---

## A3 — "The agent's world: observation, action, episode"

**On slide**
- **Observation (S):** AdvancedObs, **107 numbers** — ball + both cars'
  position / velocity / orientation, boost, + relative positions. The agent's
  *partial* view of the true sim state.
- **Action (A):** LookupAction, **90 discrete** — a curated table of controller
  combos. We **discretize** the naturally-continuous controls *(S9 s47)* — far
  more stable to train than continuous heads.
- **States / the loop:** a **state setter** spawns each episode (kickoff, aerial,
  dribble, random); the agent acts every **8 ticks → 15 Hz**; the episode ends on
  a **goal or timeout** (terminal conditions).

**SAY (~35s):** "Three pieces define what the agent experiences. The observation
is a 107-number vector — the ball, both cars' positions, velocities and
orientation, boost, plus relative positions — a partial view of the true
simulator state. The action space is 90 discrete moves: we deliberately
discretize the continuous controls into a lookup table, because discrete heads
train far more stably. And the loop: a state-setter decides where each episode
starts — a kickoff, an aerial drill, a random scramble — the agent acts fifteen
times a second, and the episode ends on a goal or a timeout."

**VISUAL:** obs-vector breakdown; the 90-action grid; an episode timeline (spawn → 15 Hz steps → goal/timeout).

---

## A4 — "rlgym-ppo runs PPO — by self-play"

**On slide**
- **Self-play:** the bot trains against a **copy of its own current policy**
- **PPO** = policy gradient + actor-critic + on-policy + **clipped objective** *(S9 s38, s43)* — the clip keeps each update in a **trust region** (stable)
- **Two networks:** **actor** picks the action (plays); **critic** scores the state `V(s)` → the **advantage** that trains the actor (training only)
- **Why PPO** (not DQN / SAC / BC): on-policy + stable + the proven self-play standard *(each rejection = an S8–S9 slide)*

**SAY (~35s):** "rlgym-ppo ties it together: many copies of the environment in
parallel, all feeding one central learner, and the bot trains by self-play —
against a copy of itself. The algorithm is PPO: policy-gradient, actor-critic,
on-policy, with a clipped objective — the clip keeps each update small and stable.
Two networks: an actor that picks the action and a critic that scores the state to
produce the learning signal. We use PPO, not DQN or SAC or behavior cloning,
because it's the stable on-policy standard for self-play — and each alternative is
its own slide in the course."

**VISUAL:** parallel-workers → learner diagram; PPO 4-word badge; actor/critic split.

---
---

# ░ PART B — papaya: my bot ░

## B1 — "papaya_1024 — my bot"

**On slide**
- A 1v1 bot trained **from scratch** (no human demos) by **self-play PPO** in that env
- Spec: **AdvancedObs 107** · **LookupAction 90** · **1024×3 net** · tick_skip 8 (15 Hz)
- **~3.5 billion** self-play steps

**SAY (~20s):** "That's the framework — now my bot, papaya. It's a 1v1 agent
trained entirely from scratch, no human demonstrations, by self-play PPO inside
that environment. Its spec: the 107-dim AdvancedObs, the 90-action lookup, a
1024-by-3 network — and it trained for about three and a half billion steps."

**VISUAL:** bot name big; the spec as cards; a still of papaya in rlviser.

---

## B2 — "How I built it: engineer the reward, iterate, measure"

**On slide**
- Algorithm is off-the-shelf — **the craft is the reward** *(reward design = S1; R = S2)*
- **ZeroSumReward** (`my_reward − opp_reward`) → keeps self-play **competitive**, not cooperative *(adversarial case, S9 s12)*
- **~20 components** = sparse goal events (**±10**) + dense scaffolding *(sparse/dense, S1)*
- Iterated **v4 → v7**, each a hypothesis tested vs a fixed opponent
- **Lesson:** more components ≠ better — v5 *removed* conflicting signals

**SAY (~35s):** "The algorithm is off-the-shelf — the real work was the reward. I
wrapped it zero-sum so self-play stays competitive instead of the two bots quietly
cooperating to farm rewards. I shaped about twenty components: sparse goal events
plus dense scaffolding that gives gradient between goals. Then I iterated from
version 4 to 7, each a hypothesis tested against a fixed opponent. The biggest
lesson: more components is not better — several were fighting each other, and v5
was about *removing* conflict, not adding signal."

**VISUAL:** ZeroSum formula; a "v4→v5→v6→v7" timeline strip; "21 → fewer, cleaner".

---

## B3 — "Results & the honest loss: papaya finished 3rd"

**On slide**
- **Measured by win-rate vs fixed opponents** — in zero-sum self-play the training
  reward sits at **≈ 0 by design** (my gain = opponent's loss), so the loss curve
  tells you nothing about skill.
- **3000-game team tournament:**
  Martin 9B (73%) › Nachi 2.9B (63%) › **papaya (56%)** › Marco (35%) › Marian (23%)
  → **I finished 3rd** (big samples → the ranking is real, not noise)
- **Why I lost — three causes:**
  - **Out-trained:** Martin ran **9B steps (≈2.5× mine)** — at that scale raw compute wins
  - **Out-committed:** Nachi beat me with **fewer steps than I had** → it wasn't just training time
  - **Over-shaped reward (the real one):** ~12 components fighting → policy never
    *sharpened* → **"stochastic-strong, deterministic-exploitable"** (worse at
    argmax — the mode it actually deploys in)
- **The humbling twist:** I thought I was winning (66%, 50 games) → **3000 games
  put me 3rd** — statistics-over-vibes, on my *own* bot
- **Next time:** lean event-driven rewards · train it to *commit* · 1000-game evals from day one · opponent diversity

**SAY (~45s):** "How do you measure a self-play bot? Not by the loss — in zero-sum
self-play the reward sits at zero by construction, so you measure by playing fixed
opponents. We ran a three-thousand-game team tournament, and I'll be straight:
papaya finished third. Martin's nine-billion-step bot won at 73%, Nachi second,
papaya third at 56. Why did I lose? Three reasons. Martin out-trained everyone —
about two-and-a-half times my steps. But Nachi beat me with *fewer* steps than I
had, so it wasn't just training time. My policy never sharpened — I over-shaped
the reward with twelve components fighting each other, so papaya is strong when
sampling but exploitable when greedy, which is exactly the mode it deploys in. And
the humbling part: early on I thought I was beating the champion 66 percent — but
that was fifty games; three thousand games corrected me to third. The
statistics-over-vibes lesson, on my own bot. Next time: leaner rewards, train it
to commit, and big-sample evaluation from the start."

**VISUAL:** 5-bot ranking bar (papaya highlighted at 3rd) → podium (🥇 Martin /
🥈 Nachi / 🥉 papaya); the 3 causes as icons; a "66% (50 games) → 3rd (3000 games)"
correction arrow; the 4 next-time bullets. End card.

> **Timing:** PART A (A1–A4) ≈ 2 min · PART B (B1–B3) ≈ 1.5 min → **~3.5 min total.**
> If you're tight: shorten A4 (PPO detail is in the appendix) and B2 to one line.
> The B3 results-and-loss is your close — the honest post-mortem is the strongest
> moment in the talk, so keep it whole.

---
---

# PART 2 — APPENDIX (backup slides + Q&A ammo)

> One appendix slide per lettered section. Don't present; surface on a probe.

## A. Class-alignment map — the whole bot → a slide

**A1. Algorithm core — all Session 9, near 1:1**

| Bot piece | Class home | Cov. |
|---|---|---|
| PPO (overall) | S9 s27, s38, s43 | ✅ |
| Policy gradient `∇θ log π(a\|s)·A` | S9 s16 (REINFORCE), s4 | ✅ |
| Actor `πθ(a\|s)` | S9 s31, s38 | ✅ |
| Critic / value net `Vφ(s)` | S9 s31–32, s38 | ✅ |
| Advantage `Aₜ = Rₜ − Vφ(sₜ)` | S9 s39 | ✅ |
| Clipped surrogate, `ε=0.2` | S9 s40 (states ε≈0.1–0.2) | ✅ |
| Full loss `L_CLIP − c₁·L_VF + c₂·H(π)` | S9 s41 | ✅ |
| Critic MSE `(G−V)²` | S9 s41 (L_VF) | ✅ |
| Entropy bonus `ent_coef` | S9 s41 (c₂·H), s32 | ✅ |
| K epochs `ppo_epochs=3` | S9 s42 ("K≈3–10") | ✅ |
| Discount `γ=0.99` | S2; S9 s52 | ✅ |
| The PPO loop (collect→advantages→K-epoch SGD→clip→repeat) | S9 s42 | ✅ |

**A2. Architecture & function approximation**

| Bot piece | Class home | Cov. |
|---|---|---|
| MLP function approx (1024×3) | S6 (FA w/ DQN); S9 (net=policy) | ✅ |
| Adam, `lr=3e-4` | S6; S9 s52 | ✅ |
| PyTorch (not Keras) | S7 s24 (eager-first for RL) | ✅ |
| Discrete actions (LookupAction 90) | S9 s47–48 (discretizing continuous control) | ⚠️ |
| Stochastic (train) vs deterministic (deploy) | S9 s11–12 | ✅ |

**A3. Environment = Session 2 MDP**

| Bot piece | Class home |
|---|---|
| RocketSim/rlgym | S2: obs=S, LookupAction=A, physics=P, reward_fn=R, γ=0.99 |
| 107-dim AdvancedObs | S1: observation of the true sim state (partial observability) |
| per-step scalar reward | S1 reward types; S2 `R(s,a,s')` |
| terminal conditions | S2 horizon `H` |

**A4. "Why not X" — every rejected choice is a slide**

| Rejected | Class home |
|---|---|
| On-policy, **no replay buffer** | S5 on/off-policy; S9 s3, s35 (the *absence* is the point) |
| not DQN | S9 s3 + s47 (off-policy, discrete-only) |
| not SAC | S9 s54–58 (off-policy continuous, 5 nets + replay) |
| not TD3/DDPG | S9 s48–52 (deterministic continuous) |
| not Behavior Cloning | S8 s13–15 (caps at expert, distribution shift, no reward) |

**A5. Bottom line:** algorithmic = **all S9**; environmental = **S2 MDP**; the
two things with no class home (RocketSim/RLBot tooling, reward-engineering craft)
both reduce to *"shaping R (S1/S2) and parallelizing the s42 rollout."*

---

## B. Things "not in the syllabus" → bridge to a slide (a professor WILL probe these)

Never say "that's not in the course." Always *"that's engineering on top of <slide>."*

- **GAE (`gae_lambda=0.95`).** S9 s42 *names* it; the machinery is the **n-step
  bias–variance dial, S9 s33**. → *"GAE is the s33 dial — λ trades trust in
  bootstrapped V(s) vs real rewards; 0.95 leans Monte-Carlo: low bias, slightly
  more variance."*
- **Self-play / non-stationary opponent.** **S9 s12** flags adversarial games as
  where a *stochastic* policy beats a deterministic one. → *"Self-play is the s12
  adversarial case; the moving opponent is also why on-policy (s3) beats off-policy
  here — the target keeps shifting."*
- **ZeroSumReward.** No slide → it's **reward design (S1)** making **R (S2)**
  competitive, tied to **s12** adversarial framing.
- **Reward shaping (~21 comps).** Concept = **S1 sparse vs dense**. → *"Events are
  the sparse true objective (goal ±10); continuous terms are dense scaffolding —
  the S1 tradeoff. Count + conflict-tuning is beyond syllabus; the anchor is S1."*
- **Parallel rollout workers (`n_proc=18`).** = **step 1 of the s42 loop**
  ("collect T steps") parallelized across 18 envs. Conceptually one rollout.
- **rlgym-ppo internals (9 grad steps = epochs×buffer/batch).** = s42's "K epochs
  of minibatch SGD"; minibatch is a VRAM split, not extra updates. *(the LR-mistake
  story, F.)*
- **wandb / MLOps.** No slide, but **every metric = an s41 loss term** (see E).
- **State setters / curriculum (aerials, flicks, kickoffs).** Lives entirely in
  **R (S2)** + the env's **start-state distribution**; changes what's rewarded and
  where episodes start — *not* the learner.

**Sessions that don't map (so you can say so):** S3 DP (we're model-free — opposite
corner); S4 MC/TD (indirect: REINFORCE=MC is back, s16; GAE=TD side, s33); S10/S11
(no component); S12 RLHF (only that **s43 calls PPO the RLHF default** — your algo
is the RLHF workhorse, the bot isn't RLHF).

---

## C. The full pipeline (two diagrams)

**Diagram A — ACTOR / policy (plays; training + deployment):**
```
Game state → AdvancedObs (107-dim) → ACTOR 107→1024→1024→1024→90 (ReLU, Softmax)
  → 90 action probs → sample(train)/argmax(deploy) → action 0..89
  → LookupAction → 8-dim controller → RocketSim advances 8 ticks → next state, reward
```
**Diagram B — CRITIC / value (training only; discarded at deploy):**
1. Same 107-dim observation.
2. Forward pass `107→1024→1024→1024→1` (ReLU hidden, **no Softmax**).
3. Output `V(s)` — one number, "how good is this state."
4. `advantage = actual return − V(s)`.
5. PPO update → nudge **actor** toward positive-advantage actions; train **critic**
   to predict V(s) better.

**Why 1 number** (critic) **vs 90** (actor): the actor *picks* among 90 actions;
the critic *scores* the single state. 90-per-action = Q(s,a) = DQN; PPO uses the
state-value baseline V(s) — lower variance.

**Why ReLU then Softmax** (actor): ReLU = cheap non-linearity, no vanishing
gradient (hidden layers); Softmax = turn 90 logits into a probability distribution
to sample from (output). Critic has no Softmax — a value is a raw real number
(can be negative).

---

## D. ALL hyperparameters (current values + meaning + history)

**PPO core**
| Param | Value | Meaning | Class | History |
|---|---|---|---|---|
| `ppo_batch_size` | 100k | steps collected per iteration | s42 | 50k→100k |
| `exp_buffer_size` | 300k | recent-experience buffer (3× batch) | s42 | gives 3 batches/epoch |
| `ppo_minibatch_size` | 100k | VRAM split (gradient accumulation, NOT extra steps) | s42 | — |
| `ppo_epochs` | **3** | K passes → **9 grad steps/iter** | s42 "K≈3–10" | **v6: 2→3** (updates too timid) |
| `ppo_ent_coef` | **0.005** | entropy bonus (explore vs commit) | s41 c₂·H | 0.01→0.005 (entropy 4.0→3.53); tried 0.0075, **reverted** |
| `policy_lr` / `critic_lr` | 3e-4 | Adam LR | s52 | unchanged; made explicit v6 |
| `clip_range` ε | 0.2 | trust-region clip | s40 | default |
| `gae_gamma` γ | 0.99 | discount | s52 | default |
| `gae_lambda` λ | 0.95 | bias/variance (GAE) | s33 | default |
| grad-norm clip | 0.5 | gradient cap | — | default |
| `standardize_returns` | True | normalize advantage targets | — | on |

**Network/env**
| Param | Value | Note |
|---|---|---|
| `policy_layer_sizes`/`critic_layer_sizes` | (1024,1024,1024) | separate actor & critic |
| obs / action | AdvancedObs 107 / LookupAction 90 | fixed at birth |
| `tick_skip` | 8 | act at 15 Hz |
| team_size / mode | 1 / self-play | 1v1 |

**Infra (throughput/safety, NOT learning)**
| Param | Value | Why |
|---|---|---|
| `n_proc` | 18 | parallel rollout workers (~1.5× physical cores) |
| `min_inference_size` | 180 | batch GPU inference |
| `save_every_ts` | 1M | v6: 100k→1M (SSD wear) |
| `n_checkpoints_to_keep` | 50 | v6: 5→50 (rollback window) |
| `timestep_limit` | 5B | v6: 2B→5B (don't self-stop mid-grind) |

**Tuning philosophy:** single-variable experiments — change one knob, train,
measure vs a fixed benchmark, keep or revert. Never two learning knobs at once.

---

## E. wandb metrics → S9 loss terms ("the dashboard is the s41 loss, live")

| Metric | What it is | Class | Read |
|---|---|---|---|
| **Policy Reward** | mean episode reward | return `J(θ)` s3/s16 | **≈0 in zero-sum self-play by design** — not "broken" |
| **Policy Entropy** | randomness (max ln90≈4.5) | `H(π)` s41/s32 | `ent_coef` sets equilibrium (0.01→4.0, 0.005→3.53) |
| **Mean KL Divergence** | how far policy moved | clip/trust-region s40 | healthy ~0.008–0.02; ours ran ~0.003 (too timid) |
| **SB3 Clip Fraction** | % samples hitting clip | s40 | healthy ~0.05–0.15 |
| **Value Function Loss** | critic MSE to GAE returns | `L_VF` s41 | low & stable ~0.03 |
| **Policy/Value Update Magnitude** | L2 of weight change/iter | — | displacement, not progress; plateau = steady state |
| **Cumulative Model Updates** | total optimizer steps | — | +9/iter → proves epochs change took effect |

**Meta:** in symmetric self-play the dashboard shows *training health*, not
*skill*. Skill = section F evals. Confusing the two is the #1 mistake.

---

## F. Evaluation (how we actually measure strength)

- **Win rate vs a fixed reference** (teammate's 2.1B champion). Headline.
- **Deterministic (argmax) vs Stochastic (sampling):** deterministic + fixed
  kickoffs = only ~5 scripted lines (hovers 45–50%, a *kickoff* check). Stochastic
  50+ games = real signal → **66%** (the benchmark). We deploy argmax, evaluate
  stochastic → 66% is a conservative floor.
- **Kickoff benchmark** (v7 attributable metric): time-to-touch, first-possession
  %, territory +3s. Baseline before v7: 20% first-possession.
- **Confidence intervals before decisions:** 6-4 and 9-11 are statistically
  identical (p≈0.5); trust only 50+-game samples.

---

## G. Reward function (ALL components)

**Wrapper:** `ZeroSumReward(team_spirit=0, opp_scale=1)` → `my − opp`. The most
important conceptual piece: makes self-play competitive (else it converges
cooperative). *(reward design S1, adversarial S9 s12.)*

**Nexto base (10) — event-dominated:** VelocityPlayerToBall 0.3 · LiuDistance-
PlayerToBall 0.3 · VelocityBallToGoal 2.0 · LiuDistanceBallToGoal 1.0 ·
AlignBallGoal 0.4 · BallYCoordinate 0.5 · FaceBall 0.3 · TouchBall 5.0 · SaveBoost
0.3 · **EventReward 12.0** (goal **+10**, concede **−10**, shot 1.5, save 3, touch
0.05, demo 0.5, boost_pickup 0.6).

**Custom mechanic stack (11):** Supersonic 0.03 · AerialBall 0.6 · **AerialTouch
2.0** · BigBoostProximity 0.8 · BackboardDefense 0.7 · BallAwayFromOwnGoal 0.6 ·
DribbleToGoal 0.20 · **KickoffReward v2 0.5** (time-decaying first-touch) · Recovery
0.15 · Flick 1.0 · BoostReserve 0.4. *(MaintainSpeed removed in v5.)*

**Why events dominate (12×):** sparse but they're the actual objective; continuous
terms are dense scaffolding for gradient between goals *(S1 sparse/dense)*.

**Curriculum (state setters, sum 1.0):** RandomState 0.35 · RandomKickoff 0.25 ·
AerialSetup 0.25 · DribbleSetup 0.15. Reward + spawn are a **pair** — a reward for
aerials is dead if the bot is never in an aerial.

**Terminal conditions:** GoalScored · NoTouchTimeout 10s · **KickoffStall 4s** (v7,
recycles unresolved kickoffs; can't cut mid-game play).

---

## H. Version history (each version = a tested hypothesis)

| Ver | What changed & why |
|---|---|
| baseline→nexto_rewards | first PPO loop; Nexto-style reward (256×3, DefaultObs) |
| nexto_plus_kickoff_512 | bigger net + kickoff drills → prior best 1.18B |
| **papaya v4** | fresh: **1024×3 + AdvancedObs** + Recovery/Flick + aerial/dribble drills |
| **papaya v5** | watched it play → *reduce conflicting rewards*: −MaintainSpeed, Supersonic↓, SaveBoost↑ +BoostReserve, ball-chase↓, defense↑, aerial↑ |
| **papaya v6** | optimizer tune (rewards frozen): epochs 2→3, ent_coef 0.01→0.005, explicit LRs, checkpoint-retention overhaul |
| **papaya v7** | fast-kickoff: KickoffReward v2 (time-decaying) + KickoffStall |

**Hard constraint:** obs/net fixed at birth (strict `load_state_dict`) → v4
abandoned the 1.18B DefaultObs bot; v5/v6/v7 resume the same checkpoint (rewards/
HPs/spawns don't touch net shape).

---

## I. Tough Q&A (drill these)

- **"Reward flat at 0 — is it learning?"** → expected in zero-sum self-play; skill = win-rate (66%), not loss.
- **"~20 rewards — over-engineered?"** → partly yes; v5 *reduced* conflict; lean event-dominated is arguably better. Own it.
- **"Improved or noise?"** → fixed-opponent eval w/ CIs; <50 games = noise.
- **"Lost a 1.18B bot — why?"** → strict load; bigger net + richer obs need a restart; the result validated it.
- **"One PPO update?"** → rollout → GAE advantages (γ0.99,λ0.95) → clipped surrogate (ε0.2) actor step → MSE critic step → entropy bonus → repeat.
- **"One network or two?"** → **two separate nets** (PPO_POLICY.pt, PPO_VALUE_NET.pt), share only the obs; critic discarded at deploy.
- **"Biggest tuning insight?"** → updates were *too timid* (KL/clip 3–5× below band); raised effective step (epochs 2→3) + let it commit (ent 0.01→0.005). The **LR mistake**: a proposed LR cut was sized for a 6× step increase that was really 1.5× — would've cancelled the change; caught it by reading the library's actual math.
- **"Why discretize a continuous control problem?"** → s47: continuous heads train unstably & blow up the action space; the 90-action LookupAction table is the pragmatic middle.
- **"Biggest weakness?"** → kickoffs (won ~2 of 5 lines); v7 targets it directly.

---

## J. What I learned

- Self-play reward ≈ 0 is normal — separate *training health* metrics from *skill* metrics.
- Architecture/obs are immutable contracts — strict weight loading = decided day one.
- Reward shaping is double-edged — dense shaping caused confident bad habits; the fix is usually *removing* signal. Pinned entropy at 4.0 = fingerprint of a conflicted reward landscape.
- Hyperparameters act through the *implementation*, not their names (the 9-grad-steps / minibatch / LR-mistake lesson).
- Statistics over vibes — a 6-4 and a 9-11 are identical; CIs before decisions.
- Single-variable discipline — even reverted an entropy change when its own chart argued against it.
- MLOps paid off — versioned configs, run-history JSONs, checkpoint archives, fixed-opponent benchmarks, a design doc per change → you can reason about a 2.85B-step run.
- Lean beats baroque — the strongest reference bot used 4 reward components; resisting over-shaping is the hard skill.

---

## K. Post-mortem — why papaya finished 3rd (defense depth)

The full root-cause for slide B4. Final tournament: **3000 stochastic games**,
Martin 9B (73%) › Nachi 2.9B (63%) › **papaya 56%** › Marco (35%) › Marian (23%).
Deterministic put papaya 4th (44.8%, ≈ tied with Marco) — a tighter, defensible
result than any earlier small-sample eval.

**Cause 1 — out-trained (compute).** Martin reached **9B steps (~2.5× papaya's
3.5B)**. At this scale raw training volume is the single biggest lever, and his
bot dominates both modes. Honest: with equal compute the gap would shrink, but he
spent it and I didn't.

**Cause 2 — out-committed, not out-trained (Nachi).** Nachi beat me at **2.9B —
*fewer* steps than my 3.5B** — so steps alone don't explain it. His policy
committed to cleaner decisions where mine stayed diffuse (entropy ~3.53, never
sharpened). More training of *my* recipe wouldn't have closed this; the recipe was
the problem.

**Cause 3 — over-shaped reward → an exploitable argmax policy (the real lesson).**
papaya carried **~12 reward components that conflicted** (boost vs speed,
chase vs hold). Symptoms, all observed: entropy pinned high the whole run; the bot
boost-dumped and overcommitted (v5 tried to fix it); and crucially papaya is
**"stochastic-strong, deterministic-exploitable"** — it scores higher sampling
(56%) than greedy (45%). Why that matters: **deployment plays argmax**, and an
argmax policy repeats the *same* line every game, so any opponent whose argmax
counters my kickoff/approach beats me *every* time. Leaner, event-driven rewards
(Martin's 4 components, Nachi) produce sharper, less-exploitable argmax play.

**Cause 4 — I misjudged my own strength (process).** My 50-game eval said "66% vs
the champion → I'm #2." The 3000-game tournament said **#3**, and even my own
300-game run wrongly ranked me above Nachi. Small samples flattered me; only the
big run told the truth. This is the *statistics-over-vibes* lesson landing on the
person who wrote it.

**What I'd do differently next time:**
1. **Lean, event-dominated reward** (≤5 components) — shape less, let goals drive it; the bots that beat me did exactly this.
2. **Train the policy to commit** — tune entropy/updates so *argmax* (the deployed mode) is decisively strong, not just sampling.
3. **Big-sample evaluation from day one** — 1000+ game fixed-opponent evals as the standing metric; never trust a 50-game read.
4. **Opponent diversity / past-self pool** — self-play vs only the current self leaves exploitable lines; a frozen-checkpoint pool punishes them.
5. **Budget compute deliberately** — if raw strength is the goal, fewer experiments + more steps on one good recipe beats many shaped recipes.

**The one-sentence version (good for the panel):** *"I lost because I out-engineered
the reward instead of out-training the policy — a leaner reward and a sharper,
less-exploitable argmax would have beaten a bot with fewer steps than mine."*
