# Reading the training graphs

A field guide to the numbers you see in the console iteration report and the wandb dashboard. Every chart in this doc maps to a specific lever you can pull when something looks wrong.

> The TL;DR: only one chart answers "is the bot getting better" — **Policy Reward**. Every other chart answers "is the training process healthy". A flat Policy Reward with otherwise healthy stats means your rewards or rollouts are wrong. A spiking KL or collapsing entropy means PPO itself is breaking down regardless of what the reward is doing.

---

## 1. Policy Reward — the only chart that matters for "is the bot improving"

**What it is:** average reward per episode in the most recent rollout batch. For our setup, summed across all the reward components (player_to_ball, ball_to_goal, event rewards), wrapped in ZeroSumReward.

**Healthy shape:**

```
reward
   ▲           ╱──────────  plateau eventually
   │        ╱─╯
   │     ╱─╯
   │ ╱──╯  noisy but trending up
   │╱
   └──────────────► timesteps
```

- Noisy upward trend over millions of timesteps.
- Plateaus are normal and expected; that's when you should change rewards, increase model capacity, or add curriculum.
- The first few hundred thousand timesteps can be flat or even slightly negative — that's the random exploration phase paying its initial cost.

**Bad shapes:**

| Shape | Meaning | Fix |
|---|---|---|
| Flat for 5M+ steps from start | Rewards too sparse or contradictory | Add denser shaping (e.g. velocity rewards), or reweight events vs continuous |
| Going up, then crashing | Policy collapsed (lost diversity) | Increase `ppo_ent_coef`, lower `ppo_epochs`, or reset to a checkpoint before the crash |
| Wildly oscillating | KL divergence is too large each update | Lower `ppo_epochs` from 2 to 1, or shrink `ppo_minibatch_size` |
| Stuck negative | Reward weighting is punishing the bot more than rewarding it | Audit reward weights; rewards normalize but their *signs* matter |

**Important caveat:** the numeric value is meaningless on its own. With ZeroSumReward and normalization, a Policy Reward of 0.5 vs 1.2 tells you very little. The *trajectory* of the value matters, not the level.

---

## 2. Policy Entropy — the exploration health gauge

**What it is:** the Shannon entropy of the policy's action distribution, averaged across the batch. For our 90 discrete actions, the theoretical maximum is `ln(90) ≈ 4.50`.

**Healthy shape:**

```
entropy
   ▲ ●
4.5│ ╲
   │  ╲___
   │      ╲___        slow, monotonic decay
   │          ╲___    over millions of steps
2-3│              ╲___
   └──────────────────► timesteps
```

- Starts near `ln(N_actions)` — the policy is fully random.
- Drops smoothly over millions of steps as the policy commits to specific behaviors in specific states.
- For a well-trained Rocket League bot at the end of training, entropy typically lives in the 2.0–3.0 range. Not 0 — a bot should still have stochasticity for exploration in new states.

**Bad shapes:**

| Shape | Meaning | Fix |
|---|---|---|
| Stuck at max (4.5) | Bot can't differentiate states; policy stays uniform | Train longer, OR fix reward signal, OR check obs is informative |
| Crashes to 0 quickly | Premature exploitation; bot locks in to one degenerate behavior | Bump `ppo_ent_coef` from 0.01 to 0.02 or 0.03 |
| Bounces back up | Reward landscape shifted (curriculum change, etc.) — usually fine | Watch a few iterations, intervene only if it stays elevated |

**Quick rule:** if entropy is below 1.0 before 10M timesteps, you probably converged too fast and the bot is stuck in a local optimum.

---

## 3. Mean KL Divergence — "how much did the policy just move"

**What it is:** the KL divergence between the policy *before* and *after* one PPO update. Measures how aggressively the gradient step changed behavior.

**Healthy values:** 0.005 to 0.05, stable across iterations.

**Healthy shape:**
```
KL
   ▲
0.05│  ▁▁▁▂▂▁▂▁▂▁▁▂▁▁  small, steady noise
0.01│
   └──────────────────► timesteps
```

**Bad shapes:**

| Shape | Meaning | Fix |
|---|---|---|
| Spikes above 0.1 | Update was too aggressive; the policy moved into territory not supported by the data | Lower `ppo_epochs` from 2 to 1 |
| Slowly creeping up over training | Learning rate too high for the current stage | Add learning rate decay (advanced — outside simple_bot.py) |
| Dropping to 0 | Policy stopped updating | Check if `ppo_ent_coef` is too high or `ppo_minibatch_size` mismatched |

**Why it matters:** PPO's whole pitch is "trust region" updates — small, safe steps that don't blow up training. If KL is uncontrolled, you've lost PPO's main safety guarantee.

---

## 4. SB3 Clip Fraction — "what fraction of gradient updates got clipped"

**What it is:** the fraction of samples in the batch where PPO's surrogate objective was clipped (i.e., the update wanted to move too far and PPO held it back).

**Healthy values:** 0.05 to 0.20.

| Value | Meaning |
|---|---|
| ~0.00 | Nothing is updating. Either nothing to learn, or learning rate too low. |
| 0.05–0.20 | Healthy. PPO is doing exactly what it's designed to do. |
| 0.30–0.50 | Updates are aggressive; some clipping is fine but this is a yellow flag. |
| >0.50 | Most updates are being clipped — your batch and policy are way out of alignment. Usually correlated with high KL. |

**Tightly coupled with KL Divergence.** If KL is high, Clip Fraction is also high. Treat them together.

---

## 5. Value Function Loss — "how wrong is the critic"

**What it is:** mean squared error between the critic's predicted future reward and the actual computed return. The critic is internal to PPO — you never query it at inference time — but if it's badly miscalibrated, advantage estimates are wrong and the policy can't update sensibly.

**Healthy shape:** drops over the first few hundred thousand timesteps, then settles into a noisy plateau. Absolute scale doesn't matter (depends on reward magnitudes); the *trend* does.

```
VF loss
   ▲
   │╲
   │ ╲___
   │     ──___           early drop, then noisy plateau
   │          ──______
   └──────────────────► timesteps
```

**Bad shapes:**

| Shape | Meaning | Fix |
|---|---|---|
| Increasing over time | Critic is diverging; rewards may be drifting or non-stationary | Check if `standardize_returns=True` (it should be) |
| NaN | Numerical blowup, very bad | Restart from earlier checkpoint, lower learning rate |
| Permanently huge (~10× the policy reward) | Critic is bad at predicting your return scale | Sometimes a larger critic network helps; mostly ignorable |

---

## 6. Policy / Value Function Update Magnitude

**What they are:** L2 norm of the parameter delta after each PPO update. "How much did the network's weights actually shift?"

**Healthy:** decreases slowly over training as the policy approaches a stable solution. Absolute scale depends on the network size.

**Bad:** sudden spikes correlate with KL spikes and usually mean a bad batch caused an oversized step. Isolated spikes are tolerable; sustained inflation isn't.

---

## 7. Collected Steps per Second & Overall Steps per Second

**What they are:** raw throughput. Collected = pure rollout phase, Overall = including the PPO update. The ratio tells you how update-bound vs rollout-bound you are.

**Healthy:** stable values throughout training. For an RTX 4070 Laptop in our setup, expect ~5,000–8,000 collected steps per second, ~4,000–6,000 overall.

**Bad:** degrading throughput over time usually means a memory leak in rollout workers. Restart the training run from the latest checkpoint.

---

## 8. Cumulative Timesteps

**What it is:** total environment steps the policy has been trained on across all sessions. Persists across resumes because the Learner saves it in `BOOK_KEEPING_VARS.json`.

**For Rocket League bots:**

| Cumulative timesteps | What to expect |
|---|---|
| 500k | Random twitchy behavior, occasional accidental ball touches. Baseline only. |
| 5M | Bot reliably chases the ball, makes contact, may shoot toward goal sometimes. |
| 25M | Recognizable Rocket League gameplay. Defends, shoots, basic boost management. |
| 100M | Strong bot. Decent 1v1 opponent for a casual human. |
| 500M+ | What top community bots ship at. Aerials, mechanical skill, game awareness. |

For the class final (45-day timeline, 1v1 vs a classmate), the realistic target is **30M to 80M cumulative timesteps**. That's reachable on a single laptop GPU running mostly overnight for two weeks.

---

## What to look for when comparing two runs in wandb

Open the dashboard, select your runs, use the chart options:

- **Smoothing slider** (top right of each chart) — bump to 0.7 or higher. Raw curves are noisy; smoothed curves reveal trends.
- **X-axis: `Cumulative Timesteps` not `_step`** — comparing by timesteps is fair, comparing by wall clock isn't.
- **Color by `Config.experiment_name` or wandb group** — see which experiment family beats which.
- **Y-axis: Policy Reward is the headline. Always.** Everything else is diagnostic.

**Run comparison heuristics:**

- If run A has higher Policy Reward at the same timestep as run B → A is better, period.
- If both reach the same Policy Reward but A uses fewer timesteps → A is **sample efficient**, which matters when compute is the bottleneck.
- If A has higher Policy Reward but its KL spiked midway through → A might be a fluke that collapses if you train it longer.

---

## When to stop training

In order of how often you'll hit them:

1. **Policy Reward plateaued for 5M+ timesteps** — the bot has saturated this reward function. To improve further, change the *rewards*, not the *training duration*.
2. **You're out of compute budget** — class final is in N days, you need a usable model now.
3. **Entropy crashed below 1.0** — bot stopped exploring. Reset or restart with more entropy bonus.
4. **Wall clock is too painful to wait** — start a longer run overnight, evaluate when you wake up.

**Resist the urge to stop early just because the curve is noisy.** Smoothing slider at 0.9 reveals the real trend.

---

## Quick decision table

| You see | Most likely cause | First thing to try |
|---|---|---|
| Reward flat, entropy high, KL low | Bot can't get any signal | Audit rewards. Are weights right? Is the bot getting any reward at all? |
| Reward flat, entropy low | Bot converged to a bad local optimum | Reset from a much earlier checkpoint, increase `ppo_ent_coef` |
| Reward goes up then crashes | Policy collapsed during update | Resume from the checkpoint *before* the crash, lower `ppo_epochs` |
| Reward improving slowly but steady | Healthy, just slow | Keep training, or try a bigger network |
| KL Divergence spiking | Updates too aggressive | Lower `ppo_epochs` to 1, or shrink minibatch |
| Entropy crashed to 0 fast | Lost exploration | Bump `ppo_ent_coef` from 0.01 to 0.03 |
| Critic VF loss exploding | Numerical instability | Ensure `standardize_returns=True`, restart |
| Steps/sec degrading | Memory leak in workers | Ctrl+C and resume from latest checkpoint |

---

## Glossary in one sentence each

- **Policy** — the neural network that maps observation → action probabilities. The "bot's brain."
- **Critic / Value network** — separate network that predicts expected future reward from a state. Internal PPO machinery only.
- **Episode** — one game from spawn to goal/timeout. Never saved to disk.
- **Rollout** — one batch of fresh gameplay collected by the current policy (50k timesteps in our setup).
- **Iteration** — one rollout plus one PPO update on it (~50k timesteps each).
- **On-policy** — PPO can only train on data its current policy generated, hence no replay buffer.
- **Entropy bonus** — extra term in the loss that rewards the policy for staying random; controls exploration.
- **KL divergence** — distance between two probability distributions; here, between old and new policy.
- **Clip fraction** — share of samples where PPO's clipping mechanism activated. Healthy 5–20%.
- **GAE** — Generalized Advantage Estimation, the algorithm PPO uses to turn rewards into "how much better than expected was this action."
