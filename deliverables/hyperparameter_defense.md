# Defending the hyperparameters and the wandb graphs

Everything below is from the real run (wandb run twotz849, resilient-dragon-12) and the
recipe file. Numbers are measured over 75,517 logged iterations across the 4.46B to 8.24B
window. Use this to answer questions with confidence.

## The one sentence story

All three graphs say the same thing: the policy makes small, stable, controlled updates and
keeps exploring. The clip fraction is tiny and flat, the KL divergence is tiny and flat, and
the entropy stays high. That is a well behaved, converged training run, not a thrashing one.

## Your measured numbers

| Metric | Mean | Std | Range | What it means |
|---|---|---|---|---|
| SB3 Clip Fraction | 0.0101 | 0.0012 | 0.0001 to 0.017 | about 1 percent of samples hit the clip boundary |
| Mean KL Divergence | 0.0013 | 0.0001 | 0.00003 to 0.014 | the policy barely shifts per update |
| Policy Entropy | 4.02 | 0.01 | 3.97 to 4.06 | high, near the max of ln(90) = 4.50 |
| Policy Reward | 11.2 (final) | | rising | learning was still happening |
| Value Function Loss | 0.029 (final) | | low | the critic fits well |

## Your hyperparameters and why

| Setting | Your value | Library default | Why yours |
|---|---|---|---|
| policy_lr and critic_lr | 1e-4 | varies | low rate, small safe steps |
| ppo_epochs | 2 | 10 | fewer reuse passes, far less policy drift per batch |
| ppo_ent_coef | 0.01 | 0.005 | double default, keeps exploration alive |
| ppo_clip_range | 0.2 | 0.2 | standard PPO trust region |
| gae_gamma (discount) | 0.99 | 0.99 | default horizon |
| gae_lambda | 0.95 | 0.95 | default GAE smoothing |
| ts_per_iteration | 50,000 | | fresh data per update |
| ppo_batch_size | 50,000 | | large batch, low variance gradient |
| exp_buffer_size | 150,000 | 100,000 | a bit more history |
| standardize_returns | true | | stabilizes the value target scale |
| standardize_obs | false | | obs already in sane physical ranges |
| n_proc | 22 | | 22 parallel sims on a 24 thread CPU |

## Graph 1: SB3 Clip Fraction (the flat 0.01 line Diego liked)

**What it is.** PPO clips the update so the new policy cannot move too far from the old one in
one step. With a clip range of 0.2, any sample whose probability ratio (new over old) leaves
the band 0.8 to 1.2 gets clipped. The clip fraction is the share of samples that get clipped.
Yours sits at about 1 percent and barely moves.

**Why yours is low and flat.** Three reasons that all push the same way:
1. Only 2 PPO epochs, against a default of 10. Each batch is reused far fewer times, so the
   policy drifts much less per iteration.
2. Learning rate 1e-4 is low, so each gradient step is small.
3. A 50,000 sample batch gives a low variance gradient, so steps are smooth, not jumpy.
Add that by the logged window the policy is already mature, so updates are fine refinements.

**Why that is good.** A low, flat clip fraction means PPO is respecting its trust region almost
perfectly. No update is destructive, there is no instability or policy collapse, and the reward
still climbed. That is exactly what you want late in a long run.

**Honest caveat, be ready for it.** The graph shows your 4.46B to 8.24B window next to papaya's
0 to 3.5B window. Part of the gap is that you are looking at a more mature phase, where clip
fractions are naturally lower. So do not claim "lower clip fraction means a better bot." Claim
"stable, controlled optimization with no instability." That is the accurate and defensible point.

## Graph 2: Mean KL Divergence

**What it is.** The average KL divergence between the old and new policy after each update. It
measures how far the policy distribution moved. Yours is about 0.0013.

**Why yours is tiny.** Same cause as the clip fraction. Few epochs, low rate, big batch. The
policy stays inside a small trust region every update.

**Why that is good.** Common PPO practice targets a KL around 0.01 to 0.02 and stops early if it
blows past that. Yours sits well under that the whole time, so there is no risk of a runaway
update or catastrophic forgetting. Stable convergence.

**Caveat.** Like the clip fraction, very small KL also means slow per update change, which is
normal this late. The reward rising while KL stayed tiny is the proof that learning was still
efficient, not stalled.

## Graph 3: Policy Entropy (if it comes up)

**What it is.** How spread out the action distribution is. Max for 90 actions is ln(90) = 4.50.
Yours is 4.02, about 89 percent of max, and flat.

**Why high.** The entropy coefficient is 0.01, double the default 0.005, which deliberately keeps
the policy exploring rather than collapsing onto one rigid line of play.

**Why that matters.** This is why both deploy modes are strong. We deploy argmax (deterministic)
for peak play at 82 percent game win rate, and because the policy kept its entropy, the stochastic
(sampling) deploy still wins 73 percent. A collapsed, overconfident policy would not be robust in
both modes.

## Likely questions and crisp answers

**"Your clip fraction is only 1 percent, is your policy even learning?"**
Yes. Policy reward kept rising across the whole window and the bot finished first in the
tournament. A low clip fraction at 4 to 8 billion steps is mature phase stability, not stalling.
Earlier in training the updates moved more.

**"Why discount 0.99 and not 0.995?"**
We used the library default 0.99. A higher 0.995 lengthens the credit assignment horizon, which
is a known lever for a long horizon game like this. We prioritized training scale and a proven
reward stack. Raising gamma is on our future tuning list.

**"Only 2 PPO epochs, the default is 10?"**
Deliberate. Fewer epochs means less off policy drift per batch, which is exactly why the clip
fraction and KL stay tiny and training is stable. With 22 parallel sims we collect fresh data
fast, so we favor more fresh data over more reuse of old data.

**"Entropy coefficient 0.01 is high, is the policy too random?"**
It keeps exploration alive and makes the policy robust. We deploy argmax for peak play. The
retained entropy is why the sampling deploy mode also holds up.

**"Why does your wandb line start at 4.4 billion?"**
That logging run captures the 4.46B to 8.24B window. Training ran from 0 to 10 billion across
runs. The full curve is the concatenation of the runs.

**"Standardize returns but not observations, why?"**
Standardizing returns keeps the value target on a stable scale, which keeps the advantages well
scaled and feeds the stable updates you see. Observations are already in sane physical ranges
(positions divided by 2300, and so on), so we left them raw.
