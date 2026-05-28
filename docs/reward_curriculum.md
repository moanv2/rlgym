# Reward-Shaping Curriculum — Bot Design

How this bot goes from a random twitching car to something that plays 1v1 Rocket
League, over ~1 billion timesteps. The design follows the RLGym-PPO-Guide's
[rewards](../../rl-bot-tutorial-repo/RLGym-PPO-Guide/rewards.md) and
[making_a_good_bot](../../rl-bot-tutorial-repo/RLGym-PPO-Guide/making_a_good_bot.md)
sections.

## The core idea

A goal reward alone teaches nothing — a random bot scores roughly never, so there's
no signal to learn from. Instead we **shape**: start with dense, easy-to-trigger
rewards that get the bot moving and touching the ball, then progressively shift the
reward toward the things that actually win games (scoring, saves, demos), removing
the training wheels as the bot outgrows them.

We do this as **four chained stages**. Each stage is one experiment config that
**warm-starts the previous stage's policy** (`learner.init_from`) and changes the
rewards, learning rate, and batch size. The neural network is continuous across the
whole curriculum — only the reward landscape it's climbing changes.

> **Locked for the whole run:** `obs=advanced`, `action=lookup`, `arch=large`
> (1024,1024,1024). Changing any of these invalidates the carried-over network, so
> they're fixed in Stage 1 and inherited down the chain.

## The stages

| # | Config | Theme | What it learns | LR | `timestep_limit` (cumulative) |
|---|--------|-------|----------------|----|------------------------------|
| 1 | `exp_004_chase` | Chase | drive at, face & **strike** the ball net-ward; keep jumping | 2.0e-4 | 100M |
| 2 | `exp_005_score` | Score & control | score & **defend** (zero-sum, big goal/concede); low-speed near-ball drill; align positioning | 1.5e-4 | 250M |
| 3 | `exp_006_mechanics` | Mechanics | aerials, boost economy, **demolitions**, saves | 1.0e-4 | 500M |
| 4 | `exp_007_polish` | Polish | aggression, realistic spawns, refine | 0.8e-4 | 1.2B |

`timestep_limit` is **cumulative** because warm-starting carries over the agent's
step counter. So Stage 2 (limit 250M) trains ~150M new steps on top of Stage 1's
100M, and the big Stage 4 run carries the bulk: ~700M steps to ~1.2B total.

### Stage 1 — Chase (`exp_004_chase`)

> *"In this stage, you primarily should focus your bot on these 2 tasks: 1. Learn to
> touch the ball. 2. Don't forget how to jump."* — the guide

```
speed_toward_ball     7.0    # dense pull toward the ball
face_ball             1.0    # don't approach the ball backwards
in_air                1.5    # keep the (first) jump alive
double_jump           1.25   # reward spending the SECOND jump (dodge / double jump)
save_boost            0.5    # sqrt(boost)
velocity_ball_to_goal 15.0   # dominant DENSE signal: drive the ball net-ward (above speed_toward_ball)
strong_touch          10.0   # touch scaled by Δ|ball_vel| — real strikes pay, taps earn ~0
event(touch=1.0)      2.0    # tiny flat-touch BOOTSTRAP only (cold-start nudge)
event(goal,shot,boost_pickup) 25.0   # MODEST scoring + boost events
```

- **This run deviated from the script.** An earlier round farmed the flat `touch`
  reward (circle-dribbling, trading touches, never scoring). So the rewards here
  switched to `strong_touch` (rewards real strikes, ~0 for circle-taps) and made
  `velocity_ball_to_goal` the dominant dense signal — the bot earns more for
  *advancing* the ball than for merely *approaching* it. A tiny flat `touch` event
  (+2 / contact) is kept only as a from-scratch cold-start bootstrap.
- A `double_jump` reward was added so the bot stops bypassing its second jump, and
  the **modest** scoring + boost events were pulled forward from phases 5–6.
- **Still not zero-sum.** Movement/touch shaping is individual tuning, and the
  scoring events are a positive nudge only (no concede penalty). Zero-sum scoring
  with a concede penalty arrives in exp_005.
- Spawns are `RandomState` with cars airborne half the time → early air-control
  practice. If the bot stops jumping, raise `in_air` / `double_jump` further.
- **Advance on behavior, not on the ceiling**: move on once it reliably strikes the
  ball net-ward and still jumps — not when it hits 100M steps.

### Stage 2 — Score & control (`exp_005_score`)

The bot strikes the ball; now teach it to *score, defend, and stop being clumsy
right next to the ball*. Warm-starts from exp_004 via `learner.init_from`.

```
speed_toward_ball     5.0   # still strong: keeps flips instrumental for chase
face_ball             0.5
in_air                0.5   # insurance — jump habit is established
double_jump           0.4   # DOWN from 1.25 — was being spammed; flips now earn
                            # their value via speed_toward_ball (flip toward ball → speed)
save_boost            0.3
velocity_ball_to_goal 5.0   # dense path to goal; goal event leads
strong_touch          5.0
align_ball_goal       1.0   # NEW: per-step positioning — approach from the goal-side
event                 80.0  # goal +1, concede -1, shot +0.3, save +0.5, boost_pickup +0.1
```

- **Now zero-sum** (`team_spirit=0`, 1v1): scoring is naturally adversarial — the
  bot gets a real gradient for *defense* (penalty when the opponent scores).
- **Drastically larger goal events** (±80, up from exp_004's ±25). The bot already
  has the mechanics, so the goal can lead the gradient now without killing
  exploration. `save +0.5` gives positive credit for defending.
- **`double_jump` lowered, `speed_toward_ball` preserved.** The flat per-jump payout
  was being spammed; the bot now earns its flips' value *indirectly* through
  `speed_toward_ball` (a flip toward the ball nets more "approaching" reward).
- **`align_ball_goal`** is the targeted fix for clumsy near-ball play — it rewards
  positioning yourself such that pushing the ball points it toward the opponent's
  net, not just being near the ball.
- **State-setter mix**: 50% `random` / 30% `default` (kickoff) / 20% `near_ball`
  (low-speed near-ball drill). The 20% slice is direct reps in the exact situation
  the bot is currently weak at — usually far more impactful than reward shaping for
  a control problem.
- Flat-touch bootstrap dropped: warm-start ≠ cold start, `strong_touch` carries all
  contact shaping alone.

### Stage 3 — Mechanics (`exp_006_mechanics`)

Grow the toolkit (guide → Middle Stages):

```
velocity_ball_to_goal 1.0
strong_touch          1.5
air_touch             1.0   # NEW: aerial touches = min(airtime frac, ball-height frac)
save_boost            0.5   # NEW: sqrt(boost) — stop wasting boost
event                 20.0  # + save +0.5, demo +0.3, boost_pickup +0.5
+ small speed_toward_ball / in_air insurance
```

- `air_touch` is gated on **both** airtime and ball height, so the bot has to commit
  to a real aerial instead of farming cheap high wall-reads.
- `save_boost` uses `sqrt` (rlgym_sim built-in) so 0→20 boost matters more than
  80→100, matching how boost actually works.
- **Demolitions** turned up here and zero-sum — a demo both rewards you and hurts the
  victim, which is exactly what a demo should do.

### Stage 4 — Polish (`exp_007_polish`)

The long run. Fundamentals are in place; refine and remove training wheels (guide →
Later Stages):

- **`aggression_bias = 0.2`**: `concede = -0.8` instead of `-1.0`. The guide's most
  general anti-passiveness lever — it nudges the bot to *fight to score* rather than
  always sit back and defend. Tune it up if the bot plays too passively.
- **Shaping shrunk to a minimum** (`speed_toward_ball` 0.1, no `in_air`):
  `velocity_ball_to_goal` + the event reward now carry the learning.
- **Game-realistic spawns**: 70% random / 30% kickoff (`weighted_sample`).
- **Lower LR (0.8e-4) + lower entropy (0.008)** → exploit and sharpen, explore less.
- *Let it cook.* Flat graphs here often mean it's improving at everything at once
  rather than visibly changing how it plays.

## A note on zero-sum scope

The repo's `ZeroSumReward` wraps the **whole** combined reward (all-or-nothing per
stage), so Stages 2-4 also make the small residual shaping rewards zero-sum. With
their weights kept low this adds only minor noise. If you want the guide's finer
control ("a reward should only be zero-sum if it's beneficial for the opponent to
prevent it"), the clean upgrade is per-component zero-sum in
[`rewards/builder.py`](../src/rlbot/rewards/builder.py) — left as a future change.

## Running the curriculum

```bash
# Stage 1: from scratch (only stage with no init_from)
python -m rlbot.training.train --config configs/experiments/exp_004_chase.yaml

# When touches are reliable (watch wandb / the visualizer), move on. Stage 2 picks
# up Stage 1's latest checkpoint automatically via learner.init_from.
python -m rlbot.training.train --config configs/experiments/exp_005_score.yaml
python -m rlbot.training.train --config configs/experiments/exp_006_mechanics.yaml
python -m rlbot.training.train --config configs/experiments/exp_007_polish.yaml
```

Re-launching the **same** stage resumes its own latest checkpoint (the warm-start
only fires when the stage has no checkpoints of its own yet). See
[training_guide.md → Curriculum chaining](training_guide.md#curriculum-chaining).

### When to advance a stage

Advance when the current stage's headline behavior is solid, not on a fixed step
count — the `timestep_limit` values are generous ceilings, not targets:

| Stage | Advance when… |
|-------|---------------|
| 1 → 2 | the bot reliably strikes the ball net-ward and still jumps |
| 2 → 3 | non-trivial self-play goal rate AND it doesn't always concede; less clumsy near the ball |
| 3 → 4 | it attempts aerials, collects boost, and goes for demos |
| 4 → 🏁 | it stops improving in eval vs. older checkpoints |

## The custom rewards

New reward functions live in [`src/rlbot/rewards/custom.py`](../src/rlbot/rewards/custom.py)
(registered names in parentheses); `save_boost` is registered from rlgym_sim in
[`builtin.py`](../src/rlbot/rewards/builtin.py).

| Name | What it does |
|------|--------------|
| `speed_toward_ball` | fraction of car speed pointed at the ball, clamped at 0 (never punishes moving away) |
| `in_air` | 1 while airborne, 0 on ground — keeps the (first) jump alive |
| `double_jump` | +1 the step the *air* jump is spent (`has_flip` True→False airborne + jump pressed) — keeps the second jump / dodge alive |
| `strong_touch` | touch reward scaled by the change in ball velocity |
| `air_touch` | `min(airtime_frac, ball_height_frac)` on an aerial touch |
| `save_boost` | `sqrt(boost_amount)` (rlgym_sim built-in) |
| `align_ball_goal` | per-step cosine alignment of (car ↔ ball) with (own_goal ↔ car) and (car ↔ opp_goal) — rewards approaching the ball from the goal-side (rlgym_sim built-in) |

Each is unit-tested in [`tests/test_rewards.py`](../tests/test_rewards.py).

## The custom state setters

Beyond `RandomState` and `DefaultState` (kickoff), the curriculum can mix in
targeted drills. Defined in [`src/rlbot/state_setters/`](../src/rlbot/state_setters/),
unit-tested in [`tests/test_state_setters.py`](../tests/test_state_setters.py).

| Name | What it does |
|------|--------------|
| `near_ball` | spawns each car 500–1500 uu from a settled ground ball, random yaw, low speed — direct reps for the low-speed near-ball moment that random spawns rarely produce |
| `weighted_sample` | mixes children by per-reset weighted sampling — local replacement for rlgym-tools' `WeightedSampleSetter` (the v2 install no longer exposes the old `extra_state_setters` path) |
