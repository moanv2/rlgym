# How our 1v1 Rocket League bot was made and trained

## TL;DR
It is a neural network that watches the game as a list of numbers, and 15 times per second
picks one of 90 preset controller inputs. It got good by **playing millions of games against
itself** with **PPO** (a reinforcement-learning algorithm), getting rewarded for useful play and
goals, and slowly adjusting itself to do more of what wins. We gave it a head start by briefly
copying a stronger bot, then let it practice solo until it surpassed it. ~9 billion steps of training.

## The simple idea (plain English)
- The bot is a brain (neural net). **Input:** a snapshot of the game (its car, the ball, the
  opponent, as numbers). **Output:** which action to take right now.
- It learns by **trial and error at huge scale**: play, see what scored points, and nudge the
  brain so good moves become more likely and bad moves less likely. Repeat billions of times.
- Analogy: like a player who first copies a pro to learn the basics, then drops the copying and
  grinds solo ranked until they are better than the pro.

## The four building blocks
| Part | What it is |
|---|---|
| **What it sees (observation)** | `AdvancedObs` — 107 numbers: its car position/velocity/rotation/boost, the ball, the opponent, all from the bot's point of view |
| **What it can do (actions)** | `LookupAction` — 90 preset controller combos (throttle, steer, jump, boost, handbrake, air control). It picks one every step, so it is a 90-way classification |
| **Its brain (network)** | A 1024 x 3 fully-connected network, actor-critic: the *actor* picks actions, the *critic* judges how good a situation is |
| **What it is rewarded for** | A shaped, zero-sum reward stack: small rewards for useful behavior (driving at the ball, facing it, pushing it toward the enemy goal, defending, aerial touches) plus big rewards for outcomes (goal +1, concede -1, shots, demos) |

## How it learned (PPO self-play)
1. **Self-play:** the bot controls *both* cars, so its opponent is always a copy of itself and
   gets harder as it improves. No human data needed.
2. **Collect then update:** it plays ~50,000 steps across **22 simulations running in parallel**,
   then PPO does a careful, clipped update of the brain toward higher-reward behavior. Repeat.
3. **Diverse practice:** episodes start from a random situation 70% of the time and a normal
   kickoff 30%, so it learns to handle all states, not just kickoffs.

## The one clever trick: kickstarting (distillation)
We did not start from zero. Early on, an extra term gently pulled our bot toward imitating a
stronger existing bot (Diego's "papaya"). That influence was **annealed to zero over the first
150M steps**, after which it was pure self-play RL — so it could go on to **beat** the bot it
learned from, not just copy it (DeepMind, "Kickstarting Deep RL", 2018).

## Training parameters (for the slide)
- Algorithm: **PPO** (Proximal Policy Optimization), self-play
- Network: **1024 x 3 MLP** actor-critic
- Learning rate: **1e-4** (policy and critic) | PPO epochs: **2** | entropy coef: **0.01**
- Steps per update: **50k** | experience buffer: **150k** | minibatch: **50k**
- Decision rate: **tick_skip 8** → ~15 actions/sec | episode ends on goal, 10s no-touch, or 300s
- Returns standardized; observations not. (Anything unlisted = rlgym-ppo defaults.)
- Scale: **~9B timesteps** (target 10B), **~9,000 steps/sec** on a Ryzen 9 3900X (12c/24t)
- Note: **CPU-bound** — the physics sim runs on CPU across 22 workers; the GPU sat ~8% idle

## How strong (team tournament, full 5-bot field, 6/21 — Diego confirmed our bot #1; exact 300-game numbers pending his post)
> Note: report these as the **8.23B checkpoint** (what was uploaded when Diego ran it). Training
> continued past 9B afterward, but we report what we actually measured. To legitimately claim a 9B
> number, we would re-run the tournament with the real 9B bot.
- **#1 in BOTH modes, 4-0 matches each:**
  - Deterministic (argmax): **77.5%** game win-rate, +22 goal diff, 12 pts — clear 1st.
  - Stochastic (sampling): **70%** game win-rate, +16 goal diff, 12 pts — clear 1st.
- The rest: Diego papaya 3.5B and Nachi 2.9B are **co-second** (they swap 2nd/3rd between modes,
  both ~55-67%); Marco 2.0B 4th, Marian 1.35B 5th.
- **Takeaway:** our 8.23B is the undisputed strongest bot, robust across BOTH deploy modes, with
  ~2-4x the training steps of the others — training scale clearly paid off. (The earlier 4-bot test
  run had papaya edging us in deterministic; with the full field that flip is gone.)
- Caveat: this run is 40 games/bot. Diego's official ~300-game run is pending and will firm up the
  exact percentages (the ranking is already stable).

## Stack
`rlgym-ppo` (trainer) + `rlgym_sim` (Rocket League physics sim). Full recipe in
`checkpoints/_recipeH_distill.yaml`; distillation code in `src/rlbot/training/distill.py`.

---

## SLIDE CONTENT (team format: policy / reward / pros & cons / future improvements)

**1. Policy.** PPO self-play agent. Sees 107 numbers (its car, the ball, the opponent), picks 1 of
90 preset controller actions ~15x/sec. Network: 1024x3 MLP actor-critic. **Report it as the 8.23B
checkpoint** — that is what the tournament actually evaluated. (Training has continued past 9B
since, but report the number you measured, not a rounder one.) Deploy stochastic or deterministic
(both strong; #1 in both in the tournament).

**2. Reward stack.** Shaped and zero-sum (purely competitive). Dense shaping: drive toward the ball,
face it, push it toward the enemy goal, defend the backboard, keep it away from own goal, aerial
touches. Sparse events: goal +1, concede -1, shot, demo. Head start: kickstarted (distilled) from
Diego's papaya for the first 150M steps, then pure self-play.

**3. Pros and cons.**
- Pros: strongest bot in the tournament — #1 in BOTH modes (deterministic 77.5%, stochastic 70%).
  Robust, general play (trained from 70% randomized states, not just kickoffs). Reused our eval and
  video harness.
- Cons: probably too many shaping rewards, which dilutes the signal — smaller bots (Nachi 2.9B,
  papaya 3.5B) stayed competitive (co-second) with far fewer steps, so our reward is likely less
  efficient per step. Ground-dominant (limited aerials). Leaned on raw timesteps more than reward quality.

**4. Future improvements / what I learned.**
- Reward QUALITY matters as much as step count: smaller bots (Nachi, papaya) stayed competitive
  with far fewer steps, so I would simplify and tune the reward stack — though our training scale
  still delivered a clear #1 in both modes.
- Kickstarting works: a brief distillation head start let the bot surpass the teacher it copied.
- This is a CPU-bound task (physics simulation, not GPU) — hardware core count set the ceiling.
- Deterministic vs stochastic deployment flips the ranking (matchups are non-transitive), so always test both.
- Class concepts used: PPO (clipped policy-gradient, actor-critic), reward shaping, exploration vs
  exploitation (entropy bonus), discount factor + GAE, and self-play.
