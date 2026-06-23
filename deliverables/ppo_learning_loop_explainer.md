# Your part: the PPO learning loop, the Markov state, and the kickstart

This is your make or break section. For each topic there is a "say this" version for the
audience and a "if asked" version with the depth for questions.

## Fact check first (Diego is in the room)
You distilled from Diego's **papaya bot at 1024x3, about 1.34 billion steps** (107 obs, 90
actions, the same architecture as your bot). Verified from the checkpoint weights and the
training log. There was a separate Diego 512 net that we only ever tested against, never
distilled from. So say "Diego's papaya 1024 bot," not 512.

---

## 1. Why is the state Markovian

**Say this.** A state is Markovian when the future depends only on the present, not on how you
got there. Rocket League physics is exactly that. Given the current positions, velocities, and
rotations of both cars and the ball, plus the controls, the next moment is fully determined by
the physics. Once you know the present, the past adds nothing.

**The insight to land.** This is why the observation includes velocities and angular velocities,
not just positions. Position alone is not enough. To know where the ball goes next you need its
speed and direction. Physics is second order, so the state has to carry both position and
velocity to be a complete summary. We build the observation to be Markovian on purpose.

**Why it matters for our design (strong point).** Our network is feedforward with no memory. That
is only valid because the state is Markovian. A memoryless network cannot look at history, so it
depends on the current observation being a full summary of the situation. The velocities and
rotations in the observation are what make that true.

**If pushed (honest caveat).** Strictly, the bot does not see every hidden variable, for example
exact boost pad respawn timers, so it is technically a partially observed MDP. But it sees all the
physically essential quantities, so it is Markovian for all practical purposes.

---

## 2. The PPO learning loop (actor and critic)

**The two networks.**
- **Actor** is the policy. Input is the 107 number state, output is a probability over the 90
  actions. It decides what to do.
- **Critic** is the value network. Same input, output is a single number, the expected future
  reward from this state. It judges how good the situation is. It never picks actions.

**The loop, one cycle.**
1. **Collect.** The current actor plays against a copy of itself for about 50,000 steps. At every
   step we record the state, the action taken, the reward, and the critic's value estimate.
2. **Score the actions.** From the rewards and the critic's estimates we compute an advantage for
   each action. Advantage is how much better the action turned out than the critic expected.
   Positive means better than expected, do more of it. Negative means worse, do less.
3. **Update the actor.** Shift the policy so positive advantage actions become more likely and
   negative ones less likely. PPO does this with a clipped objective that caps how far the policy
   can move in one step. That cap is the trust region, and it is why our clip fraction and KL
   divergence stayed tiny and stable.
4. **Update the critic.** Train the value network to predict the real returns better, so next
   round the advantages are more accurate.
5. **Repeat.** The slightly better actor plays again. Over 10 billion steps this compounds into a
   champion.

**Why a critic at all (this is the make or break idea).** Without the critic you would learn from
raw outcomes, which are extremely noisy. One lucky goal would wrongly reward the hundred random
actions that came before it. The critic gives a baseline expectation, so we learn from the
surprise, the gap between what happened and what was expected. That cuts the noise massively and
is what makes the training stable.

**If asked for more depth.**
- The advantage is computed with GAE (generalized advantage estimation), which blends short and
  long term outcomes using the discount gamma 0.99 and lambda 0.95.
- The clipped objective limits the ratio of new policy to old policy to a band around 1, set by
  the clip range 0.2. This is the trust region that keeps updates safe.
- The critic is trained by regression toward the observed returns.
- Actor and critic are separate networks here, both 1024 by 3, trained together every cycle.

---

## 3. Distillation, the head start (kickstarting)

**Say this.** We did not start from random flailing. For the first 150 million steps we added a
term that pulled our actor toward matching Diego's papaya bot, our teacher. It is like learning
by shadowing a good player. Then we faded that pull to zero and let pure self play take over, so
the bot could surpass the teacher instead of only copying it.

**The mechanism if asked.** We added a term that penalized our policy for disagreeing with the
teacher's action distribution, weighted by a factor beta. Beta started at 0.3 and decayed
linearly to 0 over 150 million steps. After that it is pure PPO. The reference is DeepMind's
Kickstarting Deep Reinforcement Learning, 2018.

**The payoff line.** We kickstarted from Diego's 1024 papaya at 1.34 billion steps, then trained
to 10 billion and now beat it. Copy to start, then surpass.

---

## One sentence flow for your section
"The game state is Markovian because physics only needs the current positions and velocities, so
a memoryless network is enough. PPO learns by playing itself, having a critic judge each move
against expectation, and nudging the actor toward the moves that beat expectation, with a clip
that keeps every update safe. We gave it a head start by shadowing Diego's papaya bot for the
first 150 million steps, then let it surpass the teacher over 10 billion steps of self play."
