# Presentation prep and session context (saved 2026-06-23)

This is the "read this first" file for building the deck and defending the bot. It captures the
context from the working session so you can pick it up from your laptop. The authoritative
content lives in the other files in this folder (see the index below). Open the deck by just
opening `martin_champion_deck.html` in any browser. It is fully self contained (inline SVG
graphics, fonts loaded from Google over the internet), no build step and no local assets.

## What is in this folder
- `martin_champion_deck.html` = the final 6 slide deck. Open in a browser. Edit the HTML directly.
- `how_the_bot_works.md` = policy, reward stack, strengths and weaknesses, future improvements.
- `hyperparameter_defense.md` = the wandb graph defense (clip fraction, KL, entropy) with the real numbers.
- `ppo_learning_loop_explainer.md` = the PPO loop, Markov state, actor critic, distillation.
- `presentation_speaker_notes.md` = speaker notes.
- `presentation_outline.md` (on the martin branch) = the full slide by slide outline and rubric map.
- `imitation_BC_DAgger_submission.zip` = the graded imitation learning deliverable (BC and DAgger).

## The bot at a glance (the numbers to quote)
- 10.04 billion training steps (exact step 10042132490). This is a real number, the bot was actually trained that far.
- Network: a 1024 by 1024 by 1024 MLP (DiscreteFF actor critic).
- Input: AdvancedObs, 107 dimensions. Output: LookupAction, 90 discrete actions. The policy acts every 8 physics ticks.
- PPO settings: 2 epochs per update, learning rate 1e-4 for policy and critic, gamma 0.99, gae_lambda 0.95,
  entropy coefficient 0.01, clip range 0.2.
- Healthy training signals from wandb: clip fraction 0.0101, mean KL divergence 0.0013, policy entropy 4.02.
  See `hyperparameter_defense.md` for what each means and why low and stable is good.
- Trained on a Ryzen 9 3900X (12 cores, 24 threads) at about 9.5k steps per second with 22 parallel workers.
- Started from a teacher (Diego papaya, a 1024 by 1024 by 1024 net, about 1.34B) by distillation, then pure
  self play RL took over and surpassed the teacher.

## Results (fair kickoff, deterministic, our 300 to 400 game evals)
- Beats Diego papaya 1.34B about 62 percent.
- Beats Marco 2.0B about 69 percent.
- Beats Diego 512 net about 60 percent.
- Beats Diego v7 about 59 percent of decisive games.
- Tournament order: Martin number one, then Nachi, then Diego, then Marco, then Marian.

## Concept cheat sheet (for the make or break part)
PPO learning loop:
1. Collect experience by self play. The current policy plays games against itself, acting every 8 ticks.
2. Estimate advantages with GAE (gamma 0.99, lambda 0.95). The advantage is how much better an action was
   than the critic expected.
3. Update the policy and the value net for a few epochs (2) using the clipped surrogate objective
   (clip 0.2 keeps each update small and stable). An entropy bonus (0.01) keeps the policy exploring.
4. Repeat. Over billions of steps the policy keeps improving.

Why the state is Markovian:
The observation encodes the full physical state needed to act optimally: ball position and velocity, both
cars position, velocity, orientation and boost, the boost pad timers, and the previous action. Given this
state the future depends only on the present state and the chosen action, not on the past, because the
RocketSim physics is deterministic given the current state and action. So the state has the Markov property
and the value and policy can be functions of the current observation alone.

Actor critic:
The actor is the policy. It outputs a probability over the 90 discrete actions. The critic is the value net.
It predicts the expected return from a state. The critic is used to compute advantages, which lowers the
variance of the policy gradient so learning is faster and more stable.

Distillation or kickstarting:
The student policy was first trained to copy a strong teacher (cross entropy or KL toward the teacher action
distribution). That gives a good starting point fast. Then the distillation weight was annealed to zero and
pure RL took over, so the student surpassed the teacher rather than being capped by it.

Stochastic versus deterministic (4 sentence version):
Stochastic means the bot samples an action from its probability distribution, so it explores and plays with
variety. Deterministic means the bot takes the single most likely action (the argmax), which is the most
consistent and usually the strongest play for deployment and evaluation. During training we want stochastic
for exploration, at test time we usually want deterministic. Our bot is number one in both modes, which means
it wins whether it is sampling or taking the argmax, so the strength is not an artifact of one setting.

Clip fraction (0.0101):
The fraction of samples where the PPO probability ratio hit the 0.2 clip bound. A low value means each update
is small and the policy is not lurching, which is a sign of stable training.

Mean KL divergence (0.0013):
The average change between the old and new policy per update. Low and steady means controlled, stable updates,
neither collapsing nor diverging.

Policy entropy (4.02):
How much the policy is still exploring. With 90 actions the maximum entropy is about 4.5, so 4.02 means the
policy kept healthy exploration rather than collapsing too early to a narrow set of actions.

## What Martin's bot did differently (the unique slide)
- Disciplined A/B testing of reward ideas. Many reward experiments (possession, touch acceleration, big
  rollouts, low entropy, concede penalty, KRC) were each tested against the champion and kept only if they won.
  The reward stack at the optimum beat every challenger, so the final stack is the result of evidence, not guesses.
- Distillation then surpassing the teacher, rather than training from scratch or staying a copy.
- Scale as the decisive lever. A strong CPU (12 cores, 24 threads) running 22 parallel workers let the bot
  reach 10 billion steps, far more experience than the rest of the field, which is the main reason it is ahead.

## The CPU and n_proc advantage narrative
PPO throughput here is bound by how many parallel game workers the CPU can run. With 22 workers on the
Ryzen 9 3900X the bot collected experience at roughly 9.5k steps per second. More workers means more games
per second means more total experience in the same wall clock, which compounds over billions of steps. The
decisive advantage was hardware that let the bot simply see and learn from far more play than the others.

## Likely defense questions and short answers
- Why only 2 epochs? Throughput. With this much fresh experience per iteration, 2 epochs already extracts the
  signal and avoids overfitting the current batch, and it keeps steps per second high.
- Why gamma 0.99 and not higher? It worked best in practice for this reward shaping and horizon. The clip
  fraction and KL stayed healthy, which is the evidence it was tuned well.
- Is 10 billion real? Yes, step 10042132490, trained over many days, frozen and pushed.
- Stochastic or deterministic for the demo? Deterministic for the cleanest, strongest play. Mention it is
  number one in both modes.

## Status of the work (so you are not surprised)
- Training is FROZEN at 10.04B and the final bot is pushed (CHAMPION 10.0B on the martin branch).
- The imitation learning half (BC and DAgger) is done and on the martin-imitation branch plus the zip here.
- The 3D highlight video idea was abandoned because rlviser cannot hold a steady cinematic camera on fast
  goals (the chase camera whips, ball cam was no better). All the generated videos were deleted to save space.
  If you ever want highlights later, the path that would actually work is a custom broadcast renderer with full
  camera control, not rlviser. The scripts are still in tools but they are not needed.
