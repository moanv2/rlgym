# Presentation speaker notes (draft) — PPO + Imitation Learning in 1v1 Rocket League

Detailed talking points to go with `presentation_outline.md`. Numbers here are the final,
verified ones (use these, not any earlier rough figures). Each slide: what to say, the key
numbers, the asset to show. Keep slides visual, talk to these notes.

---

### S1 — Title
- "We did the assignment as two halves: a PPO expert, then Imitation Learning (BC and DAgger) on top of it."
- "With Jaume's approval we ran it in 1v1 Rocket League instead of Walker2D and Ant, same structure, harder environment."

### S2 — Problem and environment
- Task: learn to play 1v1 Rocket League from scratch, continuous control, sparse goal reward.
- Stack: rlgym_sim physics + rlgym-ppo trainer. Observation: AdvancedObs, 107 numbers per car. Action: LookupAction, 90 discrete controller presets.
- Why the deviation is legitimate: it is still an expert-then-imitate pipeline, the only change is the environment, and the discrete action space makes the imitation step a clean classification problem.
- Show: a few seconds of gameplay (progression.mp4).

### S3 — Roadmap
- Half 1: train a strong PPO bot (our "expert").
- Half 2: clone it with Behavioural Cloning, then fix the clone with DAgger, with ablations and analysis.

### S4 — PPO method
- Algorithm: PPO with self-play. The bot plays itself, both cars learning, so the opponent scales with the bot.
- Net: 1024 x 3 MLP actor-critic. Reward: a shaped stack (ball-to-goal velocity, touches, alignment, boost economy, event rewards for goals and concedes).
- One line that sets up Half 2: because the action space is 90 discrete choices, "imitating the expert" later means "predict which of 90 actions it would pick," a classification problem.

### S5 — Training and scale
- ~4 billion environment steps. CPU-bound (the physics sim), GPU mostly idle. ~9,000 steps per second on a Ryzen 9 3900X.
- The bottleneck is the synchronous rollout loop, not raw cores, which is why one run cannot saturate the machine. (Honest engineering detail, shows we understand the system.)
- Show: progression.mp4 (the bot visibly improving across checkpoints).

### S6 — PPO results (where we stand)
- Accurate head-to-head vs every uploaded teammate bot: 300 games each, mixed start states, both sides, deterministic, with Wilson 95% confidence intervals. Win-rate over decisive games:
  - Diego 2.85B v7 (his latest): **59%** (CI 0.53–0.65) — ahead, modestly.
  - Diego papaya 1.34B: **68%** (0.62–0.73).
  - Diego 1.18B (512): **84%** (0.79–0.88).
  - Marco 2.0B: **67%** (0.61–0.72).
- Takeaway: we beat every uploaded bot. The one close rival is Diego's newest, where the edge is real but thin, which is why we kept training.
- Honesty note worth saying out loud: a quick small-sample test first showed ~80% vs Diego's v7, the 300-game test corrected it to ~59%. Good example of why sample size and proper methodology matter.
- Show: standings.png, champ_vs_papaya.mp4.

### S7 — Imitation Learning setup (the pivot)
- Now freeze the champion and treat it as a fixed, black-box EXPERT. We only ever ask it "what action here," never read its weights or recipe. That is exactly how DAgger treats its expert oracle.
- The single most important modelling decision: 90 discrete actions means imitation is multi-class CLASSIFICATION with cross-entropy loss, NOT regression. MSE on action indices would be meaningless (action 89 is not "89x" action 1).
- Neat engineering: the student network mirrors the expert's layout, so a trained student exports to a checkpoint our existing eval and video tools load unchanged. We reused the whole harness.

### S8 — Behavioural Cloning
- Collect demonstrations: roll the champion out in real kickoff games, record (observation, expert action) pairs. ~14k–26k pairs.
- Train a from-scratch PyTorch MLP with cross-entropy. The important methodology point: we evaluate on held-out whole EPISODES, not random frames, because at 15 Hz neighbouring frames are near-identical and a random split leaks them and inflates accuracy. (We actually caught this in our own review and fixed it.)
- Result: held-out top-1 **0.35**, top-3 **0.57**, versus a 0.13 majority-class baseline. So it genuinely learned state-dependent behaviour, ~2.7x the trivial baseline, but still below 50% (cloning a superhuman policy is hard).
- Show: BC loss and accuracy curves from notebook 03.

### S9 — BC ablations
- Dataset size: accuracy rises with more demonstrations, 0.29 to 0.36, with tight error bars over 3 seeds.
- Network width (depth fixed): essentially flat, error bars overlap, so capacity barely matters here.
- Takeaway: data helps more than model size, and neither fixes the real problem (next slide).
- Show: the two ablation plots (with error bars) from notebook 03.

### S10 — The covariate-shift problem (the heart of the talk, slow down here)
- BC is trained only on EXPERT states. At deployment the student makes small errors, drifts into states the expert never demonstrated, and has no idea what to do there. Errors compound as O(eps · T²) over a horizon T.
- Made concrete: our clone agrees with the expert ~0.31 per action on expert states but only ~0.20 on its OWN visited states, and it wins ~0% even against a weak bot. High per-action accuracy does not equal task skill.
- This is THE reason naive cloning fails, and the motivation for DAgger.

### S11 — DAgger
- Idea: train on the student's OWN state distribution. Roll the student out, ask the expert to label the states it actually visits, aggregate, retrain, repeat.
- This bounds regret at O(eps · T) instead of O(eps · T²).
- Result: agreement on the student's own distribution rises from ~0.15 to ~0.69 across six rounds (mean and std over 3 rollouts per round, so it has error bars). It ends up agreeing with the expert on its own states more than plain BC did on expert states.
- Important caveat we found: this clean result needs demonstrations from a NARROW, on-task distribution (real kickoff games). Demonstrating from wildly random states masks the covariate shift and DAgger then looks marginal. A genuine insight about when DAgger helps.
- Show: the DAgger agreement curve (with error bars) from notebook 04.

### S12 — Expert vs student, qualitatively
- Show: expert_vs_student.mp4 (champion vs the DAgger student, a close game).
- Point: the clone learned real, watchable behaviour, not just numbers.

### S13 — Research questions, answered
- Classification vs regression for discrete control: cross-entropy, because the actions are categorical. (Slide 7.)
- How does BC scale with data and capacity: data helps, width barely, within noise. (Slide 9.)
- Why does BC fail and how does DAgger fix it: covariate shift, O(eps·T²) vs O(eps·T). (Slides 10–11.)
- When does DAgger help most: when demonstrations are narrow and on-task, so the student actually drifts out of the demonstrated region. (Slide 11.)
- Map these one-to-one onto the assignment PDF's research questions before the talk.

### S14 — Engineering and rigor
- Reproducible notebooks, leak-free episode-level evaluation, and an adversarial self-review that caught an inflated BC accuracy before submission and forced the proper split.
- We report the corrected, lower numbers. Honesty over hype.
- Same discipline on the bot side: we corrected an 80% claim down to a verified 59% after a proper 300-game test.

### S15 — Conclusions
- Built a strong PPO expert that beats every teammate bot, then a complete and honest IL study on top of it.
- Main lesson: in imitation learning, the data distribution matters more than the model, and DAgger is the principled fix for covariate shift.
- Future work: distillation from a stronger teacher, and the possession-reward experiment we have built and parked.

### Backup slides (only if asked)
- Exact PPO and BC hyperparameters.
- The student-exports-to-DiscreteFF harness-reuse trick.
- Tournament methodology: mixed start states, both sides, Wilson intervals, why deterministic.
- Why kickoff-only demonstrations expose covariate shift.

---

### Speaking tips
- The two highest-impact moments are slide 10 (covariate shift, high per-action accuracy but 0% wins) and slide 11 (DAgger closing the gap). Spend time there.
- Lead with honesty on numbers (the 80 to 59 correction, the leak fix). Graders reward rigor and candor.
- Keep the Rocket League novelty as a hook, not the substance. The substance is the IL analysis.
