# Presentation outline: PPO + Imitation Learning (1v1 Rocket League)

Reinforcement Learning and Autonomous Systems, group project.

How to use this: one section per slide. Each slide lists the point to make, the asset to
show, and a one line speaker note. Numbers marked (nb) come straight from the executed
notebooks so they stay defensible. Swap in the final scaled-up figures once that run lands.

Assets we already have:
- `deliverables/progression.mp4` and `progression_vs_origin.mp4` (bot getting stronger over training)
- `deliverables/champ_vs_papaya.mp4` (champion vs Diego papaya match)
- `deliverables/standings.png`, `deliverables/heatmap.png`, `deliverables/report.html` (tournament Elo + analysis)
- `imitation/artifacts/expert_vs_student.mp4` (expert vs cloned student)
- BC and DAgger result plots inside `imitation/notebooks/03_*.ipynb` and `04_*.ipynb`

---

## 1. Title
- Project title, team names, course.
- One line: "We trained a PPO expert and then studied imitation learning (Behavioural Cloning and DAgger) using that expert."
- Note: set the frame that the project has two halves, RL and IL.

## 2. Problem and environment (the deviation)
- Task: 1v1 Rocket League, learn to play from scratch.
- We deviated from Walker2D / Ant to Rocket League (rlgym_sim + rlgym-ppo), approved by Jaume.
- Why it is the same assignment: still a continuous-control agent, a PPO expert, and the full IL pipeline on top. The structure of the rubric is unchanged.
- Asset: a short clip of gameplay (progression.mp4 first few seconds).
- Note: address the deviation head on so it is clearly a feature, not a shortcut.

## 3. Roadmap
- Half 1: PPO expert (how we built a strong bot).
- Half 2: Imitation Learning (clone that expert with BC, fix it with DAgger).
- One sentence each on what the grader will see.

## 4. PPO method
- Algorithm: PPO with self-play (rlgym-ppo).
- Observation: AdvancedObs, 107 dim. Action: LookupAction, 90 discrete controller presets.
- Network: 1024 x 3 MLP actor-critic.
- Why discrete actions matter (sets up the IL classification framing later).
- Note: keep it crisp, this is context for the IL half.

## 5. PPO training and scale
- Self-play curriculum, billions of environment steps (~4B).
- Throughput reality: CPU-bound rollouts, GPU mostly idle, ~8 to 9k steps/sec on a Ryzen 9 3900X.
- Asset: progression.mp4 (bot visibly improving across checkpoints).
- Note: this is the "we actually did the engineering" slide.

## 6. PPO results (strength)
- Head to head vs teammates' bots, deterministic, fair kickoff, both sides.
- Numbers: clearly beats both Diego bots (papaya 1.34B ~67 to 74% of decisive games, the 512 ~76%), and the live bot is ahead of Marco's 2.0B (~73% of decisive games).
- Continued training matters: vs Marco we went from roughly even to ~73% over the last ~500M steps.
- Asset: standings.png (Elo table), champ_vs_papaya.mp4.
- Note: honest caveat that these are 60 game samples with wide intervals, directionally clear.

## 7. Imitation Learning setup
- The champion is now a fixed BLACK BOX expert. We only call it to act or to label states, never read its weights.
- Because actions are the 90-way discrete LookupAction, imitation is a CLASSIFICATION problem with cross-entropy loss, not MSE regression. This is the central modelling decision.
- Reuse trick: the student mirrors the expert network layout, so a trained student exports to a checkpoint the existing eval and video tools load unchanged.
- Note: the classification point is the single most important thing to land.

## 8. Behavioural Cloning
- Method: from-scratch PyTorch MLP, cross-entropy on (observation, expert action) demonstrations collected from real kickoff games.
- Leak-free evaluation: we hold out whole EPISODES, because a random frame split leaks near-identical 15 Hz neighbours and inflates accuracy. (nb: explain we caught and fixed this.)
- Result: held-out top-1 around 0.31 and top-3 around 0.51, versus a 0.13 majority-class baseline (nb, refresh from scaled run).
- Note: be honest that absolute accuracy is below 50%, which sets up the covariate-shift point.

## 9. BC ablations
- Dataset size: accuracy rises with more demonstrations, with error bars over seeds (nb plot).
- Network width: roughly flat within noise at fixed depth (nb plot).
- Takeaway: data helps more than capacity in this regime, and neither fixes the real problem.
- Asset: the two ablation plots from notebook 03.

## 10. The covariate shift problem (the key insight)
- BC agrees with the expert far more on expert states than on its own visited states (the gap).
- Punchline: the clone has decent per-action accuracy yet wins ~0% even against a weak bot, because small action errors compound over a ~200 step episode.
- This is the textbook BC failure, error grows as O(eps * T^2).
- Note: this is the intellectual heart of the talk, slow down here.

## 11. DAgger
- Method: roll the student out, let the expert label the states the student actually visits, aggregate, retrain, repeat.
- Result: agreement on the student's own state distribution rises sharply across rounds (around 0.17 to 0.85), with error bars (nb plot from notebook 04).
- Why it works: DAgger trains on the deployment distribution, cutting regret to O(eps * T).
- Asset: the DAgger agreement curve.

## 12. Expert vs student, qualitatively
- Asset: expert_vs_student.mp4 (champion vs the DAgger student, a close game).
- Note: shows the clone learned real behaviour, not just numbers on a slide.

## 13. Research questions and answers
- Classification vs regression for discrete control: why cross-entropy, why MSE is wrong.
- How does BC scale with data and capacity: the ablation answer.
- Why does BC fail at deployment and how does DAgger fix it: covariate shift, O(eps*T^2) vs O(eps*T).
- When does DAgger help most: when demonstrations are narrow and on-task (we showed broad random-state demos mask the effect).
- Note: map these to the exact research questions in the assignment PDF.

## 14. Engineering and rigor
- Reproducible notebooks, leak-free evaluation, an adversarial self-review that caught an inflated accuracy before submission and forced the episode-level split.
- Honest reporting throughout (we report the corrected, lower numbers).
- Note: this slide signals maturity, graders like seeing the process.

## 15. Conclusions
- We built a strong PPO expert and a complete, honest IL study on top of it.
- Main lesson: in imitation, the data distribution matters more than the model, and DAgger is the principled fix for covariate shift.
- One forward-looking line (distillation from a stronger teacher as future work).

## Backup slides (only if asked)
- Exact hyperparameters (PPO and BC).
- The harness-reuse mechanism (student exports to a DiscreteFF-compatible checkpoint).
- Tournament methodology (Bradley-Terry Elo, Wilson intervals, common-random-number seeds).
- Why kickoff-only demonstrations (narrow distribution exposes covariate shift).

---

### Mapping to the rubric
Align the slide content to the exact M-milestones in the assignment PDF. Rough mapping:
- Problem and environment -> M1
- PPO method and training -> M2 to M3
- BC implementation and ablations -> M4 to M5
- DAgger -> M6
- Evaluation, video, research questions -> M7
- Reproducibility and discussion -> M8

Confirm the numbering against the PDF before the talk.
