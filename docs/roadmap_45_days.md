# 45-Day Roadmap

**Start:** 2026-05-07 · **Final presentation:** 2026-06-21 · **Goal:** beat a classmate's bot in a 1v1 match.

The most important thing is that **week 1 must end with a long training run actually running**. Compute, not code, is the bottleneck on a 45-day timeline.

---

## Week 1 — Days 1-7 (May 07 – May 13): Foundations

**Objective:** baseline bot training in the background by Sunday night.

| Day | Task                                                                                  |
|-----|---------------------------------------------------------------------------------------|
| 1   | Scaffold repo (this commit). Read [docs/setup.md](setup.md). Install Python + CUDA.   |
| 2   | Install all deps. Dump `collision_meshes/`. `make test-fast` green.                   |
| 3   | `python -m rlbot.training.train --config configs/experiments/exp_001_baseline.yaml --dry-run` works end-to-end. |
| 4   | First real training: launch `exp_001_baseline` (10M timesteps, ~2-6h). Set up wandb.  |
| 5   | Inspect wandb graphs (Policy Reward, KL, FPS). Read [intro.md][intro] + [graphs.md][graphs] from the tutorial. |
| 6   | Watch the bot in the visualizer. Record a clip for later comparison.                  |
| 7   | Kick off `exp_002_advanced_obs` (50M, will run into week 2).                          |

**Exit criteria:** non-zero policy reward trend on wandb, bot visibly chases ball.

[intro]: ../../rl-bot-turorial-repo/RLGym-PPO-Guide/intro.md
[graphs]: ../../rl-bot-turorial-repo/RLGym-PPO-Guide/graphs.md

---

## Week 2 — Days 8-14 (May 14 – May 20): Reward shaping

**Objective:** the bot reliably hits the ball toward the opponent's goal.

- Read [rewards.md][rewards] + [making_a_good_bot.md][good] thoroughly.
- Stage 1 → Stage 2 reward shift (see `configs/reward_weights/stage_*.yaml`).
- Add custom rewards if needed (e.g. `BallHeightReward`, `KickoffReward`). Each
  one gets a unit test.
- Start the long-run experiment `exp_003_long_run` if `exp_002` looks healthy.

**Exit criteria:** non-trivial goal rate (>0.1 goals per minute self-play),
SB3 clip fraction stable in 0.05–0.20 range.

[rewards]: ../../rl-bot-turorial-repo/RLGym-PPO-Guide/rewards.md
[good]: ../../rl-bot-turorial-repo/RLGym-PPO-Guide/making_a_good_bot.md

---

## Week 3 — Days 15-21 (May 21 – May 27): Long training + curriculum

**Objective:** the bot is recognizably playing Rocket League.

- `exp_003_long_run` runs 24/7. Resist the urge to stop and tweak — every restart
  costs days.
- Build the eval harness (`src/rlbot/evaluation/evaluate.py`) and run weekly
  bot-vs-bot tournaments between checkpoints.
- Curriculum: shift state-setter mix toward more game-realistic spawns.

**Exit criteria:** newest checkpoint beats the week-1 baseline >70% in 100-episode eval.

---

## Week 4 — Days 22-28 (May 28 – Jun 03): Hyperparameter + architecture

**Objective:** squeeze the last bit of strength out before deployment work.

- Try `arch: medium` (or a longer run on `small`) — pick whichever wandb dashboard
  says is improving faster.
- Experiment with `ppo_ent_coef` (lower = more exploitative late-stage).
- Hard freeze on training changes by **end of week 4** — week 5 is for deployment,
  not new experiments.

**Exit criteria:** chosen "presentation candidate" checkpoint identified.

---

## Week 5 — Days 29-35 (Jun 04 – Jun 10): RLBot deployment

**Objective:** bot can play in real Rocket League against another RLBot agent.

- Implement `src/rlbot/deployment/bot.py`: load checkpoint, build same obs as training,
  run forward pass, decode `LookupAction` to controller state.
- Write `scripts/export_model.py` to package the bot folder.
- Test locally vs. an RLBot like Nexto/Necto/Chip on easy difficulty.

**Exit criteria:** bot plays a full match against another RLBot without crashing.

---

## Week 6 — Days 36-42 (Jun 11 – Jun 17): Polish + scouting

**Objective:** maximize win rate vs. *the specific classmate's bot*.

- If you can get a copy of their bot or a similar RLBot, run a 100-match private eval.
- Spot weaknesses: do they over-commit? bad in air? slow rotations? — add a *targeted*
  reward shaping run if needed (don't restart from scratch — fine-tune).
- Final wandb dashboard cleanup for the presentation slides.

---

## Days 43-45 (Jun 18 – Jun 20): Presentation prep

- Day 43: Lock the final checkpoint. No code changes.
- Day 44: Slides — architecture diagram, training curves, eval numbers, demo clip.
- Day 45 (Jun 20): Practice run. Sleep early.

## Day 45 (Jun 21): Present.

---

## Risk register

| Risk                                | Mitigation                                                          |
|-------------------------------------|---------------------------------------------------------------------|
| Single-GPU training too slow        | Pick `small` arch; extend horizon by reducing eval/visualization    |
| rocketsim install fails             | Ask early; have a CPU-only fallback config (slow but works)         |
| Bot plateaus and refuses to improve | Reset with revised obs/rewards; budget one full reset by end of W3  |
| RLBot deployment breaks late        | Spike a "load checkpoint, run forward pass" smoke test in W2 not W5 |
| Lose the wandb run id               | Always check `checkpoints/<exp>/run_metadata.json` before deleting  |
