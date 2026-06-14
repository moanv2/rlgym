# v7 — fast-kickoff package (IMPLEMENTED 2026-06-12; baseline captured)

> Status: all Phase 0/1 items below are implemented, unit-tested (8/8) and
> integration-tested through papaya's real build_env. Pre-v7 baseline vs
> Martin's champion (deterministic, all 5 spawns x both colors):
> **first_possession_rate 0.20** (1 of 5 lines), papaya touch 2.2s vs Martin
> 2.8s when he commits, mean ball territory 3s after kickoff **-1109uu**
> (in papaya's half). JSON: history_and_summary/kickoff_benchmark_pre_v7_*.json.
> Rollback archive: diego-bots/checkpoints/_archive/papaya_1024_PRE_V7_1.92B/.

Source material: two PDFs read 2026-06-12 — the RLGym "Game Values" constants
reference, and a fast-kickoff guide (reward shaping for early ball contact,
short no-touch timeouts, self-play as the speed benchmark).

Why kickoffs: the deterministic-20 eval showed papaya wins only ~2-3 of the 5
scripted kickoff lines vs Martin's champion (45%), while stochastic-50 shows
overall skill is clearly ahead (66%). The real presentation match runs both
bots near-deterministic, so kickoff lines get an outsized vote. Kickoff is the
highest-leverage remaining skill gap.

## Ground rules (learned the hard way, non-negotiable)

1. **Translate concepts, not code.** The PDF's snippets are RLGym 2.0 API
   (`rlgym.api`, `KickoffMutator`, `RepeatAction`) — papaya is rlgym_sim. Our
   equivalents already exist: `RandomKickoffSetter` (= KickoffMutator),
   `KickoffReward`, kickoff drills at 0.25 of spawns.
2. **No action-space change, no tick_skip change, no obs change.** Any of those
   orphans the 1.9B-step checkpoint (strict load_state_dict). The PDF's
   "frame-perfect flip cancel" point is noted: at tick_skip=8 the bot acts every
   66.7ms vs the ~75ms human flip-cancel window — coarse but learnable (all the
   strong community bots run tick_skip 8 and still kickoff fast).
3. **Don't hardcode a speed flip.** Per the PDF (and our own philosophy): shape
   the *outcome* (early, fast first touch), let PPO find the motor sequence.
4. **One attributable package.** v7 = kickoff-targeted reward + termination +
   constants. Nothing else moves (v6 optimizer stays frozen).

## Phase 0 — benchmark BEFORE changing anything

Build `scripts/kickoff_benchmark.py` (rlgym_sim, AdvancedObs, deterministic):
for each of the 5 canonical kickoff spawns x both colors, run papaya vs a fixed
opponent (Martin's champion, and papaya-mirror) and record:
- **time-to-first-touch** (the PDF's core metric),
- **first-possession %** (who touched first),
- **post-kickoff outcome** (ball heading into whose half 3s after first touch).
Output: per-spawn table + JSON in `history_and_summary/`. This is the metric
v7 answers to; without it the package is vibes.

## Phase 1 — the changes (all rlgym_sim-native, resume-safe)

1. **`KickoffReward` v2 — time-decaying early-touch** (the PDF's central idea:
   early touch must pay more than late touch). Current reward is binary
   (touch while ball near center = 1.0). New: at the FIRST touch of a kickoff,
   pay `max(0, 1 - t_touch / T_MAX)` (T_MAX ~ 4s), 0 for later touches; keep a
   small alignment term for touching it *toward* the opponent half. Stateful
   per episode like FlickReward (needs reset()).
2. **`KickoffStallCondition`** (translate "3-5s no-touch timeout for kickoff
   training"): a TerminalCondition that ends the episode if the ball is still
   inside the center kickoff radius after ~4s. Our global 10s NoTouchTimeout
   stays for normal play; this only kills stalled kickoffs, so drill episodes
   recycle ~2.5x faster = more kickoff reps per hour at zero extra spawn share.
3. **Constants** — add to `rl_constants.py` from the Game Values PDF:
   `DOUBLEJUMP_MAX_DELAY = 1.25`, `FLIP_TORQUE_TIME = 0.65` (jump-physics
   section; documents the flip-cancel timing the policy is implicitly learning).
4. **Weights** — `KickoffReward` weight 0.35 -> ~0.5 (it's now better-shaped and
   sparser-per-step); kickoff spawn share STAYS 0.25 (the stall condition
   already multiplies effective reps). Everything else untouched.

## Phase 2 — run + verify

- Overnight run with v7 (resumes the 1.9B checkpoint; reward-shift transient
  expected for a few M steps, same as v5/v6 transitions).
- Morning gates, in order:
  1. `kickoff_benchmark.py` delta — time-to-first-touch down, first-possession
     up vs Phase-0 baseline (the attributable metric).
  2. Deterministic-20 vs Martin — kickoff lines won should move from 2-3/5
     toward 3-4/5.
  3. Stochastic-50 vs Martin — the guard: must stay >= ~55%. A kickoff gain
     that costs general play is a net loss.
- Rollback: same machinery as v6 (50 retained checkpoints + archive copy
  taken before the v7 restart).

## Explicitly rejected from the PDF guidance

- Migrating to RLGym 2.0 / KickoffMutator (API break, zero benefit now).
- Kickoff-only training env (papaya must stay a complete player; drills-within-
  the-mix preserve that).
- Lower tick_skip / RepeatAction tuning (orphans the net).
- Hardcoded speed-flip macro (against the whole training philosophy; also the
  deployment runs the same policy net — a macro wouldn't exist in rlgym_sim).
