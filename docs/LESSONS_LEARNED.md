# LESSONS LEARNED

Hard-won insights from this project. A fresh Claude session that ignores these WILL
repeat mistakes that cost real time. Read before touching anything.

---

## RL / training insights

### 1. Under ZeroSumReward, Policy Reward is NOT a progress metric
In 1v1 self-play with `ZeroSumReward`, each player's reward = `own − opponent`. With two
identical policies that's mathematically ~0 forever, regardless of skill. **The Policy
Reward chart will look flat/noisy around 0 even as the bot gets dramatically better.**
- ❌ Don't judge progress from Policy Reward.
- ✅ Use `scripts/eval_checkpoint_progression.py` → win rate vs a fixed reference. That's
  the ground truth. (We built this whole script specifically because of this.)

### 2. Bigger networks have DELAYED payoff — don't judge them early
The 512×3 experiment looked WORSE than 256×3 at 97M (35% win rate, entropy stuck at 4.0
while 256×3 had committed to 3.85). We almost reverted. **We didn't, and by 125M it surged
to 65%, by 202M to 90%, and at 416M it takes 40% off a 1.35B bot.** Bigger nets start slow
(more params to fit) then overtake. Give a 512×3 net at least 100-150M before judging.

### 3. Own-goals under pressure ≠ goal confusion
The bot scored own-goals against the much-stronger Marian bot. It was NOT confused about
which net was which (it won 90% vs the baseline, scoring on the right net). It was
**defensive panic** — pinned near its own goal by a stronger opponent, mishitting clears.
Fixed with `BallAwayFromOwnGoalReward` (penalize ball moving toward own net in defensive
half) + the bot simply training more. Confirmed fixed at 416M (zero own-goals).
Lesson: diagnose by testing vs a weaker opponent before assuming a fundamental bug.

### 4. Changing rewards invalidates the value function
The critic is trained to predict returns under the current reward. Change the reward →
critic predictions are wrong → expect a recalibration wobble (~5-15M timesteps) where win
rate may dip before recovering. This is fine for a FINE-TUNE (resume strong policy, let
critic catch up). Just don't panic at the temporary dip, and don't eval in the first ~10M
after a reward change.

### 5. Reward magnitudes: events sparse+large, continuous small
Events (goals) happen rarely so need large weights (10-12x) to register in gradients.
Continuous rewards fire every step so accumulate fast — keep them small (0.05-2). Adding
MORE reward components increases signal density (each gives the policy another axis to
differentiate actions) — this is what broke the original plateau.

### 6. Metric quick-reference (what healthy looks like)
- **Policy Entropy**: starts ~4.5 (ln(90) max for 90 actions), drops slowly. Lower = more
  committed. ~3.85 was a strong committed bot. Stuck near 4.5 = not learning.
- **Mean KL Divergence**: 0.001-0.05 healthy. Spikes >0.1 = updates too aggressive.
- **SB3 Clip Fraction**: 0.05-0.20 healthy. ~0 = nothing updating.
- **Policy Update Magnitude**: 0.25-0.4 stable band is fine. Sharp downward spikes = session
  restart points (optimizer momentum resets), harmless.

---

## Tooling / environment gotchas

### 7. Python must be 3.10 or 3.11 — NOT 3.13
`rlgym-ppo` and `rocketsim` don't support 3.12+. The `ds311` conda env was actually Python
3.13 (misleading name). We created `rlbot310` (Python 3.10). `rocketsim` max version for
3.10 is **2.2.1** (2.2.4 needs 3.11+).

### 8. RocketSim imports as `RocketSim`, not `rocketsim`
The PyPI package `rocketsim` installs the module as `RocketSim` (capital R, S). Our code
goes through `rlgym_sim` so it rarely matters, but `import rocketsim` fails — use `import RocketSim`.

### 9. rlviser version must MATCH rlviser-py
`rlviser-py 0.6.13` only works with `rlviser.exe` **v0.8.7**. The latest binary (v0.9.x)
changed the wire protocol → crashes with `memory allocation of 72057594037927944 bytes`.
Download v0.8.7 specifically: `github.com/VirxEC/rlviser/releases/download/v0.8.7/rlviser.exe`.
Also: don't pre-launch rlviser.exe AND run a render script — they fight for the UDP port
(error 10048). Let the script auto-launch it, or kill existing instances first.

### 10. wandb needs >=0.27 for the new key format
The `wandb_v1_...` long-form API keys are rejected by wandb <0.18 with "API key must be 40
characters long." We upgraded to 0.27. Also: NEVER paste API keys in chat — rotate if leaked.

### 11. rlgym_ppo's Ctrl+C bypasses Python `finally` blocks
The Learner uses `os._exit()` on multiprocess teardown, which skips `try/finally` AND
`except`. Our `save_run_summary` initially never fired on Ctrl+C. Fixed by ALSO registering
it via `atexit` (which survives most exit paths) + making it idempotent with a
`_summary_saved` flag. If you add exit-time logic, use atexit, not just finally.

### 12. Architecture change → shape mismatch on resume
Changing `policy_layer_sizes` then resuming a checkpoint trained with the old size throws
`size mismatch for model.0.weight`. To change arch: start a NEW experiment (fresh from 0)
OR match the checkpoint's saved layer sizes. `_load_policy` reads layer sizes from each
checkpoint's `BOOK_KEEPING_VARS.json`, so eval handles mixed-arch fine — only resume-training
is strict.

### 13. n_checkpoints_to_keep=5 rotates old checkpoints out
rlgym_ppo keeps only the latest 5 saves by default. Older checkpoints get DELETED as
training advances. To preserve a milestone (for eval comparisons or as a presentation
"before"), COPY it to a `MILESTONE_*` folder manually before continuing training. We lost
the ability to make fine-grained progression curves on `nexto_plus_kickoff_512` because of
this — consider bumping `n_checkpoints_to_keep` to a large number for future runs.

### 14. checkpoint folder layouts differ between the two code paths
- `diego-bots/simple_bot.py` saves to `diego-bots/checkpoints/<EXP>/<EXP>-<unix>/<timestep>/`
- `src/rlbot` (Marian's) saves to `checkpoints/<EXP>/<timestep>/`
The eval scripts handle BOTH. The progression script's `find_checkpoints()` scans both layouts.

---

## Workflow conventions that worked

- **Commit before long runs / overnight** — code on GitHub survives laptop crashes.
- **Always compile-check + build_env() smoke test after edits** before declaring done.
- **Preserve milestone checkpoints** before any reward/arch change (point 13).
- **Eval vs a FIXED reference** for fair progression curves (point 1).
- **One experiment per EXPERIMENT_NAME**, bump it for fundamental changes, resume same name
  for fine-tunes.
- **Watch the bot in rlviser periodically** — the win-rate number doesn't tell you it's
  dribbling into walls or panic-clearing. Visual inspection caught both the own-goal and
  the dribble-into-poles issues that pure metrics missed.

---

## The story arc so far (for the presentation, and for context)

1. **baseline (256×3, 3 rewards)** → plateaued at 11M, entropy stuck at max (4.48). Diagnosed
   as too-sparse reward signal.
2. **nexto_rewards (256×3, 10 rewards)** → broke the plateau, reached 130M, 100% win vs baseline.
   Clean 10%→100% progression curve (great presentation visual).
3. **nexto_plus_kickoff (256×3, +custom rewards +kickoff)** → 17.6M, abandoned for bigger net.
4. **nexto_plus_kickoff_512 (512×3, v1 then v2 rewards)** → the champion. v1 took 35%→90% vs
   the 130M baseline; v2 reward tuning (aerial touch, anti-own-goal, ball-distance boost)
   eliminated own-goals and added aerials. At 416M: 40% vs Marian's 1.35B.
5. **Next**: dribbling-to-goal reward, kickoff logic, then cloud-GPU push toward billions.
