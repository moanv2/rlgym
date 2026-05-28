# HANDOFF — Read This First

You are a fresh Claude Code session picking up an in-progress Rocket League RL bot
project. This doc orients you. Read it fully, then `docs/NEXT_TASKS.md` for what to do
and `docs/LESSONS_LEARNED.md` for gotchas that will bite you if you don't know them.

---

## The project in one paragraph

Diego (data-science student, Python-only) is building a 1v1 Rocket League bot for a
class final worth 40% of his grade. The opponent at the final is a classmate's bot —
and the bar has risen: both sides are now training to **billions of timesteps**. The bot
trains via self-play PPO using `rlgym_sim` (RocketSim C++ physics under a Python wrapper)
+ `rlgym_ppo`. Everything Diego authors is Python. The current champion bot is strong:
40% win rate vs a teammate's 1.35B-timestep bot, 90%+ vs earlier baselines.

## Environment / machine facts

- **Repo**: `C:\Users\Diego\Desktop\rlgym-root\rlgym\` (GitHub: github.com/moanv2/rlgym)
- **Current git branch**: `diego` (has Diego's `diego-bots/` + merged `marian/setup-fixes` `src/rlbot/`)
- **Conda env**: `rlbot310` (Python 3.10.20) — activate before anything: `conda activate rlbot310`
- **Python interpreter for scripts**: `C:/Users/Diego/miniconda3/envs/rlbot310/python.exe`
- **GPU**: NVIDIA RTX 4070 Laptop, 8 GB VRAM, CUDA 12.x driver, torch 2.5.1+cu121
- **CPU**: 12 physical / 24 logical cores
- **wandb**: entity `diego08-ie-university`, project `rlgym-finalproject`
- **Stack versions** (critical — see LESSONS): rocketsim 2.2.1, rlgym-sim 1.2.6, rlgym-ppo 1.3.13, rlviser-py 0.6.13 + rlviser.exe **v0.8.7** (NOT v0.9.x), wandb >=0.27

## Two code paths (both call rlgym_ppo.Learner under the hood)

1. **`diego-bots/simple_bot.py`** — THE active training script. Standalone, all knobs
   inline, no YAML. This is what Diego actually runs. **Work here for training changes.**
2. **`src/rlbot/`** — Marian's YAML-config-driven framework (train via
   `python -m rlbot.training.train --config configs/experiments/<exp>.yaml`). Diego doesn't
   use this for training, but its `evaluation/evaluate.py` helpers ARE reused by the eval scripts.

## Key files

| File | Purpose |
|---|---|
| `diego-bots/simple_bot.py` | Active training script (run: `python diego-bots/simple_bot.py`) |
| `diego-bots/simple_bot_play.py` | Watch ONE checkpoint self-play in rlviser |
| `diego-bots/eval_render.py` | Watch TWO checkpoints 1v1 in rlviser (blue vs orange) |
| `scripts/eval_checkpoint_progression.py` | Eval every checkpoint vs a fixed reference → win-rate curve to wandb |
| `src/rlbot/rewards/nexto_style.py` | `build_nexto_style_reward()` — 10-component Nexto-inspired base |
| `src/rlbot/rewards/custom_rl.py` | 6 custom rewards (see below) |
| `src/rlbot/rewards/zero_sum.py` | `ZeroSumReward` wrapper (1v1 competitive) |
| `src/rlbot/utils/rl_constants.py` | Canonical RL physics constants (field, car, boost, kickoff coords) |
| `src/rlbot/state_setters/kickoff_scenarios.py` | `RandomKickoffSetter`, `FixedKickoffSetter` |
| `src/rlbot/evaluation/evaluate.py` | `_load_policy`, `_build_eval_env`, `_resolve_checkpoint_path`, `_action_to_int` (reused everywhere) |

## Current champion: experiment `nexto_plus_kickoff_512`

- **Checkpoint**: `diego-bots/checkpoints/nexto_plus_kickoff_512/nexto_plus_kickoff_512-1779995764491146500/416220408` (~416M timesteps)
- **Architecture**: 512×3 MLP (both actor + critic)
- **Reward**: v2 stack (see below), all wrapped in `ZeroSumReward(team_spirit=0, opp_scale=1)`
- **Results**: 40% win vs Marian's 1.35B bot, 90%+ vs the old 130M `nexto_rewards` baseline
- **Confirmed behaviors** (Diego watched in rlviser): aerials emerging, dribbling improved,
  **zero own-goals** (the anti-own-goal reward worked), goes for aerial challenges.

### Current v2 reward stack (in `simple_bot.py` build_env)

```
CombinedReward(
  nexto_base (unwrapped 10-component)   weight 1.0
  SupersonicReward()                     weight 0.05
  AerialBallReward()                     weight 0.5
  AerialTouchReward()                    weight 1.5   # real aerial contact
  BigBoostProximityReward()              weight 0.5   # ball-distance aware
  BackboardDefenseReward()               weight 0.4
  BallAwayFromOwnGoalReward()            weight 0.6   # anti own-goal
) → wrapped in ZeroSumReward
```

State setter: `WeightedSampleSetter([RandomState 0.7, RandomKickoffSetter 0.3])`.

## Experiment lineage (all checkpoints preserved for comparison)

| Experiment | Arch | Reward | Reached | Notes |
|---|---|---|---|---|
| `baseline` | 256×3 | 3-component | 16.8M | OG plateau, weak |
| `nexto_rewards` | 256×3 | 10-component Nexto | 130M | Strong; wins 100% vs baseline |
| `nexto_plus_kickoff` | 256×3 | 14-component + kickoff | 17.6M | Abandoned for 512 |
| `nexto_plus_kickoff_512` | 512×3 | v2 (current) | 416M+ | **CHAMPION** |
| `marian_iterations` | 512×3 | (teammate's) | 1.35B | Reference opponent, not ours |

## How to check what's training / latest checkpoint

```powershell
# Latest checkpoint of the active experiment:
Get-ChildItem -Recurse -Directory diego-bots\checkpoints\nexto_plus_kickoff_512 | Where-Object { $_.Name -match '^\d+$' } | Sort-Object { [int]$_.Name } -Descending | Select-Object -First 1
```

## The eval workflow (how Diego measures progress)

Policy Reward is **useless** under ZeroSum (it hovers at 0 — see LESSONS). Progress is
measured by **win rate vs a fixed reference**:

```powershell
python scripts/eval_checkpoint_progression.py --experiment nexto_plus_kickoff_512 --reference "diego-bots/checkpoints/nexto_rewards/nexto_rewards-1779876636941376400/130506086" --episodes-per-checkpoint 20 --subsample-every 5
```

Logs `blue_win_rate vs cumulative_timesteps` to wandb. THE progress metric.

## Immediate priorities (detail in docs/NEXT_TASKS.md)

1. **Dribbling-to-goal reward** — bot dribbles well but into the side walls/poles. Add a
   small reward scaling ball control by proximity to the ENEMY goal.
2. **Kickoff logic** — add kickoff-specific reward/behavior.
3. **Billions-scale training** — laptop can't reach billions in time. Plan a cloud-GPU path.

## Conventions you must follow

- **One experiment = one `EXPERIMENT_NAME`** at top of `simple_bot.py`. Changing rewards/arch
  meaningfully → new name (fresh checkpoint folder) OR resume same name (fine-tune).
- **Changing architecture requires a fresh experiment** (or matching the saved checkpoint's
  layer sizes) — shape-mismatch error otherwise.
- **wandb runs auto-rename on exit** to `baseline_<endts>_r<N>` via the atexit hook.
- **Preserve milestone checkpoints manually** — `n_checkpoints_to_keep=5` rotates old ones out.
- **Verify edits**: `python -m py_compile <file>` + a build_env() smoke test before declaring done.
