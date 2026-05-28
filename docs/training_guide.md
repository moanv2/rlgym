# Training Guide

Operational notes for actually running experiments.

## Launching an experiment

```bash
make train EXP=exp_001_baseline
# or
python -m rlbot.training.train --config configs/experiments/exp_001_baseline.yaml
```

A run produces:

- `logs/<experiment_name>.run_metadata.json` — config snapshot + git SHA (kept out
  of the checkpoint folder so rlgym_ppo's checkpoint saver doesn't choke on it)
- `checkpoints/<experiment_name>/<timestep>/` — periodic policy snapshots
- `logs/<experiment_name>.log` — local log file
- a wandb run in project `rlgym-finalproject`, group `<experiment_name>`

## Resuming

rlgym-ppo's `Learner` auto-resumes if it finds a checkpoint in
`checkpoints_save_folder`. Just rerun the same command.

## Curriculum chaining

The reward-shaping curriculum (`exp_004_chase` → `exp_007_polish`, see
[reward_curriculum.md](reward_curriculum.md)) carries **one** policy across stages by
warm-starting. A stage opts in with:

```yaml
learner:
  init_from: exp_004_chase     # previous stage's experiment_name
  add_unix_timestamp: false    # required so the previous stage's checkpoints are findable
```

On launch, `train.py` warm-starts from `init_from`'s latest checkpoint **only if this
stage has no checkpoints of its own** — so re-launching mid-stage resumes normally.
Because warm-starting carries over the agent's step counter, each stage's
`timestep_limit` is a **cumulative** budget across the chain, not a per-stage count.

A warm-started stage starts a **fresh wandb run** (it doesn't resume the previous
stage's run). Train the stages in order; launching a stage before its `init_from`
stage has any checkpoint raises a clear error.

## Learning rate

`learner.policy_lr` / `learner.critic_lr` are now wired through to the `Learner`
(default `3e-4` if unset). The curriculum lowers LR as the bot matures
(`2e-4 → 1e-4 → 0.8e-4`), per the guide. If KL divergence spikes, lower LR.

## Keeping the machine awake

Training inhibits system sleep for the duration of the run (Windows
`SetThreadExecutionState`, macOS `caffeinate`, Linux `systemd-inhibit`) so a
multi-day run doesn't die when the machine idles. Pass `--no-keep-awake` to disable.
The display is allowed to turn off; only system sleep is blocked.

## Stopping

Ctrl+C in the terminal. The Learner saves before exiting. Don't kill -9.

## Reading the report

Each iteration prints:

```
Policy Reward: 0.038      # avg episode reward — monotonic-ish growth = good
Policy Entropy: 0.807     # exploration. Drops over time. Crashing fast = bad
Value Function Loss: ...  # critic training signal
Mean KL Divergence: ...   # how fast policy moves. Should be small, stable.
SB3 Clip Fraction: ...    # PPO clip rate. 0.05–0.20 is healthy.
Collected Steps per Second: ...
Cumulative Timesteps: ...
```

If KL divergence shoots up, lower learning rate or `ppo_epochs`. If clip fraction
stays near 0, the bot is barely changing — increase `ppo_ent_coef`.

## When to reset vs. continue

| Symptom                                  | Action                          |
|------------------------------------------|---------------------------------|
| Reward growing, slow but steady          | Continue                        |
| Reward flat for >50M steps               | Try reward weight tweak         |
| Reward flat for >150M steps              | Reset with new obs/rewards      |
| Reward dropped suddenly + KL spike       | Roll back to previous checkpoint|
| Bot stuck doing one degenerate behavior  | Reset, raise entropy            |

Resetting = new `experiment_name`, new checkpoint folder. Never reuse a name.

## Conventions for experiment names

`exp_NNN_short_description.yaml` where NNN is monotonic. Why monotonic numbers
(not git SHA): grading committee finds it easier to follow `exp_001 → exp_007`
in slides than 7 hex strings.

## Dry runs

```bash
python -m rlbot.training.train --config configs/experiments/exp_001_baseline.yaml --dry-run
```

Builds the env once and exits. Good for sanity-checking config changes without
spinning up rollout workers.
