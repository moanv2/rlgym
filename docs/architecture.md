# Architecture

## Why this layout

A bot project drifts into spaghetti fast: rewards entangled with obs, hyperparameters
hardcoded inside training scripts, "the run we presented" being whichever weights
happened to be on disk last. This layout is built around three rules:

1. **One YAML defines one experiment.** No hardcoded hyperparameters anywhere.
2. **Components are registered, not imported.** New rewards / obs / state setters
   become available to configs by registering a name; configs reference names.
3. **Checkpoints carry their context.** Every run's folder contains the config
   that produced it and the git SHA — you can always reproduce.

## Data flow

```
configs/experiments/exp_NNN.yaml
        │
        ▼
   load_config()       <-- src/rlbot/utils/config.py (deep-merges `extends:`)
        │
        ▼
   make_env_builder()  <-- src/rlbot/env/builder.py
        ├── build_obs()              src/rlbot/obs/
        ├── build_action_parser()    src/rlbot/actions/
        ├── build_reward()           src/rlbot/rewards/  (wraps in ZeroSumReward)
        ├── build_state_setter()     src/rlbot/state_setters/
        └── build_terminal_conditions()  src/rlbot/terminal/
        │
        ▼
   rlgym_ppo.Learner   <-- multi-process rollout + PPO update
        │
        ▼
   checkpoints/<experiment_name>/
       ├── run_metadata.json       (config snapshot + git SHA)
       └── <timestep>.pt
```

## Registries

`src/rlbot/utils/registry.py` is a generic `name -> factory` mapping. The
rewards module uses one (`REWARDS`) so configs can reference rewards by string.

To add a new reward:

```python
# src/rlbot/rewards/custom.py
from rlgym_sim.utils import RewardFunction
from rlbot.rewards.registry import REWARDS

@REWARDS.register("aerial_hit")
class AerialHitReward(RewardFunction):
    def __init__(self, scale: float = 1.0):
        ...
```

Then import it from `rlbot/rewards/__init__.py` (so the decorator runs at package
import time), and reference it from a config:

```yaml
rewards:
  components:
    - name: aerial_hit
      weight: 0.5
      kwargs:
        scale: 1.5
```

Obs, action, and state-setter modules use straight `if/elif` builders rather than
registries because we expect very few of each — convert them to registries if/when
the count grows.

## Reproducibility floor

Every run snapshots:

- `configs/experiments/exp_NNN.yaml` (full merged form, into `run_metadata.json`)
- `git rev-parse --short HEAD`
- the wandb run id (auto-logged by rlgym-ppo)
- a deterministic `seed` (set across `random`, `numpy`, `torch`)

If any of these are missing for a checkpoint, treat it as untrusted.

## Why ZeroSumReward by default

In 1v1 self-play, both copies of the policy share parameters. A non-zero-sum
positive reward (e.g. "+1 for moving toward ball") rewards both agents
simultaneously and lets them converge on cooperative behavior that doesn't
generalize to playing against an *adversary*. ZeroSumReward subtracts the
opponent's reward from yours, forcing the policy to actually compete.

## Why LookupAction

`ContinuousAction` makes exploration brutally hard for an early bot.
`DiscreteAction` is actually MultiDiscrete (combinatorial). `LookupAction`
enumerates the ~90 useful input combinations into a single discrete head — the
right default for a 45-day project on a single GPU.

## Where things will get added

- `src/rlbot/models/policy.py` — only if we customize beyond MLP layer sizes
  (e.g. attention over teammates for 2v2/3v3 — not needed for 1v1).
- `src/rlbot/training/callbacks.py` — wandb media uploads, periodic eval-vs-baseline,
  early stopping. Skipped at scaffold time; add when needed.
- `src/rlbot/deployment/bot.py` — RLBot adapter, week 5.
