# System Diagram — How Everything Connects

Visual reference for the rlgym-finalproject pipeline. Four diagrams: the full stack, one training iteration, the on-disk file layout, and the sequence of what happens when you run training.

---

## 1. The full stack — top to bottom

```
┌─────────────────────────────────────────────────────────────────────────┐
│  YOU (writing Python, watching wandb dashboards, pressing Ctrl+C)         │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │ writes / runs / monitors
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          YOUR CODE (Python)                              │
│                                                                          │
│  ┌────────────────────────────┐    ┌────────────────────────────────┐   │
│  │ diego-bots/                │    │ src/rlbot/    (modular framework)│  │
│  │  • simple_bot.py           │    │  • training/train.py            │   │
│  │    (standalone, all knobs  │    │    (YAML config driven)         │   │
│  │     visible inline)        │    │  • env/, obs/, actions/,        │   │
│  │  • simple_bot_play.py      │    │    rewards/, state_setters/     │   │
│  │  • checkpoints/<exp>/...   │    │  • utils/ (config, registry,    │   │
│  │                            │    │           logging, seeding)     │   │
│  └────────────────────────────┘    └────────────────────────────────┘   │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │ imports / calls
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                EXTERNAL LIBRARIES (pip installed)                        │
│                                                                          │
│  ┌──────────────────┐  ┌──────────────────┐  ┌────────────────────────┐ │
│  │ rlgym_ppo        │  │ rlgym_sim        │  │ PyTorch (CUDA wheel)    │ │
│  │ (Learner,        │──│ (env wrapper,    │  │ wandb (logging)         │ │
│  │  PPO algorithm,  │  │  gym-style API)  │  │ rlviser-py (vis bindings│ │
│  │  rollout mgmt)   │  │                  │  │                        ) │ │
│  └────────┬─────────┘  └─────────┬────────┘  └────────────────────────┘ │
└───────────┼──────────────────────┼──────────────────────────────────────┘
            │                      │
            │ uses                 ▼
            │           ┌─────────────────────┐
            │           │ RocketSim.pyd       │  ← compiled C++ extension
            │           │ (physics engine)    │     installed as a binary .pyd
            │           └──────────┬──────────┘
            │                      │ reads at startup
            │                      ▼
            │           ┌─────────────────────────┐
            │           │ collision_meshes/soccar/│  ← from Rocket League dump
            │           │ 16 .cmf files           │     gitignored; shared via Drive
            │           └─────────────────────────┘
            │
            ▼ writes during training
┌─────────────────────────────────────────────────────────────────────────┐
│                     OUTPUTS (artifacts saved)                            │
│                                                                          │
│  ┌──────────────────────────┐ ┌──────────────────┐ ┌──────────────────┐ │
│  │ diego-bots/checkpoints/  │ │ wandb (cloud)    │ │ history_and_     │ │
│  │ <experiment>/            │ │ runs live forever│ │ summary/         │ │
│  │ <session>-<unix_ts>/     │ │ at wandb.ai/     │ │ run_NNN_<exp>.   │ │
│  │ <timestep>/              │ │ <entity>/        │ │   json           │ │
│  │ PPO_POLICY.pt etc.       │ │ <project>        │ │ (one per session)│ │
│  └──────────────────────────┘ └──────────────────┘ └──────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. One training iteration — what the Learner does in a loop

```
                  rlgym_ppo.Learner.learn() — loops until Ctrl+C or timestep_limit

                  ┌───────────────────────────────────────────────────┐
                  │                                                   │
                  │     Iteration N  (collects 50,000 timesteps)       │
                  │                                                   │
                  │  ┌───────────────────────────────────────────┐    │
                  │  │ 1. ROLLOUT (8 parallel workers)           │    │
                  │  │                                           │    │
                  │  │  for each step:                           │    │
                  │  │    state → obs_builder → obs vector       │    │
                  │  │    obs → policy network → action probs    │    │
                  │  │    sample action → LookupAction lookup    │    │
                  │  │    8-dim controller → env.step()          │    │
                  │  │    → new state + reward                   │    │
                  │  │  (all in RAM, no disk writes)             │    │
                  │  └───────────────────┬───────────────────────┘    │
                  │                      ▼                            │
                  │  ┌───────────────────────────────────────────┐    │
                  │  │ 2. COMPUTE ADVANTAGES (GAE)               │    │
                  │  │    reward, value_estimate → advantage     │    │
                  │  └───────────────────┬───────────────────────┘    │
                  │                      ▼                            │
                  │  ┌───────────────────────────────────────────┐    │
                  │  │ 3. PPO UPDATE (2 epochs over the batch)   │    │
                  │  │    policy network ← policy + Adam step    │    │
                  │  │    critic network ← critic + Adam step    │    │
                  │  │    (clip update size — the PPO trick)     │    │
                  │  └───────────────────┬───────────────────────┘    │
                  │                      ▼                            │
                  │  ┌───────────────────────────────────────────┐    │
                  │  │ 4. LOG TO WANDB                           │    │
                  │  │    Policy Reward, Entropy, KL,            │    │
                  │  │    Clip Fraction, FPS, etc.               │    │
                  │  └───────────────────┬───────────────────────┘    │
                  │                      ▼                            │
                  │  ┌───────────────────────────────────────────┐    │
                  │  │ 5. THROW AWAY ROLLOUT DATA                │    │
                  │  │    on-policy: data from the OLD policy    │    │
                  │  │    can't be reused after policy updates   │    │
                  │  │    (no replay buffer like DQN/SAC)        │    │
                  │  └───────────────────┬───────────────────────┘    │
                  │                      ▼                            │
                  │  ┌───────────────────────────────────────────┐    │
                  │  │ 6. SAVE CHECKPOINT (every 100k cumulative)│    │
                  │  │    PPO_POLICY.pt, PPO_VALUE_NET.pt,       │    │
                  │  │    optimizer states, BOOK_KEEPING_VARS    │    │
                  │  └───────────────────┬───────────────────────┘    │
                  │                      │                            │
                  │   ◄──── goto 1 ──────┘                            │
                  └────────────────────────────────────────────────────┘
                                        │
                              Ctrl+C OR timestep_limit reached
                                        ▼
                  ┌────────────────────────────────────────────────────┐
                  │  FINALLY (simple_bot.py try/finally):              │
                  │   • Learner saves final checkpoint                 │
                  │   • Learner calls wandb.finish() — run sealed      │
                  │   • simple_bot.py writes run_NNN.json to           │
                  │     history_and_summary/                           │
                  └────────────────────────────────────────────────────┘
```

---

## 3. File layout on disk

```
rlgym/                                  (= project root, this whole tree)
│
├── diego-bots/                          your personal training scripts
│   ├── simple_bot.py                    main training script
│   ├── simple_bot_play.py               replay a checkpoint in rlviser
│   └── checkpoints/                     [gitignored]
│       ├── baseline/                     archived first experiment (11M plateau)
│       ├── richer_rewards/               (if you train this experiment)
│       └── nexto_rewards/                current experiment
│           └── nexto_rewards-<unix_ts>/  one folder per training session
│               ├── 100000/               one folder per save_every_ts
│               │   ├── PPO_POLICY.pt
│               │   ├── PPO_VALUE_NET.pt
│               │   ├── *_OPTIMIZER.pt
│               │   └── BOOK_KEEPING_VARS.json
│               ├── 200000/
│               └── ...
│
├── src/rlbot/                           modular framework (for advanced YAML-driven runs)
│   ├── training/train.py                YAML-driven entrypoint
│   ├── env/builder.py                   wires obs+actions+rewards+state into rlgym_sim env
│   ├── obs/, actions/, state_setters/, terminal/   component selectors
│   ├── rewards/                         registry pattern; ZeroSumReward, builder, custom
│   ├── models/architectures.py          named MLP shapes (tiny/small/medium/large)
│   └── utils/                           config loader, registry, logging, seeding
│
├── configs/                             experiment YAML configs (for src/rlbot/)
│   ├── default.yaml
│   ├── experiments/exp_001_baseline.yaml etc.
│   └── reward_weights/stage_*.yaml
│
├── history_and_summary/                 [tracked] session summary JSONs
│   ├── run_001_nexto_rewards.json
│   ├── run_002_nexto_rewards.json
│   └── ...
│
├── docs/                                all written documentation
│   ├── architecture.md                  why the framework is structured this way
│   ├── setup.md                         first-time install instructions
│   ├── training_guide.md                day-to-day operational notes
│   ├── roadmap_45_days.md               week-by-week plan
│   ├── reading_training_graphs.md       interpreting wandb charts
│   └── system_diagram.md                ← this file
│
├── collision_meshes/                    [gitignored]
│   └── soccar/mesh_0..15.cmf            16 arena meshes from RL dump
│
├── rlviser.exe                          [gitignored] visualizer binary v0.8.7
├── wandb/                               [gitignored] local wandb cache
├── settings.txt                         [gitignored] rlviser UI config
│
├── tests/                               pytest unit tests
├── scripts/                             CLI wrappers (train, evaluate, visualize)
├── pyproject.toml + requirements*.txt   dependency definitions
├── Makefile                             common dev tasks
├── .github/workflows/ci.yml             GitHub Actions CI
├── README.md                            project overview
└── rlvisor_claude_setup.md              teammate onboarding runbook
```

---

## 4. Sequence — what happens when you run training

```
You type:  python diego-bots/simple_bot.py
   │
   ▼
[simple_bot.py begins executing]
   │
   ├─→ Reads top of file:  EXPERIMENT_NAME = "nexto_rewards"
   │
   ├─→ import rlgym_sim, rlgym_ppo, wandb, RocketSim
   │
   ├─→ find_latest_checkpoint("nexto_rewards")
   │     • scans diego-bots/checkpoints/nexto_rewards/
   │     • returns path to latest cumulative-timestep folder, or None
   │
   ├─→ Builds wandb run name from start cumulative steps:
   │     "nexto_rewards_15M"  (or "nexto_rewards_0M" if fresh)
   │
   ├─→ Defines build_env (factory closure, NOT invoked yet)
   │     Inside build_env when called by workers later:
   │       • makes a CombinedReward of 5 components
   │       • plugs in DefaultObs, LookupAction, RandomState
   │       • returns one rlgym_sim env instance
   │
   ├─→ Instantiates Learner(build_env, n_proc=8, ...)
   │     │
   │     ├─→ Spawns 8 worker processes
   │     │    each worker:
   │     │      • imports rlgym_sim
   │     │      • calls build_env() to make its own env
   │     │      • loads collision_meshes/soccar/ into its RocketSim
   │     │      • waits for policy actions to step the env
   │     │
   │     ├─→ Calls wandb.init(project="rlgym-finalproject",
   │     │                     name="nexto_rewards_15M", ...)
   │     │    creates a cloud run, returns URL
   │     │    prints:  wandb: View run at: https://wandb.ai/.../runs/xyz
   │     │
   │     └─→ if checkpoint_load_folder is set:
   │            loads PPO_POLICY.pt → actor network weights
   │            loads PPO_VALUE_NET.pt → critic weights
   │            loads optimizer states (so Adam momentum preserved)
   │            loads BOOK_KEEPING_VARS.json (cumulative_timesteps counter)
   │
   ├─→ Records  _session_started_at = now()
   │
   ├─→ try:
   │      learner.learn()    ← BLOCKS HERE for minutes to hours
   │        loops the training iteration shown in diagram 2
   │
   ├─→ except KeyboardInterrupt:    ← user pressed Ctrl+C
   │      _session_stop_reason = "keyboard_interrupt"
   │      Learner catches the interrupt, finishes the current iteration,
   │      saves a final checkpoint, calls wandb.finish()
   │
   └─→ finally:
         save_run_summary(...)
            • Determines next run number from existing files
            • Captures wandb.run.id, .url, .summary
            • Reads cumulative_timesteps from latest BOOK_KEEPING_VARS.json
            • Captures config snapshot (rewards, weights, arch, etc.)
            • Writes JSON to:
                history_and_summary/run_NNN_nexto_rewards.json
            • Prints: [summary] saved run #N to ...

[Python process exits cleanly]
```

---

## 5. Why two parallel code paths? `diego-bots/` vs `src/rlbot/`

| Aspect | `diego-bots/simple_bot.py` | `src/rlbot/training/train.py` |
|---|---|---|
| **Purpose** | Learn-by-doing, all knobs visible | Production-grade modular framework |
| **Config** | Hardcoded in the file | YAML-driven (`configs/*.yaml`) |
| **Resume** | Auto-finds latest checkpoint per experiment | YAML field `checkpoint_load_folder` |
| **Wandb** | Inline naming logic | Reads from YAML + threads `wandb_entity` |
| **Best for** | Tweaking rewards quickly, learning PPO | Final structured experiments, A/B at scale |

Both end up calling the same `rlgym_ppo.Learner`. The framework path is what you'll lean on once experiments stabilize; the standalone script is your educational playground until then.

---

## 6. The data ownership map (who writes what)

| Artifact | Written by | Persists where |
|---|---|---|
| Policy weights (`.pt`) | `rlgym_ppo.Learner` | `diego-bots/checkpoints/<exp>/.../<timestep>/` |
| `BOOK_KEEPING_VARS.json` | `rlgym_ppo.Learner` | Inside each checkpoint folder |
| wandb run | `rlgym_ppo.Learner` (calls `wandb.init`) | Cloud at `wandb.ai/<entity>/<project>/runs/<id>` |
| `run_NNN_<exp>.json` | `simple_bot.py` `finally:` block | `history_and_summary/` |
| Gameplay episodes | **Nobody — they're discarded** | RAM only, gone after each iteration |
| Game state mid-step | `RocketSim` (C++) | Process memory of rollout workers |
| Reward values per step | `rlgym_sim.utils.reward_functions.CombinedReward` | Computed on the fly, summarized into the rollout buffer |

The "nobody saves episodes" row is the key insight people miss. PPO is on-policy: every iteration's gameplay is generated by the current policy, consumed by one gradient update, and discarded. There is no replay buffer. All "learning" persists only as updated network weights.
