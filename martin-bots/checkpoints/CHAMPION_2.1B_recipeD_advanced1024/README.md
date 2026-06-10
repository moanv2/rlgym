# Martin's champion bot — 1v1

A 1v1 Rocket League PPO bot (rlgym-ppo + rlgym_sim).

| | |
|---|---|
| **Policy** | `PPO_POLICY.pt` (the runnable bot — load with rlgym_ppo `DiscreteFF`) |
| **Trained** | ~2.13B timesteps, pure self-play |
| **Obs** | `AdvancedObs` (107-dim) |
| **Action** | `LookupAction` (90 discrete) |
| **Net** | 1024 x 3 MLP (shared actor/critic), inferred from the weights |
| **Deploy** | **Deterministic / argmax** inference (much stronger than sampling for this policy) |

## Strength vs the shared Diego 1.18B (512x3), 200 games each
- **Deterministic: 93.4% decisive** (171-12-17), Wilson 95% [88.9, 96.2]
- Stochastic: 74.7% decisive (118-40-42), Wilson 95% [67.4, 80.8]

## Running it in a tournament
Compatible with any harness that loads a `DiscreteFF` policy and uses `LookupAction` (90). Obs differs from a DefaultObs bot (107 vs 89), so use a cross-obs adapter that feeds each bot its own obs from the shared game state. Run it **deterministic** for its real strength.
