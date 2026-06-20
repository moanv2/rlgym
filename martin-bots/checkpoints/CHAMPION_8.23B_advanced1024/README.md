# Martin's champion bot (latest) — 1v1

Our current strongest 1v1 Rocket League PPO bot (rlgym-ppo + rlgym_sim).
**Supersedes** `CHAMPION_3.99B_advanced1024` — use this one.

| | |
|---|---|
| **Policy** | `PPO_POLICY.pt` (the runnable bot — load with rlgym_ppo `DiscreteFF`) |
| **Trained** | ~8.23B timesteps (PPO, rlgym-ppo self-play) |
| **Obs** | `AdvancedObs` (107-dim) |
| **Action** | `LookupAction` (90 discrete) |
| **Net** | 1024 x 3 MLP (auto-inferred from the weights) |
| **Deploy** | **Deterministic / argmax** inference — noticeably stronger than sampling for this policy |
| **Bookkeeping** | `BOOK_KEEPING_VARS.json` — timestep count, model updates, and the full training config |

## Recipe (now shared)
- **Reward stack** (zero-sum, team_spirit 0): `velocity_player_to_ball` 0.1, `face_ball` 0.05,
  `velocity_ball_to_goal` 0.3, `event` 8.0 (goal +1, concede -1, shot 0.1, demo 0.1).
- **PPO**: arch large (1024x3), `ppo_epochs` 2, `ent_coef` 0.01, `ts_per_iteration` 50k,
  `ppo_batch`/`minibatch` 50k, `exp_buffer` 150k, `standardize_returns` true, `standardize_obs` false.
- **State setter**: 70% randomized states (ball + cars random pos/vel, off-ground allowed) + 30% default kickoff.
- **Lineage**: distilled (Kickstarting) from Diego's papaya 1.34B, beta annealed to 0, then pure PPO self-play.
- Full config in `BOOK_KEEPING_VARS.json` and `checkpoints/_recipeH_distill.yaml` in this repo.

## Strength (last rigorous eval was at ~4B; this checkpoint has +4.2B more training on top)
300 games each, mixed start states, both sides, deterministic, Wilson 95% CIs, decisive win-rate:
- Diego **2.85B v7**: **59%** (0.53–0.65)
- Diego **papaya 1.34B**: **68%** (0.62–0.73)
- Diego **1.18B (512)**: **84%** (0.79–0.88)
- Marco **2.0B**: **67%** (0.61–0.72)

Beat every uploaded teammate bot at 4B; this 8.23B is materially stronger (a fresh head-to-head eval is pending). Spar it **deterministic** for its real level.

## Running it
Loads in any harness that rebuilds a `DiscreteFF` (hidden sizes auto-inferred from the weights)
with the 90-action `LookupAction` and `AdvancedObs` (107-dim). Against a `DefaultObs` (89-dim)
bot, build each bot its own obs from the shared game state — see `scripts/tournament.py`. Deploy **deterministic**.
