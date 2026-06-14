# Martin's champion bot (latest) — 1v1

Our current strongest 1v1 Rocket League PPO bot (rlgym-ppo + rlgym_sim).
**Supersedes** `CHAMPION_3.37B_advanced1024` — use this one.

| | |
|---|---|
| **Policy** | `PPO_POLICY.pt` (the runnable bot — load with rlgym_ppo `DiscreteFF`) |
| **Trained** | ~3.99B timesteps (PPO, rlgym-ppo self-play) |
| **Obs** | `AdvancedObs` (107-dim) |
| **Action** | `LookupAction` (90 discrete) |
| **Net** | 1024 x 3 MLP, inferred from the weights |
| **Deploy** | **Deterministic / argmax** inference — noticeably stronger than sampling for this policy |

## Strength (thorough eval: 300 games each, mixed start states, both sides, deterministic, Wilson 95% CIs)
Win-rate over decisive games (draws excluded):
- Diego **2.85B v7** (his latest): **59%** (95% CI 0.53–0.65) — ahead, but modestly.
- Diego **papaya 1.34B**: **68%** (0.62–0.73).
- Diego **1.18B (512)**: **84%** (0.79–0.88).
- Marco **2.0B**: **67%** (0.61–0.72).

Beats every uploaded teammate bot. Run it yourself and spar it **deterministic** for its real level.

## Running it
Loads in any harness that rebuilds a `DiscreteFF` policy (hidden sizes auto-inferred from the
weights) and uses the 90-action `LookupAction`. Its obs is `AdvancedObs` (107-dim), so against a
`DefaultObs` (89-dim) bot use a cross-obs adapter that builds each bot its own obs from the shared
game state — see `scripts/tournament.py` in this repo. Deploy **deterministic**.
