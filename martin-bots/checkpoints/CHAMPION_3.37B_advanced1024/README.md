# Martin's champion bot (latest) — 1v1

Our current strongest 1v1 Rocket League PPO bot (rlgym-ppo + rlgym_sim).
**Supersedes** `CHAMPION_2.1B_recipeD_advanced1024` — use this one.

| | |
|---|---|
| **Policy** | `PPO_POLICY.pt` (the runnable bot — load with rlgym_ppo `DiscreteFF`) |
| **Trained** | ~3.37B timesteps (PPO, rlgym-ppo self-play) |
| **Obs** | `AdvancedObs` (107-dim) |
| **Action** | `LookupAction` (90 discrete) |
| **Net** | 1024 x 3 MLP (shared actor/critic), inferred from the weights |
| **Deploy** | **Deterministic / argmax** inference — noticeably stronger than sampling for this policy |

## Strength
Beats both shared Diego bots — **papaya 1.34B** and the **1.18B 512x3** — in deterministic
head-to-head eval. (Run it yourself for exact numbers; spar it deterministic for its real level.)

## Running it
Loads in any harness that rebuilds a `DiscreteFF` policy (hidden sizes auto-inferred from the
weights) and uses the 90-action `LookupAction`. Its obs is `AdvancedObs` (107-dim), so against a
`DefaultObs` (89-dim) bot use a cross-obs adapter that builds each bot its own obs from the shared
game state — see `scripts/tournament.py` in this repo, which handles exactly that and can also run
a full round-robin + Elo + a championship final. Deploy **deterministic**.
