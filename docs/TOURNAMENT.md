# Bot Tournament (cross-obs round-robin + Elo)

A shared rig to run **everyone's bots against each other** and rank them by Elo — even
though our bots use different observation builders and network sizes. Built for the final
presentation (a real bracket) and for measuring where each bot stands.

Runner: [`scripts/tournament.py`](../scripts/tournament.py) · entrant list: [`configs/tournament_bots.example.yaml`](../configs/tournament_bots.example.yaml)

## Why it's needed

Our bots are **not** drop-in interchangeable:

| | Observation | Action | Net |
|---|---|---|---|
| Diego | DefaultObs (89-dim) | LookupAction (90) | 512x3 |
| Martin | AdvancedObs (107-dim) | LookupAction (90) | 1024x3 |
| Nachi / Marian | (default in their config: DefaultObs + LookupAction) | LookupAction (90) | — |

The standard eval (`rlbot.evaluation.evaluate`) feeds **one** shared observation to both
policies, so it only works when both bots use the same obs builder. Two bots can play each
other **iff** they share the **action space** (the 90-action `LookupAction`) — the
observation builder and the network width can differ.

## How it works

`tournament.py` builds **each bot its own observation** from the shared game state every
step (Diego's car gets a DefaultObs vector, Martin's gets an AdvancedObs vector), and both
cars' chosen action indices go through the single env's `LookupAction`. Network width is
auto-detected from each checkpoint's weights. It plays every pair both sides (to cancel
kickoff/side bias) and ranks everyone with a Bradley-Terry Elo.

## Compatibility — what makes a bot eligible

- **Required:** the 90-action `LookupAction` (the repo default). Different action sets can't play in one env.
- **Supported observations:** `advanced` (AdvancedObs, 107-dim) or `default` (DefaultObs, 89-dim).
- **Custom obs builder:** add it to `make_obs()` in `scripts/tournament.py` (it must be importable), otherwise that bot can't be entered.
- Network size (512x3 / 1024x3 / anything) does **not** matter — it's inferred from the weights.

## Submitting your bot

1. Zip your latest `PPO_POLICY.pt` (just that file is enough to *play*; the optimizers aren't needed for eval).
2. Tell us your **obs builder** (`advanced` or `default`) and confirm you used the 90-action `LookupAction`.
3. We add a line to `configs/tournament_bots.yaml` and run the bracket.

## Running it

```bash
cp configs/tournament_bots.example.yaml configs/tournament_bots.yaml   # edit paths
python scripts/tournament.py --manifest configs/tournament_bots.yaml --games 30
```

Output: a printed Elo standings table + `tournament_results.json` (per-matchup W-L-D + Elo).
Add `--deterministic` for greedy play; `--games` controls episodes per side per matchup.

> Note for teammates' Claude sessions: this is the team's tournament rig. The richer
> experiment harnesses it was distilled from (a per-checkpoint cross-obs adapter, an
> auto-archiving training supervisor, an rlviser live viewer, behavior-metrics eval) live
> under `checkpoints/` on Martin's machine (gitignored); `scripts/tournament.py` is the
> clean, shareable version.
