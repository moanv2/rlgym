# Checkpoint Progression Eval Dashboard — Plan

## The problem

`Policy Reward` under `ZeroSumReward` is mathematically forced to ~0 in
symmetric self-play. It cannot tell you whether the bot is improving, only
that two identical copies are evenly matched. We need a different metric.

## The solution

A standalone evaluation script that walks every saved checkpoint in an
experiment folder, runs head-to-head matches against a **fixed reference
opponent**, and logs the resulting **win rate / goal differential per
checkpoint** to wandb. That gives a monotonic-ish progress curve that
actually reflects skill improvement.

## Script: `scripts/eval_checkpoint_progression.py`

### Inputs (CLI args)
- `--experiment` — experiment folder name under `diego-bots/checkpoints/` or `checkpoints/`
- `--reference` — path to the fixed reference opponent checkpoint, or `latest:<exp>` shorthand
- `--episodes-per-checkpoint` — default 20–30 (more = lower noise, longer runtime)
- `--device` — `cpu` (default, fine for eval) or `cuda`
- `--deterministic` — flag, default True
- `--subsample-every` — optional int, e.g. `5` = only evaluate every 5th checkpoint to save time
- `--wandb-project` — default `rlgym-finalproject`
- `--wandb-run-name` — default `eval_progression_<exp>_vs_<ref_short>_<timestamp>`
- `--no-wandb` — flag for local-only execution

### Behavior
1. Walk experiment folder, find all numeric-timestep subdirs across all
   session folders, sort by timestep ascending.
2. Optionally subsample (every Nth checkpoint).
3. Load the reference policy once (it does not change).
4. For each blue checkpoint:
   - Load it via the existing `_load_policy` from `src/rlbot/evaluation/evaluate.py`
   - Build a fresh eval env via `_build_eval_env()` (kickoff-based, headless)
   - Play `--episodes-per-checkpoint` games
   - Aggregate metrics
5. Log each checkpoint's metrics as one wandb step keyed by `cumulative_timesteps`.
6. Also write a local JSON to `history_and_summary/eval_progression_<exp>_<unix_ts>.json`
   so a crash mid-run does not lose work.
7. Print a tabular summary at the end.

### Metrics logged per checkpoint
- `cumulative_timesteps` — X-axis on all charts
- `blue_win_rate` — primary metric (0.0–1.0)
- `blue_wins`, `orange_wins`, `draws` — raw counts
- `goal_differential` — total blue_goals − orange_goals across episodes
- `avg_episode_seconds` — match duration proxy (faster wins = more decisive)
- `goals_scored_per_min` — blue offensive output
- `goals_conceded_per_min` — blue defensive failures
- `(optional)` `avg_supersonic_time_pct` — how aggressive the bot plays
- `(optional)` `avg_max_car_z` — proxy for aerial ability per episode

### Wandb visualization
- One wandb run = one full progression eval
- X-axis: `cumulative_timesteps` (global on workspace, already set)
- Charts: `blue_win_rate` as the headline, `goal_differential`, `goals_scored_per_min`,
  `goals_conceded_per_min` as supporting
- The wandb run can sit alongside the training runs in the same project —
  filter by tag `eval_progression` to keep them separate from training runs

### Sample invocation

```powershell
python scripts/eval_checkpoint_progression.py `
    --experiment nexto_plus_kickoff `
    --reference "diego-bots/checkpoints/nexto_rewards/nexto_rewards-1779876636941376400/130506086" `
    --episodes-per-checkpoint 30 `
    --deterministic `
    --subsample-every 2
```

### Why a fixed reference (not self-play among checkpoints)

Two options were considered:
1. **Fixed reference** (chosen): every checkpoint plays the same opponent. Win
   rate directly measures "how much better than the reference does this
   checkpoint play?". Clean, monotonic-ish, easy to interpret.
2. **Round-robin self-play**: every checkpoint plays every other. Produces
   an ELO-like ranking but takes O(N²) time and the absolute strength
   reference shifts (because the field improves together).

Fixed reference wins on simplicity, runtime, and interpretability. Use
your old 130M `nexto_rewards` bot or Marian's 1.35B as the reference. They
do not change so the metric does not drift.

## Runtime estimates

| Setup | Time per checkpoint | Time total at 20 checkpoints |
|---|---|---|
| 20 eps, deterministic, CPU | ~2–4 min | ~40–80 min |
| 30 eps, sampling, CPU | ~3–6 min | ~60–120 min |
| 50 eps, deterministic, CPU | ~5–10 min | ~100–200 min |

If training is running concurrently, expect 20–30% slowdown due to CPU contention.

## Implementation reuse — what already exists

Marian's `src/rlbot/evaluation/evaluate.py` already has:
- `_resolve_checkpoint_path()` — handles `latest:<exp>` shorthand
- `_load_policy()` — loads any checkpoint, reads layer sizes from `BOOK_KEEPING_VARS.json` (so 256×3 and 512×3 checkpoints both work)
- `_build_eval_env()` — kickoff-based 1v1 env
- `_action_to_int()` — action shape normalization
- `evaluate()` — runs N episodes between two policies and returns W/L/D dict

The new script should **import these helpers and loop over checkpoints**,
not re-implement them. Estimated new code: ~150 lines.

## Future extensions (do later if useful)

- **Sliding-window self-play**: each checkpoint also plays the checkpoint
  at 50% of its training. Detects "is the bot improving over its own past?"
  on top of the absolute reference comparison.
- **Multiple references on one chart**: same script, run twice, overlay
  in wandb. E.g. blue_win_rate_vs_130M and blue_win_rate_vs_marian as two
  series on one chart.
- **Per-skill metrics**: count aerial touches, demos, saves, kickoff
  win/loss per kickoff position. Surface which behaviors are emerging.
- **wandb Report integration**: generate the final presentation visualization
  as a wandb Report with markdown narration between charts.
- **Streamlit dashboard variant**: if wandb is not enough, the JSON output
  can feed a Streamlit app for an embedded interactive dashboard.

## Testing checklist (when implementing)

- [ ] Script handles checkpoints from BOTH `diego-bots/checkpoints/<exp>/<session>/<ts>/`
      and `checkpoints/<exp>/<ts>/` (Marian's layout)
- [ ] Skips folders without `BOOK_KEEPING_VARS.json` gracefully
- [ ] Handles architecture mismatch between blue and reference (already
      handled by `_load_policy` reading layer sizes per checkpoint)
- [ ] Writes incremental JSON so a Ctrl+C mid-run keeps partial results
- [ ] Reports progress (`[12/30] cumulative=24M win_rate=0.42 ...`)
- [ ] `--no-wandb` flag works for offline use
- [ ] Final summary table prints regardless of wandb state

## Estimated implementation time

~2 hours including testing. The hard parts are already solved in Marian's
`evaluate.py`. The new script is mostly orchestration + wandb integration.
