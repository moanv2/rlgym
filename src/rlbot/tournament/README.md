# Bot Tournament

Single-elimination tournament for the team's 1v1 bots: discover everyone's
strongest checkpoint, seed by training steps, play a bracket, rank 1..N, and
record a ~60s rlviser video of each person's bracket match for the presentation.

Built to run **tomorrow** with whatever checkpoints are present — missing
teammates are skipped (or fetched via the download script) rather than crashing.

## Roster

Edit `ROSTER_CONFIG` in `roster.py`. Each entry is `{owner, name, path}` where
`path` points at a checkpoint folder (or any ancestor — the latest `<timestep>`
dir beneath it is auto-selected). Architecture and obs dimension are read straight
from the saved weights, so DefaultObs (89-dim) and AdvancedObs (107-dim) bots of
any width are all handled automatically — including in the same match (each car is
fed the obs it was trained on, via `obs.PerSideObs`).

Present in the repo today: **diego** (papaya_1024, AdvancedObs), **martin**
(champion 2.1B, AdvancedObs), **marian** (1.35B, DefaultObs). Pending: **nachi**,
**marco**.

## Run it

```bash
conda activate rlbot310

# 0) (optional) fetch teammates' checkpoints listed in manifest.json
python -m rlbot.tournament.download --list      # see what's configured
python -m rlbot.tournament.download             # fetch all with a source set

# 1) run the bracket (headless, full sim speed) -> writes results JSON
python -m rlbot.tournament.run                  # best-of-5, deterministic
#   -> history_and_summary/tournament_results.json  + a printed 1..N ranking

# 2) record each person's bracket match for video (open rlviser.exe first!)
python -m rlbot.tournament.record --all --capture
#   -> videos/<owner>_bracket.mp4   (one ~60s clip per bot)
```

`--capture` screen-records the rlviser window with ffmpeg `gdigrab`. If ffmpeg
isn't installed it prints the exact command so you can capture with OBS instead.
The rlviser window title defaults to `RLViser` — override with `--window-title`.

## Match rules

Best-of-5, **deterministic** (argmax — the mode Martin's champion is strongest in
and reproducible for replays). Sides swap every game; `DefaultState` randomises the
kickoff each game so deterministic policies still play distinct games. Ties break on
aggregate goal differential, then stochastic sudden-death, then seed.

## Layout

| File | Role |
|------|------|
| `roster.py` | Roster config, checkpoint resolution, arch/obs auto-detection (pure + lazy torch) |
| `bracket.py` | Seeding, byes, bracket run, 1..N ranking — **pure**, unit-tested |
| `match.py` | Best-of-N runner + pure `tally_games`/`decide` (unit-tested) |
| `obs.py` | `PerSideObs` env so cross-obs bots can play (lazy rlgym_sim) |
| `policy_io.py` | Load a `DiscreteFF` from weights (lazy torch) |
| `run.py` | CLI: run the tournament, write JSON |
| `record.py` | CLI: render a bot's bracket match, optional ffmpeg capture |
| `download.py` | CLI: fetch teammate checkpoints from `manifest.json` |

Pure logic (bracket/ranking/match-decision) imports no torch or rlgym_sim, so the
tests in `tests/test_tournament_*.py` run anywhere. The heavy env/policy code is
imported lazily only when a match is actually played.
