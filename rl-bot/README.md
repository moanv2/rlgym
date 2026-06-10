# MoanV2 RLBot — RLBot v5 deployment bot

The agent that plays **real Rocket League** (1v1) against another RLBot — e.g. the
classmate's bot at the final. This is the *deployment* side and is deliberately
separate from the RLGym **training** pipeline in `../diego-bots/` and `../src/rlbot/`:

| | Training (`rlgym` / `diego-bots`) | Deployment (this `rl-bot/`) |
|---|---|---|
| Purpose | Learn a policy via self-play PPO | Drive a car in the real game |
| Engine | `rlgym_sim` + RocketSim (headless) | Rocket League + `RLBotServer` |
| Env | conda `rlbot310` (Python **3.10**) | venv / conda `rlbotv5` (Python **3.12+**) |
| Deps | torch, rlgym-ppo, rocketsim | just `rlbot` (v5) |

The bot now drives with the **trained 1B-timestep `nexto_plus_kickoff_512`
policy** (the `src/policy/` learned controller). The original
Always-Towards-Ball logic in `bot.py` remains as an automatic **fallback** that
takes over only if the policy's deps/weights are missing or it hits a runtime
error — so the bot always does *something* sane.

See **"Learned controller"** below for how it works and how to test it.

---

## Requirements

- **Windows 11** (gameplay is Windows-only — see "Why no gameplay Docker?").
- **Python 3.12+** (v5 requirement; *not* the 3.10 `rlbot310` training env).
- **Rocket League** owned on Steam or Epic.
- **RLBot v5 launcher / GUI + `RLBotServer`**, installed separately:
  <https://github.com/RLBot/launcher/releases/tag/installer>

## Setup (Windows / PowerShell)

```powershell
# from this rl-bot/ folder
.\scripts\setup.ps1          # creates venv/ (Python 3.12), installs deps, runs validate.py
.\venv\Scripts\activate
```

Or with conda (mirrors how you manage `rlbot310`):

```powershell
conda env create -f environment.yml
conda activate rlbotv5
python validate.py
```

## Run

```powershell
python run.py              # RLBotServer + 1v1 from rlbot.toml: my bot vs a Psyonix AllStar
python run.py human.toml   # my bot (Blue) vs YOU, a human (Orange) — grab a controller/keyboard

# during development (server already running, iterate on the bot without restarting the match):
python run_only.py   # uses dev.toml semantics: rendering on, you start the bot process yourself
```

On startup the bot prints one of:

```
[policy] learned controller ready: 1B-step nexto_plus_kickoff_512 (greedy)
[policy] learned controller disabled, using baseline (<reason>)
```

The first means the trained policy is driving. The second means it fell back to
the baseline (usually a missing dep or checkpoint) — fix the reason, then rerun.

---

## Learned controller

`src/policy/` runs your trained PPO checkpoint in the real game. It reproduces
the EXACT observation/action pipeline the policy trained against in
`diego-bots/`, so the bot plays at its trained strength:

```
packet --(rlgym-compat V1GameState)--> rlgym-v1-style GameState
       --(vendored DefaultObs)-------> 89-dim obs
       --(DiscreteFF 89->512x3->90)--> action index 0..89
       --(vendored LookupAction)-----> 8-dim controls --> ControllerState
```

| Piece | File | Notes |
|---|---|---|
| Weights | `src/policy/weights/PPO_POLICY.pt` | Bundled copy of the 1B-step `nexto_plus_kickoff_512` checkpoint (`BOOK_KEEPING_VARS.json` alongside it records provenance) |
| Network | `src/policy/discrete_ff.py` | Vendored `DiscreteFF` — layout must match the weights exactly |
| Observation | `src/policy/default_obs.py` | Vendored `DefaultObs`, verified byte-identical to `rlgym_sim`'s |
| Action table | `src/policy/lookup_action.py` | Vendored `LookupAction`, verified identical to the training table |
| Glue | `src/policy/__init__.py` | `Policy`: load net, hold tick_skip=8 cadence, track `previous_action`, decode controls; graceful baseline fallback |

**Determinism.** Greedy argmax by default (cleaner play vs a human). For
stochastic sampling (how it explored in training), set `Policy.DETERMINISTIC =
False` in `src/policy/__init__.py`.

**Updating to a newer checkpoint.** Copy the new `PPO_POLICY.pt` (and its
`BOOK_KEEPING_VARS.json`) over `src/policy/weights/`. If the new run changed the
network width or obs/action space, also update `Policy.LAYER_SIZES` /
`INPUT_SIZE` / `N_ACTIONS` to match — otherwise `load_state_dict` will reject it.

> **Final gameplay verification is Windows-only and needs Rocket League running.**
> The net load, action table, and observation builder are unit-verified offline
> (see "validate.py" / the equivalence checks), but whether the bot *plays well*
> can only be confirmed by launching a real match on your machine.

---

## Reproducibility (IaC)

So it "works on my machine → works on everyone's machine", three layers, pick what fits:

1. **`environment.yml`** — `conda env create -f environment.yml` builds an identical
   Python 3.12 `rlbotv5` env on any machine. Closest analog to your `rlbot310`.
2. **`scripts/setup.ps1` / `setup.sh`** — one-command venv bootstrap (Windows / *nix).
3. **`requirements.lock`** — after the first install run `make freeze` (or
   `pip freeze > requirements.lock`) and commit it. Teammates then get byte-for-byte
   identical dependency versions.
4. **Docker (`Dockerfile` + `docker-compose.yml`)** — a **validation/CI** image only:
   ```powershell
   docker compose run --rm validate   # installs deps + runs validate.py in a clean Linux box
   ```

### Why no gameplay Docker?

RLBot v5 drives the **actual Rocket League game** through the native `RLBotServer`,
both Windows-only. A Linux container can't run the game, so there is no container that
"plays" Rocket League. The Docker image here therefore validates the project
(deps install, configs parse, `agent_id` consistent, no v4 API symbols, bot imports) —
it is the reproducibility/CI guarantee, not a runtime. Matches always run on Windows.

### `validate.py`

Offline self-check (no server/GPU needed), used by the setup scripts and the Docker image:

```powershell
python validate.py
```
Checks: no v4 symbols (`BaseAgent`, `SimpleControllerState`, `initialize_agent`, `.cfg`),
all TOML parses with required keys, `agent_id` matches across `src/bot.toml` / `src/bot.py` / `.env`,
and the bot module imports cleanly when `rlbot` is installed.

---

## Project layout

```
rl-bot/
├─ run.py / run_only.py       # start server+match / match-only
├─ rlbot.toml                 # my bot vs a Psyonix AllStar (default match)
├─ human.toml                 # my bot vs a HUMAN (python run.py human.toml)
├─ dev.toml                   # dev match (rendering on, manual bot start)
├─ requirements.txt           # rlbot (v5) + torch + numpy + rlgym-compat
├─ environment.yml            # conda env (Python 3.12)
├─ scripts/setup.ps1|sh       # env bootstrap
├─ Dockerfile / compose       # reproducible validation image (NOT gameplay)
├─ validate.py                # offline self-check / CI gate
└─ src/
   ├─ bot.py                  # MyBot(Bot): learned policy + baseline fallback
   ├─ bot.toml                # agent config (agent_id = moanv2/myrlbot)
   ├─ loadout.toml            # car appearance
   ├─ util/                   # template helpers (vec, drive, orientation, sequence,
   │                          #   boost_pad_tracker, ball_prediction_analysis)
   └─ policy/                 # LEARNED CONTROLLER (runs the trained checkpoint)
      ├─ __init__.py          #   Policy: net load, tick loop, decode, fallback
      ├─ discrete_ff.py       #   vendored DiscreteFF network
      ├─ default_obs.py       #   vendored DefaultObs (89-dim)
      ├─ lookup_action.py     #   vendored LookupAction table (90 actions)
      └─ weights/             #   PPO_POLICY.pt + BOOK_KEEPING_VARS.json (1B step)
```

`agent_id` is `moanv2/myrlbot` and **must stay identical** in `src/bot.py`
(`MyBot("moanv2/myrlbot")`) and `src/bot.toml` (`agent_id = ...`). `validate.py` enforces this.

---

## Next steps

- **Learned controller** — DONE. `policy.Policy.decide` loads the 1B-step
  checkpoint, rebuilds the trained `DefaultObs`, runs a forward pass, and decodes
  to controls (see "Learned controller" above).
- **Play-test & tune**: run `python run.py` (vs Psyonix) and `python run.py
  human.toml` (vs you) on Windows; if it looks twitchy, try
  `Policy.DETERMINISTIC = False`.
- **Optional strategy overrides**: layer kickoff / shot-selection heuristics on
  top of the policy for situations it handles poorly.
- **Hivemind / botpack packaging (PyInstaller)** — only if needed.

> v5 API was mirrored from the official template (`RLBot/python-example`). If the live
> template and this code ever disagree on a method name/signature, the template wins —
> re-verify against `python-example/src/bot.py` and the `rlbot` package.
