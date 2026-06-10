# RLBot v5 Bot — Skeleton Build Brief

> Feed this file to Claude Code as the task spec. It tells you (Claude Code) **what to build, against which API version, and what NOT to do.** Follow the "Ground rules" before writing any code.

---

## 0. What I want

Scaffold a minimal but correct **RLBot v5 (beta)** Python bot project that compiles, connects to `RLBotServer`, and drives a car. Start from an Always-Towards-Ball baseline so I can confirm the loop works, then leave clear seams where I plug in real strategy later.

**My bot's intended behavior (edit this before building):**
- v0: drive toward the ball, boost when aligned, basic flip/jump near the ball.
- Future: structured decision layer (kickoff / defense / shot), and an optional pluggable policy module so I can later swap in a learned controller. Leave a `policy/` seam but do **not** build ML now.

---

## 1. Ground rules (read first — this is the part people get wrong)

1. **Target RLBot v5, NOT v4.** v4 and v5 have *different* Python APIs. Most Google/StackOverflow results show v4. Do not use any of these v4 symbols:
   - ❌ `from rlbot.agents.base_agent import BaseAgent, SimpleControllerState`
   - ❌ `class X(BaseAgent)`, `initialize_agent()`, `get_field_info()`, `self.renderer.draw_string_3d` (v4 signatures)
   - ❌ `.cfg` config files
2. **Source of truth = the official v5 template.** Before writing code, fetch and read these and mirror their structure/API exactly:
   - Template repo (the skeleton to copy): https://github.com/RLBot/python-example
   - Python interface (the library API): https://github.com/RLBot/python-interface and its wiki https://github.com/RLBot/python-interface/wiki
   - v5 wiki (avoid v4 pages): https://wiki.rlbot.org/v5/ — esp. Config Files, Game Data, Rendering, Ball path prediction.
   - flatbuffer schema (field/packet definitions): https://github.com/RLBot/flatbuffers-schema
3. **If my skeleton below disagrees with the live template on exact method names or signatures, the template wins.** I wrote the skeleton from the v5 API shape, but verify each import and method against `python-example/src/bot.py` and the `rlbot` package source before finalizing.
4. **120 Hz only.** v5 removed non-standard tick rates. Every `get_output` call must return fast. Any heavy computation goes on a separate thread, never inline in the tick.
5. **TOML, not CFG.** All config is `.toml`.
6. Don't install or wire up any ML framework, RLGym, or PyTorch in this pass. Leave the seam, keep deps minimal.

---

## 2. My environment (target this exactly)

- **OS:** Windows 11, PowerShell.
- **Python:** use a dedicated project **venv** (the v5 template asks for Python 3.12+; my Miniconda base is 3.11.9, so create a fresh 3.12 venv rather than reusing the `ds311` conda env). All commands below are PowerShell.
- **RLBotServer / GUI:** assume installed separately via the v5 Windows installer (https://github.com/RLBot/launcher/releases/tag/installer). The bot project does NOT bundle the server.
- Keep everything cross-platform-friendly where trivial (config has `run_command` and `run_command_linux`), but I only need Windows to work.

---

## 3. Target project structure

Create this layout. Match filenames to the official template; only the bot's own module names are mine.

```
rl-bot/
├─ .env                     # RLBOT_AGENT_ID etc. (mirror template)
├─ .gitignore              # venv/, __pycache__/, *.spec build artifacts, etc.
├─ requirements.txt        # just: rlbot   (v5 package; pin a version)
├─ rlbot.toml              # match config used by run.py (which bots, mutators, map)
├─ dev.toml                # dev-tweaked match config (rendering on, state setting on)
├─ run.py                  # starts RLBotServer + a match from rlbot.toml
├─ run_only.py             # runs ONLY the bot process (server already running)
├─ README.md              # how to set up + run, Windows-first
└─ src/
   ├─ bot.py               # entry point: defines the Bot subclass + __main__
   ├─ bot.toml             # this bot's agent config (agent_id, name, run_command, loadout)
   ├─ loadout.toml         # car appearance
   └─ policy/              # EMPTY SEAM for future strategy/ML — add __init__.py + a stub
      └─ __init__.py
```

---

## 4. The v5 API contract (verify against template, then implement)

The v5 Python pattern is approximately this. **Confirm exact names against `python-example/src/bot.py` and the `rlbot` package — adjust if they differ.**

```python
# src/bot.py
from rlbot.flat import ControllerState, GamePacket   # flat types come from rlbot.flat
from rlbot.managers import Bot                        # base class lives in rlbot.managers


class MyBot(Bot):
    def initialize(self):
        # Called once after the framework hands us field_info + match_config.
        # Available here (verify exact attribute names): self.index, self.team,
        # self.name, self.field_info, self.match_config, self.renderer
        pass

    def get_output(self, packet: GamePacket) -> ControllerState:
        # Runs every tick @120Hz. Must be fast. Return a ControllerState.
        # Read state from `packet` (cars, ball, boost pads, game_info).
        # Ball prediction is available via the bot's prediction accessor — verify name.
        controls = ControllerState()
        controls.throttle = 1.0
        return controls


if __name__ == "__main__":
    # The agent_id string MUST match the agent_id in src/bot.toml.
    # The base class typically reads RLBOT_AGENT_ID from env and falls back to this.
    MyBot("moanv2/myrlbot").run()
```

Key things to confirm and wire correctly from the real API:
- Exact import paths (`rlbot.managers`, `rlbot.flat`) and class name for the base (`Bot`).
- The lifecycle method name (`initialize` vs other) and what's populated by the time it runs.
- How **ball prediction** is exposed (per-tick struct + a `find_slice_at_time`-style helper) — see the v5 "Ball path prediction" wiki page.
- The **rendering** API (v5 reworked it: `RenderAnchor`s, per-bot toggle, no keyboard shortcuts). Wire one debug render (e.g. a line car→ball) but guard it so it no-ops when rendering is off.
- Reading the packet: respect the count fields when iterating fixed-length lists.

---

## 5. Config files to generate

Generate these and keep `agent_id` consistent between `src/bot.toml` and the constructor in `bot.py`. Use the template's exact keys — verify against `python-example` and the v5 "Configuration Files" wiki page.

**`src/bot.toml`** — must include (at minimum): a `[settings]` table with `name`, a unique `agent_id` (e.g. `"moanv2/myrlbot"`), `run_command` (Windows, points at running `src/bot.py` in the venv) and `run_command_linux`, plus `loadout_file = "loadout.toml"`. If I ever make it a hivemind, that's `hivemind = true` here — leave it false/commented for now.

**`rlbot.toml`** — a match config: `[rlbot]` (launcher = "Steam" or "Epic"), `[match]` (game_mode, map, rendering/state-setting flags), one `[[cars]]` entry for my bot pointing at `src/bot.toml`, and an opponent (`type = "Psyonix"`, a skill level) so I can test 1v1. Include a `[mutators]` block with defaults.

**`dev.toml`** — same as `rlbot.toml` but with `enable_rendering = true` and `enable_state_setting = true` for debugging.

**`src/loadout.toml`** — minimal car appearance; copy the template's keys.

**`.env`** — mirror the template (e.g. `RLBOT_AGENT_ID`).

---

## 6. Build steps (do these in order)

1. Fetch/read the four reference sources in §1.2 and reconcile the API in §4 with reality.
2. Create the folder structure (§3) and `requirements.txt` containing `rlbot` (pin the installed 5.x version).
3. Write `src/bot.py` as an Always-Towards-Ball baseline:
   - compute a steer toward the ball's ground position,
   - `throttle = 1.0`, `boost` when roughly facing the ball and not already max speed,
   - simple `jump`/flip when very close to the ball,
   - one guarded debug render (car → target line).
4. Write all config files (§5) with consistent `agent_id`.
5. Add the `src/policy/` seam with an `__init__.py` and a `Policy` stub class exposing `decide(packet) -> ControllerState` that `bot.py` could delegate to later (don't use it yet).
6. Write `README.md` with the Windows setup + run instructions (§7).
7. Add `.gitignore`.
8. **Self-check:** confirm no v4 symbols remain (grep for `BaseAgent`, `SimpleControllerState`, `initialize_agent`, `.cfg`). Confirm `get_output` does no blocking/heavy work. Confirm `agent_id` matches in both places.

---

## 7. Run instructions to put in README (Windows / PowerShell)

```powershell
# from project root
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

# make sure RLBotServer/GUI is installed (v5 launcher) and Rocket League is owned on Steam/Epic
python run.py            # launches server + match from rlbot.toml
# during dev, with server already up:
python run_only.py
```

---

## 8. Out of scope for this pass (note in README under "Next steps")

- ML/learned policy (RLGym, PyTorch on the RTX 4070) — the `policy/` seam is the future hook.
- Hivemind (multiple drones, one process).
- Botpack packaging (PyInstaller + `bob.toml`) — only add if I ask.

---

## 9. Acceptance criteria

- `pip install -r requirements.txt` succeeds in a fresh 3.12 venv.
- `python run.py` starts a match and the bot car drives toward the ball in-game.
- No v4 API symbols anywhere.
- `agent_id` is consistent; configs are valid TOML matching v5 keys.
- `get_output` is non-blocking and returns a valid `ControllerState` every tick.
- Code has the `policy/` seam wired as a no-op delegate, ready to extend.
