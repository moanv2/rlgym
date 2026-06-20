# Making rlviser render the full RL arena + cars (Windows, self-serve)

**Goal:** Stop rlviser crashing the moment a map loads, and get the real Rocket League
arena + car meshes rendering. We fix this ourselves — no need to ask the bot's author.

This is a **config/asset** problem, not a version mismatch. (The version trap — rlviser
v0.8.7 vs rlviser-py — produces a *different* error:
`memory allocation of 72057594037927944 bytes failed`. That is not what we're fixing here.)

---

## 1. Why it crashes (root cause)

rlviser renders the **ball + menu** from built-in/procedural geometry, so it is stable when
launched alone. The **arena and cars are NOT built in.** rlviser loads them from an
`assets/` folder it expects **next to `rlviser.exe`**, and it populates that folder on first
run by **uncooking the real Rocket League game files with UModel**.

When a sim connects and sends the map-load packet (the `Connect → FieldExtra → Field`
states in rlviser's `main.rs`), rlviser tries to read the arena/car meshes. If there is no
`assets/` folder and no way to build one, it crashes.

Confirmed from rlviser's own source (`src/assets.rs`) and the live state of this repo:

| File / folder                | Expected location          | Status here |
|------------------------------|----------------------------|-------------|
| `rlviser.exe`                | repo root                  | ✅ present (53 MB) |
| `settings.txt`               | repo root (auto-written)   | ✅ present — proves the binary itself runs |
| `collision_meshes/soccar/*`  | repo root                  | ✅ present — **but these are RocketSim PHYSICS meshes, not rlviser VISUAL assets. Different system. Do not confuse them.** |
| `umodel.exe`                 | next to `rlviser.exe`      | ❌ MISSING |
| `assets.path`                | next to `rlviser.exe`      | ❌ MISSING |
| `assets/` (+ `assets/files.txt`) | next to `rlviser.exe`  | ❌ MISSING |
| `cache/`                     | next to `rlviser.exe`      | ❌ MISSING |

So rlviser has **no UModel to extract with** and **no extracted assets to read** → map load
= crash. This is the textbook fingerprint.

A second, sneakier failure mode: rlviser's first-run path is **interactive**
(`io::stdin().read_line`, "Try to automatically find the path? (y/n):"). When rlviser is
launched indirectly by `rlviser-py` from inside `env.render()` there is no console to answer
the prompt, so it hangs/aborts exactly on map load. Pre-creating `assets.path` (Step 3)
removes that prompt entirely.

---

## 2. Exactly where rlviser expects everything (verbatim from `src/assets.rs`)

All paths are **relative to the working directory** (the folder rlviser runs from), which
for us is the **repo root** where `rlviser.exe` lives:

```
c:\Users\Lasca\Desktop\Pro\IE\Courses\REINFORCEMENT LEARNING & AUTONOMOUS SYSTEMS\Group Project\rlgym\
```

- `OUT_DIR   = "./assets/"`        → where uncooked visual assets must end up
- `OUT_DIR_VER = "./assets/files.txt"` → manifest/version marker; its presence is how rlviser
  decides assets are "already built" (so subsequent runs skip extraction)
- `UMODEL    = ".\umodel.exe"` (Windows) → the uncooker, **must sit next to `rlviser.exe`**
- `./cache/` (`./cache/mesh/...`, `./cache/material/...`) → mesh/material cache, auto-created
- `assets.path` → plain-text file in the base folder; one line = absolute path to your
  `...\rocketleague\TAGame\CookedPCConsole` folder

**There is NO environment variable** (no `RL_PATH`, no `ROCKET_LEAGUE`, etc.). Discovery is
purely by file presence: `umodel.exe`, `assets.path` in the working dir.

The 11 cooked `.upk` files rlviser uncooks (arena maps + the 6 default car bodies):

```
Startup.upk, MENU_Main_p.upk, Stadium_P.upk, HoopsStadium_P.upk, ShatterShot_P.upk,
Body_MuscleCar_SF.upk, Body_Darkcar_SF.upk, Body_CarCar_SF.upk,
Body_Venom_SF.upk, Body_Force_SF.upk, Body_Vanquish_SF.upk
```

The exact UModel command rlviser builds per file (it drives UModel itself — you don't type
this, it's shown so you know what's happening):

```
umodel.exe -path=<CookedPCConsole> -out=./assets/ -game=rocketleague ^
           -export -nooverwrite -nolightmap -uncook -uc <file.upk>
```

Note: rlviser's pipeline uses **only UModel's `-game=rocketleague -uncook`**. There is **no
RLUPKTool / no separate decryption step in rlviser's code.** RLUPKTool is only for the older
manual Blender/modding workflow, not this one.

---

## 3. The CookedPCConsole path (where your RL cooked assets live)

The decisive subpath is always `...\rocketleague\TAGame\CookedPCConsole`. Only the install
root differs:

- **Steam (most likely):**
  `C:\Program Files (x86)\Steam\steamapps\common\rocketleague\TAGame\CookedPCConsole`
  - Moved Steam library? Steam → right-click Rocket League → Properties → Installed Files →
    Browse → then go into `TAGame\CookedPCConsole`.
- **Epic Games:**
  `C:\Program Files\Epic Games\rocketleague\TAGame\CookedPCConsole`
  - Custom Epic dir? Epic Launcher → Library → Rocket League → ⋯ → Manage → shows install
    location; append `TAGame\CookedPCConsole`.

**Verify before continuing:** that folder must actually contain `Stadium_P.upk` and the
`Body_*_SF.upk` files. If it doesn't, you've got the wrong folder (a common mistake is
pointing at the install root instead of `...\TAGame\CookedPCConsole`).

It must be a **Windows** RL install — rlviser asserts the `TAGame/CookedPCConsole` folder
exists and warns it must be "a Windows version of Rocket League" (a Proton/Linux layout
fails the `.is_dir()` assert).

---

## 4. The fix — do this once

Run everything from the **repo root** (where `rlviser.exe` is). Rocket League must be
**installed locally** but **not running** during extraction.

### Step 1 — Get UModel and put it next to `rlviser.exe`

Two options. **Prefer the RL-aware fork** for guaranteed auto-decrypt:

- **Recommended (RL fork, auto-decrypts RL packages):**
  https://github.com/AltimorTASDK/UModel/releases — download the Windows release, rename the
  executable to `umodel.exe`.
- **Fallback (stock Gildor build):** `umodel_win32.zip` from
  https://www.gildor.org/en/projects/umodel — extract `umodel.exe`. This works with rlviser's
  `-game=rocketleague` flag in most cases; if extraction fails with a decryption error, switch
  to the fork above.

Place `umodel.exe` **in the repo root, next to `rlviser.exe`**:

```
c:\Users\Lasca\Desktop\Pro\IE\Courses\REINFORCEMENT LEARNING & AUTONOMOUS SYSTEMS\Group Project\rlgym\umodel.exe
```

If this is missing you'll see: `Couldn't find UModel! Make sure it's in the same folder as
the executable. Using default assets!` — followed by a crash when a real map arrives.

### Step 2 — Confirm your CookedPCConsole path

Use the appropriate path from Section 3 and confirm it contains the `.upk` files. PowerShell:

```powershell
# Adjust the path to YOUR install (Steam example shown):
$cooked = "C:\Program Files (x86)\Steam\steamapps\common\rocketleague\TAGame\CookedPCConsole"
Test-Path "$cooked\Stadium_P.upk"          # must print True
Get-ChildItem "$cooked\Body_*_SF.upk" | Select-Object Name   # should list the car bodies
```

If `Test-Path` is `False`, find the real folder first (Section 3) before continuing.

### Step 3 — Create `assets.path` next to `rlviser.exe`

This tells rlviser where RL is and **skips the interactive prompt** that otherwise crashes
under `env.render()`. One line, no newline, ASCII. From the repo root in PowerShell:

```powershell
# Use the SAME $cooked path you verified in Step 2:
Set-Content -Path ".\assets.path" `
  -Value "C:\Program Files (x86)\Steam\steamapps\common\rocketleague\TAGame\CookedPCConsole" `
  -Encoding ascii -NoNewline
```

Sanity check it points at the right place:

```powershell
Get-Content .\assets.path
```

Common follow-on mistake: pointing `assets.path` at the **install root** instead of
`...\TAGame\CookedPCConsole`. If extraction finds nothing, this is why.

### Step 4 — Pre-extract ONCE from a real terminal (not via Python)

Do the first extraction with stdout/stdin available so any prompt can be answered and you can
read UModel's progress/errors. From the repo root:

```powershell
.\rlviser.exe
```

Because `assets.path` already exists, it skips the "find RocketLeague.exe?" prompt and starts
uncooking. You should see:

```
Uncooking assets from Rocket League...
Processing file 1/11 (...)
...
Processing file 11/11 (...)
```

This takes ~30–90 s the first time. When it finishes, **close the rlviser window.**

### Step 5 — Verify the assets were built

```powershell
Test-Path .\assets\files.txt    # must print True
Get-ChildItem .\assets | Select-Object -First 20
Test-Path .\cache               # True after first render
```

You should now see `assets\`, `assets\files.txt`, and (after a render) `cache\` next to
`rlviser.exe`. From here on, every run reuses them instantly and the full arena + cars render.

---

## 5. Corrected run flow

Once Steps 1–5 are done, the working dir already has `umodel.exe`, `assets.path`, and a
populated `assets\`. So when `rlviser-py` launches the bundled rlviser binary from
`env.render()`, map-load reads real meshes instead of crashing.

**Critical CWD caveat:** rlviser uses **CWD-relative paths** (`./assets/`, `./umodel.exe`,
`assets.path`, `./cache/`). `rlviser-py` extracts/launches its **own bundled** rlviser binary
and runs it **from the current working directory of your Python process.** So:

1. Always launch your render/sim script with the **working directory = repo root** (the folder
   that has the populated `assets\`). E.g. run `python ...` from
   `c:\Users\Lasca\...\rlgym\`, not from a subfolder.
2. If `rlviser-py`'s bundled binary still can't see the assets (different CWD), **copy the
   generated `assets\` + `cache\` (and `umodel.exe` + `assets.path`)** to wherever that
   process's CWD actually is.

Note on this repo's `spar_env.py`: its `SparEnv.render()` (line ~106) is a **no-op `pass`** —
that wrapper is for training rollout workers and intentionally does not draw anything. To
actually *see* the bot, render via the **underlying `rlgym_sim` env** (the one built in
`src/rlbot/env/builder.py` calling `rlgym_sim.make(...)`), constructing it with
`rlgym_sim.make(..., render=...)` / calling `env.render()` in your play/eval script — not the
spar wrapper. `train.py` correctly keeps `render=False` (line ~152) for headless training.

Typical play/eval loop (conceptual):

```python
import rlgym_sim
env = rlgym_sim.make(...)         # your normal obs/action/reward config
obs = env.reset()
done = False
while not done:
    actions = policy(obs)         # your trained policy
    obs, reward, done, info = env.step(actions)
    env.render()                  # first call boots rlviser; assets/ already built → renders
```

---

## 6. Diagnosis checklist (if it still misbehaves)

- **No `assets\` next to the binary `rlviser-py` launches** → guaranteed crash on map load.
  (This was our case.) → redo Steps 1–5; mind the CWD caveat (Section 5).
- **`umodel.exe` missing next to `rlviser.exe`** → "Couldn't find UModel! ... Using default
  assets!" then crash on real map. → Step 1.
- **`assets.path` empty or wrong** → "Your 'assets.path' file is empty!" /
  "Couldn't find the directory specified in your 'assets.path'!" → Step 3, point at
  `...\TAGame\CookedPCConsole`.
- **Non-Windows / Proton RL layout** → "Couldn't find 'rocketleague/TAGame/CookedPCConsole'
  folder! Make sure you select the correct path to a Windows version of Rocket League." →
  use a Windows RL install.
- **Hangs on first render with no output** → the interactive prompt fired (no `assets.path`).
  → create `assets.path` (Step 3) and pre-extract from a console (Step 4).
- **`memory allocation of 72057594037927944 bytes failed`** → this is the SEPARATE
  version-mismatch trap (rlviser binary vs `rlviser-py`), NOT an asset problem. Keep the
  v0.8.7 binary that matches your `rlviser-py` pin; the asset fix is orthogonal.

---

## 7. Don't commit the extracted assets

`assets*`, `assets.path`, `umodel*`, and `cache` are in rlviser's own `.gitignore` — they are
intentionally local-only derived files, and the extracted Psyonix meshes are **not
redistributable.** Make sure this repo's `.gitignore` covers them and do **not** commit them.

Suggested `.gitignore` additions (repo root):

```
/assets/
/cache/
/assets.path
/umodel.exe
```

---

## Fallback: manual extraction (skip rlviser driving UModel)

Because rlviser keys off the presence of `assets/files.txt` + the uncooked tree, you can build
`assets/` by hand if Step 4 won't run interactively:

1. (Only if your UModel build doesn't auto-decrypt) Decrypt each `.upk` in CookedPCConsole with
   RLUPKTool (https://github.com/AltimorTASDK/RLUPKTool — drag-and-drop the `.upk` files;
   outputs `<name>_Decrypted.upk`).
2. Run UModel yourself with the exact flags from Section 2, pointed at CookedPCConsole (or the
   decrypted files), with `-out=./assets/`, for each of the 11 `UPK_FILES`.
3. Confirm `assets/files.txt` and the mesh tree exist, then run normally.

The RL-aware UModel fork (Section 3, Step 1) collapses decryption + uncook into one step, so
the cleanest path remains Steps 1–5.

---

### Source references

- rlviser `src/assets.rs` — `OUT_DIR="./assets/"`, `OUT_DIR_VER`, `UMODEL=".\umodel.exe"`,
  `assets.path`, `CookedPCConsole`, the 11 `UPK_FILES`, the umodel command, all panic strings.
- rlviser `src/mesh.rs` — embedded stadium layout JSON + `.pskx` mesh loading.
- rlviser `src/main.rs` — `Connect → FieldExtra → Field` states (map loads on connect).
- rlviser PR #2 — "On launch, RLViser looks for `umodel.exe`/`umodel` in the same directory —
  it will pull the assets needed for you."
- rlviser `.gitignore` — `assets*`, `assets.path`, `umodel*`, `cache`, `settings.txt`.
- UModel: https://www.gildor.org/en/projects/umodel ; RL fork:
  https://github.com/AltimorTASDK/UModel/releases ; RLUPKTool:
  https://github.com/AltimorTASDK/RLUPKTool
- rocket-league-gym-sim — "acquire assets from a copy of Rocket League you own with an asset
  dumper."
