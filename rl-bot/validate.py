"""Offline self-check for the RLBot v5 skeleton.

Runs WITHOUT a server, Rocket League, or a GPU — so it works locally, in CI, and
inside the Docker image. It enforces the brief's acceptance checks:

    - no v4 API symbols anywhere (BaseAgent, SimpleControllerState,
      initialize_agent) and no .cfg config files,
    - all .toml files parse and contain the required keys,
    - agent_id is consistent across src/bot.toml, src/bot.py, and .env,
    - (best effort) the bot module imports cleanly when `rlbot` is installed.

Exit code 0 = all good; non-zero = at least one failure (CI-friendly).

Usage:  python validate.py
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).parent
SRC = ROOT / "src"

failures: list[str] = []
notes: list[str] = []


def fail(msg: str) -> None:
    failures.append(msg)


def ok(msg: str) -> None:
    notes.append(msg)


# ---------------------------------------------------------------------------
# 1. No v4 API symbols / no .cfg files.
# ---------------------------------------------------------------------------
V4_SYMBOLS = ("BaseAgent", "SimpleControllerState", "initialize_agent")
# Skip generated/third-party trees — we only validate OUR project's files, not
# the venv, build artifacts, or vendored package data (which legitimately ship
# their own .cfg / setup.cfg files).
IGNORED_DIRS = {"venv", ".venv", "env", "build", "dist", "__pycache__", ".git"}


def is_ignored(path: Path) -> bool:
    return any(part in IGNORED_DIRS for part in path.relative_to(ROOT).parts)


for py_file in SRC.rglob("*.py"):
    text = py_file.read_text(encoding="utf-8")
    for sym in V4_SYMBOLS:
        if sym in text:
            fail(f"v4 symbol '{sym}' found in {py_file.relative_to(ROOT)}")
stray_cfgs = [p for p in ROOT.rglob("*.cfg") if not is_ignored(p)]
if stray_cfgs:
    listing = ", ".join(str(p.relative_to(ROOT)) for p in stray_cfgs)
    fail(f"found .cfg config file(s); v5 uses .toml only: {listing}")
ok("v4-symbol scan complete")

# ---------------------------------------------------------------------------
# 2. TOML parses + required keys.
# ---------------------------------------------------------------------------
def load_toml(rel: str) -> dict:
    path = ROOT / rel
    if not path.exists():
        fail(f"missing config file: {rel}")
        return {}
    try:
        with path.open("rb") as fh:
            return tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        fail(f"invalid TOML in {rel}: {exc}")
        return {}


bot_cfg = load_toml("src/bot.toml")
rlbot_cfg = load_toml("rlbot.toml")
dev_cfg = load_toml("dev.toml")
load_toml("src/loadout.toml")

settings = bot_cfg.get("settings", {})
for key in ("name", "agent_id", "loadout_file", "run_command"):
    if key not in settings:
        fail(f"src/bot.toml [settings] missing required key '{key}'")

if "match" not in rlbot_cfg:
    fail("rlbot.toml missing [match] table")
if "rlbot" not in rlbot_cfg:
    fail("rlbot.toml missing [rlbot] table")
cars = rlbot_cfg.get("cars", [])
if not isinstance(cars, list) or len(cars) < 2:
    fail("rlbot.toml should define at least two [[cars]] (bot + opponent)")
else:
    if not any("config_file" in c for c in cars):
        fail("rlbot.toml has no [[cars]] entry pointing at a bot config_file")
ok("TOML structure check complete")

# ---------------------------------------------------------------------------
# 3. agent_id consistency: bot.toml <-> bot.py constructor <-> .env.
# ---------------------------------------------------------------------------
toml_agent_id = settings.get("agent_id")
bot_py = (SRC / "bot.py").read_text(encoding="utf-8") if (SRC / "bot.py").exists() else ""
m = re.search(r'MyBot\(\s*["\']([^"\']+)["\']\s*\)', bot_py)
py_agent_id = m.group(1) if m else None

if toml_agent_id and py_agent_id and toml_agent_id != py_agent_id:
    fail(f"agent_id mismatch: bot.toml='{toml_agent_id}' vs bot.py='{py_agent_id}'")
elif not py_agent_id:
    fail("could not find MyBot(\"...\") agent id in src/bot.py")
else:
    ok(f"agent_id consistent: '{toml_agent_id}'")

env_path = ROOT / ".env"
if env_path.exists():
    env_text = env_path.read_text(encoding="utf-8")
    em = re.search(r"^RLBOT_AGENT_ID\s*=\s*(.+)$", env_text, re.MULTILINE)
    if em and toml_agent_id and em.group(1).strip() != toml_agent_id:
        fail(f".env RLBOT_AGENT_ID='{em.group(1).strip()}' != bot.toml agent_id='{toml_agent_id}'")

# ---------------------------------------------------------------------------
# 4. Best-effort import smoke check (needs `rlbot` installed).
# ---------------------------------------------------------------------------
sys.path.insert(0, str(SRC))
try:
    import importlib

    importlib.import_module("policy")
    importlib.import_module("util.vec")
    importlib.import_module("bot")  # __main__ guard prevents .run()
    ok("bot module imports cleanly")
except ModuleNotFoundError as exc:
    if exc.name in ("rlbot", "rlbot_flatbuffers"):
        notes.append(f"skipped import check (rlbot not installed: {exc.name})")
    else:
        fail(f"import error: {exc}")
except Exception as exc:  # noqa: BLE001 - surface any import-time error
    fail(f"import raised {type(exc).__name__}: {exc}")

# ---------------------------------------------------------------------------
# Report.
# ---------------------------------------------------------------------------
for n in notes:
    print(f"  ok   - {n}")
for f in failures:
    print(f"  FAIL - {f}")

if failures:
    print(f"\nvalidate.py: {len(failures)} failure(s).")
    sys.exit(1)
print("\nvalidate.py: all checks passed.")
