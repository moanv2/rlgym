#!/usr/bin/env bash
# RLBot v5 bot — Linux/macOS bootstrap (dev/CI parity).
#
# NOTE: you cannot actually PLAY Rocket League on Linux/macOS via this skeleton
# (the game + RLBotServer are Windows). This exists so teammates on other OSes
# can install deps, run validate.py, and edit/lint the bot. Real matches run on
# Windows (scripts/setup.ps1 + run.py).
#
# Usage from the rl-bot/ project root:
#     bash scripts/setup.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Require Python 3.12+.
PY="${PYTHON:-python3.12}"
if ! command -v "$PY" >/dev/null 2>&1; then
  PY="python3"
fi
echo "[setup] Using interpreter: $PY"
"$PY" --version

if [ ! -d "venv" ]; then
  echo "[setup] Creating venv/ ..."
  "$PY" -m venv venv
else
  echo "[setup] venv/ already exists, reusing it."
fi

VENV_PY="$ROOT/venv/bin/python"
echo "[setup] Upgrading pip ..."
"$VENV_PY" -m pip install --upgrade pip

if [ -f "requirements.lock" ]; then
  echo "[setup] Installing from requirements.lock (pinned) ..."
  "$VENV_PY" -m pip install -r requirements.lock
else
  echo "[setup] Installing from requirements.txt ..."
  "$VENV_PY" -m pip install -r requirements.txt
fi

echo "[setup] Running offline validation ..."
"$VENV_PY" validate.py

echo ""
echo "[setup] Done. Activate with:  source venv/bin/activate"
