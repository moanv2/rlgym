# RLBot v5 bot — Windows bootstrap (PowerShell).
#
# Creates a dedicated Python 3.12 venv and installs deps, so the project runs
# identically on any Windows machine. Run from the rl-bot/ project root:
#
#     .\scripts\setup.ps1
#
# Requires Python 3.12+ on PATH (`py -3.12` or `python`). v5 does NOT support
# the 3.10 used by the rlbot310 training env — keep them separate.

$ErrorActionPreference = "Stop"

# Resolve project root (parent of this script's folder).
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

# Pick a 3.12 interpreter: prefer the py launcher, else fall back to `python`.
$Py = $null
if (Get-Command py -ErrorAction SilentlyContinue) {
    try { & py -3.12 --version *> $null; if ($LASTEXITCODE -eq 0) { $Py = "py -3.12" } } catch {}
}
if (-not $Py) { $Py = "python" }

Write-Host "[setup] Using interpreter: $Py" -ForegroundColor Cyan
Invoke-Expression "$Py --version"

# Create the venv if missing.
if (-not (Test-Path "venv")) {
    Write-Host "[setup] Creating venv/ ..." -ForegroundColor Cyan
    Invoke-Expression "$Py -m venv venv"
} else {
    Write-Host "[setup] venv/ already exists, reusing it." -ForegroundColor Yellow
}

$VenvPy = Join-Path $Root "venv\Scripts\python.exe"

Write-Host "[setup] Upgrading pip ..." -ForegroundColor Cyan
& $VenvPy -m pip install --upgrade pip

# Prefer the lockfile for reproducibility if it exists, else requirements.txt.
if (Test-Path "requirements.lock") {
    Write-Host "[setup] Installing from requirements.lock (pinned) ..." -ForegroundColor Cyan
    & $VenvPy -m pip install -r requirements.lock
} else {
    Write-Host "[setup] Installing from requirements.txt ..." -ForegroundColor Cyan
    & $VenvPy -m pip install -r requirements.txt
}

Write-Host "[setup] Running offline validation ..." -ForegroundColor Cyan
& $VenvPy validate.py

Write-Host ""
Write-Host "[setup] Done. Activate with:  .\venv\Scripts\activate" -ForegroundColor Green
Write-Host "[setup] Then:  python run.py   (needs RLBotServer + Rocket League installed)" -ForegroundColor Green
