"""Tournament roster: which bot belongs to whom, where its checkpoint lives, and
its architecture/obs dimension (auto-detected from the saved weights).

Design notes
------------
- A "checkpoint" is an rlgym-ppo folder containing PPO_POLICY.pt +
  BOOK_KEEPING_VARS.json (the numeric <timestep> directory). ROSTER_CONFIG may
  point at that folder directly OR at any ancestor -- resolve_checkpoint() picks
  the latest (highest-timestep) checkpoint beneath it.
- obs_dim is read straight from the policy's first-layer weight shape: 89 ==
  DefaultObs, 107 == AdvancedObs. This is what lets cross-architecture matches
  work (see rlbot.tournament.obs.make_env).
- Owners whose checkpoint is missing/empty (e.g. a teammate who hasn't pushed
  yet) are SKIPPED with a warning rather than crashing the whole tournament.
  Run `python -m rlbot.tournament.download` to fetch them first.

This module is import-light on purpose: torch is imported lazily inside
detect_arch() so the pure bracket/test code never pays for it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# Repo root = three parents up from this file (src/rlbot/tournament/roster.py).
REPO_ROOT = Path(__file__).resolve().parents[3]

# Known obs builders by 1v1 vector size. Keep in sync with rlbot.tournament.obs.
OBS_DEFAULT = 89   # rlgym_sim DefaultObs
OBS_ADVANCED = 107  # rlgym_sim AdvancedObs / rlbot.obs.advanced_obs
N_ACTIONS = 90      # LookupAction (shared by every bot on the team)

# The five team bots. `path` is relative to the repo root and may be a parent
# folder (latest checkpoint beneath it is auto-selected). Teammates who are not
# yet present resolve to empty teammates/<name>/ folders and are skipped until
# their weights are downloaded.
ROSTER_CONFIG: list[dict[str, str]] = [
    {"owner": "diego",  "name": "Diego — papaya_1024",     "path": "diego-bots/checkpoints/papaya_1024"},
    {"owner": "martin", "name": "Martin — champion 2.1B",  "path": "martin-bots/checkpoints/CHAMPION_2.1B_recipeD_advanced1024"},
    {"owner": "marian", "name": "Marian — 1.35B",          "path": "checkpoints/marian_iterations/1349081288"},
    {"owner": "nachi",  "name": "Nachi",                   "path": "teammates/nachi"},
    {"owner": "marco",  "name": "Marco",                   "path": "teammates/marco"},
]


@dataclass(frozen=True)
class Bot:
    """A tournament entrant resolved to a concrete, loadable checkpoint."""

    owner: str
    name: str
    checkpoint: Path
    obs_dim: int
    hidden_sizes: tuple[int, ...]
    timesteps: int  # cumulative training steps — used to seed the bracket

    @property
    def label(self) -> str:
        return f"{self.name} [{self.obs_dim}d {self.hidden_sizes} @ {self.timesteps:,}]"


def _as_abs(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else REPO_ROOT / p


def resolve_checkpoint(path_str: str) -> Path | None:
    """Return a checkpoint folder containing PPO_POLICY.pt, or None if there is none.

    Accepts the leaf checkpoint folder directly, or any ancestor -- in which case
    the highest-numbered <timestep> dir with weights beneath it is chosen. Handles
    both the nested papaya layout (<exp>/<exp>-<unix>/<timestep>/) and the flat
    teammate layout (<exp>/<timestep>/). Returns None for missing/empty folders so
    callers can skip absent teammates gracefully.
    """
    p = _as_abs(path_str)
    if not p.exists():
        return None
    if (p / "PPO_POLICY.pt").is_file():
        return p
    candidates = [
        d for d in p.rglob("*")
        if d.is_dir() and d.name.isdigit() and (d / "PPO_POLICY.pt").is_file()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda d: int(d.name))


def detect_arch(ckpt_dir: Path) -> tuple[int, tuple[int, ...], int]:
    """Infer (obs_dim, hidden_sizes, n_actions) from the saved policy weights.

    We never trust a hand-written arch: the input/hidden/output dims come straight
    from PPO_POLICY.pt, so a 512x3 DefaultObs bot and a 1024x3 AdvancedObs bot are
    both handled with zero config. torch is imported here (lazily) so importing
    this module stays cheap for the pure bracket logic and its tests.
    """
    import torch

    sd = torch.load(ckpt_dir / "PPO_POLICY.pt", map_location="cpu", weights_only=True)
    weight_keys = [k for k in sd if k.endswith("weight")]
    if not weight_keys:
        raise ValueError(f"No linear weights found in {ckpt_dir / 'PPO_POLICY.pt'}")
    obs_dim = int(sd[weight_keys[0]].shape[1])
    n_actions = int(sd[weight_keys[-1]].shape[0])
    hidden_sizes = tuple(int(sd[k].shape[0]) for k in weight_keys[:-1])
    return obs_dim, hidden_sizes, n_actions


def read_timesteps(ckpt_dir: Path) -> int:
    """Cumulative training steps from BOOK_KEEPING_VARS.json (0 if unavailable).

    Falls back to the numeric checkpoint folder name when the bookkeeping file is
    absent, so teammate checkpoints saved in odd layouts still seed sensibly.
    """
    book = ckpt_dir / "BOOK_KEEPING_VARS.json"
    if book.is_file():
        try:
            data = json.loads(book.read_text())
            ts = data.get("cumulative_timesteps")
            if isinstance(ts, (int, float)):
                return int(ts)
        except (json.JSONDecodeError, OSError):
            pass
    return int(ckpt_dir.name) if ckpt_dir.name.isdigit() else 0


def build_roster(config: list[dict[str, str]] | None = None, *, verbose: bool = True) -> list[Bot]:
    """Resolve ROSTER_CONFIG into loadable Bot entries, skipping absent owners.

    Returns the present bots only (no fabrication for missing teammates). The
    caller decides whether the count is enough to run (>=2 needed).
    """
    config = ROSTER_CONFIG if config is None else config
    bots: list[Bot] = []
    for entry in config:
        owner, name, path = entry["owner"], entry["name"], entry["path"]
        ckpt = resolve_checkpoint(path)
        if ckpt is None:
            if verbose:
                print(f"  [skip] {owner:<7} no checkpoint under {path!r} "
                      f"(run download, or this teammate hasn't pushed yet)")
            continue
        obs_dim, hidden_sizes, n_actions = detect_arch(ckpt)
        if n_actions != N_ACTIONS:
            if verbose:
                print(f"  [skip] {owner:<7} {ckpt} has {n_actions} actions, expected {N_ACTIONS}")
            continue
        if obs_dim not in (OBS_DEFAULT, OBS_ADVANCED):
            if verbose:
                print(f"  [skip] {owner:<7} {ckpt} has unknown obs dim {obs_dim} "
                      f"(expected {OBS_DEFAULT} or {OBS_ADVANCED})")
            continue
        bot = Bot(owner, name, ckpt, obs_dim, hidden_sizes, read_timesteps(ckpt))
        bots.append(bot)
        if verbose:
            print(f"  [ok]   {owner:<7} {bot.label}")
    return bots
