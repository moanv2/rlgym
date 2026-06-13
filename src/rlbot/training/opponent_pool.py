"""Opponent pool for Fictitious Self-Play (FSP).

We maintain a directory of past policy snapshots and sample one as the
orange-team opponent at every episode reset. This is the single biggest lever
for 1v1 strength: training against diverse past selves prevents the bot from
falling into local minima where it only beats the *current* version of itself.

Each entry in the pool is a directory in rlgym_ppo's checkpoint format
(`PPO_POLICY.pt`, `PPO_VALUE_NET.pt`, `BOOK_KEEPING_VARS.json`). The pool
directory layout mirrors `checkpoints/<experiment>/<timestep>/`.
"""
from __future__ import annotations

import random
import shutil
from pathlib import Path


class OpponentPool:
    def __init__(self, pool_dir: str | Path, max_size: int = 20, seed: int | None = None):
        self.pool_dir = Path(pool_dir)
        self.pool_dir.mkdir(parents=True, exist_ok=True)
        self.max_size = max_size
        self._rng = random.Random(seed)

    def snapshots(self) -> list[Path]:
        return sorted(
            (p for p in self.pool_dir.iterdir() if p.is_dir() and (p / "PPO_POLICY.pt").exists()),
            key=lambda p: int(p.name) if p.name.isdigit() else 0,
        )

    def is_empty(self) -> bool:
        return len(self.snapshots()) == 0

    def sample(self) -> Path | None:
        snaps = self.snapshots()
        if not snaps:
            return None
        return self._rng.choice(snaps)

    def add(self, checkpoint_dir: str | Path) -> Path | None:
        """Copy a checkpoint into the pool. Evicts oldest if past max_size."""
        src = Path(checkpoint_dir)
        if not (src / "PPO_POLICY.pt").exists():
            return None
        dst = self.pool_dir / src.name
        if dst.exists():
            return dst
        shutil.copytree(src, dst)
        self._evict_if_needed()
        return dst

    def _evict_if_needed(self) -> None:
        snaps = self.snapshots()
        while len(snaps) > self.max_size:
            shutil.rmtree(snaps[0])
            snaps = self.snapshots()

    def latest_in(self, checkpoint_root: str | Path) -> Path | None:
        root = Path(checkpoint_root)
        if not root.exists():
            return None
        candidates = [p for p in root.iterdir() if p.is_dir() and p.name.isdigit()
                      and (p / "PPO_POLICY.pt").exists()]
        if not candidates:
            return None
        return max(candidates, key=lambda p: int(p.name))
