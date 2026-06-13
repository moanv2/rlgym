"""Unit tests for the self-play opponent pool. These don't require torch or
rlgym_sim — just filesystem manipulation."""
from __future__ import annotations

from pathlib import Path

import pytest

from rlbot.training.opponent_pool import OpponentPool


def _fake_checkpoint(root: Path, ts: int) -> Path:
    """Create a fake rlgym_ppo checkpoint dir at root/<ts>/ with the
    files OpponentPool considers valid."""
    d = root / str(ts)
    d.mkdir(parents=True, exist_ok=True)
    (d / "PPO_POLICY.pt").write_bytes(b"\x00")
    (d / "BOOK_KEEPING_VARS.json").write_text("{}")
    return d


def test_pool_starts_empty(tmp_path):
    pool = OpponentPool(tmp_path / "pool")
    assert pool.is_empty()
    assert pool.sample() is None
    assert pool.snapshots() == []


def test_pool_picks_up_existing_snapshots(tmp_path):
    pool_dir = tmp_path / "pool"
    pool_dir.mkdir()
    _fake_checkpoint(pool_dir, 1000)
    _fake_checkpoint(pool_dir, 2000)
    pool = OpponentPool(pool_dir)
    snaps = pool.snapshots()
    assert len(snaps) == 2
    assert [s.name for s in snaps] == ["1000", "2000"]  # sorted by ts


def test_pool_ignores_dirs_without_policy_pt(tmp_path):
    pool_dir = tmp_path / "pool"
    pool_dir.mkdir()
    # Valid
    _fake_checkpoint(pool_dir, 1000)
    # Invalid: missing PPO_POLICY.pt
    bad = pool_dir / "2000"
    bad.mkdir()
    (bad / "BOOK_KEEPING_VARS.json").write_text("{}")
    pool = OpponentPool(pool_dir)
    assert len(pool.snapshots()) == 1


def test_pool_add_and_evict(tmp_path):
    src_root = tmp_path / "src"
    src_root.mkdir()
    pool = OpponentPool(tmp_path / "pool", max_size=2)
    for ts in (100, 200, 300, 400):
        pool.add(_fake_checkpoint(src_root, ts))
    snaps = pool.snapshots()
    assert len(snaps) == 2
    assert [s.name for s in snaps] == ["300", "400"]  # oldest evicted


def test_pool_sample_is_deterministic_with_seed(tmp_path):
    pool_dir = tmp_path / "pool"
    pool_dir.mkdir()
    for ts in (100, 200, 300):
        _fake_checkpoint(pool_dir, ts)
    a = OpponentPool(pool_dir, seed=42)
    b = OpponentPool(pool_dir, seed=42)
    picks_a = [a.sample().name for _ in range(10)]
    picks_b = [b.sample().name for _ in range(10)]
    assert picks_a == picks_b


def test_pool_latest_in(tmp_path):
    ckpt_root = tmp_path / "checkpoints" / "exp_xyz"
    ckpt_root.mkdir(parents=True)
    _fake_checkpoint(ckpt_root, 100)
    _fake_checkpoint(ckpt_root, 500)
    _fake_checkpoint(ckpt_root, 250)
    pool = OpponentPool(tmp_path / "pool")
    latest = pool.latest_in(ckpt_root)
    assert latest is not None
    assert latest.name == "500"


def test_pool_latest_in_returns_none_for_empty(tmp_path):
    pool = OpponentPool(tmp_path / "pool")
    assert pool.latest_in(tmp_path / "nonexistent") is None
    empty = tmp_path / "empty"
    empty.mkdir()
    assert pool.latest_in(empty) is None


# --- SelfPlayWrapper tests -------------------------------------------------
# Mocked env + mocked _FrozenPolicy so we don't need torch or rlgym_sim
# to validate the action-override logic.

class _MockEnv:
    """Behaves like rlgym_sim's Gym env for a 1v1 setup: reset returns a
    list of 2 obs vectors, step takes a list of 2 actions and echoes them
    back in info so tests can assert what the wrapper actually applied."""

    def __init__(self):
        self.last_actions = None

    def reset(self):
        return [[0.0] * 4, [0.0] * 4]

    def step(self, actions):
        self.last_actions = list(actions)
        return ([[0.0] * 4, [0.0] * 4], [0.0, 0.0], False, {"actions": list(actions)})


class _MockFrozenPolicy:
    def __init__(self, action_value):
        self.action_value = action_value
        self.ckpt_name = "mock"

    def act(self, obs):
        return self.action_value


def test_wrapper_passes_through_when_pool_empty(tmp_path):
    from rlbot.env.self_play_wrapper import SelfPlayWrapper

    inner = _MockEnv()
    w = SelfPlayWrapper(inner, pool_dir=tmp_path / "pool", latest_prob=0.0, seed=1)
    w.reset()
    w.step([7, 42])
    assert inner.last_actions == [7, 42]  # no override


def test_wrapper_overrides_orange_when_opponent_loaded(tmp_path, monkeypatch):
    from rlbot.env import self_play_wrapper as spw_mod

    pool_dir = tmp_path / "pool"
    pool_dir.mkdir()
    fake = pool_dir / "1000"
    fake.mkdir()
    (fake / "PPO_POLICY.pt").write_bytes(b"\x00")

    # Force opponent sampling and stub out _FrozenPolicy
    monkeypatch.setattr(spw_mod, "_FrozenPolicy",
                        lambda ckpt, device="cpu": _MockFrozenPolicy(99))

    inner = _MockEnv()
    w = spw_mod.SelfPlayWrapper(inner, pool_dir=pool_dir, latest_prob=0.0, seed=1)
    w.reset()
    w.step([7, 42])
    # Blue (index 0) untouched, orange (index 1) replaced with opponent's choice
    assert inner.last_actions == [7, 99]


# --- Auto-resume tests -----------------------------------------------------

def test_find_latest_checkpoint_returns_highest(tmp_path):
    from rlbot.training.train import _find_latest_checkpoint

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    for ts in (100, 50000, 250000, 10000):
        d = run_dir / str(ts)
        d.mkdir()
        (d / "PPO_POLICY.pt").write_bytes(b"\x00")
    result = _find_latest_checkpoint(run_dir)
    assert result is not None
    assert result.endswith("250000")


def test_find_latest_checkpoint_ignores_dirs_without_policy(tmp_path):
    from rlbot.training.train import _find_latest_checkpoint

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "100").mkdir()
    (run_dir / "100" / "PPO_POLICY.pt").write_bytes(b"\x00")
    (run_dir / "200").mkdir()   # no PPO_POLICY.pt — should be ignored
    result = _find_latest_checkpoint(run_dir)
    assert result.endswith("100")


def test_find_latest_checkpoint_none_when_dir_missing(tmp_path):
    from rlbot.training.train import _find_latest_checkpoint

    assert _find_latest_checkpoint(tmp_path / "does_not_exist") is None


def test_find_latest_checkpoint_none_when_empty(tmp_path):
    from rlbot.training.train import _find_latest_checkpoint

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    assert _find_latest_checkpoint(run_dir) is None


def test_wrapper_preserves_action_array_shape(tmp_path, monkeypatch):
    """Regression for the production crash where _FrozenPolicy returned a
    bare int but the Learner sends actions as np.array([id], shape=(1,)).
    The mismatch made LookupAction.parse_actions raise an inhomogeneous-shape
    ValueError when it tried np.asarray() on [array([7]), 99]."""
    import numpy as np
    from rlbot.env import self_play_wrapper as spw_mod

    pool_dir = tmp_path / "pool"
    pool_dir.mkdir()
    (pool_dir / "1000").mkdir()
    (pool_dir / "1000" / "PPO_POLICY.pt").write_bytes(b"\x00")

    monkeypatch.setattr(spw_mod, "_FrozenPolicy",
                        lambda ckpt, device="cpu": _MockFrozenPolicy(99))

    inner = _MockEnv()
    w = spw_mod.SelfPlayWrapper(inner, pool_dir=pool_dir, latest_prob=0.0, seed=1)
    w.reset()
    # Simulate the Learner sending shape-(1,) numpy arrays
    actions = [np.array([7], dtype=np.int64), np.array([42], dtype=np.int64)]
    w.step(actions)

    # The wrapper must preserve the np.ndarray container so the parser
    # downstream can call np.asarray(actions) without an inhomogeneous-shape
    # error. Both elements have to be arrays, not a mix of int + array.
    assert isinstance(inner.last_actions[0], np.ndarray)
    assert isinstance(inner.last_actions[1], np.ndarray)
    assert inner.last_actions[0].shape == (1,)
    assert inner.last_actions[1].shape == (1,)
    assert int(inner.last_actions[1][0]) == 99


def test_wrapper_falls_back_when_policy_load_fails(tmp_path, monkeypatch):
    from rlbot.env import self_play_wrapper as spw_mod

    pool_dir = tmp_path / "pool"
    pool_dir.mkdir()
    fake = pool_dir / "1000"
    fake.mkdir()
    (fake / "PPO_POLICY.pt").write_bytes(b"\x00")

    def _broken(*a, **kw):
        raise RuntimeError("corrupt checkpoint")

    monkeypatch.setattr(spw_mod, "_FrozenPolicy", _broken)

    inner = _MockEnv()
    w = spw_mod.SelfPlayWrapper(inner, pool_dir=pool_dir, latest_prob=0.0, seed=1)
    w.reset()
    w.step([7, 42])
    assert inner.last_actions == [7, 42]  # graceful fallback
