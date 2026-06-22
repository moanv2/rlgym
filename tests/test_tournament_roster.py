"""Tests for roster discovery (checkpoint resolution, seeding metadata, config sanity).

Kept torch-free: resolve_checkpoint / read_timesteps are pure, and build_roster is
only exercised on *absent* paths (which short-circuit before the torch-backed
arch detection), so these run anywhere.
"""
from __future__ import annotations

import json

from rlbot.tournament.roster import (
    N_ACTIONS,
    OBS_ADVANCED,
    OBS_DEFAULT,
    ROSTER_CONFIG,
    build_roster,
    read_timesteps,
    resolve_checkpoint,
)


def _make_ckpt(parent, timestep: str, with_weights: bool = True):
    d = parent / timestep
    d.mkdir(parents=True)
    if with_weights:
        (d / "PPO_POLICY.pt").write_bytes(b"")  # presence is all resolve_checkpoint checks
    return d


def test_resolve_direct_checkpoint(tmp_path):
    d = _make_ckpt(tmp_path, "12345")
    assert resolve_checkpoint(str(d)) == d


def test_resolve_picks_latest_timestep(tmp_path):
    exp = tmp_path / "exp"
    _make_ckpt(exp, "1000")
    _make_ckpt(exp, "5000")
    _make_ckpt(exp, "300")
    assert resolve_checkpoint(str(exp)).name == "5000"


def test_resolve_handles_nested_layout(tmp_path):
    # papaya-style: <exp>/<exp>-<unix>/<timestep>/PPO_POLICY.pt
    run = tmp_path / "papaya" / "papaya-1780000000"
    _make_ckpt(run, "999")
    _make_ckpt(run, "111")
    assert resolve_checkpoint(str(tmp_path / "papaya")).name == "999"


def test_resolve_missing_or_empty_returns_none(tmp_path):
    assert resolve_checkpoint(str(tmp_path / "does_not_exist")) is None
    empty = tmp_path / "empty"
    empty.mkdir()
    assert resolve_checkpoint(str(empty)) is None  # exists but no PPO_POLICY.pt anywhere


def test_read_timesteps_from_bookkeeping(tmp_path):
    d = _make_ckpt(tmp_path, "42")
    (d / "BOOK_KEEPING_VARS.json").write_text(json.dumps({"cumulative_timesteps": 1349081288}))
    assert read_timesteps(d) == 1349081288


def test_read_timesteps_fallback_to_folder_name(tmp_path):
    d = _make_ckpt(tmp_path, "987654")
    assert read_timesteps(d) == 987654  # no bookkeeping file -> numeric folder name


def test_build_roster_skips_absent_owners(tmp_path):
    # All paths absent -> empty roster, no torch touched, no crash.
    cfg = [
        {"owner": "ghost1", "name": "G1", "path": str(tmp_path / "nope1")},
        {"owner": "ghost2", "name": "G2", "path": str(tmp_path / "nope2")},
    ]
    assert build_roster(cfg, verbose=False) == []


def test_roster_config_is_well_formed():
    owners = [e["owner"] for e in ROSTER_CONFIG]
    assert len(owners) == len(set(owners)), "duplicate owners in ROSTER_CONFIG"
    for entry in ROSTER_CONFIG:
        assert {"owner", "name", "path"} <= entry.keys()
    # the five expected teammates are all represented
    assert set(owners) == {"diego", "martin", "marian", "nachi", "marco"}


def test_obs_constants_sane():
    assert OBS_DEFAULT == 89
    assert OBS_ADVANCED == 107
    assert N_ACTIONS == 90
