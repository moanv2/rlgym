"""Tests for the cross-platform keep-awake context manager."""
from __future__ import annotations

from rlbot.utils.power import keep_awake


def test_keep_awake_enters_and_exits_cleanly():
    # On the host platform this actually sets and clears the sleep-inhibit state;
    # it must not raise and must run the body.
    ran = False
    with keep_awake("pytest"):
        ran = True
    assert ran


def test_keep_awake_is_reentrant():
    # Nesting / repeated use shouldn't blow up (state is idempotent per platform).
    with keep_awake("pytest-outer"), keep_awake("pytest-inner"):
        pass
    with keep_awake("pytest-again"):
        pass
