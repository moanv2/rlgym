"""Smoke tests for the generic Registry helper."""

from __future__ import annotations

import pytest

from rlbot.utils.registry import Registry


def test_register_and_get():
    reg: Registry = Registry("things")

    @reg.register("foo")
    class Foo:
        pass

    assert "foo" in reg
    assert reg.get("foo") is Foo
    assert reg.keys() == ["foo"]


def test_duplicate_raises():
    reg: Registry = Registry("things")
    reg.register("a")(lambda: None)
    with pytest.raises(ValueError, match="already registered"):
        reg.register("a")(lambda: None)


def test_unknown_key_raises():
    reg: Registry = Registry("things")
    with pytest.raises(KeyError, match="not registered"):
        reg.get("nope")
