"""Make sure the named architectures stay consistent."""

from __future__ import annotations

import pytest

from rlbot.models.architectures import ARCHITECTURES, get_layer_sizes


def test_known_archs_present():
    assert {"tiny", "small", "medium", "large"}.issubset(ARCHITECTURES.keys())


def test_get_layer_sizes_returns_tuple():
    sizes = get_layer_sizes("small")
    assert isinstance(sizes, tuple)
    assert all(isinstance(x, int) and x > 0 for x in sizes)


def test_unknown_arch_raises():
    with pytest.raises(KeyError):
        get_layer_sizes("nope")
