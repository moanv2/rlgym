"""Tests for the Kinesthetic Reward Combination (KRC) — signed geometric mean."""
from __future__ import annotations

import math

import numpy as np
import pytest

from rlbot.rewards.krc import KRCReward


class _ConstReward:
    """Mock sub-reward that always returns a fixed value."""
    def __init__(self, value):
        self.value = value
        self.reset_called = False
    def reset(self, initial_state):
        self.reset_called = True
    def pre_step(self, state):
        pass
    def get_reward(self, player, state, previous_action):
        return self.value
    def get_final_reward(self, player, state, previous_action):
        return self.value


def _krc(*values):
    return KRCReward([_ConstReward(v) for v in values])


def test_all_positive_is_geometric_mean():
    # geomean(2, 8) = sqrt(16) = 4
    r = _krc(2.0, 8.0)
    assert r.get_reward(None, None, None) == pytest.approx(4.0)


def test_three_components_geometric_mean():
    # geomean(1, 2, 4) = cbrt(8) = 2
    r = _krc(1.0, 2.0, 4.0)
    assert r.get_reward(None, None, None) == pytest.approx(2.0)


def test_any_negative_flips_sign():
    # one negative -> magnitude geomean(2,8)=4 but sign negative
    r = _krc(2.0, -8.0)
    assert r.get_reward(None, None, None) == pytest.approx(-4.0)


def test_all_negative_is_negative():
    r = _krc(-2.0, -8.0)
    assert r.get_reward(None, None, None) == pytest.approx(-4.0)


def test_zero_component_zeros_reward():
    # geometric mean with a zero -> 0 (no compound skill this step)
    r = _krc(5.0, 0.0)
    assert r.get_reward(None, None, None) == 0.0


def test_magnitude_robust_small_drags_down():
    # a tiny component drags the product down (vs a weighted sum which wouldn't)
    krc_val = _krc(0.01, 100.0).get_reward(None, None, None)  # geomean = sqrt(1) = 1
    weighted_sum = 0.01 + 100.0                                # = 100.01
    assert krc_val == pytest.approx(1.0)
    assert krc_val < weighted_sum  # KRC punishes neglecting the small component


def test_reset_delegates_to_subrewards():
    subs = [_ConstReward(1.0), _ConstReward(2.0)]
    r = KRCReward(subs)
    r.reset(None)
    assert all(s.reset_called for s in subs)


def test_empty_group_raises():
    with pytest.raises(ValueError):
        KRCReward([])


def test_builder_supports_krc_group():
    from rlbot.rewards.builder import _build_one
    spec = {
        "name": "krc",
        "group": [
            {"name": "velocity_player_to_ball"},
            {"name": "velocity_ball_to_goal"},
        ],
    }
    fn = _build_one(spec)
    assert isinstance(fn, KRCReward)
    assert len(fn.reward_functions) == 2
