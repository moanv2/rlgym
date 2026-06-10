"""Unit tests for the in-game wandb metrics logger.

Needs ``rlgym_ppo`` (for the MetricsLogger base) and ``rlgym_sim`` (for the
game-state classes), but no live env — like test_rewards, runs under `test-fast`.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("rlgym_ppo")
pytest.importorskip("rlgym_sim")

from rlgym_sim.utils.gamestates import GameState, PhysicsObject, PlayerData

from rlbot.training.metrics import (
    _AVG_BOOST,
    _FRAC_AIR,
    _FRAC_EMPTY,
    _GOALS,
    _SAVES,
    _SHOTS,
    BotMetricsLogger,
)


def _player(*, car_id=1, team=0, boost=0.0, on_ground=True, shots=0, saves=0) -> PlayerData:
    p = PlayerData()
    p.car_id = car_id
    p.team_num = team
    p.boost_amount = boost
    p.on_ground = on_ground
    p.match_shots = shots
    p.match_saves = saves
    p.car_data = PhysicsObject(position=np.array([0.0, 0.0, 17.0], dtype=np.float32))
    return p


def _state(players, *, ball_z=93.0, blue=0, orange=0) -> GameState:
    s = GameState()
    s.ball = PhysicsObject(position=np.array([0.0, 0.0, ball_z], dtype=np.float32))
    s.players = players
    s.blue_score = blue
    s.orange_score = orange
    return s


def _vec(logger, state):
    """Run a single collection step and return the raw metric vector."""
    return logger._collect_metrics(state)[0]


# --------------------------------------------------------------------------- #
# Per-step instantaneous metrics
# --------------------------------------------------------------------------- #
def test_avg_boost_and_empty_fraction():
    m = BotMetricsLogger(empty_threshold=0.10)
    full = _player(car_id=1, boost=0.8)
    empty = _player(car_id=2, boost=0.05)
    v = _vec(m, _state([full, empty]))
    assert v[_AVG_BOOST] == pytest.approx(0.425)
    assert v[_FRAC_EMPTY] == pytest.approx(0.5)  # one of two cars under threshold


def test_airborne_fraction():
    m = BotMetricsLogger()
    v = _vec(m, _state([_player(car_id=1, on_ground=False), _player(car_id=2, on_ground=True)]))
    assert v[_FRAC_AIR] == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# Stateful event detection (goals / shots / saves as increments)
# --------------------------------------------------------------------------- #
def test_goal_counted_once_on_score_increment():
    m = BotMetricsLogger()
    car = _player()
    # No score yet.
    assert _vec(m, _state([car], blue=0, orange=0))[_GOALS] == 0
    # Blue scores → +1 exactly on the step the score increments.
    assert _vec(m, _state([car], blue=1, orange=0))[_GOALS] == 1
    # Same score held the next step → not counted again.
    assert _vec(m, _state([car], blue=1, orange=0))[_GOALS] == 0


def test_score_reset_is_not_a_goal():
    m = BotMetricsLogger()
    car = _player()
    _vec(m, _state([car], blue=2, orange=1))
    # Episode reset: scores drop to 0 — a negative delta must NOT register as goals.
    assert _vec(m, _state([car], blue=0, orange=0))[_GOALS] == 0


def test_shots_and_saves_counted_on_stat_increment():
    m = BotMetricsLogger()
    car = _player(car_id=7, shots=0, saves=0)
    assert _vec(m, _state([car]))[_SHOTS] == 0  # first sighting: baseline, no event
    car.match_shots = 1
    car.match_saves = 1
    v = _vec(m, _state([car]))
    assert v[_SHOTS] == 1
    assert v[_SAVES] == 1
    # No further increment → no event.
    assert _vec(m, _state([car]))[_SHOTS] == 0


# --------------------------------------------------------------------------- #
# Iteration-level aggregation -> wandb report
# --------------------------------------------------------------------------- #
class _FakeWandb:
    def __init__(self):
        self.logged = None
        self.commit = None

    def log(self, data, commit=True):
        self.logged = data
        self.commit = commit


def test_report_aggregates_means_and_rates():
    m = BotMetricsLogger()
    # Two steps: boosts 0.0 and 1.0 (mean 0.5 -> 50%); one goal across two steps.
    steps = [
        [np.array([0.0, 1.0, 100.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)],
        [np.array([1.0, 0.0, 100.0, 1.0, 1.0, 2.0, 0.0], dtype=np.float64)],
    ]
    wb = _FakeWandb()
    m._report_metrics(steps, wb, cumulative_timesteps=1000)

    assert wb.commit is False  # must merge onto the learner's step
    assert wb.logged["boost/avg_held_pct"] == pytest.approx(50.0)
    assert wb.logged["play/pct_airborne"] == pytest.approx(50.0)
    # 1 goal / 2 steps * 1000 = 500 per 1k steps; 2 shots / 2 * 1000 = 1000.
    assert wb.logged["score/goals_per_1k_steps"] == pytest.approx(500.0)
    assert wb.logged["score/shots_per_1k_steps"] == pytest.approx(1000.0)


def test_report_noop_without_wandb():
    m = BotMetricsLogger()
    # Must not raise when wandb is disabled.
    m._report_metrics([[np.zeros(7)]], None, cumulative_timesteps=0)
