"""Tests for the advanced reward components (rlbot.rewards.advanced)."""
from __future__ import annotations

import numpy as np
import pytest

from rlgym_sim.utils.common_values import (
    BLUE_TEAM,
    ORANGE_TEAM,
    CEILING_Z,
    SUPERSONIC_THRESHOLD,
    BLUE_GOAL_BACK,
    ORANGE_GOAL_BACK,
)
from rlbot.rewards.advanced import (
    AerialTouchReward,
    SupersonicReward,
    BallAwayFromOwnGoalReward,
)


class _Phys:
    def __init__(self, position=(0, 0, 0), linear_velocity=(0, 0, 0)):
        self.position = np.array(position, dtype=np.float32)
        self.linear_velocity = np.array(linear_velocity, dtype=np.float32)


class _Player:
    def __init__(self, ball_touched=False, on_ground=True, team_num=BLUE_TEAM, car_vel=(0, 0, 0)):
        self.ball_touched = ball_touched
        self.on_ground = on_ground
        self.team_num = team_num
        self.car_data = _Phys(linear_velocity=car_vel)


class _State:
    def __init__(self, ball_pos=(0, 0, 0)):
        self.ball = _Phys(position=ball_pos)


# --- AerialTouchReward ---
def test_aerial_touch_airborne_high_ball_is_scaled():
    r = AerialTouchReward(height_scale=1.5)
    p = _Player(ball_touched=True, on_ground=False)
    s = _State(ball_pos=(0, 0, CEILING_Z))  # ball at ceiling -> height_frac ~1
    assert r.get_reward(p, s, None) == pytest.approx(1.5, abs=1e-2)


def test_aerial_touch_on_ground_is_zero():
    r = AerialTouchReward()
    p = _Player(ball_touched=True, on_ground=True)  # touched but grounded
    assert r.get_reward(p, _State(ball_pos=(0, 0, CEILING_Z)), None) == 0.0


def test_aerial_touch_no_touch_is_zero():
    r = AerialTouchReward()
    p = _Player(ball_touched=False, on_ground=False)  # airborne but no touch
    assert r.get_reward(p, _State(ball_pos=(0, 0, CEILING_Z)), None) == 0.0


def test_aerial_touch_scales_with_height():
    r = AerialTouchReward(height_scale=1.0)
    p = _Player(ball_touched=True, on_ground=False)
    low = r.get_reward(p, _State(ball_pos=(0, 0, 200)), None)
    high = r.get_reward(p, _State(ball_pos=(0, 0, 1800)), None)
    assert high > low > 0


# --- SupersonicReward ---
def test_supersonic_fast_is_one():
    r = SupersonicReward()
    p = _Player(car_vel=(SUPERSONIC_THRESHOLD + 50, 0, 0))
    assert r.get_reward(p, _State(), None) == 1.0


def test_supersonic_slow_is_zero():
    r = SupersonicReward()
    p = _Player(car_vel=(1000, 0, 0))
    assert r.get_reward(p, _State(), None) == 0.0


# --- BallAwayFromOwnGoalReward ---
def test_ball_at_own_goal_near_zero():
    r = BallAwayFromOwnGoalReward()
    p = _Player(team_num=BLUE_TEAM)
    assert r.get_reward(p, _State(ball_pos=BLUE_GOAL_BACK), None) == pytest.approx(0.0, abs=1e-3)


def test_ball_at_enemy_goal_near_one():
    r = BallAwayFromOwnGoalReward()
    p = _Player(team_num=BLUE_TEAM)
    assert r.get_reward(p, _State(ball_pos=ORANGE_GOAL_BACK), None) == pytest.approx(1.0, abs=1e-2)


def test_ball_away_from_own_goal_is_team_aware():
    r = BallAwayFromOwnGoalReward()
    s = _State(ball_pos=ORANGE_GOAL_BACK)  # ball at orange net: great for blue, bad for orange
    assert r.get_reward(_Player(team_num=BLUE_TEAM), s, None) > 0.9
    assert r.get_reward(_Player(team_num=ORANGE_TEAM), s, None) < 0.1


# --- builder integration ---
def test_builder_supports_all_advanced_names():
    from rlbot.rewards.builder import _build_one

    for name in ["aerial_touch", "supersonic", "ball_away_from_own_goal", "save_boost", "align_ball_goal"]:
        fn = _build_one({"name": name})
        assert fn is not None, f"{name} failed to build"
