"""Unit tests for the custom curriculum rewards.

These need ``rlgym_sim`` importable (for the RewardFunction base + game-state
classes) but NOT the collision meshes / a live env, so they run under `test-fast`.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("rlgym_sim")

from rlgym_sim.utils.common_values import (
    BACK_WALL_Y,
    BALL_MAX_SPEED,
    CAR_MAX_SPEED,
    CEILING_Z,
)
from rlgym_sim.utils.gamestates import GameState, PhysicsObject, PlayerData

from rlbot.rewards.custom import (
    AirTouchReward,
    DoubleJumpReward,
    InAirReward,
    ShotTowardGoalReward,
    SpeedTowardBallReward,
    StrongTouchReward,
)
from rlbot.rewards.custom_rl import (
    BackboardDefenseReward,
    BallAwayFromOwnGoalReward,
    BigBoostProximityReward,
    BoostReserveReward,
    RecoveryReward,
)
from rlbot.rewards.registry import REWARDS

# Parsed controller vectors: [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]
JUMP = np.array([0, 0, 0, 0, 0, 1, 0, 0], dtype=np.float32)
NO_JUMP = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)

# Quaternions (w, x, y, z) for car orientation in fixtures (see PhysicsObject.up/forward):
UPRIGHT = (1.0, 0.0, 0.0, 0.0)        # up=[0,0,1],  forward=[1,0,0]
UPSIDE_DOWN = (0.0, 1.0, 0.0, 0.0)    # roll pi → up=[0,0,-1], forward=[1,0,0]


def _player(*, car_id=1, team=0, on_ground=True, ball_touched=False, has_flip=False,
            pos=(0.0, 0.0, 17.0), vel=(0.0, 0.0, 0.0), boost=0.0, quat=None) -> PlayerData:
    p = PlayerData()
    p.car_id = car_id
    p.team_num = team
    p.on_ground = on_ground
    p.ball_touched = ball_touched
    p.has_flip = has_flip
    p.boost_amount = float(boost)
    kwargs = dict(
        position=np.array(pos, dtype=np.float32),
        linear_velocity=np.array(vel, dtype=np.float32),
    )
    if quat is not None:
        kwargs["quaternion"] = np.array(quat, dtype=np.float64)
    p.car_data = PhysicsObject(**kwargs)
    return p


def _state(players, *, ball_pos=(0.0, 0.0, 93.0), ball_vel=(0.0, 0.0, 0.0)) -> GameState:
    s = GameState()
    s.ball = PhysicsObject(
        position=np.array(ball_pos, dtype=np.float32),
        linear_velocity=np.array(ball_vel, dtype=np.float32),
    )
    s.players = players
    return s


# --------------------------------------------------------------------------- #
# Registry wiring
# --------------------------------------------------------------------------- #
def test_curriculum_rewards_are_registered():
    import rlbot.rewards  # noqa: F401  (triggers builtin + custom registration)

    for name in (
        "speed_toward_ball",
        "in_air",
        "double_jump",
        "strong_touch",
        "air_touch",
        "shot_toward_goal",
        "save_boost",
        "align_ball_goal",
        # adopted from Diego's branch
        "big_boost_proximity",
        "boost_reserve",
        "ball_away_from_own_goal",
        "recovery",
        "backboard_defense",
    ):
        assert name in REWARDS, f"{name!r} should be registered"


# --------------------------------------------------------------------------- #
# SpeedTowardBallReward
# --------------------------------------------------------------------------- #
def test_speed_toward_ball_positive_when_driving_at_ball():
    r = SpeedTowardBallReward()
    car = _player(pos=(0, 0, 17), vel=(CAR_MAX_SPEED / 2, 0, 0))
    state = _state([car], ball_pos=(1000, 0, 17))
    r.reset(state)
    assert r.get_reward(car, state, None) == pytest.approx(0.5, abs=0.01)


def test_speed_toward_ball_zero_when_driving_away():
    r = SpeedTowardBallReward()
    car = _player(pos=(0, 0, 17), vel=(-CAR_MAX_SPEED, 0, 0))  # away from ball at +x
    state = _state([car], ball_pos=(1000, 0, 17))
    r.reset(state)
    assert r.get_reward(car, state, None) == 0.0  # never punishes moving away


def test_speed_toward_ball_clamped_at_one():
    r = SpeedTowardBallReward()
    car = _player(pos=(0, 0, 17), vel=(3 * CAR_MAX_SPEED, 0, 0))  # supersonic-ish
    state = _state([car], ball_pos=(1000, 0, 17))
    r.reset(state)
    assert r.get_reward(car, state, None) == 1.0


# --------------------------------------------------------------------------- #
# InAirReward
# --------------------------------------------------------------------------- #
def test_in_air_reward():
    r = InAirReward()
    grounded = _player(on_ground=True)
    airborne = _player(on_ground=False)
    state = _state([grounded])
    r.reset(state)
    assert r.get_reward(grounded, state, None) == 0.0
    assert r.get_reward(airborne, state, None) == 1.0


# --------------------------------------------------------------------------- #
# DoubleJumpReward
# --------------------------------------------------------------------------- #
def test_double_jump_rewards_spending_the_air_jump():
    r = DoubleJumpReward()
    # Reset airborne with the flip token still available...
    airborne_with_flip = _player(on_ground=False, has_flip=True)
    r.reset(_state([airborne_with_flip]))
    # ...then the token is consumed in the air WITH a jump press → the air jump.
    used_jump = _player(on_ground=False, has_flip=False)
    assert r.get_reward(used_jump, _state([used_jump]), JUMP) == 1.0


def test_double_jump_zero_when_flip_window_expires_without_jumping():
    r = DoubleJumpReward()
    airborne_with_flip = _player(on_ground=False, has_flip=True)
    r.reset(_state([airborne_with_flip]))
    # Same has_flip True→False transition, but no jump pressed → flip window just timed
    # out. Must NOT reward.
    expired = _player(on_ground=False, has_flip=False)
    assert r.get_reward(expired, _state([expired]), NO_JUMP) == 0.0


def test_double_jump_zero_on_landing():
    r = DoubleJumpReward()
    airborne_with_flip = _player(on_ground=False, has_flip=True)
    r.reset(_state([airborne_with_flip]))
    # has_flip resets to False on landing; that is not an air jump even with jump held.
    landed = _player(on_ground=True, has_flip=False)
    assert r.get_reward(landed, _state([landed]), JUMP) == 0.0


def test_double_jump_first_ground_jump_not_rewarded():
    r = DoubleJumpReward()
    # Start grounded with no flip token (the normal pre-jump state).
    grounded = _player(on_ground=True, has_flip=False)
    r.reset(_state([grounded]))
    # First jump off the ground GAINS the flip token (False→True) — not the air jump.
    took_off = _player(on_ground=False, has_flip=True)
    assert r.get_reward(took_off, _state([took_off]), JUMP) == 0.0


def test_double_jump_fires_only_once_per_air_jump():
    r = DoubleJumpReward()
    airborne_with_flip = _player(on_ground=False, has_flip=True)
    r.reset(_state([airborne_with_flip]))
    used = _player(on_ground=False, has_flip=False)
    assert r.get_reward(used, _state([used]), JUMP) == 1.0
    # Still airborne, still no flip, still mashing jump → no new transition, no reward.
    assert r.get_reward(used, _state([used]), JUMP) == 0.0


# --------------------------------------------------------------------------- #
# StrongTouchReward
# --------------------------------------------------------------------------- #
def test_strong_touch_scales_with_velocity_change():
    r = StrongTouchReward()
    car = _player(ball_touched=True)
    start = _state([car], ball_vel=(0, 0, 0))
    r.reset(start)
    # Ball jumps from rest to half BALL_MAX_SPEED on the step the car touches it.
    hit = _state([car], ball_vel=(BALL_MAX_SPEED / 2, 0, 0))
    assert r.get_reward(car, hit, None) == pytest.approx(0.5, abs=0.01)


def test_strong_touch_zero_without_touch():
    r = StrongTouchReward()
    car = _player(ball_touched=False)
    start = _state([car], ball_vel=(0, 0, 0))
    r.reset(start)
    moving = _state([car], ball_vel=(BALL_MAX_SPEED, 0, 0))
    assert r.get_reward(car, moving, None) == 0.0  # only a touch earns reward


def test_strong_touch_prev_velocity_shared_within_step():
    """Both players in a step must compare against the SAME prior-step velocity."""
    r = StrongTouchReward()
    p0 = _player(car_id=1, team=0, ball_touched=True)
    p1 = _player(car_id=2, team=1, ball_touched=True)
    start = _state([p0, p1], ball_vel=(0, 0, 0))
    r.reset(start)
    hit = _state([p0, p1], ball_vel=(BALL_MAX_SPEED / 2, 0, 0))
    r0 = r.get_reward(p0, hit, None)
    r1 = r.get_reward(p1, hit, None)
    assert r0 == pytest.approx(r1)  # prev_vel only advances after both are scored
    assert r0 == pytest.approx(0.5, abs=0.01)


# --------------------------------------------------------------------------- #
# AirTouchReward
# --------------------------------------------------------------------------- #
def test_air_touch_zero_on_ground():
    r = AirTouchReward()
    car = _player(on_ground=True, ball_touched=True)
    state = _state([car], ball_pos=(0, 0, CEILING_Z))
    r.reset(state)
    assert r.get_reward(car, state, None) == 0.0


def test_air_touch_rewards_sustained_high_aerial():
    r = AirTouchReward(max_time_in_air=1.75, tick_skip=8)
    car = _player(on_ground=False, ball_touched=False)
    high_ball = _state([car], ball_pos=(0, 0, CEILING_Z))
    r.reset(high_ball)
    # Accumulate well past max_time_in_air without touching (no reward yet).
    for _ in range(40):
        assert r.get_reward(car, high_ball, None) == 0.0
    # Now touch with a high ball + saturated airtime → full reward.
    car.ball_touched = True
    assert r.get_reward(car, high_ball, None) == pytest.approx(1.0, abs=1e-6)


def test_air_touch_gated_by_height():
    r = AirTouchReward(max_time_in_air=1.75, tick_skip=8)
    car = _player(on_ground=False, ball_touched=False)
    low_ball = _state([car], ball_pos=(0, 0, CEILING_Z / 4))  # height_frac = 0.25
    r.reset(low_ball)
    for _ in range(40):  # saturate airtime
        r.get_reward(car, low_ball, None)
    car.ball_touched = True
    # min(air_frac=1.0, height_frac=0.25) == 0.25
    assert r.get_reward(car, low_ball, None) == pytest.approx(0.25, abs=0.01)


# --------------------------------------------------------------------------- #
# ShotTowardGoalReward
# --------------------------------------------------------------------------- #
def test_shot_toward_goal_rewards_goalward_strike():
    """Blue attacks +y; a touch that sends the ball at the net earns ~goalward Δspeed."""
    r = ShotTowardGoalReward()
    car = _player(team=0, ball_touched=True)
    start = _state([car], ball_pos=(0, 0, 93), ball_vel=(0, 0, 0))
    r.reset(start)
    # From midfield (dist to goal > BACK_WALL_Y, so dist_frac == 1): a +y strike at
    # half BALL_MAX_SPEED scores ~0.5 (minus a hair from the elevated goal-back).
    hit = _state([car], ball_pos=(0, 0, 93), ball_vel=(0, BALL_MAX_SPEED / 2, 0))
    assert r.get_reward(car, hit, None) == pytest.approx(0.498, abs=0.02)


def test_shot_toward_goal_zero_when_struck_away_from_goal():
    r = ShotTowardGoalReward()
    car = _player(team=0, ball_touched=True)  # blue attacks +y
    start = _state([car], ball_pos=(0, 0, 93), ball_vel=(0, 0, 0))
    r.reset(start)
    # Ball driven toward the bot's OWN net (-y) → no goalward gain → no reward.
    away = _state([car], ball_pos=(0, 0, 93), ball_vel=(0, -BALL_MAX_SPEED / 2, 0))
    assert r.get_reward(car, away, None) == 0.0


def test_shot_toward_goal_zero_without_touch():
    r = ShotTowardGoalReward()
    car = _player(team=0, ball_touched=False)
    start = _state([car], ball_pos=(0, 0, 93), ball_vel=(0, 0, 0))
    r.reset(start)
    moving = _state([car], ball_pos=(0, 0, 93), ball_vel=(0, BALL_MAX_SPEED, 0))
    assert r.get_reward(car, moving, None) == 0.0  # only a touch earns reward


def test_shot_toward_goal_distance_weighted():
    """An identical strike from near the net is worth less than one from afar."""
    far = ShotTowardGoalReward()
    car = _player(team=0, ball_touched=True)
    far.reset(_state([car], ball_pos=(0, 0, 93), ball_vel=(0, 0, 0)))
    far_r = far.get_reward(
        car, _state([car], ball_pos=(0, 0, 93), ball_vel=(0, BALL_MAX_SPEED / 2, 0)), None
    )

    near = ShotTowardGoalReward()
    near.reset(_state([car], ball_pos=(0, BACK_WALL_Y - 620, 93), ball_vel=(0, 0, 0)))
    near_r = near.get_reward(
        car,
        _state([car], ball_pos=(0, BACK_WALL_Y - 620, 93), ball_vel=(0, BALL_MAX_SPEED / 2, 0)),
        None,
    )
    assert 0.0 < near_r < far_r


def test_shot_toward_goal_respects_team_direction():
    """Orange attacks -y; the same -y strike that earns nothing for blue scores here."""
    r = ShotTowardGoalReward()
    car = _player(team=1, ball_touched=True)
    r.reset(_state([car], ball_pos=(0, 0, 93), ball_vel=(0, 0, 0)))
    hit = _state([car], ball_pos=(0, 0, 93), ball_vel=(0, -BALL_MAX_SPEED / 2, 0))
    assert r.get_reward(car, hit, None) == pytest.approx(0.498, abs=0.02)


# --------------------------------------------------------------------------- #
# BigBoostProximityReward  (adopted: active routing to a big pad)
# --------------------------------------------------------------------------- #
# A big pad sits at (3584, 0); half-way (750 uu) from it gives pad_proximity 0.5.
def test_big_boost_proximity_rewards_low_boost_near_pad_ball_far():
    r = BigBoostProximityReward()
    car = _player(pos=(2834, 0, 17), boost=0.20)            # 750 uu from the (3584,0) pad
    state = _state([car], ball_pos=(2834, 5000, 93))         # ball 5000 uu away → ball_factor 1.0
    r.reset(state)
    assert r.get_reward(car, state, None) == pytest.approx(0.5, abs=0.02)


def test_big_boost_proximity_zero_when_boost_high():
    r = BigBoostProximityReward()
    car = _player(pos=(2834, 0, 17), boost=0.80)            # above the 0.40 threshold
    state = _state([car], ball_pos=(2834, 5000, 93))
    r.reset(state)
    assert r.get_reward(car, state, None) == 0.0


def test_big_boost_proximity_zero_when_ball_close():
    r = BigBoostProximityReward()
    car = _player(pos=(2834, 0, 17), boost=0.20)
    state = _state([car], ball_pos=(2834, 800, 93))         # ball only 800 uu away → in play
    r.reset(state)
    assert r.get_reward(car, state, None) == 0.0


def test_big_boost_proximity_zero_when_far_from_any_pad():
    r = BigBoostProximityReward()
    car = _player(pos=(0, 0, 17), boost=0.20)               # 3584 uu from nearest big pad
    state = _state([car], ball_pos=(0, 5000, 93))
    r.reset(state)
    assert r.get_reward(car, state, None) == 0.0


# --------------------------------------------------------------------------- #
# BoostReserveReward  (adopted; sqrt-shaped improvement, gated to non-attacking)
# --------------------------------------------------------------------------- #
_SQRT_BE = float(np.sqrt(0.34))


def test_boost_reserve_positive_when_topped_up_and_ball_far():
    r = BoostReserveReward()
    car = _player(team=0, pos=(0, 0, 17), boost=1.0)
    state = _state([car], ball_pos=(0, 5000, 93))           # far → active
    r.reset(state)
    assert r.get_reward(car, state, None) == pytest.approx(1.0 - _SQRT_BE, abs=1e-4)


def test_boost_reserve_penalizes_empty_when_active():
    r = BoostReserveReward()
    car = _player(team=0, pos=(0, 0, 17), boost=0.0)
    state = _state([car], ball_pos=(0, 5000, 93))
    r.reset(state)
    assert r.get_reward(car, state, None) == pytest.approx(-_SQRT_BE, abs=1e-4)


def test_boost_reserve_silent_when_attacking_near_ball():
    r = BoostReserveReward()
    car = _player(team=0, pos=(0, 0, 17), boost=0.10)        # near-empty, but...
    state = _state([car], ball_pos=(0, 1000, 93))            # close + attacking half (y>0) → silent
    r.reset(state)
    assert r.get_reward(car, state, None) == 0.0


def test_boost_reserve_active_in_defensive_half_even_if_near():
    r = BoostReserveReward()
    car = _player(team=0, pos=(0, 0, 17), boost=1.0)
    state = _state([car], ball_pos=(0, -1000, 93))           # close but defensive half → active
    r.reset(state)
    assert r.get_reward(car, state, None) == pytest.approx(1.0 - _SQRT_BE, abs=1e-4)


# --------------------------------------------------------------------------- #
# BallAwayFromOwnGoalReward  (adopted; threat-scaled improvement)
# --------------------------------------------------------------------------- #
def test_ball_away_positive_on_clear_scaled_by_threat():
    r = BallAwayFromOwnGoalReward()
    car = _player(team=0)                                    # blue defends -Y
    # ball at mid of own half (threat 0.5) moving +Y (away from own net) at full speed.
    state = _state([car], ball_pos=(0, -BACK_WALL_Y / 2, 93), ball_vel=(0, BALL_MAX_SPEED, 0))
    r.reset(state)
    assert r.get_reward(car, state, None) == pytest.approx(0.5, abs=1e-3)


def test_ball_away_negative_when_ball_drifts_to_own_net():
    r = BallAwayFromOwnGoalReward()
    car = _player(team=0)
    state = _state([car], ball_pos=(0, -BACK_WALL_Y / 2, 93), ball_vel=(0, -BALL_MAX_SPEED, 0))
    r.reset(state)
    assert r.get_reward(car, state, None) == pytest.approx(-0.5, abs=1e-3)


def test_ball_away_zero_in_attacking_half():
    r = BallAwayFromOwnGoalReward()
    car = _player(team=0)
    state = _state([car], ball_pos=(0, 2000, 93), ball_vel=(0, -BALL_MAX_SPEED, 0))
    r.reset(state)
    assert r.get_reward(car, state, None) == 0.0


def test_ball_away_threat_grows_toward_goal_line():
    r = BallAwayFromOwnGoalReward()
    car = _player(team=0)
    vel = (0, BALL_MAX_SPEED, 0)
    deep = _state([car], ball_pos=(0, -BACK_WALL_Y, 93), ball_vel=vel)     # threat 1.0
    mid = _state([car], ball_pos=(0, -BACK_WALL_Y / 4, 93), ball_vel=vel)  # threat 0.25
    r.reset(deep)
    assert r.get_reward(car, deep, None) == pytest.approx(1.0, abs=1e-3)
    assert r.get_reward(car, mid, None) == pytest.approx(0.25, abs=1e-3)


# --------------------------------------------------------------------------- #
# RecoveryReward  (adopted)
# --------------------------------------------------------------------------- #
def test_recovery_full_when_upright_and_aligned_to_motion():
    r = RecoveryReward()
    car = _player(on_ground=False, pos=(0, 0, 500), vel=(1000, 0, 0), quat=UPRIGHT)
    state = _state([car], ball_pos=(0, 0, 93))              # low ball → not an aerial target
    r.reset(state)
    assert r.get_reward(car, state, None) == pytest.approx(1.0, abs=1e-4)


def test_recovery_half_when_upside_down_but_moving_forward():
    r = RecoveryReward()
    car = _player(on_ground=False, pos=(0, 0, 500), vel=(1000, 0, 0), quat=UPSIDE_DOWN)
    state = _state([car], ball_pos=(0, 0, 93))
    r.reset(state)
    assert r.get_reward(car, state, None) == pytest.approx(0.5, abs=1e-4)


def test_recovery_zero_on_ground():
    r = RecoveryReward()
    car = _player(on_ground=True, pos=(0, 0, 17), vel=(1000, 0, 0), quat=UPRIGHT)
    state = _state([car], ball_pos=(0, 0, 93))
    r.reset(state)
    assert r.get_reward(car, state, None) == 0.0


def test_recovery_silent_during_aerial_play():
    r = RecoveryReward()
    car = _player(on_ground=False, pos=(0, 0, 500), vel=(1000, 0, 0), quat=UPRIGHT)
    # ball high (>=400) AND near (flat dist <=1200) → mid-aerial, leave to aerial rewards.
    state = _state([car], ball_pos=(500, 0, 600))
    r.reset(state)
    assert r.get_reward(car, state, None) == 0.0


# --------------------------------------------------------------------------- #
# BackboardDefenseReward  (adopted; continuous improvement over binary)
# --------------------------------------------------------------------------- #
def test_backboard_defense_full_on_the_line_goal_side():
    r = BackboardDefenseReward()
    car = _player(team=0, pos=(0, -3000, 17))               # goal-side, on x=0 line
    state = _state([car], ball_pos=(0, -2000, 93))          # ball in own (defensive) half
    r.reset(state)
    assert r.get_reward(car, state, None) == pytest.approx(1.0, abs=1e-3)


def test_backboard_defense_less_when_wide_of_the_line():
    r = BackboardDefenseReward()
    on_line = _player(team=0, pos=(0, -3000, 17))
    wide = _player(team=0, pos=(2000, -3000, 17))
    state = _state([on_line], ball_pos=(0, -2000, 93))
    r.reset(state)
    full = r.get_reward(on_line, state, None)
    less = r.get_reward(wide, state, None)
    assert 0.0 < less < full == pytest.approx(1.0, abs=1e-3)


def test_backboard_defense_zero_when_ball_side_of_ball():
    r = BackboardDefenseReward()
    car = _player(team=0, pos=(0, -1000, 17))               # upfield of the ball (not shadowing)
    state = _state([car], ball_pos=(0, -2000, 93))
    r.reset(state)
    assert r.get_reward(car, state, None) == 0.0


def test_backboard_defense_zero_when_ball_in_attacking_half():
    r = BackboardDefenseReward()
    car = _player(team=0, pos=(0, -3000, 17))
    state = _state([car], ball_pos=(0, 2000, 93))           # opp half → no defensive bonus
    r.reset(state)
    assert r.get_reward(car, state, None) == 0.0
