"""Unit tests for the custom state setters used in the curriculum.

These need ``rlgym_sim`` importable for the ``StateSetter`` base + ``StateWrapper``,
but NOT collision meshes or a live env, so they run under ``make test-fast``.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("rlgym_sim")

from rlgym_sim.utils.state_setters import StateSetter
from rlgym_sim.utils.state_setters.wrappers import StateWrapper

from rlbot.state_setters.aerial_ball import AerialBallState
from rlbot.state_setters.near_ball import NearBallState
from rlbot.state_setters.weighted_random import WeightedRandomSetter


def _wrapper(blue: int = 1, orange: int = 1) -> StateWrapper:
    return StateWrapper(blue_count=blue, orange_count=orange)


def _xy_dist(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a[:2]) - np.asarray(b[:2])))


# --------------------------------------------------------------------------- #
# NearBallState
# --------------------------------------------------------------------------- #
def test_near_ball_places_cars_in_distance_band():
    s = NearBallState(min_radius=500.0, max_radius=1500.0)
    # rlgym_sim's state setters use the module-level np.random; seed for repeatability.
    np.random.seed(0)
    for _ in range(20):
        w = _wrapper(blue=1, orange=1)
        s.reset(w)
        for car in w.cars:
            d = _xy_dist(car.position, w.ball.position)
            assert 500.0 - 1e-6 <= d <= 1500.0 + 1e-6, f"car-ball xy distance {d} out of band"


def test_near_ball_grounded_and_stationary_by_default():
    s = NearBallState(max_car_speed=0.0, max_ball_speed=0.0)
    np.random.seed(1)
    w = _wrapper(blue=1, orange=1)
    s.reset(w)
    assert w.ball.position[2] == pytest.approx(NearBallState.BALL_Z)
    assert tuple(w.ball.linear_velocity) == (0.0, 0.0, 0.0)
    for car in w.cars:
        assert car.position[2] == pytest.approx(NearBallState.CAR_Z)
        assert tuple(car.linear_velocity) == (0.0, 0.0, 0.0)
        # Only yaw is randomized — pitch/roll = 0 so the car can drive normally.
        assert car.rotation[0] == 0.0  # pitch
        assert car.rotation[2] == 0.0  # roll


def test_near_ball_boost_respects_floor():
    s = NearBallState(min_boost=0.5)
    np.random.seed(2)
    for _ in range(20):
        w = _wrapper(blue=1, orange=1)
        s.reset(w)
        for car in w.cars:
            assert 0.5 <= car.boost <= 1.0


def test_near_ball_separates_multiple_cars():
    s = NearBallState(min_radius=500.0, max_radius=1500.0)
    np.random.seed(3)
    # With 2 cars spread in azimuth, they should be meaningfully apart (not stacked).
    for _ in range(20):
        w = _wrapper(blue=1, orange=1)
        s.reset(w)
        assert _xy_dist(w.cars[0].position, w.cars[1].position) > 300.0


def test_near_ball_invalid_args():
    with pytest.raises(ValueError):
        NearBallState(min_radius=-1.0)
    with pytest.raises(ValueError):
        NearBallState(min_radius=1000.0, max_radius=500.0)
    with pytest.raises(ValueError):
        NearBallState(min_boost=1.5)


# --------------------------------------------------------------------------- #
# AerialBallState
# --------------------------------------------------------------------------- #
def test_aerial_ball_floats_at_reachable_height():
    s = AerialBallState(min_height=350.0, max_height=1400.0)
    np.random.seed(0)
    for _ in range(20):
        w = _wrapper(blue=1, orange=1)
        s.reset(w)
        z = float(w.ball.position[2])
        assert 350.0 - 1e-6 <= z <= 1400.0 + 1e-6, f"ball height {z} out of band"


def test_aerial_ball_cars_grounded_with_boost_and_in_bounds():
    s = AerialBallState(min_boost=0.5)
    np.random.seed(1)
    for _ in range(20):
        w = _wrapper(blue=1, orange=1)
        s.reset(w)
        for car in w.cars:
            assert car.position[2] == pytest.approx(AerialBallState.CAR_Z)  # on the ground
            assert 0.5 <= car.boost <= 1.0
            assert abs(car.position[0]) <= AerialBallState.MAX_CAR_X + 1e-6
            assert abs(car.position[1]) <= AerialBallState.MAX_CAR_Y + 1e-6
            # Drivable spawn: only yaw set, pitch/roll flat.
            assert car.rotation[0] == 0.0
            assert car.rotation[2] == 0.0


def test_aerial_ball_horizontal_offset_band():
    s = AerialBallState(min_offset=600.0, max_offset=1600.0)
    np.random.seed(2)
    for _ in range(20):
        w = _wrapper(blue=1, orange=1)
        s.reset(w)
        for car in w.cars:
            # Clamping can only pull a car CLOSER, so the raw band is an upper bound.
            d = _xy_dist(car.position, w.ball.position)
            assert d <= 1600.0 + 1e-6, f"car-ball xy offset {d} exceeds max"


def test_aerial_ball_falling_bias():
    s = AerialBallState(max_ball_speed=350.0)
    np.random.seed(3)
    for _ in range(20):
        w = _wrapper(blue=1, orange=1)
        s.reset(w)
        assert w.ball.linear_velocity[2] <= 0.0  # vertical velocity biased downward


def test_aerial_ball_invalid_args():
    with pytest.raises(ValueError):
        AerialBallState(min_height=-1.0)
    with pytest.raises(ValueError):
        AerialBallState(min_height=1000.0, max_height=500.0)
    with pytest.raises(ValueError):
        AerialBallState(min_offset=1000.0, max_offset=500.0)
    with pytest.raises(ValueError):
        AerialBallState(min_boost=1.5)


# --------------------------------------------------------------------------- #
# WeightedRandomSetter
# --------------------------------------------------------------------------- #
class _CountingSetter(StateSetter):
    """Test double — records each reset call without touching the state."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def reset(self, state_wrapper: StateWrapper) -> None:
        self.calls += 1


def test_weighted_random_always_picks_only_nonzero_weighted():
    a, b = _CountingSetter(), _CountingSetter()
    s = WeightedRandomSetter(state_setters=[a, b], weights=[1.0, 0.0])
    np.random.seed(0)
    for _ in range(50):
        s.reset(_wrapper())
    assert (a.calls, b.calls) == (50, 0)


def test_weighted_random_respects_opposite_weights():
    a, b = _CountingSetter(), _CountingSetter()
    s = WeightedRandomSetter(state_setters=[a, b], weights=[0.0, 1.0])
    np.random.seed(0)
    for _ in range(50):
        s.reset(_wrapper())
    assert (a.calls, b.calls) == (0, 50)


def test_weighted_random_samples_proportionally():
    a, b = _CountingSetter(), _CountingSetter()
    s = WeightedRandomSetter(state_setters=[a, b], weights=[3.0, 1.0])
    np.random.seed(42)
    n = 4000
    for _ in range(n):
        s.reset(_wrapper())
    # Expect ~75 / 25; allow generous slack so the test stays stable under reseeds.
    assert 0.70 * n < a.calls < 0.80 * n
    assert a.calls + b.calls == n


def test_weighted_random_validates_inputs():
    a = _CountingSetter()
    with pytest.raises(ValueError):
        WeightedRandomSetter(state_setters=[], weights=[])
    with pytest.raises(ValueError):
        WeightedRandomSetter(state_setters=[a], weights=[1.0, 1.0])  # length mismatch
    with pytest.raises(ValueError):
        WeightedRandomSetter(state_setters=[a], weights=[-1.0])
    with pytest.raises(ValueError):
        WeightedRandomSetter(state_setters=[a, _CountingSetter()], weights=[0.0, 0.0])
