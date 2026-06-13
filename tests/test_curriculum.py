"""Tests for the curriculum state setters. These build a StateWrapper
directly without instantiating an rlgym_sim env (which would need collision
meshes), so they run on any machine with rlgym_sim installed."""
from __future__ import annotations

import math
import random

import pytest
from rlgym_sim.utils.common_values import BLUE_TEAM, ORANGE_TEAM
from rlgym_sim.utils.state_setters import StateWrapper

from rlbot.state_setters.builder import build_state_setter
from rlbot.state_setters.curriculum import AerialState, DefensiveState, KickoffState


def _fresh_wrapper() -> StateWrapper:
    # 1v1 layout
    return StateWrapper(blue_count=1, orange_count=1)


def test_kickoff_places_ball_at_center():
    random.seed(0)
    w = _fresh_wrapper()
    KickoffState().reset(w)
    assert w.ball.position[0] == 0.0
    assert w.ball.position[1] == 0.0
    # Ball velocity zero
    assert tuple(w.ball.linear_velocity) == (0.0, 0.0, 0.0)


def test_kickoff_places_cars_symmetrically():
    random.seed(0)
    w = _fresh_wrapper()
    KickoffState().reset(w)
    blue = next(c for c in w.cars if c.team_num == BLUE_TEAM)
    orange = next(c for c in w.cars if c.team_num == ORANGE_TEAM)
    # Cars on ground
    assert blue.position[2] == pytest.approx(17.0)
    assert orange.position[2] == pytest.approx(17.0)
    # Mirrored across the field center
    assert blue.position[0] == pytest.approx(-orange.position[0])
    assert blue.position[1] == pytest.approx(-orange.position[1])
    # Blue car in its own half (negative Y), orange in its (positive Y)
    assert blue.position[1] < 0
    assert orange.position[1] > 0


def test_kickoff_corner_yaws_face_ball_diagonally():
    """Right-corner blue (x=-2048, y=-2560) faces 0.25π (upper-right toward
    ball at origin), not straight forward — this is the wiki-canonical value
    and was the bug we fixed."""
    import math

    # Force the right-corner kickoff
    while True:
        random.seed(random.randint(0, 100000))
        w = _fresh_wrapper()
        KickoffState().reset(w)
        blue = next(c for c in w.cars if c.team_num == BLUE_TEAM)
        if math.isclose(blue.position[0], -2048.0) and math.isclose(blue.position[1], -2560.0):
            break
    # rotation is (pitch, yaw, roll); we set yaw at index 1
    assert blue.rotation[1] == pytest.approx(0.25 * math.pi)


def test_kickoff_kickoff_boost_amount():
    random.seed(0)
    w = _fresh_wrapper()
    KickoffState().reset(w)
    for car in w.cars:
        assert car.boost == pytest.approx(0.33)


def test_defensive_ball_moves_toward_a_goal():
    # Run a few seeds; defensive scenario should always have ball velocity
    # pointing meaningfully toward one of the goals.
    for seed in range(5):
        random.seed(seed)
        w = _fresh_wrapper()
        DefensiveState().reset(w)
        vy = w.ball.linear_velocity[1]
        # Some non-trivial Y velocity, either + or -
        assert abs(vy) > 100.0, f"seed={seed}: weak ball Y-vel {vy}"


def test_defensive_defender_between_ball_and_own_goal():
    # The defender's Y should be between the ball's Y and that team's goal Y.
    # With randomness, allow some slack — we just check the *median* over many
    # seeds is on the defending side of the ball.
    blue_defends_count = 0
    on_correct_side = 0
    for seed in range(50):
        random.seed(seed)
        w = _fresh_wrapper()
        DefensiveState().reset(w)
        ball_y = w.ball.position[1]
        ball_vy = w.ball.linear_velocity[1]
        # Defending team = whichever goal the ball is heading toward
        defending_goal_y = -5120.0 if ball_vy < 0 else 5120.0
        defending_team = BLUE_TEAM if defending_goal_y < 0 else ORANGE_TEAM
        if defending_team == BLUE_TEAM:
            blue_defends_count += 1
        defender = next(c for c in w.cars if c.team_num == defending_team)
        # Defender Y should be between ball Y and own goal Y
        if defending_goal_y < 0:
            if defender.position[1] < ball_y + 100.0:  # slack for randomness
                on_correct_side += 1
        else:
            if defender.position[1] > ball_y - 100.0:
                on_correct_side += 1
    # Both teams defend roughly equally
    assert 10 < blue_defends_count < 40
    # Most of the time the defender is goal-side of the ball
    assert on_correct_side >= 35, f"only {on_correct_side}/50 defenders correctly positioned"


def test_aerial_ball_is_high():
    for seed in range(5):
        random.seed(seed)
        w = _fresh_wrapper()
        AerialState().reset(w)
        assert w.ball.position[2] >= 1200.0


def test_aerial_cars_have_boost():
    for seed in range(5):
        random.seed(seed)
        w = _fresh_wrapper()
        AerialState().reset(w)
        for car in w.cars:
            assert car.boost >= 0.5


def test_builder_resolves_curriculum_names():
    for name in ("kickoff", "defensive", "aerial"):
        setter = build_state_setter({"name": name})
        assert setter is not None


def test_builder_weighted_sample_with_curriculum():
    cfg = {
        "name": "weighted_sample",
        "components": [
            {"name": "kickoff", "weight": 0.5},
            {"name": "aerial", "weight": 0.5},
        ],
    }
    setter = build_state_setter(cfg)
    # Sample many times — both component types should activate
    random.seed(0)
    seen_high_ball = 0
    seen_center_ball = 0
    for _ in range(50):
        w = _fresh_wrapper()
        setter.reset(w)
        if w.ball.position[2] >= 1000.0:
            seen_high_ball += 1
        elif w.ball.position[0] == 0.0 and w.ball.position[1] == 0.0:
            seen_center_ball += 1
    assert seen_high_ball > 0
    assert seen_center_ball > 0
