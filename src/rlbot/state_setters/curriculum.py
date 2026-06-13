"""Curriculum state setters: scenario-specific episode starts.

These complement RandomState (which spawns everything uniformly) with
scenario-shaped starts that teach specific skills. Mix them via
WeightedSampleSetter (already wired into the builder) to get a curriculum.

Constants are pulled from rlgym_sim.utils.common_values where they exist
(BALL_RADIUS, BACK_WALL_Y, SIDE_WALL_X, etc.) so we stay in sync with the
physics engine. Kickoff spawn positions + yaws are from the RLBot wiki's
authoritative spawn-locations table.
"""
from __future__ import annotations

import math
import random

import numpy as np
from rlgym_sim.utils.common_values import (
    BACK_WALL_Y,
    BALL_RADIUS,
    BLUE_TEAM,
    ORANGE_TEAM,
)
from rlgym_sim.utils.state_setters import StateSetter, StateWrapper

# Car ride height when wheels are on the ground. Octane is 17.01, Dominus
# 17.05, Batmobile/Plank 18.65 — 17.0 is close enough for any body the
# default agent will spawn as.
CAR_GROUND_Z = 17.0

# Standard 1v1 kickoff positions from the RLBot wiki spawn-locations table.
# Each entry is (x, y, yaw) for the blue team. Orange mirrors by negating
# x, y and yaw. The yaw makes each car face the ball at field center.
#   Right corner: faces upper-right (toward ball) at 45° = 0.25π
#   Left corner:  faces upper-left  (toward ball) at 135° = 0.75π
#   Back rows:    face straight forward (toward orange goal) at 90° = 0.5π
_BLUE_KICKOFFS = [
    (-2048.0, -2560.0, 0.25 * math.pi),   # right corner
    (2048.0, -2560.0, 0.75 * math.pi),    # left corner
    (-256.0, -3840.0, 0.5 * math.pi),     # back right
    (256.0, -3840.0, 0.5 * math.pi),      # back left
    (0.0, -4608.0, 0.5 * math.pi),        # far back center
]


def _yaw_toward(from_xy, to_xy) -> float:
    """Return the yaw (rotation around Z) so a car at `from_xy` faces `to_xy`."""
    dx = to_xy[0] - from_xy[0]
    dy = to_xy[1] - from_xy[1]
    return math.atan2(dy, dx)


class KickoffState(StateSetter):
    """Random 1v1 kickoff. Ball at center, cars at one of 5 standard positions
    with the canonical yaws (corner kickoffs face diagonally toward the ball,
    not straight forward). Teaches the bot to win kickoffs — the most common
    situation it'll see in any real match."""

    def reset(self, state_wrapper: StateWrapper) -> None:
        # Pick one of the 5 standard kickoff positions; orange mirrors blue
        # across the field center (negate x, y, and yaw + π).
        blue_x, blue_y, blue_yaw = random.choice(_BLUE_KICKOFFS)
        orange_x, orange_y, orange_yaw = -blue_x, -blue_y, blue_yaw - math.pi

        # Ball: dead center, sitting on the ground, zero velocity
        state_wrapper.ball.set_pos(0.0, 0.0, BALL_RADIUS)
        state_wrapper.ball.set_lin_vel(0.0, 0.0, 0.0)
        state_wrapper.ball.set_ang_vel(0.0, 0.0, 0.0)

        for car in state_wrapper.cars:
            if car.team_num == BLUE_TEAM:
                x, y, yaw = blue_x, blue_y, blue_yaw
            else:
                x, y, yaw = orange_x, orange_y, orange_yaw
            car.set_pos(x, y, CAR_GROUND_Z)
            car.set_rot(0.0, yaw, 0.0)
            car.set_lin_vel(0.0, 0.0, 0.0)
            car.set_ang_vel(0.0, 0.0, 0.0)
            car.boost = 0.33   # standard kickoff boost (33/100)


class DefensiveState(StateSetter):
    """Ball + opponent attacking our goal; we spawn between ball and our net.
    Teaches positional defense and shot-blocking. Half the time the roles
    flip (blue defends / orange attacks) so the bot learns both sides."""

    def reset(self, state_wrapper: StateWrapper) -> None:
        # Pick the defending team
        blue_defends = random.random() < 0.5
        defending_team = BLUE_TEAM if blue_defends else ORANGE_TEAM
        our_goal_y = -BACK_WALL_Y if blue_defends else BACK_WALL_Y

        # Ball: at ~halfway between center and our goal, moving toward us
        ball_y = our_goal_y * 0.4 + random.uniform(-500.0, 500.0)
        ball_x = random.uniform(-2500.0, 2500.0)
        state_wrapper.ball.set_pos(ball_x, ball_y, BALL_RADIUS)
        # Velocity toward our goal at moderate speed
        speed = random.uniform(800.0, 1800.0)
        direction = np.array([0.0, our_goal_y - ball_y, 0.0])
        direction /= max(np.linalg.norm(direction), 1e-6)
        state_wrapper.ball.set_lin_vel(*(direction * speed))
        state_wrapper.ball.set_ang_vel(0.0, 0.0, 0.0)

        for car in state_wrapper.cars:
            if car.team_num == defending_team:
                # Defender: between ball and goal, slightly back from ball
                x = ball_x + random.uniform(-700.0, 700.0)
                y = (ball_y + our_goal_y) * 0.5 + random.uniform(-400.0, 400.0)
                yaw = _yaw_toward((x, y), (ball_x, ball_y))
                car.boost = random.uniform(0.3, 1.0)
            else:
                # Attacker: behind the ball, chasing
                x = ball_x + random.uniform(-700.0, 700.0)
                y = ball_y - direction[1] * 800.0 + random.uniform(-300.0, 300.0)
                yaw = _yaw_toward((x, y), (our_goal_y, 0.0))
                car.boost = random.uniform(0.4, 1.0)
            car.set_pos(x, y, CAR_GROUND_Z)
            car.set_rot(0.0, yaw, 0.0)
            car.set_lin_vel(0.0, 0.0, 0.0)
            car.set_ang_vel(0.0, 0.0, 0.0)


class AerialState(StateSetter):
    """Ball thrown high in the air, cars on the ground. Teaches the bot to
    jump and fly — without this, RandomState rarely produces meaningful
    aerial training data because the bot has no reason to leave the ground."""

    def reset(self, state_wrapper: StateWrapper) -> None:
        # Ball high in the air with some horizontal drift
        ball_x = random.uniform(-3000.0, 3000.0)
        ball_y = random.uniform(-2000.0, 2000.0)
        ball_z = random.uniform(1200.0, 1800.0)
        state_wrapper.ball.set_pos(ball_x, ball_y, ball_z)
        state_wrapper.ball.set_lin_vel(
            random.uniform(-500.0, 500.0),
            random.uniform(-500.0, 500.0),
            random.uniform(-200.0, 200.0),
        )
        state_wrapper.ball.set_ang_vel(0.0, 0.0, 0.0)

        for car in state_wrapper.cars:
            # Spawn both cars below the ball with random offsets — whoever
            # learns to fly first gets the touch.
            x = ball_x + random.uniform(-1500.0, 1500.0)
            y = ball_y + random.uniform(-1500.0, 1500.0)
            yaw = _yaw_toward((x, y), (ball_x, ball_y))
            car.set_pos(x, y, CAR_GROUND_Z)
            car.set_rot(0.0, yaw, 0.0)
            car.set_lin_vel(0.0, 0.0, 0.0)
            car.set_ang_vel(0.0, 0.0, 0.0)
            car.boost = random.uniform(0.5, 1.0)   # need boost to aerial
