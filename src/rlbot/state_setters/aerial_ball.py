"""``AerialBallState``: spawn the ball in the air with the car below it, boost loaded.

A targeted drill for the *the ball is up, go get it* situation. ``RandomState`` only
pops the ball up occasionally (and rarely with a car positioned to chase it), and the
``shooting``/``near_ball`` drills both settle the ball on the ground — so by exp_009
the bot has little reason to leave the floor and average ball height drifts down. This
setter guarantees aerial reps: a reachable airborne ball, a grounded car a short drive
away facing it, and enough boost to actually fly. Pairs with ``air_touch`` (rewards a
high, sustained aerial touch) and ``double_jump`` (the air jump that unlocks height).

Heights are kept in the *reachable* band (well under the 2044 ceiling) so the bot
learns ordinary aerials, not ceiling reads it can't yet pull off.
"""
from __future__ import annotations

import numpy as np
from numpy import random as rand
from rlgym_sim.utils.state_setters import StateSetter
from rlgym_sim.utils.state_setters.wrappers import StateWrapper


class AerialBallState(StateSetter):
    """Each reset floats the ball at a random reachable height and drops every car on
    the ground a short distance away, yawed at the ball, with a healthy boost tank."""

    CAR_Z = 17.0           # ground level for cars
    # Keep cars in-bounds regardless of where the ball + offset land (walls x=±4096,
    # back wall y=±5120). We clamp to a safe margin inside those.
    MAX_CAR_X = 3900.0
    MAX_CAR_Y = 4900.0

    def __init__(
        self,
        min_height: float = 350.0,
        max_height: float = 1400.0,
        ball_x_range: float = 2200.0,
        ball_y_range: float = 3000.0,
        min_offset: float = 600.0,
        max_offset: float = 1600.0,
        max_ball_speed: float = 350.0,
        min_boost: float = 0.5,
    ) -> None:
        """
        :param min_height: lowest the ball spawns (uu). ~350 ≈ a low pop the bot must
            still jump for.
        :param max_height: highest the ball spawns (uu). 1400 is a real aerial but well
            below the 2044 ceiling — reachable with a jump + boost.
        :param ball_x_range: half-width of the ball's spawn box along x (uu).
        :param ball_y_range: half-depth of the ball's spawn box along y (uu) — kept off
            the nets so the drill is about flying, not finishing.
        :param min_offset: min horizontal car↔ball distance (uu) — a short run-up.
        :param max_offset: max horizontal car↔ball distance (uu).
        :param max_ball_speed: cap on per-axis ball velocity (uu/s). The vertical
            component is biased downward so the ball tends to *fall* — a realistic read.
        :param min_boost: lower bound on randomized car boost (upper 1.0). Aerials need
            boost, so we never start the car dry.
        """
        super().__init__()
        if min_height < 0 or max_height < min_height:
            raise ValueError("require 0 <= min_height <= max_height")
        if min_offset < 0 or max_offset < min_offset:
            raise ValueError("require 0 <= min_offset <= max_offset")
        if not 0.0 <= min_boost <= 1.0:
            raise ValueError("min_boost must be in [0, 1]")
        self.min_height = float(min_height)
        self.max_height = float(max_height)
        self.ball_x_range = float(ball_x_range)
        self.ball_y_range = float(ball_y_range)
        self.min_offset = float(min_offset)
        self.max_offset = float(max_offset)
        self.max_ball_speed = float(max_ball_speed)
        self.min_boost = float(min_boost)

    def reset(self, state_wrapper: StateWrapper) -> None:
        # --- ball: random field position, floating at a reachable height, gently falling ---
        bx = float(rand.uniform(-self.ball_x_range, self.ball_x_range))
        by = float(rand.uniform(-self.ball_y_range, self.ball_y_range))
        bz = float(rand.uniform(self.min_height, self.max_height))
        state_wrapper.ball.set_pos(bx, by, bz)
        if self.max_ball_speed > 0:
            vx = float(rand.uniform(-self.max_ball_speed, self.max_ball_speed))
            vy = float(rand.uniform(-self.max_ball_speed, self.max_ball_speed))
            vz = float(rand.uniform(-self.max_ball_speed, 0.0))  # downward bias — a falling read
            state_wrapper.ball.set_lin_vel(vx, vy, vz)
        else:
            state_wrapper.ball.set_lin_vel(0.0, 0.0, 0.0)
        state_wrapper.ball.set_ang_vel(0.0, 0.0, 0.0)

        # --- cars: grounded a short drive from under the ball, yawed at it, boost loaded ---
        n = max(len(state_wrapper.cars), 1)
        base_angle = float(rand.uniform(0.0, 2.0 * np.pi))
        for i, car in enumerate(state_wrapper.cars):
            theta = base_angle + (2.0 * np.pi * i / n) + float(rand.uniform(-0.3, 0.3))
            r = float(rand.uniform(self.min_offset, self.max_offset))
            cx = float(np.clip(bx + r * np.cos(theta), -self.MAX_CAR_X, self.MAX_CAR_X))
            cy = float(np.clip(by + r * np.sin(theta), -self.MAX_CAR_Y, self.MAX_CAR_Y))
            car.set_pos(cx, cy, self.CAR_Z)

            # Yaw toward the ball's ground projection so the first move drives under it;
            # the bot must add its own jump + pitch to actually go up.
            yaw = float(np.arctan2(by - cy, bx - cx))
            car.set_rot(pitch=0.0, yaw=yaw, roll=0.0)
            car.set_lin_vel(0.0, 0.0, 0.0)
            car.set_ang_vel(0.0, 0.0, 0.0)
            car.boost = float(rand.uniform(self.min_boost, 1.0))
