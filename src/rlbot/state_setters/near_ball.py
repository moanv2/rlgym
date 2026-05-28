"""``NearBallState``: spawn cars adjacent to the ball at low speed.

A targeted drill for the *I'm next to the ball and have to turn / nudge / first-touch
it* situation, which ``RandomState`` produces rarely (random spawns tend to put cars
far from the ball, often at high speed). Mixing this in via ``weighted_sample`` gives
the policy repeated reps in exactly the scenario where low-speed near-ball play is
clumsy. Pairs with ``align_ball_goal`` so the bot also learns *which side* of the
ball to approach from.
"""
from __future__ import annotations

import numpy as np
from numpy import random as rand
from rlgym_sim.utils.state_setters import StateSetter
from rlgym_sim.utils.state_setters.wrappers import StateWrapper


class NearBallState(StateSetter):
    """Each reset places the ball on the ground inside the field and every car at
    a random radius/angle around it (separated in azimuth so cars don't overlap)."""

    # Ball spawn box: well inside the field (walls at x=±4096, goals at y=±5120),
    # so a car placed at ``max_radius`` from the ball stays safely in-bounds.
    BALL_X_RANGE = 2000.0
    BALL_Y_RANGE = 3000.0
    BALL_Z = 93.0          # ball radius — resting on the ground
    CAR_Z = 17.0           # ground level for cars

    def __init__(
        self,
        min_radius: float = 500.0,
        max_radius: float = 1500.0,
        max_car_speed: float = 0.0,
        max_ball_speed: float = 0.0,
        min_boost: float = 0.3,
    ) -> None:
        """
        :param min_radius: min distance from car to ball at spawn (uu). 500 ≈ 4 car
            lengths — close enough that the bot can't just chase, but not overlapping.
        :param max_radius: max distance (uu). 1500 ≈ a short controlled approach.
        :param max_car_speed: cap on random per-axis car velocity. 0 = stationary,
            which is the actual "low-speed" drill we want most reps of.
        :param max_ball_speed: cap on random per-axis ball velocity. 0 = settled.
        :param min_boost: lower bound on randomized car boost (upper is 1.0).
        """
        super().__init__()
        if min_radius < 0 or max_radius < min_radius:
            raise ValueError("require 0 <= min_radius <= max_radius")
        if not 0.0 <= min_boost <= 1.0:
            raise ValueError("min_boost must be in [0, 1]")
        self.min_radius = float(min_radius)
        self.max_radius = float(max_radius)
        self.max_car_speed = float(max_car_speed)
        self.max_ball_speed = float(max_ball_speed)
        self.min_boost = float(min_boost)

    def reset(self, state_wrapper: StateWrapper) -> None:
        # --- ball: random field position, ground level, optional tiny velocity ---
        bx = float(rand.uniform(-self.BALL_X_RANGE, self.BALL_X_RANGE))
        by = float(rand.uniform(-self.BALL_Y_RANGE, self.BALL_Y_RANGE))
        state_wrapper.ball.set_pos(bx, by, self.BALL_Z)
        if self.max_ball_speed > 0:
            state_wrapper.ball.set_lin_vel(*self._rand_vec(self.max_ball_speed))
        else:
            state_wrapper.ball.set_lin_vel(0.0, 0.0, 0.0)
        state_wrapper.ball.set_ang_vel(0.0, 0.0, 0.0)

        # --- cars: evenly distributed azimuths around the ball (with jitter) so
        # multi-car spawns don't overlap; each at a random radius in the band ---
        n = max(len(state_wrapper.cars), 1)
        base_angle = float(rand.uniform(0.0, 2.0 * np.pi))
        for i, car in enumerate(state_wrapper.cars):
            theta = base_angle + (2.0 * np.pi * i / n) + float(rand.uniform(-0.3, 0.3))
            r = float(rand.uniform(self.min_radius, self.max_radius))
            car.set_pos(bx + r * np.cos(theta), by + r * np.sin(theta), self.CAR_Z)

            # Random yaw is the actual drill: the bot must turn to face the ball
            # from a stationary start, then control the first touch.
            car.set_rot(pitch=0.0, yaw=float(rand.uniform(-np.pi, np.pi)), roll=0.0)

            if self.max_car_speed > 0:
                car.set_lin_vel(*self._rand_vec(self.max_car_speed))
            else:
                car.set_lin_vel(0.0, 0.0, 0.0)
            car.set_ang_vel(0.0, 0.0, 0.0)

            car.boost = float(rand.uniform(self.min_boost, 1.0))

    @staticmethod
    def _rand_vec(cap: float) -> tuple[float, float, float]:
        """Uniform-in-a-cube; cheap and adequate for small-noise initial velocities."""
        x, y, z = rand.uniform(-cap, cap, size=3).tolist()
        return float(x), float(y), float(z)
