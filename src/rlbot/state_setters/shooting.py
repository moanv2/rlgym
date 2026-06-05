"""``ShootingState``: spawn the car behind a midfield ball so it must shoot at the net.

A targeted drill for the *line up and launch the ball at the far goal from distance*
situation. ``RandomState`` rarely produces a clean "ball ahead of me, net beyond it"
look, and ``NearBallState`` puts the car at a random azimuth around the ball (no
goal-relative structure at all). This setter instead places the ball near midfield —
deliberately far from both nets — and drops each car on the side of the ball *away*
from the goal it attacks, facing the ball. From there the only productive action is a
long strike toward the net, which is exactly the rep the bot is weak at.

Per-team attack direction (1v1): blue (team 0) attacks +y, orange (team 1) attacks -y,
so the two cars naturally end up on opposite sides of the ball. Pairs with
``shot_toward_goal`` (which rewards goalward strike speed, distance-weighted) and
``align_ball_goal`` (approach from the goal-side).
"""
from __future__ import annotations

import numpy as np
from numpy import random as rand
from rlgym_sim.utils.common_values import BLUE_TEAM
from rlgym_sim.utils.state_setters import StateSetter
from rlgym_sim.utils.state_setters.wrappers import StateWrapper


class ShootingState(StateSetter):
    """Each reset settles the ball at midfield and places every car behind it (relative
    to that car's attacking net), facing the ball, with enough boost to drive a shot."""

    BALL_Z = 93.0   # ball radius — resting on the ground
    CAR_Z = 17.0    # ground level for cars

    def __init__(
        self,
        ball_x_range: float = 2500.0,
        ball_y_range: float = 1500.0,
        min_behind: float = 1200.0,
        max_behind: float = 2600.0,
        max_lateral: float = 900.0,
        min_boost: float = 0.4,
    ) -> None:
        """
        :param ball_x_range: half-width of the ball's spawn box along x (uu). Kept inside
            the side walls (x=±4096) so the car placed beside it stays in-bounds.
        :param ball_y_range: half-depth of the ball's spawn box along y (uu). Small on
            purpose: the ball stays near midfield so every shot is a *long* one.
        :param min_behind: min car setback behind the ball toward its own net (uu).
            ~1200 ≈ a real run-up, not a tap.
        :param max_behind: max car setback (uu) — a long approach from deep.
        :param max_lateral: max sideways offset of the car from the ball (uu), so the
            shot is not always dead-straight; the bot must still steer onto the line.
        :param min_boost: lower bound on randomized car boost (upper is 1.0). Shooting
            from distance wants speed, so we don't start the car empty.
        """
        super().__init__()
        if min_behind < 0 or max_behind < min_behind:
            raise ValueError("require 0 <= min_behind <= max_behind")
        if not 0.0 <= min_boost <= 1.0:
            raise ValueError("min_boost must be in [0, 1]")
        self.ball_x_range = float(ball_x_range)
        self.ball_y_range = float(ball_y_range)
        self.min_behind = float(min_behind)
        self.max_behind = float(max_behind)
        self.max_lateral = float(max_lateral)
        self.min_boost = float(min_boost)

    def reset(self, state_wrapper: StateWrapper) -> None:
        # --- ball: midfield, on the ground, settled ---
        bx = float(rand.uniform(-self.ball_x_range, self.ball_x_range))
        by = float(rand.uniform(-self.ball_y_range, self.ball_y_range))
        state_wrapper.ball.set_pos(bx, by, self.BALL_Z)
        state_wrapper.ball.set_lin_vel(0.0, 0.0, 0.0)
        state_wrapper.ball.set_ang_vel(0.0, 0.0, 0.0)

        # --- cars: behind the ball relative to the net they attack, facing the ball ---
        for car in state_wrapper.cars:
            attack_dir = 1.0 if car.team_num == BLUE_TEAM else -1.0  # blue attacks +y
            behind = float(rand.uniform(self.min_behind, self.max_behind))
            lateral = float(rand.uniform(-self.max_lateral, self.max_lateral))
            cx = bx + lateral
            cy = by - attack_dir * behind  # set the car back toward its own goal
            car.set_pos(cx, cy, self.CAR_Z)

            # Yaw toward the ball so the very first action can be a shot, not a turn.
            yaw = float(np.arctan2(by - cy, bx - cx))
            car.set_rot(pitch=0.0, yaw=yaw, roll=0.0)
            car.set_lin_vel(0.0, 0.0, 0.0)
            car.set_ang_vel(0.0, 0.0, 0.0)
            car.boost = float(rand.uniform(self.min_boost, 1.0))
