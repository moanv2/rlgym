"""State setters that DRILL specific mechanics: aerials and dribbles/flicks.

The reward stack has AerialTouchReward / AerialBallReward / FlickReward, but
those are near-useless if the bot is rarely airborne or carrying the ball during
training. Random spawns almost never produce a clean aerial setup or a dribble.
These setters manufacture those situations so the mechanic rewards actually fire
often enough for the policy to learn from them.

Use inside a WeightedSampleSetter alongside RandomState (broad coverage) and the
kickoff drills, e.g.:

    WeightedSampleSetter(
        state_setters=[RandomState(...), RandomKickoffSetter(),
                       AerialSetupState(), DribbleSetupState()],
        weights=[0.45, 0.25, 0.15, 0.15],
    )

Coordinate conventions (see rl_constants.py): +Y is toward orange's net, so blue
attacks +Y and orange attacks -Y. Boost on a StateWrapper car is 0..100.
"""
from __future__ import annotations

import numpy as np
from rlgym_sim.utils.state_setters import StateSetter
from rlgym_sim.utils.state_setters.wrappers import StateWrapper

from rlbot.utils.rl_constants import (
    BACK_WALL_Y,
    OCTANE_HEIGHT_AT_REST,
    SIDE_WALL_X,
)


def _clamp(v: float, lo: float, hi: float) -> float:
    return float(np.clip(v, lo, hi))


class AerialSetupState(StateSetter):
    """Ball floating high (900–1500 uu) near midfield with mild drift; both cars
    on the ground below it with plenty of boost and facing it. Whoever commits
    gets aerial reps — this is the single biggest unlock for AerialTouchReward.
    """

    def reset(self, state: StateWrapper) -> None:
        bx = _clamp(np.random.uniform(-2000.0, 2000.0), -SIDE_WALL_X + 300, SIDE_WALL_X - 300)
        by = _clamp(np.random.uniform(-1500.0, 1500.0), -BACK_WALL_Y + 300, BACK_WALL_Y - 300)
        bz = float(np.random.uniform(900.0, 1500.0))
        state.ball.set_pos(bx, by, bz)
        state.ball.set_lin_vel(
            float(np.random.uniform(-300.0, 300.0)),
            float(np.random.uniform(-300.0, 300.0)),
            float(np.random.uniform(-200.0, 100.0)),
        )
        state.ball.set_ang_vel(0.0, 0.0, 0.0)

        for car in state.cars:
            # Blue (team 0) sits on the -Y side, orange on the +Y side, so each
            # approaches the high ball from its own half.
            side = -1.0 if car.team_num == 0 else 1.0
            cx = _clamp(bx + np.random.uniform(-600.0, 600.0), -SIDE_WALL_X + 200, SIDE_WALL_X - 200)
            cy = _clamp(by + side * np.random.uniform(800.0, 1600.0), -BACK_WALL_Y + 200, BACK_WALL_Y - 200)
            car.set_pos(cx, cy, OCTANE_HEIGHT_AT_REST)
            yaw = float(np.arctan2(by - cy, bx - cx))  # face the ball
            car.set_rot(0.0, yaw, 0.0)
            car.set_lin_vel(0.0, 0.0, 0.0)
            car.set_ang_vel(0.0, 0.0, 0.0)
            car.boost = float(np.random.uniform(60.0, 100.0))


class DribbleSetupState(StateSetter):
    """One randomly chosen car carries the ball on its roof near midfield, rolling
    toward the enemy net with boost; the opponent is set back near its own goal.
    Drives dribble control + FlickReward reps.
    """

    def reset(self, state: StateWrapper) -> None:
        cars = list(state.cars)
        if not cars:
            return
        carrier_idx = int(np.random.randint(0, len(cars)))

        for i, car in enumerate(cars):
            # Direction toward the enemy net for THIS car.
            attack_dir = 1.0 if car.team_num == 0 else -1.0

            if i == carrier_idx:
                cx = _clamp(np.random.uniform(-1500.0, 1500.0), -SIDE_WALL_X + 300, SIDE_WALL_X - 300)
                cy = _clamp(np.random.uniform(-1500.0, 1500.0), -BACK_WALL_Y + 300, BACK_WALL_Y - 300)
                car.set_pos(cx, cy, OCTANE_HEIGHT_AT_REST)
                yaw = (np.pi / 2.0) if attack_dir > 0 else (-np.pi / 2.0)  # face the enemy net
                car.set_rot(0.0, float(yaw), 0.0)
                fwd_speed = float(np.random.uniform(500.0, 1100.0))
                car.set_lin_vel(0.0, attack_dir * fwd_speed, 0.0)
                car.set_ang_vel(0.0, 0.0, 0.0)
                car.boost = float(np.random.uniform(50.0, 100.0))

                # Ball balanced just above the roof, matching the car's velocity
                # so it sits in a carry rather than rolling off immediately.
                state.ball.set_pos(cx, cy, OCTANE_HEIGHT_AT_REST + 150.0)
                state.ball.set_lin_vel(0.0, attack_dir * fwd_speed, 0.0)
                state.ball.set_ang_vel(0.0, 0.0, 0.0)
            else:
                # Defender: set back toward its OWN net (opposite of its attack dir).
                own_goal_dir = -attack_dir
                ox = _clamp(np.random.uniform(-1000.0, 1000.0), -SIDE_WALL_X + 200, SIDE_WALL_X - 200)
                oy = _clamp(own_goal_dir * np.random.uniform(2500.0, 3500.0), -BACK_WALL_Y + 200, BACK_WALL_Y - 200)
                car.set_pos(ox, oy, OCTANE_HEIGHT_AT_REST)
                yaw = (np.pi / 2.0) if attack_dir > 0 else (-np.pi / 2.0)  # face up-field toward the play
                car.set_rot(0.0, float(yaw), 0.0)
                car.set_lin_vel(0.0, 0.0, 0.0)
                car.set_ang_vel(0.0, 0.0, 0.0)
                car.boost = float(np.random.uniform(30.0, 80.0))
