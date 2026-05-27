"""Custom reward functions that use hardcoded RL physics constants.

These rewards 'exploit' game knowledge — they encode information about the
field, car physics, and boost economy that the obs builder might not capture
sharply. Use them on top of the Nexto-style base reward stack, not as a
replacement.

Add to a CombinedReward like:

    from rlbot.rewards.custom_rl import (
        SupersonicReward,
        AerialBallReward,
        BigBoostProximityReward,
        BackboardDefenseReward,
    )
    ...
    CombinedReward(
        reward_functions=(..., SupersonicReward(), AerialBallReward(), ...),
        reward_weights=(..., 0.05, 0.5, ...),
    )
"""
from __future__ import annotations

import numpy as np
from rlgym_sim.utils import RewardFunction
from rlgym_sim.utils.gamestates import GameState, PlayerData

from rlbot.utils.rl_constants import (
    BACK_WALL_Y,
    BIG_BOOST_PAD_POSITIONS,
    CAR_SUPERSONIC_THRESHOLD,
    OCTANE_HEIGHT_AT_REST,
)


class SupersonicReward(RewardFunction):
    """Small constant bonus while the car is at supersonic speed (>= 2200 uu/s).

    Purpose: encourage the bot to use boost meaningfully and play at pace.
    Cheap signal — fires every step the car is supersonic. Weight low (0.05).
    """

    def reset(self, initial_state: GameState) -> None:
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        speed = float(np.linalg.norm(player.car_data.linear_velocity))
        return 1.0 if speed >= CAR_SUPERSONIC_THRESHOLD else 0.0

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return 0.0


class AerialBallReward(RewardFunction):
    """Bonus when the car is airborne AND the ball is high (>= 500 uu).

    Purpose: encourage aerial play. The bot is rewarded only when both
    conditions hold, so it does not earn this just by jumping near the ground.
    Combine with an aerial-scenario state setter for fastest learning.
    """

    BALL_HEIGHT_THRESHOLD = 500.0
    # Octane on the ground sits at z=17.01; anything 50+ above that is clearly airborne.
    CAR_AIRBORNE_HEIGHT = OCTANE_HEIGHT_AT_REST + 50.0

    def reset(self, initial_state: GameState) -> None:
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        ball_z = state.ball.position[2]
        car_z = player.car_data.position[2]
        # Use both height AND the engine's on_ground flag; either-alone gives false positives.
        car_airborne = (not player.on_ground) and car_z >= self.CAR_AIRBORNE_HEIGHT
        if ball_z >= self.BALL_HEIGHT_THRESHOLD and car_airborne:
            return 1.0
        return 0.0

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return 0.0


class BigBoostProximityReward(RewardFunction):
    """When the bot is low on boost, reward proximity to a big boost pad.

    Purpose: teach the bot the boost economy — big pads give 100, respawn in 10s,
    and live at predictable arena positions. Linear falloff from the closest big
    pad, only active when boost < threshold so we do not constantly drag the bot
    away from plays.
    """

    LOW_BOOST_THRESHOLD = 0.30   # boost is normalized 0..1 in rlgym_sim
    MAX_REWARD_DISTANCE = 1500.0 # uu — only reward within this radius of a big pad

    def reset(self, initial_state: GameState) -> None:
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if player.boost_amount > self.LOW_BOOST_THRESHOLD:
            return 0.0
        car_xy = np.asarray(player.car_data.position[:2])
        min_distance = float("inf")
        for pad in BIG_BOOST_PAD_POSITIONS:
            pad_xy = np.asarray(pad[:2])
            d = float(np.linalg.norm(pad_xy - car_xy))
            if d < min_distance:
                min_distance = d
        if min_distance > self.MAX_REWARD_DISTANCE:
            return 0.0
        return 1.0 - (min_distance / self.MAX_REWARD_DISTANCE)

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return 0.0


class BackboardDefenseReward(RewardFunction):
    """Reward being between the ball and the bot's own goal when the ball is
    in the bot's defensive half.

    Purpose: encode shadow-defense positioning. This is on top of AlignBallGoal
    in the Nexto stack — that one rewards goal-ball line generally, this one
    specifically activates only when the ball is in the bot's half and threatens.
    """

    def reset(self, initial_state: GameState) -> None:
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        ball_y = state.ball.position[1]
        car_y = player.car_data.position[1]

        # Own goal Y depends on team (blue=−BACK_WALL_Y, orange=+BACK_WALL_Y).
        # Active only when the ball is in the bot's defensive half AND the bot
        # is between ball and own goal.
        if player.team_num == 0:  # blue defends at y = -BACK_WALL_Y
            if ball_y > 0:
                return 0.0  # ball is in opp half, no defensive bonus
            return 1.0 if car_y < ball_y else 0.0  # bot is goal-side of ball
        else:  # orange defends at y = +BACK_WALL_Y
            if ball_y < 0:
                return 0.0
            return 1.0 if car_y > ball_y else 0.0

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return 0.0
