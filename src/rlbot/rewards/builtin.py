"""Registers built-in rlgym_sim rewards under stable string names.

Adding a new reward? Either:
  1. Register an existing rlgym_sim/rlgym-tools reward here, or
  2. Write a new RewardFunction subclass and decorate it with @REWARDS.register("my_name")
"""
from __future__ import annotations

import numpy as np

from rlbot.rewards.registry import REWARDS

# Lazy-imported to keep tests cheap; actual training imports rlgym_sim anyway.
try:
    from rlgym_sim.utils.reward_functions.common_rewards import (
        EventReward,
        FaceBallReward,
        TouchBallReward,
        VelocityBallToGoalReward,
        VelocityPlayerToBallReward,
    )
    from rlgym_sim.utils.reward_functions import RewardFunction
    from rlgym_sim.utils.gamestates import GameState, PlayerData

    REWARDS.register("velocity_player_to_ball")(VelocityPlayerToBallReward)
    REWARDS.register("velocity_ball_to_goal")(VelocityBallToGoalReward)
    REWARDS.register("face_ball")(FaceBallReward)
    REWARDS.register("touch_ball")(TouchBallReward)
    REWARDS.register("event")(EventReward)

    @REWARDS.register("supersonic")
    class SupersonicReward(RewardFunction):
        """Reward for maintaining supersonic speed (>=2200 uu/s) when approaching ball.

        A supersonic car hits the ball at much higher speed (up to 6000 uu/s ball speed
        vs ~2000 normally). Encourages fast, aggressive play.
        """
        SUPERSONIC_THRESHOLD = 2200.0  # uu/s from RLBot wiki

        def reset(self, initial_state: GameState) -> None:
            pass

        def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
            vel = player.car_data.linear_velocity
            speed = float(np.linalg.norm(vel))
            if speed >= self.SUPERSONIC_THRESHOLD:
                return 1.0
            # Partial reward proportional to speed above 1000 uu/s
            return max(0.0, (speed - 1000.0) / (self.SUPERSONIC_THRESHOLD - 1000.0))

        def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
            return self.get_reward(player, state, previous_action)

    @REWARDS.register("in_air")
    class InAirReward(RewardFunction):
        """Reward for being off the ground.

        Encourages aerial positioning. Explicitly recommended by RLGym docs
        (weight 0.002). Different from aerial_touch — does not require touching
        the ball, just being airborne.
        """

        def reset(self, initial_state: GameState) -> None:
            pass

        def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
            return 0.0 if player.on_ground else 1.0

        def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
            return self.get_reward(player, state, previous_action)

    @REWARDS.register("aerial_touch")
    class AerialTouchReward(RewardFunction):
        """Reward for touching the ball while airborne (not on ground).

        Encourages the bot to develop aerial skills instead of relying
        purely on ground play. Returns 1.0 on aerial touch, 0.0 otherwise.
        """

        def reset(self, initial_state: GameState) -> None:
            pass

        def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
            if player.ball_touched and not player.on_ground:
                return 1.0
            return 0.0

        def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
            return self.get_reward(player, state, previous_action)

except ImportError:
    # rlgym_sim not installed — registry stays empty until it is.
    pass
