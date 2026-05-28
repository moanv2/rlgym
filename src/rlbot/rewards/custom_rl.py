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
    BALL_MAX_SPEED,
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
    """When the bot is low on boost, reward proximity to a big boost pad —
    but ONLY when the ball is far away, scaled by how far the ball is.

    Purpose: teach SMART boost economy. The original version rewarded boost
    proximity whenever boost was low, which could drag the bot off a play to
    chase a pad. This version only rewards boost-seeking when the ball is far
    enough that grabbing boost is the sensible move (not abandoning an
    immediate contest). Reward = pad_proximity * ball_distance_factor.
    """

    LOW_BOOST_THRESHOLD = 0.30    # boost is normalized 0..1 in rlgym_sim
    MAX_REWARD_DISTANCE = 1500.0  # uu — only reward within this radius of a big pad
    MIN_BALL_DISTANCE = 1500.0    # ball must be at least this far before boost-seeking is rewarded
    BALL_DISTANCE_SCALE = 3000.0  # how quickly the ball-distance factor ramps to 1.0

    def reset(self, initial_state: GameState) -> None:
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if player.boost_amount > self.LOW_BOOST_THRESHOLD:
            return 0.0

        car_pos = np.asarray(player.car_data.position)
        ball_pos = np.asarray(state.ball.position)
        ball_distance = float(np.linalg.norm(ball_pos - car_pos))

        # Only reward boost-seeking when the ball is NOT in immediate play.
        if ball_distance < self.MIN_BALL_DISTANCE:
            return 0.0

        # Ball-distance factor: 0 at MIN_BALL_DISTANCE, ramping to 1.0 as the
        # ball gets farther. The farther the ball, the safer it is to grab boost.
        ball_factor = min(1.0, (ball_distance - self.MIN_BALL_DISTANCE) / self.BALL_DISTANCE_SCALE)

        car_xy = car_pos[:2]
        min_pad_distance = float("inf")
        for pad in BIG_BOOST_PAD_POSITIONS:
            d = float(np.linalg.norm(np.asarray(pad[:2]) - car_xy))
            if d < min_pad_distance:
                min_pad_distance = d
        if min_pad_distance > self.MAX_REWARD_DISTANCE:
            return 0.0

        pad_proximity = 1.0 - (min_pad_distance / self.MAX_REWARD_DISTANCE)
        return pad_proximity * ball_factor

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return 0.0


class AerialTouchReward(RewardFunction):
    """Large bonus when the bot actually TOUCHES the ball while airborne and
    the ball is off the ground.

    Stronger and more specific than AerialBallReward — that one rewards merely
    being airborne near a high ball; this one requires real ball contact in the
    air. This is the signal that teaches actual aerial hits, not just floating
    near the ball. Give it a high weight.
    """

    BALL_HEIGHT_THRESHOLD = 300.0   # lower than AerialBallReward's 500 so lower aerials still count
    CAR_AIRBORNE_HEIGHT = OCTANE_HEIGHT_AT_REST + 50.0

    def reset(self, initial_state: GameState) -> None:
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        car_airborne = (not player.on_ground) and player.car_data.position[2] >= self.CAR_AIRBORNE_HEIGHT
        if (
            player.ball_touched
            and car_airborne
            and state.ball.position[2] >= self.BALL_HEIGHT_THRESHOLD
        ):
            return 1.0
        return 0.0

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return 0.0


class BallAwayFromOwnGoalReward(RewardFunction):
    """Reward the ball moving AWAY from the bot's own goal when the ball is in
    the bot's defensive half. Penalizes it moving toward the own net.

    Purpose: directly discourage own-goals and reward clean defensive clears.
    The bot saw few high-pressure defensive situations in balanced self-play,
    so under heavy pressure (a much stronger opponent) it mishits clears into
    its own net. This reward makes "ball moving toward my own goal" explicitly
    costly, teaching the bot to clear toward the sides/upfield instead.

    Output is in roughly [-1, 1]: positive when the ball moves away from the
    bot's own goal, negative when it moves toward it.
    """

    def reset(self, initial_state: GameState) -> None:
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        ball_y = state.ball.position[1]
        ball_vel_y = state.ball.linear_velocity[1]

        # Only active when the ball is in the bot's defensive half (where own
        # goals happen). Blue defends -Y, orange defends +Y.
        if player.team_num == 0:  # blue
            if ball_y > 0:
                return 0.0
            # Blue's own goal is at -Y. Ball moving in -Y (negative vel_y) is
            # toward own goal (bad). Reward the opposite.
            vel_toward_own_goal = -ball_vel_y
        else:  # orange
            if ball_y < 0:
                return 0.0
            # Orange's own goal is at +Y. Ball moving in +Y is toward own goal.
            vel_toward_own_goal = ball_vel_y

        # Normalize by ball max speed. Positive reward = ball moving away from
        # own goal; negative = toward own goal.
        return -vel_toward_own_goal / BALL_MAX_SPEED

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
