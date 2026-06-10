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

    LOW_BOOST_THRESHOLD = 0.40    # boost is normalized 0..1 in rlgym_sim; v3: 0.30→0.40 so the bot tops up proactively (pros refill before empty)
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


class DribbleToGoalReward(RewardFunction):
    """Reward keeping the ball in a dribble pose (balanced near/above the car)
    AND close to the enemy goal. Scales the possession reward by enemy-goal
    proximity so the bot learns to dribble TOWARD the net, not into the side
    walls/poles where it currently loses the ball.

    Keep the weight small (~0.15) — this is a directional nudge, not a
    dominant signal.
    """

    MAX_HORIZONTAL_DIST = 170.0   # ball within this horizontal dist of the car
    MIN_BALL_HEIGHT = 110.0       # ball above the car roof (a real dribble, not just nearby)
    MAX_BALL_HEIGHT = 400.0

    def reset(self, initial_state: GameState) -> None:
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        car = player.car_data.position
        ball = state.ball.position
        horiz = ((ball[0] - car[0]) ** 2 + (ball[1] - car[1]) ** 2) ** 0.5
        height = ball[2] - car[2]
        dribbling = (
            horiz < self.MAX_HORIZONTAL_DIST
            and self.MIN_BALL_HEIGHT < height < self.MAX_BALL_HEIGHT
        )
        if not dribbling:
            return 0.0
        # Blue attacks +BACK_WALL_Y, orange attacks -BACK_WALL_Y.
        enemy_goal_y = BACK_WALL_Y if player.team_num == 0 else -BACK_WALL_Y
        dist_to_goal = abs(enemy_goal_y - ball[1])
        proximity = 1.0 - min(1.0, dist_to_goal / (2 * BACK_WALL_Y))
        return proximity   # [0,1], higher the closer the dribble is to the enemy net

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return 0.0


class KickoffReward(RewardFunction):
    """Active only when the ball is still near field center (the kickoff state).
    Rewards committing hard to the ball off kickoff — touching it first.

    At kickoff the ball is exactly at (0,0,93), so 'ball near center' is a good
    approximation of 'we are in a kickoff'. Teaches the bot to win the kickoff
    instead of hesitating. Weight ~0.3.
    """

    KICKOFF_BALL_DIST = 300.0   # ball still within this flat dist of center → likely a kickoff

    def reset(self, initial_state: GameState) -> None:
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        ball = state.ball.position
        ball_near_center = (ball[0] ** 2 + ball[1] ** 2) ** 0.5 < self.KICKOFF_BALL_DIST
        if not ball_near_center:
            return 0.0
        return 1.0 if player.ball_touched else 0.0

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return 0.0


class MaintainSpeedReward(RewardFunction):
    """Continuous reward for moving fast, saturating at supersonic.

    Purpose: encode the gamesense that pace = the ability to rotate back on
    defense the instant the ball is cleared. SupersonicReward only fires at the
    full 2200 uu/s threshold; this fills the gap below it so the bot is rewarded
    for *keeping* momentum, not just briefly hitting supersonic. The no-boost
    cap (1410) is the natural cruising speed — above it the bot is spending boost
    or carrying momentum, which is the behavior we want. Weight low (~0.1) so it
    nudges pace without making the bot zoom around ignoring the ball.
    """

    def reset(self, initial_state: GameState) -> None:
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        speed = float(np.linalg.norm(player.car_data.linear_velocity))
        return min(speed / CAR_SUPERSONIC_THRESHOLD, 1.0)

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return 0.0


class RecoveryReward(RewardFunction):
    """Reward orienting upright and toward your motion while airborne OUTSIDE an
    aerial play — i.e. recovering after a bump, a clear, or coming down from a
    challenge. This is the mechanic your stack is missing: with no recovery
    signal the bot tumbles after every aerial/contact and wastes time.

    Gated so it does NOT fire during an intentional aerial (a high, near ball),
    where the car SHOULD be tilted toward the ball — otherwise it would fight
    AerialTouchReward / AerialBallReward. While airborne and the ball is not a
    viable aerial target, the car earns:
        0.5 * upright   (its up vector points up → ready to land wheels-down)
      + 0.5 * vel_align (it points where it is moving → clean landing / half-flip
                         / wavedash setup)
    Output in [0, 1]. Keep the weight small — this is a polish signal, not a
    dominant one.
    """

    BALL_AERIAL_HEIGHT = 400.0     # above this the ball may be a real aerial target
    BALL_AERIAL_RADIUS = 1200.0    # within this flat dist an airborne car may be aerialing

    def reset(self, initial_state: GameState) -> None:
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if player.on_ground:
            return 0.0

        car = player.car_data
        ball = state.ball.position
        car_pos = car.position

        # If the ball is high AND near, the car is probably mid-aerial — leave it
        # to the aerial rewards and don't demand uprightness here.
        flat_dist = float(((ball[0] - car_pos[0]) ** 2 + (ball[1] - car_pos[1]) ** 2) ** 0.5)
        if ball[2] >= self.BALL_AERIAL_HEIGHT and flat_dist <= self.BALL_AERIAL_RADIUS:
            return 0.0

        up_alignment = float(np.clip(car.up()[2], 0.0, 1.0))  # 1.0 = wheels-down ready

        vel = car.linear_velocity
        speed = float(np.linalg.norm(vel))
        if speed > 100.0:
            vel_alignment = float(np.clip(np.dot(car.forward(), vel / speed), 0.0, 1.0))
        else:
            vel_alignment = 0.5  # nearly stationary in air — uprightness is what matters

        return 0.5 * up_alignment + 0.5 * vel_alignment

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return 0.0


class FlickReward(RewardFunction):
    """Reward flicking the ball off a dribble: launching it UP and FORWARD toward
    the enemy net out of a carry. This is the premier 1v1 finishing mechanic, and
    DribbleToGoalReward only rewards the carry itself, never the finish.

    Stateful (per car): a short timer marks 'this car was just dribbling'. If the
    player touches the ball while that timer is live AND the ball leaves with real
    upward velocity, the touch is treated as a flick and rewarded by how fast the
    ball goes up and toward the enemy goal. Fires only on the flick contact;
    output in [0, 1]. Pairs with DribbleSetupState so the bot is in carries often
    enough to learn this.
    """

    MAX_HORIZONTAL_DIST = 170.0   # dribble pose: ball horizontally over the car
    MIN_BALL_HEIGHT = 110.0       # ball above the roof (a real carry)
    MAX_BALL_HEIGHT = 400.0
    DRIBBLE_WINDOW = 8            # steps after a dribble pose during which a touch counts as a flick
    MIN_FLICK_UP_VEL = 300.0      # ball must leave with at least this upward speed
    UP_VEL_CAP = 1500.0           # normalizes the upward-launch factor
    GOAL_VEL_CAP = 2500.0         # normalizes the toward-goal factor

    def __init__(self):
        # car_id -> remaining steps in the dribble window. Per-instance state;
        # one reward object lives per rollout env, shared across its cars.
        self._dribble_timer: dict[int, int] = {}

    def reset(self, initial_state: GameState) -> None:
        self._dribble_timer = {}

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        cid = player.car_id
        car = player.car_data.position
        ball = state.ball.position
        ball_vel = state.ball.linear_velocity

        horiz = float(((ball[0] - car[0]) ** 2 + (ball[1] - car[1]) ** 2) ** 0.5)
        height = float(ball[2] - car[2])
        in_dribble_pose = (
            horiz < self.MAX_HORIZONTAL_DIST
            and self.MIN_BALL_HEIGHT < height < self.MAX_BALL_HEIGHT
        )

        timer = self._dribble_timer.get(cid, 0)
        if in_dribble_pose:
            timer = self.DRIBBLE_WINDOW

        reward = 0.0
        if player.ball_touched and timer > 0 and ball_vel[2] >= self.MIN_FLICK_UP_VEL:
            vel_toward_goal = ball_vel[1] if player.team_num == 0 else -ball_vel[1]
            up_factor = float(np.clip(ball_vel[2] / self.UP_VEL_CAP, 0.0, 1.0))
            goal_factor = float(np.clip(vel_toward_goal / self.GOAL_VEL_CAP, 0.0, 1.0))
            reward = 0.5 * up_factor + 0.5 * goal_factor

        self._dribble_timer[cid] = max(0, timer - 1)
        return reward

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return 0.0


class BoostReserveReward(RewardFunction):
    """Reward keeping a boost reserve when NOT in an immediate offensive play, so
    the bot has boost left to recover/save after a mistake. This directly counters
    the old behavior of dumping all boost to farm SupersonicReward.

    Only active when the bot is NOT attacking near the ball — i.e. the ball is far
    away OR sitting in the bot's defensive half. In those neutral/defensive moments
    it should be topped up, ready to react. It deliberately stays silent when the
    bot is near the ball in the attacking half, so it never discourages spending
    boost on a real play.

    Output centered on the kickoff boost amount (~0.34): holding more than that
    earns positive reward, running near-empty is penalized. Range ~[-0.34, +0.66].
    """

    FAR_BALL_DIST = 2000.0   # ball at least this far => not an immediate play
    BREAK_EVEN = 0.34        # ~kickoff boost; above = reward, below = penalty

    def reset(self, initial_state: GameState) -> None:
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        car = np.asarray(player.car_data.position)
        ball = np.asarray(state.ball.position)
        dist = float(np.linalg.norm(ball - car))

        # Defensive half: blue (team 0) defends -Y, orange (team 1) defends +Y.
        ball_y = float(ball[1])
        in_defensive_half = (ball_y < 0.0) if player.team_num == 0 else (ball_y > 0.0)

        # Near the ball in the attacking half -> active play; don't reward hoarding.
        if dist < self.FAR_BALL_DIST and not in_defensive_half:
            return 0.0

        # boost_amount is normalized 0..1 in rlgym_sim.
        return float(player.boost_amount) - self.BREAK_EVEN

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return 0.0
