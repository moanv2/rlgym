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

        # Own goal Y depends on team (blue=-BACK_WALL_Y, orange=+BACK_WALL_Y).
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


class KRCOffensivePotentialReward(RewardFunction):
    """Lucy-SKG 'Offensive Potential' via a KRC (geometric-mean) combination instead of the
    additive position terms (AlignBallGoal + BackboardDefense), which pay off FAR from the ball
    and reward camping/standoffs (our observed regression). It combines three normalized
    components — measured TOWARD the opponent goal:
        align : car is behind the ball on the ball->goal line (pushing it goalward)   in [-1, 1]
        vel   : ball velocity projected toward the opponent goal                      in [-1, 1]
        prox  : how close the ball is to the opponent goal                            in [ 0, 1]
    combined as a SIGN-adjusted geometric mean:
        reward = sign * (|align| * |vel| * |prox|) ** (1/3)
    The geometric mean means ALL THREE must be high to score -> it CANNOT be farmed by maxing one
    component alone (the additive-terms exploit). `sign` is +1 only when the moment is genuinely
    offensive (aligned AND the ball is moving goalward), otherwise -1, so being out of position or
    knocking the ball the wrong way is penalized. Output in [-1, 1]. Replaces AlignBallGoal +
    BackboardDefense; weight ~0.6 (Lucy-SKG, re-scaled to our anchor).
    """

    GOAL_CENTER_Z = 321.0  # ~half the goal opening height; goal aim point
    DIST_NORM = 2.0 * BACK_WALL_Y  # max ball->opp-goal distance (ball at own goal) ~ 10240 uu

    def reset(self, initial_state: GameState) -> None:
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        sign_team = 1.0 if player.team_num == 0 else -1.0  # blue (team 0) attacks +Y
        opp_goal = np.array([0.0, sign_team * BACK_WALL_Y, self.GOAL_CENTER_Z], dtype=np.float64)
        ball = np.asarray(state.ball.position, dtype=np.float64)
        car = np.asarray(player.car_data.position, dtype=np.float64)
        ball_vel = np.asarray(state.ball.linear_velocity, dtype=np.float64)

        ball_to_goal = opp_goal - ball
        d_bg = float(np.linalg.norm(ball_to_goal))
        if d_bg < 1e-6:
            return 0.0
        bg_dir = ball_to_goal / d_bg

        # 1) alignment: car behind the ball, on the ball->goal line
        car_to_ball = ball - car
        d_cb = float(np.linalg.norm(car_to_ball))
        align = float(np.dot(car_to_ball / d_cb, bg_dir)) if d_cb > 1e-6 else 0.0
        # 2) ball velocity toward the opponent goal (clamped to [-1, 1])
        vel = max(-1.0, min(1.0, float(np.dot(ball_vel, bg_dir)) / BALL_MAX_SPEED))
        # 3) proximity of the ball to the opponent goal (1 at the goal, 0 far away)
        prox = max(0.0, 1.0 - d_bg / self.DIST_NORM)

        sign = 1.0 if (align >= 0.0 and vel >= 0.0) else -1.0
        return sign * (abs(align) * abs(vel) * abs(prox)) ** (1.0 / 3.0)

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return 0.0


class PossessionEventReward(RewardFunction):
    """Anti-passivity lever: a sparse EVENT reward for WINNING the ball — gaining last-touch from
    the other team (a won challenge/50-50) or from a neutral state (the first touch off a kickoff).

    Distinct from the continuous touch rewards (TouchBall, TouchBallToGoalAcceleration): those pay
    for contact/quality every touch, this fires ONCE per possession transition. It directly rewards
    the two highest-leverage moments in 1v1 — winning the kickoff race and winning a contested ball —
    which is exactly the aggression the reward-research flagged as untested (we tried Nexto-style,
    KRC, concede, touch-accel, low-entropy, but never a possession-transition signal). It also
    attacks the documented weakness of bots like Nexto (giving the ball away).

    Designed for the ZERO-SUM stack (rewards.zero_sum: true): it emits +1 only to the player whose
    team just gained the ball, and ZeroSumReward turns that into the symmetric penalty for the team
    that just lost it, so there is no double counting.

    Stateful + shared-instance-safe: the possession transition is computed EXACTLY ONCE per physics
    step (detected by a change in ball position), so both players in a 1v1 see a consistent event and
    repeated same-step calls (the other player, or ZeroSumReward's second pass) are no-ops.
    """

    def reset(self, initial_state: GameState) -> None:
        self._possessor = -1  # team_num of the last toucher (-1 = neutral / kickoff)
        self._gainer = -1     # team that gained possession THIS physics step (-1 = none)
        self._cur_ball_pos = np.asarray(initial_state.ball.position, dtype=np.float64)

    def _advance(self, state: GameState) -> None:
        pos = np.asarray(state.ball.position, dtype=np.float64)
        if np.array_equal(pos, self._cur_ball_pos):
            return  # same physics step -> no-op (other player / zero-sum second pass)
        self._cur_ball_pos = pos
        self._gainer = -1
        toucher = next((int(p.team_num) for p in state.players if p.ball_touched), None)
        if toucher is not None and toucher != self._possessor:
            self._gainer = toucher        # a possession transition (won the ball or first touch)
            self._possessor = toucher

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        self._advance(state)
        return 1.0 if int(player.team_num) == self._gainer else 0.0

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return 0.0


class TouchBallToGoalAccelerationReward(RewardFunction):
    """Reward a ball TOUCH by how much it accelerates the ball toward the OPPONENT
    goal — Lucy-SKG's Touch-Ball-to-Goal-Acceleration, the upgrade that let it beat
    Necto/Nexto. Replaces the flat per-touch reward (TouchBallReward weight 5), which
    is farmable by aimless taps/dribble-pokes that make no goalward progress.

    On the step the player touches the ball, reward = the signed change in the ball's
    goal-axis (Y) velocity between the previous and current step, normalized so a
    ~2300 uu/s goalward swing ≈ 1.0 (Lucy's scale). A weak touch pays ~0; a touch that
    knocks the ball toward the player's own goal pays negative. Touches that genuinely
    drive the ball at the net are what pay — outcome, not contact.

    Stateful: tracks the previous ball velocity. To stay correct AND symmetric under a
    shared 1v1 reward instance (get_reward is called once per player per step on the
    SAME state, and ZeroSumReward may call it again), the velocity window is rolled
    forward EXACTLY ONCE per physics step — detected by a change in ball position — so
    both players in a step compare against the same previous velocity.
    """

    GOAL_ACCEL_NORM = 2300.0  # uu/s; a solid goalward touch ≈ 1.0 (Lucy-SKG scale)

    def reset(self, initial_state: GameState) -> None:
        self._prev_ball_vel = np.asarray(initial_state.ball.linear_velocity, dtype=np.float64)
        self._cur_ball_vel = self._prev_ball_vel.copy()
        self._cur_ball_pos = np.asarray(initial_state.ball.position, dtype=np.float64)

    def _advance(self, state: GameState) -> None:
        pos = np.asarray(state.ball.position, dtype=np.float64)
        # Only roll forward on a genuinely new step (ball moved). Same-step repeat
        # calls (the other player, or ZeroSumReward's second pass) are no-ops.
        if not np.array_equal(pos, self._cur_ball_pos):
            self._prev_ball_vel = self._cur_ball_vel
            self._cur_ball_vel = np.asarray(state.ball.linear_velocity, dtype=np.float64)
            self._cur_ball_pos = pos

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        self._advance(state)
        if not player.ball_touched:
            return 0.0
        # Blue (team 0) attacks +Y, orange attacks -Y; sign makes "toward opp goal" positive.
        sign = 1.0 if player.team_num == 0 else -1.0
        cur_goalward = sign * self._cur_ball_vel[1]
        prev_goalward = sign * self._prev_ball_vel[1]
        return float((cur_goalward - prev_goalward) / self.GOAL_ACCEL_NORM)

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return 0.0
