"""Game-knowledge reward functions adopted from Diego's `origin/diego` branch.

These encode field geometry / boost-economy knowledge that the obs builder does
not capture sharply. Diego wrote them but never wired them in — they carried no
``@REWARDS.register`` decorator and were not imported, so they never ran. Here
they are registered into the YAML registry (so the curriculum can use them by
name), sourced from our single constants module, and unit-tested.

Adopted (with the curriculum's design constraints in mind):

  big_boost_proximity   : route to a big pad — only when low-boost AND ball far
  boost_reserve         : hold a reserve when NOT in an active play (sqrt-shaped)
  ball_away_from_own_goal: clear the ball away from your net (scaled by threat)
  recovery              : land wheels-down / oriented after a non-aerial air moment
  backboard_defense     : shadow-defense positioning, goal-side on the ball→goal line

Changes from Diego's originals (improvements flagged in review):
  - boost_reserve uses a sqrt shape (matches save_boost's diminishing returns)
    instead of a linear ``boost - break_even``.
  - ball_away_from_own_goal scales by how deep the ball is in your half, so a
    clear off your doorstep counts more than one at midfield.
  - backboard_defense is continuous (cosine alignment to the ball→own-goal line
    while goal-side) instead of a binary y-only "am I behind the ball" check.

Same rlgym_sim call contract as custom.py: ``CombinedReward`` forwards
reset/get_reward/get_final_reward but NOT pre_step, and calls get_reward once
per player per step. All rewards here are stateless across steps, so there is no
cross-step bookkeeping to get wrong. We follow custom.py and do not override
get_final_reward (the base class default applies, as for the existing customs).
"""
from __future__ import annotations

# Lazy import block — mirrors custom.py so importing rlbot.rewards stays cheap and
# does not require rlgym_sim to be installed (e.g. in unit-test-only environments).
try:
    import numpy as np
    from rlgym_sim.utils import RewardFunction
    from rlgym_sim.utils.gamestates import GameState, PlayerData

    from rlbot.rewards.registry import REWARDS
    from rlbot.utils.rl_constants import (
        BACK_WALL_Y,
        BALL_MAX_SPEED,
        BIG_BOOST_PAD_POSITIONS,
        BLUE_TEAM,
    )
except ImportError:
    pass
else:

    @REWARDS.register("big_boost_proximity")
    class BigBoostProximityReward(RewardFunction):
        """Reward heading to a big boost pad — but ONLY when low on boost AND the
        ball is far enough that grabbing boost is the sensible move.

        This is the *active routing* signal the dense `save_boost` lacks: rather
        than rewarding the end-state of having boost (which exp_010 showed barely
        moves the needle — the bot stayed empty ~67% of the time), it rewards the
        *act* of closing distance to a pad when it is safe to do so. The ball-far
        gate keeps it from dragging the bot off an immediate contest.

        ``reward = pad_proximity * ball_distance_factor`` in [0, 1], and exactly 0
        unless (boost < threshold) AND (ball far) AND (within range of a big pad).
        """

        def __init__(
            self,
            low_boost_threshold: float = 0.40,   # boost is normalized 0..1; top up proactively (pros refill before empty)
            max_reward_distance: float = 1500.0,  # uu — only reward within this radius of a big pad
            min_ball_distance: float = 1500.0,    # ball must be at least this far before boost-seeking pays
            ball_distance_scale: float = 3000.0,  # how fast the ball-distance factor ramps to 1.0
        ) -> None:
            super().__init__()
            self.low_boost_threshold = float(low_boost_threshold)
            self.max_reward_distance = float(max_reward_distance)
            self.min_ball_distance = float(min_ball_distance)
            self.ball_distance_scale = float(ball_distance_scale)
            # Precompute the big-pad XY positions once.
            self._pads_xy = np.asarray([(p[0], p[1]) for p in BIG_BOOST_PAD_POSITIONS], dtype=np.float64)

        def reset(self, initial_state: GameState) -> None:
            pass

        def get_reward(self, player: PlayerData, state: GameState, previous_action) -> float:
            if player.boost_amount > self.low_boost_threshold:
                return 0.0

            car_pos = np.asarray(player.car_data.position, dtype=np.float64)
            ball_pos = np.asarray(state.ball.position, dtype=np.float64)
            ball_distance = float(np.linalg.norm(ball_pos - car_pos))

            # Only reward boost-seeking when the ball is NOT in immediate play.
            if ball_distance < self.min_ball_distance:
                return 0.0
            ball_factor = min(1.0, (ball_distance - self.min_ball_distance) / self.ball_distance_scale)

            dists = np.linalg.norm(self._pads_xy - car_pos[:2], axis=1)
            min_pad_distance = float(dists.min())
            if min_pad_distance > self.max_reward_distance:
                return 0.0

            pad_proximity = 1.0 - (min_pad_distance / self.max_reward_distance)
            return pad_proximity * ball_factor

    @REWARDS.register("boost_reserve")
    class BoostReserveReward(RewardFunction):
        """Reward keeping a boost reserve when NOT in an immediate offensive play,
        so the bot has boost left to recover/save after a mistake.

        Active only when the bot is NOT attacking near the ball — the ball is far
        OR sitting in the bot's defensive half. In those neutral/defensive moments
        it should be topped up; it stays silent when near the ball in the attacking
        half so it never discourages spending boost on a real play.

        Shape (improvement over Diego's linear ``boost - break_even``): centered on
        ``sqrt`` like save_boost — ``sqrt(boost) - sqrt(break_even)``. The sqrt
        makes the penalty for running toward empty steep and the reward for topping
        past full shallow, so it punishes being dry without rewarding hoarding.
        """

        def __init__(self, far_ball_dist: float = 2000.0, break_even: float = 0.34) -> None:
            super().__init__()
            self.far_ball_dist = float(far_ball_dist)
            self.break_even = float(break_even)
            self._sqrt_break_even = float(np.sqrt(break_even))

        def reset(self, initial_state: GameState) -> None:
            pass

        def get_reward(self, player: PlayerData, state: GameState, previous_action) -> float:
            car = np.asarray(player.car_data.position, dtype=np.float64)
            ball = np.asarray(state.ball.position, dtype=np.float64)
            dist = float(np.linalg.norm(ball - car))

            ball_y = float(ball[1])
            in_defensive_half = (ball_y < 0.0) if player.team_num == BLUE_TEAM else (ball_y > 0.0)

            # Near the ball in the attacking half -> active play; don't reward hoarding.
            if dist < self.far_ball_dist and not in_defensive_half:
                return 0.0

            # boost_amount is normalized 0..1 in rlgym_sim.
            return float(np.sqrt(max(0.0, player.boost_amount))) - self._sqrt_break_even

    @REWARDS.register("ball_away_from_own_goal")
    class BallAwayFromOwnGoalReward(RewardFunction):
        """Reward the ball moving AWAY from the bot's own goal when it is in the
        bot's defensive half; penalize it moving toward the own net.

        Directly discourages own-goals and rewards clean clears. Active only in the
        bot's defensive half (where own goals happen). Blue defends -Y, orange +Y.

        Improvement over Diego's original: scaled by a ``threat`` factor — how deep
        the ball is in the bot's half (0 at the halfway line, 1 on the goal line) —
        so a clear off the doorstep is worth more than nudging the ball at midfield.
        Output ~[-1, 1].
        """

        def reset(self, initial_state: GameState) -> None:
            pass

        def get_reward(self, player: PlayerData, state: GameState, previous_action) -> float:
            ball_y = float(state.ball.position[1])
            ball_vel_y = float(state.ball.linear_velocity[1])

            if player.team_num == BLUE_TEAM:  # blue defends -Y
                if ball_y > 0:
                    return 0.0
                vel_toward_own_goal = -ball_vel_y      # ball moving -Y is toward own goal
                threat = min(1.0, -ball_y / BACK_WALL_Y)  # 0 at midfield, 1 on the goal line
            else:  # orange defends +Y
                if ball_y < 0:
                    return 0.0
                vel_toward_own_goal = ball_vel_y
                threat = min(1.0, ball_y / BACK_WALL_Y)

            return (-vel_toward_own_goal / BALL_MAX_SPEED) * threat

    @REWARDS.register("recovery")
    class RecoveryReward(RewardFunction):
        """Reward orienting upright and toward your motion while airborne OUTSIDE
        an aerial play — recovering after a bump, a clear, or coming down from a
        challenge. The stack has no recovery signal, so the bot tumbles after every
        contact and wastes time.

        Gated so it does NOT fire during an intentional aerial (a high, near ball),
        where the car SHOULD be tilted toward the ball — otherwise it fights the
        aerial rewards. While airborne and the ball is not a viable aerial target:
            0.5 * upright    (up vector points up → ready to land wheels-down)
          + 0.5 * vel_align  (nose points where it is moving → clean landing)
        Output in [0, 1].
        """

        def __init__(self, ball_aerial_height: float = 400.0, ball_aerial_radius: float = 1200.0) -> None:
            super().__init__()
            self.ball_aerial_height = float(ball_aerial_height)
            self.ball_aerial_radius = float(ball_aerial_radius)

        def reset(self, initial_state: GameState) -> None:
            pass

        def get_reward(self, player: PlayerData, state: GameState, previous_action) -> float:
            if player.on_ground:
                return 0.0

            car = player.car_data
            ball = state.ball.position
            car_pos = car.position

            # If the ball is high AND near, the car is probably mid-aerial — leave it
            # to the aerial rewards and don't demand uprightness here.
            flat_dist = float(((ball[0] - car_pos[0]) ** 2 + (ball[1] - car_pos[1]) ** 2) ** 0.5)
            if ball[2] >= self.ball_aerial_height and flat_dist <= self.ball_aerial_radius:
                return 0.0

            up_alignment = float(np.clip(car.up()[2], 0.0, 1.0))  # 1.0 = wheels-down ready

            vel = car.linear_velocity
            speed = float(np.linalg.norm(vel))
            if speed > 100.0:
                vel_alignment = float(np.clip(np.dot(car.forward(), vel / speed), 0.0, 1.0))
            else:
                vel_alignment = 0.5  # nearly stationary in air — uprightness is what matters

            return 0.5 * up_alignment + 0.5 * vel_alignment

    @REWARDS.register("backboard_defense")
    class BackboardDefenseReward(RewardFunction):
        """Reward shadow-defense positioning: be goal-side of the ball AND on the
        ball→own-goal line, when the ball is in the bot's defensive half.

        Complements AlignBallGoal (general goal-ball line) with a defense-specific,
        own-half-gated signal.

        Improvement over Diego's original: his version was binary and only checked
        the Y ordering (``car_y < ball_y``), so a car goal-side in Y but far out wide
        scored full marks. This version is continuous — the cosine alignment between
        (car→own_goal) and (ball→own_goal) in the XY plane — so it rewards actually
        sitting on the defensive line, not just being deep. Output in [0, 1], and 0
        when the bot is ball-side of the ball.
        """

        def __init__(self) -> None:
            super().__init__()
            self._blue_goal_xy = np.asarray((0.0, -BACK_WALL_Y), dtype=np.float64)
            self._orange_goal_xy = np.asarray((0.0, BACK_WALL_Y), dtype=np.float64)

        def reset(self, initial_state: GameState) -> None:
            pass

        def get_reward(self, player: PlayerData, state: GameState, previous_action) -> float:
            ball_y = float(state.ball.position[1])
            if player.team_num == BLUE_TEAM:
                if ball_y > 0:
                    return 0.0  # ball in opp half — no defensive bonus
                own_goal = self._blue_goal_xy
                goal_side = float(player.car_data.position[1]) < ball_y
            else:
                if ball_y < 0:
                    return 0.0
                own_goal = self._orange_goal_xy
                goal_side = float(player.car_data.position[1]) > ball_y

            if not goal_side:
                return 0.0  # bot is upfield of the ball — not shadowing

            car_xy = np.asarray(player.car_data.position[:2], dtype=np.float64)
            ball_xy = np.asarray(state.ball.position[:2], dtype=np.float64)

            car_to_goal = own_goal - car_xy
            ball_to_goal = own_goal - ball_xy
            n1 = float(np.linalg.norm(car_to_goal))
            n2 = float(np.linalg.norm(ball_to_goal))
            if n1 < 1e-6 or n2 < 1e-6:
                return 1.0  # essentially on the goal line / on top of it
            cos = float(np.dot(car_to_goal, ball_to_goal) / (n1 * n2))
            return max(0.0, cos)
