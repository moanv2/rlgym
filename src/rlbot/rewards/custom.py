"""Custom reward functions for the shaping curriculum.

These implement the rewards recommended in the RLGym-PPO-Guide's *rewards* and
*making a good bot* sections that rlgym_sim does not ship:

  speed_toward_ball : SpeedTowardBallReward  — never-negative "move at the ball"
  in_air            : InAirReward            — anti "forgetting how to jump"
  double_jump       : DoubleJumpReward       — reward spending the air jump (dodge/2nd jump)
  strong_touch      : StrongTouchReward      — touch scaled by change in ball speed
  air_touch         : AirTouchReward         — aerial touches (height x air-time)

Two rlgym_sim quirks shape these implementations:

  1. ``CombinedReward`` forwards ``reset``/``get_reward``/``get_final_reward`` to its
     children but **not** ``pre_step``. So any cross-step state (previous ball
     velocity, previous ``has_flip``) must be maintained inside ``get_reward`` itself.
  2. ``PlayerData`` has no ``air_time`` field, so AirTouchReward tracks per-car air
     time itself.

The call contract we rely on: within one env step, ``get_reward`` is invoked exactly
once per player, in player order. (rlgym_sim's CombinedReward iterates players in the
outer loop; ZeroSumReward does the same.)
"""
from __future__ import annotations

from rlbot.rewards.registry import REWARDS

# Lazy import block — mirrors builtin.py so importing rlbot.rewards stays cheap and
# does not require rlgym_sim to be installed (e.g. in unit-test-only environments).
try:
    import numpy as np
    from rlgym_sim.utils import RewardFunction
    from rlgym_sim.utils.common_values import (
        BACK_WALL_Y,
        BALL_MAX_SPEED,
        BLUE_GOAL_BACK,
        BLUE_TEAM,
        CAR_MAX_SPEED,
        CEILING_Z,
        ORANGE_GOAL_BACK,
    )
    from rlgym_sim.utils.gamestates import GameState, PlayerData
except ImportError:
    pass
else:

    @REWARDS.register("speed_toward_ball")
    class SpeedTowardBallReward(RewardFunction):
        """Fraction of the car's speed pointed at the ball, clamped to [0, 1].

        Preferred over the built-in ``velocity_player_to_ball``: it never punishes
        moving *away* from the ball (many good behaviours — rotating back, lining up a
        shot — require that), so it adds no noise while still pulling a fresh bot toward
        the ball. (Guide: "rewards" section.)
        """

        def reset(self, initial_state: GameState) -> None:
            pass

        def get_reward(self, player: PlayerData, state: GameState, previous_action) -> float:
            pos_diff = state.ball.position - player.car_data.position
            dist = float(np.linalg.norm(pos_diff))
            if dist < 1e-6:
                return 0.0
            dir_to_ball = pos_diff / dist
            speed_toward = float(np.dot(player.car_data.linear_velocity, dir_to_ball))
            if speed_toward <= 0.0:
                return 0.0
            return min(speed_toward / CAR_MAX_SPEED, 1.0)

    @REWARDS.register("in_air")
    class InAirReward(RewardFunction):
        """1.0 while airborne, 0.0 on the ground.

        A small weight on this keeps the bot pressing jump. Fresh bots quickly stop
        jumping (ground control is easier) and then struggle to rediscover it.
        (Guide: "making a good bot" — *why do bots forget how to jump?*)
        """

        def reset(self, initial_state: GameState) -> None:
            pass

        def get_reward(self, player: PlayerData, state: GameState, previous_action) -> float:
            return 0.0 if player.on_ground else 1.0

    @REWARDS.register("double_jump")
    class DoubleJumpReward(RewardFunction):
        """Reward the instant a car spends its *air jump* — a double jump or air dodge.

        Bots that learn to chase on the ground often abandon their second jump entirely
        (it's harder to time than driving), which kills aerials and fast flip-approaches.
        ``in_air`` keeps the *first* jump alive; this rewards the *second*. It fires a
        one-off +1 the step the air jump is used, detected as ``has_flip`` going
        True→False while airborne.

        The jump-press gate (``previous_action[JUMP]``) rules out the false positive
        where ``has_flip`` simply expires at the end of the ~1.25s flip window without
        the car ever jumping. ``previous_action`` is the parsed 8-dim controller vector
        ``[throttle, steer, pitch, yaw, roll, jump, boost, handbrake]`` (rlgym_sim's
        Match passes ``_prev_actions[i]``), so index 5 is the jump button.

        Per-car ``has_flip`` is tracked here because the air jump is a per-step
        transition, not a standing PlayerData field. (Guide: "making a good bot" — keep
        the bot using its jump; it's the gateway to aerials.)
        """

        _JUMP_IDX = 5  # index of the jump button in the controller action vector

        def __init__(self) -> None:
            super().__init__()
            self._had_flip: dict[int, bool] = {}

        def reset(self, initial_state: GameState) -> None:
            self._had_flip = {p.car_id: bool(p.has_flip) for p in initial_state.players}

        def get_reward(self, player: PlayerData, state: GameState, previous_action) -> float:
            had_flip = self._had_flip.get(player.car_id, bool(player.has_flip))
            self._had_flip[player.car_id] = bool(player.has_flip)

            # Air jump "spent": the flip token was consumed this step while airborne.
            used_air_jump = had_flip and not player.has_flip and not player.on_ground
            # ...and only if the car actually pressed jump (else the flip window just
            # timed out). Defensive about a missing/short/None previous_action.
            jump_pressed = (
                previous_action is not None
                and len(previous_action) > self._JUMP_IDX
                and float(previous_action[self._JUMP_IDX]) > 0.0
            )
            return 1.0 if (used_air_jump and jump_pressed) else 0.0

    @REWARDS.register("strong_touch")
    class StrongTouchReward(RewardFunction):
        """Reward a ball touch scaled by how much it changed the ball's velocity.

        A gentle dribble-push barely moves the ball and earns ~0; a powerful shot or
        clear earns close to 1. This is far less farmable than a flat touch reward,
        which a bot will happily exploit by nudging the ball forever. (Guide: "making a
        good bot" — *a better ball-touch reward*.)

        Tracks the previous step's ball velocity internally; it is advanced once all
        players have been scored for the step (see module docstring).
        """

        def __init__(self, max_delta: float = float(BALL_MAX_SPEED)):
            super().__init__()
            self.max_delta = float(max_delta)
            self._prev_ball_vel = np.zeros(3, dtype=np.float32)
            self._n_players = 1
            self._seen = 0

        def reset(self, initial_state: GameState) -> None:
            self._prev_ball_vel = np.asarray(initial_state.ball.linear_velocity, dtype=np.float32)
            self._n_players = max(len(initial_state.players), 1)
            self._seen = 0

        def get_reward(self, player: PlayerData, state: GameState, previous_action) -> float:
            cur = np.asarray(state.ball.linear_velocity, dtype=np.float32)
            reward = 0.0
            if player.ball_touched:
                delta = float(np.linalg.norm(cur - self._prev_ball_vel))
                reward = min(delta / self.max_delta, 1.0)
            # Advance the "previous" velocity once the whole step has been scored, so
            # every player in the step compares against the genuine prior-step velocity.
            self._seen += 1
            if self._seen >= self._n_players:
                self._prev_ball_vel = cur
                self._seen = 0
            return reward

    @REWARDS.register("air_touch")
    class AirTouchReward(RewardFunction):
        """Reward an aerial touch, scaled by ``min(air-time frac, ball-height frac)``.

        Height alone is farmable via cheap high wall-reads, so we gate it on how long
        the car has been airborne: full reward needs both a sustained aerial *and* a
        high ball. Per-car air time is tracked here because PlayerData has none.
        (Guide: "making a good bot" — *a good air-touch reward*.)
        """

        def __init__(self, max_time_in_air: float = 1.75, tick_skip: int = 8):
            super().__init__()
            self.max_time = float(max_time_in_air)
            self.dt = tick_skip / 120.0  # seconds of game time per policy step
            self._air_time: dict[int, float] = {}

        def reset(self, initial_state: GameState) -> None:
            self._air_time = {p.car_id: 0.0 for p in initial_state.players}

        def get_reward(self, player: PlayerData, state: GameState, previous_action) -> float:
            if player.on_ground:
                self._air_time[player.car_id] = 0.0
                return 0.0
            air_time = self._air_time.get(player.car_id, 0.0) + self.dt
            self._air_time[player.car_id] = air_time
            if not player.ball_touched:
                return 0.0
            air_frac = min(air_time, self.max_time) / self.max_time
            height_frac = min(float(state.ball.position[2]) / CEILING_Z, 1.0)
            return min(air_frac, height_frac)

    @REWARDS.register("shot_toward_goal")
    class ShotTowardGoalReward(RewardFunction):
        """Reward a touch by how much goalward speed it adds, weighted by distance.

        This is the *directional* cousin of ``strong_touch``: instead of rewarding any
        change in ball speed, it rewards only the component of the velocity change that
        points at the opponent's net — ``dot(Δball_vel, dir_to_goal)``. A powerful shot
        sent straight at the net earns ~1; an equally powerful clear sideways or backward
        earns ~0. That fixes the gap ``strong_touch`` leaves open: a bot can max out a
        flat power reward by smacking the ball *anywhere*, which is exactly the aimless
        long-range play we're trying to cure.

        ``dist_frac`` (clamped distance from the ball to the goal over ``BACK_WALL_Y``)
        scales the reward up the farther the strike originates, so the signal concentrates
        on the weakness — aiming *from afar*. Point-blank finishing is left to the sparse
        ``event.goal`` reward, not double-counted here.

        Not farmable: you cannot gain reward without genuinely accelerating the ball
        toward the net, and once it's moving fast goalward there's little Δv left to add.

        Like ``StrongTouchReward`` it tracks the previous step's ball velocity internally
        and advances it only once every player has been scored for the step, so both cars
        in a 1v1 compare against the same genuine prior-step velocity (see module
        docstring for the call contract).
        """

        def __init__(self, max_delta: float = float(BALL_MAX_SPEED), far_ref: float = float(BACK_WALL_Y)):
            super().__init__()
            self.max_delta = float(max_delta)
            self.far_ref = float(far_ref)
            self._prev_ball_vel = np.zeros(3, dtype=np.float32)
            self._n_players = 1
            self._seen = 0

        def reset(self, initial_state: GameState) -> None:
            self._prev_ball_vel = np.asarray(initial_state.ball.linear_velocity, dtype=np.float32)
            self._n_players = max(len(initial_state.players), 1)
            self._seen = 0

        def get_reward(self, player: PlayerData, state: GameState, previous_action) -> float:
            cur = np.asarray(state.ball.linear_velocity, dtype=np.float32)
            reward = 0.0
            if player.ball_touched:
                objective = np.array(
                    ORANGE_GOAL_BACK if player.team_num == BLUE_TEAM else BLUE_GOAL_BACK,
                    dtype=np.float32,
                )
                to_goal = objective - np.asarray(state.ball.position, dtype=np.float32)
                dist = float(np.linalg.norm(to_goal))
                if dist > 1e-6:
                    dir_to_goal = to_goal / dist
                    gained = float(np.dot(cur - self._prev_ball_vel, dir_to_goal))
                    if gained > 0.0:
                        dist_frac = min(dist / self.far_ref, 1.0)
                        reward = min(gained / self.max_delta, 1.0) * dist_frac
            # Advance the shared "previous" velocity once the whole step has been scored.
            self._seen += 1
            if self._seen >= self._n_players:
                self._prev_ball_vel = cur
                self._seen = 0
            return reward
