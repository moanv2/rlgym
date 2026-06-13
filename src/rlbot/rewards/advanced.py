"""Advanced reward components — richer signal density.

Diego's documented lever (his LESSONS_LEARNED, same repo/opponents as us):
    "the sparse-reward baseline plateaued at 11M with entropy stuck at max;
     adding ~10 reward components broke the plateau — increased signal density
     accelerates learning."

Our original exp_007 stack had only 5 basic components (touch / velocity-to-ball /
face / velocity-to-goal / event) and rewarded NO advanced mechanics. These add the
high-value ones his champion uses and ours completely ignored:
  * aerial_touch          — the single biggest gap (his bot's emergent aerials)
  * supersonic            — fast, boost-using, aggressive play
  * ball_away_from_own_goal — anti-own-goal / rewards clears (he flagged this fixed own-goals)
plus two rlgym_sim built-ins he relies on: save_boost (boost economy) and
align_ball_goal (defensive/offensive alignment).

Wrapped in try/except so the registry degrades gracefully if rlgym_sim is absent
(same pattern as builtin.py).
"""
from __future__ import annotations

import numpy as np

try:
    from rlgym_sim.utils.reward_functions import RewardFunction
    from rlgym_sim.utils.gamestates import GameState, PlayerData
    from rlgym_sim.utils.common_values import (
        BLUE_TEAM,
        BLUE_GOAL_BACK,
        ORANGE_GOAL_BACK,
        BACK_NET_Y,
        CEILING_Z,
        SUPERSONIC_THRESHOLD,
    )
    from rlgym_sim.utils.reward_functions.common_rewards import (
        SaveBoostReward,
        AlignBallGoal,
    )

    from rlbot.rewards.registry import REWARDS

    @REWARDS.register("aerial_touch")
    class AerialTouchReward(RewardFunction):
        """Touching the ball while AIRBORNE, scaled by ball height (0 on the ground
        -> ~1 at the ceiling). Our old reward gave nothing for air play; this is the
        mechanic Diego flags as separating his bot."""

        def __init__(self, height_scale: float = 1.0):
            super().__init__()
            self.height_scale = float(height_scale)

        def reset(self, initial_state: GameState):
            pass

        def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
            if player.ball_touched and not player.on_ground:
                height_frac = min(1.0, float(state.ball.position[2]) / CEILING_Z)
                return height_frac * self.height_scale
            return 0.0

    @REWARDS.register("supersonic")
    class SupersonicReward(RewardFunction):
        """1.0 while the car is at supersonic speed, else 0.0 — rewards fast,
        boost-using, aggressive movement."""

        def reset(self, initial_state: GameState):
            pass

        def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
            speed = float(np.linalg.norm(player.car_data.linear_velocity))
            return 1.0 if speed >= SUPERSONIC_THRESHOLD else 0.0

    @REWARDS.register("ball_away_from_own_goal")
    class BallAwayFromOwnGoalReward(RewardFunction):
        """Reward proportional to the ball's distance from the player's OWN goal:
        ~0 when the ball is in/at our net, ~1 near the enemy net. Discourages letting
        the ball sit on our goal (anti-own-goal; rewards getting it out). Team-aware."""

        def reset(self, initial_state: GameState):
            pass

        def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
            own_goal = BLUE_GOAL_BACK if player.team_num == BLUE_TEAM else ORANGE_GOAL_BACK
            dist = float(np.linalg.norm(state.ball.position - np.asarray(own_goal, dtype=np.float32)))
            return min(1.0, dist / (2.0 * BACK_NET_Y))

    # Reuse rlgym_sim built-ins Diego relies on: boost economy + ball/goal alignment.
    REWARDS.register("save_boost")(SaveBoostReward)
    REWARDS.register("align_ball_goal")(AlignBallGoal)

except ImportError:
    # rlgym_sim not installed — these simply won't be registered until it is.
    pass
