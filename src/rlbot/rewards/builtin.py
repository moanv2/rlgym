"""Registers built-in rlgym_sim rewards under stable string names.

Adding a new reward? Either:
  1. Register an existing rlgym_sim/rlgym-tools reward here, or
  2. Write a new RewardFunction subclass and decorate it with @REWARDS.register("my_name")
"""
from __future__ import annotations

from rlbot.rewards.registry import REWARDS

# Lazy-imported to keep tests cheap; actual training imports rlgym_sim anyway.
try:
    from rlgym_sim.utils.reward_functions.common_rewards import (
        AlignBallGoal,
        EventReward,
        FaceBallReward,
        SaveBoostReward,
        TouchBallReward,
        VelocityBallToGoalReward,
        VelocityPlayerToBallReward,
    )

    REWARDS.register("velocity_player_to_ball")(VelocityPlayerToBallReward)
    REWARDS.register("velocity_ball_to_goal")(VelocityBallToGoalReward)
    REWARDS.register("face_ball")(FaceBallReward)
    REWARDS.register("touch_ball")(TouchBallReward)
    REWARDS.register("event")(EventReward)
    # sqrt(boost_amount) — rewards keeping boost, weighted toward having *some* over none.
    REWARDS.register("save_boost")(SaveBoostReward)
    # AlignBallGoal(defense=1.0, offense=1.0) — per-step cosine alignment of
    # (car ↔ ball) with (own_goal ↔ car) and (car ↔ opp_goal). Teaches *positioning*:
    # approach the ball from the side that pushes it toward the opponent's net.
    # Helpful for low-speed near-ball play where the bot has to choose which side
    # of the ball to be on, not just whether to chase it.
    REWARDS.register("align_ball_goal")(AlignBallGoal)
except ImportError:
    # rlgym_sim not installed — registry stays empty until it is.
    pass
