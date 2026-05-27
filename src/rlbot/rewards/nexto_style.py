"""Nexto-inspired reward function for Rocket League PPO training.

This is a faithful re-implementation of the *shape* of Nexto's reward design,
using only reward primitives available in rlgym_sim.utils.reward_functions.common_rewards
plus our ZeroSumReward wrapper. It is NOT a 1:1 port of Nexto's exact weights —
those were tuned over millions of dollars of cloud compute and a year of
iteration. These weights are a sensible starting point for a 45-day project.

Design philosophy (the same one Nexto uses):
    1. Event rewards (goals, saves, shots) carry the most weight overall —
       they are sparse but represent the actual game objectives.
    2. Multiple overlapping continuous signals so the policy can differentiate
       similar states. Each component covers a different axis of "is this good":
         - velocity-based:  is the bot moving toward the right place?
         - distance-based:  is the bot/ball close to the right place?
         - alignment-based: is the bot positioned between ball and goal?
         - territorial:     is the ball in the right half of the field?
    3. Wrapped in ZeroSumReward so 1v1 self-play stays competitive — what blue
       gains, orange loses. Without this, self-play converges on cooperative
       behavior that does not transfer to playing against an adversary.

Usage in simple_bot.py:
    from rlbot.rewards.nexto_style import build_nexto_style_reward
    reward_fn = build_nexto_style_reward()

Or with custom team_spirit (only relevant for 2v2+):
    reward_fn = build_nexto_style_reward(team_spirit=0.0, zero_sum=True)

Skills this reward shape encourages (without explicit per-skill rewards):
    - Aerials      → emerges from TouchBallReward + EventReward(touch/shot)
                     when the ball is in the air. Add an aerial state setter
                     for faster learning.
    - Dribbling    → emerges from TouchBallReward + low-distance continuous
                     reward. The bot learns ball is on its roof = high reward.
    - Demos        → EventReward(demo=0.5) directly rewards demolitions.
    - Shadow def.  → AlignBallGoal(defense=1.0) rewards positioning between
                     ball and own goal.
    - Boost mgmt.  → SaveBoostReward + EventReward(boost_pickup) reward
                     both saving and grabbing boost.
"""
from __future__ import annotations

from rlgym_sim.utils.reward_functions import CombinedReward
from rlgym_sim.utils.reward_functions.common_rewards import (
    AlignBallGoal,
    BallYCoordinateReward,
    EventReward,
    FaceBallReward,
    LiuDistanceBallToGoalReward,
    LiuDistancePlayerToBallReward,
    SaveBoostReward,
    TouchBallReward,
    VelocityBallToGoalReward,
    VelocityPlayerToBallReward,
)

from rlbot.rewards.zero_sum import ZeroSumReward


def build_nexto_style_reward(
    zero_sum: bool = True,
    team_spirit: float = 0.0,
    opp_scale: float = 1.0,
):
    """Construct the full Nexto-style reward stack.

    Parameters
    ----------
    zero_sum : bool
        Wrap the combined reward in ZeroSumReward. Strongly recommended for
        1v1 self-play. Set to False only if you have a specific reason.
    team_spirit : float
        ZeroSumReward team_spirit (0=pure individual, 1=pure team avg). For
        1v1 this is irrelevant (one player per team) — keep at 0.
    opp_scale : float
        How strongly the opponent's reward is subtracted from yours. 1.0 is
        balanced. Higher values make the policy more competitive/defensive,
        lower values closer to non-zero-sum.

    Returns
    -------
    A RewardFunction (CombinedReward or ZeroSumReward wrapping it) ready to
    pass to rlgym_sim.make(reward_fn=...).

    Notes on the weights:
        - Event rewards dominate at 12x — sparse but they matter most.
        - Touch ball at 5x because contact is rare and represents a phase
          transition for the bot (random → engaged).
        - Velocity-to-goal at 2x is the strongest offensive continuous signal.
        - LiuDistance rewards are exponential proximity — they peak sharply
          right at the target, giving the policy a clearer gradient than
          linear velocity rewards alone.
        - AlignBallGoal at 0.4 with defense=1, offense=1 — balances offense
          and defense positioning.
        - BallYCoordinate at 0.5 — rewards pushing the ball into the
          opponent's half (positive y by convention). Cheap territorial bonus.
        - SaveBoost at 0.05 — tiny weight so the bot does not become miserly
          with boost at the expense of plays.
    """
    components = (
        # --- Approach signals: get the bot to the ball ---
        VelocityPlayerToBallReward(),       # linear chase
        LiuDistancePlayerToBallReward(),    # exponential proximity — peaks when very close
        # --- Offensive signals: push the ball at the opp goal ---
        VelocityBallToGoalReward(),         # ball velocity toward opp net
        LiuDistanceBallToGoalReward(),      # ball geometric closeness to opp net
        # --- Positional signals ---
        AlignBallGoal(defense=1.0, offense=1.0),  # be between ball and the right goal
        BallYCoordinateReward(exponent=1),  # ball in opp half = positive
        # --- Engagement signals ---
        FaceBallReward(),                   # orientation toward ball
        TouchBallReward(),                  # big bonus while in contact with ball
        # --- Resource management ---
        SaveBoostReward(),                  # cheap nudge to not waste boost
        # --- Event rewards: rare but high-stakes ---
        EventReward(
            goal=10.0,
            concede=-10.0,
            shot=1.5,
            save=3.0,
            touch=0.05,        # constant small per-touch event (separate from continuous TouchBallReward)
            demo=0.5,
            boost_pickup=0.3,
        ),
    )

    weights = (
        # approach
        0.6,   # velocity_player_to_ball
        0.7,   # liu_distance_player_to_ball
        # offensive
        2.0,   # velocity_ball_to_goal
        1.0,   # liu_distance_ball_to_goal
        # positional
        0.4,   # align_ball_goal (both offense+defense components weighted internally)
        0.5,   # ball_y_coordinate
        # engagement
        0.3,   # face_ball
        5.0,   # touch_ball
        # resource
        0.05,  # save_boost
        # event
        12.0,  # event_reward
    )

    assert len(components) == len(weights), (
        f"reward stack mismatch: {len(components)} components vs {len(weights)} weights"
    )

    combined = CombinedReward(reward_functions=components, reward_weights=weights)

    if zero_sum:
        return ZeroSumReward(combined, team_spirit=team_spirit, opp_scale=opp_scale)
    return combined
