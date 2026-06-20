"""Build a CombinedReward from a YAML reward config.

Config schema (excerpt):
    rewards:
      team_spirit: 0.0
      opp_scale: 1.0
      components:
        - name: velocity_player_to_ball
          weight: 0.05
        - name: velocity_ball_to_goal
          weight: 0.5
        - name: event
          weight: 10.0
          kwargs:
            goal: 1.0
            concede: -1.0
            demo: 0.1
"""

from __future__ import annotations

from typing import Any

from rlbot.rewards.registry import REWARDS
from rlbot.rewards.zero_sum import ZeroSumReward


def build_reward(config: dict[str, Any]):
    """Build the top-level reward function used by the env."""
    from rlgym_sim.utils.reward_functions import CombinedReward

    # Recipe B challenger: Diego's full nexto-style + custom-RL reward stack (vendored
    # from origin/diego). Triggered by `rewards.recipe: nexto_full` in the config.
    if config.get("recipe") == "nexto_full":
        return _build_nexto_full(config)

    components = config.get("components", [])
    if not components:
        raise ValueError("rewards.components must be a non-empty list")

    fns = []
    weights = []
    for spec in components:
        name = spec["name"]
        weight = float(spec.get("weight", 1.0))
        kwargs = spec.get("kwargs", {}) or {}
        cls = REWARDS.get(name)
        fns.append(cls(**kwargs))
        weights.append(weight)

    combined = CombinedReward(reward_functions=tuple(fns), reward_weights=tuple(weights))

    if config.get("zero_sum", True):
        return ZeroSumReward(
            combined,
            team_spirit=float(config.get("team_spirit", 0.0)),
            opp_scale=float(config.get("opp_scale", 1.0)),
        )
    return combined


def _build_nexto_full(config: dict[str, Any]):
    """Diego's full reward recipe (vendored from origin/diego): the 10-component
    nexto-style base + 6 custom RL-physics rewards (Supersonic, AerialBall, AerialTouch,
    BigBoostProximity, BackboardDefense, BallAwayFromOwnGoal), wrapped in ZeroSumReward.
    Weights default to his tuned values; override any via `rewards.custom_weights`.
    Obs/action stay ours, so this is a clean rewards-only A/B vs the champion.
    """
    from rlgym_sim.utils.reward_functions import CombinedReward

    from rlbot.rewards.custom_rl import (
        AerialBallReward,
        AerialTouchReward,
        BackboardDefenseReward,
        BallAwayFromOwnGoalReward,
        BigBoostProximityReward,
        KRCOffensivePotentialReward,
        PossessionEventReward,
        SupersonicReward,
        TouchBallToGoalAccelerationReward,
    )
    from rlbot.rewards.nexto_style import build_nexto_style_reward

    w = config.get("custom_weights", {}) or {}
    # Opt-in (rewards.touch_accel: true): swap the base's flat TouchBall (weight 5,
    # farmable by aimless taps) for Lucy-SKG's outcome-scaled Touch-Ball-to-Goal-
    # Acceleration. Default false -> identical to the original recipe (sweep1/sweep2 unaffected).
    touch_accel = bool(config.get("touch_accel", False))
    # Opt-in (rewards.krc: true): replace the two ADDITIVE position terms (AlignBallGoal in the
    # base + BackboardDefenseReward here) with one KRC geometric-mean Offensive-Potential reward
    # (Lucy-SKG) that can't be farmed by camping. Default false -> identical to the champion.
    krc = bool(config.get("krc", False))
    # Opt-in (rewards.possession: true): add a sparse POSSESSION-EVENT reward (+1 on winning the
    # ball or the kickoff first-touch, zero-summed into a penalty for giving it away). Reward-research
    # anti-passivity lever we had NOT tested (distinct from continuous touch). Default false ->
    # identical to the champion. Weight via custom_weights.possession (default 2.0).
    possession = bool(config.get("possession", False))
    # custom_weights.concede overrides the event concede penalty (default -10, symmetric
    # with goal). Reward-research rank-1 anti-passivity lever: e.g. -7.5 = concede at 75%
    # of goal, so the policy stops over-pricing "don't lose" over "go score".
    nexto_base = build_nexto_style_reward(
        zero_sum=False,
        include_touch=not touch_accel,
        concede=float(w.get("concede", -10.0)),
        include_align=not krc,  # KRC replaces AlignBallGoal
    )

    fns = [
        nexto_base,
        SupersonicReward(),
        AerialBallReward(),
        AerialTouchReward(),
        BigBoostProximityReward(),
    ]
    weights = [
        float(w.get("nexto_base", 1.0)),
        float(w.get("supersonic", 0.05)),
        float(w.get("aerial_ball", 0.5)),
        float(w.get("aerial_touch", 1.5)),
        float(w.get("big_boost", 0.5)),
    ]
    if not krc:  # KRC also replaces the additive BackboardDefense term
        fns.append(BackboardDefenseReward())
        weights.append(float(w.get("backboard_defense", 0.4)))
    fns.append(BallAwayFromOwnGoalReward())
    weights.append(float(w.get("ball_away_own_goal", 0.6)))
    if touch_accel:
        fns.append(TouchBallToGoalAccelerationReward())
        weights.append(float(w.get("touch_accel", 1.5)))
    if krc:
        fns.append(KRCOffensivePotentialReward())
        weights.append(float(w.get("krc_weight", 0.6)))
    if possession:
        fns.append(PossessionEventReward())
        weights.append(float(w.get("possession", 2.0)))

    combined = CombinedReward(reward_functions=tuple(fns), reward_weights=tuple(weights))
    if config.get("zero_sum", True):
        return ZeroSumReward(
            combined,
            team_spirit=float(config.get("team_spirit", 0.0)),
            opp_scale=float(config.get("opp_scale", 1.0)),
        )
    return combined
