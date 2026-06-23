"""rlgym_sim-compatible AdvancedObs.

Why this exists
---------------
rlgym_tools 2.6.4 (the version pinned in rlbot310) does NOT ship a drop-in
AdvancedObs for the rlgym_sim API. Its only relative-style builder,
RelativeDefaultObs, subclasses the RLGym 2.0 `rlgym.api` ObsBuilder, which has a
completely different `build_obs` signature and is NOT a subclass of
`rlgym_sim.utils.obs_builders.ObsBuilder` — so it crashes inside
`rlgym_sim.make(obs_builder=...)`. (This is why `configs/experiments/
exp_002_advanced_obs.yaml` silently fell back to `name: default`.)

This is the classic community AdvancedObs, ported to subclass rlgym_sim's
ObsBuilder so it Just Works in `rlgym_sim.make()`. Compared to DefaultObs it
adds, per car, the RELATIVE position and velocity to the ball, and per OTHER
car the relative position and velocity to the observing car. Relative features
let the policy reason about "where is the ball/opponent w.r.t. me" without
having to subtract two absolute vectors itself.

Obs size (1v1, team_size=1):
    ball(9) + prev_action(8) + pads(34)
    + self  (rel_ball_pos 3 + rel_ball_vel 3 + pos 3 + fwd 3 + up 3
             + linvel 3 + angvel 3 + misc 4 = 25)
    + enemy (same 25 + rel_to_self_pos 3 + rel_to_self_vel 3 = 31)
    = 107   (vs DefaultObs = 89)

NOTE: changing obs builder changes the input shape, so a policy trained with
AdvancedObs cannot play / be evaluated against a DefaultObs policy (and vice
versa) without a size mismatch. This is intentional and known.
"""
from __future__ import annotations

import math
from typing import Any, List

import numpy as np

from rlgym_sim.utils import common_values
from rlgym_sim.utils.gamestates import GameState, PlayerData
from rlgym_sim.utils.obs_builders import ObsBuilder


class AdvancedObs(ObsBuilder):
    # Normalisation constants. POS_STD roughly covers the field half-diagonal /
    # max car speed; ANG_STD normalises angular velocity to ~[-1, 1].
    POS_STD = 2300.0
    ANG_STD = math.pi

    def __init__(self):
        super().__init__()

    def reset(self, initial_state: GameState) -> None:
        # Stateless obs — nothing to reset between episodes.
        pass

    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> Any:
        # Orange sees a mirrored (inverted) world so the policy is team-agnostic:
        # it always plays as if attacking the same goal.
        if player.team_num == common_values.ORANGE_TEAM:
            inverted = True
            ball = state.inverted_ball
            pads = state.inverted_boost_pads
        else:
            inverted = False
            ball = state.ball
            pads = state.boost_pads

        obs = [
            ball.position * (1.0 / self.POS_STD),
            ball.linear_velocity * (1.0 / self.POS_STD),
            ball.angular_velocity * (1.0 / self.ANG_STD),
            previous_action,
            pads,
        ]

        player_car = self._add_player_to_obs(obs, player, ball, inverted)

        allies: List = []
        enemies: List = []

        for other in state.players:
            if other.car_id == player.car_id:
                continue

            team_obs = allies if other.team_num == player.team_num else enemies
            other_car = self._add_player_to_obs(team_obs, other, ball, inverted)

            # Relative position / velocity of the other car w.r.t. THIS car.
            team_obs.extend([
                (other_car.position - player_car.position) * (1.0 / self.POS_STD),
                (other_car.linear_velocity - player_car.linear_velocity) * (1.0 / self.POS_STD),
            ])

        obs.extend(allies)
        obs.extend(enemies)

        return np.concatenate(obs)

    def _add_player_to_obs(self, obs: List, player: PlayerData, ball, inverted: bool):
        player_car = player.inverted_car_data if inverted else player.car_data

        # Relative-to-ball features — the core "advanced" addition over DefaultObs.
        rel_pos = ball.position - player_car.position
        rel_vel = ball.linear_velocity - player_car.linear_velocity

        obs.extend([
            rel_pos * (1.0 / self.POS_STD),
            rel_vel * (1.0 / self.POS_STD),
            player_car.position * (1.0 / self.POS_STD),
            player_car.forward(),
            player_car.up(),
            player_car.linear_velocity * (1.0 / self.POS_STD),
            player_car.angular_velocity * (1.0 / self.ANG_STD),
            [
                player.boost_amount,
                int(player.on_ground),
                int(player.has_flip),
                int(player.is_demoed),
            ],
        ])

        return player_car
