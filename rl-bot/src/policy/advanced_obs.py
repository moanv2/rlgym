"""Vendored AdvancedObs -- the 107-dim observation papaya trained on.

Reproduces the canonical AdvancedObs (== rlgym_sim's built-in AdvancedObs, which
is numerically identical to the custom rlbot.obs.advanced_obs.AdvancedObs papaya
trained against -- verified to ~1e-7) but operating on rlgym-compat's
``V1GameState`` / ``V1PlayerData`` objects, exactly like ``default_obs.py`` does
for DefaultObs.

Vendored (rather than importing rlgym_sim) so the deployment env needs only
numpy + torch + rlgym-compat, not the full rocketsim training stack.

For 1v1 (team_size=1) the vector is 107-dim:
    ball pos/lin/ang (9) + previous_action (8) + boost_pads (34)
    + self block (25) + one-enemy block (31) = 107.
Versus DefaultObs, each car block adds the car->ball RELATIVE position+velocity,
and the enemy block additionally adds the enemy's pos+vel relative to the
observing car.

Normalization MUST match training (POS_STD=2300, ANG_STD=pi); standardize_obs was
False during training, so these coefficients are the only scaling the net saw.
"""
from __future__ import annotations

import math
from typing import List

import numpy as np

ORANGE_TEAM = 1

POS_STD = 2300.0
ANG_STD = math.pi


def _add_player_to_obs(obs: List, player, ball, inverted: bool):
    """Append a car block (with car->ball relative features) and return the car
    data so the caller can compute enemy-relative-to-self features."""
    player_car = player.inverted_car_data if inverted else player.car_data

    rel_pos = ball.position - player_car.position
    rel_vel = ball.linear_velocity - player_car.linear_velocity

    obs.extend(
        [
            rel_pos / POS_STD,
            rel_vel / POS_STD,
            player_car.position / POS_STD,
            player_car.forward(),
            player_car.up(),
            player_car.linear_velocity / POS_STD,
            player_car.angular_velocity / ANG_STD,
            [
                player.boost_amount,
                int(player.on_ground),
                int(player.has_flip),
                int(player.is_demoed),
            ],
        ]
    )
    return player_car


def build_obs(player, state, previous_action: np.ndarray) -> np.ndarray:
    """Build the 107-dim AdvancedObs vector for ``player`` from a V1GameState.

    ``previous_action`` is the 8-dim controller vector the policy emitted last
    decision (zeros on the first tick), fed back as part of the observation.
    """
    if player.team_num == ORANGE_TEAM:
        inverted = True
        ball = state.inverted_ball
        pads = state.inverted_boost_pads
    else:
        inverted = False
        ball = state.ball
        pads = state.boost_pads

    obs = [
        ball.position / POS_STD,
        ball.linear_velocity / POS_STD,
        ball.angular_velocity / ANG_STD,
        previous_action,
        pads,
    ]

    player_car = _add_player_to_obs(obs, player, ball, inverted)

    allies: List = []
    enemies: List = []
    for other in state.players:
        if other.car_id == player.car_id:
            continue
        team_obs = allies if other.team_num == player.team_num else enemies
        other_car = _add_player_to_obs(team_obs, other, ball, inverted)
        team_obs.extend(
            [
                (other_car.position - player_car.position) / POS_STD,
                (other_car.linear_velocity - player_car.linear_velocity) / POS_STD,
            ]
        )

    obs.extend(allies)
    obs.extend(enemies)
    return np.concatenate(obs)
