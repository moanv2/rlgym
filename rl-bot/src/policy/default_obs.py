"""Vendored DefaultObs — byte-for-byte the observation the policy trained on.

This reproduces ``rlgym_sim.utils.obs_builders.DefaultObs.build_obs`` exactly,
but operating on rlgym-compat's ``V1GameState`` / ``V1PlayerData`` objects
(which mirror the rlgym_sim GameState API: ``car_data.forward()/up()``,
``.position``, ``.linear_velocity``, ``.angular_velocity``, ``state.ball`` /
``inverted_ball``, ``state.boost_pads`` / ``inverted_boost_pads``,
``player.boost_amount / on_ground / has_flip / is_demoed``).

Vendored (rather than importing rlgym_sim) so the deployment env needs only
numpy + torch + rlgym-compat, not the full rocketsim training stack.

For 1v1 (team_size=1) the vector is 89-dim:
    ball pos/lin/ang (9) + previous_action (8) + boost_pads (34)
    + self block (19) + one-enemy block (19) = 89.

The normalization coefficients are DefaultObs's defaults and MUST stay
identical to training (standardize_obs was False during training, so these
coefficients are the only scaling the network ever saw).
"""
from __future__ import annotations

import math
from typing import List

import numpy as np

ORANGE_TEAM = 1

POS_COEF = 1 / 2300
ANG_COEF = 1 / math.pi
LIN_VEL_COEF = 1 / 2300
ANG_VEL_COEF = 1 / math.pi


def _add_player_to_obs(obs: List, player, inverted: bool) -> None:
    player_car = player.inverted_car_data if inverted else player.car_data
    obs.extend(
        [
            player_car.position * POS_COEF,
            player_car.forward(),
            player_car.up(),
            player_car.linear_velocity * LIN_VEL_COEF,
            player_car.angular_velocity * ANG_VEL_COEF,
            [
                player.boost_amount,
                int(player.on_ground),
                int(player.has_flip),
                int(player.is_demoed),
            ],
        ]
    )


def build_obs(player, state, previous_action: np.ndarray) -> np.ndarray:
    """Build the 89-dim observation for ``player`` from a V1GameState.

    ``previous_action`` is the 8-dim controller vector the policy emitted last
    decision (zeros on the first tick) — DefaultObs feeds it back as part of
    the observation, so we must track and pass it.
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
        ball.position * POS_COEF,
        ball.linear_velocity * LIN_VEL_COEF,
        ball.angular_velocity * ANG_VEL_COEF,
        previous_action,
        pads,
    ]

    _add_player_to_obs(obs, player, inverted)

    allies: List = []
    enemies: List = []
    for other in state.players:
        if other.car_id == player.car_id:
            continue
        team_obs = allies if other.team_num == player.team_num else enemies
        _add_player_to_obs(team_obs, other, inverted)

    obs.extend(allies)
    obs.extend(enemies)
    return np.concatenate(obs)
