"""DefaultPlusObs: DefaultObs's exact 89 features + 18 new relative features = 107.

The first 89 outputs are byte-for-byte identical to DefaultObs (we just call
super().build_obs), so a policy trained on DefaultObs(89) can be transplanted
into a 107-input net by zero-padding the 18 new input columns — the original
89 weights still read the exact same features.

The 18 extras are ego-relative geometry (pos/vel to ball & opponent + a few
scalars) — info DefaultObs only encodes absolutely, so it's genuinely new.
"""
from __future__ import annotations

import numpy as np
from rlgym_sim.utils.obs_builders import DefaultObs

_POS = 2300.0
_VEL = 2300.0


class DefaultPlusObs(DefaultObs):
    def build_obs(self, player, state, previous_action):
        base = np.asarray(super().build_obs(player, state, previous_action), dtype=np.float32)

        car = player.car_data
        ball = state.ball
        opp = next((p for p in state.players if p.team_num != player.team_num), player)
        oc = opp.car_data

        rel_ball_p = (ball.position - car.position) / _POS
        rel_ball_v = (ball.linear_velocity - car.linear_velocity) / _VEL
        rel_opp_p = (oc.position - car.position) / _POS
        rel_opp_v = (oc.linear_velocity - car.linear_velocity) / _VEL

        to_ball = ball.position - car.position
        to_opp = oc.position - car.position
        d_ball = float(np.linalg.norm(to_ball)) / 5000.0
        d_opp = float(np.linalg.norm(to_opp)) / 5000.0
        car_spd = float(np.linalg.norm(car.linear_velocity)) / 2300.0
        ball_spd = float(np.linalg.norm(ball.linear_velocity)) / 6000.0
        fwd = car.forward()
        align_ball = float(np.dot(fwd, to_ball / (np.linalg.norm(to_ball) + 1e-6)))
        align_opp = float(np.dot(fwd, to_opp / (np.linalg.norm(to_opp) + 1e-6)))

        extra = np.concatenate([
            rel_ball_p, rel_ball_v, rel_opp_p, rel_opp_v,
            [d_ball, d_opp, car_spd, ball_spd, align_ball, align_opp],
        ]).astype(np.float32)
        return np.concatenate([base, extra]).astype(np.float32)
