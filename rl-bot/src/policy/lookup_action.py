"""Vendored LookupAction table — dependency-free.

Same 90-row table the policy was trained against (rlgym-tools v1, mirrored in
``src/rlbot/actions/lookup_action.py``). Vendored here WITHOUT the rlgym_sim
``ActionParser`` base class / GameState import so the deployment env stays lean
(numpy only). Each row is the 8-dim controller vector:

    [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]

The policy outputs an integer 0..89; ``LOOKUP_TABLE[index]`` is the controls.
This MUST match the training table exactly or action indices map to the wrong
controls.
"""
from __future__ import annotations

import numpy as np


def make_lookup_table() -> np.ndarray:
    actions: list[list[float]] = []
    # Ground
    for throttle in (-1, 0, 1):
        for steer in (-1, 0, 1):
            for boost in (0, 1):
                for handbrake in (0, 1):
                    if boost == 1 and throttle != 1:
                        continue
                    actions.append(
                        [throttle or boost, steer, 0, steer, 0, 0, boost, handbrake]
                    )
    # Aerial
    for pitch in (-1, 0, 1):
        for yaw in (-1, 0, 1):
            for roll in (-1, 0, 1):
                for jump in (0, 1):
                    for boost in (0, 1):
                        if jump == 1 and yaw != 0:
                            continue
                        if pitch == roll == jump == 0:
                            continue
                        handbrake = jump == 1 and (pitch != 0 or yaw != 0 or roll != 0)
                        actions.append(
                            [boost, yaw, pitch, yaw, roll, jump, boost, int(handbrake)]
                        )
    return np.array(actions, dtype=np.float32)


LOOKUP_TABLE = make_lookup_table()  # shape (90, 8)
