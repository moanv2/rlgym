"""LookupAction — fully-discrete action parser with ~90 useful input permutations.

Vendored from rlgym-tools v1 (the v2 ecosystem moved this file). Stable, well-tested
across the bot community. The action space is a single Discrete head; each index
maps to an 8-dim controller vector: [throttle, steer, pitch, yaw, roll, jump, boost, handbrake].
"""
from __future__ import annotations

import numpy as np
from gym.spaces import Discrete
from rlgym_sim.utils.action_parsers import ActionParser
from rlgym_sim.utils.gamestates import GameState


class LookupAction(ActionParser):
    def __init__(self):
        super().__init__()
        self._lookup_table = self.make_lookup_table()

    @staticmethod
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

    def get_action_space(self) -> Discrete:
        return Discrete(len(self._lookup_table))

    def parse_actions(self, actions: np.ndarray, state: GameState) -> np.ndarray:
        actions = np.asarray(actions).astype(int).flatten()
        return self._lookup_table[actions]
