"""ZeroSumReward — wrap a reward so what blue gains, orange loses.

Adapted from Zealan's RLGym-PPO-Guide. Critical for 1v1 self-play: prevents the agent
from optimizing for shared positive-sum behavior that doesn't translate to wins.

Usage:
    ZeroSumReward(YourOtherReward(), team_spirit=0.0, opp_scale=1.0)

team_spirit (0-1): how much each player shares the team's average reward.
                   1.0 = pure team reward, 0.0 = pure individual.
opp_scale: how strongly to penalize the opponent's gain (usually 1.0).

NOTE: rlgym-sim does not pass `previous_action` to child rewards.
"""
from __future__ import annotations

import numpy as np
from rlgym_sim.utils import RewardFunction
from rlgym_sim.utils.gamestates import GameState, PlayerData


class ZeroSumReward(RewardFunction):
    def __init__(self, child_reward: RewardFunction, team_spirit: float = 0.0, opp_scale: float = 1.0):
        self.child_reward = child_reward
        self.team_spirit = team_spirit
        self.opp_scale = opp_scale
        self._update_next = True
        self._rewards_cache: dict[int, float] = {}

    def reset(self, initial_state: GameState) -> None:
        self.child_reward.reset(initial_state)

    def pre_step(self, state: GameState) -> None:
        self.child_reward.pre_step(state)
        self._update_next = True

    def update(self, state: GameState, is_final: bool) -> None:
        self._rewards_cache = {}

        individual: dict[int, float] = {}
        team_lists: list[list[float]] = [[], []]
        for p in state.players:
            r = (
                self.child_reward.get_final_reward(p, state, None)
                if is_final
                else self.child_reward.get_reward(p, state, None)
            )
            individual[p.car_id] = r
            team_lists[int(p.team_num)].append(r)

        for i in range(2):
            if not team_lists[i]:
                team_lists[i].append(0.0)

        team_avg = np.average(team_lists, axis=1)

        for p in state.players:
            t = int(p.team_num)
            self._rewards_cache[p.car_id] = (
                individual[p.car_id] * (1 - self.team_spirit)
                + team_avg[t] * self.team_spirit
                - team_avg[1 - t] * self.opp_scale
            )

    def _get(self, player: PlayerData, state: GameState, is_final: bool) -> float:
        if self._update_next:
            self.update(state, is_final)
            self._update_next = False
        return self._rewards_cache[player.car_id]

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return self._get(player, state, False)

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return self._get(player, state, True)
