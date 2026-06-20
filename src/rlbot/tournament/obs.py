"""Per-side observation routing so two bots with *different* obs builders can play
in the same 1v1 env.

A normal rlgym_sim env has ONE obs builder feeding both cars, so an 89-dim
DefaultObs bot and a 107-dim AdvancedObs bot cannot share it. `make_env` installs
a PerSideObs that hands each car the obs its own checkpoint was trained on, keyed
by the dims you pass in. This generalises the hardcoded MixedObs in
diego-bots/oldobs_vs_advanced_viewer.py to any (blue_dim, orange_dim) pairing.

Everything here imports rlgym_sim lazily (inside make_env) so the module stays
import-safe for the pure bracket logic and its tests.
"""
from __future__ import annotations

from .roster import OBS_ADVANCED, OBS_DEFAULT

# ~200 simulated seconds max per episode (3000 steps x tick_skip 8 / 120 Hz) so a
# scoreless game still terminates instead of running forever.
TIMEOUT_STEPS = 3000


def make_env(blue_dim: int, orange_dim: int):
    """Build a 1v1 env feeding blue `blue_dim`-obs and orange `orange_dim`-obs.

    Both dims must be 89 (DefaultObs) or 107 (AdvancedObs). DefaultState gives a
    fresh randomly-chosen kickoff each reset, so even deterministic policies play
    distinct games across a best-of-N.
    """
    import rlgym_sim
    from rlgym_sim.utils.obs_builders import AdvancedObs, DefaultObs, ObsBuilder
    from rlgym_sim.utils.reward_functions import DefaultReward
    from rlgym_sim.utils.state_setters import DefaultState
    from rlgym_sim.utils.terminal_conditions.common_conditions import (
        GoalScoredCondition,
        TimeoutCondition,
    )

    from rlbot.actions.lookup_action import LookupAction

    def builder_for(dim: int) -> ObsBuilder:
        if dim == OBS_DEFAULT:
            return DefaultObs()
        if dim == OBS_ADVANCED:
            return AdvancedObs()
        raise ValueError(f"unsupported obs dim {dim} (expected {OBS_DEFAULT} or {OBS_ADVANCED})")

    class PerSideObs(ObsBuilder):
        """team 0 (blue) -> blue builder, team 1 (orange) -> orange builder.

        Returns different-length vectors per car, which is fine: each policy only
        ever consumes its own car's obs.
        """

        def __init__(self) -> None:
            super().__init__()
            self._blue = builder_for(blue_dim)
            self._orange = builder_for(orange_dim)

        def reset(self, initial_state) -> None:
            self._blue.reset(initial_state)
            self._orange.reset(initial_state)

        def build_obs(self, player, state, previous_action):
            builder = self._blue if player.team_num == 0 else self._orange
            return builder.build_obs(player, state, previous_action)

    return rlgym_sim.make(
        tick_skip=8,
        team_size=1,
        spawn_opponents=True,
        obs_builder=PerSideObs(),
        action_parser=LookupAction(),
        reward_fn=DefaultReward(),
        state_setter=DefaultState(),
        terminal_conditions=[GoalScoredCondition(), TimeoutCondition(TIMEOUT_STEPS)],
    )
