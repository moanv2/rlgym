"""KickoffStallCondition -- end episodes where the kickoff never resolves.

Fast-kickoff training translation of "use a short (3-5s) no-touch timeout for
kickoff scenarios" to the rlgym_sim API. The global NoTouchTimeoutCondition
(10s) stays for normal play; this condition specifically kills episodes where
the ball is STILL sitting in the center kickoff zone after T_MAX seconds --
i.e. both cars whiffed, stalled, or perfectly cancelled out. Recycling those
episodes ~2.5x faster multiplies kickoff reps per hour without increasing the
kickoff share of the spawn mix.

Safety property: the condition ARMS only while the ball has never left the
center radius since reset. The moment the ball leaves the zone (kickoff
resolved -- or the episode didn't start as a kickoff at all, e.g. a RandomState
spawn elsewhere), it disarms for the remainder of the episode. This guarantees
it can never terminate normal mid-game play when the ball happens to roll
through midfield.
"""
from __future__ import annotations

from rlgym_sim.utils.gamestates import GameState
from rlgym_sim.utils.terminal_conditions import TerminalCondition


class KickoffStallCondition(TerminalCondition):
    def __init__(
        self,
        max_kickoff_seconds: float = 4.0,
        center_radius: float = 300.0,
        tick_skip: int = 8,
    ):
        super().__init__()
        self._max_steps = int(max_kickoff_seconds * 120 / tick_skip)
        self.center_radius = center_radius
        self._steps = 0
        self._armed = True

    def reset(self, initial_state: GameState) -> None:
        self._steps = 0
        # Arm only for episodes that actually START as a kickoff (ball spawned
        # in the center zone). RandomState spawns elsewhere never arm at all.
        try:
            ball = initial_state.ball.position
            self._armed = (ball[0] ** 2 + ball[1] ** 2) ** 0.5 <= self.center_radius
        except Exception:
            self._armed = False

    def is_terminal(self, current_state: GameState) -> bool:
        if not self._armed:
            return False

        ball = current_state.ball.position
        flat_dist = (ball[0] ** 2 + ball[1] ** 2) ** 0.5
        if flat_dist > self.center_radius:
            # Kickoff resolved (or never was one) -- never fire this episode.
            self._armed = False
            return False

        self._steps += 1
        return self._steps >= self._max_steps
