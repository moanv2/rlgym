"""Custom wandb metrics for the curriculum — boost economy + scoring/defense.

rlgym_ppo logs PPO internals (reward, entropy, losses) out of the box but nothing
about *what the bot is actually doing* in-game. exp_008 (boost economy) needs to
watch boost-held and scoring move together to tune the two boost levers, and those
panels don't exist yet. This adds them.

How rlgym_ppo metrics logging works (so the bookkeeping below makes sense):
  - ``_collect_metrics(game_state)`` runs **per step inside each env worker**. The
    logger instance is pickled into every worker, so each worker keeps its OWN
    copy of the per-step event state below — which is exactly what we need to
    detect goals/shots/saves as score/stat *increments* along that worker's stream.
  - ``_report_metrics(...)`` runs **once per iteration on the learner** with every
    collected step's vector, and writes the aggregate to wandb.

Logging contract: the learner calls our ``report_metrics`` immediately BEFORE its
own ``reporting.report_metrics`` in the same loop iteration, and rlgym_ppo logs
with an auto-incrementing wandb step. So we log with ``commit=False`` to merge our
panels onto the SAME step as the main report (otherwise our series would land on
alternating steps and never line up on the Cumulative-Timesteps x-axis).
"""
from __future__ import annotations

import numpy as np
from rlgym_ppo.util.metrics_logger import MetricsLogger

# Indices into the per-step metric vector returned by ``_collect_metrics``.
_AVG_BOOST = 0      # mean boost_amount across players, [0, 1]
_FRAC_EMPTY = 1     # fraction of players below the "empty" threshold this step
_BALL_HEIGHT = 2    # ball z (uu) — proxy for aerial / above-ground play
_FRAC_AIR = 3       # fraction of players off the ground this step
_GOALS = 4          # goals scored this step (both teams)
_SHOTS = 5          # shots registered this step (all players)
_SAVES = 6          # saves registered this step (all players)
_N_METRICS = 7


class BotMetricsLogger(MetricsLogger):
    """Track boost economy and scoring/defense rates and log them to wandb.

    :param empty_threshold: boost fraction below which a car counts as "running
        empty" (default 0.10 ≈ 10 boost) — the state exp_008 is trying to eliminate.
    """

    def __init__(self, empty_threshold: float = 0.10) -> None:
        super().__init__()
        self.empty_threshold = float(empty_threshold)
        # Per-worker event state — detects goals/shots/saves as increments. Each
        # worker process gets its own unpickled copy, so these never collide.
        self._prev_blue = 0
        self._prev_orange = 0
        self._prev_shots: dict[int, int] = {}
        self._prev_saves: dict[int, int] = {}

    def _collect_metrics(self, game_state) -> list[np.ndarray]:
        players = game_state.players
        n = max(len(players), 1)

        boosts = [float(p.boost_amount) for p in players]
        avg_boost = sum(boosts) / n
        frac_empty = sum(1 for b in boosts if b < self.empty_threshold) / n
        frac_air = sum(1 for p in players if not p.on_ground) / n
        ball_height = float(game_state.ball.position[2])

        # Goals: positive increments of either team's score. A reset drops the score
        # to 0 (negative delta) and is correctly ignored.
        blue = int(game_state.blue_score)
        orange = int(game_state.orange_score)
        goals = max(0, blue - self._prev_blue) + max(0, orange - self._prev_orange)
        self._prev_blue, self._prev_orange = blue, orange

        # Shots/saves: positive increments of each car's match stat. ``.get`` defaults
        # to the current value the first time a car is seen → 0 delta, no false event.
        shots = 0
        saves = 0
        for p in players:
            cid = p.car_id
            s_now, v_now = int(p.match_shots), int(p.match_saves)
            shots += max(0, s_now - self._prev_shots.get(cid, s_now))
            saves += max(0, v_now - self._prev_saves.get(cid, v_now))
            self._prev_shots[cid] = s_now
            self._prev_saves[cid] = v_now

        vec = np.empty(_N_METRICS, dtype=np.float32)
        vec[_AVG_BOOST] = avg_boost
        vec[_FRAC_EMPTY] = frac_empty
        vec[_BALL_HEIGHT] = ball_height
        vec[_FRAC_AIR] = frac_air
        vec[_GOALS] = goals
        vec[_SHOTS] = shots
        vec[_SAVES] = saves
        return [vec]

    def _report_metrics(self, collected_metrics, wandb_run, cumulative_timesteps) -> None:
        if wandb_run is None or not collected_metrics:
            return

        # Each collected step deserializes back to ``[vec]`` (the list we returned).
        rows = np.asarray([m[0] for m in collected_metrics], dtype=np.float64)
        n_steps = len(rows)
        if n_steps == 0:
            return

        means = rows.mean(axis=0)
        sums = rows.sum(axis=0)
        per_1k = 1000.0 / n_steps  # event counts -> events per 1k steps

        report = {
            "boost/avg_held_pct": float(means[_AVG_BOOST] * 100.0),
            "boost/pct_steps_empty": float(means[_FRAC_EMPTY] * 100.0),
            "play/avg_ball_height": float(means[_BALL_HEIGHT]),
            "play/pct_airborne": float(means[_FRAC_AIR] * 100.0),
            "score/goals_per_1k_steps": float(sums[_GOALS] * per_1k),
            "score/shots_per_1k_steps": float(sums[_SHOTS] * per_1k),
            "score/saves_per_1k_steps": float(sums[_SAVES] * per_1k),
        }
        # commit=False: merge onto the same wandb step as the learner's own report,
        # which is logged right after this call in the same iteration.
        wandb_run.log(report, commit=False)
