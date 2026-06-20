"""Run one bracket match (best-of-N with side-swapping) and decide the winner.

`tally_games` / `decide` are pure and unit-tested. `play_match` / `play_game` run
the actual rlgym_sim env and import torch lazily, so importing this module is cheap.

Match rules
-----------
- best-of-N (default 5), DETERMINISTIC (argmax) -- the mode Martin's champion is
  strongest in, and reproducible for the video replays.
- Sides swap every game so neither bot gets a permanent blue/orange advantage.
- DefaultState randomises the kickoff each game, so deterministic policies still
  produce N distinct games.
- Winner = more games won; tie -> aggregate goal differential; still tied -> up to
  `sudden_death_cap` STOCHASTIC decider games; still tied -> higher seed (more
  timesteps) wins, recorded as decided_by="seed".
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class MatchOutcome:
    """Result of one best-of-N match, from bot_a's perspective. JSON-friendly via to_detail()."""

    a_owner: str
    b_owner: str
    a_wins: int
    b_wins: int
    draws: int
    a_goal_diff: int                    # aggregate signed goals (+ = a scored more overall)
    games: list[int] = field(default_factory=list)  # per-game goal diff from a's perspective
    winner_owner: str = ""
    decided_by: str = "games"           # games | goal_diff | sudden_death | seed

    def to_detail(self) -> dict:
        return {
            "score": f"{self.a_wins}-{self.b_wins}" + (f" ({self.draws}D)" if self.draws else ""),
            "a_owner": self.a_owner,
            "b_owner": self.b_owner,
            "a_wins": self.a_wins,
            "b_wins": self.b_wins,
            "draws": self.draws,
            "a_goal_diff": self.a_goal_diff,
            "games": self.games,
            "decided_by": self.decided_by,
        }


def tally_games(per_game_a_diff: list[int]) -> tuple[int, int, int, int]:
    """Return (a_wins, b_wins, draws, aggregate_goal_diff) from a's-perspective diffs."""
    a_wins = sum(1 for d in per_game_a_diff if d > 0)
    b_wins = sum(1 for d in per_game_a_diff if d < 0)
    draws = sum(1 for d in per_game_a_diff if d == 0)
    return a_wins, b_wins, draws, sum(per_game_a_diff)


def decide(a_wins: int, b_wins: int, agg_goal_diff: int) -> str | None:
    """'a' / 'b' winner, or None if undecided (equal wins AND equal goals)."""
    if a_wins != b_wins:
        return "a" if a_wins > b_wins else "b"
    if agg_goal_diff != 0:
        return "a" if agg_goal_diff > 0 else "b"
    return None


# ── heavy path (real env) ──────────────────────────────────────────────────────

def play_game(env, blue_policy, orange_policy, *, deterministic: bool,
              render: bool = False, step_delay: float = 0.006) -> int:
    """Play one episode to termination. Returns info['result'] (+ blue, - orange, 0 draw)."""
    import torch

    from .policy_io import action_to_int

    obs_list = env.reset()
    blue_obs, orange_obs = obs_list[0], obs_list[1]
    done = False
    info: dict = {}
    while not done:
        with torch.no_grad():
            b_act, _ = blue_policy.get_action(blue_obs, deterministic=deterministic)
            o_act, _ = orange_policy.get_action(orange_obs, deterministic=deterministic)
        obs_list, _r, done, info = env.step([action_to_int(b_act), action_to_int(o_act)])
        blue_obs, orange_obs = obs_list[0], obs_list[1]
        if render:
            env.render()
            time.sleep(step_delay)
    return int(info.get("result", 0))


def play_match(bot_a, bot_b, *, games: int = 5, deterministic: bool = True,
               render: bool = False, step_delay: float = 0.006,
               sudden_death_cap: int = 4) -> MatchOutcome:
    """Play a best-of-`games` match between two roster.Bot objects.

    Envs are built lazily and cached per (blue_dim, orange_dim) side-assignment, so
    each match builds at most two rlgym_sim envs regardless of game count.
    """
    from .obs import make_env
    from .policy_io import load_policy

    pol_a = load_policy(bot_a.checkpoint, bot_a.obs_dim)
    pol_b = load_policy(bot_b.checkpoint, bot_b.obs_dim)

    env_cache: dict[tuple[int, int], object] = {}

    def env_for(blue_dim: int, orange_dim: int):
        key = (blue_dim, orange_dim)
        if key not in env_cache:
            env_cache[key] = make_env(blue_dim, orange_dim)
        return env_cache[key]

    def one_game(a_is_blue: bool, deterministic_: bool) -> int:
        """Return goal diff from a's perspective for a single game."""
        if a_is_blue:
            env = env_for(bot_a.obs_dim, bot_b.obs_dim)
            result = play_game(env, pol_a, pol_b, deterministic=deterministic_,
                               render=render, step_delay=step_delay)
            return result               # + means blue(=a) scored more
        env = env_for(bot_b.obs_dim, bot_a.obs_dim)
        result = play_game(env, pol_b, pol_a, deterministic=deterministic_,
                           render=render, step_delay=step_delay)
        return -result                  # + result means blue(=b) scored more -> bad for a

    try:
        per_game: list[int] = []
        for g in range(games):
            per_game.append(one_game(a_is_blue=(g % 2 == 0), deterministic_=deterministic))

        a_wins, b_wins, draws, agg = tally_games(per_game)
        decided_by = "games" if a_wins != b_wins else "goal_diff"
        verdict = decide(a_wins, b_wins, agg)

        # Sudden death: stochastic deciders to break a dead-even best-of-N.
        sd = 0
        while verdict is None and sd < sudden_death_cap:
            per_game.append(one_game(a_is_blue=(sd % 2 == 0), deterministic_=False))
            a_wins, b_wins, draws, agg = tally_games(per_game)
            verdict = decide(a_wins, b_wins, agg)
            decided_by = "sudden_death"
            sd += 1

        if verdict is None:  # truly inseparable -> higher seed (more timesteps) wins
            verdict = "a" if bot_a.timesteps >= bot_b.timesteps else "b"
            decided_by = "seed"
    finally:
        for env in env_cache.values():
            env.close()

    winner = bot_a if verdict == "a" else bot_b
    return MatchOutcome(
        a_owner=bot_a.owner, b_owner=bot_b.owner,
        a_wins=a_wins, b_wins=b_wins, draws=draws, a_goal_diff=agg,
        games=per_game, winner_owner=winner.owner, decided_by=decided_by,
    )


def make_play_match_fn(*, games: int = 5, deterministic: bool = True):
    """Adapter producing the (bot_a, bot_b, round_name) -> (winner, detail) callable
    that bracket.run_bracket expects."""

    def _fn(bot_a, bot_b, round_name: str):
        print(f"\n=== {round_name.upper()}: {bot_a.owner} vs {bot_b.owner} "
              f"(best of {games}, {'deterministic' if deterministic else 'stochastic'}) ===")
        outcome = play_match(bot_a, bot_b, games=games, deterministic=deterministic)
        winner = bot_a if outcome.winner_owner == bot_a.owner else bot_b
        print(f"--> {outcome.winner_owner} wins {outcome.a_wins}-{outcome.b_wins} "
              f"(by {outcome.decided_by})")
        return winner, outcome.to_detail()

    return _fn
