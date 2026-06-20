"""Pure-logic tests for match tallying / decision (no env, no torch)."""
from __future__ import annotations

from rlbot.tournament.match import decide, tally_games


def test_tally_counts_wins_draws_and_goal_diff():
    # a perspective: +2, -1, 0, +3, -1  -> a 2 wins, b 2 wins, 1 draw, agg +3
    a_wins, b_wins, draws, agg = tally_games([2, -1, 0, 3, -1])
    assert (a_wins, b_wins, draws, agg) == (2, 2, 1, 3)


def test_decide_by_wins():
    assert decide(3, 2, -5) == "a"   # more wins beats goal diff
    assert decide(1, 4, 10) == "b"


def test_decide_by_goal_diff_when_wins_tied():
    assert decide(2, 2, 4) == "a"
    assert decide(2, 2, -1) == "b"


def test_decide_undecided_when_fully_even():
    assert decide(2, 2, 0) is None
    assert decide(0, 0, 0) is None


def test_best_of_five_clean_sweep():
    a_wins, b_wins, draws, agg = tally_games([1, 1, 1, 1, 1])
    assert decide(a_wins, b_wins, agg) == "a"
    assert (a_wins, b_wins, draws) == (5, 0, 0)


def test_three_two_split_decided_by_wins_not_goals():
    # a wins 3 narrowly, b wins 2 in blowouts -> a still wins the match (more games)
    per_game = [1, 1, 1, -5, -5]
    a_wins, b_wins, _draws, agg = tally_games(per_game)
    assert (a_wins, b_wins, agg) == (3, 2, -7)
    assert decide(a_wins, b_wins, agg) == "a"
