"""Pure-logic tests for the single-elim bracket and ranking.

These import only rlbot.tournament.bracket (no torch / rlgym_sim), so they run
anywhere pytest runs. The match runner is faked: the higher-seeded (more timesteps)
bot always wins, which makes outcomes fully deterministic and easy to assert.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from rlbot.tournament.bracket import next_pow2, run_bracket, seed_entrants, seed_order


@dataclass(frozen=True)
class FakeBot:
    owner: str
    timesteps: int
    name: str = ""

    def __post_init__(self):
        if not self.name:
            object.__setattr__(self, "name", self.owner)


def make_bots(n: int) -> list[FakeBot]:
    # owner b0 strongest ... b{n-1} weakest
    return [FakeBot(f"b{i}", timesteps=(n - i) * 1000) for i in range(n)]


def higher_seed_wins(a: FakeBot, b: FakeBot, round_name: str):
    winner = a if a.timesteps >= b.timesteps else b
    return winner, {"score": "5-0", "round": round_name}


def test_next_pow2():
    assert [next_pow2(i) for i in (1, 2, 3, 4, 5, 8, 9)] == [1, 2, 4, 4, 8, 8, 16]


def test_seed_order_standard_brackets():
    assert seed_order(2) == [1, 2]
    assert seed_order(4) == [1, 4, 2, 3]
    assert seed_order(8) == [1, 8, 4, 5, 2, 7, 3, 6]


def test_seed_order_rejects_non_power_of_two():
    with pytest.raises(ValueError):
        seed_order(5)


def test_seed_entrants_strongest_first():
    bots = [FakeBot("weak", 10), FakeBot("strong", 100), FakeBot("mid", 50)]
    assert [b.owner for b in seed_entrants(bots)] == ["strong", "mid", "weak"]


def test_two_bots_is_a_single_final():
    bots = make_bots(2)
    res = run_bracket(bots, higher_seed_wins)
    assert res["champion"] == "b0"
    assert res["ranking"][0]["owner"] == "b0"
    assert res["ranking"][1]["owner"] == "b1"
    # exactly one round, named 'final'
    assert [r["round"] for r in res["rounds"]] == ["final"]


def test_three_bots_top_seed_gets_bye():
    bots = make_bots(3)
    res = run_bracket(bots, higher_seed_wins)
    assert res["bracket_size"] == 4
    # b0 (seed 1) byes the semifinal; b1 vs b2 is contested.
    first_round = res["rounds"][0]
    byes = [m for m in first_round["matches"] if m["type"] == "bye"]
    assert [m["advanced"] for m in byes] == ["b0"]
    assert res["champion"] == "b0"


def test_five_bots_full_ranking_and_third_place():
    bots = make_bots(5)
    res = run_bracket(bots, higher_seed_wins)
    assert res["bracket_size"] == 8
    assert res["champion"] == "b0"
    # higher seed always wins -> strict seed order ranking.
    positions = [(r["position"], r["owner"]) for r in res["ranking"]]
    assert positions == [(1, "b0"), (2, "b1"), (3, "b2"), (4, "b3"), (5, "b4")]
    # a 3rd-place playoff must have happened between the two semifinal losers.
    assert res["third_place"] is not None
    assert {res["third_place"]["winner"], res["third_place"]["loser"]} == {"b2", "b3"}


def test_ranking_is_complete_and_unique():
    for n in (2, 3, 4, 5, 6, 7, 8):
        res = run_bracket(make_bots(n), higher_seed_wins)
        positions = sorted(r["position"] for r in res["ranking"])
        owners = {r["owner"] for r in res["ranking"]}
        assert positions == list(range(1, n + 1)), f"n={n} positions {positions}"
        assert len(owners) == n, f"n={n} duplicate owners"


def test_upset_changes_champion():
    # Lowest seed always wins -> the weakest bot should take the title.
    def lower_seed_wins(a, b, round_name):
        winner = a if a.timesteps <= b.timesteps else b
        return winner, {"round": round_name}

    res = run_bracket(make_bots(4), lower_seed_wins)
    assert res["champion"] == "b3"  # weakest


def test_needs_two_entrants():
    with pytest.raises(ValueError):
        run_bracket(make_bots(1), higher_seed_wins)
