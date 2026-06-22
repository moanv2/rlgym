"""Single-elimination bracket logic and final ranking — pure, no torch/env.

The match runner is *injected* (`play_match_fn`), so this whole module is unit
testable with a fake that just compares seeds. `run_bracket` returns a structure
that serialises cleanly to JSON for the presentation.

Seeding & byes
--------------
Entrants are seeded 1..N by strength (most timesteps = seed 1). The bracket is
padded up to the next power of two; the (P - N) missing seeds are byes that always
land on the *top* seeds, via the standard bracket order:
    size 2 -> [1, 2]
    size 4 -> [1, 4, 2, 3]
    size 8 -> [1, 8, 4, 5, 2, 7, 3, 6]
So with 5 bots (padded to 8) only the 4-vs-5 quarterfinal is actually contested;
seeds 1/2/3 get byes.

Ranking
-------
1st = final winner, 2nd = final loser. The two semifinal losers contest a
3rd-place playoff -> 3rd / 4th. Anyone knocked out earlier is ranked by how far
they got (later round = better), breaking ties by seed.
"""
from __future__ import annotations

from collections.abc import Callable

# play_match_fn(bot_a, bot_b, round_name) -> (winner, detail_dict)
PlayMatchFn = Callable[[object, object, str], "tuple[object, dict]"]


def next_pow2(n: int) -> int:
    """Smallest power of two >= n (>=1)."""
    if n < 1:
        raise ValueError("need at least 1 entrant")
    p = 1
    while p < n:
        p *= 2
    return p


def seed_order(size: int) -> list[int]:
    """Standard single-elim seed positions for a power-of-two bracket.

    seed_order(4) -> [1, 4, 2, 3];  seed_order(8) -> [1, 8, 4, 5, 2, 7, 3, 6].
    """
    if size < 1 or (size & (size - 1)) != 0:
        raise ValueError(f"size must be a power of two, got {size}")
    order = [1]
    while len(order) < size:
        n = len(order) * 2
        nxt: list[int] = []
        for x in order:
            nxt.append(x)
            nxt.append(n + 1 - x)
        order = nxt
    return order


def _round_name(num_slots: int) -> str:
    """Human label for a round with `num_slots` participants entering it."""
    return {2: "final", 4: "semifinal", 8: "quarterfinal", 16: "round-of-16"}.get(
        num_slots, f"round-of-{num_slots}"
    )


def seed_entrants(entrants: list, key=lambda b: b.timesteps) -> list:
    """Order entrants strongest-first (seed 1 = highest key), deterministic ties."""
    return sorted(entrants, key=lambda b: (-key(b), b.owner))


def run_bracket(entrants: list, play_match_fn: PlayMatchFn) -> dict:
    """Run a seeded single-elim bracket. Returns a JSON-serialisable result dict.

    `entrants` is any list of objects with at least `.owner` and `.timesteps`
    (e.g. roster.Bot). `play_match_fn(a, b, round_name)` must return
    (winner, detail) where winner is `a` or `b` (object identity) and detail is a
    JSON-friendly dict describing the games played.
    """
    if len(entrants) < 2:
        raise ValueError(f"need >=2 entrants to run a bracket, got {len(entrants)}")

    seeds = seed_entrants(entrants)
    n = len(seeds)
    size = next_pow2(n)
    # seat[position] = bot or None(bye). seed s (1-indexed) -> seeds[s-1] if present.
    seats = [seeds[s - 1] if s <= n else None for s in seed_order(size)]

    eliminated_in: dict[int, str] = {}   # id(bot) -> round_name it lost in
    rounds: list[dict] = []
    semifinal_losers: list = []
    final_match: dict | None = None

    current = seats
    while len(current) > 1:
        rname = _round_name(len(current))
        winners: list = []
        match_records: list[dict] = []
        for i in range(0, len(current), 2):
            a, b = current[i], current[i + 1]
            if a is None and b is None:
                winners.append(None)
                continue
            if a is None or b is None:
                winners.append(a or b)  # bye: the present bot advances
                match_records.append({
                    "round": rname, "type": "bye",
                    "advanced": (a or b).owner,
                })
                continue
            winner, detail = play_match_fn(a, b, rname)
            loser = b if winner is a else a
            eliminated_in[id(loser)] = rname
            if rname == "semifinal":
                semifinal_losers.append(loser)
            winners.append(winner)
            rec = {"round": rname, "type": "match",
                   "winner": winner.owner, "loser": loser.owner, **detail}
            match_records.append(rec)
            if rname == "final":
                final_match = rec
        rounds.append({"round": rname, "matches": match_records})
        current = winners

    champion = current[0]

    # 3rd-place playoff between the two semifinal losers (if any).
    third_place: dict | None = None
    if len(semifinal_losers) == 2:
        w, detail = play_match_fn(semifinal_losers[0], semifinal_losers[1], "third-place")
        loser = semifinal_losers[1] if w is semifinal_losers[0] else semifinal_losers[0]
        third_place = {"round": "third-place", "type": "match",
                       "winner": w.owner, "loser": loser.owner, **detail}

    ranking = _rank(seeds, champion, final_match, semifinal_losers, third_place, eliminated_in)

    return {
        "n_entrants": n,
        "bracket_size": size,
        "seeds": [{"seed": i + 1, "owner": b.owner, "name": getattr(b, "name", b.owner),
                   "timesteps": getattr(b, "timesteps", 0)} for i, b in enumerate(seeds)],
        "rounds": rounds,
        "third_place": third_place,
        "champion": champion.owner,
        "ranking": ranking,
    }


_ROUND_DEPTH = {"final": 100, "semifinal": 90, "quarterfinal": 80,
                "round-of-16": 70, "round-of-32": 60}


def _rank(seeds, champion, final_match, semifinal_losers, third_place, eliminated_in) -> list[dict]:
    """Assemble the 1..N ranking from bracket outcomes."""
    seed_of = {id(b): i + 1 for i, b in enumerate(seeds)}
    placed: list[tuple[int, object]] = []  # (position, bot)

    placed.append((1, champion))
    if final_match is not None:
        runner_up = next(b for b in seeds if b.owner == final_match["loser"])
        placed.append((2, runner_up))

    if third_place is not None:
        third = next(b for b in seeds if b.owner == third_place["winner"])
        fourth = next(b for b in seeds if b.owner == third_place["loser"])
        placed.append((3, third))
        placed.append((4, fourth))
    else:
        # No 3rd-place game (e.g. only 2 entrants): semifinal losers (if any) tie next.
        for b in sorted(semifinal_losers, key=lambda b: seed_of[id(b)]):
            placed.append((len(placed) + 1, b))

    ranked_ids = {id(b) for _, b in placed}
    # Everyone else: deeper round reached first, then better seed.
    rest = [b for b in seeds if id(b) not in ranked_ids]
    rest.sort(key=lambda b: (-_ROUND_DEPTH.get(eliminated_in.get(id(b), ""), 0), seed_of[id(b)]))
    pos = len(placed) + 1
    for b in rest:
        placed.append((pos, b))
        pos += 1

    placed.sort(key=lambda pb: pb[0])
    return [{"position": p, "owner": b.owner, "name": getattr(b, "name", b.owner),
             "seed": seed_of[id(b)]} for p, b in placed]
