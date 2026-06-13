"""Bot-vs-bot TOURNAMENT runner — round-robin + Bradley-Terry Elo, across DIFFERENT bots.

The whole point: our teammates' bots use DIFFERENT observation builders (Diego's is
DefaultObs = 89-dim, others use AdvancedObs = 107-dim), but they all share the same
90-action LookupAction. So we can't use a single shared-obs eval. This runner builds
EACH bot its own observation from the shared game state every step and feeds it the obs
it was trained on, while both cars' discrete actions go through the one env's LookupAction.
That lets any mix of 512x3 / 1024x3 / DefaultObs / AdvancedObs bots fight fairly.

Usage:
    python scripts/tournament.py --manifest configs/tournament_bots.yaml --games 30

Manifest (YAML): a list of entrants, each with a label, a path to its PPO_POLICY.pt
(or the folder containing it), and which obs builder it was trained on:

    bots:
      - {label: martin_1024,  policy: checkpoints/<run>/<step>, obs: advanced}
      - {label: diego_512,    policy: diego-bots/checkpoints/MILESTONE_1.18B_nexto_plus_kickoff_512, obs: default}

`obs` is `advanced` (AdvancedObs, 107-dim 1v1) or `default` (DefaultObs, 89-dim). The
network width is auto-detected from the weights, so you don't specify it. A bot trained
on a CUSTOM obs builder can't be entered unless its builder is importable here.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")  # CPU eval; leave the GPU for training

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

torch.cuda.is_available = lambda: False

from rlbot.env import make_env_builder  # noqa: E402
from rlbot.utils.config import load_config  # noqa: E402


def make_obs(name):
    if name == "advanced":
        from rlgym_sim.utils.obs_builders import AdvancedObs

        return AdvancedObs()
    if name == "default":
        from rlgym_sim.utils.obs_builders import DefaultObs

        return DefaultObs()
    raise ValueError(f"Unsupported obs builder {name!r} (use 'advanced' or 'default', or add it here)")


def resolve_policy(path):
    """Accept a PPO_POLICY.pt file or a folder containing one."""
    if os.path.isdir(path):
        cand = os.path.join(path, "PPO_POLICY.pt")
        if os.path.isfile(cand):
            return cand
        raise FileNotFoundError(f"No PPO_POLICY.pt under {path}")
    return path


def load_policy(policy_path, obs_dim, n_actions):
    """Rebuild a DiscreteFF policy, inferring hidden-layer sizes from the saved weights."""
    from rlgym_ppo.ppo import DiscreteFF

    sd = torch.load(resolve_policy(policy_path), map_location="cpu", weights_only=True)
    wk = sorted((k for k in sd if k.endswith(".weight")), key=lambda k: int(k.split(".")[1]))
    layer_sizes = tuple(int(sd[k].shape[0]) for k in wk[:-1])
    saved_in = int(sd[wk[0]].shape[1])
    if saved_in != obs_dim:
        raise ValueError(
            f"obs mismatch: weights expect {saved_in} inputs but the '{'?'}' obs builder gives {obs_dim}. "
            f"Wrong obs name in the manifest?"
        )
    policy = DiscreteFF(obs_dim, n_actions, layer_sizes, "cpu")
    policy.load_state_dict(sd)
    policy.eval()
    return policy


def build_env(config_path, max_seconds, states="kickoff"):
    full = load_config(config_path).to_dict()
    # Tournament start conditions:
    #   "kickoff" (default, FAIREST) — standard symmetric kickoff (rlgym's default setter cycles
    #       the 5 standard kickoff positions). Both bots start equal, every game, no positional
    #       luck. This is the RLBot/RL convention. Caveat: with DETERMINISTIC bots a given
    #       (kickoff position, side) yields the same game, so distinct games ~= 5 positions x 2
    #       sides; run enough games and lean on the championship final, and read the Wilson CIs
    #       knowing the effective sample is the distinct openings, not the raw game count.
    #   "mixed" — kickoffs + random on-ground states: many more DISTINCT deterministic games
    #       (tighter, more robust Elo) but injects per-game positional luck (averages out over
    #       many games). Use if you want a broader skill measure rather than pure fairness.
    if states == "mixed":
        full["state_setter"] = {
            "name": "weighted_sample",
            "components": [
                {"name": "default", "weight": 1.0},
                {"name": "random", "weight": 1.0, "cars_on_ground": True},
            ],
        }
    else:
        full["state_setter"] = {"name": "default"}  # symmetric standard kickoff — fairest
    full["terminal"]["timeout_seconds"] = int(max_seconds)
    env_cfg = dict(full["env"])
    env_cfg["team_size"] = 1
    env_cfg["spawn_opponents"] = True
    return make_env_builder(env_cfg, full)()


def play(env, lut, blue, blue_obs, orange, orange_obs, games, deterministic):
    """blue/orange are (policy, obs_builder). Returns (blue_wins, orange_wins, draws)."""
    bw = ow = dr = 0
    for _ in range(games):
        _, info = env.reset(return_info=True)
        state = info["state"]
        blue_obs.reset(state)
        orange_obs.reset(state)
        prev = {0: np.zeros(8, dtype=np.float32), 1: np.zeros(8, dtype=np.float32)}
        done, result = False, 0.0
        while not done:
            acts = []
            for pl in state.players:
                if pl.team_num == 0:
                    ob = blue_obs.build_obs(pl, state, prev[0])
                    pol = blue
                else:
                    ob = orange_obs.build_obs(pl, state, prev[1])
                    pol = orange
                with torch.no_grad():
                    idx = int(
                        pol.get_action(np.asarray(ob, dtype=np.float32), deterministic=deterministic)[0]
                    )
                acts.append([idx])
                prev[pl.team_num] = lut[idx]
            _, _, done, info = env.step(np.array(acts))
            state = info["state"]
            result = info["result"]
        if result > 0:
            bw += 1
        elif result < 0:
            ow += 1
        else:
            dr += 1
    return bw, ow, dr


def wilson(wins, n, z=1.96):
    """Wilson score 95% CI for a win proportion wins/n. Returns (low, high); (None, None) if n=0.

    Used to make rankings statistically defensible: a result is only called decisive when
    the whole interval clears 50%, not when a small-sample point estimate happens to.
    """
    if n == 0:
        return None, None
    p = wins / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return center - half, center + half


def play_matchup(env, lut, bots, x, y, games_per_side, deterministic):
    """Play x vs y for games_per_side on EACH side (cancels kickoff/side bias).
    Returns (x_wins, y_wins, draws)."""
    bx, by = bots[x], bots[y]
    b1, o1, d1 = play(
        env,
        lut,
        bx["policy"],
        make_obs(bx["obs"]),
        by["policy"],
        make_obs(by["obs"]),
        games_per_side,
        deterministic,
    )
    b2, o2, d2 = play(
        env,
        lut,
        by["policy"],
        make_obs(by["obs"]),
        bx["policy"],
        make_obs(bx["obs"]),
        games_per_side,
        deterministic,
    )
    return b1 + o2, o1 + b2, d1 + d2


def bradley_terry_elo(labels, wins, games):
    s = {b: 1.0 for b in labels}
    W = {b: sum(wins.get((b, o), 0) for o in labels) for b in labels}
    for _ in range(1000):
        for i in labels:
            denom = sum(
                games.get((i, j), 0) / (s[i] + s[j]) for j in labels if j != i and games.get((i, j), 0)
            )
            if denom > 0 and W[i] > 0:
                s[i] = W[i] / denom
        m = sum(s.values()) / len(s)
        for i in labels:
            s[i] /= m
    return {b: round(400 * math.log10(max(s[b], 1e-9)) + 1000) for b in labels}


def main():
    p = argparse.ArgumentParser(
        description="Round-robin bot tournament with Elo (handles mixed obs builders)."
    )
    p.add_argument("--manifest", required=True, help="YAML listing the bots (see module docstring)")
    p.add_argument("--games", type=int, default=30, help="games per side per matchup")
    p.add_argument(
        "--config",
        default="configs/experiments/exp_003_long_run.yaml",
        help="env config (action space source)",
    )
    p.add_argument("--max-seconds", type=int, default=60)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument(
        "--states",
        choices=["kickoff", "mixed"],
        default="kickoff",
        help="start conditions: 'kickoff' (default, fairest — symmetric standard kickoff) or "
        "'mixed' (kickoffs + random on-ground states for a broader, higher-sample skill measure).",
    )
    p.add_argument(
        "--final-games",
        type=int,
        default=200,
        help="championship final: games PER SIDE between the Elo top 2 (default 200 = 400 total, "
        "enough for a ~5pt-resolution Wilson CI). 0 disables the final.",
    )
    p.add_argument("--out", default="tournament_results.json")
    a = p.parse_args()

    with open(a.manifest, encoding="utf-8") as f:
        entrants = yaml.safe_load(f)["bots"]
    env = build_env(a.config, a.max_seconds, a.states)
    print(f"start conditions: {a.states}", flush=True)
    n_actions = int(env.action_space.n)
    from rlbot.actions.lookup_action import LookupAction

    lut = LookupAction().make_lookup_table()

    bots = {}
    for e in entrants:
        ob = make_obs(e["obs"])
        # measure this obs builder's dim from a reset
        _, info = env.reset(return_info=True)
        st = info["state"]
        ob.reset(st)
        dim = int(np.asarray(ob.build_obs(st.players[0], st, np.zeros(8, dtype=np.float32))).shape[0])
        bots[e["label"]] = {"policy": load_policy(e["policy"], dim, n_actions), "obs": e["obs"]}
        print(f"loaded {e['label']:<22} obs={e['obs']} ({dim}-dim)", flush=True)

    labels = list(bots)
    record = {b: {"W": 0, "L": 0, "D": 0} for b in labels}
    wins, games, matches = {}, {}, []
    for x, y in itertools.combinations(labels, 2):
        xw, yw, dd = play_matchup(env, lut, bots, x, y, a.games, a.deterministic)
        wins[(x, y)] = xw
        wins[(y, x)] = yw
        games[(x, y)] = xw + yw
        games[(y, x)] = xw + yw
        record[x]["W"] += xw
        record[x]["L"] += yw
        record[x]["D"] += dd
        record[y]["W"] += yw
        record[y]["L"] += xw
        record[y]["D"] += dd
        lo, hi = wilson(xw, xw + yw)
        matches.append(
            {
                "a": x,
                "b": y,
                "a_wins": xw,
                "b_wins": yw,
                "draws": dd,
                "wilson95_low": round(lo, 3) if lo is not None else None,
                "wilson95_high": round(hi, 3) if hi is not None else None,
            }
        )
        print(f"  {x} {xw} - {yw} {y}  (draws {dd})", flush=True)

    elo = bradley_terry_elo(labels, wins, games)
    standings = sorted(labels, key=lambda b: elo[b], reverse=True)

    # --- CHAMPIONSHIP FINAL: a big head-to-head between the Elo top 2 so the #1 spot is
    # statistically defensible, not a small-sample coin flip. The Elo round-robin SEEDS the
    # final; the final DECIDES 1st/2nd. Verdict rule: only the Wilson 95% CI clearing 50%
    # (either way) overturns or confirms decisively — otherwise the Elo order is kept and
    # the result is reported as statistically inseparable.
    final = None
    if a.final_games > 0 and len(labels) >= 2:
        s1, s2 = standings[0], standings[1]
        print(f"\n=== CHAMPIONSHIP FINAL: {s1} vs {s2} ({a.final_games}/side) ===", flush=True)
        fw, fl, fd = play_matchup(env, lut, bots, s1, s2, a.final_games, a.deterministic)
        dec = fw + fl
        p1 = fw / dec if dec else None
        lo, hi = wilson(fw, dec)
        if lo is not None and lo > 0.5:
            champion, decisive = s1, True
        elif hi is not None and hi < 0.5:
            champion, decisive = s2, True
            standings[0], standings[1] = s2, s1  # the final overturns the seeding
        else:
            champion, decisive = s1, False  # inseparable -> keep Elo order
        final = {
            "a": s1,
            "b": s2,
            "a_wins": fw,
            "b_wins": fl,
            "draws": fd,
            "games": 2 * a.final_games,
            "a_winrate_decisive": round(p1, 3) if p1 is not None else None,
            "wilson95_low": round(lo, 3) if lo is not None else None,
            "wilson95_high": round(hi, 3) if hi is not None else None,
            "decisive": decisive,
            "champion": champion,
        }
        ci = f"[{lo:.1%}, {hi:.1%}]" if lo is not None else "n/a"
        verdict = "DECISIVE (95% CI clears 50%)" if decisive else "inseparable at 95% -> Elo order kept"
        print(f"  {s1} {fw} - {fl} {s2}  (draws {fd})", flush=True)
        print(f"  {s1} decisive win rate {p1:.1%} {ci}  -> {verdict}", flush=True)
        print(f"  CHAMPION: {champion}", flush=True)

    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(
            {"matches": matches, "record": record, "elo": elo, "standings": standings, "final": final},
            f,
            indent=2,
        )

    print("\n=== STANDINGS (Elo seeds the final, the final decides 1st/2nd) ===", flush=True)
    for i, b in enumerate(standings, 1):
        r = record[b]
        print(f"  {i}. {b:<22} Elo {elo[b]}   W-L-D {r['W']}-{r['L']}-{r['D']}", flush=True)
    print(f"\nwrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
