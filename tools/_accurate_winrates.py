"""Thorough, accurate win-rate eval: our champion vs every uploaded teammate bot.

High game count + MIXED start states (each deterministic game is distinct -> real statistical
power) + BOTH sides (no kickoff bias) + Wilson 95% CIs. Single env so it barely touches the
live champion training. Run from repo root with PYTHONPATH=src.
"""
import sys, glob
from pathlib import Path
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
import tournament as T
from rlbot.actions.lookup_action import LookupAction

GAMES_PER_SIDE = 150          # 300 per matchup
CFG = "configs/experiments/exp_003_long_run.yaml"

def latest_complete(pat):
    ds = sorted([Path(p) for p in glob.glob(pat) if Path(p).name.isdigit()], key=lambda d: int(d.name))
    return ds[-2] if len(ds) >= 2 else ds[-1]

champ = latest_complete("checkpoints/exp_recipeH_distill-*/*")
print(f"OUR CHAMPION: step {champ.name} (~{int(champ.name)/1e9:.2f}B)", flush=True)

BOTS = [
    ("Diego 2.85B v7 (latest)", "checkpoints/_eval_snapshots/diego_2.85B_v7",     "advanced", 107),
    ("Diego papaya 1.34B",      "checkpoints/_eval_snapshots/diego_papaya_1.34B", "advanced", 107),
    ("Diego 1.18B (512)",       "checkpoints/_eval_snapshots/diego_1.18B",        "default",  89),
    ("Marco 2.0B",              "checkpoints/_eval_snapshots/marco_2.0B",         "default",  89),
]

lut = LookupAction().make_lookup_table()
env = T.build_env(CFG, 60, "mixed")          # mixed states = many distinct deterministic games
champ_pol = T.load_policy(str(champ), 107, 90)
champ_obs = T.make_obs("advanced")

print(f"\n{'opponent':28s} {'winrate':>8s} {'W-L-D':>14s} {'decisive':>9s} {'Wilson95':>14s}", flush=True)
print("-" * 78, flush=True)
results = []
for name, path, obs, dim in BOTS:
    try:
        opp = T.load_policy(path, dim, 90); opp_obs = T.make_obs(obs)
        cw, ow, d  = T.play(env, lut, champ_pol, champ_obs, opp, opp_obs, GAMES_PER_SIDE, True)   # champ blue
        ow2, cw2, d2 = T.play(env, lut, opp, opp_obs, champ_pol, champ_obs, GAMES_PER_SIDE, True) # champ orange
        CW, OW, D = cw + cw2, ow + ow2, d + d2
        tot = CW + OW + D; dec = CW + OW
        lo, hi = T.wilson(CW, dec)
        print(f"{name:28s} {CW/tot:>7.1%} {f'{CW}-{OW}-{D}':>14s} {CW/dec:>8.1%} {f'{lo:.2f}-{hi:.2f}':>14s}", flush=True)
        results.append((name, CW/tot, CW/dec, lo, hi, CW, OW, D))
    except Exception as e:
        print(f"{name:28s}  ERROR: {repr(e)[:60]}", flush=True)
print("\nNote: 'decisive' excludes draws. Wilson95 is the 95% CI on the decisive win-rate.", flush=True)
print("DONE", flush=True)
