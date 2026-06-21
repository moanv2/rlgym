"""Render the presentation charts from a tournament stats JSON.

Four figures (each a standalone PNG for dropping into slides), plus a combined
2x2 dashboard:

  1. winrate_convergence : per-bot cumulative win-rate vs game number (one colored
     line per bot) -- shows the average stabilising over ~300 games.
  2. goals_saves_demos   : per-bot average goals / saves / demos per game, STACKED.
  3. goal_margins        : aggregate goal margin per matchup, sorted -- the biggest
     blowout (top) down to the closest game (bottom).
  4. fun_facts           : per-bot % airborne / possession / supersonic / dribbling,
     plus a highlights box (most aerial, most possession, fastest, etc.).

Usage:
    python -m rlbot.tournament.charts                       # newest stats JSON
    python -m rlbot.tournament.charts history_and_summary/tournament_stats_*.json
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = REPO_ROOT / "tournament_results" / "figures"

# Fixed colour per person (papaya = orange, naturally).
COLORS = {
    "diego": "#ff7f0e", "martin": "#1f77b4", "nachi": "#2ca02c",
    "marco": "#9467bd", "marian": "#d62728",
}
plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.axisbelow": True})


def _load(path_arg: str | None) -> tuple[dict, Path]:
    if path_arg:
        p = Path(path_arg)
    else:
        cands = sorted(glob.glob(str(REPO_ROOT / "history_and_summary" / "tournament_stats_*.json")))
        if not cands:
            raise SystemExit("No tournament_stats_*.json found — run rlbot.tournament.stats_round_robin first.")
        p = Path(cands[-1])
    return json.loads(p.read_text()), p


def _ordered(per_bot: dict) -> list[str]:
    """Owners sorted by win rate (best first) for consistent bar ordering."""
    return sorted(per_bot, key=lambda o: per_bot[o]["win_rate"], reverse=True)


def _color(o: str) -> str:
    return COLORS.get(o, "#777777")


def fig_convergence(data, ax):
    per = data["per_bot"]
    for o in _ordered(per):
        seq = np.array(per[o]["seq"], dtype=float)
        if len(seq) == 0:
            continue
        cum = np.cumsum(seq) / (np.arange(len(seq)) + 1)
        ax.plot(np.arange(1, len(seq) + 1), 100 * cum, color=_color(o), lw=2,
                label=f"{per[o]['name'].split('—')[0].strip()} ({100*per[o]['win_rate']:.0f}%)")
    ax.axhline(50, color="k", ls=":", lw=1, alpha=0.5)
    ax.set_xlabel("game number (per bot)"); ax.set_ylabel("cumulative win rate (%)")
    ax.set_title("1 · Win-rate convergence over the tournament")
    ax.legend(fontsize=8, loc="upper right"); ax.set_ylim(0, 100)


def fig_goals_saves_demos(data, ax):
    per = data["per_bot"]; order = _ordered(per)
    names = [per[o]["name"].split("—")[0].strip() for o in order]
    goals = np.array([per[o]["avg_goals"] for o in order])
    saves = np.array([per[o]["avg_saves"] for o in order])
    demos = np.array([per[o]["avg_demos"] for o in order])
    x = np.arange(len(order))
    ax.bar(x, goals, color="#4c78a8", label="goals/game")
    ax.bar(x, saves, bottom=goals, color="#54a24b", label="saves/game (clears)")
    ax.bar(x, demos, bottom=goals + saves, color="#e45756", label="demos/game")
    for i in range(len(order)):
        ax.text(i, goals[i] + saves[i] + demos[i] + 0.05,
                f"{goals[i]+saves[i]+demos[i]:.1f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("avg per game"); ax.set_title("2 · Goals / saves / demos per game (stacked)")
    ax.legend(fontsize=8)


def fig_goal_margins(data, ax):
    ms = sorted(data["matchups"], key=lambda m: abs(m["margin"]), reverse=True)
    labels, vals, cols = [], [], []
    for m in ms:
        if m["margin"] >= 0:
            win, lose, wg, lg = m["a"], m["b"], m["a_goals"], m["b_goals"]
        else:
            win, lose, wg, lg = m["b"], m["a"], m["b_goals"], m["a_goals"]
        labels.append(f"{win} {wg}-{lg} {lose}")
        vals.append(abs(m["margin"])); cols.append(_color(win))
    y = np.arange(len(labels))[::-1]  # biggest at top
    ax.barh(y, vals, color=cols)
    for yi, v in zip(y, vals):
        ax.text(v + 0.3, yi, f"+{v}", va="center", fontsize=8)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("aggregate goal margin")
    ax.set_title("3 · Biggest blowout (top) → closest game (bottom)")
    ax.margins(x=0.12)


def fig_fun_facts(data, ax):
    per = data["per_bot"]; order = _ordered(per)
    names = [per[o]["name"].split("—")[0].strip() for o in order]
    metrics = [("possession_pct", "possession %"), ("air_pct", "airborne %"),
               ("supersonic_pct", "supersonic %"), ("dribble_pct", "dribble %")]
    x = np.arange(len(order)); w = 0.2
    for i, (key, lab) in enumerate(metrics):
        ax.bar(x + (i - 1.5) * w, [per[o][key] for o in order], w, label=lab)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("% of play time"); ax.set_title("4 · Fun facts — playstyle profile")
    ax.legend(fontsize=8, ncol=2)


def highlights(data) -> list[str]:
    per = data["per_bot"]
    def top(key):
        o = max(per, key=lambda x: per[x][key]); return per[o]["name"].split("—")[0].strip(), per[o][key]
    air = top("air_pct"); poss = top("possession_pct"); ss = top("supersonic_pct"); drib = top("dribble_pct")
    low_boost = min(per, key=lambda x: per[x]["avg_boost"])
    return [
        f"Most aerial: {air[0]} ({air[1]:.0f}% airborne)",
        f"Most possession: {poss[0]} ({poss[1]:.0f}%)",
        f"Fastest (most supersonic): {ss[0]} ({ss[1]:.1f}%)",
        f"Best dribbler: {drib[0]} ({drib[1]:.1f}% on the roof)",
        f"Biggest boost hog (runs lowest): {per[low_boost]['name'].split('—')[0].strip()} "
        f"({100*per[low_boost]['avg_boost']:.0f}% avg tank)",
    ]


def main() -> None:
    data, src = _load(sys.argv[1] if len(sys.argv) > 1 else None)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Source: {src}  ({data['total_games']} games, {data['mode']})")

    # individual figures
    for name, fn, size in [
        ("winrate_convergence", fig_convergence, (8, 5)),
        ("goals_saves_demos", fig_goals_saves_demos, (8, 5)),
        ("goal_margins", fig_goal_margins, (8, 5)),
        ("fun_facts", fig_fun_facts, (8, 5)),
    ]:
        fig, ax = plt.subplots(figsize=size)
        fn(data, ax)
        fig.suptitle(f"papaya tournament · {data['total_games']} games ({data['mode']})",
                     fontsize=9, y=0.99, alpha=0.6)
        fig.tight_layout()
        out = OUT_DIR / f"{name}.png"
        fig.savefig(out, bbox_inches="tight"); plt.close(fig)
        print(f"  wrote {out.relative_to(REPO_ROOT)}")

    # combined dashboard
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig_convergence(data, axes[0, 0]); fig_goals_saves_demos(data, axes[0, 1])
    fig_goal_margins(data, axes[1, 0]); fig_fun_facts(data, axes[1, 1])
    fig.suptitle(f"papaya tournament dashboard · {data['total_games']} {data['mode']} games", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    dash = OUT_DIR / "dashboard.png"
    fig.savefig(dash, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {dash.relative_to(REPO_ROOT)}")

    print("\nFun facts:")
    for h in highlights(data):
        print(f"  • {h}")


if __name__ == "__main__":
    main()
