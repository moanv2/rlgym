"""Turn a tournament_results.json into a self-contained report.html (+ PNGs).

Reads the JSON written by scripts/tournament.py (keys: matches, record, elo, standings)
and produces, in --out-dir:
  - standings.png   : Elo leaderboard table (champion highlighted)
  - heatmap.png     : NxN head-to-head decisive-win% matrix (row beats col)
  - report.html     : one shareable file with both images embedded (base64) + a styled
                      win-matrix table. No internet needed to view it.

Run with the conda python:
  C:\\Users\\Lasca\\miniconda3\\envs\\rl-group-project\\python.exe tools/make_report.py \
    --results tournament_results.json --out-dir deliverables
"""

from __future__ import annotations

import argparse
import base64
import json
import os

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def winrate_matrix(labels, matches):
    """Symmetric decisive-win-rate matrix: cell[i][j] = i's wins / (i+j decisive) vs j."""
    idx = {b: i for i, b in enumerate(labels)}
    n = len(labels)
    wins = np.zeros((n, n))
    games = np.zeros((n, n))
    for m in matches:
        if m["a"] not in idx or m["b"] not in idx:
            continue
        i, j = idx[m["a"]], idx[m["b"]]
        aw, bw = m["a_wins"], m["b_wins"]
        wins[i, j] += aw
        wins[j, i] += bw
        games[i, j] += aw + bw
        games[j, i] += aw + bw
    wr = np.full((n, n), np.nan)
    mask = games > 0
    wr[mask] = wins[mask] / games[mask]
    return wr


def leaderboard_png(data, path):
    standings, elo, rec = data["standings"], data["elo"], data["record"]
    fig, ax = plt.subplots(figsize=(7.5, 0.8 + 0.5 * len(standings)))
    ax.axis("off")
    header = ["#", "Bot", "Elo", "W", "L", "D"]
    rows = [header]
    for i, b in enumerate(standings, 1):
        r = rec[b]
        rows.append([str(i), b, str(elo[b]), str(r["W"]), str(r["L"]), str(r["D"])])
    tbl = ax.table(cellText=rows, loc="center", cellLoc="center", colWidths=[0.06, 0.42, 0.13, 0.1, 0.1, 0.1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(1, 1.6)
    for (r, _c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#444")
        if r == 0:
            cell.set_facecolor("#222")
            cell.set_text_props(color="white", fontweight="bold")
        elif r == 1:
            cell.set_facecolor("#f6d365")
            cell.set_text_props(fontweight="bold")  # champion
        else:
            cell.set_facecolor("#f3f3f3" if r % 2 else "#ffffff")
    ax.set_title("Tournament Standings — Bradley-Terry Elo", fontweight="bold", fontsize=14, pad=14)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def heatmap_png(labels, wr, path):
    n = len(labels)
    fig, ax = plt.subplots(figsize=(1.6 + 0.75 * n, 1.4 + 0.75 * n))
    im = ax.imshow(wr, cmap="RdYlGn", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=9)
    for i in range(n):
        for j in range(n):
            if not np.isnan(wr[i, j]):
                ax.text(j, i, f"{wr[i, j] * 100:.0f}", ha="center", va="center", fontsize=9, color="black")
    ax.set_title("Head-to-head decisive win % (row beats column)", fontsize=12, pad=10)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def html_matrix(labels, wr):
    th = "".join(f"<th>{b}</th>" for b in labels)
    rows = ""
    for i, b in enumerate(labels):
        cells = ""
        for j in range(len(labels)):
            if i == j:
                cells += '<td style="background:#333;color:#888">—</td>'
            elif np.isnan(wr[i, j]):
                cells += "<td></td>"
            else:
                p = wr[i, j]
                # green (1.0) -> red (0.0)
                r = int(220 * (1 - p) + 30 * p)
                g = int(40 * (1 - p) + 180 * p)
                cells += f'<td style="background:rgb({r},{g},60);color:#fff">{p * 100:.0f}</td>'
        rows += f"<tr><th>{b}</th>{cells}</tr>"
    return f'<table class="mat"><tr><th></th>{th}</tr>{rows}</table>'


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--out-dir", default="deliverables")
    p.add_argument("--videos", nargs="*", default=[], help="optional mp4 paths to embed")
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    data = load(a.results)
    labels = data["standings"]
    wr = winrate_matrix(labels, data["matches"])

    lb_png = os.path.join(a.out_dir, "standings.png")
    hm_png = os.path.join(a.out_dir, "heatmap.png")
    leaderboard_png(data, lb_png)
    heatmap_png(labels, wr, hm_png)

    champ = labels[0]
    vids = ""
    for v in a.videos:
        if os.path.isfile(v):
            vids += f'<video controls width="640" src="data:video/mp4;base64,{b64(v)}"></video>'
    report = os.path.join(a.out_dir, "report.html")
    with open(report, "w", encoding="utf-8") as f:
        f.write(f"""<!doctype html><html><head><meta charset="utf-8"><title>Bot Tournament</title>
<style>
body{{font-family:system-ui,Segoe UI,Arial;background:#11141a;color:#eee;margin:0;padding:28px}}
h1{{margin:.2em 0}} .sub{{color:#9aa}} .card{{background:#1b1f27;border-radius:12px;padding:18px;margin:18px 0;box-shadow:0 2px 12px #0006}}
img{{max-width:100%;border-radius:8px}} table.mat{{border-collapse:collapse;font-size:13px}}
table.mat th,table.mat td{{padding:6px 10px;text-align:center;border:1px solid #2a2f3a;min-width:38px}}
.champ{{color:#f6d365;font-weight:700}}
</style></head><body>
<h1>🏆 1v1 Bot Tournament</h1>
<div class="sub">Bradley-Terry Elo · double round-robin (both sides) · deterministic play · {len(labels)} entrants</div>
<div class="card"><h2>Champion: <span class="champ">{champ}</span></h2></div>
<div class="card"><h2>Standings</h2><img src="data:image/png;base64,{b64(lb_png)}"></div>
<div class="card"><h2>Head-to-head win %</h2>{html_matrix(labels, wr)}<p class="sub">row's decisive win rate vs column.</p>
<img src="data:image/png;base64,{b64(hm_png)}"></div>
{f'<div class="card"><h2>Match videos</h2>{vids}</div>' if vids else ''}
</body></html>""")
    print(f"wrote {report}\nwrote {lb_png}\nwrote {hm_png}\nchampion: {champ}", flush=True)


if __name__ == "__main__":
    main()
