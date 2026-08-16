#!/usr/bin/env python3
"""Figures from the overnight paper runs: long-horizon LOB-Bench (winner vs SS at 2h) and the
step-count ablation (DDPM-100 vs DDIM-10 on a fixed checkpoint)."""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, GREEN, GREY = "#2a78d6", "#eb6834", "#1c8256", "#8a897f"
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": "#c9c8c2", "axes.linewidth": 0.8, "axes.grid": False,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
})
def despine(ax, keep=("left", "bottom")):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in keep)

OUT = "analysis/plots/longhorizon"
os.makedirs(OUT, exist_ok=True)
NICE = {"spread": "spread", "log_inter_arrival": "inter-arrival", "orderbook_imbalance": "book imbalance",
        "orderflow_imbalance": "flow imbalance", "limit_depth_ask": "limit depth", "cancel_depth_ask": "cancel depth"}
METRICS = list(NICE)


def load(p):
    df = pd.read_csv(p); df = df[df.metric == "wasserstein"]
    return {s: g.set_index("score")["distance"].to_dict() for s, g in df.groupby("sampler")}


def bars(scores, labels, colors, title, sub, sub2, fname, xmax=1.05):
    y = np.arange(len(METRICS))[::-1]; h = 0.36
    fig, ax = plt.subplots(figsize=(9.8, 5.0))
    for (lab, key), c, off in zip(labels, colors, [h/2, -h/2]):
        vals = [scores[key][m] for m in METRICS]
        ax.barh(y + off, vals, h, color=c, zorder=3,
                label=f"{lab}   (mean {np.mean(vals):.3f})")
    ax.set_yticks(y); ax.set_yticklabels([NICE[m] for m in METRICS])
    ax.set_xlabel("Wasserstein distance to real  (lower = more realistic)")
    ax.set_xlim(0, xmax); despine(ax)
    ax.legend(loc="lower right", frameon=False, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.85])
    fig.text(0.02, 0.955, title, fontsize=13, fontweight="bold", color=INK, ha="left")
    fig.text(0.02, 0.905, sub, fontsize=10.3, color=INK2, ha="left")
    if sub2:
        fig.text(0.02, 0.862, sub2, fontsize=10.3, color=INK2, ha="left")
    fig.savefig(f"{OUT}/{fname}", dpi=160); plt.close(fig)


lh = load("lob_bench_paper/longhorizon_0129/lob_bench_scores.csv")
bars(lh, [("0.724 baseline", "WINNER_2h"), ("SS epoch 4", "SS_E4_2h")], [ORANGE, GREEN],
     "Two-hour horizon: scheduled sampling pulls decisively ahead",
     "INTC 2015-01-29, 10:00-12:00. At 30 minutes the two are near-identical; over two hours",
     "SS epoch 4 is better on every metric bar one, driven by spread (0.575 -> 0.077).",
     "2_longhorizon_lobbench.png")

sc = load("lob_bench_paper/stepcount_0130/lob_bench_scores.csv")
bars(sc, [("DDPM, 100 steps", "DDPM100"), ("DDIM, 10 steps", "DDIM10")], [GREY, BLUE],
     "Step-count ablation: ten steps beat one hundred",
     "INTC 2015-01-30, checkpoint 0.724, identical decode config -- only the step count differs.",
     "NOT a TRADES baseline: this checkpoint carries our data-pipeline fixes.",
     "3_stepcount_ablation.png")

with open(f"{OUT}/lobbench_tables.md", "w") as f:
    f.write("# LOB-Bench, overnight paper runs\n\n## Two-hour horizon (INTC 2015-01-29, 10:00-12:00)\n\n")
    f.write("| metric | 0.724 baseline | SS epoch 4 |\n|---|---|---|\n")
    for m in METRICS:
        f.write(f"| {NICE[m]} | {lh['WINNER_2h'][m]:.4f} | {lh['SS_E4_2h'][m]:.4f} |\n")
    f.write(f"| **grand mean** | **{np.mean([lh['WINNER_2h'][m] for m in METRICS]):.4f}** | "
            f"**{np.mean([lh['SS_E4_2h'][m] for m in METRICS]):.4f}** |\n\n")
    f.write("## Step-count ablation (INTC 2015-01-30, ckpt 0.724, same decode config)\n\n")
    f.write("| metric | DDPM-100 | DDIM-10 |\n|---|---|---|\n")
    for m in METRICS:
        f.write(f"| {NICE[m]} | {sc['DDPM100'][m]:.4f} | {sc['DDIM10'][m]:.4f} |\n")
    f.write(f"| **grand mean** | **{np.mean([sc['DDPM100'][m] for m in METRICS]):.4f}** | "
            f"**{np.mean([sc['DDIM10'][m] for m in METRICS]):.4f}** |\n\n")
    f.write("Ten steps score *better* than one hundred on the same checkpoint. This is a step-count "
            "ablation on our own model, not a comparison against the published TRADES configuration.\n")

print("wrote", OUT)
for fn in sorted(os.listdir(OUT)): print("  ", fn)
