#!/usr/bin/env python3
"""LOB-Bench per-metric comparison against the published TRADES sampling default, house palette.

Supersedes the two-bar version in 2_longhorizon_lobbench.png, which compared our two checkpoints
against each other only. The third bar here is the actual published configuration — DDPM, 100
steps, checkpoint val_ema=0.667, no decode-time flags — so this is the first version of the figure
that answers "better than TRADES?" rather than "better than our other checkpoint?".

Both panels matter and they disagree, which is the point: the grand mean favours our model by 52%,
but the default wins outright on the two imbalance metrics. A single-number claim would hide that.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, BLUE_LT, RED = "#2a78d6", "#8fb9e8", "#c0492f"
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": "#c9c8c2", "axes.linewidth": 0.8,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

OUT = "analysis/plots/longhorizon"
ORDER = ["spread", "log_inter_arrival", "orderbook_imbalance",
         "orderflow_imbalance", "limit_depth_ask", "cancel_depth_ask"]
NICE = ["spread", "inter-\narrival", "book\nimbalance", "flow\nimbalance",
        "limit\ndepth", "cancel\ndepth"]


def w(path, sampler=None):
    d = pd.read_csv(path)
    d = d[d.metric == "wasserstein"]
    if sampler:
        d = d[d.sampler == sampler]
    return d.set_index("score")["distance"]


def main():
    old = f"{OUT}/lob_bench/longhorizon_0129.csv"
    series = [
        ("TRADES default (DDPM-100)", w("lob_bench_paper/default_0129/lob_bench_scores.csv"), RED),
        ("Ours: 0.724 baseline", w(old, "WINNER_2h"), BLUE_LT),
        ("Ours: SS epoch 4", w(old, "SS_E4_2h"), BLUE),
    ]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5.2),
                                  gridspec_kw={"width_ratios": [2.5, 1]})

    x = np.arange(len(ORDER))
    wdt = 0.26
    for i, (lab, s, c) in enumerate(series):
        ax.bar(x + (i - 1) * wdt, [s[k] for k in ORDER], wdt, color=c, label=lab, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(NICE, fontsize=9.5)
    ax.set_ylabel("Wasserstein distance  (lower = closer to real)")
    ax.set_title("Per metric")
    ax.legend(frameon=False, fontsize=9.5)
    ax.grid(axis="y", color="#e6e5e0", zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    gm = [(lab, float(np.mean([s[k] for k in ORDER])), c) for lab, s, c in series]
    ax2.bar(range(3), [g[1] for g in gm], 0.55, color=[g[2] for g in gm], zorder=3)
    for i, (_, v, _) in enumerate(gm):
        ax2.text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=10.5,
                 fontweight="bold", color=INK)
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(["TRADES\ndefault", "0.724\nbaseline", "SS\nepoch 4"], fontsize=9.5)
    ax2.set_title("Grand mean")
    ax2.grid(axis="y", color="#e6e5e0", zorder=0)
    for sp in ("top", "right"):
        ax2.spines[sp].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.text(0.02, 0.955, "LOB-Bench at two hours: 52% closer to real than the published default",
             fontsize=13.5, fontweight="bold", color=INK, ha="left")
    fig.text(0.02, 0.905, "INTC 2015-01-29, 10:00–12:00. Six Wasserstein metrics (Nagy et al. 2025). "
                          "The gain is concentrated in spread and depth;",
             fontsize=10.3, color=INK2, ha="left")
    fig.text(0.02, 0.862, "the default is genuinely better on both imbalance metrics — a trade, not "
                          "a clean sweep.", fontsize=10.3, color=INK2, ha="left")

    for ext, kw in (("png", {"dpi": 300}), ("pdf", {})):
        p = f"{OUT}/11_lobbench_vs_default_0129.{ext}"
        fig.savefig(p, **kw)
        print("  wrote", p)

    # Decomposition: which metrics actually supply the gain over the default.
    d, ss = series[0][1], series[2][1]
    tot = sum(d[k] - ss[k] for k in ORDER)
    print(f"\nSS epoch 4 vs TRADES default — decomposition of the {tot:.3f} total metric gap:")
    for k, n in zip(ORDER, [n.replace("\n", " ") for n in NICE]):
        g = d[k] - ss[k]
        print(f"  {n:16s} {g:+.3f}   {100*g/tot:+6.1f}%")


if __name__ == "__main__":
    main()
