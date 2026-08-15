#!/usr/bin/env python3
"""LOB-Bench comparison figures for the replication baselines: TRADES's released model scored
against ours on the same benchmark and the same real data.

IMPORTANT CAVEAT baked into the figure: the released CSVs carry only 0.1s timestamp resolution
(32% distinct stamps, up to 27 events sharing one) against microsecond resolution in our runs. That
systematically inflates their log_inter_arrival distance --- it is an artifact of the released file
format, not of their model. The headline comparison therefore EXCLUDES inter-arrival, and the metric
is drawn separately, greyed, so the exclusion is visible rather than hidden."""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORANGE, BLUE, GREY = "#eb6834", "#2a78d6", "#8a897f"
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

BASE = "analysis/replication_baselines"
OUT = f"{BASE}/figures"
os.makedirs(OUT, exist_ok=True)

NICE = {"spread": "spread", "log_inter_arrival": "inter-arrival*", "orderbook_imbalance": "book imbalance",
        "orderflow_imbalance": "flow imbalance", "limit_depth_ask": "limit depth", "cancel_depth_ask": "cancel depth"}
ORDER = ["spread", "orderbook_imbalance", "orderflow_imbalance", "limit_depth_ask",
         "cancel_depth_ask", "log_inter_arrival"]      # inter-arrival last: excluded from headline
FAIR = [m for m in ORDER if m != "log_inter_arrival"]


def load(day):
    p = f"{BASE}/lob_bench/INTC_{day}/lob_bench_scores.csv"
    df = pd.read_csv(p)
    df = df[df.metric == "wasserstein"]
    return {s: g.set_index("score")["distance"].to_dict() for s, g in df.groupby("sampler")}


def panel(ax, scores, labels, colors, title, sub):
    y = np.arange(len(ORDER))[::-1]
    h = 0.36
    offs = [h/2, -h/2]
    for (lab, key), c, off in zip(labels, colors, offs):
        vals = [scores[key][m] for m in ORDER]
        bars = ax.barh(y + off, vals, h, color=c, zorder=3,
                       label=f"{lab}  (mean* {np.mean([scores[key][m] for m in FAIR]):.3f})")
        # grey out the excluded inter-arrival bar
        bars[ORDER.index("log_inter_arrival")].set_alpha(0.32)
    ax.set_yticks(y); ax.set_yticklabels([NICE[m] for m in ORDER])
    ax.set_xlabel("Wasserstein distance to real  (lower = more realistic)")
    ax.set_xlim(0, 1.85); despine(ax)
    ax.legend(loc="lower right", frameon=False, fontsize=9.5)
    ax.set_title(title, loc="left", fontsize=12, pad=6)


# ---- headline: like-for-like DDPM replication, 2015-01-30 ----
s30 = load("2015-01-30")
fig, ax = plt.subplots(figsize=(9.8, 5.0))
panel(ax, s30, [("TRADES released (DDPM)", "THEIRS_DDPM"), ("Our DDPM-100 replication", "OURS_DDPM100")],
      [ORANGE, BLUE], "", "")
fig.tight_layout(rect=[0, 0, 1, 0.83])
fig.text(0.02, 0.955, "Replication check: our DDPM-100 against TRADES's released model",
         fontsize=13, fontweight="bold", color=INK, ha="left")
fig.text(0.02, 0.905, "INTC 2015-01-30, both scored on LOB-Bench against the same real market data.",
         fontsize=10.3, color=INK2, ha="left")
fig.text(0.02, 0.862, "*inter-arrival excluded from the mean: the released CSVs have only 0.1s timestamp "
                      "resolution, which inflates it.", fontsize=9.2, color=INK2, ha="left")
fig.savefig(f"{OUT}/lobbench_replication_ddpm_0130.png", dpi=160); plt.close(fig)

# ---- secondary: their DDPM vs our single-step, 2015-01-29 ----
s29 = load("2015-01-29")
fig, ax = plt.subplots(figsize=(9.8, 5.0))
panel(ax, s29, [("TRADES released (DDPM)", "THEIRS_DDPM"), ("Our DDIM-1 (single step)", "OURS_DDIM1")],
      [ORANGE, GREY], "", "")
fig.tight_layout(rect=[0, 0, 1, 0.83])
fig.text(0.02, 0.955, "Benchmark blind spot: the collapsing single-step run scores well",
         fontsize=13, fontweight="bold", color=INK, ha="left")
fig.text(0.02, 0.905, "INTC 2015-01-29. Our DDIM-1 diverges to $31.86 against real's $33.76, yet scores "
                      "BELOW their working model.", fontsize=10.3, color=INK2, ha="left")
fig.text(0.02, 0.862, "LOB-Bench measures distributional fidelity, not price-path validity --- which is why "
                      "it must not be read alone.", fontsize=9.2, color=INK2, ha="left")
fig.savefig(f"{OUT}/lobbench_blindspot_ddim1_0129.png", dpi=160); plt.close(fig)

# ---- tables ----
os.makedirs(f"{BASE}/tables", exist_ok=True)
with open(f"{BASE}/tables/lobbench_theirs_vs_ours.md", "w") as f:
    f.write("# LOB-Bench: TRADES released vs our replication\n\n")
    f.write("Wasserstein distance to real (lower = more realistic). `*` = excluded from the mean, see note.\n\n")
    for day, sc, labs in [("2015-01-30", s30, [("TRADES released (DDPM)", "THEIRS_DDPM"),
                                               ("Our DDPM-100", "OURS_DDPM100")]),
                          ("2015-01-29", s29, [("TRADES released (DDPM)", "THEIRS_DDPM"),
                                               ("Our DDIM-1", "OURS_DDIM1")])]:
        f.write(f"\n## INTC {day}\n\n| metric | " + " | ".join(l for l, _ in labs) + " |\n")
        f.write("|---|" + "---|" * len(labs) + "\n")
        for m in ORDER:
            star = "*" if m == "log_inter_arrival" else ""
            f.write(f"| {NICE[m].rstrip('*')}{star} | " + " | ".join(f"{sc[k][m]:.4f}" for _, k in labs) + " |\n")
        f.write("| **mean (excl. inter-arrival)** | " +
                " | ".join(f"**{np.mean([sc[k][m] for m in FAIR]):.4f}**" for _, k in labs) + " |\n")
        f.write("| mean (all six) | " +
                " | ".join(f"{np.mean([sc[k][m] for m in ORDER]):.4f}" for _, k in labs) + " |\n")
    f.write("\n\n`*` The released TRADES-LOB CSVs carry only 0.1s timestamp resolution (32% distinct "
            "stamps; up to 27 events share one), against microsecond resolution in our runs. This "
            "systematically inflates their inter-arrival distance and is an artifact of the released "
            "file format, not of their model. Exclude it from any headline comparison.\n")

print("wrote figures + tables under", BASE)
for fn in sorted(os.listdir(OUT)): print("   figures/", fn)
