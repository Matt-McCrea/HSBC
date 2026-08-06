#!/usr/bin/env python3
"""Marginal-distribution comparisons for the replication baselines: spread, order size and order
type, real vs TRADES's released model vs our DDPM-100 replication. INTC 2015-01-30.

These are the standard replication exhibits --- if our reproduction matches theirs, the marginals
should sit on top of each other. Complements the LOB-Bench scoring (which is a distance summary)
by showing the distributions themselves."""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1c8256"
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": "#c9c8c2", "axes.linewidth": 0.8, "axes.grid": False,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold",
})
def despine(ax, keep=("left", "bottom")):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in keep)

OUT = "analysis/replication_baselines/figures"
os.makedirs(OUT, exist_ok=True)

SRC = [
    ("Real market",            "/tmp/realref/real_0130_union.csv",                      BLUE),
    ("TRADES released",        "/tmp/trades_lob_fixed/INTC_2015-01-30.csv",             ORANGE),
    ("Our DDPM-100",           "sweep_results/DDPM_100/DDPM_100_generated_orders.csv",  GREEN),
]


def load(p):
    df = pd.read_csv(p)
    out = {}
    if "SPREAD" in df.columns:
        s = pd.to_numeric(df["SPREAD"], errors="coerce")
    else:
        s = (pd.to_numeric(df["ask_price_1"], errors="coerce")
             - pd.to_numeric(df["bid_price_1"], errors="coerce")) / 10000.0
    s = s[(s > 0) & (s < 0.25)]
    out["spread_tk"] = (s * 100).round().astype(int)
    sz = pd.to_numeric(df.get("SIZE"), errors="coerce").abs()
    out["size"] = sz[(sz > 0) & (sz < 5000)]
    out["type"] = df["TYPE"].astype(str) if "TYPE" in df.columns else pd.Series(dtype=str)
    return out


data = [(lab, load(p), c) for lab, p, c in SRC]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

# --- spread (ticks) ---
ax = axes[0]
maxtk = 6
for lab, d, c in data:
    v = d["spread_tk"]; v = v[v <= maxtk]
    counts = v.value_counts(normalize=True).sort_index()
    ax.plot(counts.index, counts.values * 100, marker="o", markersize=5, lw=1.6, color=c, label=lab)
ax.set_xlabel("spread (ticks)"); ax.set_ylabel("% of observations")
ax.set_title("Bid-ask spread", loc="left"); despine(ax); ax.legend(frameon=False, fontsize=9.5)

# --- order size (log-x CCDF) ---
ax = axes[1]
for lab, d, c in data:
    v = np.sort(d["size"].values)
    if len(v) == 0: continue
    ccdf = 1.0 - np.arange(len(v)) / len(v)
    ax.plot(v, ccdf, lw=1.7, color=c, label=lab)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("order size (shares)"); ax.set_ylabel("P(size > x)")
ax.set_title("Order size (CCDF)", loc="left"); despine(ax); ax.legend(frameon=False, fontsize=9.5)

# --- order type mix ---
ax = axes[2]
cats = ["LIMIT_ORDER", "ORDER_CANCELLED", "ORDER_EXECUTED"]
nice = ["limit", "cancel", "executed"]
x = np.arange(len(cats)); w = 0.26
for i, (lab, d, c) in enumerate(data):
    vc = d["type"].value_counts(normalize=True) * 100
    ax.bar(x + (i - 1) * w, [vc.get(k, 0) for k in cats], w, color=c, label=lab, zorder=3)
ax.set_xticks(x); ax.set_xticklabels(nice)
ax.set_ylabel("% of orders"); ax.set_title("Order type mix", loc="left")
despine(ax); ax.legend(frameon=False, fontsize=9.5)

fig.tight_layout(rect=[0, 0, 1, 0.86])
fig.text(0.01, 0.955, "Marginal distributions: real vs TRADES released vs our replication",
         fontsize=13, fontweight="bold", color=INK, ha="left")
fig.text(0.01, 0.905, "INTC 2015-01-30. Close agreement between the two generated series is the "
                      "replication check; both diverge from real in the same direction.",
         fontsize=10.3, color=INK2, ha="left")
fig.savefig(f"{OUT}/marginal_distributions_0130.png", dpi=160)
plt.close(fig)

# numbers for the caption
rows = []
for lab, d, _ in data:
    vc = d["type"].value_counts(normalize=True) * 100
    rows.append((lab, d["spread_tk"].median(), d["size"].median(),
                 vc.get("LIMIT_ORDER", 0), vc.get("ORDER_CANCELLED", 0), vc.get("ORDER_EXECUTED", 0)))
with open("analysis/replication_baselines/tables/marginal_distributions.md", "w") as f:
    f.write("# Marginal distributions, INTC 2015-01-30\n\n")
    f.write("| series | median spread (tk) | median size | limit % | cancel % | executed % |\n")
    f.write("|---|---|---|---|---|---|\n")
    for r in rows:
        f.write(f"| {r[0]} | {r[1]:.0f} | {r[2]:.0f} | {r[3]:.1f} | {r[4]:.1f} | {r[5]:.1f} |\n")
    f.write("\nReal reference spans 09:30-11:00; TRADES released 09:45-11:00; our DDPM-100 "
            "09:30-10:30. Marginals are close to stationary intraday so the differing windows are "
            "not material here, unlike for price-path comparisons.\n")

print("wrote", f"{OUT}/marginal_distributions_0130.png")
print(open("analysis/replication_baselines/tables/marginal_distributions.md").read())
