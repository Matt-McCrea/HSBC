#!/usr/bin/env python3
"""Methodology figures for the TRADES replication (1-step DDIM, ckpt 0.763, INTC 2015-01-29,
10:00-12:00) against the matching real replay — the same window, so the comparison is like-for-like.

Figure 1: mid-price trace with the documented price-OOD boundary (~$33.50) marked. Real dips below
it and recovers; the replication crosses it and collapses irrecoverably. Colour convention matches
the stylized-facts batteries these sit alongside: blue = real, orange = generated."""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE = "#2a78d6", "#eb6834"
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

OUT = "analysis/plots/methodology_replication"
os.makedirs(OUT, exist_ok=True)
BOUNDARY = 33.50   # documented price-OOD threshold (z ~ -4) for the pre-reanchor conditioning

REAL = "ABIDES/log/market_replay_INTC_2015-01-29_12-00-00_30/processed_orders.csv"
GEN  = ("ABIDES/log/world_agent_INTC_2015-01-29_12-00-00_30_DDIM_0.0_1_val_ema=0.763/"
        "processed_orders.csv")


def load_mid(path):
    df = pd.read_csv(path)
    df["dt"] = pd.to_datetime(df.iloc[:, 0], errors="coerce")
    for c in ("ask_price_1", "bid_price_1"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    d = df[(df.ask_price_1 > 0) & (df.bid_price_1 > 0)
           & (df.ask_price_1.abs() < 9e9) & (df.bid_price_1.abs() < 9e9)].copy()
    d["mid"] = (d.ask_price_1 + d.bid_price_1) / 2 / 10000.0
    d = d.dropna(subset=["dt", "mid"])
    return d.set_index("dt")["mid"].resample("1s").last().ffill().dropna()


real, gen = load_mid(REAL), load_mid(GEN)
n = min(len(real), len(gen))
real, gen = real.iloc[:n], gen.iloc[:n]
t = np.arange(n) / 60.0     # minutes into the session

fig, ax = plt.subplots(figsize=(10.5, 5.0))
ax.axhline(BOUNDARY, color=INK2, lw=1.1, ls="--", zorder=2)
ax.plot(t, real.values, color=BLUE, lw=1.3, zorder=4, label="Real market")
ax.plot(t, gen.values, color=ORANGE, lw=1.3, zorder=3, label="TRADES replication (1-step DDIM)")

# mark where the replication crosses the boundary for the last time
below = gen < BOUNDARY
if below.any():
    cross_i = int(np.argmax(below.values))
    ax.axvline(t[cross_i], color=ORANGE, lw=0.9, ls=":", alpha=0.8, zorder=2)
    ax.annotate("crosses $33.50\nand never recovers",
                xy=(t[cross_i], gen.values[cross_i]), xytext=(t[cross_i] + 6, 33.1),
                fontsize=9.5, color=INK2,
                arrowprops=dict(arrowstyle="->", color=INK2, lw=0.9))

ax.text(1.5, BOUNDARY + 0.045, "documented price-OOD boundary  ~$33.50  (z ≈ −4)",
        fontsize=9.2, color=INK2, va="bottom")
ax.set_xlabel("minutes into session  (10:00–12:00)")
ax.set_ylabel("mid price ($)")
ax.set_xlim(0, t[-1]); despine(ax)
ax.legend(loc="lower left", frameon=False, fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.87])
fig.text(0.02, 0.955, "Replication failure: irrecoverable divergence at the conditioning boundary",
         fontsize=13, fontweight="bold", color=INK, ha="left")
fig.text(0.02, 0.905, "INTC 2015-01-29. The real market dips below $33.50 and recovers; the replication "
                      "crosses the same level and collapses.", fontsize=10.3, color=INK2, ha="left")
fig.savefig(f"{OUT}/1_replication_price_ood_collapse.png", dpi=160)
plt.close(fig)

# ---- summary numbers for the caption / text ----
def stats(s, name):
    below = s < BOUNDARY
    first = s.index[int(np.argmax(below.values))] if below.any() else None
    after_max = s.loc[first:].max() if first is not None else np.nan
    d60 = s.diff(60)
    return dict(series=name, start=s.iloc[0], end=s.iloc[-1], lo=s.min(), hi=s.max(),
                first_below=None if first is None else first.time(),
                max_after=after_max, recovers=bool(after_max >= BOUNDARY) if first is not None else None,
                worst_60s=d60.min())

rows = [stats(real, "real"), stats(gen, "replication")]
with open(f"{OUT}/summary.md", "w") as f:
    f.write("# TRADES replication failure — summary numbers\n\n")
    f.write("INTC 2015-01-29, 10:00-12:00, 1-step DDIM, checkpoint val_ema=0.763.\n")
    f.write(f"Boundary reference: ${BOUNDARY:.2f}.\n\n")
    f.write("| series | start | end | min | max | first below boundary | max after | recovers? | worst 60s move |\n")
    f.write("|---|---|---|---|---|---|---|---|---|\n")
    for r in rows:
        f.write(f"| {r['series']} | {r['start']:.2f} | {r['end']:.2f} | {r['lo']:.2f} | {r['hi']:.2f} | "
                f"{r['first_below']} | {r['max_after']:.2f} | {r['recovers']} | {r['worst_60s']:.3f} |\n")

print("wrote", OUT)
for fn in sorted(os.listdir(OUT)):
    print("  ", fn)
