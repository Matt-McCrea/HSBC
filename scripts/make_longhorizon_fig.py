#!/usr/bin/env python3
"""The long-horizon figure: over the two-hour window in which single-step DDIM collapses, both
final-model candidates track the real market instead.

This is the claim that could not be made before these runs existed --- every prior evaluation of
the fixed configuration stopped at 30 minutes, i.e. an hour before the failure mode it is supposed
to fix even appears."""
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
B = "ABIDES/log/paper_runs_downloaded"

SERIES = [
    ("Real market",        "/tmp/realref/real_0129_1000_1200.csv", BLUE,  1.6, 4),
    ("Ours: SS epoch 4",   f"{B}/world_agent_INTC_2015-01-29_12-00-00_30_DDIM_0.0_10_val_ema=0.69__tdprior_sr_dn0.3/processed_orders.csv", GREEN, 1.4, 3),
    ("Ours: 0.724",        f"{B}/world_agent_INTC_2015-01-29_12-00-00_30_DDIM_0.0_10_val_ema=0.724_tdprior_sr_dn0.3/processed_orders.csv", ORANGE, 1.4, 3),
    ("TRADES single-step", "ABIDES/log/world_agent_INTC_2015-01-29_12-00-00_30_DDIM_0.0_1_val_ema=0.763/processed_orders.csv", GREY, 1.4, 2),
]


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


loaded = [(lab, load_mid(p), c, lw, z) for lab, p, c, lw, z in SERIES]
n = min(len(s) for _, s, _, _, _ in loaded)
t = np.arange(n) / 60.0

fig, ax = plt.subplots(figsize=(11, 5.4))
for lab, s, c, lw, z in loaded:
    ax.plot(t, s.values[:n], color=c, lw=lw, zorder=z, label=lab)

ax.axvline(73, color=GREY, lw=0.9, ls=":", zorder=1)
ax.annotate("single-step collapse\nbegins (~min 73)", xy=(73, 32.4), xytext=(78, 32.2),
            fontsize=9.5, color=INK2,
            arrowprops=dict(arrowstyle="->", color=INK2, lw=0.9))

ax.set_xlabel("minutes into session  (10:00–12:00)")
ax.set_ylabel("mid price ($)")
ax.set_xlim(0, t[-1]); despine(ax)
ax.legend(loc="lower left", frameon=False, fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.87])
fig.text(0.02, 0.955, "Two-hour horizon: the corrected configuration holds where single-step fails",
         fontsize=13, fontweight="bold", color=INK, ha="left")
fig.text(0.02, 0.905, "INTC 2015-01-29. Both final-model candidates stay inside the real envelope for the "
                      "full two hours;", fontsize=10.3, color=INK2, ha="left")
fig.text(0.02, 0.862, "the single-step replication crosses the conditioning boundary at ~minute 73 and "
                      "never recovers.", fontsize=10.3, color=INK2, ha="left")
fig.savefig(f"{OUT}/1_longhorizon_2h.png", dpi=160)
plt.close(fig)

# summary numbers
rows = []
for lab, s, _, _, _ in loaded:
    r = np.log(s).diff().dropna()
    rows.append((lab, s.iloc[0], s.iloc[-1], s.min(), s.max(),
                 (s.max() - s.min()) * 100, r.std() * 1e4, s.round(3).nunique()))
with open(f"{OUT}/summary.md", "w") as f:
    f.write("# Two-hour horizon, INTC 2015-01-29 10:00-12:00\n\n")
    f.write("| series | start | end | min | max | range (tk) | ret1s_std (bp) | unique mids |\n")
    f.write("|---|---|---|---|---|---|---|---|\n")
    for r in rows:
        f.write(f"| {r[0]} | {r[1]:.2f} | {r[2]:.2f} | {r[3]:.2f} | {r[4]:.2f} | {r[5]:.0f} | {r[6]:.2f} | {r[7]} |\n")

print("wrote", OUT)
print(open(f"{OUT}/summary.md").read())
