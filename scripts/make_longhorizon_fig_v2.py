#!/usr/bin/env python3
"""Long-horizon two-hour figure, recoloured to the house red/blue palette, for the dissertation.

Differs from scripts/make_longhorizon_fig.py (which it does NOT overwrite):
  * house palette: real = neutral dark grey, ours = blue, TRADES single-step = red. The failure is
    now the most salient line on the chart rather than the most recessive.
  * emits BOTH a three-series version (real / adopted model / single-step) and a four-series one
    that adds the 0.724 baseline in a lighter blue, so the two "ours" lines read as one family.
  * 300 dpi PNG plus a PDF with TrueType fonts (pdf.fonttype=42) for LaTeX.
  * rebuilds the real reference from the raw LOBSTER files if absent. The original script read it
    from /tmp/realref/, which does not survive a reboot; the rebuild is deterministic and was
    verified to reproduce every statistic recorded in summary.md (start 33.97, end 33.99, min 33.47,
    max 34.02, 56 tk, 1.26 bp, 92 unique mids).
"""
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# House palette (scripts/make_hsbc_figs_stability.py). Red/blue pair, per the figure brief.
BLUE, BLUE_LT, RED = "#2a78d6", "#8fb9e8", "#c0492f"
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
REAL_GREY = "#52514e"          # neutral dark grey: the reference, present but not competing
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": "#c9c8c2", "axes.linewidth": 0.8, "axes.grid": False,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
    "pdf.fonttype": 42, "ps.fonttype": 42,          # embed TrueType, not Type 3, for LaTeX
})


def despine(ax, keep=("left", "bottom")):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in keep)


RED_LT = "#e08a76"             # second TRADES line: same family as RED, clearly subordinate

OUT = "analysis/plots/longhorizon"
B = "ABIDES/log/paper_runs_downloaded"
REAL = f"{OUT}/real_0129_1000_1200.csv"      # kept beside the figure, not in /tmp
REAL_30 = f"{OUT}/real_0130_1000_1200.csv"

SS_E4 = f"{B}/world_agent_INTC_2015-01-29_12-00-00_30_DDIM_0.0_10_val_ema=0.69__tdprior_sr_dn0.3/processed_orders.csv"
BASE_0724 = f"{B}/world_agent_INTC_2015-01-29_12-00-00_30_DDIM_0.0_10_val_ema=0.724_tdprior_sr_dn0.3/processed_orders.csv"
SINGLE = "ABIDES/log/world_agent_INTC_2015-01-29_12-00-00_30_DDIM_0.0_1_val_ema=0.763/processed_orders.csv"

# The vanilla DDPM-100 replication of the published TRADES sampling default: 100 steps, checkpoint
# val_ema=0.667, and NO decode-time flags. Earlier "TRADES" rows in this folder were either a
# step-count ablation on OUR checkpoint (0.724, which carries PRICE_REANCHOR/UNCLAMP_DEPTH) or the
# DDIM-1 single-step run — neither is the paper's default, so neither could carry this comparison.
DDPM_29 = f"{B}/world_agent_INTC_2015-01-29_12-00-00_30_DDPM_0.0_100_val_ema=0.667/processed_orders.csv"
DDPM_30 = f"{B}/world_agent_INTC_2015-01-30_12-00-00_30_DDPM_0.0_100_val_ema=0.667/processed_orders.csv"
SS_E4_30 = f"{B}/world_agent_INTC_2015-01-30_12-00-00_30_DDIM_0.0_10_val_ema=0.69__tdprior_sr_dn0.3/processed_orders.csv"
BASE_30 = f"{B}/world_agent_INTC_2015-01-30_12-00-00_30_DDIM_0.0_10_val_ema=0.724_tdprior_sr_dn0.3/processed_orders.csv"


def ensure_real(path=REAL, date="2015-01-29"):
    if os.path.exists(path):
        return
    print(f"[real] {path} absent — rebuilding from LOBSTER")
    subprocess.run([sys.executable, "-m", "evaluation.stylized_custom.lobster_real_reference",
                    "--ticker", "INTC", "--date", date,
                    "--st", "10:00:00", "--et", "12:00:00", "--out", path], check=True)


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


def draw(series, stem, title=None, sub1=None, sub2=None, mark_collapse=True, day="2015-01-29",
         legend_loc="lower left"):
    """series: list of (label, mid_series, colour, linewidth, zorder)

    mark_collapse gates the minute-73 annotation: it describes the DDIM-1 single-step trace, so on
    any figure that does not plot that series it would point at nothing.
    """
    n = min(len(s) for _, s, _, _, _ in series)
    t = np.arange(n) / 60.0
    fig, ax = plt.subplots(figsize=(11, 5.4))
    for lab, s, c, lw, z in series:
        ax.plot(t, s.values[:n], color=c, lw=lw, zorder=z, label=lab)

    if mark_collapse:
        ax.axvline(73, color="#8a897f", lw=0.9, ls=":", zorder=1)
        # Sits in the empty band between the model traces (~33.8+) and the collapsed line (~32.8-),
        # with the arrow on the drop itself. The original position (32.4/32.2) overlapped the
        # single-step trace, which only became obvious once that line was recoloured red.
        ax.annotate("single-step collapse\nbegins (~min 73)", xy=(73.5, 33.15), xytext=(79, 33.42),
                    fontsize=9.5, color=INK2,
                    arrowprops=dict(arrowstyle="->", color=INK2, lw=0.9))

    ax.set_xlabel("minutes into session  (10:00–12:00)")
    ax.set_ylabel("mid price ($)")
    ax.set_xlim(0, t[-1])
    despine(ax)
    ax.legend(loc=legend_loc, frameon=False, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.87])
    fig.text(0.02, 0.955,
             title or "Two-hour horizon: the corrected configuration holds where single-step fails",
             fontsize=13, fontweight="bold", color=INK, ha="left")
    fig.text(0.02, 0.905,
             sub1 or f"INTC {day}. Both final-model candidates stay inside the real envelope for the "
                     "full two hours;", fontsize=10.3, color=INK2, ha="left")
    fig.text(0.02, 0.862,
             sub2 or "the single-step replication crosses the conditioning boundary at ~minute 73 and "
                     "never recovers.", fontsize=10.3, color=INK2, ha="left")
    for ext, kw in (("png", {"dpi": 300}), ("pdf", {})):
        p = f"{OUT}/{stem}.{ext}"
        fig.savefig(p, **kw)
        print("  wrote", p)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    ensure_real()
    real = load_mid(REAL)
    ss = load_mid(SS_E4)
    base = load_mid(BASE_0724)
    single = load_mid(SINGLE)

    # Three-series: the argument at its cleanest — reference, the model, the failure.
    draw([("Real market", real, REAL_GREY, 1.6, 4),
          ("Ours: SS epoch 4", ss, BLUE, 1.6, 3),
          ("TRADES single-step", single, RED, 1.6, 2)],
         "1_longhorizon_2h_3series")

    # Four-series: adds the baseline in a lighter blue so both "ours" lines read as one family
    # and the red failure still dominates.
    draw([("Real market", real, REAL_GREY, 1.6, 5),
          ("Ours: SS epoch 4", ss, BLUE, 1.6, 4),
          ("Ours: 0.724 baseline", base, BLUE_LT, 1.4, 3),
          ("TRADES single-step", single, RED, 1.6, 2)],
         "1_longhorizon_2h_4series")

    # --- vs the published TRADES default -------------------------------------------------
    # The comparison the dissertation actually needs: our adopted model against the sampling
    # configuration the TRADES paper specifies (DDPM, 100 steps, no decode-time corrections),
    # rather than against the single-step variant we introduced for acceleration.
    ddpm29 = load_mid(DDPM_29)
    draw([("Real market", real, REAL_GREY, 1.6, 4),
          ("Ours: SS epoch 4 (DDIM-10)", ss, BLUE, 1.6, 3),
          ("TRADES default (DDPM-100)", ddpm29, RED, 1.6, 2)],
         "6_longhorizon_2h_vs_default", mark_collapse=False,
         title="Two-hour horizon: our model tracks the real envelope more closely than the TRADES default",
         sub1="INTC 2015-01-29, 10:00–12:00. The published sampling default (DDPM, 100 steps, no "
              "decode-time correction) runs",
         sub2="3.2x the real one-second volatility (4.04 bp vs 1.26 bp); the adopted configuration "
              "runs 1.65 bp.")

    # All four, so the two TRADES configurations can be compared against each other directly.
    draw([("Real market", real, REAL_GREY, 1.6, 5),
          ("Ours: SS epoch 4 (DDIM-10)", ss, BLUE, 1.6, 4),
          ("Ours: 0.724 baseline (DDIM-10)", base, BLUE_LT, 1.4, 3),
          ("TRADES default (DDPM-100)", ddpm29, RED, 1.6, 2),
          ("TRADES single-step (DDIM-1)", single, RED_LT, 1.4, 1)],
         "7_longhorizon_2h_all",
         title="Two-hour horizon: both TRADES sampling configurations leave the real envelope",
         sub1="INTC 2015-01-29, 10:00–12:00. Single-step collapses downward at ~minute 73; the "
              "100-step default drifts upward",
         sub2="through the whole session. Both of our configurations stay inside the real range.")

    # --- second test day ------------------------------------------------------------------
    # 2015-01-30 is the other held-out day and the drift is far more pronounced there, so showing
    # only 01-29 would understate it. Same series, same palette.
    ensure_real(REAL_30, "2015-01-30")
    real30, ss30, base30, ddpm30 = (load_mid(p) for p in (REAL_30, SS_E4_30, BASE_30, DDPM_30))
    draw([("Real market", real30, REAL_GREY, 1.6, 5),
          ("Ours: SS epoch 4 (DDIM-10)", ss30, BLUE, 1.6, 4),
          ("Ours: 0.724 baseline (DDIM-10)", base30, BLUE_LT, 1.4, 3),
          ("TRADES default (DDPM-100)", ddpm30, RED, 1.6, 2)],
         "8_longhorizon_2h_0130_vs_default", mark_collapse=False, day="2015-01-30",
         # Legend sits upper-left: the band below it carries the real trace, and at lower left the
         # default "lower left" placement overprinted it.
         legend_loc="upper left",
         title="Second test day: the TRADES default ends $2.00 above the real close",
         sub1="INTC 2015-01-30, 10:00–12:00. A single +$0.94 second at minute 39 supplies 47% of "
              "that gap; the rest accrues as drift.",
         sub2="The default traverses 222 ticks against the real day's 33. Both corrected "
              "configurations stay within 45 ticks.")

    # Numbers behind the figures, so the captions can be checked against them.
    for day, rows in (("2015-01-29", (("Real market", real), ("Ours: SS epoch 4", ss),
                                      ("Ours: 0.724", base), ("TRADES DDPM-100", ddpm29),
                                      ("TRADES single-step", single))),
                      ("2015-01-30", (("Real market", real30), ("Ours: SS epoch 4", ss30),
                                      ("Ours: 0.724", base30), ("TRADES DDPM-100", ddpm30)))):
        print(f"\nINTC {day}, 10:00–12:00")
        print("| series | start | end | min | max | range (tk) | ret1s_std (bp) | unique mids |")
        print("|---|---|---|---|---|---|---|---|")
        for lab, s in rows:
            r = np.log(s).diff().dropna()
            print(f"| {lab} | {s.iloc[0]:.2f} | {s.iloc[-1]:.2f} | {s.min():.2f} | {s.max():.2f} | "
                  f"{(s.max()-s.min())*100:.0f} | {r.std()*1e4:.2f} | {s.round(3).nunique()} |")


if __name__ == "__main__":
    main()
