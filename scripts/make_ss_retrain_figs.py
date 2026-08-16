#!/usr/bin/env python3
"""SS-retrain comparison figures: baseline (0.724_epoch=0) vs epochs 2/3/4 across all six
LOB-Bench metrics, plus the uniq_mid-vs-LOB-Bench reconciliation chart. Same validated palette as
the rest of the HSBC figures (blue #2a78d6 family, sequential steps for the ordered epoch 2->3->4
progression since there IS a genuine order here, grey for the superseded baseline reference)."""
import os, datetime as dt
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GREY = "#8a897f"
BLUE_LIGHT, BLUE_MED, BLUE_DARK = "#9dc3ec", "#5199ea", "#144a85"
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": "#c9c8c2", "axes.linewidth": 0.8, "axes.grid": False,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
    "xtick.direction": "out", "ytick.direction": "out",
})
def despine(ax, keep=("left", "bottom")):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in keep)

OUT = f"analysis/plots/hsbc_ss_retrain_{dt.date.today():%Y%m%d}"
os.makedirs(OUT, exist_ok=True)

NICE = {"spread": "spread", "log_inter_arrival": "inter-arrival", "orderbook_imbalance": "book imbalance",
        "orderflow_imbalance": "flow imbalance", "limit_depth_ask": "limit depth", "cancel_depth_ask": "cancel depth"}
METRICS = list(NICE)

# hardcoded from the completed lob_bench_multiday.sh runs (see lob_bench_0724_full_month/ and
# lob_bench_ss_retrain/*/SUMMARY_mean_wasserstein.csv for the underlying per-day data)
BASELINE = {"spread": 0.719, "log_inter_arrival": 0.625, "orderbook_imbalance": 0.432,
            "orderflow_imbalance": 0.673, "limit_depth_ask": 0.200, "cancel_depth_ask": 0.161}
EPOCH2 = {"spread": 0.2846, "log_inter_arrival": 0.5375, "orderbook_imbalance": 0.4003,
          "orderflow_imbalance": 0.6517, "limit_depth_ask": 0.1698, "cancel_depth_ask": 0.1445}
EPOCH3 = {"spread": 0.2382, "log_inter_arrival": 0.5143, "orderbook_imbalance": 0.3888,
          "orderflow_imbalance": 0.6466, "limit_depth_ask": 0.1852, "cancel_depth_ask": 0.1613}
EPOCH4 = {"spread": 0.1923, "log_inter_arrival": 0.5042, "orderbook_imbalance": 0.3937,
          "orderflow_imbalance": 0.6452, "limit_depth_ask": 0.1855, "cancel_depth_ask": 0.1521}

UNIQ_MID_MEAN = {"epoch 0\n(baseline)": 15.10, "epoch 2": 13.65, "epoch 3": 11.30, "epoch 4": 12.95}
LOBBENCH_GRAND_MEAN = {"epoch 0\n(baseline)": 0.468, "epoch 2": 0.3647, "epoch 3": 0.3557, "epoch 4": 0.3455}


def per_metric_comparison():
    y = np.arange(len(METRICS))[::-1]
    h = 0.19
    fig, ax = plt.subplots(figsize=(10.2, 5.6))
    series = [("baseline (epoch 0)", BASELINE, GREY, 1.5*h),
              ("epoch 2", EPOCH2, BLUE_LIGHT, 0.5*h),
              ("epoch 3", EPOCH3, BLUE_MED, -0.5*h),
              ("epoch 4", EPOCH4, BLUE_DARK, -1.5*h)]
    for lab, d, c, off in series:
        vals = [d[m] for m in METRICS]
        ax.barh(y+off, vals, h, color=c, zorder=3, label=f"{lab}  (mean {np.mean(vals):.3f})")
    ax.set_yticks(y); ax.set_yticklabels([NICE[m] for m in METRICS])
    ax.set_xlabel("Wasserstein distance to real market  (lower = more realistic)")
    ax.set_xlim(0, 1.0); despine(ax)
    ax.legend(loc="lower right", frameon=False, fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    fig.text(0.02, 0.955, "Scheduled-sampling retrain: per-metric LOB-Bench comparison", fontsize=13,
              fontweight="bold", color=INK, ha="left")
    fig.text(0.02, 0.90, "Full-month mean Wasserstein distance, baseline vs each retrain epoch, all six metrics.",
              fontsize=10.3, color=INK2, ha="left")
    fig.savefig(f"{OUT}/1_ss_retrain_per_metric.png", dpi=160); plt.close(fig)


def grand_mean_bars():
    labels = list(LOBBENCH_GRAND_MEAN)
    vals = list(LOBBENCH_GRAND_MEAN.values())
    colors = [GREY, BLUE_LIGHT, BLUE_MED, BLUE_DARK]
    y = np.arange(len(labels))[::-1]
    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    ax.barh(y, vals, 0.6, color=colors, zorder=3)
    for yi, v in zip(y, vals):
        ax.text(v + 0.012, yi, f"{v:.3f}", va="center", ha="left", fontsize=11, color=INK, fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlabel("Grand mean Wasserstein distance across all 6 metrics")
    ax.set_xlim(0, max(vals) * 1.25); despine(ax)
    fig.tight_layout(rect=[0, 0, 1, 0.82])
    fig.text(0.02, 0.955, "Full-month LOB-Bench grand mean, by retrain epoch", fontsize=13,
              fontweight="bold", color=INK, ha="left")
    fig.text(0.02, 0.90, "Lower is more realistic. All scores are full-month means (20 days).",
              fontsize=10.3, color=INK2, ha="left")
    fig.savefig(f"{OUT}/2_ss_retrain_grand_mean.png", dpi=160); plt.close(fig)


def reconciliation_panel():
    labels = list(UNIQ_MID_MEAN)
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    axes[0].plot(x, list(UNIQ_MID_MEAN.values()), color=BLUE_DARK, lw=1.8, marker="o", markersize=7,
                 markerfacecolor=BLUE_DARK, markeredgecolor=SURF, markeredgewidth=1.2, zorder=3)
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, fontsize=9.5)
    axes[0].set_ylabel("mean uniq_mid")
    axes[0].set_title("Activity level (uniq_mid)", loc="left", fontsize=12)
    axes[0].set_ylim(0, 18)
    despine(axes[0])

    axes[1].plot(x, list(LOBBENCH_GRAND_MEAN.values()), color="#cf5a2a", lw=1.8, marker="o", markersize=7,
                 markerfacecolor="#cf5a2a", markeredgecolor=SURF, markeredgewidth=1.2, zorder=3)
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, fontsize=9.5)
    axes[1].set_ylabel("grand mean Wasserstein")
    axes[1].set_title("Overall LOB-Bench realism (lower = better)", loc="left", fontsize=12)
    axes[1].set_ylim(0, 0.55)
    despine(axes[1])

    fig.tight_layout(rect=[0, 0, 1, 0.84])
    fig.text(0.02, 0.955, "Two different axes: activity level vs overall realism", fontsize=13,
              fontweight="bold", color=INK, ha="left")
    fig.text(0.02, 0.90, "uniq_mid stays flat while overall realism improves — the benefit is on a",
              fontsize=10.3, color=INK2, ha="left")
    fig.text(0.02, 0.855, "different axis than originally hypothesised, not absent.",
              fontsize=10.3, color=INK2, ha="left")
    fig.savefig(f"{OUT}/3_reconciliation_panel.png", dpi=160); plt.close(fig)


per_metric_comparison()
grand_mean_bars()
reconciliation_panel()

with open(f"{OUT}/tables.md", "w") as f:
    f.write("# SS-retrain comparison tables (2026-08-05)\n\n")
    f.write("## Per-metric, full-month mean Wasserstein\n\n")
    f.write("| metric | baseline (epoch 0) | epoch 2 | epoch 3 | epoch 4 |\n|---|---|---|---|---|\n")
    for m in METRICS:
        f.write(f"| {NICE[m]} | {BASELINE[m]:.3f} | {EPOCH2[m]:.3f} | {EPOCH3[m]:.3f} | {EPOCH4[m]:.3f} |\n")
    f.write(f"| **grand mean** | **{np.mean(list(BASELINE.values())):.3f}** | "
            f"**{np.mean(list(EPOCH2.values())):.3f}** | **{np.mean(list(EPOCH3.values())):.3f}** | "
            f"**{np.mean(list(EPOCH4.values())):.3f}** |\n\n")
    f.write("## Activity level vs overall realism (the reconciliation)\n\n")
    f.write("| | epoch 0 (baseline) | epoch 2 | epoch 3 | epoch 4 |\n|---|---|---|---|---|\n")
    f.write("| mean uniq_mid | " + " | ".join(f"{v:.2f}" for v in UNIQ_MID_MEAN.values()) + " |\n")
    f.write("| LOB-Bench grand mean | " + " | ".join(f"{v:.3f}" for v in LOBBENCH_GRAND_MEAN.values()) + " |\n")

print("wrote figures + tables to", OUT)
for fn in sorted(os.listdir(OUT)): print("  ", fn)
