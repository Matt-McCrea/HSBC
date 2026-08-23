#!/usr/bin/env python3
"""Price-path, variance-ratio and flow statistics for the vanilla DDPM-100 TRADES-default runs.

The two runs added here (INTC 2015-01-29 and 2015-01-30, 10:00-12:00, checkpoint val_ema=0.667,
DDPM with 100 steps and NO decode-time flags) are the first genuine replication of the published
TRADES sampling configuration at the two-hour horizon. Every earlier "TRADES" row in the results
folder was either a step-count ablation on OUR checkpoint (0.724, which carries PRICE_REANCHOR and
UNCLAMP_DEPTH) or the DDIM-1 single-step run — neither is the paper's default.

Columns follow analysis/plots/longhorizon/summary.md, plus the variance-ratio pair that
variance_ratio_analysis.md established must travel together:

  VR(q) = Var(q-period return) / (q * Var(1-period return))

  nz%   = share of 1s bars with a non-zero return. A sparse jump series is biased toward VR=1
          MECHANICALLY — with few non-zero returns the q-period variance is dominated by the same
          handful of jumps as the 1-period variance. Quoting VR without nz% invites reading a
          frozen book as a well-behaved random walk, which is the exact error the vanilla 0.956
          reading produced earlier in this project (55 non-zero returns out of 877).
"""
import os
import sys

import numpy as np
import pandas as pd

OUT = "analysis/plots/longhorizon"
B = "ABIDES/log/paper_runs_downloaded"
LOG = "ABIDES/log"

# (day, {label: path}). The real reference is built by lobster_real_reference.py, NOT the ABIDES
# market replay — they differ slightly and the rest of the project quotes the LOBSTER-derived one.
DAYS = {
    "2015-01-29": {
        "Real market":          f"{OUT}/real_0129_1000_1200.csv",
        "Ours: SS epoch 4":     f"{B}/world_agent_INTC_2015-01-29_12-00-00_30_DDIM_0.0_10_val_ema=0.69__tdprior_sr_dn0.3/processed_orders.csv",
        "Ours: 0.724 baseline": f"{B}/world_agent_INTC_2015-01-29_12-00-00_30_DDIM_0.0_10_val_ema=0.724_tdprior_sr_dn0.3/processed_orders.csv",
        "TRADES DDPM-100":      f"{B}/world_agent_INTC_2015-01-29_12-00-00_30_DDPM_0.0_100_val_ema=0.667/processed_orders.csv",
        "TRADES single-step":   f"{LOG}/world_agent_INTC_2015-01-29_12-00-00_30_DDIM_0.0_1_val_ema=0.763/processed_orders.csv",
    },
    "2015-01-30": {
        "Real market":          f"{OUT}/real_0130_1000_1200.csv",
        "Ours: SS epoch 4":     f"{B}/world_agent_INTC_2015-01-30_12-00-00_30_DDIM_0.0_10_val_ema=0.69__tdprior_sr_dn0.3/processed_orders.csv",
        "Ours: 0.724 baseline": f"{B}/world_agent_INTC_2015-01-30_12-00-00_30_DDIM_0.0_10_val_ema=0.724_tdprior_sr_dn0.3/processed_orders.csv",
        "TRADES DDPM-100":      f"{B}/world_agent_INTC_2015-01-30_12-00-00_30_DDPM_0.0_100_val_ema=0.667/processed_orders.csv",
    },
}

SENTINEL = 9e9      # ABIDES writes +/-9999999999 into empty book levels


def load(path):
    """-> (1s mid series in dollars, flow-composition Series or None)."""
    df = pd.read_csv(path)
    df["dt"] = pd.to_datetime(df.iloc[:, 0], errors="coerce")
    for c in ("ask_price_1", "bid_price_1"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop rows where either side of the book is empty. MID_PRICE in the raw file is computed
    # straight off the sentinels on those rows (giving values like -499983), so it cannot be used.
    d = df[(df.ask_price_1 > 0) & (df.bid_price_1 > 0)
           & (df.ask_price_1.abs() < SENTINEL) & (df.bid_price_1.abs() < SENTINEL)].copy()
    d["mid"] = (d.ask_price_1 + d.bid_price_1) / 2 / 10000.0
    d = d.dropna(subset=["dt", "mid"])
    mid = d.set_index("dt")["mid"].resample("1s").last().ffill().dropna()

    flow = None
    if "TYPE" in df.columns:
        flow = df["TYPE"].value_counts(normalize=True)
    return mid, flow


WARMUP_S = 15 * 60      # analysis/plots/longhorizon/variance_ratio_analysis.md, "The measure"


def vr(mid, q, warmup=WARMUP_S):
    """Variance ratio at q seconds, on 1s log returns, after discarding a warm-up.

    The warm-up discard is NOT optional book-keeping — it is the established convention in
    variance_ratio_analysis.md (15 min for two-hour sessions, 5 min for thirty-minute ones) and it
    changes the answer by an order of magnitude. The opening minutes are dominated by the model
    settling out of its conditioning block: on SS epoch 4 the first 14 minutes contain a swing
    larger than the remaining 106, which inflates Var(1-period) and drags VR toward 1. Computing VR
    over the full session gives 0.521 where the recorded post-warm-up figure is 0.047.

    Price-path columns (start/end/min/max/range) stay FULL-session, matching the headline table in
    README.md and what the figures actually plot. The two conventions coexist in this project by
    design; the tables label which is which.
    """
    r = np.log(mid.iloc[warmup:]).diff().dropna()
    if len(r) < q * 3:
        return np.nan
    v1 = r.var()
    vq = r.rolling(q).sum().dropna().var()
    return np.nan if v1 == 0 else float(vq / (q * v1))


def row(label, mid, flow):
    r = np.log(mid).diff().dropna()
    # nz% qualifies the VR columns, so it is measured on the same post-warm-up sample they are.
    rw = np.log(mid.iloc[WARMUP_S:]).diff().dropna()
    nz = float((rw != 0).mean() * 100)
    ex = float(flow.get("ORDER_EXECUTED", 0.0) * 100) if flow is not None else np.nan
    return {
        "series": label,
        "start": mid.iloc[0], "end": mid.iloc[-1],
        "min": mid.min(), "max": mid.max(),
        "range_tk": (mid.max() - mid.min()) * 100,
        "ret1s_bp": r.std() * 1e4,
        "uniq_mid": mid.round(3).nunique(),
        "vr10": vr(mid, 10), "vr60": vr(mid, 60), "vr300": vr(mid, 300),
        "nz_pct": nz,
        "exec_pct": ex,
        "bars": len(mid),
    }


def emit(day, rows):
    print(f"\n### INTC {day}, 10:00-12:00 (2 h)\n")
    print("| series | start | end | min | max | range (tk) | ret1s_std (bp) | unique mids "
          "| VR(10s) | VR(60s) | VR(300s) | nz % | exec % |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for x in rows:
        f = lambda v, p=2: "--" if pd.isna(v) else f"{v:.{p}f}"
        print(f"| {x['series']} | {x['start']:.2f} | {x['end']:.2f} | {x['min']:.2f} | "
              f"{x['max']:.2f} | {x['range_tk']:.0f} | {x['ret1s_bp']:.2f} | {x['uniq_mid']} | "
              f"{f(x['vr10'],3)} | {f(x['vr60'],3)} | {f(x['vr300'],3)} | "
              f"{x['nz_pct']:.1f} | {f(x['exec_pct'],1)} |")


def main():
    allrows = {}
    for day, series in DAYS.items():
        rows = []
        for label, path in series.items():
            if not os.path.exists(path):
                print(f"  !! missing, skipped: {label}  ({path})", file=sys.stderr)
                continue
            mid, flow = load(path)
            rows.append(row(label, mid, flow))
        allrows[day] = rows
        emit(day, rows)

    # Flow composition, separately: it is the execution-share argument and deserves its own table.
    print("\n### Flow composition (share of generated events)\n")
    print("| day | series | limit % | cancel % | market % |")
    print("|---|---|---|---|---|")
    for day, series in DAYS.items():
        for label, path in series.items():
            if not os.path.exists(path):
                continue
            _, flow = load(path)
            if flow is None:
                continue
            g = lambda k: float(flow.get(k, 0.0) * 100)
            print(f"| {day} | {label} | {g('LIMIT_ORDER'):.1f} | "
                  f"{g('ORDER_CANCELLED'):.1f} | {g('ORDER_EXECUTED'):.1f} |")

    pd.DataFrame([r | {"day": d} for d, rs in allrows.items() for r in rs]).to_csv(
        f"{OUT}/ddpm_default_stats.csv", index=False)
    print(f"\nwrote {OUT}/ddpm_default_stats.csv")


if __name__ == "__main__":
    main()
