"""
drift_profile.py — time-bucketed profile of a simulated session: WHAT drives long-horizon drift?

Context: DDIM10+dn0.3+sr+prior is realistic over 30 min (mid inside real's envelope, exec 7.9%
vs real 7.0%) but over 75 min drifts -5% (real: -1.8%) with exec creeping to 11.2% and the
conditioning going OOD late (cond_z[price] -6.86, cond_z[depth] -1259). Two rival explanations:
  (a) over-execution compounds: too many crossings thin the book, each subsequent crossing moves
      price more, drift accelerates  -> fix = feedback-calibrated sigma (hold exec rate at target)
  (b) directional feedback: a buy/sell imbalance in crossings pushes price, the drifted price
      puts conditioning OOD, the OOD model distorts further  -> fix = different (re-anchoring)
The aggregates can't distinguish cause from effect; the TIMELINE can. This prints per-bucket:
mid, spread, exec counts split by aggressor side, book depth at touch, order-flow imbalance —
so drift onset can be lined up against exec-rate rise and side-imbalance.

CPU-only, runs on any processed_orders.csv (works for real market-replay output too — run it on
both and compare columns side by side).

Usage:
    python evaluation/diagnostics/drift_profile.py <processed_orders.csv> [--bucket-min 5]
    python evaluation/diagnostics/drift_profile.py gen.csv --real real.csv   # aligned comparison
"""
import argparse
import sys

import numpy as np
import pandas as pd


def profile(path, bucket_min):
    df = pd.read_csv(path, index_col=0)
    idx = pd.to_datetime(df.index, errors="coerce")
    df = df[~idx.isna()]
    idx = idx[~idx.isna()]
    df["_t"] = idx
    df["MID"] = (df["ask_price_1"] + df["bid_price_1"]) / 2.0
    # LOBSTER sentinels / empty-book rows would poison mins and mids
    ok = (df["ask_price_1"].abs() < 9_000_000_000) & (df["bid_price_1"].abs() < 9_000_000_000) \
         & (df["ask_price_1"] > 0) & (df["bid_price_1"] > 0)
    df = df[ok]

    t0 = df["_t"].iloc[0]
    df["_bucket"] = ((df["_t"] - t0).dt.total_seconds() // (bucket_min * 60)).astype(int)

    rows = []
    for b, g in df.groupby("_bucket"):
        execs = g[g["TYPE"] == "ORDER_EXECUTED"]
        n = len(g)
        # BUY_SELL_FLAG on an execution row is the resting order's side in this log format;
        # the AGGRESSOR is the opposite. aggressive-buy pressure = executions against resting sells.
        exec_vs_sell = int((execs["BUY_SELL_FLAG"] == False).sum())   # noqa: E712 — aggressor bought
        exec_vs_buy = int((execs["BUY_SELL_FLAG"] == True).sum())    # noqa: E712 — aggressor sold
        lim = g[g["TYPE"] == "LIMIT_ORDER"]
        ofi = (lim["BUY_SELL_FLAG"] == True).sum() - (lim["BUY_SELL_FLAG"] == False).sum()  # noqa: E712
        rows.append({
            "min": int(b * bucket_min),
            "n": n,
            "mid_open": g["MID"].iloc[0] / 10000.0,
            "mid_close": g["MID"].iloc[-1] / 10000.0,
            "mid_lo": g["MID"].min() / 10000.0,
            "spread_med": float((g["ask_price_1"] - g["bid_price_1"]).median()) / 100.0,  # ticks
            "exec%": 100.0 * len(execs) / max(1, n),
            "aggr_buy": exec_vs_sell,
            "aggr_sell": exec_vs_buy,
            "lim_ofi": int(ofi),
            "bid1_med": float(g["bid_size_1"].median()),
            "ask1_med": float(g["ask_size_1"].median()),
        })
    return pd.DataFrame(rows)


def render(tag, p):
    print(f"\n══ {tag} ══ (per-bucket; aggr_buy = executions where the aggressor BOUGHT)")
    hdr = (f"{'min':>4} {'events':>7} {'mid_open':>9} {'mid_close':>9} {'mid_lo':>8} "
           f"{'spr(tk)':>8} {'exec%':>6} {'aggrB':>6} {'aggrS':>6} {'B-S':>6} {'limOFI':>7} "
           f"{'bid1':>7} {'ask1':>7}")
    print(hdr); print("-" * len(hdr))
    for _, r in p.iterrows():
        print(f"{r['min']:>4.0f} {r['n']:>7.0f} {r['mid_open']:>9.4f} {r['mid_close']:>9.4f} "
              f"{r['mid_lo']:>8.4f} {r['spread_med']:>8.1f} {r['exec%']:>6.1f} "
              f"{r['aggr_buy']:>6.0f} {r['aggr_sell']:>6.0f} {r['aggr_buy']-r['aggr_sell']:>6.0f} "
              f"{r['lim_ofi']:>7.0f} {r['bid1_med']:>7.0f} {r['ask1_med']:>7.0f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("gen_csv")
    ap.add_argument("--real", default=None, help="real processed_orders.csv for aligned comparison")
    ap.add_argument("--bucket-min", type=int, default=5)
    args = ap.parse_args()

    render("GENERATED: " + args.gen_csv, profile(args.gen_csv, args.bucket_min))
    if args.real:
        render("REAL: " + args.real, profile(args.real, args.bucket_min))

    print("\nREAD: line up WHERE mid_close starts running away against (1) exec% rising and "
          "(2) B-S / limOFI going one-sided and (3) the thin side's touch size collapsing. "
          "If exec% rises FIRST -> over-execution drives it (fix: feedback-calibrated sigma). "
          "If B-S goes one-sided at healthy exec% -> directional feedback (fix: needs re-anchoring "
          "or direction-aware treatment). If spread widens and exec% FALLS while mid jumps -> "
          "book-thinning/teleporting (different again).")


if __name__ == "__main__":
    main()
