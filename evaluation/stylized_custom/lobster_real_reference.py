#!/usr/bin/env python3
"""Build a real-market reference CSV for any day/window straight from the raw LOBSTER files.

Motivation: the stylized-fact and diagnostic tooling expects an ABIDES-style processed_orders.csv
for the real side, but market-replay runs only exist locally for a couple of days and windows. When
the real reference covers a different window from the generated session, every distributional panel
silently compares two different market periods --- which is easy to miss and invalidates the figure.

This reads the LOBSTER message+orderbook pair directly and emits a CSV with the columns the tooling
needs, restricted to the requested window, so real and generated can always be aligned.

LOBSTER message columns: time (seconds after midnight), type, order_id, size, price, direction
    type 1 = new limit, 2 = partial cancel, 3 = total delete, 4 = visible execution,
         5 = hidden execution, 6 = cross, 7 = halt

Usage:
    python evaluation/stylized_custom/lobster_real_reference.py \
        --date 2015-01-29 --st 10:00:00 --et 12:00:00 --out real_0129.csv
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

# INTC January 2015 remains the default so existing callers are unaffected. For any other ticker or
# period, pass --day-dir, or set TICKER/TRADING_START/TRADING_END (the cross-ticker pipeline exports
# these) and the directory is resolved from them.
DEFAULT_DIR = "data/INTC/INTC_2015-01-02_2015-01-30_10"
TYPE_MAP = {1: "LIMIT_ORDER", 2: "ORDER_CANCELLED", 3: "ORDER_CANCELLED",
            4: "ORDER_EXECUTED", 5: "ORDER_EXECUTED", 6: "ORDER_EXECUTED", 7: "HALT"}


def hhmmss_to_sec(s):
    h, m, sec = (int(x) for x in s.split(":"))
    return h * 3600 + m * 60 + sec


def resolve_day_dir(ticker, day_dir=None):
    """Directory holding the LOBSTER pairs. Explicit --day-dir wins, then the env-driven layout
    (canonical or level-suffixed), then the INTC default."""
    if day_dir:
        return day_dir
    start, end = os.environ.get("TRADING_START"), os.environ.get("TRADING_END")
    if start and end:
        base = f"data/{ticker}/{ticker}_{start}_{end}"
        if os.path.isdir(base):
            return base
        for cand in sorted(glob.glob(base + "_*")):
            if os.path.isdir(cand):
                return cand
    return DEFAULT_DIR


def _find_pair(day_dir, ticker, date, levels):
    """Locate the message/orderbook pair by globbing, so the session-seconds stamp in the filename
    (34140000_57660000 on INTC, different elsewhere) does not have to be known or hardcoded."""
    dashed = date if "-" in date else f"{date[:4]}-{date[4:6]}-{date[6:]}"
    out = []
    for kind in ("message", "orderbook"):
        hits = sorted(glob.glob(os.path.join(day_dir, f"{ticker}_{dashed}_*_{kind}_{levels}.csv")))
        if not hits:
            raise SystemExit(
                f"!! no {kind} file for {ticker} {dashed} (levels={levels}) in {day_dir}\n"
                f"   contents: {', '.join(sorted(os.listdir(day_dir))[:4]) if os.path.isdir(day_dir) else 'DIR MISSING'} ...")
        out.append(hits[0])
    return out


def build(date, st, et, out, ticker="INTC", levels=10, day_dir=None):
    day_dir = resolve_day_dir(ticker, day_dir)
    msg_p, ob_p = _find_pair(day_dir, ticker, date, levels)
    print(f"[real-ref] {ticker} {date} {st}-{et}  <- {os.path.basename(msg_p)}")

    msg = pd.read_csv(msg_p, header=None,
                      names=["time", "type", "oid", "size", "price", "dir"], usecols=[0, 1, 2, 3, 4, 5])
    ob = pd.read_csv(ob_p, header=None)
    ncols = 4 * levels
    ob = ob.iloc[:, :ncols]
    cols = []
    for i in range(1, levels + 1):
        cols += [f"ask_price_{i}", f"ask_size_{i}", f"bid_price_{i}", f"bid_size_{i}"]
    ob.columns = cols

    df = pd.concat([msg.reset_index(drop=True), ob.reset_index(drop=True)], axis=1)
    lo, hi = hhmmss_to_sec(st), hhmmss_to_sec(et)
    df = df[(df.time >= lo) & (df.time < hi)].copy()
    if df.empty:
        raise SystemExit(f"!! no rows in {date} {st}-{et}")

    df["TYPE"] = df["type"].map(TYPE_MAP).fillna("OTHER")
    df["SIZE"] = df["size"]
    df["PRICE"] = df["price"] / 10000.0
    df["BUY_SELL_FLAG"] = df["dir"] > 0
    df["ORDER_ID"] = df["oid"]
    ts = pd.Timestamp(date) + pd.to_timedelta(df["time"], unit="s")
    df.insert(0, "", ts.dt.strftime("%Y-%m-%d %H:%M:%S.%f"))

    ok = (df.ask_price_1 > 0) & (df.bid_price_1 > 0)
    df = df[ok]
    df["MID_PRICE"] = (df.ask_price_1 + df.bid_price_1) / 2 / 10000.0
    df["SPREAD"] = (df.ask_price_1 - df.bid_price_1) / 10000.0

    keep = [""] + ["ORDER_ID", "PRICE", "SIZE", "BUY_SELL_FLAG", "TYPE"] + cols + ["MID_PRICE", "SPREAD"]
    df[keep].to_csv(out, index=False)

    vc = df["TYPE"].value_counts(normalize=True) * 100
    print(f"  wrote {out}")
    print(f"  {date} {st}-{et}   rows={len(df)}   mid {df.MID_PRICE.min():.2f}-{df.MID_PRICE.max():.2f}")
    print("  flow: " + "  ".join(f"{k} {v:.1f}%" for k, v in vc.items()))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--st", required=True)
    ap.add_argument("--et", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ticker", default="INTC")
    ap.add_argument("--day-dir", default=None,
                    help="directory of LOBSTER pairs; else resolved from TICKER/TRADING_START/TRADING_END")
    ap.add_argument("--levels", type=int, default=10)
    a = ap.parse_args()
    build(a.date, a.st, a.et, a.out, a.ticker, a.levels, a.day_dir)
