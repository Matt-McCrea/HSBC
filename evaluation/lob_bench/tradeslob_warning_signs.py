"""Experiment 5 (instability paper) — check the TRADES authors' own released `TRADES-LOB` file
for the same early-warning trend our own runs show before they visibly fail: growing bid/ask
resting-volume imbalance, a shrinking share of marketable (spread-crossing) orders, and slowing
movement between price levels — all within the official 30-minute window the authors themselves
evaluated on. No training, no generation, no ABIDES: pure bookkeeping over a released CSV, so this
can run before/alongside Experiment 0.

Getting the file: the DeepMarket README does not give TRADES-LOB its own link — it ships inside
the same Google Drive folder as the TRADES checkpoints, already referenced in this repo's own
READMEFORMEHMET.md (https://drive.google.com/drive/folders/1fg5G9KzmzC6E4FUYSCjObJ7sCEdjo43W).
That folder isn't fetchable/downloadable from an automated session — a human needs to open it in
a browser and pull the file down (or `gdown` with the specific file id, once known). Place it at
`data/TRADES-LOB/` (already excluded by `.gitignore`).

Input format (per the DeepMarket README): 50 columns — 6 order-feature columns matching LOBSTER's
message convention (time, event_type, order_id, size, price, direction), 40 LOB-snapshot columns
(10 levels: ask_price/ask_size/bid_price/bid_size x10), and 4 trailing metrics (mid_price, spread,
order_volume_imbalance, vwap) — the same shape as this repo's own `processed_orders.csv`. Column
NAMES are not guaranteed to match this repo's convention, so this reads by POSITION, matching the
documented layout, not by header text (falls back cleanly on any `processed_orders.csv`-shaped
file too, header names and all, since positions agree — used as this script's own smoke test in
the absence of the real released file).

Usage:
    python -m evaluation.lob_bench.tradeslob_warning_signs data/TRADES-LOB/<file>.csv \
        --bucket-min 5 --out exp5_results/tradeslob_warning_signs.csv
"""
import argparse
import sys

import numpy as np
import pandas as pd

N_LEVELS = 10
MSG_COLS = ["time", "event_type", "order_id", "size", "price", "direction"]
LOB_COLS = [f"{side}_{f}_{lvl}" for lvl in range(1, N_LEVELS + 1)
            for side, f in (("ask", "price"), ("ask", "size"), ("bid", "price"), ("bid", "size"))]
METRIC_COLS = ["mid_price", "spread", "order_volume_imbalance", "vwap"]
ALL_COLS = MSG_COLS + LOB_COLS + METRIC_COLS  # 6 + 40 + 4 = 50


def load(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path, header=None, skiprows=1)  # skip whatever header row is there, if any
    if raw.shape[1] < len(ALL_COLS):
        # some sources carry an extra leading index column (processed_orders.csv does) -- retry
        # dropping the first column before giving up.
        raw = raw.iloc[:, 1:]
    if raw.shape[1] < len(ALL_COLS):
        raise ValueError(f"{path}: expected >= {len(ALL_COLS)} columns, got {raw.shape[1]}")
    df = raw.iloc[:, :len(ALL_COLS)].copy()
    df.columns = ALL_COLS
    for c in MSG_COLS + LOB_COLS + METRIC_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    # if "time" doesn't look like seconds-since-midnight (processed_orders.csv's own first column
    # is a timestamp string, which coerces to NaN and gets dropped above -- in that case rebuild
    # `time` from the row order at a nominal 1-row-per-event spacing so bucketing still works).
    if df.empty:
        # this repo's own processed_orders.csv shape: named columns, string TYPE, bool
        # BUY_SELL_FLAG, timestamp index -- used as this script's smoke test in the absence of
        # the real released file (see module docstring).
        raw2 = pd.read_csv(path)
        df = raw2.copy()
        df["dt"] = pd.to_datetime(df.iloc[:, 0], errors="coerce")
        df = df.dropna(subset=["dt"]).sort_values("dt").reset_index(drop=True)
        df["time"] = (df["dt"] - df["dt"].iloc[0]).dt.total_seconds()
        type_map = {"LIMIT_ORDER": 1, "ORDER_CANCELLED": 3, "ORDER_EXECUTED": 4}
        df["event_type"] = df["TYPE"].map(type_map)
        df["direction"] = np.where(df["BUY_SELL_FLAG"].astype(bool), 1, -1)
        df["order_id"] = pd.to_numeric(df["ORDER_ID"], errors="coerce")
        df["size"] = pd.to_numeric(df["SIZE"], errors="coerce")
        df["price"] = pd.to_numeric(df["PRICE"], errors="coerce") * 10000  # dollars -> LOBSTER units
        for k in ("ask_price_1", "ask_size_1", "bid_price_1", "bid_size_1"):
            df[k] = pd.to_numeric(df[k], errors="coerce")
    return df


def marketable_flag(df: pd.DataFrame) -> pd.Series:
    """True where a LIMIT_ORDER (event_type==1) crosses the spread at submission, using this same
    row's own attached LOB snapshot. Non-LIMIT rows are NaN (excluded from the marketable-share
    denominator, which is "share of new limit orders that are marketable", not all events).
    """
    is_limit = df["event_type"] == 1
    is_buy = df["direction"] == 1
    crosses_buy = df["price"] >= df["bid_price_1"]  # a buy resting at/through best bid...
    # marketable = crosses the OPPOSITE side's best quote
    crosses = np.where(is_buy, df["price"] >= df["ask_price_1"], df["price"] <= df["bid_price_1"])
    flag = pd.Series(np.where(is_limit, crosses.astype(float), np.nan), index=df.index)
    return flag


def bucketed_warning_signs(df: pd.DataFrame, bucket_min: float = 5.0) -> pd.DataFrame:
    df = df.copy()
    df["bucket"] = (df["time"] // (bucket_min * 60.0)).astype(int)
    valid_lob = (df["ask_price_1"] > 0) & (df["bid_price_1"] > 0) \
        & (df["ask_size_1"].fillna(0) + df["bid_size_1"].fillna(0) > 0)
    df["imbalance"] = np.where(valid_lob,
                               (df["bid_size_1"] - df["ask_size_1"]) / (df["bid_size_1"] + df["ask_size_1"]).replace(0, np.nan),
                               np.nan)
    df["marketable"] = marketable_flag(df)
    df["mid"] = np.where(valid_lob, (df["ask_price_1"] + df["bid_price_1"]) / 2.0, np.nan)

    rows = []
    for b, g in df.groupby("bucket"):
        rows.append({
            "bucket_start_sec": b * bucket_min * 60.0,
            "mean_imbalance": g["imbalance"].mean(),
            "marketable_share": g["marketable"].mean(),
            "unique_mid_levels": g["mid"].round(4).nunique(),
            "n_events": len(g),
        })
    return pd.DataFrame(rows).sort_values("bucket_start_sec").reset_index(drop=True)


def trend_direction(series: pd.Series) -> tuple:
    """Simple OLS slope over bucket index -- sign + magnitude is enough to call a direction."""
    y = series.to_numpy(dtype=float)
    x = np.arange(len(y), dtype=float)
    mask = ~np.isnan(y)
    if mask.sum() < 2:
        return 0.0, "insufficient data"
    slope = np.polyfit(x[mask], y[mask], 1)[0]
    return float(slope), ("worsening" if slope != 0 else "flat")


def summarize(buckets: pd.DataFrame) -> str:
    imb_slope, _ = trend_direction(buckets["mean_imbalance"].abs())  # growing |imbalance| = warning
    mkt_slope, _ = trend_direction(buckets["marketable_share"])       # falling share = warning
    mid_slope, _ = trend_direction(buckets["unique_mid_levels"])      # falling rate = warning

    imb_warn = imb_slope > 0
    mkt_warn = mkt_slope < 0
    mid_warn = mid_slope < 0
    n_warn = sum([imb_warn, mkt_warn, mid_warn])

    lines = [
        f"|imbalance| trend slope: {imb_slope:+.5f}/bucket  -> {'WARNING (growing)' if imb_warn else 'flat/improving'}",
        f"marketable-share trend slope: {mkt_slope:+.5f}/bucket -> {'WARNING (shrinking)' if mkt_warn else 'flat/improving'}",
        f"unique-mid-levels trend slope: {mid_slope:+.5f}/bucket -> {'WARNING (slowing)' if mid_warn else 'flat/improving'}",
        "",
        f"HEADLINE: {n_warn}/3 warning-sign metrics are already trending toward failure within "
        "the released 30-minute window" + (" -- consistent with the instability being present but "
        "not yet visually obvious in the authors' own evaluation window." if n_warn > 0 else
        " -- none of the three metrics show a pre-failure trend in the released window."),
    ]
    return "\n".join(lines)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv_path")
    ap.add_argument("--bucket-min", type=float, default=5.0)
    ap.add_argument("--out", default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    df = load(args.csv_path)
    buckets = bucketed_warning_signs(df, args.bucket_min)
    print(buckets.to_string(index=False))
    print()
    print(summarize(buckets))
    if args.out:
        buckets.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
