"""
Convert a TRADES/ABIDES `processed_orders.csv` into LOBSTER-format message + orderbook
CSVs, which is what LOB-Bench (peernagy/lob_bench) consumes.

`processed_orders.csv` layout (index = timestamp; 49 columns):
    ORDER_ID, PRICE($), SIZE, BUY_SELL_FLAG(bool), TYPE(str),
    ask_price_1, ask_size_1, bid_price_1, bid_size_1, … ×10 levels (40 cols, LOBSTER units),
    MID_PRICE, SPREAD, ORDER_VOLUME_IMBALANCE, VWAP

LOBSTER message (headerless): time, event_type, order_id, size, price, direction
LOBSTER orderbook (headerless): ask_price_1, ask_size_1, bid_price_1, bid_size_1, … (40 cols)

Mapping:
    time       = seconds since midnight (from the timestamp index)
    event_type = LIMIT_ORDER→1, ORDER_CANCELLED→3, ORDER_EXECUTED→4
    order_id   = ORDER_ID
    size       = SIZE
    price      = PRICE × 10000   (dollars → LOBSTER integer units)
    direction  = BUY_SELL_FLAG → +1 (buy) / −1 (sell)
The 40 LOB columns are already in LOBSTER order and units, so they pass through directly.

Usage (standalone):
    python evaluation/lob_bench/to_lobster.py path/to/processed_orders.csv \
        --out-prefix out/INTC_2015-01-30 [--window 09:45]
    # writes out/INTC_2015-01-30_message.csv and _orderbook.csv
"""

import argparse
import warnings

import numpy as np
import pandas as pd

_TYPE_TO_EVENT = {"LIMIT_ORDER": 1, "ORDER_CANCELLED": 3, "ORDER_EXECUTED": 4}
_LOB_COLS = [f"{side}_{f}_{lvl}"
             for lvl in range(1, 11)
             for side, f in (("ask", "price"), ("ask", "size"), ("bid", "price"), ("bid", "size"))]
_PRICE_SCALE = 10000  # dollars → LOBSTER integer price units


def _seconds_since_midnight(index) -> np.ndarray:
    """Robust timestamp → seconds-since-midnight. Requires a parseable absolute time."""
    idx = pd.to_datetime(index, errors="coerce")
    if idx.isna().all():
        raise ValueError(
            "Could not parse the timestamp index as absolute datetimes. The released "
            "TRADES-LOB files ship a truncated 'MM:SS.f' index and cannot be time-aligned; "
            "run this on a full-datetime processed_orders.csv (any world_agent_sim output).")
    if idx.isna().any():
        warnings.warn(f"{int(idx.isna().sum())} unparseable timestamps dropped.")
    t = (idx - idx.normalize()).total_seconds()
    return t.to_numpy()


def convert(source, window_start: str | None = None):
    """source: path or DataFrame of processed_orders.csv. Returns (message_df, orderbook_df)."""
    df = source if isinstance(source, pd.DataFrame) else pd.read_csv(source, index_col=0)

    if window_start:  # keep only rows at/after HH:MM on the first day (drop the replay warm-up)
        idx = pd.to_datetime(df.index, errors="coerce")
        cutoff = idx[0].normalize() + pd.to_timedelta(window_start + ":00")
        df = df[idx >= cutoff]

    # drop LOBSTER sentinel rows where the top of book is absent (empty book at t0)
    ok = (df["ask_price_1"].abs() < 9_000_000_000) & (df["bid_price_1"].abs() < 9_000_000_000)
    df = df[ok]

    unknown = set(df["TYPE"].unique()) - set(_TYPE_TO_EVENT)
    if unknown:
        warnings.warn(f"Unmapped TYPE values dropped: {unknown}")
        df = df[df["TYPE"].isin(_TYPE_TO_EVENT)]

    message = pd.DataFrame({
        "time": _seconds_since_midnight(df.index),
        "event_type": df["TYPE"].map(_TYPE_TO_EVENT).astype(int).to_numpy(),
        "order_id": df["ORDER_ID"].astype("int64").to_numpy(),
        "size": df["SIZE"].round().astype("int64").to_numpy(),
        "price": (df["PRICE"].to_numpy() * _PRICE_SCALE).round().astype("int64"),
        "direction": np.where(df["BUY_SELL_FLAG"].astype(bool).to_numpy(), 1, -1),
    })

    orderbook = df[_LOB_COLS].round().astype("int64").reset_index(drop=True)
    return message.reset_index(drop=True), orderbook


def convert_to_files(source, out_prefix: str, window_start: str | None = None):
    message, orderbook = convert(source, window_start)
    msg_path, ob_path = f"{out_prefix}_message.csv", f"{out_prefix}_orderbook.csv"
    message.to_csv(msg_path, header=False, index=False)
    orderbook.to_csv(ob_path, header=False, index=False)
    return msg_path, ob_path, len(message)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("processed_csv")
    ap.add_argument("--out-prefix", required=True, help="output path prefix (…_message.csv/_orderbook.csv)")
    ap.add_argument("--window", default=None, help="keep rows at/after HH:MM (drops replay warm-up)")
    ap.add_argument("--self-test", action="store_true", help="print a summary of the converted output and exit")
    args = ap.parse_args()

    msg, ob, n = convert_to_files(args.processed_csv, args.out_prefix, args.window)
    print(f"wrote {msg} and {ob}  ({n} events)")
    if args.self_test:
        m = pd.read_csv(msg, header=None, names=["time", "event_type", "order_id", "size", "price", "direction"])
        o = pd.read_csv(ob, header=None)
        print("message head:\n", m.head(3).to_string())
        print(f"event_type counts: {m['event_type'].value_counts().to_dict()}")
        print(f"time range (s): {m['time'].min():.1f}–{m['time'].max():.1f}   "
              f"price range: {m['price'].min()}–{m['price'].max()}   dir±: {sorted(m['direction'].unique())}")
        print(f"orderbook shape: {o.shape} (expect 40 cols)")


if __name__ == "__main__":
    main()
