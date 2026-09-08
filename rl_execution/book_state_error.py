"""Experiment 0: does ABIDES's own bookkeeping error on REAL order flow stay bounded, or grow
unbounded, the longer a replay runs? No model involved -- this compares two independent
reconstructions of "the book" at regular intervals through a session:

  - ground truth: rl_execution.orderbook_reconstructor replayed straight off the raw LOBSTER
    message CSV for the day. Needs the reconstructor's own crossing/matching logic, because raw
    exchange feeds have gaps (see that module's docstring on "ghost orders").
  - ABIDES's own record: the same day's ABIDES EXCHANGE_AGENT.bz2 order stream (from a
    `world_agent_sim` real-replay run, no diffusion model involved), converted to LOBSTER-message
    format via ABIDES's own `util/formatting/convert_order_stream.py` (reused as-is), then
    replayed with *pure* add/reduce/remove bookkeeping -- no re-derived crossing logic, because
    ABIDES's own stream already reports the outcome of every match it performed internally;
    re-deriving crossings here would double-count fills against the same resting orders.
    (`processed_orders.csv` is NOT used for this side: it's built with
    `ignore_cancellations=True`, which drops cancellation rows -- exactly the events this
    experiment needs to not lose.)

Diffing the two at regular intervals gives, per timestamp: shares missing (ground truth resting
shares minus ABIDES's own resting shares), depth error (top-N-level shares diff), and whether
ABIDES's own book has gone one-sided (a whole side at zero depth) -- the specific pathology this
experiment exists to rule in or out as an ABIDES-vs-model question.

Usage:
    python -m rl_execution.book_state_error \
        --message-csv data/INTC/INTC_2015-01-02_2015-01-30/INTC_2015-01-30_34140000_57660000_message_10.csv \
        --exchange-bz2 ABIDES/log/market_replay_INTC_2015-01-30_10-00-00_30/EXCHANGE_AGENT.bz2 \
        --out exp0_results/2015-01-30/30.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ABIDES")))

from rl_execution.orderbook_reconstructor import (  # noqa: E402
    DEFAULT_MARKET_OPEN, MESSAGE_COLUMNS, RestingOrder, _MatchingBook,
    aggregate_levels, read_messages,
)
from util.formatting.convert_order_stream import convert_stream_to_format  # noqa: E402


def load_abides_own_messages(exchange_bz2_path: str) -> pd.DataFrame:
    """ABIDES's own complete record of every order event it processed for a real-replay run,
    in the same time/event_type/order_id/size/price/direction convention as a raw LOBSTER file.
    """
    stream_df = pd.read_pickle(exchange_bz2_path)
    lobster_df = convert_stream_to_format(stream_df.reset_index(), fmt="LOBSTER")
    lobster_df = lobster_df.rename(columns={
        "Time": "time", "Type": "event_type", "Order ID": "order_id",
        "Size": "size", "Price": "price", "Direction": "direction",
    })
    return lobster_df[MESSAGE_COLUMNS].reset_index(drop=True)


def _snapshot(by_id: dict) -> dict:
    """Deep-enough copy: RestingOrder.size is mutated in place during replay, so a later
    mutation must not silently leak into an earlier stored snapshot.
    """
    return {oid: RestingOrder(o.order_id, o.side, o.price, o.size, o.entry_time)
            for oid, o in by_id.items()}


def _iter_rows(messages: pd.DataFrame):
    cols = messages[["time", "event_type", "order_id", "size", "price", "direction"]]
    return cols.itertuples(index=False, name=None)


def ground_truth_snapshots(messages: pd.DataFrame, sample_times,
                            market_open: float = DEFAULT_MARKET_OPEN) -> dict:
    """Crossing-aware replay (reuses orderbook_reconstructor's own matching book) -- a single
    pass collecting a snapshot at each requested time, instead of that module's own
    reconstruct_book_from_frame called once per sample (which would re-parse from market open
    every time: O(n * samples) instead of O(n)).
    """
    messages = messages.sort_values("time")
    book = _MatchingBook()
    sample_times = sorted(sample_times)
    snapshots, si = {}, 0
    for t, et, oid, size, price, direction in _iter_rows(messages):
        while si < len(sample_times) and t > sample_times[si]:
            snapshots[sample_times[si]] = _snapshot(book.by_id)
            si += 1
        if t < market_open:
            continue
        if et == 1:
            side = "buy" if direction == 1 else "sell"
            book.insert_new(RestingOrder(int(oid), side, int(price), int(size), float(t)))
        elif et in (2, 4):
            book.reduce_or_remove(int(oid), int(size))
        elif et == 3:
            book.remove(int(oid))
    while si < len(sample_times):
        snapshots[sample_times[si]] = _snapshot(book.by_id)
        si += 1
    return snapshots


def bookkeeping_replay_snapshots(messages: pd.DataFrame, sample_times,
                                  market_open: float = DEFAULT_MARKET_OPEN) -> dict:
    """Pure add/reduce/remove replay, no crossing logic -- appropriate for a stream that already
    reports every match's outcome itself (ABIDES's own converted event log), unlike raw LOBSTER
    data which needs orderbook_reconstructor's crossing-aware matching book.
    """
    messages = messages.sort_values("time")
    by_id: dict = {}
    sample_times = sorted(sample_times)
    snapshots, si = {}, 0
    for t, et, oid, size, price, direction in _iter_rows(messages):
        while si < len(sample_times) and t > sample_times[si]:
            snapshots[sample_times[si]] = _snapshot(by_id)
            si += 1
        if t < market_open:
            continue
        if et == 1:
            side = "buy" if direction == 1 else "sell"
            by_id[int(oid)] = RestingOrder(int(oid), side, int(price), int(size), float(t))
        elif et in (2, 4):
            o = by_id.get(int(oid))
            if o is not None:
                o.size -= int(size)
                if o.size <= 0:
                    del by_id[int(oid)]
        elif et == 3:
            by_id.pop(int(oid), None)
    while si < len(sample_times):
        snapshots[sample_times[si]] = _snapshot(by_id)
        si += 1
    return snapshots


def diff_snapshot(gt_book: dict, ab_book: dict, n_levels: int = 10) -> dict:
    gt_shares = sum(o.size for o in gt_book.values())
    ab_shares = sum(o.size for o in ab_book.values())
    gt_levels = aggregate_levels(gt_book, n_levels)
    ab_levels = aggregate_levels(ab_book, n_levels)
    gt_depth10 = sum(sz for _, sz in gt_levels["bids"]) + sum(sz for _, sz in gt_levels["asks"])
    ab_depth10 = sum(sz for _, sz in ab_levels["bids"]) + sum(sz for _, sz in ab_levels["asks"])
    # "one-sided" means ABIDES's own book has gone unquoted on a side while the REAL market at
    # that same instant has not -- both sides being empty at/near market open (nothing has
    # happened yet on either reconstruction) is not the pathology this experiment is looking for.
    gt_one_sided = (len(gt_levels["bids"]) == 0) or (len(gt_levels["asks"]) == 0)
    ab_one_sided = (len(ab_levels["bids"]) == 0) or (len(ab_levels["asks"]) == 0)
    one_sided = ab_one_sided and not gt_one_sided
    return {
        "gt_shares": gt_shares, "ab_shares": ab_shares, "shares_missing": gt_shares - ab_shares,
        "gt_depth10": gt_depth10, "ab_depth10": ab_depth10, "depth_error": gt_depth10 - ab_depth10,
        "one_sided": int(one_sided),
    }


def hhmmss(seconds_since_midnight: float) -> str:
    s = int(round(seconds_since_midnight))
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def run(message_csv: str, exchange_bz2: str, out_csv: str, interval_min: float = 5.0,
        market_open: float = DEFAULT_MARKET_OPEN, market_close: float | None = None) -> pd.DataFrame:
    real_messages = read_messages(message_csv)
    abides_messages = load_abides_own_messages(exchange_bz2)

    end_time = market_close
    if end_time is None:
        end_time = min(real_messages["time"].max(), abides_messages["time"].max())
    sample_times = list(np.arange(market_open, end_time, interval_min * 60.0)) + [end_time]

    gt_snaps = ground_truth_snapshots(real_messages, sample_times, market_open)
    ab_snaps = bookkeeping_replay_snapshots(abides_messages, sample_times, market_open)

    rows = []
    for t in sample_times:
        d = diff_snapshot(gt_snaps[t], ab_snaps[t])
        d["timestamp"] = hhmmss(t)
        d["seconds_since_open"] = t - market_open
        rows.append(d)
    df = pd.DataFrame(rows)[["timestamp", "seconds_since_open", "shares_missing", "depth_error",
                             "one_sided", "gt_shares", "ab_shares", "gt_depth10", "ab_depth10"]]
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--message-csv", required=True, help="raw LOBSTER message CSV for the day (ground truth)")
    ap.add_argument("--exchange-bz2", required=True, help="EXCHANGE_AGENT.bz2 from a world_agent_sim real-replay run")
    ap.add_argument("--out", required=True, help="output CSV path")
    ap.add_argument("--interval-min", type=float, default=5.0)
    ap.add_argument("--market-open", type=float, default=DEFAULT_MARKET_OPEN)
    ap.add_argument("--market-close", type=float, default=None,
                    help="seconds since midnight; default = min(last real msg, last ABIDES msg) time")
    return ap.parse_args()


def main():
    args = parse_args()
    df = run(args.message_csv, args.exchange_bz2, args.out, args.interval_min,
             args.market_open, args.market_close)
    final = df.iloc[-1]
    early = df.iloc[min(2, len(df) - 1)]
    growing = abs(final["shares_missing"]) > 1.5 * max(abs(early["shares_missing"]), 1)
    any_one_sided = bool(df["one_sided"].any())
    print(f"wrote {args.out}  ({len(df)} samples)")
    print(f"  shares_missing: early={early['shares_missing']:.0f} final={final['shares_missing']:.0f}"
          f"  {'GROWING' if growing else 'roughly flat'}")
    print(f"  depth_error final={final['depth_error']:.0f}  one_sided_ever={any_one_sided}")


if __name__ == "__main__":
    main()
