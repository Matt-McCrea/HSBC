"""
check_raw_depth_distribution.py — does REAL LOBSTER data ever produce negative depth at all?

Diagnostic for why an unclamped retrain produced BIT-FOR-BIT IDENTICAL normalization stats to the
clamped baseline (mean_depth=1.3847392706370658, std_depth=2.677663621501452) despite cst.UNCLAMP_DEPTH
confirmed True at both call sites. Two things checked on one real trading day, fast, CPU-only,
independent of the training pipeline:

  A) The REAL preprocess_data() code path (respects the current UNCLAMP_DEPTH flag) — depth stats
     broken down by event_type.
  B) A manual recompute using the PRE-event orderbook snapshot (index=j-1) for EVERY event type,
     instead of preprocess_data's `index = j if event_type==1 else j-1`. Hypothesis: for a new-order
     submission (event_type==1), using the POST-event snapshot (index=j) is self-referential — for a
     marketable order that partially fills and rests its remainder, the "best bid" being compared
     against becomes the order's OWN resting remainder, washing depth to 0 instead of negative. Using
     the pre-event snapshot for all types would avoid this.

If (A) shows ~zero negatives but (B) shows a real negative population, that confirms the indexing
bug as the actual reason clamping never had anything to remove, independent of the UNCLAMP_DEPTH
flag itself (which is correctly wired).

Usage:
    python scripts/check_raw_depth_distribution.py [day_dir] [date]
    (defaults to data/INTC/INTC_2015-01-02_2015-01-30 / 2015-01-30)
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import constants as cst
from utils.utils_data import preprocess_data, reset_indexes


COLUMNS_NAMES = {
    "orderbook": [f"{s}{lvl}" for lvl in range(1, 11) for s in ("sell", "vsell", "buy", "vbuy")],
    "message": ["time", "event_type", "order_id", "size", "price", "direction"],
}


def load_day(day_dir, date):
    msg_path = f"{day_dir}/{date}_34140000_57660000_message_10.csv"
    ob_path = f"{day_dir}/{date}_34140000_57660000_orderbook_10.csv"
    messages = pd.read_csv(msg_path, names=COLUMNS_NAMES["message"], usecols=range(6))
    orderbook = pd.read_csv(ob_path, names=COLUMNS_NAMES["orderbook"])
    return messages, orderbook


def report(name, depth, event_type):
    print(f"\n── {name} ──")
    print(f"n={len(depth)}  min={depth.min()}  max={depth.max()}  mean={depth.mean():.4f}  std={depth.std():.4f}")
    print(f"negative: {(depth < 0).sum()} ({(depth < 0).mean():.2%})   zero: {(depth == 0).sum()} ({(depth == 0).mean():.2%})")
    for et, label in ((1, "LIMIT(1)"), (3, "CANCEL(3)"), (4, "EXEC(4)")):
        mask = event_type == et
        if mask.sum() == 0:
            continue
        d = depth[mask]
        print(f"  {label:<10} n={mask.sum():<8} neg={int((d<0).sum()):<7} ({(d<0).mean():.2%})   "
              f"min={d.min()}  mean={d.mean():.3f}")


def main():
    day_dir = sys.argv[1] if len(sys.argv) > 1 else "data/INTC/INTC_2015-01-02_2015-01-30"
    date = sys.argv[2] if len(sys.argv) > 2 else "2015-01-30"
    print(f"cst.UNCLAMP_DEPTH = {cst.UNCLAMP_DEPTH}")
    print(f"loading {day_dir} / {date}")

    # ── A) the real preprocess_data() path ──
    messages, orderbook = load_day(day_dir, date)
    ob_a, msg_a = preprocess_data([messages.copy(), orderbook.copy()], cst.N_LOB_LEVELS, cst.Models.TRADES)
    report("A) preprocess_data() — current code path", msg_a["depth"].to_numpy(), msg_a["event_type"].to_numpy())

    # ── B) manual recompute, PRE-event snapshot (index=j-1) for every event type ──
    messages, orderbook = load_day(day_dir, date)
    dataframes = reset_indexes([messages, orderbook])
    dataframes[1] = dataframes[1].iloc[:, :cst.N_LOB_LEVELS * cst.LEN_LEVEL]
    drop_idx = dataframes[0][dataframes[0]["event_type"].isin([2, 5, 6, 7])].index
    dataframes[0] = dataframes[0].drop(drop_idx)
    dataframes[1] = dataframes[1].drop(drop_idx)
    dataframes = reset_indexes(dataframes)

    prices = dataframes[0]["price"].values
    directions = dataframes[0]["direction"].values
    event_types = dataframes[0]["event_type"].values
    bid_sides = dataframes[1].iloc[:, 2::4].values
    ask_sides = dataframes[1].iloc[:, 0::4].values
    depths_b = np.zeros(len(prices), dtype=int)
    for j in range(1, len(prices)):
        index = j - 1  # ALWAYS pre-event, unlike preprocess_data's `j if event_type==1 else j-1`
        if directions[j] == 1:
            depths_b[j] = (bid_sides[index, 0] - prices[j]) // 100
        else:
            depths_b[j] = (prices[j] - ask_sides[index, 0]) // 100
    report("B) manual — PRE-event snapshot for ALL event types", depths_b[1:], event_types[1:])

    print("\n── verdict ──")
    neg_a = (msg_a["depth"].to_numpy() < 0).mean()
    neg_b = (depths_b[1:] < 0).mean()
    if neg_a < 0.001 and neg_b > 0.01:
        print(f"CONFIRMED: (A) has ~zero negatives ({neg_a:.3%}) but (B) has a real negative population "
              f"({neg_b:.3%}). The event_type==1 post-event indexing is the actual bug — it's washing "
              f"out marketable orders' negativity independent of the UNCLAMP_DEPTH clamp.")
    elif neg_a < 0.001 and neg_b < 0.001:
        print(f"Neither shows negatives ({neg_a:.3%} / {neg_b:.3%}) — the indexing theory is NOT it; "
              f"negative depth may genuinely be near-absent in this raw data under any indexing choice. "
              f"Needs a different explanation.")
    else:
        print(f"(A) neg={neg_a:.3%}  (B) neg={neg_b:.3%} — inspect both breakdowns above directly.")


if __name__ == "__main__":
    main()
