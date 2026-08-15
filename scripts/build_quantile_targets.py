"""
build_quantile_targets.py — extract real-data target distributions for quantile matching.

Prerequisite for a quantile-matching fix (whichever channel — depth or size — tonight's
DDIM10-vs-DDPM comparison points us toward): the TARGET half of the transform. The target must
be REAL data, not DDPM's own output — DDPM has its own known biases (e.g. resting depth
systematically deeper than real, per the LOB-Bench finding), so matching to DDPM would import
DDPM's distortions rather than fix them.

Pure CPU, no GPU needed — safe to run alongside training/eval jobs. Processes every trading day
found in the given LOBSTER directory (globs *_message_10.csv), reusing the exact preprocessing
pipeline (utils_data.preprocess_data) so the extracted values are computed identically to what
training itself sees — including this session's index-bug fix (pre-event snapshot, not post-event).

Saves sorted float32 .npy arrays to --out-dir, ready for np.interp-based quantile remapping:
  real_depth_limit.npy    — signed depth, LIMIT(1) events only (the only type with genuine
                             negative/marketable depth; CANCEL/EXEC are structurally ~0, see
                             check_raw_depth_distribution.py's earlier findings)
  real_size_limit.npy, real_size_cancel.npy, real_size_market.npy
                           — size per decoded type, restricted to [0,1000] to match the model's
                             own representable/valid output range (sizes are capped/dropped
                             beyond this in WorldAgent, so an unbounded real target would include
                             values the model can never actually produce)

Usage:
    python scripts/build_quantile_targets.py \
        --day-dir data/INTC/INTC_2015-01-02_2015-01-30 --out-dir data/quantile_targets
"""
import argparse
import glob
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import constants as cst
from utils.utils_data import preprocess_data


COLUMNS_NAMES = {
    "orderbook": [f"{s}{lvl}" for lvl in range(1, 11) for s in ("sell", "vsell", "buy", "vbuy")],
    "message": ["time", "event_type", "order_id", "size", "price", "direction"],
}


def find_days(day_dir):
    """Discover (date, message_path) pairs from *message*10.csv files in day_dir.

    Handles BOTH LOBSTER naming conventions seen in the wild (and on the remote):
      2015-01-30_34140000_57660000_message_10.csv          (bare date first)
      INTC_2015-01-30_34140000_57660000_message_10.csv     (ticker-prefixed)
    The original re.match(r'^date_') silently found ZERO files against the prefixed
    convention, which crashed every reshape cell of the 2026-07-14 overnight run.
    De-dupes by date (same day present under both namings counts once)."""
    by_date = {}
    for f in sorted(glob.glob(os.path.join(day_dir, "*message*10.csv"))):
        m = re.search(r"(\d{4}-\d{2}-\d{2})", os.path.basename(f))
        if m and m.group(1) not in by_date:
            by_date[m.group(1)] = f
    return sorted(by_date.items())


def load_day(msg_path):
    """Load a (message, orderbook) pair given the message path; the orderbook path is derived
    by substring replacement so the pairing can never mismatch across naming conventions."""
    ob_path = msg_path.replace("message", "orderbook")
    if not os.path.isfile(ob_path):
        return None, None
    messages = pd.read_csv(msg_path, names=COLUMNS_NAMES["message"], usecols=range(6))
    orderbook = pd.read_csv(ob_path, names=COLUMNS_NAMES["orderbook"])
    return messages, orderbook


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--day-dir", default="data/INTC/INTC_2015-01-02_2015-01-30")
    ap.add_argument("--out-dir", default="data/quantile_targets")
    ap.add_argument("--stock", default="INTC")
    args = ap.parse_args()

    print(f"cst.UNCLAMP_DEPTH = {cst.UNCLAMP_DEPTH}  (should be True — targets must reflect signed depth)")
    days = find_days(args.day_dir)
    if not days:
        print(f"ERROR: no *message*10.csv files found in {args.day_dir}")
        sys.exit(1)   # nonzero so shell callers actually abort (silent exit-0 broke the 07-14 night)
    print(f"found {len(days)} trading days: {days[0][0]} .. {days[-1][0]}")

    depth_limit = []
    size_by_type = {1: [], 3: [], 4: []}

    for date, msg_path in days:
        messages, orderbook = load_day(msg_path)
        if messages is None:
            print(f"  skip {date} (orderbook pair not found for {msg_path})")
            continue
        _, msg = preprocess_data([messages.copy(), orderbook.copy()], cst.N_LOB_LEVELS, cst.Models.TRADES)
        et = msg["event_type"].to_numpy()
        depth = msg["depth"].to_numpy()
        size = msg["size"].to_numpy()
        depth_limit.append(depth[et == 1])
        for t in (1, 3, 4):
            size_by_type[t].append(size[et == t])
        print(f"  {date}: n={len(msg)}  limit_depth_neg={(depth[et==1]<0).sum()}")

    if not depth_limit:
        print("ERROR: every trading day was skipped — no target data extracted")
        sys.exit(1)
    os.makedirs(args.out_dir, exist_ok=True)

    depth_limit = np.sort(np.concatenate(depth_limit).astype(np.float32))
    out = os.path.join(args.out_dir, "real_depth_limit.npy")
    np.save(out, depth_limit)
    print(f"\nsaved {out}  n={len(depth_limit)}  neg={(depth_limit<0).sum()} "
          f"({(depth_limit<0).mean():.3%})  min={depth_limit.min():.0f}  max={depth_limit.max():.0f}")

    for t, name in ((1, "limit"), (3, "cancel"), (4, "market")):
        sizes = np.concatenate(size_by_type[t]).astype(np.float32)
        valid = sizes[(sizes >= 0) & (sizes <= 1000)]   # match the model's representable range
        out = os.path.join(args.out_dir, f"real_size_{name}.npy")
        np.save(out, np.sort(valid))
        dropped = len(sizes) - len(valid)
        print(f"saved {out}  n={len(valid)}  (dropped {dropped} outside [0,1000], "
              f"{dropped/len(sizes):.1%})  mean={valid.mean():.1f}  std={valid.std():.1f}")

    print(f"\nAll targets in {args.out_dir}/ — sorted float32 arrays, ready for "
          f"np.interp(quantile, linspace(0,1,len(target)), target) quantile remapping.")


if __name__ == "__main__":
    main()
