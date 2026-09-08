"""Experiment 2/4 scoring — pre-registered freeze/diverge definitions for the instability paper,
fixed in `analysis/instability_paper_log.md` / the execution plan BEFORE any Exp-2 result exists,
per the plan's own instruction to decide this in advance rather than after looking at the data.

    FROZEN:   no new *unique* mid-price level visited within a trailing WINDOW_MIN window.
              (Generalizes scripts/checkpoint_stability.sh's own empirical anchor: "FREEZE =
              uniq_mid <~10" over a fixed 30-minute window, to a rolling window so it applies at
              any horizon.)
    DIVERGED: EITHER (a) one side's top-of-book size is 0 for a continuous >= ONE_SIDED_SEC span
              (the literal "unquoted side" pathology), OR (b) mid price departs from the run's own
              arrival mid by more than RANGE_MULTIPLE x the real day's own high-low range.
              (Loosened from scripts/long_session_stability.sh's "~2x real" 90-minute envelope,
              since Exp 2 runs much longer.)

A run that hits neither survives to the end of its horizon -- that's a data point too, not a
discard.

Usage:
    python -m rl_execution.survival_metrics --gen <processed_orders.csv> --real <real-replay processed_orders.csv>
"""
from __future__ import annotations  # remote box runs Python 3.9 -- `float | None` needs this to not crash at def-time

import argparse
import json

import numpy as np
import pandas as pd

WINDOW_MIN = 10.0
ONE_SIDED_SEC = 60.0
RANGE_MULTIPLE = 3.0


def _load(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["dt"] = pd.to_datetime(df.iloc[:, 0], errors="coerce")
    for c in ("ask_price_1", "bid_price_1", "ask_size_1", "bid_size_1"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["dt"])
    # invalid/sentinel LOB rows (no quote yet, or an overflow sentinel) must be dropped BEFORE
    # computing mid -- otherwise a single garbage row can poison arrival_mid or the day range.
    df = df[(df["ask_price_1"].abs() < 9e9) & (df["bid_price_1"].abs() < 9e9)
            & (df["ask_price_1"] > 0) & (df["bid_price_1"] > 0)]
    df["mid"] = (df["ask_price_1"] + df["bid_price_1"]) / 2 / 10000.0
    return df.sort_values("dt").reset_index(drop=True)


def real_day_range(real_csv: str) -> float:
    """The real day's own high-low mid-price range (dollars) -- the yardstick for 'departed'."""
    df = _load(real_csv)
    return float(df["mid"].max() - df["mid"].min())


def time_to_freeze(df: pd.DataFrame, window_min: float = WINDOW_MIN):
    """First time by which `window_min` have elapsed with no new unique mid level. None if it
    never happens.
    """
    if df.empty:
        return None
    mids = df["mid"].round(3).to_numpy()
    times = df["dt"].to_numpy()
    seen = set()
    last_new_time = times[0]
    for t, m in zip(times, mids):
        if m not in seen:
            seen.add(m)
            last_new_time = t
        elif (t - last_new_time) / np.timedelta64(1, "m") >= window_min:
            return pd.Timestamp(last_new_time) + pd.Timedelta(minutes=window_min)
    return None


def time_to_diverge(df: pd.DataFrame, real_day_range_dollars: float,
                     range_multiple: float = RANGE_MULTIPLE, one_sided_sec: float = ONE_SIDED_SEC,
                     arrival_mid: float | None = None):
    """Returns (time_or_None, reason_or_None) -- the EARLIER of a price-departure crossing and a
    continuous one-sided span, since both count as "diverged."
    """
    if df.empty:
        return None, None
    if arrival_mid is None:
        arrival_mid = float(df["mid"].iloc[0])
    threshold = range_multiple * max(real_day_range_dollars, 1e-9)

    departed = df[(df["mid"] - arrival_mid).abs() > threshold]
    price_t = departed["dt"].iloc[0] if len(departed) else None

    bid = df.get("bid_size_1", pd.Series(np.nan, index=df.index)).fillna(0)
    ask = df.get("ask_size_1", pd.Series(np.nan, index=df.index)).fillna(0)
    one_sided_mask = (bid <= 0) | (ask <= 0)
    onesided_t = None
    run_start = None
    for i in range(len(df)):
        if one_sided_mask.iloc[i]:
            if run_start is None:
                run_start = df["dt"].iloc[i]
            elif (df["dt"].iloc[i] - run_start) / np.timedelta64(1, "s") >= one_sided_sec:
                onesided_t = run_start
                break
        else:
            run_start = None

    candidates = [(t, r) for t, r in ((price_t, "price_departure"), (onesided_t, "one_sided"))
                  if t is not None]
    if not candidates:
        return None, None
    candidates.sort(key=lambda tr: tr[0])
    return candidates[0]


def score_run(gen_csv: str, real_csv: str) -> dict:
    df = _load(gen_csv)
    rng = real_day_range(real_csv)
    freeze_t = time_to_freeze(df)
    diverge_t, diverge_reason = time_to_diverge(df, rng)
    return {
        "time_to_freeze": str(freeze_t) if freeze_t is not None else None,
        "time_to_diverge": str(diverge_t) if diverge_t is not None else None,
        "diverge_reason": diverge_reason,
        "survived_to_end": freeze_t is None and diverge_t is None,
        "real_day_range": rng,
        "n_rows": int(len(df)),
        "last_timestamp": str(df["dt"].iloc[-1]) if len(df) else None,
    }


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen", required=True, help="generated run's processed_orders.csv")
    ap.add_argument("--real", required=True, help="matching real-replay processed_orders.csv (for the day's own range)")
    ap.add_argument("--out", default=None, help="optional JSON output path")
    return ap.parse_args()


def main():
    args = parse_args()
    result = score_run(args.gen, args.real)
    print(json.dumps(result, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
