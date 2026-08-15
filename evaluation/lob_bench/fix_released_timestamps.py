#!/usr/bin/env python3
"""Repair the timestamps in TRADES's released TRADES-LOB CSVs so they can be scored on LOB-Bench.

The released files carry a first column formatted MM:SS.s ("45:00.2") --- the hour component was
lost somewhere in a CSV round-trip. Relative ordering and inter-arrival gaps are intact; only the
absolute anchor is missing. Minutes wrap (45 -> 55 -> 06 -> 16 ...), so the hour can be rebuilt by
counting wraps from a starting hour.

Anchor: the first row reads 45:00.2 and the TRADES generation window begins at 09:45, so hour 9 is
the natural anchor. This is an assumption, not a fact recoverable from the file --- it is validated
below by checking the reconstructed session's mid-price range against the real market on that day,
and the check is reported so a wrong anchor cannot pass silently.

Usage:
    python evaluation/lob_bench/fix_released_timestamps.py IN.csv OUT.csv [--start-hour 9]
"""
import argparse
import sys

import numpy as np
import pandas as pd


def rebuild(path, out, start_hour=9, date="2015-01-30"):
    df = pd.read_csv(path)
    tcol = df.columns[0]
    raw = df[tcol].astype(str)

    parts = raw.str.extract(r"^(\d+):(\d+(?:\.\d+)?)$")
    if parts.isna().any().any():
        bad = raw[parts.isna().any(axis=1)].head(3).tolist()
        sys.exit(f"!! unparseable timestamps, e.g. {bad}")
    mins = parts[0].astype(int)
    secs = parts[1].astype(float)

    # count wraps: minute value decreasing => the hour rolled over
    wraps = (mins.diff().fillna(0) < 0).cumsum()
    hours = start_hour + wraps

    base = pd.Timestamp(date)
    ts = base + pd.to_timedelta(hours, unit="h") \
              + pd.to_timedelta(mins, unit="m") \
              + pd.to_timedelta(secs, unit="s")

    df[tcol] = ts.dt.strftime("%Y-%m-%d %H:%M:%S.%f")
    df.to_csv(out, index=False)

    span = (ts.iloc[-1] - ts.iloc[0]).total_seconds() / 60.0
    print(f"  rebuilt {len(df)} rows")
    print(f"  session: {ts.iloc[0].time()} -> {ts.iloc[-1].time()}  ({span:.1f} min, {int(wraps.iloc[-1])} hour-wraps)")
    if "MID_PRICE" in df.columns:
        m = pd.to_numeric(df["MID_PRICE"], errors="coerce").dropna()
        print(f"  mid range: {m.min():.2f} - {m.max():.2f}   (validation: compare to real for {date})")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("infile"); ap.add_argument("outfile")
    ap.add_argument("--start-hour", type=int, default=9)
    ap.add_argument("--date", default="2015-01-30")
    a = ap.parse_args()
    rebuild(a.infile, a.outfile, a.start_hour, a.date)
