"""Offline analysis of the JSON-lines run logs (deliverable 7's payoff: the
Results chapter should be writable from saved logs without rerunning anything).

Pure pandas/numpy -- no GPU, no model, no simulation. Safe to run anywhere.

    python -m rl_execution.analyze_logs logs/train_sell.jsonl
    python -m rl_execution.analyze_logs logs/train_sell.jsonl --episode 56
    python -m rl_execution.analyze_logs logs/eval_final.jsonl --by policy_name

Reports robust statistics alongside the mean/stderr, because the shortfall
distribution has heavy tails: a single episode where the simulated book drains
and the terminal sweep walks an empty book can be ~100x a typical episode and
will dominate any mean. Reporting median and a trimmed mean next to the mean
is what makes the tail visible rather than silently baked into one number.
"""

import argparse
import json

import numpy as np
import pandas as pd

from rl_execution.logging_utils import read_episodes


def load(path) -> pd.DataFrame:
    df = pd.DataFrame(read_episodes(path))
    if "shortfall_bps" not in df.columns or df["shortfall_bps"].isna().all():
        # older logs predate the bps field; derive it (both inputs are logged)
        df["shortfall_bps"] = df["shortfall"] / df["p_arrival"] * 10_000.0
    df["episode"] = np.arange(1, len(df) + 1)
    return df


def _robust_stats(x: pd.Series) -> dict:
    x = x.dropna()
    n = len(x)
    if n == 0:
        return {}
    stats = {
        "n": n,
        "mean": x.mean(),
        "stderr": x.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0,
        "median": x.median(),
        "iqr_lo": x.quantile(0.25),
        "iqr_hi": x.quantile(0.75),
        "min": x.min(),
        "max": x.max(),
    }
    if n >= 5:  # 10% trimmed mean, i.e. mean with the extreme tails removed
        lo, hi = x.quantile(0.05), x.quantile(0.95)
        stats["trimmed_mean"] = x[(x >= lo) & (x <= hi)].mean()
    return stats


def _fmt(stats: dict, label: str) -> str:
    if not stats:
        return f"  {label}: no data"
    out = (f"  {label:28s} n={stats['n']:3d}  mean={stats['mean']:10.2f}  "
           f"stderr={stats['stderr']:9.2f}  median={stats['median']:9.2f}  "
           f"IQR=[{stats['iqr_lo']:.1f}, {stats['iqr_hi']:.1f}]")
    if "trimmed_mean" in stats:
        out += f"  trimmed_mean={stats['trimmed_mean']:9.2f}"
    return out


def summarize(df: pd.DataFrame, by=None):
    print("=" * 100)
    print(f"{len(df)} episodes")
    print("=" * 100)

    print("\nSHORTFALL (bps of arrival mid; negative = better than arrival)")
    if by and by in df.columns:
        for key, grp in df.groupby(by):
            print(_fmt(_robust_stats(grp["shortfall_bps"]), str(key)))
    else:
        print(_fmt(_robust_stats(df["shortfall_bps"]), "all"))

    print("\nWALL-CLOCK (s per episode)")
    for col, label in (("wall_clock_total_s", "total"),
                       ("wall_clock_reconstruct_s", "reconstruct"),
                       ("wall_clock_simulate_s", "simulate")):
        if col in df.columns:
            print(_fmt(_robust_stats(df[col]), label))

    print("\nMARKET DIAGNOSTICS")
    for col in ("execution_rate", "unique_mid_count", "n_resting_orders"):
        if col in df.columns:
            print(_fmt(_robust_stats(df[col]), col))

    _outliers(df)


def _outliers(df: pd.DataFrame, k=3.0):
    """Flag episodes whose shortfall sits far outside the interquartile range --
    these are the ones worth opening individually (see --episode), because they
    are usually a drained book rather than a trading decision.
    """
    x = df["shortfall_bps"].dropna()
    if len(x) < 5:
        return
    q1, q3 = x.quantile(0.25), x.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    out = df[(df["shortfall_bps"] < lo) | (df["shortfall_bps"] > hi)]
    print(f"\nOUTLIERS ({len(out)} of {len(df)} episodes outside {k}x IQR)")
    if out.empty:
        print("  none")
        return
    cols = [c for c in ("episode", "shortfall_bps", "seed_day", "t0", "side", "Q",
                        "execution_rate", "unique_mid_count", "n_resting_orders") if c in out.columns]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(out[cols].to_string(index=False))
    contribution = out["shortfall_bps"].sum() / df["shortfall_bps"].sum() if df["shortfall_bps"].sum() else float("nan")
    print(f"  these episodes account for {contribution:.1%} of the summed shortfall "
          f"-- compare mean vs median/trimmed_mean above before quoting a mean")


def show_episode(df: pd.DataFrame, episode: int):
    row = df[df["episode"] == episode]
    if row.empty:
        print(f"no episode {episode} in this log ({len(df)} episodes)")
        return
    rec = row.iloc[0].to_dict()
    fills = rec.pop("fills", None)
    print(json.dumps({k: v for k, v in rec.items()}, indent=2, default=str))
    if fills:
        f = np.array(fills, dtype=float)
        qty, px = f[:, 0], f[:, 1]
        print(f"\nfills: {len(f)} executions, total qty={qty.sum():.0f}")
        print(f"  price range: {px.min():.0f} .. {px.max():.0f}  (arrival mid {rec.get('p_arrival')})")
        print(f"  qty-weighted avg fill: {(qty * px).sum() / qty.sum():.1f}")
        worst = f[np.argmax(np.abs(px - float(rec.get("p_arrival", 0))))]
        print(f"  furthest fill from arrival: qty={worst[0]:.0f} @ {worst[1]:.0f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log", help="path to a .jsonl run log")
    parser.add_argument("--episode", type=int, default=None, help="dump one episode in full")
    parser.add_argument("--by", default=None, help="group shortfall stats by a column, e.g. policy_name")
    args = parser.parse_args()

    df = load(args.log)
    if args.episode is not None:
        show_episode(df, args.episode)
    else:
        summarize(df, by=args.by)
