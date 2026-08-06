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


RUN_GAP_SECONDS = 1800.0  # a gap this large between consecutive rows implies a separate run


def load(path) -> pd.DataFrame:
    df = pd.DataFrame(read_episodes(path))
    if "shortfall_bps" not in df.columns or df["shortfall_bps"].isna().all():
        # older logs predate the bps field; derive it (both inputs are logged)
        df["shortfall_bps"] = df["shortfall"] / df["p_arrival"] * 10_000.0
    df["run_id"] = _assign_runs(df)
    # Episode numbers restart per run, matching what the training loop printed. Numbering
    # straight through the file is wrong whenever a filename was reused across runs, and
    # silently points every "episode N" lookup at the wrong row.
    df["episode"] = df.groupby("run_id").cumcount() + 1
    df["row"] = np.arange(1, len(df) + 1)
    return df


def _assign_runs(df: pd.DataFrame) -> pd.Series:
    if "run_id" in df.columns and df["run_id"].notna().all():
        return df["run_id"]
    # Logs written before run_id existed: fall back to splitting on large time gaps.
    ts = pd.to_numeric(df.get("timestamp"), errors="coerce")
    if ts.isna().all():
        return pd.Series(["run1"] * len(df), index=df.index)
    boundary = ts.diff().fillna(0) > RUN_GAP_SECONDS
    inferred = "inferred" + (boundary.cumsum() + 1).astype(str)
    if "run_id" in df.columns:
        return df["run_id"].fillna(pd.Series(inferred, index=df.index))
    return pd.Series(inferred, index=df.index)


def decode_cond_z(rec) -> dict:
    """cond_z is stored as the WorldAgent's raw accumulator [min, max, sum, count]
    per channel; report it as the min/mean/max the DIAG lines print."""
    out = {}
    for chan, v in (rec or {}).items():
        if isinstance(v, (list, tuple)) and len(v) == 4:
            lo, hi, total, n = v
            out[chan] = {"min": lo, "mean": (total / n if n else float("nan")), "max": hi, "n": n}
    return out


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
    print(f"{len(df)} rows in file")
    runs = df.groupby("run_id", sort=False)
    if len(runs) > 1:
        print(f"WARNING: {len(runs)} separate runs share this file (append-only log, reused filename).")
        print("         Episode numbers below restart per run, matching the training loop's output.")
        print("         Use --run <id> to analyse one, or --last-run for the most recent.")
        for rid, grp in runs:
            print(f"           {rid}: {len(grp)} episodes (file rows {grp['row'].min()}-{grp['row'].max()})")
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

    _drift(df)

    print("\nMARKET DIAGNOSTICS")
    for col in ("execution_rate", "unique_mid_count", "n_resting_orders"):
        if col in df.columns:
            print(_fmt(_robust_stats(df[col]), col))

    _outliers(df)


def _drift(df: pd.DataFrame):
    """Separate 'where the market went' from 'how well the policy traded'.

    With a single-sided (e.g. SELL-only) design, any directional drift in the
    generated price systematically flatters or penalises the agent -- and TRADES has
    a documented directional-drift failure mode. If shortfall tracks drift closely,
    the shortfall level is mostly the market moving, not the policy choosing well.
    """
    if "drift_bps" not in df.columns or df["drift_bps"].isna().all():
        print("\nMARKET DRIFT: not logged in this run "
              "(p_final added later; drift cannot be separated from execution here)")
        return
    d = df["drift_bps"].dropna()
    print("\nMARKET DRIFT over the episode (bps; +ve = price rose, which favours a SELLER)")
    print(_fmt(_robust_stats(d), "drift"))
    both = df[["shortfall_bps", "drift_bps"]].dropna()
    if len(both) >= 3:
        r = both["shortfall_bps"].corr(both["drift_bps"])
        print(f"  correlation(shortfall, drift) = {r:+.3f}")
        if abs(r) > 0.6:
            print("  -> shortfall is largely explained by where the market went, not by the policy;")
            print("     quote the POLICY-VS-BENCHMARK difference rather than the absolute level.")


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
    cols = [c for c in ("episode", "row", "shortfall_bps", "seed_day", "t0", "side", "Q",
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
    parser.add_argument("--run", default=None, help="analyse only this run_id (see the warning header)")
    parser.add_argument("--last-run", action="store_true", help="analyse only the most recent run in the file")
    args = parser.parse_args()

    df = load(args.log)
    if args.run:
        df = df[df["run_id"] == args.run]
        if df.empty:
            raise SystemExit(f"no rows with run_id={args.run}")
    elif args.last_run:
        last = df["run_id"].iloc[-1]
        df = df[df["run_id"] == last]
        print(f"(showing only the most recent run: {last}, {len(df)} episodes)\n")
    if args.episode is not None:
        show_episode(df, args.episode)
    else:
        summarize(df, by=args.by)
