"""
Run LOB-Bench (peernagy/lob_bench) on a TRADES simulation vs real market data.

Both inputs are `processed_orders.csv` files (the world_agent_sim output format): one for the
real market-replay, one for the generated simulation. This script converts each to LOBSTER
message+orderbook CSVs (via to_lobster), lays them out in the folder structure LOB-Bench's
`Simple_Loader` expects, and calls the benchmark.

    python evaluation/lob_bench/run_lob_bench.py \
        --real ABIDES/log/market_replay_INTC_2015-01-30_10-00-00_30/processed_orders.csv \
        --gen  ABIDES/log/world_agent_INTC_..._DDPM_.../processed_orders.csv \
        --out-dir lob_bench_run --window 09:45 --n-splits 1

By default one whole-session sequence goes in each folder (point estimates, no error bars).
Use --n-splits N to chop the session into N time-contiguous sequences (restores bootstrapped
error bars; no conditional-generation machinery needed). --prepare-only stops after writing the
LOBSTER folders (guaranteed to work) without invoking the lob_bench API.

Install the benchmark first:  pip install -r requirements-eval.txt
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from to_lobster import convert  # noqa: E402


def _date_str(processed_csv):
    idx = pd.to_datetime(pd.read_csv(processed_csv, index_col=0, nrows=1).index, errors="coerce")
    return idx[0].strftime("%Y-%m-%d") if not idx.isna().all() else "0000-00-00"


def _write_splits(message, orderbook, out_dir, name_fn, n_splits):
    """Write n contiguous (message, orderbook) sequence pairs using name_fn(kind, seq_id)->filename."""
    os.makedirs(out_dir, exist_ok=True)
    bounds = np.linspace(0, len(message), n_splits + 1).astype(int)
    for k in range(n_splits):
        lo, hi = bounds[k], bounds[k + 1]
        message.iloc[lo:hi].to_csv(os.path.join(out_dir, name_fn("message", k)), header=False, index=False)
        orderbook.iloc[lo:hi].to_csv(os.path.join(out_dir, name_fn("orderbook", k)), header=False, index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--real", required=True, help="real market-replay processed_orders.csv")
    ap.add_argument("--gen", required=True, help="generated simulation processed_orders.csv")
    ap.add_argument("--out-dir", default="lob_bench_run")
    ap.add_argument("--window", default="09:45", help="keep rows at/after HH:MM (drop replay warm-up)")
    ap.add_argument("--n-splits", type=int, default=1, help="sequences per folder (1 = whole session)")
    ap.add_argument("--prepare-only", action="store_true", help="write LOBSTER folders, skip the benchmark call")
    args = ap.parse_args()

    date = _date_str(args.real)
    real_dir = os.path.join(args.out_dir, "real")
    gen_dir = os.path.join(args.out_dir, "generated")

    # NOTE: filename convention reverse-engineered from Simple_Loader's globs
    #   real:  *message*.csv / *orderbook*.csv ; date = basename.split('_')[1];
    #          real_id = (substring after 'message').split('_')[0].split('.')[0]
    #   gen:   *{date}*message*real_id_{id}_gen_id_*.csv
    # If Simple_Loader fails to pair files, adjust these two lambdas (and re-run --prepare-only).
    real_name = lambda kind, k: f"INTC_{date}_{kind}{k:02d}.csv"
    gen_name = lambda kind, k: f"INTC_{date}_{kind}_real_id_{k:02d}_gen_id_00.csv"

    print(f"Converting real  → {real_dir}")
    rm, ro = convert(args.real, window_start=args.window)
    _write_splits(rm, ro, real_dir, real_name, args.n_splits)
    print(f"Converting gen   → {gen_dir}")
    gm, go = convert(args.gen, window_start=args.window)
    _write_splits(gm, go, gen_dir, gen_name, args.n_splits)
    print(f"  real events={len(rm)}  gen events={len(gm)}  splits={args.n_splits}")
    print(f"  example files: {real_name('message',0)} | {gen_name('message',0)}")

    if args.prepare_only:
        print("\n--prepare-only: LOBSTER folders written. To benchmark:")
        print("  from lob_bench import data_loading, scoring")
        print(f"  loader = data_loading.Simple_Loader('{real_dir}', '{gen_dir}', '{gen_dir}')")
        print("  results = scoring.run_benchmark(loader, score_cfg, metric_cfg)")
        return

    # ── Invoke the benchmark ──────────────────────────────────────────────────
    try:
        from lob_bench import data_loading, scoring
    except Exception as e:  # noqa: BLE001
        print(f"\n[!] Could not import lob_bench ({e}).")
        print("    pip install -r requirements-eval.txt  — or run with --prepare-only and call it yourself.")
        return

    # cond_path is optional; we pass gen_dir as a harmless placeholder (we don't use conditional scores).
    loader = data_loading.Simple_Loader(real_dir, gen_dir, gen_dir)

    # score_cfg / metric_cfg structure is defined by lob_bench — start from their README/example
    # (e.g. lob_bench's default configs) and pass here. Left as None so the failure is explicit
    # rather than silently wrong.
    score_cfg = getattr(scoring, "DEFAULT_SCORE_CFG", None)
    metric_cfg = getattr(scoring, "DEFAULT_METRIC_CFG", None)
    if score_cfg is None or metric_cfg is None:
        print("\n[!] lob_bench imported and LOBSTER folders are ready, but this script does not know "
              "your score_cfg/metric_cfg. Fill them from lob_bench's example, then call:")
        print(f"      loader = data_loading.Simple_Loader('{real_dir}', '{gen_dir}', '{gen_dir}')")
        print("      scoring.run_benchmark(loader, score_cfg, metric_cfg)")
        return

    print("\nRunning benchmark…")
    results = scoring.run_benchmark(loader, score_cfg, metric_cfg)
    out = os.path.join(args.out_dir, "results.txt")
    with open(out, "w") as f:
        f.write(str(results))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
