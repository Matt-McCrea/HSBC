"""
Batch LOB-Bench: score several TRADES samplers against real data in one comparative table.

Runs entirely inside lob_bench's venv (which has numpy+pandas, so the converter works too):

    external/lob_bench_env/bin/python evaluation/lob_bench/score_batch.py \
        --real-lobster data/INTC/INTC_2015-01-02_2015-01-30_10/INTC_2015-01-30_34140000_57660000_orderbook_10.csv \
        --gen DDPM=sweep_results/DDPM_100/DDPM_100_generated_orders.csv \
        --gen DDIM10=sweep_results/DDIM_10/DDIM_10_generated_orders.csv \
        --gen DPMpp=sweep_results/DPM_SOLVER_PP_10/DPM_SOLVER_PP_10_generated_orders.csv \
        --out-dir lob_bench_batch --window 09:45

Prints a distance table (rows = score/metric, cols = samplers; lower = closer to real) and
writes it to <out-dir>/lob_bench_scores.csv. Each generated CSV's real data is sliced to that
generated session's own window, so comparisons are apples-to-apples.
"""

import argparse
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from to_lobster import convert                                   # noqa: E402
from run_lob_bench import _load_lobster_pair, _slice_time, _write_splits, _date_str  # noqa: E402


def _build_score_cfg(lbe):
    """The metrics confirmed to run on TRADES output (single-Series eval fns)."""
    return {
        "spread":              {"fn": lambda m, b: lbe.spread(m, b).values, "discrete": True},
        "log_inter_arrival":   {"fn": lambda m, b: np.log(
                                    lbe.inter_arrival_time(m).replace({0: 1e-9}).values.astype(float))},
        "orderbook_imbalance": {"fn": lambda m, b: lbe.orderbook_imbalance(m, b).values},
        "orderflow_imbalance": {"fn": lambda m, b: lbe.orderflow_imbalance(m, b).values},
        "limit_depth_ask":     {"fn": lambda m, b: lbe.limit_order_depth(m, b)[0].values, "discrete": True},
        "cancel_depth_ask":    {"fn": lambda m, b: lbe.cancellation_depth(m, b)[0].values, "discrete": True},
    }


def _prep(real_arg, real_is_lobster, gen_csv, out_dir, window, n_splits):
    date = _date_str(gen_csv)
    real_dir, gen_dir = os.path.join(out_dir, "data_real"), os.path.join(out_dir, "data_gen")
    real_name = lambda kind, k: f"INTC_{date}_{kind}_real_id_{k:02d}.csv"
    gen_name = lambda kind, k: f"INTC_{date}_{kind}_real_id_{k:02d}_gen_id_00.csv"
    gm, go = convert(gen_csv, window_start=window)
    t_lo, t_hi = float(gm["time"].min()), float(gm["time"].max())
    if real_is_lobster:
        rm, ro = _load_lobster_pair(real_arg)
    else:
        rm, ro = convert(real_arg, window_start=window)
    rm, ro = _slice_time(rm, ro, t_lo, t_hi)
    for d in (real_dir, gen_dir):
        shutil.rmtree(d, ignore_errors=True)
    _write_splits(rm, ro, real_dir, real_name, n_splits)
    _write_splits(gm, go, gen_dir, gen_name, n_splits)
    return real_dir, gen_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--real")
    ap.add_argument("--real-lobster")
    ap.add_argument("--gen", action="append", required=True, metavar="LABEL=PATH",
                    help="generated processed_orders.csv, labelled (repeatable)")
    ap.add_argument("--out-dir", default="lob_bench_batch")
    ap.add_argument("--window", default="09:45")
    ap.add_argument("--n-splits", type=int, default=1)
    ap.add_argument("--lob-bench-path", default="external/lob_bench")
    ap.add_argument("--metric", default="wasserstein", choices=["wasserstein", "l1"],
                    help="which distance to tabulate (both are computed)")
    args = ap.parse_args()
    if not (args.real or args.real_lobster):
        ap.error("supply --real or --real-lobster")

    sys.path.insert(0, args.lob_bench_path)
    import data_loading, scoring, metrics       # noqa: E402
    import eval as lbe                            # noqa: E402
    import warnings; warnings.filterwarnings("ignore")

    metric_cfg = {"l1": metrics.l1_by_group, "wasserstein": metrics.wasserstein}
    score_cfg = _build_score_cfg(lbe)

    results = {}   # label -> {score -> {metric -> (est, lo, hi)}}
    for spec in args.gen:
        if "=" not in spec:
            ap.error(f"--gen must be LABEL=PATH, got {spec!r}")
        label, gen_csv = spec.split("=", 1)
        print(f"\n=== {label} : {gen_csv} ===")
        real_dir, gen_dir = _prep(args.real_lobster or args.real, bool(args.real_lobster),
                                  gen_csv, args.out_dir, args.window, args.n_splits)
        loader = data_loading.Simple_Loader(real_dir, gen_dir, gen_dir)
        scores, _, _ = scoring.run_benchmark(loader, score_cfg, default_metric=metric_cfg)
        results[label] = {name: {m: (v[0], v[1][0], v[1][1]) for m, v in md.items()}
                          for name, md in scores.items()}

    # ── comparative table (rows = score, cols = samplers; the chosen metric) ──
    labels = list(results)
    score_names = list(score_cfg)
    w = max(18, max(len(s) for s in score_names) + 2)
    print(f"\n\n{args.metric.upper()} distance to real  (lower = more realistic)\n")
    print("score".ljust(w) + "".join(f"{l:>14}" for l in labels))
    print("-" * (w + 14 * len(labels)))
    for s in score_names:
        row = s.ljust(w)
        for l in labels:
            est = results[l].get(s, {}).get(args.metric, (float("nan"),))[0]
            row += f"{est:>14.4f}"
        print(row)

    # ── CSV (both metrics + CIs) ──
    out_csv = os.path.join(args.out_dir, "lob_bench_scores.csv")
    with open(out_csv, "w") as f:
        f.write("sampler,score,metric,distance,ci_lo,ci_hi\n")
        for l in labels:
            for s in score_names:
                for m in ("wasserstein", "l1"):
                    est, lo, hi = results[l].get(s, {}).get(m, (float("nan"),) * 3)
                    f.write(f"{l},{s},{m},{est:.6f},{lo:.6f},{hi:.6f}\n")
    print(f"\nsaved {out_csv}")


if __name__ == "__main__":
    main()
