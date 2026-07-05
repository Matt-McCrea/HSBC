#!/usr/bin/env python3
"""
Evaluate one generated processed_orders.csv against real market-replay data.

Runs all quantitative (KL/JS) and qualitative (stylized facts, order type,
spread distribution) metrics and saves results to --out-dir tagged by --tag.

Usage
-----
    python evaluation/sweep_eval.py \
        --real  ABIDES/log/paper/market_replay_.../processed_orders.csv \
        --gen   ABIDES/log/world_agent_.../processed_orders.csv \
        --tag   DPM_SOLVER_PP_10 \
        --out-dir sweep_results/2026-07-03/
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Repo root is one level up from evaluation/
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

parser = argparse.ArgumentParser()
parser.add_argument("--real",    default=None, help="Path to real market-replay processed_orders.csv (optional — KL/JS and comparisons skipped if absent)")
parser.add_argument("--gen",     required=True, help="Path to generated processed_orders.csv")
parser.add_argument("--tag",     required=True, help="Label for this run, e.g. DPM_SOLVER_PP_10")
parser.add_argument("--out-dir", required=True, help="Directory to write all outputs into")
args = parser.parse_args()

real_path = Path(args.real) if args.real else None
gen_path  = Path(args.gen)
out_dir   = Path(args.out_dir)
tag       = args.tag
out_dir.mkdir(parents=True, exist_ok=True)

print(f"\n{'='*60}")
print(f"[sweep_eval] tag={tag}")
print(f"  real: {real_path}")
print(f"  gen:  {gen_path}")
print(f"  out:  {out_dir}")
print(f"{'='*60}\n")

results = {"tag": tag, "real": str(real_path), "gen": str(gen_path)}

# ── 1. KL / JS Divergence ────────────────────────────────────────────────────
print("[1/4] KL / JS divergence...")
if real_path is None:
    print("  SKIP: no --real provided")
    results["kl_js"] = "skipped"
else:
    try:
        from evaluation.quantitative_eval.kl_divergence import compute_distribution_distances
        kl_results = compute_distribution_distances(str(real_path), str(gen_path))
        results["kl_js"] = kl_results
        kl_path = out_dir / f"{tag}_kl.json"
        kl_path.write_text(json.dumps(kl_results, indent=2))
        print(f"  -> saved {kl_path}")
        for feat, vals in kl_results.items():
            if isinstance(vals, dict):
                js = vals.get("js", vals.get("JS", "n/a"))
                kl = vals.get("kl", vals.get("KL", "n/a"))
                print(f"  {feat:12s}  JS={js:.4f}  KL={kl:.4f}" if isinstance(js, float) else f"  {feat}: {vals}")
    except Exception as e:
        print(f"  WARNING: KL/JS failed: {e}")
        results["kl_js"] = {"error": str(e)}

# ── 2. Stylized Facts (6-panel combined figure) ───────────────────────────────
print("\n[2/4] Stylized facts (6-panel figure)...")
try:
    from evaluation.stylized_custom.combined_stylized_facts_overlay import (
        load_processed_orders, make_figure
    )
    datasets = [load_processed_orders(str(gen_path), tag)]
    if real_path is not None:
        datasets.insert(0, load_processed_orders(str(real_path), "Real"))
    sf_path = out_dir / f"{tag}_stylized_facts.pdf"
    make_figure(datasets, sf_path, title=tag)
    print(f"  -> saved {sf_path}")
    results["stylized_facts_pdf"] = str(sf_path)
except Exception as e:
    print(f"  WARNING: stylized facts failed: {e}")
    results["stylized_facts_pdf"] = {"error": str(e)}

# ── 3. Order Type Distribution ────────────────────────────────────────────────
print("\n[3/4] Order type distribution...")
if real_path is None:
    print("  SKIP: no --real provided")
    results["order_type_pdf"] = "skipped"
else:
    try:
        from evaluation.visualizations.comparison_distribution_order_type import main as order_type_main
        # Script saves to os.path.dirname(TRADES_path) — use a temp dir then move
        tmp_dir = out_dir / "_tmp_order_type"
        tmp_dir.mkdir(exist_ok=True)
        tmp_csv = tmp_dir / "gen.csv"
        shutil.copy(gen_path, tmp_csv)
        order_type_main(str(real_path), str(tmp_csv), str(real_path), str(real_path))
        plt.close("all")
        src = tmp_dir / "order_type_join.pdf"
        dst = out_dir / f"{tag}_order_type.pdf"
        if src.exists():
            shutil.move(str(src), str(dst))
            print(f"  -> saved {dst}")
            results["order_type_pdf"] = str(dst)
        else:
            print("  WARNING: order_type_join.pdf not found in expected location")
        shutil.rmtree(tmp_dir, ignore_errors=True)
    except Exception as e:
        print(f"  WARNING: order type distribution failed: {e}")
        results["order_type_pdf"] = {"error": str(e)}

# ── 4. Spread Distribution ────────────────────────────────────────────────────
print("\n[4/4] Spread distribution...")
if real_path is None:
    print("  SKIP: no --real provided")
    results["spread_pdf"] = "skipped"
else:
    try:
        import pandas as pd
        from scipy.stats import gaussian_kde

        def _load_spread(csv_path):
            df = pd.read_csv(csv_path, index_col=0)
            if "SPREAD" in df.columns:
                s = df["SPREAD"].dropna()
            elif "ask_price_1" in df.columns and "bid_price_1" in df.columns:
                s = (df["ask_price_1"] - df["bid_price_1"]).dropna()
            else:
                return None
            # Filter out sentinel values (LOB not yet initialised)
            s = s[(s > 0) & (s < 1)]
            return s.values

        r_spread = _load_spread(str(real_path))
        g_spread = _load_spread(str(gen_path))

        if r_spread is not None and g_spread is not None and len(r_spread) > 1 and len(g_spread) > 1:
            fig, ax = plt.subplots(figsize=(6, 4))
            for vals, label, color in [(r_spread, "Real", "black"), (g_spread, tag, "orange")]:
                lo, hi = vals.min(), vals.max()
                if hi > lo:
                    xs = np.linspace(lo, hi, 300)
                    kde = gaussian_kde(vals)
                    ax.plot(xs, kde(xs), label=label, color=color)
            ax.set_xlabel("Bid-Ask Spread ($)")
            ax.set_ylabel("Density")
            ax.set_title(f"Spread Distribution — {tag}")
            ax.legend()
            dst = out_dir / f"{tag}_spread.pdf"
            fig.savefig(str(dst), bbox_inches="tight")
            plt.close(fig)
            print(f"  -> saved {dst}")
            results["spread_pdf"] = str(dst)
        else:
            print("  WARNING: could not extract spread values from CSVs")
            results["spread_pdf"] = "no_data"
    except Exception as e:
        print(f"  WARNING: spread distribution failed: {e}")
        results["spread_pdf"] = {"error": str(e)}

# ── Summary ───────────────────────────────────────────────────────────────────
summary_path = out_dir / f"{tag}_summary.json"
summary_path.write_text(json.dumps(results, indent=2))
print(f"\n[sweep_eval] Complete. Summary written to {summary_path}")

# Print one-line CSV row for the master summary (captured by sweep.sh)
kl_js = results.get("kl_js", {})
def _get(feat, metric):
    block = kl_js.get(feat, {})
    if isinstance(block, dict):
        return block.get(metric, block.get(metric.upper(), "NA"))
    return "NA"

print(
    f"SWEEP_ROW,{tag},"
    f"js_size={_get('SIZE','js')},"
    f"js_price={_get('PRICE','js')},"
    f"js_time={_get('TIME','js')},"
    f"kl_size={_get('SIZE','kl')},"
    f"kl_price={_get('PRICE','kl')},"
    f"kl_time={_get('TIME','kl')}"
)
