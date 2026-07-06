"""
Order-flow mix diagnostic: percentage of LIMIT_ORDER / ORDER_CANCELLED / ORDER_EXECUTED
in the generation phase (after the 09:45 replay warm-up), plus mid-price and top-of-book
volume health checks. Use this to judge whether a sampler/checkpoint config actually
produces a moving, non-degenerate market before trusting JS/KL scores.

Usage
-----
    python evaluation/quantitative_eval/flow_mix.py --gen path/to/generated_orders.csv
    python evaluation/quantitative_eval/flow_mix.py --gen gen.csv --real real.csv
"""

import argparse

import pandas as pd

from evaluation.quantitative_eval.kl_divergence import _load_and_filter


def report_flow_mix(path: str, label: str) -> None:
    df = _load_and_filter(path)
    print(f"\n=== {label} ({path}) ===")
    print(f"rows (post-09:45): {len(df)}")

    if "TYPE" in df.columns:
        mix = (df["TYPE"].value_counts(normalize=True) * 100).round(1)
        print("flow mix (%):")
        print(mix.to_string())

    if "MID_PRICE" in df.columns:
        mid = df["MID_PRICE"]
        mid = mid[(mid > 0) & (mid < 1_000_000)]
        print(f"unique mid-prices: {mid.nunique()}")
        if len(mid):
            print(f"mid range: {mid.min():.4f} - {mid.max():.4f}")

    for col in ("bid_size_1", "ask_size_1"):
        if col in df.columns:
            s = df[col].dropna()
            s = s[(s > 0) & (s < 1e12)]
            if len(s):
                print(f"{col}: min={s.min():.0f} max={s.max():.0f} mean={s.mean():.0f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gen", required=True, help="Generated processed_orders.csv")
    ap.add_argument("--real", default=None, help="Optional real market-replay processed_orders.csv")
    args = ap.parse_args()

    if args.real:
        report_flow_mix(args.real, "REAL")
    report_flow_mix(args.gen, "GENERATED")


if __name__ == "__main__":
    main()
