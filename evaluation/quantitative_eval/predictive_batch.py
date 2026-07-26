#!/usr/bin/env python3
"""Batch the TRADES predictive score (MAE) over many generated files against one real reference.

Mirrors evaluation/lob_bench/score_batch.py: give one --real and any number of
--gen LABEL=path, and it trains an LSTM per label (train on that generated series, test on the
real series), reports a mean+/-std MAE per label plus one shared real-on-real baseline, and
writes a results CSV. Lower MAE = the generated series is as useful as real data for predicting
the real market.

  python -m evaluation.quantitative_eval.predictive_batch \
      --real REAL.csv \
      --gen dn0.3=.../dn0.3/processed_orders.csv \
      --gen bt2.0=.../bt2.0r0.5/processed_orders.csv \
      --out-dir predictive_scores --seeds 3
"""
import argparse, csv, json, os, sys

from evaluation.quantitative_eval.predictive_score import (
    predictive_score, load_series, add_common_args, cfg_from_args, TARGET)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--real', required=True, help='real reference processed_orders.csv')
    ap.add_argument('--gen', action='append', required=True, metavar='LABEL=PATH',
                    help='a labelled generated csv; repeatable')
    ap.add_argument('--out-dir', default='predictive_scores')
    add_common_args(ap)
    a = ap.parse_args()

    cfg = cfg_from_args(a)
    os.makedirs(a.out_dir, exist_ok=True)
    rows, baseline = [], None
    for i, spec in enumerate(a.gen):
        if '=' not in spec:
            ap.error(f"--gen must be LABEL=PATH, got {spec!r}")
        label, path = spec.split('=', 1)
        # only the first run needs to compute the (label-independent) real-on-real baseline
        cfg_i = {**cfg, 'real_baseline': cfg['real_baseline'] and baseline is None}
        try:
            res = predictive_score(a.real, path, cfg_i, device=a.device, verbose=False)
        except Exception as e:
            print(f"[{label}] ERROR: {e}")
            rows.append({'label': label, 'gen_mae_mean': '', 'gen_mae_std': '', 'n_gen': '', 'error': str(e)})
            continue
        if 'real_mae_mean' in res:
            baseline = (res['real_mae_mean'], res['real_mae_std'])
        rows.append({'label': label, 'gen_mae_mean': res['gen_mae_mean'], 'gen_mae_std': res['gen_mae_std'],
                     'n_gen': res['n_gen'], 'error': ''})
        with open(os.path.join(a.out_dir, f"{label}.json"), 'w') as f:
            json.dump(res, f, indent=2)
        print(f"  scored {label}: MAE {res['gen_mae_mean']:.4f} +/- {res['gen_mae_std']:.4f}")

    # results table + csv
    out_csv = os.path.join(a.out_dir, 'predictive_scores.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['label', 'gen_mae_mean', 'gen_mae_std', 'n_gen', 'error'])
        w.writeheader(); w.writerows(rows)

    ok = [r for r in rows if r['error'] == '']
    ok.sort(key=lambda r: r['gen_mae_mean'])
    hdr = f"{'config':<24}{'MAE':>9}{'+/-':>9}{'n_gen':>9}"
    print("\n==== PREDICTIVE SCORE (MAE, lower = more realistic) ====")
    if baseline:
        print(f"  real-on-real baseline: {baseline[0]:.4f} +/- {baseline[1]:.4f} "
              f"(the reference a real training set achieves on the same test set)")
    print(hdr); print('-' * len(hdr))
    for r in ok:
        print(f"{r['label']:<24}{r['gen_mae_mean']:>9.4f}{r['gen_mae_std']:>9.4f}{r['n_gen']:>9}")
    for r in rows:
        if r['error']:
            print(f"{r['label']:<24}   ERROR: {r['error']}")
    print(f"\nwrote {out_csv} (+ per-config JSON in {a.out_dir}/)")


if __name__ == '__main__':
    main()
