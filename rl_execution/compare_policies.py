"""Paired policy comparison -- the headline number for the RL chapter.

Every policy walks the SAME held-out seed list, so each seed gives one episode per
policy on the same day, t0, side and Q. That pairing is not a nicety here, it is the
whole comparison: shortfall correlates -0.837 with market drift over the episode, so
the between-seed spread is dominated by where the market happened to go, not by what
the policy did. Comparing unpaired means throws that structure away and buries a real
effect under a standard error four to ten times too large.

The paired difference d_i = shortfall(policy_i) - shortfall(baseline_i) cancels the
drift common to both arms of a seed, which is exactly the "quote the POLICY-VS-BENCHMARK
difference rather than the absolute level" that analyze_logs recommends but does not
itself compute.

    python -m rl_execution.compare_policies logs/eval_frontier_lam.jsonl
    python -m rl_execution.compare_policies logs/eval_frontier_*.jsonl --baseline twap

Pure pandas/numpy over a finished log: no kernel, no model, runs in milliseconds.
"""

import argparse
import glob
import itertools
import json
import math
import os

import numpy as np
import pandas as pd

# A seed is the market situation an episode was handed. Two episodes are the same seed
# when all four match -- Q and side included, since evaluate.generate_held_out_seeds
# randomises them per seed and a BUY is not comparable with a SELL.
SEED_KEY = ["seed_day", "t0", "side", "Q"]


def load(paths):
    rows = []
    for path in paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("error"):
                    continue
                rows.append(rec)
    if not rows:
        raise SystemExit(f"no usable episodes found in {paths}")
    df = pd.DataFrame(rows)
    if "shortfall_bps" not in df.columns or df["shortfall_bps"].isna().all():
        df["shortfall_bps"] = df["shortfall"] / df["p_arrival"] * 10_000.0
    return df


def _t_sf(t, dof):
    """Two-sided p-value for Student's t. Uses the exact regularised incomplete beta via
    the continued fraction in math.lgamma terms would be overkill; scipy is not a
    dependency here, so fall back to the normal approximation above 30 dof and a
    conservative note below it."""
    try:
        from scipy import stats
        return float(2 * stats.t.sf(abs(t), dof))
    except ImportError:
        return float(math.erfc(abs(t) / math.sqrt(2.0)))


def paired(df, baseline="twap", metric="shortfall_bps"):
    """One row per (policy vs baseline) with the paired statistics."""
    wide = df.pivot_table(index=SEED_KEY, columns="policy_name", values=metric, aggfunc="mean")
    if baseline not in wide.columns:
        raise SystemExit(f"baseline policy {baseline!r} not in log; found {list(wide.columns)}")

    out = []
    for policy in wide.columns:
        if policy == baseline:
            continue
        both = wide[[baseline, policy]].dropna()
        n = len(both)
        if n < 2:
            continue
        d = (both[policy] - both[baseline]).to_numpy(dtype=float)
        mean_d = float(d.mean())
        se_d = float(d.std(ddof=1) / math.sqrt(n))
        t = mean_d / se_d if se_d > 0 else float("nan")
        # What the pairing actually bought: the naive stderr of the difference of means,
        # which is what an unpaired reading of the evaluation summary would have used.
        se_unpaired = math.sqrt(
            both[policy].std(ddof=1) ** 2 / n + both[baseline].std(ddof=1) ** 2 / n)
        out.append({
            "policy": policy, "n": n,
            "baseline_mean": float(both[baseline].mean()),
            "policy_mean": float(both[policy].mean()),
            "mean_diff": mean_d, "stderr": se_d,
            "t": t, "p": _t_sf(t, n - 1) if se_d > 0 else float("nan"),
            "ci_lo": mean_d - 1.96 * se_d, "ci_hi": mean_d + 1.96 * se_d,
            "wins": int((d < 0).sum()), "se_unpaired": se_unpaired,
            "variance_reduction": se_unpaired / se_d if se_d > 0 else float("nan"),
        })
    return pd.DataFrame(out).sort_values("mean_diff") if out else pd.DataFrame()


def frontier(df):
    """Mean cost against mean inventory risk, per policy -- the Almgren-Chriss frontier.

    Why this is not optional. Mean shortfall IS the lambda=0 objective, and TWAP is the
    lambda=0 optimum of the AC family, so an AC schedule at kappa*T=2 must score worse on
    mean cost than TWAP by construction. Reporting that alone reads as evidence against
    AC when it is really a statement about which objective was measured. AC buys reduced
    exposure to price moves, which shows up as sum x_t^2 -- how slowly inventory was drawn
    down -- and is invisible on the cost axis.

    A policy is only genuinely worse if it loses on BOTH axes.
    """
    from rl_execution.qlearning import inventory_risk
    rows = []
    for name, grp in df.groupby("policy_name"):
        risks = [inventory_risk(t) for t in grp.get("trajectory", []) if t]
        if not risks:
            continue
        cost = grp["shortfall_bps"].dropna()
        rows.append({"policy": name, "n": len(risks),
                     "mean_cost_bps": float(cost.mean()),
                     "mean_risk": float(np.mean(risks)),
                     "risk_sd": float(np.std(risks, ddof=1)) if len(risks) > 1 else 0.0})
    return pd.DataFrame(rows).sort_values("mean_risk") if rows else pd.DataFrame()


def report(df, baseline="twap", metric="shortfall_bps"):
    lines = []
    A = lines.append
    A("=" * 78)
    A(f"PAIRED POLICY COMPARISON   metric={metric}   baseline={baseline}")
    A("negative mean_diff = the policy BEAT the baseline on the same seeds")
    A("=" * 78)

    per = df.groupby("policy_name")[metric].agg(["count", "mean", "std"])
    A("\nUNPAIRED (what the evaluation summary prints -- drift still in the noise)")
    for name, r in per.iterrows():
        se = r["std"] / math.sqrt(r["count"]) if r["count"] > 1 else 0.0
        A(f"  {name:<14} n={int(r['count']):>3}  mean={r['mean']:>9.3f} bps  stderr={se:>8.3f}")

    res = paired(df, baseline, metric)
    if res.empty:
        A("\nno policy shares enough seeds with the baseline to pair")
        return "\n".join(lines), res

    A(f"\nPAIRED vs {baseline} (drift cancels within each seed)")
    for _, r in res.iterrows():
        star = "***" if r["p"] < 0.001 else "**" if r["p"] < 0.01 else "*" if r["p"] < 0.05 else ""
        A(f"  {r['policy']:<14} n={int(r['n']):>3}  diff={r['mean_diff']:>+9.3f} bps  "
          f"stderr={r['stderr']:>7.3f}  t={r['t']:>+6.2f}  p={r['p']:.4f}{star}")
        A(f"  {'':<14} 95% CI [{r['ci_lo']:>+8.3f}, {r['ci_hi']:>+8.3f}]  "
          f"beat baseline on {int(r['wins'])}/{int(r['n'])} seeds")
        A(f"  {'':<14} pairing shrank the stderr {r['variance_reduction']:.1f}x "
          f"(unpaired would be {r['se_unpaired']:.3f})")

    fr = frontier(df)
    if not fr.empty and len(fr) > 1:
        A("\nCOST / RISK FRONTIER   mean shortfall vs mean inventory risk (sum x_t^2)")
        A(f"  {'policy':<14} {'n':>5} {'mean cost':>12} {'mean risk':>11} {'risk sd':>9}")
        for _, r in fr.iterrows():
            A(f"  {r['policy']:<14} {int(r['n']):>5} {r['mean_cost_bps']:>9.3f} bps "
              f"{r['mean_risk']:>11.3f} {r['risk_sd']:>9.3f}")
        A("  Lower risk = inventory drawn down faster = less exposure to price moves.")
        A("  Mean shortfall alone is the lambda=0 objective, which TWAP optimises by")
        A("  construction, so a front-loaded schedule MUST look worse on that axis. Judge a")
        A("  policy worse only if it loses on both.")

    A("\nREADING THIS")
    sig = res[res["p"] < 0.05]
    worse, better = sig[sig["mean_diff"] > 0], sig[sig["mean_diff"] < 0]
    n_min = int(res["n"].min())

    if len(better):
        for _, r in better.iterrows():
            A(f"  {r['policy']} BEATS {baseline} by {-r['mean_diff']:.2f} bps (p={r['p']:.4f}).")
    if len(worse) == len(res):
        A(f"  Every policy tested is significantly WORSE than {baseline}. That is a legitimate")
        A("  result rather than a bug: TWAP is the risk-neutral Almgren-Chriss optimum, so")
        A("  beating it needs either exploitable structure or risk aversion, and a tabular")
        A("  agent trained for ~10^2 episodes has too little data to find either.")
    elif len(worse):
        A(f"  Significantly worse than {baseline}: {', '.join(worse['policy'])}.")
        A(f"  Not separated at 5%: {', '.join(res[res['p'] >= 0.05]['policy']) or 'none'}.")
    elif not len(better):
        A(f"  No policy separates from {baseline} at the 5% level. State the confidence")
        A("  interval rather than 'no difference': the interval below is wide enough that a")
        A("  modest effect could not have been detected either way.")

    if n_min < 8:
        A(f"  CAUTION: the smallest paired sample here is n={n_min}. Treat every p-value in")
        A("  this block as indicative only -- at that size the test has little power and the")
        A("  estimate moves a lot with any single seed.")
    A("=" * 78)
    return "\n".join(lines), res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("logs", nargs="+", help="one or more eval jsonl logs (globs allowed)")
    p.add_argument("--baseline", default="twap")
    p.add_argument("--metric", default="shortfall_bps")
    p.add_argument("--by-side", action="store_true",
                   help="split the comparison by BUY/SELL. Worth doing when drift is "
                        "directional: an upward-drifting market flatters sellers, so a "
                        "policy can look good on one side purely from that")
    p.add_argument("--csv", default=None, help="write the paired table here")
    args = p.parse_args()

    paths = list(itertools.chain.from_iterable(glob.glob(g) or [g] for g in args.logs))
    missing = [q for q in paths if not os.path.exists(q)]
    if missing:
        raise SystemExit(f"no such log(s): {missing}")
    df = load(paths)

    text, res = report(df, args.baseline, args.metric)
    print(text)

    if args.by_side:
        for side, grp in df.groupby("side"):
            print(f"\n{'=' * 78}\nSIDE = {side}   ({grp['policy_name'].nunique()} policies, "
                  f"{len(grp)} episodes)\n{'=' * 78}")
            print(report(grp, args.baseline, args.metric)[0])

    if args.csv and not res.empty:
        res.to_csv(args.csv, index=False)
        print(f"\nwritten to {args.csv}")


if __name__ == "__main__":
    raise SystemExit(main())
