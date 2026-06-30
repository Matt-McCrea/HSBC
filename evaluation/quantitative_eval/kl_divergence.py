"""
Distribution distance metrics between real and generated LOB order streams.

Computes Jensen-Shannon divergence (bounded, symmetric) and KL divergence
per feature for: SIZE, PRICE, inter-arrival TIME, DEPTH, and order TYPE.

Usage
-----
From Python:
    from evaluation.quantitative_eval.kl_divergence import compute_distribution_distances
    results = compute_distribution_distances("path/to/real.csv", "path/to/generated.csv")

The generated CSV is the processed_orders.csv written by world_agent_sim.py after a
simulation run; the real CSV is the equivalent market-replay output.
"""

import numpy as np
import pandas as pd
from scipy.stats import entropy

# Columns expected in processed_orders.csv
_REAL_COLS = {"SIZE", "PRICE", "TIME", "DEPTH", "TYPE"}
_N_BINS = 50
_EPS = 1e-10  # smoothing to avoid log(0)


def _load_and_filter(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Drop warm-up period (first 15 min), matching WorldAgent's convention
    if "Time" in df.columns:
        df["_time_str"] = df["Time"].astype(str).str.slice(11, 19)
        df = df[df["_time_str"] >= "09:45:00"].drop(columns=["_time_str"])
    # Remove extreme price levels used as sentinels
    for col in ("ask_price_1", "bid_price_1"):
        if col in df.columns:
            df = df[(df[col] < 9_999_999) & (df[col] > -9_999_999)]
    return df.reset_index(drop=True)


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence (base-2 bits, in [0, 1])."""
    p = p + _EPS
    q = q + _EPS
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * entropy(p, m, base=2) + 0.5 * entropy(q, m, base=2))


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q) in nats with epsilon smoothing."""
    p = p + _EPS
    q = q + _EPS
    p /= p.sum()
    q /= q.sum()
    return float(entropy(p, q))


def _continuous_divergences(real_vals: np.ndarray, gen_vals: np.ndarray) -> dict:
    lo = min(real_vals.min(), gen_vals.min())
    hi = max(real_vals.max(), gen_vals.max())
    if hi == lo:
        return {"js": 0.0, "kl": 0.0}
    bins = np.linspace(lo, hi, _N_BINS + 1)
    p, _ = np.histogram(real_vals, bins=bins, density=False)
    q, _ = np.histogram(gen_vals,  bins=bins, density=False)
    p = p.astype(float)
    q = q.astype(float)
    return {"js": _js_divergence(p.copy(), q.copy()), "kl": _kl_divergence(p, q)}


def _discrete_divergences(real_vals: pd.Series, gen_vals: pd.Series) -> dict:
    categories = sorted(set(real_vals.unique()) | set(gen_vals.unique()))
    p = np.array([real_vals.value_counts().get(c, 0) for c in categories], dtype=float)
    q = np.array([gen_vals.value_counts().get(c, 0)  for c in categories], dtype=float)
    return {"js": _js_divergence(p.copy(), q.copy()), "kl": _kl_divergence(p, q)}


def compute_distribution_distances(real_path: str, generated_path: str) -> dict:
    """Compare per-feature distributions between real and generated order streams.

    Parameters
    ----------
    real_path : str
        Path to the market-replay processed_orders.csv (real data).
    generated_path : str
        Path to the TRADES simulation processed_orders.csv (generated data).

    Returns
    -------
    dict
        ``{feature: {"js": float, "kl": float}}`` for each feature, plus
        ``"summary": {"mean_js": float, "mean_kl": float}``.
    """
    df_r = _load_and_filter(real_path)
    df_g = _load_and_filter(generated_path)

    results = {}
    features_computed = []

    continuous_map = {
        "SIZE":  "SIZE",
        "PRICE": "PRICE",
        "TIME":  "TIME",   # inter-arrival time
        "DEPTH": "DEPTH",
    }
    for label, col in continuous_map.items():
        if col in df_r.columns and col in df_g.columns:
            r_vals = df_r[col].dropna().values
            g_vals = df_g[col].dropna().values
            if len(r_vals) > 1 and len(g_vals) > 1:
                results[label] = _continuous_divergences(r_vals, g_vals)
                features_computed.append(label)

    # Discrete: order type
    type_col = next((c for c in ("TYPE", "type", "event_type") if c in df_r.columns), None)
    if type_col and type_col in df_g.columns:
        results["TYPE"] = _discrete_divergences(df_r[type_col], df_g[type_col])
        features_computed.append("TYPE")

    if not results:
        print("[kl_divergence] No matching columns found — check CSV column names.")
        return {}

    # Summary
    js_vals = [v["js"] for v in results.values()]
    kl_vals = [v["kl"] for v in results.values()]
    results["summary"] = {"mean_js": float(np.mean(js_vals)), "mean_kl": float(np.mean(kl_vals))}

    _print_table(results, real_path, generated_path)
    return results


def _print_table(results: dict, real_path: str, gen_path: str) -> None:
    print("\n" + "=" * 62)
    print("  Distribution Distance: Real vs Generated")
    print(f"  Real:      {real_path}")
    print(f"  Generated: {gen_path}")
    print("=" * 62)
    print(f"  {'Feature':<10}  {'JS div (bits)':>14}  {'KL div (nats)':>14}")
    print("-" * 62)
    for feat, vals in results.items():
        if feat == "summary":
            continue
        print(f"  {feat:<10}  {vals['js']:>14.4f}  {vals['kl']:>14.4f}")
    print("-" * 62)
    s = results["summary"]
    print(f"  {'Mean':<10}  {s['mean_js']:>14.4f}  {s['mean_kl']:>14.4f}")
    print("=" * 62)
    print("  JS divergence: 0 = identical, 1 = maximally different (base-2 bits)")
    print("  KL divergence: 0 = identical, higher = more different (nats)")
    print()
