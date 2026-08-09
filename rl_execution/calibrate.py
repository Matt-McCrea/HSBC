"""Estimate the Almgren-Chriss parameters from real episodes, so the risk penalty
and the AC baseline are calibrated to this simulator rather than guessed.

Three quantities, all recoverable from the per-step `mid` and `fills` now in the
trajectory log:

  sigma  std of mid changes over one decision interval (price units).
  eta    AC's temporary-impact coefficient, from regressing per-step slippage on
         trade rate. AC model the execution price as S~ = S - (eps + eta * v) for
         trade rate v, so a straight line through (rate, slippage) gives the
         half-spread term as intercept and eta as slope.
  drift  mean per-episode price move and its t-statistic, which TESTS AC's
         martingale assumption. TRADES has documented directional drift, and a
         seller is systematically flattered by an upward-drifting market, so this
         is a result to report rather than a knob to tune.

lambda is a preference, not a measurement, so it is pinned by requiring a target
kappa*T -- the dimensionless risk aversion that fixes the trajectory's shape, with
kappa*T -> 0 being TWAP. Working through AC's kappa^2 = lambda * sigma^2 / eta and
the per-share convention this code uses for reward (shortfall is divided by Q):

    lam = (kappa_T)^2 * eta * Q / (N^2 * tau)

Note sigma cancels: targeting kappa*T rather than fixing lambda is what makes the
calibration robust to a noisy sigma estimate, and kappa*T is the economically
meaningful quantity anyway (it is what determines how front-loaded the schedule is).
"""

import argparse
import math

import numpy as np

from rl_execution.execution_agent import N_DECISIONS
from rl_execution.logging_utils import read_episodes

DECISION_SECONDS = 30.0
TARGET_KAPPA_T = 2.0        # meaningfully front-loaded, still far from immediate liquidation
KAPPA_T_BOUNDS = (0.25, 10.0)
FALLBACK_LAM = 50.0         # the hand-picked value used before calibration existed
FALLBACK_KAPPAS = (1.0, 3.0)


def _step_observations(records):
    """(trade rate, slippage) per decision point, plus the mid series per episode."""
    rates, slippages, mid_series = [], [], []
    for rec in records:
        traj = rec.get("trajectory") or []
        side = str(rec.get("side", "SELL"))
        mids = [s.get("mid") for s in traj if s.get("mid") is not None]
        if len(mids) >= 2:
            mid_series.append([float(m) for m in mids])
        for step in traj:
            fills = step.get("fills") or []
            mid = step.get("mid")
            if not fills or mid is None:
                continue
            qty = sum(q for q, _ in fills)
            if qty <= 0:
                continue
            vwap = sum(q * p for q, p in fills) / qty
            # positive slippage = executed worse than the prevailing mid
            slip = (float(mid) - vwap) if side == "SELL" else (vwap - float(mid))
            rates.append(qty / DECISION_SECONDS)
            slippages.append(slip)
    return np.asarray(rates, float), np.asarray(slippages, float), mid_series


def estimate_sigma(mid_series):
    """Std of mid changes over one decision interval, pooled across episodes."""
    diffs = [np.diff(np.asarray(m, float)) for m in mid_series if len(m) >= 2]
    if not diffs:
        return None
    pooled = np.concatenate(diffs)
    return float(np.std(pooled, ddof=1)) if len(pooled) > 1 else None


def estimate_eta(rates, slippages):
    """Least-squares fit of slippage = eps + eta * rate. Returns (eta, eps, r2, n)."""
    if len(rates) < 5 or np.ptp(rates) <= 0:
        return None, None, None, len(rates)
    slope, intercept = np.polyfit(rates, slippages, 1)
    predicted = slope * rates + intercept
    ss_res = float(np.sum((slippages - predicted) ** 2))
    ss_tot = float(np.sum((slippages - slippages.mean()) ** 2))
    r2 = (1.0 - ss_res / ss_tot) if ss_tot > 0 else None
    return float(slope), float(intercept), r2, len(rates)


def estimate_drift(records):
    """Mean drift in bps and its t-statistic -- the martingale check."""
    drifts = [r["drift_bps"] for r in records if r.get("drift_bps") is not None]
    if len(drifts) < 3:
        return None, None, len(drifts)
    arr = np.asarray(drifts, float)
    stderr = arr.std(ddof=1) / math.sqrt(len(arr))
    t_stat = float(arr.mean() / stderr) if stderr > 0 else None
    return float(arr.mean()), t_stat, len(arr)


def lam_from_kappa(kappa_t, eta, mean_q, n_decisions=N_DECISIONS, tau=DECISION_SECONDS):
    """Convert a target kappa*T into this code's per-step inventory penalty."""
    return (kappa_t ** 2) * eta * mean_q / (n_decisions ** 2 * tau)


def calibrate(records, target_kappa_t=TARGET_KAPPA_T):
    """Returns a dict of estimates plus the derived settings, and never raises:
    a failed estimate degrades to documented fallbacks and says so in `warnings`."""
    rates, slippages, mid_series = _step_observations(records)
    sigma = estimate_sigma(mid_series)
    eta, eps, r2, n_obs = estimate_eta(rates, slippages)
    drift_mean, drift_t, n_drift = estimate_drift(records)
    qs = [r["Q"] for r in records if r.get("Q")]
    mean_q = float(np.mean(qs)) if qs else None

    out = {
        "sigma": sigma, "eta": eta, "eps": eps, "eta_r2": r2, "n_impact_obs": n_obs,
        "drift_bps_mean": drift_mean, "drift_t_stat": drift_t, "n_drift_obs": n_drift,
        "mean_Q": mean_q, "target_kappa_T": target_kappa_t,
        "warnings": [], "calibrated": False,
    }

    # eta <= 0 means the fit says trading bigger is CHEAPER -- noise, not a market.
    if eta is None or eta <= 0 or mean_q is None:
        out["warnings"].append(
            f"impact regression unusable (eta={eta}, n={n_obs}); falling back to "
            f"lam={FALLBACK_LAM} and fixed kappa {FALLBACK_KAPPAS}")
        out["lam"] = FALLBACK_LAM
        out["kappas"] = list(FALLBACK_KAPPAS)
        return out

    lam = lam_from_kappa(target_kappa_t, eta, mean_q)
    if not (KAPPA_T_BOUNDS[0] <= target_kappa_t <= KAPPA_T_BOUNDS[1]) or not math.isfinite(lam) or lam <= 0:
        out["warnings"].append(
            f"derived lam={lam} implausible; falling back to lam={FALLBACK_LAM} "
            f"and fixed kappa {FALLBACK_KAPPAS}")
        out["lam"] = FALLBACK_LAM
        out["kappas"] = list(FALLBACK_KAPPAS)
        return out

    out["lam"] = float(lam)
    out["kappas"] = [target_kappa_t]
    out["calibrated"] = True
    if drift_t is not None and abs(drift_t) > 2.0:
        out["warnings"].append(
            f"drift is significantly non-zero (mean={drift_mean:.2f}bps, t={drift_t:.2f}): "
            "the simulator violates Almgren-Chriss's martingale assumption, so a mismatch "
            "against the analytic frontier is expected and should be reported as such")
    return out


def format_report(c):
    lines = ["=" * 74, "CALIBRATION (Almgren-Chriss parameters, measured from real episodes)", "=" * 74]
    lines.append(f"  sigma (per {DECISION_SECONDS:.0f}s, price units) : {_fmt(c['sigma'])}")
    lines.append(f"  eta   (impact slope)                : {_fmt(c['eta'])}   "
                 f"R^2={_fmt(c['eta_r2'])} over {c['n_impact_obs']} decision points")
    lines.append(f"  eps   (half-spread intercept)       : {_fmt(c['eps'])}")
    lines.append(f"  mean parent order Q                 : {_fmt(c['mean_Q'])}")
    lines.append(f"  drift                               : {_fmt(c['drift_bps_mean'])} bps  "
                 f"t={_fmt(c['drift_t_stat'])} over {c['n_drift_obs']} episodes")
    lines.append("")
    lines.append(f"  target kappa*T                      : {c['target_kappa_T']}")
    lines.append(f"  -> inventory penalty lam            : {_fmt(c['lam'])}"
                 f"{'' if c['calibrated'] else '   (FALLBACK)'}")
    lines.append(f"  -> AC baseline kappa values         : {c['kappas']}")
    for w in c["warnings"]:
        lines.append(f"  WARNING: {w}")
    lines.append("=" * 74)
    return "\n".join(lines)


def _fmt(v):
    return "n/a" if v is None else (f"{v:.4g}" if isinstance(v, float) else str(v))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log", help="calibration .jsonl containing per-step trajectories")
    parser.add_argument("--target-kappa-t", type=float, default=TARGET_KAPPA_T)
    args = parser.parse_args()
    print(format_report(calibrate(read_episodes(args.log), args.target_kappa_t)))
