"""Market-impact measurement for the execution agent, computed from logs already
collected -- no new simulation required.

Per-step trajectories record the prevailing mid and the fills at every decision
point, which is enough to recover both halves of the standard Almgren-Chriss
decomposition:

  TEMPORARY impact  h(v) = eps + eta*v
      Slippage of our own fills against the mid available when the order was sent.
      This is the cost we pay to cross the spread and consume depth; it does not
      persist.

  PERMANENT impact  g(v) = gamma*v
      The part of the price move our trading causes that stays. Estimated by
      regressing the SUBSEQUENT mid change on our signed traded volume. Because mids
      are recorded at all ten decision points, this is available at several horizons,
      which traces an impact DECAY curve rather than a single number.

Identification caveat, which belongs in the write-up rather than in a footnote:
regressing mid changes on own volume attributes to us any price move correlated
with our trading. In the generative arm the world agent also reacts to the book we
perturb, so "permanent impact" here means the total price response associated with
our trade, not a causally isolated effect. A clean causal estimate needs paired
counterfactual episodes (same seed, agent disabled); in replay mode that
counterfactual is exact, because replayed orders do not react to us at all.
"""

import argparse
import math

import numpy as np

from rl_execution.calibrate import DECISION_SECONDS
from rl_execution.logging_utils import read_episodes


def _signed_trades(rec):
    """Per step: (index, signed volume, mid). Sign is +1 when our trading pushes
    price up (a buy) and -1 for a sell, so a positive gamma always means 'our trade
    moved the price against us'."""
    traj = rec.get("trajectory") or []
    sign = 1.0 if str(rec.get("side", "SELL")) == "BUY" else -1.0
    out = []
    for k, step in enumerate(traj):
        mid = step.get("mid")
        if mid is None:
            continue
        qty = sum(q for q, _ in (step.get("fills") or []))
        out.append((k, sign * float(qty), float(mid)))
    return out


def temporary_impact(records):
    """Regress own-fill slippage on trade rate: slippage = eps + eta * (n/tau)."""
    rates, slips = [], []
    for rec in records:
        side = str(rec.get("side", "SELL"))
        for step in rec.get("trajectory") or []:
            fills, mid = step.get("fills") or [], step.get("mid")
            if not fills or mid is None:
                continue
            qty = sum(q for q, _ in fills)
            if qty <= 0:
                continue
            vwap = sum(q * p for q, p in fills) / qty
            slips.append((float(mid) - vwap) if side == "SELL" else (vwap - float(mid)))
            rates.append(qty / DECISION_SECONDS)
    return _fit(np.asarray(rates), np.asarray(slips))


def permanent_impact(records, horizons=(1, 2, 3, 5)):
    """Regress the mid change over h decision points on our signed volume.

    Returns {h: fit}. The slope is impact per share at that horizon; comparing
    across h shows how much of the initial move persists.
    """
    out = {}
    for h in horizons:
        vols, moves = [], []
        for rec in records:
            steps = _signed_trades(rec)
            by_index = {k: (v, m) for k, v, m in steps}
            for k, signed_vol, mid in steps:
                if signed_vol == 0 or (k + h) not in by_index:
                    continue
                vols.append(signed_vol)
                moves.append(by_index[k + h][1] - mid)
        out[h] = _fit(np.asarray(vols), np.asarray(moves))
    return out


def participation_impact(records):
    """Impact against the square root of size, the empirical 'square-root law' that
    the market-impact literature reports in preference to a linear form."""
    roots, slips = [], []
    for rec in records:
        side = str(rec.get("side", "SELL"))
        for step in rec.get("trajectory") or []:
            fills, mid = step.get("fills") or [], step.get("mid")
            if not fills or mid is None:
                continue
            qty = sum(q for q, _ in fills)
            if qty <= 0:
                continue
            vwap = sum(q * p for q, p in fills) / qty
            slips.append((float(mid) - vwap) if side == "SELL" else (vwap - float(mid)))
            roots.append(math.sqrt(qty))
    return _fit(np.asarray(roots), np.asarray(slips))


def _fit(x, y):
    """Least squares with a t-statistic on the slope; None when unidentifiable."""
    n = len(x)
    if n < 5 or np.ptp(x) <= 0:
        return {"n": n, "slope": None, "intercept": None, "r2": None, "t": None}
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = (1.0 - ss_res / ss_tot) if ss_tot > 0 else None
    # standard error of the slope
    sxx = float(np.sum((x - x.mean()) ** 2))
    t = None
    if n > 2 and sxx > 0 and ss_res >= 0:
        se = math.sqrt(ss_res / (n - 2) / sxx) if sxx > 0 else None
        t = float(slope / se) if se and se > 0 else None
    return {"n": n, "slope": float(slope), "intercept": float(intercept), "r2": r2, "t": t}


def analyse(records):
    return {
        "temporary": temporary_impact(records),
        "permanent": permanent_impact(records),
        "sqrt_law": participation_impact(records),
        "n_episodes": len([r for r in records if r.get("trajectory")]),
    }


def format_report(a):
    L = ["=" * 78, "MARKET IMPACT OF THE EXECUTION AGENT'S OWN ORDERS", "=" * 78,
         f"  episodes with trajectories: {a['n_episodes']}", ""]

    t = a["temporary"]
    L.append("TEMPORARY IMPACT   slippage vs prevailing mid = eps + eta * (shares/sec)")
    L.append(f"  eta (slope)      : {_f(t['slope'])}   t={_f(t['t'])}  R^2={_f(t['r2'])}  n={t['n']}")
    L.append(f"  eps (intercept)  : {_f(t['intercept'])}")
    if t["intercept"] is not None and t["intercept"] < 0:
        L.append("  NOTE: a negative intercept means the average fill BEAT the mid at zero size,")
        L.append("        which is impossible for pure liquidity taking. It is the signature of an")
        L.append("        action space that also PROVIDES liquidity: a passive sell rests at the ask")
        L.append("        and fills above the mid. Split passive and aggressive fills before reading")
        L.append("        eps as a half-spread.")

    L.append("")
    L.append("PERMANENT IMPACT   mid move over h decisions vs signed volume (per share)")
    L.append(f"  {'horizon':>9} {'seconds':>8} {'gamma':>12} {'t':>8} {'R^2':>8} {'n':>7}")
    for h, f in sorted(a["permanent"].items()):
        L.append(f"  {h:>9} {h * DECISION_SECONDS:>8.0f} {_f(f['slope']):>12} "
                 f"{_f(f['t']):>8} {_f(f['r2']):>8} {f['n']:>7}")
    # Retention is a ratio of two regression slopes, so it is only meaningful when both
    # are actually distinguishable from zero and share a sign. On a log where every
    # horizon is insignificant and the signs alternate, the ratio is noise over noise:
    # gamma(1)=-0.019 and gamma(5)=+0.020 print as "-105% retained", which reads like a
    # measured full reversal and is really just two zeros with different rounding.
    fits = [(h, f) for h, f in sorted(a["permanent"].items()) if f["slope"] is not None]
    if len(fits) >= 2:
        (h0, f0), (h1, f1) = fits[0], fits[-1]
        significant = all(abs(f.get("t") or 0.0) >= 2.0 for f in (f0, f1))
        same_sign = (f0["slope"] > 0) == (f1["slope"] > 0)
        if significant and same_sign and f0["slope"]:
            decay = f1["slope"] / f0["slope"]
            L.append(f"  retained at h={h1} vs h={h0}: {decay:.0%}"
                     "   (<100% = the move partly reverts, i.e. some impact was transient)")
        else:
            L.append("  retention not reported: permanent impact is not distinguishable from zero "
                     "at both\n  horizons, so their ratio would describe noise rather than decay.")

    s = a["sqrt_law"]
    L.append("")
    L.append("SQUARE-ROOT LAW    slippage vs sqrt(shares)")
    L.append(f"  slope            : {_f(s['slope'])}   t={_f(s['t'])}  R^2={_f(s['r2'])}")
    if s["r2"] is not None and t["r2"] is not None:
        # Both forms are near-collinear over the size range a single parent order spans,
        # so their R^2 land within a hair of each other and whichever is nominally higher
        # flips between logs -- linear on the 114-episode training log (0.1973 vs 0.1943),
        # square-root on the 54-episode evaluation log (0.1175 vs 0.1128). Declaring a
        # winner on a margin of ~0.005 would put an unsupported claim in the write-up, so
        # only call it when the gap is big enough to survive that.
        gap = abs(s["r2"] - t["r2"])
        if gap < 0.02:
            L.append(f"  better fit       : INDISTINGUISHABLE (R^2 {_f(s['r2'])} vs {_f(t['r2'])}, "
                     f"gap {gap:.4f})")
            L.append("                     Do not report either functional form as supported: over "
                     "one parent\n                     order's size range the two are near-collinear, "
                     "and which one leads\n                     flips between logs.")
        else:
            better = "square-root" if s["r2"] > t["r2"] else "linear"
            L.append(f"  better fit       : {better} (compare R^2 {_f(s['r2'])} vs {_f(t['r2'])}, "
                     f"gap {gap:.4f})")

    L.append("")
    L.append("CAVEAT: permanent impact here is the price response ASSOCIATED with our trade,")
    L.append("not a causally isolated effect -- the world agent also reacts to the book we")
    L.append("perturb. A clean estimate needs paired counterfactual episodes (same seed, agent")
    L.append("disabled), which in replay mode is exact.")
    L.append("=" * 78)
    return "\n".join(L)


def _f(v):
    return "n/a" if v is None else f"{v:.4g}"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("log", help="any .jsonl run log containing per-step trajectories")
    args = p.parse_args()
    print(format_report(analyse(read_episodes(args.log))))
