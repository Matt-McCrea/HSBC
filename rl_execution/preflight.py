"""Invariant checks on real episodes, run before a long session commits to a
configuration.

These cannot be unit tests: they check properties that only hold once the ABIDES
kernel, the exchange's matching engine and the execution agent interact. Per-step
reward attribution, order cancellation and the forced terminal market order have all
been unit-tested but never run against the real kernel, and each has a failure mode
that looks like a plausible result rather than a crash -- unsold inventory, for
instance, SHRINKS reported shortfall, so a broken terminal sweep reads as excellent
performance. Catching that before a 24-hour training run is the whole point.
"""

from rl_execution.execution_agent import N_DECISIONS

# Fills may exceed Q slightly through partial-fill rounding; well beyond that means
# stale orders filled on top of the terminal sweep, i.e. the parent order was breached.
OVEREXECUTION_TOLERANCE = 1.02
# Reward attribution is exact arithmetic, so this only absorbs float noise.
REWARD_SUM_TOLERANCE = 1e-6
# cond_z far outside this means the conditioning is out of distribution and the
# generated market is not trustworthy (see the drift diagnostics in analysis/).
COND_Z_ABS_LIMIT = 12.0


def check_episode(rec, reward_mode="per-step", reward_benchmark="arrival"):
    """Returns a list of human-readable failures for one logged episode."""
    failures = []
    traj = rec.get("trajectory") or []

    if len(traj) != N_DECISIONS:
        failures.append(f"trajectory has {len(traj)} steps, expected {N_DECISIONS}")

    rem = rec.get("rem_quantity")
    if rem is None:
        failures.append("rem_quantity not logged")
    elif rem > 0:
        failures.append(
            f"terminal liquidation incomplete: {rem} shares unsold. Almgren-Chriss "
            f"requires x_N=0, and unsold inventory shrinks reported shortfall, so this "
            f"would read as good performance")

    fills = rec.get("fills") or []
    quantity = rec.get("Q")
    if quantity:
        traded = sum(q for q, _ in fills)
        if traded > quantity * OVEREXECUTION_TOLERANCE:
            failures.append(
                f"over-execution: filled {traded:.0f} against a parent order of {quantity} "
                f"-- stale child orders are filling after the terminal sweep")

    # Per-step rewards must re-attribute the SAME total, or the agent is being scored
    # on something other than what gets reported.
    if reward_mode == "per-step" and reward_benchmark == "arrival" and traj:
        shortfall = rec.get("shortfall")
        if shortfall is not None:
            total = sum(float(s.get("r") or 0.0) for s in traj)
            if abs(total + float(shortfall)) > max(REWARD_SUM_TOLERANCE,
                                                    abs(float(shortfall)) * 1e-6):
                failures.append(
                    f"per-step rewards sum to {total:.6f} but -shortfall is "
                    f"{-float(shortfall):.6f} -- attribution is not preserving the total")

    cond_z = rec.get("cond_z") or {}
    for channel, stats in cond_z.items():
        if isinstance(stats, (list, tuple)) and len(stats) == 4:
            lo, hi = stats[0], stats[1]
            if abs(lo) > COND_Z_ABS_LIMIT or abs(hi) > COND_Z_ABS_LIMIT:
                failures.append(
                    f"cond_z[{channel}] out of distribution (min={lo:.1f} max={hi:.1f}); "
                    f"the generated market may not be trustworthy")

    if rec.get("error"):
        failures.append(f"episode recorded an error: {rec['error']}")
    return failures


def run_checks(records, reward_mode="per-step", reward_benchmark="arrival"):
    """Returns (ok, report_lines). ok is False if ANY episode failed any invariant."""
    lines, all_failures = [], []
    if not records:
        return False, ["PREFLIGHT FAILED: no episodes were produced"]

    for i, rec in enumerate(records, start=1):
        failures = check_episode(rec, reward_mode, reward_benchmark)
        all_failures.extend(failures)
        status = "OK" if not failures else "FAIL"
        lines.append(f"  episode {i}: {status}"
                     f"  rem_qty={rec.get('rem_quantity')}"
                     f"  steps={len(rec.get('trajectory') or [])}"
                     f"  shortfall={_fmt(rec.get('shortfall'))}")
        for f in failures:
            lines.append(f"      - {f}")

    ok = not all_failures
    header = (f"PREFLIGHT {'PASSED' if ok else 'FAILED'} "
              f"({len(records)} episodes, {len(all_failures)} invariant violations)")
    return ok, [header] + lines


def _fmt(v):
    return "n/a" if v is None else f"{v:.2f}"


if __name__ == "__main__":
    import argparse
    from rl_execution.logging_utils import read_episodes

    parser = argparse.ArgumentParser()
    parser.add_argument("log")
    parser.add_argument("--reward-mode", default="per-step")
    parser.add_argument("--reward-benchmark", default="arrival")
    args = parser.parse_args()
    ok, report = run_checks(read_episodes(args.log), args.reward_mode, args.reward_benchmark)
    print("\n".join(report))
    raise SystemExit(0 if ok else 1)
