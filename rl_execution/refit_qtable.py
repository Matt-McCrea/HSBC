"""Re-fit a Q-table OFFLINE from trajectories already logged by a training run.

Simulating one episode costs ~10 minutes of GPU. Fitting values to episodes
already simulated costs milliseconds. Once per-step trajectories are logged
(see logging_utils.trajectory_step), those two are decoupled: any number of
learning-rule variants can be tried against the same expensive data.

    python -m rl_execution.refit_qtable logs/train_v2.jsonl --alpha-mode visit-count
    python -m rl_execution.refit_qtable logs/train_v2.jsonl --drift-adjust --out checkpoints/refit.npz

Two levers, both aimed at the same diagnosis -- that the greedy policy churns
between checkpoints because per-episode reward noise is larger than the gap
between actions it is meant to resolve:

  --alpha-mode visit-count   alpha = 1/N(s,a) instead of a fixed rate. With
      gamma=1 and a single terminal reward, Q IS the mean return, so a running
      average is the principled estimator; a fixed alpha=0.3 keeps only an
      effective ~3-episode window and tracks noise.

  --drift-adjust   subtract each episode's market drift from its reward. Drift
      is common-mode -- it moves every action in an episode identically and so
      carries no information about which action was better -- but it dominates
      the variance. Removing it is a control variate, not a fudge: it changes
      the noise, not the ranking.
"""

import argparse

import numpy as np

from rl_execution.logging_utils import read_episodes
from rl_execution.execution_agent import signed_cost
from rl_execution.qlearning import (N_ACTIONS, N_STATES, QLearningPolicy, inventory_penalty,
                                     inventory_risk, state_to_index)


def _obs_from_step(step):
    return {
        "time_remaining_frac": step["t_rem"],
        "inventory_remaining_frac": step["inv_rem"],
        "spread_bucket": step["spread"],
        "vol_bucket": step["vol"],
        "ofi_bucket": step["ofi"],
    }


def _recomputed_reward(step, rec, benchmark):
    """Rebuild a step's reward under a different benchmark price.

    Returns None when the log lacks the per-step mid/fills needed (runs predating
    that logging), so the caller falls back to the reward as recorded.

    The arrival-vs-prevailing choice is a real change of objective, not a
    normalisation: arrival keeps true implementation shortfall and makes the agent
    bear timing risk; prevailing scores execution against the price available at the
    time, removing timing risk (and the drift variance with it) so that urgency comes
    only from the inventory penalty. Being able to switch offline means a long run
    does not lock that decision in.
    """
    fills = step.get("fills")
    if not fills:
        return 0.0 if step.get("mid") is not None else None
    if benchmark == "prevailing":
        price = step.get("mid")
    else:
        price = rec.get("p_arrival")
    if price is None:
        return None
    quantity = rec.get("Q")
    if not quantity:
        return None
    direction = str(rec.get("side", "SELL"))
    cost = sum(signed_cost(q, p, float(price), direction) for q, p in fills)
    return -cost / float(quantity)


def refit(records, alpha_mode="visit-count", alpha=0.3, gamma=1.0, drift_adjust=False,
          inventory_lambda=0.0, reward_benchmark=None):
    q = np.zeros((N_STATES, N_ACTIONS), dtype=np.float64)
    visits = np.zeros((N_STATES, N_ACTIONS), dtype=np.int64)
    n_eps = n_steps = 0
    skipped = 0

    for rec in records:
        traj = rec.get("trajectory")
        if not traj:
            skipped += 1
            continue
        drift = rec.get("drift_bps") if drift_adjust else None
        # Reward is logged in raw price units; drift is in bps of arrival. Convert the
        # drift back to the reward's units before subtracting, or the correction is
        # silently ~4 orders of magnitude too small.
        drift_raw = None
        if drift is not None and rec.get("p_arrival"):
            drift_raw = float(drift) / 10_000.0 * float(rec["p_arrival"])

        n_eps += 1
        for i, step in enumerate(traj):
            s = state_to_index(_obs_from_step(step))
            a = int(step["a"])
            r = float(step["r"])
            if reward_benchmark is not None:
                recomputed = _recomputed_reward(step, rec, reward_benchmark)
                if recomputed is not None:
                    r = recomputed
            done = bool(step["done"])
            if done and drift_raw is not None:
                # Sell-side sign convention matches _compute_shortfall: a rising market
                # (positive drift) lowers a seller's shortfall, hence raises the reward.
                r = r - (drift_raw if str(rec.get("side")) == "SELL" else -drift_raw)
            # Shaping is applied HERE, at fit time, never in the log -- which is what lets
            # lambda be re-swept over already-simulated episodes at no GPU cost.
            r += inventory_penalty(step["inv_rem"], inventory_lambda)

            visits[s, a] += 1
            target = r
            if not done and i + 1 < len(traj):
                s_next = state_to_index(_obs_from_step(traj[i + 1]))
                target += gamma * q[s_next].max()

            step_alpha = (1.0 / visits[s, a]) if alpha_mode == "visit-count" else alpha
            q[s, a] += step_alpha * (target - q[s, a])
            n_steps += 1

    return q, visits, n_eps, n_steps, skipped


def stability(records, n_points=5, **kw):
    """Answers "is it converging, or just tracking noise?" -- refit on growing
    prefixes of the SAME episodes and count how many greedy actions change
    between successive fits.

    A converging fit settles: later prefixes change few or no actions. A fit
    dominated by reward noise keeps reshuffling no matter how many episodes are
    added. This is the direct, quantitative version of eyeing successive policy
    grids and noticing cells flip -- and because it refits already-simulated
    episodes, it costs milliseconds rather than a re-run.
    """
    usable = [r for r in records if r.get("trajectory")]
    if len(usable) < n_points * 2:
        return []
    cuts = [int(len(usable) * (i + 1) / n_points) for i in range(n_points)]
    rows, prev_greedy, prev_visited = [], None, None
    for cut in cuts:
        q, visits, *_ = refit(usable[:cut], **kw)
        visited = (visits > 0).any(axis=1)
        greedy = q.argmax(axis=1)
        changed = None
        if prev_greedy is not None:
            both = visited & prev_visited
            changed = int((greedy[both] != prev_greedy[both]).sum()), int(both.sum())
        rows.append({"episodes": cut, "states": int(visited.sum()), "changed": changed})
        prev_greedy, prev_visited = greedy, visited
    return rows


def penalty_sweep(records, lambdas, **kw):
    """Refit once per lambda over the SAME episodes and report how the policy changes.

    What this can and cannot tell you matters. It produces POLICIES for free -- no
    simulation -- so it is the right way to screen which lambdas are worth GPU time.
    It cannot tell you the resulting shortfall: measuring that requires running the
    policy in the market, because a different policy produces different fills. So the
    workflow is: sweep here to pick two or three lambdas, then spend GPU measuring
    those, and plot the cost/risk frontier from the measured runs.
    """
    usable = [r for r in records if r.get("trajectory")]
    rows = []
    baseline = None
    for lam in lambdas:
        q, visits, n_eps, _, _ = refit(usable, inventory_lambda=lam, **kw)
        visited = (visits > 0).any(axis=1)
        greedy = q.argmax(axis=1)
        if baseline is None:
            baseline = (greedy.copy(), visited.copy())
            changed = None
        else:
            both = visited & baseline[1]
            changed = (int((greedy[both] != baseline[0][both]).sum()), int(both.sum()))
        rows.append({
            "lam": lam,
            "mean_action": float(greedy[visited].mean()) if visited.any() else float("nan"),
            "changed_vs_lam0": changed,
            "n_states": int(visited.sum()),
        })
    observed_risk = [inventory_risk(r["trajectory"]) for r in usable]
    return rows, observed_risk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log", help="training .jsonl containing per-step trajectories")
    parser.add_argument("--stability", action="store_true",
                        help="refit on growing prefixes and report how many greedy actions "
                             "change between them -- the convergence check")
    parser.add_argument("--alpha-mode", choices=["visit-count", "fixed"], default="visit-count")
    parser.add_argument("--alpha", type=float, default=0.3, help="used only with --alpha-mode fixed")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--reward-benchmark", choices=["arrival", "prevailing"], default=None,
                        help="recompute per-step rewards against this benchmark instead of using "
                             "the logged ones (needs per-step mid/fills in the log). arrival keeps "
                             "true shortfall and makes the agent bear timing risk; prevailing removes "
                             "timing risk and its variance, leaving urgency to the inventory penalty")
    parser.add_argument("--inventory-penalty", type=float, default=0.0,
                        help="Almgren-Chriss running risk penalty lambda*x_t^2 per step")
    parser.add_argument("--penalty-sweep", default=None,
                        help="comma-separated lambdas to screen offline, e.g. 0,10,25,50,100")
    parser.add_argument("--drift-adjust", action="store_true",
                        help="subtract each episode's market drift from its terminal reward")
    parser.add_argument("--out", default=None, help="write the refit Q-table to this .npz")
    args = parser.parse_args()

    records = read_episodes(args.log)
    q, visits, n_eps, n_steps, skipped = refit(
        records, alpha_mode=args.alpha_mode, alpha=args.alpha,
        gamma=args.gamma, drift_adjust=args.drift_adjust,
        inventory_lambda=args.inventory_penalty, reward_benchmark=args.reward_benchmark)

    print("=" * 74)
    print(f"REFIT from {args.log}")
    print("=" * 74)
    print(f"episodes used     : {n_eps}   ({n_steps} decision points)")
    if skipped:
        print(f"episodes skipped  : {skipped}  (no trajectory logged -- predates trajectory logging)")
    if n_eps == 0:
        print("\nNothing to fit. Runs logged before trajectory logging existed cannot be refit;")
        print("their per-step decisions were never recorded and are not recoverable.")
        return
    print(f"alpha mode        : {args.alpha_mode}" + (f" ({args.alpha})" if args.alpha_mode == "fixed" else ""))
    print(f"drift-adjusted    : {args.drift_adjust}")

    visited = (visits > 0).any(axis=1)
    n_visited = int(visited.sum())
    print(f"states visited    : {n_visited}/{N_STATES}")
    print(f"visits per entry  : median={np.median(visits[visits > 0]):.0f}  max={visits.max()}")

    greedy = q.argmax(axis=1)[visited]
    counts = np.bincount(greedy, minlength=N_ACTIONS)
    print("\ngreedy action over visited states: " +
          "  ".join(f"{i}:{c}" for i, c in enumerate(counts)))
    differs = int((greedy != 1).sum())  # 1 = TWAP's action
    print(f"differs from TWAP : {differs}/{n_visited} ({differs / n_visited:.0%})")
    spread = q[visited].max(axis=1) - q[visited].min(axis=1)
    print(f"Q-value spread    : median={np.median(spread):.2f}  max={spread.max():.2f}")

    if args.penalty_sweep:
        lambdas = [float(x) for x in args.penalty_sweep.split(",")]
        rows, observed_risk = penalty_sweep(
            records, lambdas, alpha_mode=args.alpha_mode, alpha=args.alpha,
            gamma=args.gamma, drift_adjust=args.drift_adjust)
        print("\nINVENTORY-PENALTY SWEEP (policies only -- shortfall needs a live run)")
        print(f"{'lambda':>9}  {'mean action':>12}  {'states':>7}  changed vs lambda=0")
        for r in rows:
            ch = "-" if r["changed_vs_lam0"] is None else f"{r['changed_vs_lam0'][0]}/{r['changed_vs_lam0'][1]}"
            print(f"{r['lam']:>9.1f}  {r['mean_action']:>12.2f}  {r['n_states']:>7d}  {ch:>18}")
        print("  mean action rising with lambda = the penalty is pushing the agent to")
        print("  liquidate faster, which is what a risk term should do.")
        if observed_risk:
            print(f"\n  observed inventory risk (sum x_t^2) over logged episodes: "
                  f"median={np.median(observed_risk):.2f}  "
                  f"[{np.min(observed_risk):.2f}, {np.max(observed_risk):.2f}]")
            print("  -> the x-axis of the cost/risk frontier; pair it with measured shortfall")
            print("     from a live run at each lambda to plot the frontier.")

    if args.stability:
        rows = stability(records, alpha_mode=args.alpha_mode, alpha=args.alpha,
                          gamma=args.gamma, drift_adjust=args.drift_adjust)
        print("\nCONVERGENCE — greedy actions changed between successive refits")
        if not rows:
            print("  too few episodes with trajectories to assess")
        else:
            for r in rows:
                if r["changed"] is None:
                    print(f"  after {r['episodes']:4d} episodes: {r['states']:2d} states visited   (baseline)")
                else:
                    ch, tot = r["changed"]
                    pct = ch / tot if tot else 0.0
                    print(f"  after {r['episodes']:4d} episodes: {r['states']:2d} states visited   "
                          f"{ch}/{tot} greedy actions changed ({pct:.0%})")
            last = rows[-1]["changed"]
            if last and last[1]:
                pct = last[0] / last[1]
                verdict = ("settling — later episodes barely move the policy" if pct <= 0.10 else
                           "still churning — the fit is tracking noise, not converging" if pct >= 0.25 else
                           "partially settled")
                print(f"  -> {verdict}")

    if args.out:
        policy = QLearningPolicy(alpha=args.alpha, gamma=args.gamma)
        policy.q = q
        policy.episodes_trained = n_eps
        policy.epsilon = policy.epsilon_min
        policy.save(args.out)
        print(f"\nwritten to {args.out} (epsilon set to its floor -- this table is for greedy evaluation)")


if __name__ == "__main__":
    main()
