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
from rl_execution.qlearning import N_ACTIONS, N_STATES, QLearningPolicy, state_to_index


def _obs_from_step(step):
    return {
        "time_remaining_frac": step["t_rem"],
        "inventory_remaining_frac": step["inv_rem"],
        "spread_bucket": step["spread"],
        "vol_bucket": step["vol"],
        "ofi_bucket": step["ofi"],
    }


def refit(records, alpha_mode="visit-count", alpha=0.3, gamma=1.0, drift_adjust=False):
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
            done = bool(step["done"])
            if done and drift_raw is not None:
                # Sell-side sign convention matches _compute_shortfall: a rising market
                # (positive drift) lowers a seller's shortfall, hence raises the reward.
                r = r - (drift_raw if str(rec.get("side")) == "SELL" else -drift_raw)

            visits[s, a] += 1
            target = r
            if not done and i + 1 < len(traj):
                s_next = state_to_index(_obs_from_step(traj[i + 1]))
                target += gamma * q[s_next].max()

            step_alpha = (1.0 / visits[s, a]) if alpha_mode == "visit-count" else alpha
            q[s, a] += step_alpha * (target - q[s, a])
            n_steps += 1

    return q, visits, n_eps, n_steps, skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log", help="training .jsonl containing per-step trajectories")
    parser.add_argument("--alpha-mode", choices=["visit-count", "fixed"], default="visit-count")
    parser.add_argument("--alpha", type=float, default=0.3, help="used only with --alpha-mode fixed")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--drift-adjust", action="store_true",
                        help="subtract each episode's market drift from its terminal reward")
    parser.add_argument("--out", default=None, help="write the refit Q-table to this .npz")
    args = parser.parse_args()

    records = read_episodes(args.log)
    q, visits, n_eps, n_steps, skipped = refit(
        records, alpha_mode=args.alpha_mode, alpha=args.alpha,
        gamma=args.gamma, drift_adjust=args.drift_adjust)

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

    if args.out:
        policy = QLearningPolicy(alpha=args.alpha, gamma=args.gamma)
        policy.q = q
        policy.episodes_trained = n_eps
        policy.epsilon = policy.epsilon_min
        policy.save(args.out)
        print(f"\nwritten to {args.out} (epsilon set to its floor -- this table is for greedy evaluation)")


if __name__ == "__main__":
    main()
