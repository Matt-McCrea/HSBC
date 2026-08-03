"""Evaluation / comparison script (deliverable 6): the trained policy and a
fixed TWAP baseline, across a held-out set of seed timestamps, for both:
  - the depth-noise sampler (fast, accelerated DDIM + the depth-noise fix)
  - full DDPM-100 (slow, the realistic-but-expensive baseline), run for as
    many episodes as fit in the SAME wall-clock budget the depth-noise arm
    used -- directly demonstrating the accelerated sampler's value, per the
    spec's own note that this is the single most important result if time is
    short.

Run on the GPU machine (never locally):
    python -m rl_execution.evaluate --checkpoint checkpoints/qtable.npz --n-seeds 30
"""

import argparse
import statistics
import time

import numpy as np

from rl_execution import coldstart, env as env_module
from rl_execution.env import ExecutionEnv
from rl_execution.logging_utils import JsonlLogger
from rl_execution.qlearning import QLearningPolicy, TWAPPolicy


def generate_held_out_seeds(data_dir, symbol, n_seeds, seq_len=256, episode_seconds=300,
                             Q_range=(1000, 5000), seed=123, seed_days=None):
    """A fixed list of (seed_day, t0, side, Q) tuples, generated once with a
    dedicated RandomState so the SAME held-out set is reused across every
    policy/sampler combination compared -- required for a fair comparison.
    """
    rng = np.random.RandomState(seed)
    seed_days = seed_days or env_module.list_trading_days(data_dir, symbol)
    seeds = []
    for _ in range(n_seeds):
        day = str(rng.choice(seed_days))
        message_path, orderbook_path = coldstart._day_paths(data_dir, symbol, day)
        messages, _ = coldstart.read_day(message_path, orderbook_path)
        lo = float(messages["time"].iloc[seq_len + 10])
        hi = float(messages["time"].iloc[-1]) - episode_seconds - 5
        t0 = float(rng.uniform(lo, hi))
        side = str(rng.choice(["BUY", "SELL"]))
        Q = int(rng.randint(Q_range[0], Q_range[1]))
        seeds.append({"seed_day": day, "t0": t0, "side": side, "Q": Q})
    return seeds


def evaluate_policy(env: ExecutionEnv, policy, seeds, logger: JsonlLogger, run_name, policy_name,
                     wall_clock_budget=None):
    """Run `policy` (greedy, no exploration) across `seeds` in order, stopping
    early if wall_clock_budget (seconds) is exhausted -- used to budget-match
    the DDPM-100 arm to the depth-noise arm's actual wall-clock usage.
    Returns (shortfalls, wall_clock_used, n_episodes_run).
    """
    shortfalls = []
    wall_clock_used = 0.0
    n_run = 0
    for s in seeds:
        if wall_clock_budget is not None and wall_clock_used >= wall_clock_budget:
            break
        start = time.perf_counter()
        obs, info = env.reset(t0=s["t0"], side=s["side"], Q=s["Q"], seed_day=s["seed_day"])
        done = False
        reward = 0.0
        while not done:
            action = policy.select_action(obs, greedy=True)
            obs, reward, done, info = env.step(action)
        elapsed = time.perf_counter() - start
        wall_clock_used += elapsed
        n_run += 1
        shortfalls.append(info["shortfall"])

        logger.log_episode(
            run_name=run_name, seed_day=info["seed_day"], t0=info["t0"], side=info["side"],
            Q=info["Q"], sampling_type=info["sampling_type"], depth_noise=info["depth_noise"],
            policy_name=policy_name, wall_clock_total_s=info["wall_clock_total_s"],
            wall_clock_reconstruct_s=info["wall_clock_reconstruct_s"],
            wall_clock_simulate_s=info["wall_clock_simulate_s"], p_arrival=info["p_arrival"],
            shortfall=info["shortfall"], reward=reward, n_resting_orders=info["n_resting_orders"],
            fills=info["fills"], cond_z=info["cond_stats"], flow_mix=info["flow_mix"],
            execution_rate=info["execution_rate"], unique_mid_count=info["unique_mid_count"],
        )
        print(f"  [{policy_name}] seed {n_run}/{len(seeds)}: shortfall={info['shortfall']:.4f} "
              f"wall_clock={elapsed:.1f}s")
    return shortfalls, wall_clock_used, n_run


def _stats(shortfalls):
    if not shortfalls:
        return {"n": 0, "mean": None, "stderr": None}
    n = len(shortfalls)
    mean = statistics.mean(shortfalls)
    stderr = (statistics.stdev(shortfalls) / (n ** 0.5)) if n > 1 else 0.0
    return {"n": n, "mean": mean, "stderr": stderr}


def run_comparison(data_dir="data", symbol="INTC", checkpoint_path=None, n_seeds=30,
                    depth_noise=0.3, ddim_nsteps=10, out_path="logs/evaluate.jsonl", eval_seed=123):
    seeds = generate_held_out_seeds(data_dir, symbol, n_seeds, seed=eval_seed)
    logger = JsonlLogger(out_path)

    if checkpoint_path:
        trained_policy = QLearningPolicy.load(checkpoint_path)
    else:
        trained_policy = None
    twap_policy = TWAPPolicy()

    results = {}

    print(f"\n=== depth-noise arm (DDIM {ddim_nsteps} steps, depth_noise={depth_noise}) ===")
    env_dn = ExecutionEnv(symbol=symbol, data_dir=data_dir, sampling_type="DDIM",
                           ddim_nsteps=ddim_nsteps, depth_noise=depth_noise, seed_days=None)
    dn_twap, dn_wallclock, _ = evaluate_policy(env_dn, twap_policy, seeds, logger, "eval_depth_noise", "twap")
    results["depth_noise/twap"] = _stats(dn_twap)
    if trained_policy is not None:
        dn_trained, dn_wallclock_trained, _ = evaluate_policy(
            env_dn, trained_policy, seeds, logger, "eval_depth_noise", "qlearning")
        results["depth_noise/qlearning"] = _stats(dn_trained)
        dn_wallclock = max(dn_wallclock, dn_wallclock_trained)

    print(f"\n=== DDPM-100 arm (budget-matched to {dn_wallclock:.1f}s) ===")
    env_ddpm = ExecutionEnv(symbol=symbol, data_dir=data_dir, sampling_type="DDPM",
                             depth_noise=0.0, seed_days=None)
    ddpm_twap, ddpm_wallclock, n_ddpm_twap = evaluate_policy(
        env_ddpm, twap_policy, seeds, logger, "eval_ddpm100", "twap", wall_clock_budget=dn_wallclock)
    results["ddpm100/twap"] = _stats(ddpm_twap)
    results["ddpm100/twap"]["n_episodes_in_budget"] = n_ddpm_twap
    if trained_policy is not None:
        ddpm_trained, _, n_ddpm_trained = evaluate_policy(
            env_ddpm, trained_policy, seeds, logger, "eval_ddpm100", "qlearning", wall_clock_budget=dn_wallclock)
        results["ddpm100/qlearning"] = _stats(ddpm_trained)
        results["ddpm100/qlearning"]["n_episodes_in_budget"] = n_ddpm_trained

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY (shortfall: lower is better; TWAP is the baseline)")
    print(f"depth-noise arm ran all {len(seeds)} held-out seeds in {dn_wallclock:.1f}s")
    for key, r in results.items():
        n_note = f" ({r['n_episodes_in_budget']}/{len(seeds)} seeds fit in budget)" if "n_episodes_in_budget" in r else ""
        print(f"  {key:25s} n={r['n']:3d}  mean_shortfall={r['mean']}  stderr={r['stderr']}{n_note}")
    print("=" * 70)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="INTC")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--checkpoint", default=None, help="trained Q-table .npz; omit to only evaluate TWAP")
    parser.add_argument("--n-seeds", type=int, default=30)
    parser.add_argument("--depth-noise", type=float, default=0.3)
    parser.add_argument("--ddim-nsteps", type=int, default=10)
    parser.add_argument("--out", default="logs/evaluate.jsonl")
    parser.add_argument("--eval-seed", type=int, default=123)
    args = parser.parse_args()

    run_comparison(data_dir=args.data_dir, symbol=args.symbol, checkpoint_path=args.checkpoint,
                    n_seeds=args.n_seeds, depth_noise=args.depth_noise, ddim_nsteps=args.ddim_nsteps,
                    out_path=args.out, eval_seed=args.eval_seed)
