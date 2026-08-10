"""Benchmarking harness (deliverable 4): runs N episodes and reports
wall-clock per episode (reconstruction vs simulation), plus the diagnostic
metrics already defined elsewhere in the project (flow mix, execution rate,
unique mid count, cond_z) per episode -- via the shared JSON-lines logging
schema (rl_execution/logging_utils.py), so pilot runs can be sanity-checked
without waiting for full training to finish.

Run on the GPU machine (never locally):
    python -m rl_execution.benchmark --n-episodes 20 --out logs/bench.jsonl
"""

import argparse
import statistics
import time

import numpy as np

from rl_execution.env import ExecutionEnv
from rl_execution.logging_utils import (JsonlLogger, drift_bps as _drift,
                                         shortfall_bps as _bps, trajectory_step as _traj)


def run_benchmark(env: ExecutionEnv, n_episodes: int, policy=None, run_name="benchmark", out_path="logs/benchmark.jsonl"):
    """policy(obs) -> action index; defaults to a uniform-random policy over
    the 5 action levels (this harness measures the environment, not a
    specific agent's quality).
    """
    rng = np.random.RandomState()
    policy = policy or (lambda obs: rng.randint(0, 5))
    logger = JsonlLogger(out_path)

    records = []
    for ep in range(n_episodes):
        ep_start = time.perf_counter()
        obs, info = env.reset()
        done = False
        trajectory = []
        while not done:
            action = policy(obs)
            prev_obs = obs
            obs, reward, done, info = env.step(action)
            trajectory.append(_traj(prev_obs, action, reward, done,
                                     mid=info.get("step_mid"),
                                     fills=info.get("step_fills"),
                                     quote=info.get("step_quote")))
        ep_elapsed = time.perf_counter() - ep_start

        record = logger.log_episode(
            run_name=run_name, seed_day=info["seed_day"], t0=info["t0"], side=info["side"],
            Q=info["Q"], sampling_type=info["sampling_type"], depth_noise=info["depth_noise"],
            ddim_nsteps=info["ddim_nsteps"], checkpoint=info["checkpoint"],
            policy_name="random", wall_clock_total_s=info["wall_clock_total_s"],
            wall_clock_reconstruct_s=info["wall_clock_reconstruct_s"],
            wall_clock_simulate_s=info["wall_clock_simulate_s"], p_arrival=info["p_arrival"], p_final=info.get("p_final"),
            shortfall=info["shortfall"], shortfall_bps=_bps(info), drift_bps=_drift(info), reward=reward,
            n_resting_orders=info["n_resting_orders"], rem_quantity=info.get("rem_quantity"),
            fills=info["fills"], cond_z=info["cond_stats"], flow_mix=info["flow_mix"],
            execution_rate=info["execution_rate"], unique_mid_count=info["unique_mid_count"],
            trajectory=trajectory,
        )
        records.append(record)
        print(f"episode {ep + 1}/{n_episodes}: wall_clock={ep_elapsed:.1f}s "
              f"(reconstruct={info['wall_clock_reconstruct_s']:.2f}s, "
              f"simulate={info['wall_clock_simulate_s']:.1f}s)  "
              f"shortfall={info['shortfall']:.4f}  exec_rate={info['execution_rate']}  "
              f"unique_mids={info['unique_mid_count']}")

    _print_summary(records)
    return records


def _print_summary(records):
    if not records:
        print("no episodes completed")
        return
    total = [r["wall_clock_total_s"] for r in records]
    recon = [r["wall_clock_reconstruct_s"] for r in records]
    sim = [r["wall_clock_simulate_s"] for r in records]
    shortfalls = [r["shortfall"] for r in records if r["shortfall"] is not None]
    print("\n" + "=" * 60)
    print(f"BENCHMARK SUMMARY over {len(records)} episodes")
    print(f"  wall-clock total:       mean={statistics.mean(total):.1f}s  "
          f"min={min(total):.1f}s  max={max(total):.1f}s")
    print(f"  wall-clock reconstruct: mean={statistics.mean(recon):.2f}s")
    print(f"  wall-clock simulate:    mean={statistics.mean(sim):.1f}s")
    if shortfalls:
        print(f"  shortfall:              mean={statistics.mean(shortfalls):.4f}  "
              f"stdev={statistics.stdev(shortfalls) if len(shortfalls) > 1 else 0.0:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-episodes", type=int, default=20)
    parser.add_argument("--symbol", default="INTC")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--sampling-type", default="DDIM")
    parser.add_argument("--ddim-nsteps", type=int, default=10)
    parser.add_argument("--depth-noise", type=float, default=0.3)
    parser.add_argument("--reward-mode", choices=["terminal", "per-step"], default="terminal",
                        help="per-step pays each fill's cost at the decision point whose child "
                             "order produced it -- identical total, but dense, which is what makes "
                             "credit assignment tractable at a few hundred episodes")
    parser.add_argument("--reward-benchmark", choices=["arrival", "prevailing"], default="arrival",
                        help="price each fill is scored against: arrival mid (reward series sums to "
                             "true shortfall) or the prevailing mid at that step (removes market "
                             "drift by construction -- drift is common-mode noise that dominates "
                             "reward variance). Reported shortfall is always vs arrival either way")
    parser.add_argument("--ckpt-path", default=None,
                        help="exact TRADES checkpoint to simulate with; default = lowest val-loss for the symbol")
    parser.add_argument("--world-mode", choices=["generative", "replay"], default="generative",
                        help="replay plays the REAL message stream from t0 rather than "
                             "generating one: no diffusion, so no model is loaded and no GPU "
                             "is needed. Use it to run the same policy against the real market")
    parser.add_argument("--out", default="logs/benchmark.jsonl")
    parser.add_argument("--run-name", default="benchmark")
    args = parser.parse_args()

    env = ExecutionEnv(symbol=args.symbol, data_dir=args.data_dir, sampling_type=args.sampling_type,
                        ddim_nsteps=args.ddim_nsteps, depth_noise=args.depth_noise,
                        checkpoint_path=args.ckpt_path,
                        reward_mode=args.reward_mode, reward_benchmark=args.reward_benchmark,
                        world_mode=args.world_mode)
    run_benchmark(env, args.n_episodes, run_name=args.run_name, out_path=args.out)
