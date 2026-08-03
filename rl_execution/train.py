"""Q-learning training loop (deliverable 5), with checkpointing so a run can
be resumed if a 72-hour GPU booking ends mid-training.

Run on the GPU machine (never locally):
    python -m rl_execution.train --n-episodes 1000 --checkpoint checkpoints/qtable.npz
    python -m rl_execution.train --n-episodes 1000 --checkpoint checkpoints/qtable.npz --resume
"""

import argparse
import os
import time

import numpy as np

from rl_execution.env import ExecutionEnv
from rl_execution.logging_utils import JsonlLogger
from rl_execution.qlearning import QLearningPolicy


def train(env: ExecutionEnv, policy: QLearningPolicy, n_episodes: int, checkpoint_path: str,
          checkpoint_every: int = 25, run_name="qlearning_train", out_path="logs/train.jsonl"):
    logger = JsonlLogger(out_path)
    start_episode = policy.episodes_trained

    for ep in range(start_episode, start_episode + n_episodes):
        ep_start = time.perf_counter()
        obs, info = env.reset()
        done = False
        while not done:
            action = policy.select_action(obs)
            next_obs, reward, done, info = env.step(action)
            policy.update(obs, action, reward, next_obs, done)
            obs = next_obs
        policy.end_episode()
        ep_elapsed = time.perf_counter() - ep_start

        logger.log_episode(
            run_name=run_name, seed_day=info["seed_day"], t0=info["t0"], side=info["side"],
            Q=info["Q"], sampling_type=info["sampling_type"], depth_noise=info["depth_noise"],
            policy_name=f"qlearning_eps{policy.epsilon:.3f}", wall_clock_total_s=info["wall_clock_total_s"],
            wall_clock_reconstruct_s=info["wall_clock_reconstruct_s"],
            wall_clock_simulate_s=info["wall_clock_simulate_s"], p_arrival=info["p_arrival"],
            shortfall=info["shortfall"], reward=reward, n_resting_orders=info["n_resting_orders"],
            fills=info["fills"], cond_z=info["cond_stats"], flow_mix=info["flow_mix"],
            execution_rate=info["execution_rate"], unique_mid_count=info["unique_mid_count"],
        )
        print(f"episode {ep + 1} (trained {policy.episodes_trained}): "
              f"wall_clock={ep_elapsed:.1f}s  shortfall={info['shortfall']:.4f}  "
              f"epsilon={policy.epsilon:.3f}  alpha={policy.alpha:.3f}")

        if (ep + 1) % checkpoint_every == 0:
            policy.save(checkpoint_path)
            print(f"  checkpoint saved to {checkpoint_path} ({policy.episodes_trained} episodes trained)")

    policy.save(checkpoint_path)
    print(f"training complete: checkpoint saved to {checkpoint_path} ({policy.episodes_trained} episodes trained)")
    return policy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-episodes", type=int, default=500)
    parser.add_argument("--symbol", default="INTC")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--sampling-type", default="DDIM")
    parser.add_argument("--ddim-nsteps", type=int, default=10)
    parser.add_argument("--depth-noise", type=float, default=0.3)
    parser.add_argument("--checkpoint", default="checkpoints/qtable.npz")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--out", default="logs/train.jsonl")
    parser.add_argument("--run-name", default="qlearning_train")
    args = parser.parse_args()

    if args.resume and os.path.exists(args.checkpoint):
        policy = QLearningPolicy.load(args.checkpoint, random_state=np.random.RandomState())
        print(f"resumed from {args.checkpoint}: {policy.episodes_trained} episodes already trained")
    else:
        policy = QLearningPolicy(random_state=np.random.RandomState())

    env = ExecutionEnv(symbol=args.symbol, data_dir=args.data_dir, sampling_type=args.sampling_type,
                        ddim_nsteps=args.ddim_nsteps, depth_noise=args.depth_noise)
    train(env, policy, args.n_episodes, args.checkpoint, checkpoint_every=args.checkpoint_every,
          run_name=args.run_name, out_path=args.out)
