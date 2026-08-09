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
from rl_execution.logging_utils import (JsonlLogger, drift_bps as _drift,
                                         shortfall_bps as _bps, trajectory_step as _traj)
from rl_execution.qlearning import (N_ACTIONS, N_STATES, STATE_FEATURES, QLearningPolicy,
                                     inventory_penalty)


def train(env: ExecutionEnv, policy: QLearningPolicy, n_episodes: int, checkpoint_path: str,
          checkpoint_every: int = 25, run_name="qlearning_train", out_path="logs/train.jsonl",
          side=None, max_hours=None, inventory_penalty_lambda=0.0):
    """max_hours: stop cleanly once this much wall-clock has elapsed, whatever the
    episode count. Episode cost varies ~7x with market regime (125s-1170s observed),
    so an episode count alone gives no usable finish time -- which matters when a
    booked GPU block has to fit training AND the evaluation that follows it.
    """
    logger = JsonlLogger(out_path)
    start_episode = policy.episodes_trained
    run_start = time.perf_counter()
    budget_s = max_hours * 3600.0 if max_hours else None

    for ep in range(start_episode, start_episode + n_episodes):
        if budget_s is not None:
            elapsed = time.perf_counter() - run_start
            if elapsed >= budget_s:
                print(f"[train] wall-clock budget reached ({elapsed / 3600:.2f}h of {max_hours}h) "
                      f"after {policy.episodes_trained} episodes -- stopping cleanly")
                break
        ep_start = time.perf_counter()
        obs, info = env.reset(side=side)
        done = False
        trajectory = []
        while not done:
            action = policy.select_action(obs)
            next_obs, reward, done, info = env.step(action)
            # Log the TRUE reward, learn from the shaped one. Baking the penalty into the
            # log would freeze lambda into the data and forfeit the ability to re-sweep it
            # offline (see refit_qtable) -- the whole point of logging trajectories.
            trajectory.append(_traj(obs, action, reward, done,
                                     mid=info.get("step_mid"),
                                     fills=info.get("step_fills"),
                                     quote=info.get("step_quote")))
            shaped = reward + inventory_penalty(obs["inventory_remaining_frac"],
                                                 inventory_penalty_lambda)
            policy.update(obs, action, shaped, next_obs, done)
            obs = next_obs
        policy.end_episode()
        ep_elapsed = time.perf_counter() - ep_start

        logger.log_episode(
            run_name=run_name, seed_day=info["seed_day"], t0=info["t0"], side=info["side"],
            Q=info["Q"], sampling_type=info["sampling_type"], depth_noise=info["depth_noise"],
            ddim_nsteps=info["ddim_nsteps"], checkpoint=info["checkpoint"],
            policy_name=f"qlearning_eps{policy.epsilon:.3f}", wall_clock_total_s=info["wall_clock_total_s"],
            wall_clock_reconstruct_s=info["wall_clock_reconstruct_s"],
            wall_clock_simulate_s=info["wall_clock_simulate_s"], p_arrival=info["p_arrival"], p_final=info.get("p_final"),
            shortfall=info["shortfall"], shortfall_bps=_bps(info), drift_bps=_drift(info), reward=reward,
            n_resting_orders=info["n_resting_orders"], rem_quantity=info.get("rem_quantity"),
            fills=info["fills"], cond_z=info["cond_stats"], flow_mix=info["flow_mix"],
            execution_rate=info["execution_rate"], unique_mid_count=info["unique_mid_count"],
            trajectory=trajectory,
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
    parser.add_argument("--side", default=None, choices=["BUY", "SELL"],
                        help="fix the parent-order side; default = randomised per episode")
    parser.add_argument("--checkpoint", default="checkpoints/qtable.npz",
                        help="where to save/resume the Q-table (not the TRADES checkpoint -- see --ckpt-path)")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--out", default="logs/train.jsonl")
    parser.add_argument("--run-name", default="qlearning_train")
    parser.add_argument("--alpha", type=float, default=0.3, help="Q-learning rate (fixed mode only)")
    parser.add_argument("--alpha-mode", choices=["fixed", "visit-count"], default="fixed",
                        help="visit-count sets alpha=1/N(s,a), making Q the running mean return -- "
                             "the principled estimator when gamma=1 and reward is terminal-only")
    parser.add_argument("--epsilon-decay", type=float, default=0.97, help="per-episode epsilon decay")
    parser.add_argument("--gamma", type=float, default=1.0, help="discount factor")
    parser.add_argument("--inventory-penalty", type=float, default=0.0,
                        help="Almgren-Chriss running risk penalty: charge lambda*x_t^2 per step on "
                             "inventory still held. Dense and deterministic, so it adds learning "
                             "signal with no added variance. Applied at LEARNING time only -- logged "
                             "rewards and reported shortfall stay true. Try 10-100; 0 = off")
    parser.add_argument("--max-hours", type=float, default=None,
                        help="stop cleanly after this much wall-clock, whatever the episode count "
                             "(episode cost varies ~7x with market regime, so a count alone gives "
                             "no usable finish time within a booked GPU block)")
    args = parser.parse_args()

    if args.resume and os.path.exists(args.checkpoint):
        policy = QLearningPolicy.load(args.checkpoint, random_state=np.random.RandomState())
        print(f"resumed from {args.checkpoint}: {policy.episodes_trained} episodes already trained")
    else:
        policy = QLearningPolicy(alpha=args.alpha, epsilon_decay=args.epsilon_decay,
                                  gamma=args.gamma, random_state=np.random.RandomState(),
                                  alpha_mode=args.alpha_mode)
    print(f"[train] states={N_STATES} (features={STATE_FEATURES}) actions={N_ACTIONS}  "
          f"alpha={policy.alpha} ({policy.alpha_mode}) eps_decay={policy.epsilon_decay} gamma={policy.gamma}  "
          f"side={args.side or 'random'} inv_penalty={args.inventory_penalty}")

    env = ExecutionEnv(symbol=args.symbol, data_dir=args.data_dir, sampling_type=args.sampling_type,
                        ddim_nsteps=args.ddim_nsteps, depth_noise=args.depth_noise,
                        checkpoint_path=args.ckpt_path,
                        reward_mode=args.reward_mode, reward_benchmark=args.reward_benchmark)
    train(env, policy, args.n_episodes, args.checkpoint, checkpoint_every=args.checkpoint_every,
          run_name=args.run_name, out_path=args.out, side=args.side, max_hours=args.max_hours,
          inventory_penalty_lambda=args.inventory_penalty)
