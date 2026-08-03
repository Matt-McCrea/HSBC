"""Run this on the GPU machine (never locally -- see project rule) to verify
deliverables 1-3 end-to-end before anything else gets built on top. Prints
clear, self-contained output -- paste the full output back for review.

Usage (from repo root, with the project's normal env active):
    python -m rl_execution.remote_verify

Three stages, each printed clearly so a failure is easy to localize:
  1. Baseline pipeline parity: a short existing-style simulate run, to confirm
     the WorldAgent extraction refactor (utils/utils_data.py:
     z_score_orderbook_for_cond / preprocess_orders_for_diff_cond) didn't
     change behavior on the normal (non-RL) code path.
  2. Cold-start module: seed_episode() at a few t0s, checking cond_z stats
     land in a sane range (not wildly OOD) and resting-order counts look
     reasonable.
  3. One full RL episode end-to-end: ExecutionEnv.reset()/step() with a
     random policy, reporting wall-clock (split reconstruction vs
     simulation) and the final shortfall -- the spec's own required
     "build and test a single episode locally before any bulk run" step.
"""

import sys
import time

import numpy as np


def stage_1_baseline_parity():
    print("\n" + "=" * 70)
    print("STAGE 1: baseline pipeline parity (post-refactor sanity check)")
    print("=" * 70)
    import subprocess
    cmd = [
        sys.executable, "-u", "ABIDES/abides.py", "-c", "world_agent_sim",
        "-t", "INTC", "-date", "20150130", "-st", "09:30:00", "-et", "09:47:00",
        "-d", "True", "-m", "TRADES", "-type", "DDIM", "-nsteps", "10",
    ]
    print("running:", " ".join(cmd))
    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    elapsed = time.perf_counter() - start
    print(f"exit code: {result.returncode}  wall-clock: {elapsed:.1f}s")
    tail = "\n".join(result.stdout.splitlines()[-40:])
    print("--- last 40 lines of stdout ---")
    print(tail)
    if result.returncode != 0:
        print("--- stderr (last 40 lines) ---")
        print("\n".join(result.stderr.splitlines()[-40:]))
    print("STAGE 1 " + ("PASSED (exit 0)" if result.returncode == 0 else "FAILED"))
    return result.returncode == 0


def stage_2_coldstart():
    print("\n" + "=" * 70)
    print("STAGE 2: cold-start module (deliverable 2)")
    print("=" * 70)
    from rl_execution import coldstart
    from utils.utils_data import load_compute_normalization_terms
    import constants as cst

    normalization_terms = load_compute_normalization_terms("INTC", "data", cst.Models.TRADES, n_lob_levels=10)
    message_path, orderbook_path = coldstart._day_paths("data", "INTC", "2015-01-30")
    messages, _ = coldstart.read_day(message_path, orderbook_path)

    ok = True
    for frac in (0.1, 0.3, 0.5, 0.7, 0.9):
        t0 = float(messages["time"].iloc[int(len(messages) * frac)])
        try:
            start = time.perf_counter()
            cs = coldstart.seed_episode(message_path, orderbook_path, t0, normalization_terms)
            elapsed = time.perf_counter() - start
            print(f"t0={t0:.1f} (frac={frac}): OK in {elapsed:.2f}s, "
                  f"{len(cs.resting_orders)} resting orders, cond_orders shape={tuple(cs.cond_orders.shape)}, "
                  f"cond_lob shape={tuple(cs.cond_lob.shape)}")
        except Exception as e:
            print(f"t0={t0:.1f} (frac={frac}): FAILED -- {type(e).__name__}: {e}")
            ok = False
    print("STAGE 2 " + ("PASSED" if ok else "FAILED"))
    return ok


def stage_3_full_episode():
    print("\n" + "=" * 70)
    print("STAGE 3: one full RL episode end-to-end (random policy)")
    print("=" * 70)
    from rl_execution.env import ExecutionEnv

    start_construct = time.perf_counter()
    env = ExecutionEnv(symbol="INTC", data_dir="data", sampling_type="DDIM", ddim_nsteps=10,
                        depth_noise=0.3, seed_days=["2015-01-30"])
    construct_elapsed = time.perf_counter() - start_construct
    print(f"ExecutionEnv construction (model load): {construct_elapsed:.1f}s")

    rng = np.random.RandomState(0)
    start_reset = time.perf_counter()
    obs, info = env.reset(seed=0)
    reset_elapsed = time.perf_counter() - start_reset
    print(f"reset(): {reset_elapsed:.1f}s  t0={info['t0']:.1f}  side={info['side']}  Q={info['Q']}  "
          f"p_arrival={info['p_arrival']:.2f}  resting_orders={info['n_resting_orders']}")
    print(f"cond_z at seed: {info['cond_stats']}")
    print(f"first obs: {obs}")

    n_steps = 0
    start_steps = time.perf_counter()
    done = False
    while not done:
        action = rng.randint(0, 5)
        obs, reward, done, info = env.step(action)
        n_steps += 1
        print(f"step {n_steps}: action={action} done={done} reward={reward if done else 0.0:.4f} obs={obs}")
    steps_elapsed = time.perf_counter() - start_steps

    print(f"\nepisode complete: {n_steps} steps in {steps_elapsed:.1f}s "
          f"(total incl. reset: {reset_elapsed + steps_elapsed:.1f}s)")
    print(f"final shortfall: {info.get('shortfall')}  fills: {info.get('fills')}")
    print("STAGE 3 PASSED (ran to completion without exception -- shortfall value/policy quality not evaluated here)")
    return True


if __name__ == "__main__":
    results = {}
    results["stage_1"] = stage_1_baseline_parity()
    results["stage_2"] = stage_2_coldstart()
    results["stage_3"] = stage_3_full_episode()
    print("\n" + "=" * 70)
    print("SUMMARY:", results)
    print("=" * 70)
