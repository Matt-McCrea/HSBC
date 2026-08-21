"""Gym-style RL optimal-execution environment wrapping ABIDES + the TRADES
world agent + the cold-start logic in rl_execution/coldstart.py.

reset() picks a random seed timestamp t0 (and day), cold-starts an episode at
t0 (no ~15-minute replay -- see coldstart.py), launches a fresh ABIDES Kernel
in a background thread, and returns the first observation. step(action)
hands the action to the running episode via a queue and blocks for the next
observation -- see execution_agent.py's module docstring for why a thread udp
bridge (rather than embedding the policy inside the ABIDES agent) is the
right shape here: it's what lets the same environment run the trained policy
*or* a fixed TWAP baseline (deliverable 6) without touching ABIDES code.

The model checkpoint is loaded ONCE at construction and reused across every
reset() -- reloading per episode would be wasted GPU/disk time against a
500-1500 episode training budget.
"""

import os
import queue
import threading
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import constants as cst
from Kernel import Kernel
from agent.ExchangeAgent import ExchangeAgent
from models.diffusers.diffusion_engine import DiffusionEngine
from utils.utils_data import load_compute_normalization_terms

from rl_execution import coldstart
from rl_execution.execution_agent import N_DECISIONS, RLExecutionAgent
from rl_execution.orderbook_reconstructor import DEFAULT_MARKET_OPEN
from rl_execution.replay_world_agent import ReplayWorldAgent, build_replay_stream
from rl_execution.rl_world_agent import RLWorldAgent

EPISODE_SECONDS = N_DECISIONS * 30  # 5 minutes, fixed by the spec

# Earliest seed timestamp, as seconds after the 09:30 open. Requiring only enough
# CONDITIONING history (256 messages) is not enough: on a liquid day those elapse within
# seconds of the open, so t0 could land at ~09:32 when the real book still holds only a few
# hundred resting orders against a typical 3-4k. Seeded that thin, generation destabilises
# and the price runs away -- one observed episode moved the mid ~17% in 5 simulated minutes
# and produced a -1350bps shortfall, ~100x any other episode, purely from the thin start.
# 30 minutes also matches the rest of the project's convention of leaving the opening
# auction period alone (the standard pipeline replays 15 min from 09:30 before generating).
MIN_SECONDS_AFTER_OPEN = 1800.0

# Below this many reconstructed resting orders the seeded book is unrepresentatively thin;
# warn rather than reject, so a genuine mid-session collapse stays visible in the logs.
THIN_BOOK_WARNING = 1000

# Agent ids within an episode's Kernel. The exchange must be 0 (TradingAgent.kernelStarting
# finds it by type, but WorldAgent._update_active_limit_orders hardcodes agents[0]).
EXCHANGE_AGENT_ID = 0
WORLD_AGENT_ID = 1
EXEC_AGENT_ID = 2


def find_best_checkpoint(symbol, chosen_model=None):
    chosen_model = chosen_model or cst.Models.TRADES
    dir_path = Path(cst.DIR_SAVED_MODEL) / chosen_model.value
    best_val_loss = np.inf
    best_file = None
    for file in dir_path.iterdir():
        if symbol not in file.name:
            continue
        try:
            val_loss = float(file.name.split("=")[1].split("_")[0])
        except (IndexError, ValueError):
            continue
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_file = file
    if best_file is None:
        raise FileNotFoundError(f"no checkpoint for {symbol} under {dir_path}")
    return best_file


def load_model(symbol, sampling_type="DDIM", ddim_nsteps=10, ddim_eta=1.0,
                tail_steps=0, guidance_scale=1.0, churn_steps=0, churn_strength=0.0,
                checkpoint_path=None):
    """Programmatic (non-CLI) model + sampler setup, mirroring
    ABIDES/config/world_agent_sim.py's checkpoint-loading block and
    evaluation/diagnostics/open_loop_eval.py's load_model() template.
    """
    checkpoint_reference = Path(checkpoint_path) if checkpoint_path else find_best_checkpoint(symbol)
    checkpoint = torch.load(checkpoint_reference, map_location=cst.DEVICE, weights_only=False)
    checkpoint["hyper_parameters"]["chosen_model"] = cst.Models.TRADES
    config = checkpoint["hyper_parameters"]["config"]
    config.IS_WANDB = False
    config.CHOSEN_MODEL = cst.Models.TRADES
    config.SAMPLING_TYPE = sampling_type
    config.HYPER_PARAMETERS[cst.LearningHyperParameter.DDIM_ETA] = ddim_eta
    config.HYPER_PARAMETERS[cst.LearningHyperParameter.DDIM_NSTEPS] = ddim_nsteps
    config.HYPER_PARAMETERS[cst.LearningHyperParameter.DDIM_TAIL_STEPS] = tail_steps
    config.HYPER_PARAMETERS[cst.LearningHyperParameter.GUIDANCE_SCALE] = guidance_scale
    config.HYPER_PARAMETERS[cst.LearningHyperParameter.CHURN_STEPS] = churn_steps
    config.HYPER_PARAMETERS[cst.LearningHyperParameter.CHURN_STRENGTH] = churn_strength
    model = DiffusionEngine.load_from_checkpoint(checkpoint_reference, config=config, map_location=cst.DEVICE)
    model.eval()
    return model, config, checkpoint_reference


def list_trading_days(data_dir, symbol):
    day_dir = None
    for cand in os.listdir(os.path.join(data_dir, symbol)):
        full = os.path.join(data_dir, symbol, cand)
        if os.path.isdir(full) and cand.startswith(f"{symbol}_"):
            day_dir = full
            break
    dates = sorted({f.split("_")[1] for f in os.listdir(day_dir) if f.startswith(f"{symbol}_")})
    return dates


class ExecutionEnv:

    def __init__(self, symbol="INTC", data_dir="data", sampling_type="DDIM", ddim_nsteps=10,
                 depth_noise=0.3, checkpoint_path=None, seed_days=None, Q_range=(1000, 5000),
                 random_state=None, min_seconds_after_open=MIN_SECONDS_AFTER_OPEN,
                 reward_mode="terminal", reward_benchmark="arrival", world_mode="generative"):
        self.symbol = symbol
        self.data_dir = data_dir
        self.min_seconds_after_open = min_seconds_after_open
        self.reward_mode = reward_mode
        self.reward_benchmark = reward_benchmark
        # "replay" plays the real message stream from t0 instead of generating one. No
        # diffusion sampling, so no model is loaded at all and no GPU is touched.
        self.world_mode = world_mode
        self.depth_noise = depth_noise
        self.sampling_type = sampling_type
        self.ddim_nsteps = ddim_nsteps
        self.Q_range = Q_range
        self.rng = random_state or np.random.RandomState()

        if world_mode == "replay":
            # Deliberately not loaded: replay must be runnable on a CPU-only machine, and
            # loading the checkpoint would both waste time and pin a GPU for nothing.
            self.model, self.config, self.checkpoint_path = None, None, None
            from configuration import Configuration
            self.config = Configuration()
        else:
            self.model, self.config, self.checkpoint_path = load_model(
                symbol, sampling_type=sampling_type, ddim_nsteps=ddim_nsteps,
                checkpoint_path=checkpoint_path)
        # Self-documenting run banner: which checkpoint and sampler produced a given set of
        # numbers is exactly what the Results chapter has to state, and the checkpoint was
        # previously auto-selected and never recorded anywhere.
        print(f"[ExecutionEnv] world_mode={world_mode}  checkpoint={self.checkpoint_path}")
        print(f"[ExecutionEnv] sampler={sampling_type} ddim_nsteps={ddim_nsteps} depth_noise={depth_noise}")
        print(f"[ExecutionEnv] reward_mode={reward_mode} benchmark={reward_benchmark}")
        self.normalization_terms = load_compute_normalization_terms(
            symbol, data_dir, cst.Models.TRADES, n_lob_levels=10)
        self.cond_seq_size = (self.config.HYPER_PARAMETERS[cst.LearningHyperParameter.SEQ_SIZE]
                               - self.config.HYPER_PARAMETERS[cst.LearningHyperParameter.MASKED_SEQ_SIZE])
        self.seq_len = self.config.HYPER_PARAMETERS[cst.LearningHyperParameter.SEQ_SIZE]

        self.seed_days = seed_days or list_trading_days(data_dir, symbol)

        self._kernel_thread = None
        self._state_queue = None
        self._action_queue = None
        self._exec_agent = None
        self._episode_active = False

    def reset(self, t0=None, side=None, Q=None, seed_day=None, seed=None):
        if self._episode_active:
            raise RuntimeError("reset() called while a previous episode is still active (call step() to done=True first)")
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        # str(...): np.random.RandomState.choice() on a list of strings returns numpy.str_,
        # which pandas.Timestamp (used below) rejects even though it's a str subclass.
        seed_day = str(seed_day) if seed_day else str(self.rng.choice(self.seed_days))
        message_path, orderbook_path = coldstart._day_paths(self.data_dir, self.symbol, seed_day)
        messages, orderbook = coldstart.read_day(message_path, orderbook_path)

        if t0 is None:
            # Enough preceding real history for the conditioning window, far enough past the
            # open for the book to be representative (see MIN_SECONDS_AFTER_OPEN), and at
            # least EPISODE_SECONDS of trading day left -- varying across the session per
            # the spec.
            lo = max(float(messages["time"].iloc[self.seq_len + 10]),
                     DEFAULT_MARKET_OPEN + self.min_seconds_after_open)
            hi = float(messages["time"].iloc[-1]) - EPISODE_SECONDS - 5
            if lo >= hi:
                raise ValueError(f"{seed_day}: no valid seed window (lo={lo} >= hi={hi})")
            t0 = float(self.rng.uniform(lo, hi))
        side = str(side) if side else str(self.rng.choice(["BUY", "SELL"]))
        Q = Q or int(self.rng.randint(self.Q_range[0], self.Q_range[1]))

        reconstruct_start = _time.perf_counter()
        cs = coldstart.seed_episode(
            message_path, orderbook_path, t0, self.normalization_terms,
            cond_seq_size=self.cond_seq_size, seq_len=self.seq_len,
            messages=messages, orderbook=orderbook)  # reuse the frames already read above
        self._reconstruct_elapsed = _time.perf_counter() - reconstruct_start
        p_arrival = float((cs.lob_raw[-1, 0] + cs.lob_raw[-1, 2]) / 2.0)
        if len(cs.resting_orders) < THIN_BOOK_WARNING:
            print(f"[ExecutionEnv] WARNING: thin seeded book -- only {len(cs.resting_orders)} resting "
                  f"orders at t0={t0:.0f} on {seed_day}. Generation is prone to price runaway from a "
                  f"thin start; treat this episode's shortfall with suspicion (n_resting_orders is logged).")

        session_date = pd.Timestamp(seed_day)
        kernel_start = session_date + pd.Timedelta(seconds=t0)
        # Headroom past the nominal episode length: the agent's decision schedule is
        # anchored at its first *tradeable* wakeup (slightly after kernel start) and it
        # finalizes one interval after the last decision, so the kernel must outlive
        # EPISODE_SECONDS by more than a hair.
        kernel_stop = kernel_start + pd.Timedelta(seconds=EPISODE_SECONDS + 60)

        exchange = ExchangeAgent(
            id=EXCHANGE_AGENT_ID, name="EXCHANGE_AGENT", type="ExchangeAgent",
            mkt_open=kernel_start, mkt_close=kernel_stop, symbols=[self.symbol],
            log_orders=False, pipeline_delay=0, computation_delay=0,
            # book_freq=None disables the per-episode order-book log. At book_freq=0 the
            # exchange accumulates every book change and spends ~5s per episode writing a
            # multi-MB archive we never read -- ~15 min and a lot of disk across a
            # 150-episode run. The RL diagnostics come from the WorldAgent instead.
            stream_history=2_500_000, book_freq=None, wide_book=True,
            random_state=np.random.RandomState(seed=self.rng.randint(0, 2**31)))

        if self.world_mode == "replay":
            # Bounded by the kernel's own stop time: anything later is never replayed.
            stream = build_replay_stream(messages, t0, cs.resting_orders.keys(),
                                          horizon_seconds=EPISODE_SECONDS + 60)
            print(f"[ExecutionEnv] replay stream: {len(stream)} real messages from t0")
            world_agent = ReplayWorldAgent(
                id=WORLD_AGENT_ID, name="REPLAY_WORLD_AGENT", type="WorldAgent",
                symbol=self.symbol, date=str(session_date.date()),
                date_trading_days=cst.DATE_TRADING_DAYS, model=None, data_dir=self.data_dir,
                log_orders=False,
                random_state=np.random.RandomState(seed=self.rng.randint(0, 2**31)),
                normalization_terms=self.normalization_terms,
                replay_stream=stream, seed_lob_snapshots=cs.lob_raw,
                seed_price_anchor=cs.price_anchor)
        else:
            world_agent = RLWorldAgent(
            id=WORLD_AGENT_ID, name="WORLD_AGENT", type="WorldAgent", symbol=self.symbol,
            date=str(session_date.date()), date_trading_days=cst.DATE_TRADING_DAYS,
            model=self.model, data_dir=self.data_dir, cond_type=self.config.COND_TYPE,
            cond_seq_size=self.cond_seq_size, seq_len=self.seq_len,
            size_type_emb=self.config.HYPER_PARAMETERS[cst.LearningHyperParameter.SIZE_TYPE_EMB],
            log_orders=False, random_state=np.random.RandomState(seed=self.rng.randint(0, 2**31)),
            normalization_terms=self.normalization_terms, chosen_model="TRADES",
            gen_seq_size=self.config.HYPER_PARAMETERS[cst.LearningHyperParameter.MASKED_SEQ_SIZE],
            depth_noise=self.depth_noise,
            seed_placed_orders=cs.orders_raw, seed_lob_snapshots=cs.lob_raw,
            seed_price_anchor=cs.price_anchor, protected_agent_ids=(EXEC_AGENT_ID,))
        self._world_agent = world_agent

        # Seeded resting orders must belong to the WorldAgent, exactly as in a normal
        # replay-based run -- see seed_exchange_book's docstring.
        coldstart.seed_exchange_book(exchange, self.symbol, cs.resting_orders, session_date,
                                      owner_agent_id=world_agent.id)

        state_queue = queue.Queue()
        action_queue = queue.Queue()
        self._state_queue = state_queue
        self._action_queue = action_queue
        exec_agent = RLExecutionAgent(
            id=EXEC_AGENT_ID, name="RL_EXECUTION_AGENT", type="ExecutionAgent", symbol=self.symbol,
            direction=side, quantity=Q, p_arrival=p_arrival, start_time=kernel_start,
            state_queue=state_queue, action_queue=action_queue,
            random_state=np.random.RandomState(seed=self.rng.randint(0, 2**31)),
            reward_mode=self.reward_mode, reward_benchmark=self.reward_benchmark)
        self._exec_agent = exec_agent

        kernel = Kernel("RL Execution Kernel", random_state=np.random.RandomState(seed=self.rng.randint(0, 2**31)))
        log_dir = f"rl_execution_{self.symbol}_{seed_day}_{int(t0)}_{int(_time.time() * 1000) % 1_000_000}"

        def _run():
            try:
                kernel.runner(agents=[exchange, world_agent, exec_agent],
                              startTime=kernel_start, stopTime=kernel_stop,
                              defaultComputationDelay=1, log_dir=log_dir,
                              skip_log=True, run_telemetry=False)
            except Exception as e:
                # Report onto THIS episode's queue, captured in the closure -- never
                # self._state_queue, which by the time a straggler thread fails may
                # already have been replaced by the next episode's queue, poisoning a
                # healthy episode with a dead one's error (observed exactly that way).
                # Without this the failure would instead hang reset()/step() forever on
                # a get() that never returns, which is far worse to debug remotely.
                state_queue.put(("error", e, {}))

        self._episode_info = {
            "seed_day": seed_day, "t0": t0, "side": side, "Q": Q, "p_arrival": p_arrival,
            "sampling_type": self.sampling_type, "depth_noise": self.depth_noise,
            "ddim_nsteps": self.ddim_nsteps, "checkpoint": str(self.checkpoint_path),
            "cond_stats": cs.cond_stats, "n_resting_orders": len(cs.resting_orders),
        }
        self._episode_active = True
        self._simulate_start = _time.perf_counter()
        self._kernel_thread = threading.Thread(target=_run, daemon=True)
        self._kernel_thread.start()

        msg = self._state_queue.get()
        obs, _, done, info = self._unpack(msg)
        if done:
            # The episode terminated before offering a single decision point (e.g. the
            # agent never became tradeable and RLExecutionAgent.kernelTerminating's
            # safety net fired). Fail loudly rather than hand back obs=None, which would
            # blow up later inside the caller's policy with a far less obvious error.
            self._episode_active = False
            raise RuntimeError(
                f"episode ended before the first decision point (info={info}) -- "
                "check the kernel start/stop window and the agent's wakeup schedule")
        return obs, info

    def step(self, action):
        if not self._episode_active:
            raise RuntimeError("step() called with no active episode -- call reset() first")
        self._action_queue.put(action)
        msg = self._state_queue.get()
        obs, reward, done, info = self._unpack(msg)
        if done:
            # Diagnostics are read off the WorldAgent below, so collect them before the
            # thread is allowed to be considered finished.
            info.update(self._collect_world_agent_diagnostics())
            # Wait for the episode's Kernel to fully unwind (kernelStopping /
            # kernelTerminating / log flushing) BEFORE this episode is declared over.
            # The agent signals "done" from inside the still-running kernel, so without a
            # blocking join the caller starts the next reset() while the previous Kernel
            # is still finishing -- two episodes' simulations overlapping, competing for
            # the GPU, and interleaving their output (observed exactly that way).
            self._kernel_thread.join(timeout=300)
            if self._kernel_thread.is_alive():
                print("[ExecutionEnv] WARNING: kernel thread still running 300s after episode end")
            self._episode_active = False
            info["wall_clock_reconstruct_s"] = self._reconstruct_elapsed
            info["wall_clock_simulate_s"] = _time.perf_counter() - self._simulate_start
            info["wall_clock_total_s"] = info["wall_clock_reconstruct_s"] + info["wall_clock_simulate_s"]
        return obs, reward, done, info

    def _collect_world_agent_diagnostics(self):
        wa = self._world_agent
        counts = wa.decoded_type_counts
        total = sum(counts.values())
        flow_mix = {k: v / total for k, v in counts.items()} if total else {}

        n_exec = len(wa._exec_outcomes)
        if n_exec:
            execution_rate = sum(wa._exec_outcomes) / n_exec
        else:
            # Replay: nothing decodes, so _exec_outcomes stays empty and this would log
            # None -- dropping the very comparison the replay arm exists to make, since
            # the generative market's 17-18% execution rate only means something against
            # the real market's. Executions per new order is the same quantity, and in a
            # LOBSTER stream it is definitional rather than estimated.
            placed = counts.get(1, 0)
            execution_rate = (counts.get(4, 0) / placed) if placed else None

        mids = set()
        for snap in wa.lob_snapshots:
            ask, bid = snap[0], snap[2]
            if 0 < ask < 9_000_000_000 and 0 < bid < 9_000_000_000:
                mids.add(round((ask + bid) / 2.0, 2))

        return {
            "flow_mix": flow_mix,
            "execution_rate": execution_rate,
            "unique_mid_count": len(mids),
        }

    def _unpack(self, msg):
        kind, payload, info = msg
        if kind == "error":
            self._episode_active = False
            raise RuntimeError("episode failed inside the Kernel thread") from payload
        info = {**self._episode_info, **info}
        if kind == "obs":
            # In per-step reward mode the agent settles the previous decision's fills and
            # sends the amount alongside this observation; in terminal mode it is always 0.
            return payload, float(info.get("reward") or 0.0), False, info
        elif kind == "done":
            return None, payload, True, info
        raise ValueError(f"unexpected message kind: {kind}")
