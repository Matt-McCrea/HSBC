"""Exercise the train / evaluate / benchmark logging paths against a fake
environment, then refit a Q-table from what they wrote.

These loops only ever run on the GPU box, where a bug costs a multi-hour run
(evaluation is chained after ~50h of training, so a broken log call there
throws that away). Everything except ExecutionEnv itself is plain Python, so
substituting a fake env exercises the real logging code locally, for free.
"""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rl_execution.benchmark import run_benchmark
from rl_execution.evaluate import evaluate_policy
from rl_execution.logging_utils import JsonlLogger, read_episodes
from rl_execution.qlearning import N_ACTIONS, QLearningPolicy, TWAPPolicy
from rl_execution.refit_qtable import penalty_sweep, refit
from rl_execution.train import train

N_DECISIONS = 10


# Participation multipliers from execution_agent.ACTION_LEVELS, as a fraction of the
# Q/10 base slice: inventory must respond to the action, or an inventory penalty has
# nothing to differentiate and the fixture silently cannot test it.
PARTICIPATION = [0.5, 1.0, 1.0, 1.5, 2.0]

# Aggression costs money (market impact) but sheds inventory faster. That tension IS the
# Almgren-Chriss trade-off, so a fixture without it cannot exhibit a cost/risk frontier:
# with cost independent of action, nothing distinguishes the actions on cost grounds and
# any apparent lambda response is noise. Sized well above IMPACT_NOISE so the signal is
# resolvable at the few-visits-per-entry counts these tests run at.
IMPACT_PER_PARTICIPATION = 400.0
IMPACT_NOISE = 60.0


class FakeEnv:
    """Mimics ExecutionEnv's reset/step contract and info payload."""

    def __init__(self, n_decisions=N_DECISIONS, seed=0):
        self.n = n_decisions
        self.rng = np.random.RandomState(seed)
        self.p_arrival = 340000.0
        self._i = 0
        self._inv = 1.0

    def _obs(self):
        return {
            "time_remaining_frac": (self.n - self._i) / self.n,
            "inventory_remaining_frac": max(0.0, self._inv),
            "spread_bucket": int(self.rng.randint(0, 4)),
            "vol_bucket": int(self.rng.randint(0, 3)),
            "ofi_bucket": int(self.rng.randint(0, 3)),
        }

    def _info(self, shortfall=None, p_final=None):
        return {
            "seed_day": "2015-01-30", "t0": 40000.0, "side": "SELL", "Q": 2000,
            "sampling_type": "DDIM", "depth_noise": 0.3, "ddim_nsteps": 10,
            "checkpoint": "fake.ckpt", "p_arrival": self.p_arrival, "p_final": p_final,
            "shortfall": shortfall, "n_resting_orders": 3700,
            "wall_clock_total_s": 600.0, "wall_clock_reconstruct_s": 0.5,
            "wall_clock_simulate_s": 599.5, "fills": [[2000, self.p_arrival - 30]],
            "cond_stats": {"price": [-1.0, 1.0, 0.0, 255]}, "flow_mix": {"1": 0.5},
            "execution_rate": 0.06, "unique_mid_count": 9,
        }

    def reset(self, **kwargs):
        self._i = 0
        self._inv = 1.0
        self._participation = []
        return self._obs(), self._info()

    def step(self, action):
        self._i += 1
        self._participation.append(PARTICIPATION[int(action)])
        self._inv = max(0.0, self._inv - PARTICIPATION[int(action)] / self.n)
        done = self._i >= self.n
        if done:
            self._inv = 0.0  # terminal sweep completes the parent order
        if not done:
            return self._obs(), 0.0, False, self._info()
        impact = IMPACT_PER_PARTICIPATION * float(np.mean(self._participation))
        shortfall = impact + float(self.rng.normal(0, IMPACT_NOISE))
        p_final = self.p_arrival * (1.0 + self.rng.normal(0.0005, 0.001))
        reward = -shortfall
        return self._obs(), reward, True, self._info(shortfall=shortfall, p_final=p_final)


def _read_one(path):
    recs = read_episodes(path)
    assert recs, f"no episodes written to {path}"
    return recs


def test_train_logs_trajectory_and_drift(tmp_path):
    env = FakeEnv()
    policy = QLearningPolicy(random_state=np.random.RandomState(0))
    log = tmp_path / "train.jsonl"
    train(env, policy, n_episodes=3, checkpoint_path=str(tmp_path / "q.npz"),
          checkpoint_every=99, out_path=str(log), side="SELL")

    recs = _read_one(log)
    assert len(recs) == 3
    for r in recs:
        assert r["trajectory"] and len(r["trajectory"]) == N_DECISIONS
        assert r["p_final"] is not None
        assert r["drift_bps"] is not None
        assert r["shortfall_bps"] is not None
        assert r["run_id"]
        last = r["trajectory"][-1]
        assert last["done"] is True
        assert 0 <= last["a"] < N_ACTIONS
        # only the terminal step carries reward
        assert all(s["r"] == 0.0 for s in r["trajectory"][:-1])
        assert last["r"] != 0.0


def test_evaluate_logs_trajectory_and_drift(tmp_path):
    """The path that runs after ~50h of training -- a break here is the expensive one."""
    env = FakeEnv(seed=1)
    log = tmp_path / "eval.jsonl"
    logger = JsonlLogger(str(log))
    seeds = [{"seed_day": "2015-01-30", "t0": 40000.0 + i, "side": "SELL", "Q": 2000}
             for i in range(3)]
    shortfalls, wallclock, n_run = evaluate_policy(env, TWAPPolicy(), seeds, logger,
                                                    "eval_test", "twap")
    assert n_run == 3 and len(shortfalls) == 3 and wallclock >= 0
    for r in _read_one(log):
        assert len(r["trajectory"]) == N_DECISIONS
        assert r["p_final"] is not None and r["drift_bps"] is not None
        assert r["policy_name"] == "twap"


def test_evaluate_respects_wall_clock_budget(tmp_path):
    env = FakeEnv(seed=2)
    logger = JsonlLogger(str(tmp_path / "eval2.jsonl"))
    seeds = [{"seed_day": "2015-01-30", "t0": 40000.0 + i, "side": "SELL", "Q": 2000}
             for i in range(5)]
    # a zero budget must still run exactly one episode, never zero and never all five
    _, _, n_run = evaluate_policy(env, TWAPPolicy(), seeds, logger, "eval_budget", "twap",
                                   wall_clock_budget=1e-9)
    assert n_run == 1


def test_benchmark_logs_trajectory(tmp_path):
    env = FakeEnv(seed=3)
    log = tmp_path / "bench.jsonl"
    run_benchmark(env, n_episodes=2, out_path=str(log))
    for r in _read_one(log):
        assert len(r["trajectory"]) == N_DECISIONS


def test_refit_reproduces_a_policy_offline(tmp_path):
    """The payoff: fit a Q-table from logged trajectories with no simulation."""
    env = FakeEnv(seed=4)
    policy = QLearningPolicy(random_state=np.random.RandomState(0))
    log = tmp_path / "train.jsonl"
    train(env, policy, n_episodes=12, checkpoint_path=str(tmp_path / "q.npz"),
          checkpoint_every=99, out_path=str(log), side="SELL")

    recs = read_episodes(str(log))
    q, visits, n_eps, n_steps, skipped = refit(recs, alpha_mode="visit-count")
    assert n_eps == 12 and n_steps == 12 * N_DECISIONS and skipped == 0
    assert visits.sum() == n_steps
    assert np.isfinite(q).all()
    assert (q != 0).any(), "refit produced an all-zero table"

    # drift adjustment must change the values without breaking the fit
    q_adj, *_ = refit(recs, alpha_mode="visit-count", drift_adjust=True)
    assert np.isfinite(q_adj).all()
    assert not np.allclose(q, q_adj), "drift adjustment had no effect on the fitted values"


def test_inventory_penalty_pushes_toward_faster_liquidation(tmp_path):
    """The point of the Almgren-Chriss risk term: charging for held inventory should
    move the greedy policy toward more aggressive actions. Verified on the SAME logged
    episodes at several lambdas -- which is exactly the free offline screen the sweep
    exists to provide."""
    env = FakeEnv(seed=5)
    policy = QLearningPolicy(random_state=np.random.RandomState(0), alpha_mode="visit-count")
    log = tmp_path / "train.jsonl"
    train(env, policy, n_episodes=60, checkpoint_path=str(tmp_path / "q.npz"),
          checkpoint_every=999, out_path=str(log), side="SELL")

    recs = read_episodes(str(log))
    rows, observed_risk = penalty_sweep(recs, [0.0, 100.0, 500.0], alpha_mode="visit-count")
    assert len(rows) == 3
    assert all(np.isfinite(r["mean_action"]) for r in rows)
    # a penalty must move the policy away from where lambda=0 put it
    assert rows[-1]["changed_vs_lam0"][0] > 0, "inventory penalty had no effect on the policy"
    # ... and specifically toward aggression: with impact making aggression costly,
    # lambda=0 should favour passive execution and a large lambda should override that.
    assert rows[-1]["mean_action"] > rows[0]["mean_action"] + 0.5, (
        f"penalising held inventory should favour faster liquidation; "
        f"got {rows[0]['mean_action']:.2f} at lambda=0 vs {rows[-1]['mean_action']:.2f} at lambda=500")
    assert observed_risk and all(r >= 0 for r in observed_risk)


@pytest.mark.parametrize("lam", [0.0, 500.0])
def test_inventory_penalty_is_not_baked_into_the_log(tmp_path, lam):
    """Shaping must happen at fit time only -- if a live run's penalty leaked into the
    logged rewards, lambda would be frozen into the data and could not be re-swept.

    Comparing two runs' reward sequences would NOT show this: a shaped agent takes
    different actions, and with impact in the fixture that legitimately changes the
    shortfall. The invariant that isolates a leak is that the logged terminal reward
    still equals the true (unshaped) reward implied by the reported shortfall, and
    that intermediate steps stay at exactly zero.
    """
    log = tmp_path / f"lam{lam}.jsonl"
    train(FakeEnv(seed=9), QLearningPolicy(random_state=np.random.RandomState(0)),
          n_episodes=4, checkpoint_path=str(tmp_path / f"q{lam}.npz"),
          checkpoint_every=999, out_path=str(log), side="SELL",
          inventory_penalty_lambda=lam)

    for rec in read_episodes(str(log)):
        traj = rec["trajectory"]
        assert traj[-1]["r"] == pytest.approx(-rec["shortfall"]), \
            "terminal reward no longer matches true shortfall -- shaping leaked into the log"
        assert all(s["r"] == 0.0 for s in traj[:-1]), \
            "intermediate rewards are non-zero -- the per-step penalty leaked into the log"


def test_refit_handles_logs_without_trajectories(tmp_path):
    """Runs predating trajectory logging must be reported, not crash."""
    log = tmp_path / "old.jsonl"
    with open(log, "w") as f:
        f.write(json.dumps({"run_name": "old", "shortfall": -100.0, "p_arrival": 340000.0}) + "\n")
    q, visits, n_eps, n_steps, skipped = refit(read_episodes(str(log)))
    assert n_eps == 0 and n_steps == 0 and skipped == 1
    assert not q.any()
