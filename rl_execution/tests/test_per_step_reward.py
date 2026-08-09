"""Tests for per-step reward attribution.

The whole claim of per-step mode is that it is a RE-ATTRIBUTION of the same
total, not a different objective: summing the per-step series must reproduce
exactly the negative of the episode's implementation shortfall. If that
identity fails, the agent is being scored on something other than what gets
reported, and every comparison against TWAP becomes meaningless.

These exercise the pure reward functions, so they need no ABIDES kernel.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rl_execution.execution_agent import N_DECISIONS, per_step_rewards, signed_cost

Q = 2000
P_ARRIVAL = 340000.0


def _shortfall(fills, direction, p_arrival=P_ARRIVAL, quantity=Q):
    """Mirrors RLExecutionAgent._compute_shortfall."""
    return sum(signed_cost(q, p, p_arrival, direction) for q, p in fills) / quantity


@pytest.mark.parametrize("direction", ["SELL", "BUY"])
def test_per_step_rewards_sum_to_negative_shortfall(direction):
    fills_by_step = {
        0: [(400, P_ARRIVAL + 50)],
        3: [(600, P_ARRIVAL - 20), (200, P_ARRIVAL - 35)],
        7: [(300, P_ARRIVAL + 10)],
        9: [(500, P_ARRIVAL - 120)],  # terminal sweep, walks the book
    }
    all_fills = [f for fills in fills_by_step.values() for f in fills]

    rewards = per_step_rewards(fills_by_step, [P_ARRIVAL] * (N_DECISIONS + 1),
                               Q, direction, N_DECISIONS + 1)
    assert sum(rewards) == pytest.approx(-_shortfall(all_fills, direction))


@pytest.mark.parametrize("direction", ["SELL", "BUY"])
def test_sign_convention_matches_shortfall(direction):
    """Positive shortfall is bad, so reward must be negative for an adverse fill."""
    adverse = P_ARRIVAL - 100 if direction == "SELL" else P_ARRIVAL + 100
    favourable = P_ARRIVAL + 100 if direction == "SELL" else P_ARRIVAL - 100

    r_adverse = per_step_rewards({0: [(Q, adverse)]}, [P_ARRIVAL] * 2, Q, direction, 2)
    r_favourable = per_step_rewards({0: [(Q, favourable)]}, [P_ARRIVAL] * 2, Q, direction, 2)
    assert r_adverse[0] < 0 < r_favourable[0]


def test_reward_lands_on_the_step_that_placed_the_order():
    """A fill must be credited to the decision whose child order produced it, not to
    whichever step happened to be current when it arrived -- otherwise a passive order
    filling several steps later blames the wrong action."""
    rewards = per_step_rewards({2: [(500, P_ARRIVAL - 80)]},
                               [P_ARRIVAL] * (N_DECISIONS + 1), Q, "SELL", N_DECISIONS + 1)
    assert rewards[2] != 0.0
    assert all(r == 0.0 for i, r in enumerate(rewards) if i != 2)


def test_prevailing_benchmark_removes_a_pure_drift_episode():
    """If the market simply drifts and every fill happens at the prevailing mid, the
    agent traded neither well nor badly and a drift-free reward should be ~0 -- while
    the arrival-referenced reward reports the drift, as true shortfall should."""
    drifting_mids = [P_ARRIVAL + 40 * t for t in range(N_DECISIONS + 1)]
    fills_by_step = {t: [(200, drifting_mids[t])] for t in range(N_DECISIONS)}

    at_prevailing = per_step_rewards(fills_by_step, drifting_mids, Q, "SELL", N_DECISIONS + 1)
    at_arrival = per_step_rewards(fills_by_step, [P_ARRIVAL] * (N_DECISIONS + 1),
                                  Q, "SELL", N_DECISIONS + 1)

    assert sum(at_prevailing) == pytest.approx(0.0), "drift leaked into the drift-free reward"
    assert sum(at_arrival) > 0, "arrival-referenced reward should capture the favourable drift"


def test_unfilled_steps_earn_nothing():
    rewards = per_step_rewards({}, [P_ARRIVAL] * (N_DECISIONS + 1), Q, "SELL", N_DECISIONS + 1)
    assert rewards == [0.0] * (N_DECISIONS + 1)


def test_out_of_range_steps_are_ignored_not_crashed():
    """Defensive: a fill attributed to a step outside the episode (a late arrival after
    the terminal sweep) must not index out of bounds."""
    rewards = per_step_rewards({999: [(100, P_ARRIVAL)]}, [P_ARRIVAL] * 2, Q, "SELL", 2)
    assert rewards == [0.0, 0.0]


def test_refit_can_switch_benchmark_offline():
    """A 26-hour run must not lock in the arrival-vs-prevailing choice. Given per-step
    mid and fills in the log, a refit must be able to rebuild rewards either way."""
    from rl_execution.refit_qtable import _recomputed_reward

    rec = {"p_arrival": P_ARRIVAL, "Q": Q, "side": "SELL"}
    step = {"mid": P_ARRIVAL + 200.0, "fills": [[500, P_ARRIVAL + 200.0]]}

    # sold exactly at the prevailing mid -> no execution edge either way
    assert _recomputed_reward(step, rec, "prevailing") == pytest.approx(0.0)
    # ... but against the arrival mid that same fill is favourable, since price rose
    assert _recomputed_reward(step, rec, "arrival") > 0

    # logs predating per-step mid/fills must fall back, not crash or fabricate
    assert _recomputed_reward({"mid": None, "fills": []}, rec, "prevailing") is None
    assert _recomputed_reward({"mid": None, "fills": [[100, P_ARRIVAL]]}, rec, "prevailing") is None
