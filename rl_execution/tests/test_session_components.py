"""Tests for the self-steering session: the AC baseline, calibration, preflight,
and above all the FALLBACK paths.

The fallbacks are the code most likely to execute unattended at 3am and least
likely to have ever been run, so they get the same scrutiny as the happy path. A
fallback that itself crashes turns a degraded run into a dead booking.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rl_execution.baselines import ACSchedulePolicy, ac_target_inventory
from rl_execution.calibrate import (DECISION_SECONDS, FALLBACK_LAM, calibrate,
                                     estimate_eta, estimate_sigma, lam_from_kappa)
from rl_execution.execution_agent import N_DECISIONS
from rl_execution.preflight import check_episode, run_checks
from rl_execution.qlearning import TWAPPolicy

P_ARRIVAL = 340000.0
Q = 3000


def _obs(t_rem, inv):
    return {"time_remaining_frac": t_rem, "inventory_remaining_frac": inv,
            "spread_bucket": 0, "vol_bucket": 0, "ofi_bucket": 0}


# ---------------------------------------------------------------- AC baseline

def test_ac_at_zero_kappa_is_exactly_twap():
    """Almgren-Chriss with linear impact makes uniform trading the risk-neutral
    optimum, so TWAP must be the kappa=0 member of the family -- not a separate
    heuristic. If this ever diverges, the framing of TWAP as a literature baseline
    stops being true."""
    ac, twap = ACSchedulePolicy(kappa=0.0), TWAPPolicy()
    inv = 1.0
    for step in range(N_DECISIONS):
        obs = _obs((N_DECISIONS - step) / N_DECISIONS, inv)
        assert ac.select_action(obs) == twap.select_action(obs)
        inv = ac.target_after_this_step(obs["time_remaining_frac"])


def test_ac_target_is_monotone_and_hits_the_endpoints():
    for kappa in (0.0, 1.0, 2.0, 5.0):
        vals = [ac_target_inventory(u / 10, kappa) for u in range(11)]
        assert vals[0] == pytest.approx(0.0), "no time left means no inventory left"
        assert vals[-1] == pytest.approx(1.0), "full window left means full inventory"
        assert all(a <= b + 1e-12 for a, b in zip(vals, vals[1:])), "trajectory must be monotone"


@pytest.mark.parametrize("u", [0.2, 0.5, 0.8])
def test_higher_kappa_front_loads(u):
    """AC's central qualitative prediction: more risk aversion means less inventory
    still held at any interior point."""
    held = [ac_target_inventory(u, k) for k in (0.0, 1.0, 2.0, 4.0)]
    assert all(a > b for a, b in zip(held, held[1:])), f"not front-loading at u={u}: {held}"


def test_ac_liquidates_faster_than_twap_in_a_simulated_run():
    inv_ac = inv_twap = 1.0
    ac = ACSchedulePolicy(kappa=3.0)
    for step in range(N_DECISIONS // 2):
        t_rem = (N_DECISIONS - step) / N_DECISIONS
        inv_ac = ac.target_after_this_step(t_rem)
        inv_twap -= 1.0 / N_DECISIONS
    assert inv_ac < inv_twap


# ---------------------------------------------------------------- calibration

def _synthetic_records(eta=8.0, eps=25.0, sigma=120.0, n_eps=12, seed=0, drift_bps=0.0):
    rng = np.random.RandomState(seed)
    recs = []
    for _ in range(n_eps):
        mid, traj = P_ARRIVAL, []
        for t in range(N_DECISIONS):
            qty = float(rng.randint(100, 600))
            slip = eps + eta * (qty / DECISION_SECONDS) + rng.normal(0, 3)
            traj.append({"t_rem": (N_DECISIONS - t) / N_DECISIONS, "inv_rem": 1 - t / N_DECISIONS,
                         "spread": 0, "vol": 0, "ofi": 0, "a": 1, "r": 0.0,
                         "done": t == N_DECISIONS - 1, "mid": mid, "fills": [[qty, mid - slip]]})
            mid += rng.normal(0, sigma)
        recs.append({"Q": Q, "side": "SELL", "p_arrival": P_ARRIVAL,
                     "drift_bps": float(rng.normal(drift_bps, 8)), "trajectory": traj})
    return recs


def test_calibration_recovers_known_impact_and_volatility():
    c = calibrate(_synthetic_records(eta=8.0, eps=25.0, sigma=120.0))
    assert c["calibrated"]
    assert c["eta"] == pytest.approx(8.0, rel=0.15)
    assert c["eps"] == pytest.approx(25.0, rel=0.30)
    assert c["sigma"] == pytest.approx(120.0, rel=0.20)
    assert c["lam"] > 0


def test_calibration_falls_back_when_impact_regression_is_degenerate():
    """Constant trade size gives the regression no variation in rate, so eta is
    unidentifiable. It must degrade to the documented constant, not emit a garbage
    penalty that would silently define the objective for 26 hours."""
    recs = _synthetic_records()
    for rec in recs:  # flatten trade sizes -> zero rate variance
        for step in rec["trajectory"]:
            step["fills"] = [[300.0, P_ARRIVAL - 50.0]]
    c = calibrate(recs)
    assert not c["calibrated"]
    assert c["lam"] == FALLBACK_LAM
    assert len(c["kappas"]) == 2, "fallback should span mild and aggressive kappa"
    assert c["warnings"]


def test_calibration_flags_a_drifting_simulator():
    """Almgren-Chriss assume a martingale. TRADES has documented directional drift,
    and a seller is flattered by an upward-drifting market, so a significant drift
    must be surfaced rather than quietly absorbed."""
    c = calibrate(_synthetic_records(drift_bps=40.0, n_eps=30))
    assert any("martingale" in w for w in c["warnings"]), c["warnings"]


def test_calibration_survives_an_empty_log():
    c = calibrate([])
    assert not c["calibrated"] and c["lam"] == FALLBACK_LAM


def test_lam_scales_with_risk_aversion():
    a = lam_from_kappa(1.0, eta=8.0, mean_q=Q)
    b = lam_from_kappa(2.0, eta=8.0, mean_q=Q)
    assert b == pytest.approx(4 * a), "lam should go as kappa^2"


# ---------------------------------------------------------------- preflight

def _good_episode():
    traj = [{"t_rem": (N_DECISIONS - t) / N_DECISIONS, "inv_rem": 1 - t / N_DECISIONS,
             "spread": 0, "vol": 0, "ofi": 0, "a": 1,
             "r": (-12.0 if t == N_DECISIONS - 1 else 0.0),
             "done": t == N_DECISIONS - 1, "mid": P_ARRIVAL, "fills": []}
            for t in range(N_DECISIONS)]
    return {"Q": Q, "side": "SELL", "p_arrival": P_ARRIVAL, "shortfall": 12.0,
            "rem_quantity": 0, "fills": [[Q, P_ARRIVAL - 10]], "trajectory": traj,
            "cond_z": {"price": [-1.5, 1.5, 0.0, 255]}, "error": None}


def test_preflight_passes_a_healthy_episode():
    ok, report = run_checks([_good_episode()])
    assert ok, "\n".join(report)


def test_preflight_catches_incomplete_liquidation():
    """The failure that matters most: unsold inventory SHRINKS reported shortfall, so
    it reads as good performance instead of a broken terminal sweep."""
    bad = _good_episode()
    bad["rem_quantity"] = 450
    failures = check_episode(bad)
    assert any("terminal liquidation incomplete" in f for f in failures)


def test_preflight_catches_over_execution():
    bad = _good_episode()
    bad["fills"] = [[Q * 1.4, P_ARRIVAL]]
    assert any("over-execution" in f for f in check_episode(bad))


def test_preflight_catches_broken_reward_attribution():
    bad = _good_episode()
    bad["trajectory"][-1]["r"] = -999.0  # no longer equals -shortfall
    assert any("attribution" in f for f in check_episode(bad))


def test_preflight_catches_out_of_distribution_conditioning():
    bad = _good_episode()
    bad["cond_z"] = {"price": [-40.0, 2.0, 0.0, 255]}
    assert any("out of distribution" in f for f in check_episode(bad))


def test_preflight_fails_loudly_on_an_empty_run():
    ok, report = run_checks([])
    assert not ok and "no episodes" in report[0]


def test_preflight_tolerates_terminal_reward_mode():
    """In terminal mode the per-step sum identity does not apply, so the check must
    not fire -- otherwise the preflight fallback would itself fail preflight."""
    rec = _good_episode()
    rec["trajectory"][-1]["r"] = -12.0
    for step in rec["trajectory"][:-1]:
        step["r"] = 0.0
    assert check_episode(rec, reward_mode="terminal") == []
