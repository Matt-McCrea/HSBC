"""The property that matters: pairing must recover a policy effect that the unpaired
comparison cannot see, because that is the exact situation the evaluation is in --
shortfall correlates -0.837 with per-episode drift, so between-seed variance swamps
the policy difference."""

import json
import os
import sys

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "ABIDES"))

from rl_execution.compare_policies import load, paired, report

P_ARRIVAL = 365150.0
TRUE_EFFECT_BPS = 3.0     # the policy is genuinely 3 bps worse
DRIFT_SD_BPS = 40.0       # and drift noise is >10x that, as in the real logs


def _write_log(tmp_path, n_seeds=18, seed=0):
    rng = np.random.RandomState(seed)
    path = tmp_path / "eval.jsonl"
    with open(path, "w") as f:
        for i in range(n_seeds):
            drift = rng.normal(0, DRIFT_SD_BPS)          # common to both policies
            for name, effect in (("twap", 0.0), ("qlearning", TRUE_EFFECT_BPS)):
                bps = drift + effect + rng.normal(0, 1.0)
                f.write(json.dumps({
                    "seed_day": "2015-01-2%d" % (i % 10), "t0": 40000.0 + i,
                    "side": "SELL" if i % 2 else "BUY", "Q": 2000 + i,
                    "policy_name": name, "p_arrival": P_ARRIVAL,
                    "shortfall": bps / 10_000.0 * P_ARRIVAL, "shortfall_bps": bps,
                }) + "\n")
    return str(path)


def test_pairing_recovers_an_effect_the_unpaired_view_misses(tmp_path):
    df = load([_write_log(tmp_path)])
    res = paired(df, baseline="twap")
    row = res[res["policy"] == "qlearning"].iloc[0]

    assert row["p"] < 0.001, "paired test must detect the known effect"
    assert abs(row["mean_diff"] - TRUE_EFFECT_BPS) < 1.0, "and estimate it accurately"
    # The unpaired stderr is what the evaluation summary prints; it is far too wide to
    # resolve a 3 bps effect sitting under 40 bps of drift.
    assert row["se_unpaired"] > 3 * TRUE_EFFECT_BPS
    assert row["variance_reduction"] > 5, "pairing should shrink the stderr several-fold"


def test_sign_convention_negative_means_the_policy_won(tmp_path):
    df = load([_write_log(tmp_path)])
    res = paired(df, baseline="twap")
    row = res[res["policy"] == "qlearning"].iloc[0]
    assert row["mean_diff"] > 0, "this synthetic policy is worse, so the diff is positive"
    assert row["wins"] < 9, "and it should lose on most seeds"


def test_only_seeds_present_for_both_policies_are_paired(tmp_path):
    """A truncated arm must not silently shift the baseline it is compared against."""
    path = _write_log(tmp_path)
    with open(path, "a") as f:
        f.write(json.dumps({
            "seed_day": "2015-01-99", "t0": 99999.0, "side": "SELL", "Q": 5000,
            "policy_name": "twap", "p_arrival": P_ARRIVAL,
            "shortfall": 0.0, "shortfall_bps": 500.0,   # extreme, baseline-only seed
        }) + "\n")
    res = paired(load([path]), baseline="twap")
    row = res[res["policy"] == "qlearning"].iloc[0]
    assert row["n"] == 18, "the unmatched baseline-only seed must be dropped"
    assert abs(row["mean_diff"] - TRUE_EFFECT_BPS) < 1.0


def test_missing_baseline_is_an_explicit_error(tmp_path):
    df = load([_write_log(tmp_path)])
    with pytest.raises(SystemExit):
        paired(df, baseline="not_a_policy")


def test_report_runs_and_names_every_policy(tmp_path):
    text, res = report(load([_write_log(tmp_path)]), baseline="twap")
    assert "PAIRED vs twap" in text and "qlearning" in text
    assert not res.empty


def _mixed_log(tmp_path, n_seeds, seed=1):
    """One policy clearly worse, one indistinguishable -- the case that produced a
    self-contradicting summary (a significant policy reported as 'no separation')."""
    rng = np.random.RandomState(seed)
    path = tmp_path / "mixed.jsonl"
    with open(path, "w") as f:
        for i in range(n_seeds):
            drift = rng.normal(0, DRIFT_SD_BPS)
            for name, effect in (("twap", 0.0), ("ac_k2", 6.0), ("qlearning", 0.05)):
                bps = drift + effect + rng.normal(0, 0.5)
                f.write(json.dumps({
                    "seed_day": "2015-01-2%d" % (i % 10), "t0": 40000.0 + i,
                    "side": "SELL", "Q": 2000 + i, "policy_name": name,
                    "p_arrival": P_ARRIVAL, "shortfall": bps / 10_000.0 * P_ARRIVAL,
                    "shortfall_bps": bps,
                }) + "\n")
    return str(path)


def test_a_significant_policy_is_never_reported_as_no_separation(tmp_path):
    text, res = report(load([_mixed_log(tmp_path, 18)]), baseline="twap")
    assert res[res["policy"] == "ac_k2"].iloc[0]["p"] < 0.05
    assert "No policy separates" not in text, "contradicts the table above it"
    assert "Significantly worse than twap: ac_k2" in text


def test_small_paired_samples_are_flagged(tmp_path):
    text, _ = report(load([_mixed_log(tmp_path, 4)]), baseline="twap")
    assert "CAUTION" in text and "n=4" in text


def test_no_caution_when_the_sample_is_adequate(tmp_path):
    text, _ = report(load([_mixed_log(tmp_path, 18)]), baseline="twap")
    assert "CAUTION" not in text
