"""Both faults reproduced from the real logs: a retention ratio built from two
insignificant, opposite-signed slopes, and a functional-form verdict resting on an
R^2 gap of ~0.005 that flips between logs."""
import os, sys
_ROOT = "/Users/Matthew/HSBC"
sys.path.insert(0, _ROOT); sys.path.insert(0, os.path.join(_ROOT, "ABIDES"))
from rl_execution.impact import format_report


def _analysis(g1, t1, g5, t5, sqrt_r2, lin_r2):
    return {
        "n_episodes": 54,
        "temporary": {"slope": 4.751, "t": 7.86, "r2": lin_r2, "intercept": -66.01, "n": 488},
        "permanent": {1: {"slope": g1, "t": t1, "r2": 0.001, "n": 434},
                      5: {"slope": g5, "t": t5, "r2": 0.0008, "n": 241}},
        "sqrt_law": {"slope": 6.524, "t": 8.045, "r2": sqrt_r2},
    }


def test_retention_suppressed_when_slopes_are_noise():
    # The real eval log: gamma(1)=-0.019 (t=-0.70), gamma(5)=+0.020 (t=0.43) -> "-105%".
    out = format_report(_analysis(-0.01896, -0.6987, 0.01992, 0.4278, 0.1175, 0.1128))
    assert "retained at" not in out, "a ratio of two zeros must not be printed as decay"
    assert "retention not reported" in out


def test_retention_reported_when_both_horizons_are_real():
    out = format_report(_analysis(0.10, 4.5, 0.08, 3.1, 0.1175, 0.1128))
    assert "retained at h=5 vs h=1: 80%" in out


def test_functional_form_not_declared_on_a_hair(): 
    out = format_report(_analysis(0.1, 4.5, 0.08, 3.1, 0.1175, 0.1128))   # gap 0.005
    assert "INDISTINGUISHABLE" in out
    assert "better fit       : square-root" not in out


def test_functional_form_declared_when_the_gap_is_real():
    out = format_report(_analysis(0.1, 4.5, 0.08, 3.1, 0.30, 0.10))       # gap 0.20
    assert "better fit       : square-root" in out
