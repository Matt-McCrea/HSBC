"""The curve must show the shape rather than assume it: a linear impact process should
give a flat per-share column, a square-root one a falling column. That distinction is
what the R^2 comparison could not resolve."""
import os, sys
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT); sys.path.insert(0, os.path.join(_ROOT, "ABIDES"))

from rl_execution.impact import impact_curve

MID = 365150.0


def _records(shape, n=600, seed=0):
    """Fills whose slippage follows a known functional form in size."""
    rng = np.random.RandomState(seed)
    recs = []
    for i in range(n // 10):
        traj = []
        for _ in range(10):
            qty = float(rng.randint(50, 1000))
            raw = 0.02 * qty if shape == "linear" else 0.6 * np.sqrt(qty)
            vwap = MID - raw + rng.normal(0, 0.5)      # SELL: fills below the mid
            traj.append({"mid": MID, "fills": [[qty, vwap]]})
        recs.append({"side": "SELL", "trajectory": traj})
    return recs


def test_linear_impact_gives_a_flat_per_share_column():
    rows = impact_curve(_records("linear"))
    per = [r["bps_per_100sh"] for r in rows]
    assert len(rows) >= 4
    assert max(per) / min(per) < 1.25, f"should be roughly flat, got {per}"


def test_square_root_impact_gives_a_falling_per_share_column():
    rows = impact_curve(_records("sqrt"))
    per = [r["bps_per_100sh"] for r in rows]
    assert per[0] > per[-1] * 1.8, f"should fall clearly with size, got {per}"


def test_slippage_rises_with_size_under_both_shapes():
    for shape in ("linear", "sqrt"):
        rows = impact_curve(_records(shape))
        slips = [r["mean_slip_bps"] for r in rows]
        assert slips == sorted(slips), f"{shape}: slippage must increase with size"


def test_buy_side_sign_matches_sell_side():
    """A buy paying above the mid is the same cost as a sell filling below it."""
    sell = impact_curve(_records("linear", seed=3))
    buy = _records("linear", seed=3)
    for rec in buy:
        rec["side"] = "BUY"
        for st in rec["trajectory"]:
            st["fills"] = [[q, 2 * MID - p] for q, p in st["fills"]]
    rows = impact_curve(buy)
    assert np.allclose([r["mean_slip_bps"] for r in rows],
                       [r["mean_slip_bps"] for r in sell], rtol=1e-6)


def test_too_little_data_returns_nothing_rather_than_noise():
    assert impact_curve(_records("linear", n=10)) == []
