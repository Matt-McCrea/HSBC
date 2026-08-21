"""Tests for the replay ("real data") arm.

The stream construction is where this can go quietly wrong, so it gets the most
attention. Starting mid-day differs from replaying from the open: some cancels and
executions in the window reference orders resting since before t0, and the standard
replay filter drops those as orphans while still looking like a successful replay.
On real INTC data the effect is small (0.1% of flow) but it is free to get right.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "ABIDES"))

from rl_execution.coldstart import MESSAGE_COLUMNS
from rl_execution.replay_world_agent import NON_BOOK_EVENTS, build_replay_stream

T0 = 40000.0


def _messages(rows):
    return pd.DataFrame(rows, columns=MESSAGE_COLUMNS)


def test_stream_starts_at_t0_and_keeps_new_orders():
    msgs = _messages([
        [T0 - 10, 1, 111, 100, 340000, -1],   # before t0, excluded
        [T0 + 1,  1, 222, 200, 340100, -1],
        [T0 + 3,  1, 333, 300, 340200, -1],
    ])
    stream = build_replay_stream(msgs, T0, seeded_order_ids=[])
    assert len(stream) == 2
    assert set(stream[:, 2].astype(int)) == {222, 333}


def test_messages_referencing_the_seeded_book_are_kept():
    """The property the standard filter gets wrong when starting mid-day. Order 111
    was placed before t0 and IS in the reconstructed book, so its cancellation is
    actionable and must be replayed."""
    msgs = _messages([
        [T0 + 1, 3, 111, 100, 340000, -1],   # cancel of a pre-t0 order that we seeded
        [T0 + 2, 1, 222, 200, 340100, -1],
        [T0 + 3, 4, 222, 200, 340100, -1],   # execution of an in-window order
    ])
    stream = build_replay_stream(msgs, T0, seeded_order_ids=[111])
    assert len(stream) == 3, "a cancel of a seeded resting order must be replayed"
    assert 111 in stream[:, 2].astype(int)


def test_messages_referencing_nothing_actionable_are_dropped():
    """A cancel for an order neither seeded nor placed in-window cannot be acted on;
    replaying it would just be ignored by the book."""
    msgs = _messages([
        [T0 + 1, 3, 999, 100, 340000, -1],   # unknown order
        [T0 + 2, 1, 222, 200, 340100, -1],
    ])
    stream = build_replay_stream(msgs, T0, seeded_order_ids=[111])
    assert set(stream[:, 2].astype(int)) == {222}


@pytest.mark.parametrize("event_type", NON_BOOK_EVENTS)
def test_non_book_events_are_dropped(event_type):
    msgs = _messages([
        [T0 + 1, event_type, 111, 100, 340000, -1],
        [T0 + 2, 1, 222, 200, 340100, -1],
    ])
    stream = build_replay_stream(msgs, T0, seeded_order_ids=[111])
    assert set(stream[:, 2].astype(int)) == {222}


def test_gaps_are_inter_arrival_times_and_never_negative():
    msgs = _messages([
        [T0 + 0.0, 1, 1, 100, 340000, -1],
        [T0 + 0.5, 1, 2, 100, 340000, -1],
        [T0 + 2.5, 1, 3, 100, 340000, -1],
    ])
    stream = build_replay_stream(msgs, T0, seeded_order_ids=[])
    assert stream[0, 0] == 0.0, "first replayed message starts immediately"
    assert stream[1, 0] == pytest.approx(0.5)
    assert stream[2, 0] == pytest.approx(2.0)
    assert (stream[:, 0] >= 0).all()


def test_empty_window_returns_an_empty_stream_not_an_error():
    msgs = _messages([[T0 - 5, 1, 1, 100, 340000, -1]])
    assert build_replay_stream(msgs, T0, seeded_order_ids=[]).shape == (0, 6)


def test_stream_preserves_the_placeOrder_column_layout():
    """WorldAgent.placeOrder indexes columns positionally, so the layout is load-bearing:
    order[1]=type, order[2]=id, order[3]=size, order[4]=price, order[5]=direction."""
    msgs = _messages([[T0 + 1, 1, 4242, 350, 340500, -1]])
    row = build_replay_stream(msgs, T0, seeded_order_ids=[])[0]
    assert int(row[1]) == 1 and int(row[2]) == 4242
    assert int(row[3]) == 350 and int(row[4]) == 340500 and int(row[5]) == -1


def test_replay_agent_needs_no_model_and_never_generates():
    from rl_execution.replay_world_agent import ReplayWorldAgent
    import inspect
    src = inspect.getsource(ReplayWorldAgent.__init__)
    assert 'kwargs["using_diffusion"] = False' in src, "replay must never switch to generation"
    assert 'kwargs["model"] = None' in src, "replay must not require a loaded model"


def test_env_replay_mode_skips_model_loading():
    """The whole point of the arm is that it runs on a CPU-only machine."""
    import inspect
    from rl_execution.env import ExecutionEnv
    src = inspect.getsource(ExecutionEnv.__init__)
    assert 'if world_mode == "replay"' in src
    assert "self.model, self.config, self.checkpoint_path = None, None, None" in src


# --- historic-data arm: the flags that let the 2x2 run ------------------------------

def test_train_exposes_world_mode_and_passes_it_to_the_env():
    """Training on replay is the conventional historical pipeline, and cell B of the
    2x2. Without this flag train.py can only ever build a generative env."""
    import inspect
    import rl_execution.train as train_mod
    src = inspect.getsource(train_mod)
    assert '"--world-mode"' in src, "train must expose --world-mode"
    assert "world_mode=args.world_mode" in src, "and forward it into ExecutionEnv"


def test_benchmark_forwards_side_to_reset():
    """Calibration episodes must be able to match the side training will use: sigma and
    the drift t-stat are side-sensitive in a drifting market."""
    import inspect
    from rl_execution.benchmark import run_benchmark
    src = inspect.getsource(run_benchmark)
    assert "env.reset(side=side)" in src


def test_forcing_side_leaves_days_t0_and_Q_untouched():
    """--side must change only the side, or the forced run stops being comparable with
    the mixed-side run it is meant to sit beside."""
    from rl_execution.evaluate import generate_held_out_seeds
    mixed = generate_held_out_seeds("data", "INTC", 6, seed=123)
    sell = generate_held_out_seeds("data", "INTC", 6, seed=123, side="SELL")
    assert all(s["side"] == "SELL" for s in sell)
    for a, b in zip(mixed, sell):
        assert (a["seed_day"], a["t0"], a["Q"]) == (b["seed_day"], b["t0"], b["Q"])


def test_episode_seed_is_positional_so_policies_share_a_market():
    """Common random numbers: seed i must map to the same episode RNG for every policy
    and every world mode, otherwise the paired comparison still carries market noise."""
    import inspect
    from rl_execution.evaluate import evaluate_policy
    src = inspect.getsource(evaluate_policy)
    assert "for i, s in enumerate(seeds)" in src
    assert "int(episode_seed_base) + i" in src
    # Absent a base the behaviour must be exactly as before -- generative runs already
    # committed to logs should not silently change meaning.
    assert "None if episode_seed_base is None else" in src


def test_run_comparison_derives_the_seed_base_from_eval_seed():
    import inspect
    from rl_execution.evaluate import run_comparison
    src = inspect.getsource(run_comparison)
    assert "episode_seed_base = eval_seed" in src, "must be stable across arms, not ad hoc"
    assert src.count("episode_seed_base=episode_seed_base") >= 2, "both arms must use it"


def test_horizon_truncates_the_stream_to_the_kernel_window():
    """Unbounded, the stream runs to end of day (631,710 messages on a real midday t0)
    while the kernel consumes ~12,000. Every reset paid for the rest."""
    rows = [[T0 + t, 1, 100 + t, 100, 340000, -1] for t in range(0, 1000, 10)]
    msgs = _messages(rows)
    full = build_replay_stream(msgs, T0, seeded_order_ids=[])
    bounded = build_replay_stream(msgs, T0, seeded_order_ids=[], horizon_seconds=360)
    assert len(full) == 100
    assert len(bounded) == 37, "inclusive of t0+360"
    assert (np.cumsum(bounded[:, 0]) <= 360).all()


def test_horizon_none_keeps_the_whole_day():
    rows = [[T0 + t, 1, 100 + t, 100, 340000, -1] for t in range(0, 1000, 10)]
    assert len(build_replay_stream(_messages(rows), T0, [], horizon_seconds=None)) == 100


def test_replay_reports_execution_rate_from_event_types():
    """execution_rate is the realism comparison the arm exists for: the generative market
    fills at 17-18% against a real 4-6%. In replay nothing decodes, so it must come from
    the stream's own event types or it logs as None and the comparison disappears."""
    import inspect
    from rl_execution.env import ExecutionEnv
    from rl_execution.replay_world_agent import ReplayWorldAgent

    src = inspect.getsource(ExecutionEnv._collect_world_agent_diagnostics)
    assert "counts.get(4, 0) / placed" in src, "executions per new order"
    assert "if n_exec:" in src, "generative path must be preferred and unchanged"

    rec = inspect.getsource(ReplayWorldAgent._record_flow)
    assert "self.decoded_type_counts[et] += 1" in rec
    assert "_exec_outcomes" not in rec.split('"""')[2], "must not feed the capped deque"


def test_flow_counters_are_updated_as_the_stream_is_walked():
    import inspect
    from rl_execution.replay_world_agent import ReplayWorldAgent
    assert "self._record_flow(order[1])" in inspect.getsource(ReplayWorldAgent.wakeup)
