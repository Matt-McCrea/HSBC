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
