"""World agent that replays the REAL order flow from t0 onward, instead of
generating it.

This is the "real data" arm. It answers the question the RL chapter is actually
for -- whether the simulator is usable for a live trading application -- by running
the same policies against the market that actually happened, and comparing.

Three properties make it worth having:

  NO MODEL.  Replay involves no diffusion sampling, which is the entire per-episode
  cost of the generative arm. Episodes should run in seconds rather than ~20 minutes,
  so hundreds of held-out seeds are affordable and the error bars can be far tighter
  than the generative arm can reach.

  NO DRIFT.  Real prices are near-martingale. The generative arm's measured +11.94bps
  per 5 minutes (t=4.84) systematically flatters a seller, so replay gives the
  uncontaminated read, and the GAP between arms quantifies the contamination.

  AN EXACT COUNTERFACTUAL.  Replayed orders do not react to us, so running a seed with
  the execution agent disabled reproduces the market exactly. That licenses a causal
  impact estimate the generative arm cannot give -- and, by the same token, means
  replay UNDERSTATES impact, because a real market would have responded. Reporting
  both arms brackets the truth rather than picking a side.

Faithfulness detail: the standard replay filter drops any cancel or execution whose
originating order is absent, which is right when replaying from the open but wrong
here. Starting mid-day, some messages legitimately reference orders resting since
before t0 -- and those orders ARE present, because the cold-start reconstruction
seeded them with their real order ids. They are kept.

Measured on INTC 2015-01-30 from a midday t0 this recovers 343 of 484,801 messages,
0.1% of the replayed flow -- worth having and cheap, but small, because order
lifetimes are short enough that most cancels reference orders placed minutes earlier.
Do not describe it in the write-up as a major correction; it is a correctness detail.
"""

import datetime

import numpy as np
import pandas as pd

from agent.WorldAgent import WorldAgent
from rl_execution.coldstart import MESSAGE_COLUMNS
from rl_execution.rl_world_agent import fresh_order_id_pool

# LOBSTER event types that never touch the visible book (hidden execution, cross
# trade, halt) -- matching WorldAgent._preprocess_events_for_market_replay.
NON_BOOK_EVENTS = (5, 6, 7)


def build_replay_stream(messages: pd.DataFrame, t0: float, seeded_order_ids,
                         horizon_seconds=None):
    """Messages to replay from t0 onward, with inter-arrival gaps.

    Returns an array of [gap_seconds, event_type, order_id, size, price, direction],
    the layout WorldAgent.placeOrder expects, where gap_seconds is the delay from the
    previous replayed message (first entry 0).

    horizon_seconds bounds the window. Without it the stream runs to the end of the
    trading day -- 631,710 messages on a midday t0 -- when the kernel stops after
    EPISODE_SECONDS + 60 and consumes only ~12,000 of them. The rest is built, held and
    thrown away on every reset. That is tolerable for a handful of evaluation episodes
    and not for a training run of several thousand, which replay's ~5s episodes now make
    affordable. Pass the kernel's own window; None keeps the whole day.
    """
    window = messages[(messages["time"] >= t0) & (~messages["event_type"].isin(NON_BOOK_EVENTS))]
    if horizon_seconds is not None:
        window = window[window["time"] <= t0 + float(horizon_seconds)]
    if window.empty:
        return np.zeros((0, 6))

    # Keep a cancel/execution only if we can actually act on it: either its order was
    # placed inside the replay window, or it is resting in the seeded book.
    placed_here = set(window.loc[window["event_type"] == 1, "order_id"].to_numpy().tolist())
    actionable = placed_here | set(int(o) for o in seeded_order_ids)
    keep = (window["event_type"] == 1) | window["order_id"].isin(actionable)
    window = window[keep]
    if window.empty:
        return np.zeros((0, 6))

    times = window["time"].to_numpy(dtype=float)
    gaps = np.diff(times, prepend=times[0])
    out = window[MESSAGE_COLUMNS].to_numpy(dtype=float)
    out[:, 0] = np.maximum(gaps, 0.0)
    return out


class ReplayWorldAgent(WorldAgent):
    """Cold-start seeded, then plays the real message stream forward from t0."""

    def __init__(self, *args, replay_stream, seed_lob_snapshots, seed_price_anchor=0.0, **kwargs):
        self._seed_price_anchor = float(seed_price_anchor or 0.0)
        kwargs["using_diffusion"] = False   # never switch to generation
        kwargs["model"] = None
        super().__init__(*args, **kwargs)
        self.historical_orders = np.asarray(replay_stream, dtype=float)
        self.historical_order_ids = (self.historical_orders[:, 2]
                                      if len(self.historical_orders) else np.zeros(0))
        self.next_historical_orders_index = 0
        self.lob_snapshots = list(seed_lob_snapshots)
        self.last_offset_time = 0.0
        # Replay places REAL orders under their real ids, so keeping the market orders it
        # creates for type-4 executions out of that range matters here even more than in
        # the generative arm. See WORLD_ORDER_ID_BASE.
        self.unused_order_ids = fresh_order_id_pool()
        # Replay never generates, so it never leaves this window.
        self.starting_time_diffusion = "157780min"

    def _load_orders_lob(self, symbol, data_dir, date, date_trading_days):
        """Skip WorldAgent's full-day CSV load: the replay stream is supplied ready-made
        by the caller, which has already read the day for the cold-start reconstruction.
        Returns one synthetic LOB row carrying the real day-open mid, because
        WorldAgent.__init__ feeds this to compute_price_anchor when PRICE_REANCHOR is on.
        """
        anchor_row = np.zeros((1, 40))
        if self._seed_price_anchor:
            anchor_row[0, 0] = self._seed_price_anchor
            anchor_row[0, 2] = self._seed_price_anchor
        return np.zeros((0, 6)), anchor_row

    def _record_flow(self, event_type):
        """Populate the same realism counters the generative agent fills during decoding.

        ExecutionEnv._collect_world_agent_diagnostics reads decoded_type_counts and
        _exec_outcomes to report flow mix and execution rate. Nothing decodes in replay,
        so without this both come back empty and execution_rate logs as None -- which
        silently removes the comparison the whole arm exists to make, since the
        generative market's 17-18% execution rate is only damning against the real 4-6%.

        Here the numbers are definitional rather than estimated: an event_type 4 in the
        LOBSTER stream IS an execution, so the ratio of type-4 to type-1 events over the
        replayed window is the real market's execution rate, measured not inferred.

        Deliberately does NOT feed _exec_outcomes. That is a deque(maxlen=1000) holding
        one flag per order the generative agent placed, so appending both placements and
        executions to it would compute executions/(placements+executions) over only the
        last thousand events -- a different quantity from the generative arm's, silently
        not comparable with it. The env derives the rate from these counts instead.
        """
        et = int(event_type)
        if et in self.decoded_type_counts:
            self.decoded_type_counts[et] += 1

    def wakeup(self, currentTime):
        self.currentTime = currentTime
        if self.first_wakeup:
            self.state = "REPLAYING"
            self.requestDataSubscription(self.symbol, levels=10)
            self.first_wakeup = False
            self.setWakeup(currentTime + datetime.timedelta(microseconds=1))
            return
        if self.next_historical_orders_index >= len(self.historical_orders):
            return  # stream exhausted; the episode simply runs out of real flow
        order = self.historical_orders[self.next_historical_orders_index]
        self.last_offset_time = order[0]
        self._record_flow(order[1])
        self.placeOrder(currentTime, order)
        self.next_historical_orders_index += 1
        if self.next_historical_orders_index < len(self.historical_orders):
            gap = self.historical_orders[self.next_historical_orders_index, 0]
            self.setWakeup(currentTime + datetime.timedelta(seconds=float(gap))
                           + datetime.timedelta(microseconds=1))
