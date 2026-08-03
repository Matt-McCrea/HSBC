"""WorldAgent variant for RL episodes: skips the ~15-minute real-order-flow
replay through the ABIDES kernel entirely.

WorldAgent's own generation logic (_generate_order) always reads its
conditioning window fresh from self.placed_orders[-cond_seq_size:] and
self.lob_snapshots[-cond_seq_size-1:] (ABIDES/agent/WorldAgent.py, around
line 490) -- it doesn't care how those arrays got populated. Normally they're
built up by ~15 minutes of live replay; here they're pre-seeded once, directly
from rl_execution.coldstart's reconstruction of the real conditioning history
ending at t0. Nothing else in WorldAgent needs to change -- once seeded, all
of its existing generation, depth-noise, diagnostics, etc. behave exactly as
in a normal run.

Only wakeup()'s *first* call is overridden, to skip the replay branch and
subscribe to market data immediately; every subsequent wakeup falls straight
through to WorldAgent's own unmodified generation branch (self.
starting_time_diffusion is set to '0min' so the very next wakeup already
satisfies its "past the replay window" check).
"""

import numpy as np
import pandas as pd

from agent.WorldAgent import WorldAgent


class RLWorldAgent(WorldAgent):

    def __init__(self, *args, seed_placed_orders, seed_lob_snapshots, seed_price_anchor=0.0,
                 protected_agent_ids=(), **kwargs):
        # Read by the _load_orders_lob override below, which runs inside super().__init__.
        self._seed_price_anchor = float(seed_price_anchor or 0.0)
        kwargs["using_diffusion"] = True
        super().__init__(*args, **kwargs)
        self.placed_orders = list(seed_placed_orders)
        self.lob_snapshots = list(seed_lob_snapshots)
        self.starting_time_diffusion = "0min"
        # Orders owned by these agents (i.e. the RL execution agent) are never selected as
        # targets for a generated cancel. WorldAgent rebuilds active_limit_orders from the
        # WHOLE book (_update_active_limit_orders), so without this the world agent can
        # cancel the RL agent's resting child orders -- unrealistic (nobody else can cancel
        # your order in a real market) and it would silently corrupt execution results.
        self.protected_agent_ids = set(protected_agent_ids)
        # WorldAgent only ever sets self.last_offset_time inside the replay loop (or when the
        # rarely-used fix_time flag is on) -- never in __init__. Since replay never runs here,
        # it would otherwise stay unset until the first ORDER_ACCEPTED/EXECUTED/CANCELLED
        # message, which reads it directly (WorldAgent.py receiveMessage, ~line 395-409) and
        # crashes with AttributeError. Analogous value: the offset of the most recent REAL
        # order in the seeded conditioning window, matching how a normal run freezes this at
        # "the last historical order's gap" once replay ends (see the fix_time comment in
        # WorldAgent.wakeup for why it freezes rather than updates, in the default config).
        self.last_offset_time = float(self.placed_orders[-1][0])

    def _load_orders_lob(self, symbol, data_dir, date, date_trading_days):
        """RL episodes never replay history, so the full-day LOBSTER CSVs
        WorldAgent normally loads here (~250MB per day, plus a ~1M-row
        preprocessing pass) are pure waste -- and they'd be paid on EVERY
        reset(), which is exactly the per-episode cost this architecture
        exists to remove.

        Returns empty history plus a single synthetic LOB row carrying the
        real day-open mid, because WorldAgent.__init__ feeds this straight
        into compute_price_anchor() when PRICE_REANCHOR is on (which raises on
        an empty book). The anchor MUST be the day's opening mid to match
        training; the caller computes it from the real full-day orderbook in
        coldstart.seed_episode and passes it in as seed_price_anchor.
        """
        anchor_row = np.zeros((1, 40))
        if self._seed_price_anchor:
            # compute_price_anchor reads ask1 (col 0) and bid1 (col 2) and returns their
            # rounded mid -- setting both to the anchor reproduces it exactly.
            anchor_row[0, 0] = self._seed_price_anchor
            anchor_row[0, 2] = self._seed_price_anchor
        return np.zeros((0, 6)), anchor_row

    def _update_active_limit_orders(self):
        super()._update_active_limit_orders()
        if not self.protected_agent_ids:
            return
        self.active_limit_orders = {
            oid: o for oid, o in self.active_limit_orders.items()
            if getattr(o, "agent_id", None) not in self.protected_agent_ids
        }

    def wakeup(self, currentTime):
        self.currentTime = currentTime
        if self.first_wakeup:
            self.state = "PRE_GENERATING"
            self.requestDataSubscription(self.symbol, levels=10)
            self.first_wakeup = False
            self.setWakeup(currentTime + pd.Timedelta(microseconds=1))
            return
        super().wakeup(currentTime)
