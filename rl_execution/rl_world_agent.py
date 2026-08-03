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

import pandas as pd

from agent.WorldAgent import WorldAgent


class RLWorldAgent(WorldAgent):

    def __init__(self, *args, seed_placed_orders, seed_lob_snapshots, **kwargs):
        kwargs["using_diffusion"] = True
        super().__init__(*args, **kwargs)
        self.placed_orders = list(seed_placed_orders)
        self.lob_snapshots = list(seed_lob_snapshots)
        self.starting_time_diffusion = "0min"

    def wakeup(self, currentTime):
        self.currentTime = currentTime
        if self.first_wakeup:
            self.state = "PRE_GENERATING"
            self.requestDataSubscription(self.symbol, levels=10)
            self.first_wakeup = False
            self.setWakeup(currentTime + pd.Timedelta(microseconds=1))
            return
        super().wakeup(currentTime)
