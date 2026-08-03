"""RL optimal-execution agent: an ABIDES TradingAgent (modeled on
ABIDES/agent/execution/POVExecutionAgent.py) that, at each of 10 decision
points spaced 30s apart, blocks on an external action supplied via a
queue.Queue rather than following a fixed rule -- this is what lets
rl_execution/env.py expose a genuine Gym-style reset()/step() interface with
the policy living outside the ABIDES agent (so the same environment can run
the trained policy, a TWAP baseline, or anything else, per deliverable 6).

Bridge mechanics: the ABIDES Kernel runs in a background thread (see env.py).
At each decision point this agent computes the state, puts it on
state_queue, then calls action_queue.get() -- a genuine blocking wait that
freezes the Kernel's single-threaded event loop until env.step(action) on the
main thread supplies an action. There is at most one Kernel thread alive at a
time (episodes run sequentially), so this is a simple, hazard-free
producer/consumer handoff, not real concurrency.
"""

import queue

import numpy as np
import pandas as pd

from agent.TradingAgent import TradingAgent

N_DECISIONS = 10
DECISION_INTERVAL = pd.Timedelta("30s")
FINALIZE_DELAY = pd.Timedelta("1s")  # lets the last order's fill confirmation arrive before done=True

# Action levels: discrete participation rate, passive -> aggressive (spec: 4-5 levels).
# "limit" orders join our own side's touch (or cross it); "market" orders execute immediately.
# price_cross: None = price at own best; "cross" = price at the opposite side's best (marketable).
ACTION_LEVELS = [
    {"name": "passive",         "order_type": "limit",  "price_cross": False, "participation": 0.5},
    {"name": "light",           "order_type": "limit",  "price_cross": False, "participation": 1.0},
    {"name": "neutral",         "order_type": "limit",  "price_cross": True,  "participation": 1.0},
    {"name": "aggressive",      "order_type": "market",  "price_cross": None, "participation": 1.5},
    {"name": "very_aggressive", "order_type": "market",  "price_cross": None, "participation": 2.0},
]

# State bucketing thresholds -- first-pass, documented constants, not calibrated against a
# full data sweep (the environment's correctness matters more than the discretization being
# perfectly tuned; these are easy to revisit once real episodes are observed).
SPREAD_TICK_UNIT = 100  # raw LOBSTER price units per one-cent tick (see orderbook_reconstructor.py)
SPREAD_BUCKETS = [1, 2, 5]  # tick thresholds -> 4 buckets: 1, 2, 3-5, 6+
VOL_BUCKETS_TICKS = [1.0, 3.0]  # realized-vol (std of mid-price returns, in ticks) -> 3 buckets
OFI_BUCKETS = [-0.2, 0.2]  # order-flow imbalance -> 3 buckets: negative / neutral / positive
VOL_LOOKBACK = 5  # decision points


class RLExecutionAgent(TradingAgent):

    def __init__(self, id, name, type, symbol, direction, quantity, p_arrival,
                 start_time, state_queue: queue.Queue, action_queue: queue.Queue,
                 starting_cash=100000, log_orders=False, random_state=None):
        super().__init__(id, name, type, starting_cash=starting_cash, log_orders=log_orders,
                          random_state=random_state)
        assert direction in ("BUY", "SELL")
        self.symbol = symbol
        self.direction = direction
        self.quantity = quantity
        self.rem_quantity = quantity
        self.p_arrival = p_arrival
        self.start_time = start_time
        self.state_queue = state_queue
        self.action_queue = action_queue

        self.decision_index = 0
        self.mid_history = []
        self.fills = []  # list of (qty, fill_price)
        self.state = "AWAITING_WAKEUP"
        self._finalizing = False
        self._finalized = False
        self._anchor = None        # sim time of the first tradeable wakeup; the schedule's origin
        self._awaiting_action = False
        self._scheduled_times = set()

    def getWakeFrequency(self):
        return DECISION_INTERVAL

    def _schedule(self, when):
        """setWakeup, deduplicated. TradingAgent schedules its own wakeup once it
        learns the market hours (TradingAgent.receiveMessage), independently of
        ours -- without dedupe, two wakeups land at the same sim time and two
        decision points fire at once, silently compressing the 10-decision
        schedule and consuming two actions at the same instant.
        """
        if when not in self._scheduled_times:
            self._scheduled_times.add(when)
            self.setWakeup(when)

    def wakeup(self, currentTime):
        can_trade = super().wakeup(currentTime)
        if self._finalizing:
            self._finalize(currentTime)
            return
        if not can_trade:
            # TradingAgent schedules the next wakeup itself once it knows market hours.
            return
        if self.decision_index >= N_DECISIONS or self._awaiting_action:
            return
        if self._anchor is None:
            # Anchor the decision schedule to the first tradeable moment rather than to
            # kernel start: the first wakeup is consumed learning the market hours, so
            # decision 0 cannot happen at exactly t0.
            self._anchor = currentTime
        target = self._anchor + self.decision_index * DECISION_INTERVAL
        if currentTime < target:
            self._schedule(target)
            return
        self._awaiting_action = True
        self.state = "AWAITING_SPREAD"
        self.getCurrentSpread(self.symbol, depth=10)

    def kernelTerminating(self):
        """Safety net: if the kernel ends for any reason before the episode
        finalized (stop time reached early, an agent error, an unexpected
        schedule), the env would otherwise block forever on state_queue.get().
        Always emit a terminal message.
        """
        if not self._finalized:
            self._finalize(self.currentTime, reason="kernel_terminated_before_finalize")
        super().kernelTerminating()

    def receiveMessage(self, currentTime, msg):
        super().receiveMessage(currentTime, msg)
        if msg.body["msg"] == "ORDER_EXECUTED":
            order = msg.body["order"]
            self.fills.append((order.quantity, order.fill_price))
            self.rem_quantity = max(0, self.quantity - sum(q for q, _ in self.fills))

    def querySpread(self, symbol, price, bids, asks, book):
        super().querySpread(symbol, price, bids, asks, book)
        if self.state != "AWAITING_SPREAD":
            return

        best_bid = bids[0][0] if bids else None
        best_ask = asks[0][0] if asks else None
        mid = (best_bid + best_ask) / 2.0 if (best_bid is not None and best_ask is not None) else self.p_arrival
        self.mid_history.append(mid)

        obs = self._build_state(best_bid, best_ask, bids, asks)
        self.state_queue.put(("obs", obs, {"decision_index": self.decision_index}))

        action = self.action_queue.get()  # blocks the Kernel thread until env.step() supplies one

        is_last = self.decision_index == N_DECISIONS - 1
        # On the last decision, sweep whatever inventory remains -- the parent order must
        # complete within the window for implementation shortfall to be well defined
        # (the standard Almgren-Chriss terminal constraint).
        qty = self.rem_quantity if is_last else self._child_order_qty(action)
        if qty > 0:
            self._place_child_order(action, qty, best_bid, best_ask)

        self.decision_index += 1
        self._awaiting_action = False
        self.state = "AWAITING_WAKEUP"
        if is_last:
            self._finalizing = True
            # Let the final orders rest/fill for one more interval, so the episode covers
            # the full 5-minute window and last-slice fills are actually captured.
            self._schedule(self.currentTime + DECISION_INTERVAL)
        else:
            self._schedule(self._anchor + self.decision_index * DECISION_INTERVAL)

    def _child_order_qty(self, action_idx):
        base_slice = self.quantity / N_DECISIONS
        qty = round(ACTION_LEVELS[action_idx]["participation"] * base_slice)
        return int(min(qty, self.rem_quantity))

    def _place_child_order(self, action_idx, qty, best_bid, best_ask):
        level = ACTION_LEVELS[action_idx]
        is_buy = self.direction == "BUY"
        if level["order_type"] == "market" or best_bid is None or best_ask is None:
            self.placeMarketOrder(self.symbol, qty, is_buy_order=is_buy)
            return
        own_best = best_bid if is_buy else best_ask
        opp_best = best_ask if is_buy else best_bid
        price = opp_best if level["price_cross"] else own_best
        self.placeLimitOrder(self.symbol, qty, is_buy_order=is_buy, limit_price=price)

    def _build_state(self, best_bid, best_ask, bids, asks):
        time_remaining_frac = (N_DECISIONS - self.decision_index) / N_DECISIONS
        inventory_remaining_frac = self.rem_quantity / self.quantity if self.quantity else 0.0

        spread_ticks = ((best_ask - best_bid) / SPREAD_TICK_UNIT) if (best_bid and best_ask) else float("nan")
        spread_bucket = _bucket(spread_ticks, SPREAD_BUCKETS)

        recent = self.mid_history[-VOL_LOOKBACK:]
        if len(recent) >= 2:
            returns = np.diff(recent)
            vol_ticks = float(np.std(returns)) / SPREAD_TICK_UNIT
        else:
            vol_ticks = 0.0
        vol_bucket = _bucket(vol_ticks, VOL_BUCKETS_TICKS)

        bid_sz = bids[0][1] if bids else 0
        ask_sz = asks[0][1] if asks else 0
        ofi = (bid_sz - ask_sz) / (bid_sz + ask_sz) if (bid_sz + ask_sz) > 0 else 0.0
        ofi_bucket = _bucket(ofi, OFI_BUCKETS)

        return {
            "time_remaining_frac": time_remaining_frac,
            "inventory_remaining_frac": inventory_remaining_frac,
            "spread_bucket": spread_bucket,
            "vol_bucket": vol_bucket,
            "ofi_bucket": ofi_bucket,
        }

    def _finalize(self, currentTime, reason="completed"):
        if self._finalized:
            return
        self._finalized = True
        self._finalizing = False
        shortfall = self._compute_shortfall()
        info = {
            "fills": list(self.fills),
            "rem_quantity": self.rem_quantity,
            "shortfall": shortfall,
            "decisions_made": self.decision_index,
            "termination_reason": reason,
        }
        self.state_queue.put(("done", -shortfall, info))

    def _compute_shortfall(self):
        if not self.fills or self.quantity == 0:
            return 0.0
        total_qty = sum(q for q, _ in self.fills)
        if total_qty == 0:
            return 0.0
        if self.direction == "SELL":
            # positive shortfall = sold below arrival price (bad)
            cost = sum(q * (self.p_arrival - p) for q, p in self.fills)
        else:
            # positive shortfall = paid above arrival price (bad)
            cost = sum(q * (p - self.p_arrival) for q, p in self.fills)
        return cost / self.quantity


def _bucket(value, thresholds):
    if value != value:  # NaN
        return 0
    for i, t in enumerate(thresholds):
        if value <= t:
            return i
    return len(thresholds)
