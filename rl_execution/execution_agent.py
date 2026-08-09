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

# Child-order ids are allocated from here upwards. WorldAgent draws its own ids from
# np.arange(0, 5_000_000) minus the historical ids, so starting well above that range
# guarantees no collision -- and collisions would silently mis-attribute fills, since
# attribution is by order id.
EXEC_ORDER_ID_BASE = 900_000_000


def signed_cost(qty, price, benchmark, direction):
    """Execution cost of one fill against a benchmark price, sign-matched to
    implementation shortfall: POSITIVE is bad (sold below / bought above)."""
    return qty * ((benchmark - price) if direction == "SELL" else (price - benchmark))


def per_step_rewards(fills_by_step, benchmarks, quantity, direction, n_steps):
    """Turn fills attributed to each decision point into a per-step reward series.

    Reward is -cost/Q, so summing the series reproduces exactly the negative of the
    episode's implementation shortfall when `benchmarks` is the arrival mid at every
    step -- the per-step form is a re-attribution of the same total, not a different
    objective.

    Passing the PREVAILING mid at each step instead measures execution against the
    price available when the order was placed, which removes market drift by
    construction rather than by estimating and subtracting it. Drift is common-mode
    noise (it moves every action in an episode identically, so it says nothing about
    which action was better) and it dominates reward variance, so removing it is the
    single biggest variance reduction available. The reported shortfall is unaffected
    either way -- it is always computed against the arrival mid.
    """
    rewards = [0.0] * n_steps
    for step, fills in fills_by_step.items():
        if not 0 <= step < n_steps:
            continue
        benchmark = benchmarks[step] if step < len(benchmarks) else benchmarks[-1]
        cost = sum(signed_cost(q, p, benchmark, direction) for q, p in fills)
        rewards[step] = -cost / quantity if quantity else 0.0
    return rewards


class RLExecutionAgent(TradingAgent):

    def __init__(self, id, name, type, symbol, direction, quantity, p_arrival,
                 start_time, state_queue: queue.Queue, action_queue: queue.Queue,
                 starting_cash=100000, log_orders=False, random_state=None,
                 reward_mode="terminal", reward_benchmark="arrival"):
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

        # "terminal": one payment at the end (original). "per-step": each fill's cost is
        # paid at the decision point whose child order produced it -- identical total, but
        # dense, which is what makes the credit assignment tractable at a few hundred
        # episodes. reward_benchmark selects arrival mid (true shortfall) or the prevailing
        # mid at each step (drift-free); see per_step_rewards.
        self.reward_mode = reward_mode
        self.reward_benchmark = reward_benchmark
        self._next_order_id = EXEC_ORDER_ID_BASE + int(id) * 1_000_000
        self._order_to_step = {}          # child order id -> decision index that placed it
        self._fills_by_step = {}          # decision index -> [(qty, fill_price), ...]
        self._paid_steps = set()          # steps whose reward has already been handed out

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
        self._cancel_outstanding()
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
            # Attribute by ORDER ID, not by arrival time: a passive child order can fill
            # after later decisions have been taken, and crediting that fill to whichever
            # decision happened to be current would blame the wrong action.
            step = self._order_to_step.get(order.order_id, self.decision_index)
            self._fills_by_step.setdefault(step, []).append((order.quantity, order.fill_price))

    def querySpread(self, symbol, price, bids, asks, book):
        super().querySpread(symbol, price, bids, asks, book)
        if self.state != "AWAITING_SPREAD":
            return

        best_bid = bids[0][0] if bids else None
        best_ask = asks[0][0] if asks else None
        mid = (best_bid + best_ask) / 2.0 if (best_bid is not None and best_ask is not None) else self.p_arrival
        self.mid_history.append(mid)

        obs = self._build_state(best_bid, best_ask, bids, asks)
        # Reward for the PREVIOUS decision, settled now that its interval has closed and
        # its fills are in. env.step(a_k) returns this alongside the next observation,
        # which is the standard (s, a, r, s') ordering.
        prev_step = self.decision_index - 1
        self.state_queue.put(("obs", obs, {
            "decision_index": self.decision_index,
            "reward": self._settle_rewards(),
            # Reported for the step being SETTLED, not the one starting: the caller logs
            # these against the (s, a) it is about to record, so they must describe that
            # step's execution -- the mid it faced and the fills it produced.
            "step_mid": (self.mid_history[prev_step] if 0 <= prev_step < len(self.mid_history) else None),
            "step_fills": list(self._fills_by_step.get(prev_step, [])),
        }))

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
        order_id = self._next_order_id
        self._next_order_id += 1
        self._order_to_step[order_id] = self.decision_index
        if level["order_type"] == "market" or best_bid is None or best_ask is None:
            self.placeMarketOrder(self.symbol, qty, is_buy_order=is_buy, order_id=order_id)
            return
        own_best = best_bid if is_buy else best_ask
        opp_best = best_ask if is_buy else best_bid
        price = opp_best if level["price_cross"] else own_best
        self.placeLimitOrder(self.symbol, qty, is_buy_order=is_buy, limit_price=price,
                             order_id=order_id)

    def _cancel_outstanding(self):
        """Cancel any child order still resting from an earlier decision point.

        Three things this fixes at once. It bounds each decision's fills to its own
        interval, which is what makes per-step attribution exact. It stops stale quotes
        from earlier decisions sitting in the book all episode distorting it. And it
        removes a real over-execution bug: rem_quantity is derived from fills, so the
        terminal sweep sizes itself against inventory that old resting orders could then
        fill on top of, selling more than the parent order. Re-quoting each interval is
        also what production execution algos do (cf. POVExecutionAgent.cancelOrders).
        """
        for order in list(self.orders.values()):
            self.cancelOrder(order)

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

    def _benchmarks(self):
        """Benchmark price per decision point: the arrival mid throughout (so the reward
        series sums to true shortfall), or the prevailing mid at each step (drift-free)."""
        if self.reward_benchmark == "prevailing" and self.mid_history:
            return list(self.mid_history)
        return [self.p_arrival] * max(1, len(self.mid_history))

    def _settle_rewards(self):
        """Pay out every step whose fills are now known and which has not been paid yet.

        In terminal mode this always returns 0 until _finalize, preserving the original
        behaviour exactly.
        """
        if self.reward_mode != "per-step":
            return 0.0
        rewards = per_step_rewards(self._fills_by_step, self._benchmarks(), self.quantity,
                                   self.direction, N_DECISIONS + 1)
        due = 0.0
        for step in sorted(self._fills_by_step):
            if step not in self._paid_steps and step < self.decision_index:
                due += rewards[step]
                self._paid_steps.add(step)
        return due

    def _final_reward(self, shortfall):
        """Terminal payment. In per-step mode most of the episode has already been paid
        out, so only the steps not yet settled are due here -- paying -shortfall again
        would double-count everything already handed to the agent."""
        if self.reward_mode != "per-step":
            return -shortfall
        rewards = per_step_rewards(self._fills_by_step, self._benchmarks(), self.quantity,
                                   self.direction, N_DECISIONS + 1)
        due = sum(r for step, r in enumerate(rewards) if step not in self._paid_steps)
        self._paid_steps.update(range(len(rewards)))
        return due

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
            # The final decision's own mid and fills, which no later observation will
            # carry because there isn't one.
            "step_mid": (self.mid_history[-1] if self.mid_history else None),
            "step_fills": list(self._fills_by_step.get(max(0, self.decision_index - 1), [])),
            # Mid at the last decision point. Shortfall alone cannot separate execution
            # quality from where the market simply went: with a SELL-only design, any
            # upward drift in the generated price flatters the seller, and TRADES has a
            # documented directional-drift failure mode (see --flow-balance). Logging the
            # closing mid makes that decomposable after the fact instead of confounded.
            "p_final": (self.mid_history[-1] if self.mid_history else None),
        }
        self.state_queue.put(("done", self._final_reward(shortfall), info))

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
