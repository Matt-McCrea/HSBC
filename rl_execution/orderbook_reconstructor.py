"""Standalone LOBSTER L3 order-book reconstructor.

Given a raw LOBSTER message CSV and a target time-of-day ``t0`` (seconds since
midnight, LOBSTER's own time convention), replays add/cancel/execute events from
market open up to ``t0`` and returns the exact resting order book state --
individual order id, side, price, size, and entry time. No ABIDES kernel, no
model inference: pure bookkeeping over the message log, the same operation any
LOBSTER-style book-building tool performs.

LOBSTER message columns (mirrors WorldAgent._load_orders_lob's own convention,
including tolerating extra trailing columns some data drops carry -- e.g. an
exchange/venue tag -- by only reading the first 6):
    time, event_type, order_id, size, price, direction

event_type: 1=new limit order, 2=partial cancel, 3=deletion (full cancel),
4=visible execution, 5=hidden execution, 6=cross trade, 7=trading halt.
Only 1/2/3/4 mutate the visible resting book; 5/6/7 are ignored here, mirroring
the existing repo's own convention in
WorldAgent._preprocess_events_for_market_replay (which drops 5/6/7 outright).

Matching: a naive replay that only trusts the log's own type-2/3/4 events to
remove resting orders is NOT sufficient -- a small number of orders (empirically
concentrated in the first few minutes after the open, likely an artifact of how
the opening auction is or isn't captured by the standard continuous-trading
event types) never get an explicit removal event even though a later, genuinely
crossing order arrives and should have matched against them. Left unhandled,
these "ghost" orders permanently corrupt the top-of-book for the rest of the day
(validated against LOBSTER's own paired orderbook.csv in
tests/test_reconstructor.py). The fix is to make the reconstructor a real
price-time-priority matching book -- like any actual exchange, or like ABIDES's
own OrderBook (ABIDES/util/OrderBook.py, not reused directly here to avoid its
live-agent notification/logging side effects) -- so an incoming order that
crosses the resting book always sweeps through it before resting the remainder,
regardless of whether the raw log's own bookkeeping for the counterparty is
complete.
"""

import heapq
from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

MESSAGE_COLUMNS = ["time", "event_type", "order_id", "size", "price", "direction"]

NEW_ORDER = 1
PARTIAL_CANCEL = 2
DELETE = 3
EXECUTE_VISIBLE = 4

# 09:30:00, seconds since midnight -- LOBSTER's own time convention. The small burst of
# messages before this (the pre-open order-entry period) is excluded by default: it's not
# part of continuous trading, and the matching book below only needs a clean starting point,
# not this specific data's pre-open quirks (those are now handled robustly by matching itself).
DEFAULT_MARKET_OPEN = 34200.0


@dataclass(eq=False)
class RestingOrder:
    order_id: int
    side: str  # "buy" or "sell"
    price: int  # raw LOBSTER price ticks (dollars * 10000)
    size: int
    entry_time: float  # seconds since midnight, LOBSTER convention

    def age(self, t0: float) -> float:
        return t0 - self.entry_time


class _MatchingBook:
    """Minimal price-time-priority limit order book, matching semantics only
    (no notifications/logging -- this is a reconstruction tool, not a live agent).
    """

    def __init__(self):
        self.bid_levels: dict[int, deque] = {}
        self.ask_levels: dict[int, deque] = {}
        self._bid_heap: list[int] = []  # negated prices: top of heap = best (highest) bid
        self._ask_heap: list[int] = []  # prices: top of heap = best (lowest) ask
        self.by_id: dict[int, RestingOrder] = {}

    def _best_bid_price(self):
        while self._bid_heap:
            p = -self._bid_heap[0]
            if self.bid_levels.get(p):
                return p
            heapq.heappop(self._bid_heap)
        return None

    def _best_ask_price(self):
        while self._ask_heap:
            p = self._ask_heap[0]
            if self.ask_levels.get(p):
                return p
            heapq.heappop(self._ask_heap)
        return None

    def _rest(self, order: RestingOrder):
        if order.side == "buy":
            self.bid_levels.setdefault(order.price, deque()).append(order)
            heapq.heappush(self._bid_heap, -order.price)
        else:
            self.ask_levels.setdefault(order.price, deque()).append(order)
            heapq.heappush(self._ask_heap, order.price)
        self.by_id[order.order_id] = order

    def _drop(self, order: RestingOrder):
        levels = self.bid_levels if order.side == "buy" else self.ask_levels
        dq = levels.get(order.price)
        if dq:
            try:
                dq.remove(order)
            except ValueError:
                pass
            if not dq:
                del levels[order.price]
        self.by_id.pop(order.order_id, None)

    def insert_new(self, order: RestingOrder):
        """Match against the opposite side first (price-time priority), then
        rest any unfilled remainder -- exactly what a real exchange does.
        """
        opposite_levels = self.ask_levels if order.side == "buy" else self.bid_levels
        remaining = order.size
        while remaining > 0:
            best_opp = self._best_ask_price() if order.side == "buy" else self._best_bid_price()
            if best_opp is None:
                break
            crosses = order.price >= best_opp if order.side == "buy" else order.price <= best_opp
            if not crosses:
                break
            dq = opposite_levels[best_opp]
            resting = dq[0]
            traded = min(remaining, resting.size)
            resting.size -= traded
            remaining -= traded
            if resting.size <= 0:
                dq.popleft()
                self.by_id.pop(resting.order_id, None)
                if not dq:
                    del opposite_levels[best_opp]
        if remaining > 0:
            order.size = remaining
            self._rest(order)

    def reduce_or_remove(self, order_id: int, delta: int):
        o = self.by_id.get(order_id)
        if o is None:
            return
        o.size -= delta
        if o.size <= 0:
            self._drop(o)

    def remove(self, order_id: int):
        o = self.by_id.get(order_id)
        if o is not None:
            self._drop(o)


def read_messages(message_csv_path) -> pd.DataFrame:
    return pd.read_csv(message_csv_path, header=None, names=MESSAGE_COLUMNS, usecols=range(6))


def reconstruct_book(message_csv_path, t0: float, market_open: float = DEFAULT_MARKET_OPEN) -> dict[int, RestingOrder]:
    """Replay LOBSTER messages up to and including t0; return resting orders.

    Returns a dict order_id -> RestingOrder for every order still resting at t0.
    ``t0`` is seconds-since-midnight, matching the LOBSTER message ``time`` column.
    """
    messages = read_messages(message_csv_path)
    return reconstruct_book_from_frame(messages, t0, market_open=market_open)


def reconstruct_book_from_frame(messages: pd.DataFrame, t0: float,
                                 market_open: float = DEFAULT_MARKET_OPEN) -> dict[int, RestingOrder]:
    window = messages[messages["time"] <= t0]
    return _replay(window, market_open=market_open)


def reconstruct_book_upto_index(messages: pd.DataFrame, end_index: int,
                                 market_open: float = DEFAULT_MARKET_OPEN) -> dict[int, RestingOrder]:
    """Replay messages.iloc[:end_index + 1]. For exact row-aligned comparison
    against a paired orderbook.csv (which is one row per message, in order) --
    avoids the tie-breaking ambiguity of filtering by timestamp alone when
    multiple messages share the same t.
    """
    return _replay(messages.iloc[: end_index + 1], market_open=market_open)


def _replay(window: pd.DataFrame, market_open: float = DEFAULT_MARKET_OPEN) -> dict[int, RestingOrder]:
    times = window["time"].to_numpy()
    event_types = window["event_type"].to_numpy()
    order_ids = window["order_id"].to_numpy()
    sizes = window["size"].to_numpy()
    prices = window["price"].to_numpy()
    directions = window["direction"].to_numpy()

    book = _MatchingBook()
    for t, et, oid, size, price, direction in zip(times, event_types, order_ids, sizes, prices, directions):
        if t < market_open:
            continue
        if et == NEW_ORDER:
            # direction==1 -> buy, direction==-1 -> sell (standard LOBSTER convention).
            side = "buy" if direction == 1 else "sell"
            book.insert_new(RestingOrder(int(oid), side, int(price), int(size), float(t)))
        elif et == PARTIAL_CANCEL or et == EXECUTE_VISIBLE:
            book.reduce_or_remove(int(oid), int(size))
        elif et == DELETE:
            book.remove(int(oid))
        # 5/6/7: hidden execution / cross trade / halt -- no visible-book effect
    return book.by_id


def aggregate_levels(book: dict[int, RestingOrder], n_levels: int = 10) -> dict:
    """Aggregate resting orders into price levels, best-first per side.

    Returns {"bids": [(price, total_size), ...], "asks": [(price, total_size), ...]},
    each truncated to n_levels, sorted best-first (bids descending, asks ascending).
    """
    bid_totals: dict[int, int] = {}
    ask_totals: dict[int, int] = {}
    for o in book.values():
        totals = bid_totals if o.side == "buy" else ask_totals
        totals[o.price] = totals.get(o.price, 0) + o.size

    bids = sorted(bid_totals.items(), key=lambda kv: -kv[0])[:n_levels]
    asks = sorted(ask_totals.items(), key=lambda kv: kv[0])[:n_levels]
    return {"bids": bids, "asks": asks}


def levels_to_lobster_row(levels: dict, n_levels: int = 10) -> np.ndarray:
    """Flatten aggregated levels into LOBSTER orderbook.csv row convention:
    [sell1, vsell1, buy1, vbuy1, sell2, vsell2, buy2, vbuy2, ...].

    Missing levels are left as NaN (this dataset never pads within 10 levels for
    a liquid symbol like INTC in practice; callers comparing against a real
    orderbook.csv row should only compare populated levels).
    """
    row = np.full(n_levels * 4, np.nan)
    for i in range(n_levels):
        if i < len(levels["asks"]):
            row[i * 4], row[i * 4 + 1] = levels["asks"][i]
        if i < len(levels["bids"]):
            row[i * 4 + 2], row[i * 4 + 3] = levels["bids"][i]
    return row
