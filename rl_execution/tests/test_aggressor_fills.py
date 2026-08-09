"""The aggressor must be told when its own order executes.

Upstream ABIDES notifies only the RESTING side of a match; the line notifying the
aggressor is commented out in OrderBook.handleLimitOrder. Any agent that tracks its
own remaining inventory from executions therefore never learns that its market order
filled -- it believes it still holds the stock, so the terminal sweep appears to do
nothing and the episode ends with inventory unsold. That failure is especially nasty
because unsold inventory SHRINKS reported shortfall, so it reads as good performance.

Observed live: three preflight episodes left 2407 / 1334 / 1688 shares unsold.

These tests drive a real OrderBook, so they cover the actual matching path rather
than a mock of it, and they pin both halves of the contract: tagged orders notify the
aggressor, untagged ones behave exactly as before (which is what keeps WorldAgent's
conditioning -- and every previously recorded run -- unchanged).
"""

import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "ABIDES"))

import pandas as pd

from util.OrderBook import NOTIFY_AGGRESSOR_TAGS, OrderBook
from util.order.LimitOrder import LimitOrder
from util.order.MarketOrder import MarketOrder

from rl_execution.execution_agent import AGGRESSOR_TAG

RESTING_AGENT, AGGRESSOR_AGENT = 1, 2
NOW = pd.Timestamp("2015-01-30 10:00:00")


class FakeExchange:
    """Minimal stand-in for ExchangeAgent: OrderBook only needs these."""

    def __init__(self):
        self.currentTime = NOW
        self.messages = []          # (recipient_id, msg_type, order)
        self.book_freq = None
        self.stream_history = 10
        self.log_orders = False

    def sendMessage(self, recipient, msg):
        self.messages.append((recipient, msg.body["msg"], msg.body.get("order")))

    def publishOrderBookData(self):
        pass

    def logEvent(self, *a, **k):
        pass


def _book_with_resting_liquidity():
    owner = FakeExchange()
    book = OrderBook(owner, "INTC")
    # Resting BUY interest for an incoming sell to hit.
    for price, qty in ((340000, 300), (339900, 300), (339800, 400)):
        book.enterOrder(LimitOrder(RESTING_AGENT, NOW, "INTC", qty, True, price,
                                   order_id=int(price)))
    return owner, book


def _executions_for(owner, agent_id):
    return [o for rid, kind, o in owner.messages if rid == agent_id and kind == "ORDER_EXECUTED"]


def test_tagged_market_order_notifies_the_aggressor():
    """The case that broke the live run: a market order fills but the sender is
    never told, so its remaining inventory never decreases."""
    owner, book = _book_with_resting_liquidity()
    book.handleMarketOrder(MarketOrder(AGGRESSOR_AGENT, NOW, "INTC", 800, False,
                                       order_id=999, tag=AGGRESSOR_TAG))

    fills = _executions_for(owner, AGGRESSOR_AGENT)
    assert fills, "aggressor was never told its market order executed"
    assert sum(f.quantity for f in fills) == 800, "aggressor should learn about the full fill"
    # the resting side must still be notified, exactly as before
    assert _executions_for(owner, RESTING_AGENT)


def test_tagged_crossing_limit_order_notifies_the_aggressor():
    owner, book = _book_with_resting_liquidity()
    book.handleLimitOrder(LimitOrder(AGGRESSOR_AGENT, NOW, "INTC", 500, False, 339900,
                                     order_id=1234, tag=AGGRESSOR_TAG))
    fills = _executions_for(owner, AGGRESSOR_AGENT)
    assert sum(f.quantity for f in fills) == 500


def test_untagged_orders_are_unchanged():
    """WorldAgent's orders carry no tag. If they started receiving aggressor
    notifications it would append extra type-4 events into the order history it
    conditions the diffusion model on, changing generation and invalidating
    comparison against every previously recorded run."""
    owner, book = _book_with_resting_liquidity()
    book.handleMarketOrder(MarketOrder(AGGRESSOR_AGENT, NOW, "INTC", 500, False, order_id=7))
    assert _executions_for(owner, AGGRESSOR_AGENT) == [], \
        "untagged aggressor must NOT be notified -- this is what protects WorldAgent"
    assert _executions_for(owner, RESTING_AGENT), "resting side must still be notified"


def test_partial_fill_reports_only_what_actually_traded():
    """Only 1000 shares of resting interest exist; a 1500-share sell must report 1000
    filled, not 1500, or remaining inventory would be understated."""
    owner, book = _book_with_resting_liquidity()
    book.handleMarketOrder(MarketOrder(AGGRESSOR_AGENT, NOW, "INTC", 1500, False,
                                       order_id=42, tag=AGGRESSOR_TAG))
    assert sum(f.quantity for f in _executions_for(owner, AGGRESSOR_AGENT)) == 1000


def test_place_market_order_forwards_its_tag():
    """TradingAgent.placeMarketOrder accepted a tag and dropped it before constructing
    the MarketOrder, which silently removed the opt-in for the one order type that is
    always the aggressor."""
    order = MarketOrder(AGGRESSOR_AGENT, NOW, "INTC", 100, False, 5, AGGRESSOR_TAG)
    assert order.tag == AGGRESSOR_TAG

    import inspect
    from agent.TradingAgent import TradingAgent
    src = inspect.getsource(TradingAgent.placeMarketOrder)
    assert "is_buy_order, order_id, tag)" in src, \
        "placeMarketOrder must forward its tag to MarketOrder"


def test_agent_tag_matches_the_book_gate():
    """Two constants in two files have to agree; if they drift apart the notification
    silently stops and inventory stops liquidating again."""
    assert AGGRESSOR_TAG in NOTIFY_AGGRESSOR_TAGS
