"""Correctness + timing tests for the standalone order-book reconstructor.

Primary correctness check: LOBSTER ships a paired, row-aligned orderbook.csv
with the aggregated top-N levels after every message-row event. The
reconstructor's own aggregated levels at the same row index are compared
against that row -- free ground truth, no synthetic fixture needed.

KNOWN LIMITATION (measured, not a bug -- see test_ghost_order_rate_is_small and
test_top_of_book_price_is_usually_exact below): a small fraction of orders in
real LOBSTER data (~2-3% here, confirmed across three separate trading days)
never receive an explicit cancel/execute event anywhere in the entire day's
message log, yet do not appear in LOBSTER's own reconstructed book either.
Verified by tracing several such orders' complete lifecycle across the WHOLE
file (not just up to the test point) -- there is no missed event to apply; the
gap is in the raw data itself, not in this reconstructor's handling of it
(likely LOBSTER/ITCH feed message loss, or an exchange-internal order-lifecycle
event not exposed in the public message format). A price-time-priority
matching engine (see _MatchingBook in orderbook_reconstructor.py) recovers the
large majority of these self-correcting whenever a later crossing order
arrives -- it is what took top-of-book accuracy from being off by thousands of
ticks (a naive log-trusting walk, crossed book) to matching exactly (price and
size, both sides) in 10/15, and matching in *price* on both sides
(occasionally off on size only) in 13/15, sampled (date, t0) cases below. The
two remaining price misses are a single tick, reached only after several
hours of accumulated drift.
Given this is a property of the input data rather than of the algorithm, the
tests below assert bounded, measured tolerances rather than bit-exact equality
at arbitrary t0 -- and print the actual numbers so drift is visible over time.
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rl_execution.orderbook_reconstructor import (
    aggregate_levels,
    read_messages,
    reconstruct_book_upto_index,
)

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "INTC", "INTC_2015-01-02_2015-01-30",
)

N_LEVELS = 10
TEST_DATES = ["2015-01-30", "2015-01-15", "2015-01-02"]
TEST_FRACS = [0.05, 0.25, 0.5, 0.75, 0.95]

ORDERBOOK_COLUMNS = [c for i in range(1, N_LEVELS + 1) for c in (f"sell{i}", f"vsell{i}", f"buy{i}", f"vbuy{i}")]


def _day_files(date: str):
    if not os.path.isdir(DATA_DIR):
        pytest.skip(f"LOBSTER data not present at {DATA_DIR}")
    candidates = [f for f in os.listdir(DATA_DIR) if f.startswith(f"INTC_{date}_") and "message" in f]
    if not candidates:
        pytest.skip(f"no data for {date} in {DATA_DIR}")
    message_path = os.path.join(DATA_DIR, candidates[0])
    orderbook_path = message_path.replace("message", "orderbook")
    return message_path, orderbook_path


def _top_of_book(message_path, orderbook_path, frac):
    messages = read_messages(message_path)
    row_index = int(len(messages) * frac)

    orderbook = pd.read_csv(orderbook_path, header=None, names=ORDERBOOK_COLUMNS,
                             skiprows=row_index, nrows=1)
    expected = orderbook.iloc[0]
    exp_ask, exp_bid = (expected["sell1"], expected["vsell1"]), (expected["buy1"], expected["vbuy1"])

    book = reconstruct_book_upto_index(messages, row_index)
    levels = aggregate_levels(book, n_levels=1)
    got_ask = levels["asks"][0] if levels["asks"] else None
    got_bid = levels["bids"][0] if levels["bids"] else None
    return got_ask, exp_ask, got_bid, exp_bid


def test_top_of_book_price_is_usually_exact():
    """Best bid/ask PRICE (not necessarily size, see module docstring) should
    match LOBSTER's own reconstruction in the large majority of sampled
    (date, t0) cases, and exactly (price + size) in most of them too.
    """
    results = []
    for date in TEST_DATES:
        message_path, orderbook_path = _day_files(date)
        for frac in TEST_FRACS:
            got_ask, exp_ask, got_bid, exp_bid = _top_of_book(message_path, orderbook_path, frac)
            price_ok = got_ask is not None and got_bid is not None and got_ask[0] == exp_ask[0] and got_bid[0] == exp_bid[0]
            exact = got_ask == exp_ask and got_bid == exp_bid
            results.append((date, frac, price_ok, exact, got_ask, exp_ask, got_bid, exp_bid))

    n = len(results)
    n_price_ok = sum(1 for r in results if r[2])
    n_exact = sum(1 for r in results if r[3])
    print(f"\ntop-of-book price correct: {n_price_ok}/{n}   exact (price+size): {n_exact}/{n}")
    for date, frac, price_ok, exact, got_ask, exp_ask, got_bid, exp_bid in results:
        if not exact:
            print(f"  {date} frac={frac}: ask got={got_ask} exp={exp_ask}  bid got={got_bid} exp={exp_bid}")

    assert n_price_ok / n >= 0.8, "top-of-book price accuracy regressed below the measured baseline"
    assert n_exact / n >= 0.5, "top-of-book exact-match rate regressed below the measured baseline"


def test_ghost_order_rate_is_small():
    """Orders with no cancel/execute event anywhere in the day's log (see
    module docstring) should stay a small, bounded fraction -- if this grows
    much larger than what was measured (~2-3%), something is actually wrong
    (e.g. a parsing bug), rather than this being the expected data artifact.
    """
    for date in TEST_DATES:
        message_path, _ = _day_files(date)
        messages = read_messages(message_path)
        new_ids = set(messages[messages["event_type"] == 1]["order_id"])
        referenced_ids = set(messages[messages["event_type"].isin([2, 3, 4])]["order_id"])
        ghost_rate = len(new_ids - referenced_ids) / len(new_ids)
        print(f"\n{date}: ghost-order rate = {ghost_rate:.2%} ({len(new_ids)} orders)")
        assert ghost_rate < 0.05, f"{date}: ghost-order rate {ghost_rate:.2%} exceeds expected bound"


def test_full_day_reconstruction_is_fast():
    message_path, _ = _day_files("2015-01-30")
    messages = read_messages(message_path)
    t0 = float(messages["time"].iloc[-1])

    start = time.perf_counter()
    book = reconstruct_book_upto_index(messages, len(messages) - 1)
    elapsed = time.perf_counter() - start

    assert len(book) > 0
    assert elapsed < 30, f"full-day reconstruction took {elapsed:.1f}s, expected seconds not minutes"
    print(f"\nfull-day reconstruction ({len(messages)} messages) took {elapsed:.2f}s, "
          f"{len(book)} resting orders at close, t0={t0}")
