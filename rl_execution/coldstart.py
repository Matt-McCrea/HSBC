"""Cold-start / context-injection for RL episodes.

Given a seed timestamp t0, produces everything needed to start a TRADES/ABIDES
episode at t0 without the expensive ~15-minute real-order-flow replay through
the ABIDES kernel:

  1. The exact resting order book at t0 (via rl_execution.orderbook_reconstructor),
     loaded directly into the ABIDES exchange's OrderBook.
  2. The model's conditioning tensors (the preceding N-1 real orders and N real
     LOB snapshots ending at t0), built as a direct slice of the raw message/
     orderbook log -- not a live replay -- using the *exact* z-score
     normalisation and price re-anchoring already built for WorldAgent
     (utils/utils_data.py:preprocess_orders_for_diff_cond /
     z_score_orderbook_for_cond, extracted from WorldAgent so this module and
     live simulation cannot silently diverge).

Windowing convention (must match what the model was actually trained on, in
utils/utils_data.py:preprocess_data -- NOT WorldAgent's replay-specific
convention, which differs -- see build_conditioning_window's docstring).
"""

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

import constants as cst
from rl_execution.orderbook_reconstructor import (
    DEFAULT_MARKET_OPEN,
    MESSAGE_COLUMNS,
    aggregate_levels,
    reconstruct_book,
)
from utils.utils_data import compute_price_anchor, preprocess_orders_for_diff_cond, z_score_orderbook_for_cond

# Event types the model was actually trained on conditioning against (matches
# utils/utils_data.py:preprocess_data, which drops 2/5/6/7 -- NOT WorldAgent's
# replay filter, which keeps 2). Getting this wrong silently shifts the
# conditioning distribution the model sees at seed time.
TRAINING_EVENT_TYPES = (1, 3, 4)

ORDERBOOK_COLUMNS = [c for i in range(1, 11) for c in (f"sell{i}", f"vsell{i}", f"buy{i}", f"vbuy{i}")]


def _day_paths(data_dir, symbol, date):
    """date: 'YYYY-MM-DD' string, matching the LOBSTER filename convention."""
    day_dir = None
    for cand in os.listdir(os.path.join(data_dir, symbol)):
        full = os.path.join(data_dir, symbol, cand)
        if os.path.isdir(full) and cand.startswith(f"{symbol}_"):
            day_dir = full
            break
    if day_dir is None:
        raise FileNotFoundError(f"no {symbol} data directory under {data_dir}")
    message_path = orderbook_path = None
    for fname in os.listdir(day_dir):
        if not fname.startswith(f"{symbol}_{date}_"):
            continue
        if "message" in fname:
            message_path = os.path.join(day_dir, fname)
        elif "orderbook" in fname:
            orderbook_path = os.path.join(day_dir, fname)
    if message_path is None or orderbook_path is None:
        raise FileNotFoundError(f"no message/orderbook pair for {symbol} {date} under {day_dir}")
    return message_path, orderbook_path


def read_day(message_path, orderbook_path, n_lob_levels=10):
    messages = pd.read_csv(message_path, header=None, names=MESSAGE_COLUMNS, usecols=range(6))
    orderbook = pd.read_csv(orderbook_path, header=None, names=ORDERBOOK_COLUMNS,
                             usecols=range(n_lob_levels * 4))
    return messages, orderbook


def build_conditioning_window(messages: pd.DataFrame, orderbook: pd.DataFrame, t0: float,
                               cond_seq_size: int = 255, seq_len: int = 256):
    """Slice the last `seq_len` real (order, LOB-snapshot) pairs ending at t0,
    filtered and windowed exactly as utils/utils_data.py:preprocess_data does
    for training (drop event_type 2/5/6/7, then treat consecutive *filtered*
    rows as adjacent for the time-diff and pre-event-depth computation -- i.e.
    dropped rows do not "count" as intervening book changes, matching what the
    model actually learned from, not naive raw-adjacency).

    Returns (orders_raw, lob_raw): orders_raw is [cond_seq_size, 6]
    (time-as-diff, event_type, order_id, size, price, direction), lob_raw is
    [seq_len, 40] raw LOBSTER units. Both pure numpy/pandas -- no torch.
    """
    if len(messages) != len(orderbook):
        raise ValueError(f"message/orderbook row count mismatch: {len(messages)} vs {len(orderbook)}")

    mask = messages["event_type"].isin(TRAINING_EVENT_TYPES) & (messages["time"] <= t0)
    idx = np.flatnonzero(mask.to_numpy())
    if len(idx) < seq_len:
        raise ValueError(
            f"only {len(idx)} eligible real events at or before t0={t0}, need {seq_len} "
            "for a full conditioning window -- pick a later t0")
    window_idx = idx[-seq_len:]

    lob_raw = orderbook.iloc[window_idx].to_numpy(dtype=float)

    order_rows = messages.iloc[window_idx[1:]].copy()
    times = messages["time"].to_numpy(dtype=float)[window_idx]
    order_rows["time"] = times[1:] - times[:-1]  # diff against the immediately preceding *filtered* row
    orders_raw = order_rows.to_numpy(dtype=float)

    return orders_raw, lob_raw


@dataclass
class ColdStartResult:
    t0: float
    resting_orders: dict
    orders_raw: np.ndarray
    lob_raw: np.ndarray
    cond_orders: "torch.Tensor"
    cond_lob: "torch.Tensor"
    cond_stats: dict
    price_anchor: float


def seed_episode(message_path, orderbook_path, t0: float, normalization_terms,
                  cond_seq_size: int = 255, seq_len: int = 256,
                  market_open: float = DEFAULT_MARKET_OPEN, cond_clip: float = 0.0,
                  fix_lob_pad: bool = False) -> ColdStartResult:
    """Build everything needed to seed an RL episode at t0: the reconstructed
    resting book plus the model's conditioning tensors. Does not touch ABIDES
    or the model -- see rl_execution/env.py for wiring this into a live Kernel
    (load the resting_orders into the exchange's OrderBook via enterOrder, and
    pass cond_orders/cond_lob straight to RLWorldAgent).
    """
    messages, orderbook = read_day(message_path, orderbook_path)

    price_anchor = 0.0
    if cst.PRICE_REANCHOR:
        price_anchor = float(compute_price_anchor(orderbook))

    resting_orders = reconstruct_book(message_path, t0, market_open=market_open)

    orders_raw, lob_raw = build_conditioning_window(
        messages, orderbook, t0, cond_seq_size=cond_seq_size, seq_len=seq_len)

    cond_stats: dict = {}
    cond_orders = preprocess_orders_for_diff_cond(
        orders_raw, lob_raw, normalization_terms, price_anchor=price_anchor, cond_stats=cond_stats)
    cond_lob_raw, clipped = z_score_orderbook_for_cond(
        lob_raw.copy(), normalization_terms, price_anchor=price_anchor,
        fix_lob_pad=fix_lob_pad, cond_clip=cond_clip)
    import torch  # local import: keeps this module importable without torch for the pure-numpy path
    cond_lob = torch.from_numpy(cond_lob_raw).to(cst.DEVICE, torch.float32)

    print(f"[coldstart] seed t0={t0}  resting_orders={len(resting_orders)}  "
          f"cond_clipped={clipped}  price_anchor={price_anchor}")
    for feat, s in sorted(cond_stats.items()):
        mean = s[2] / s[3] if s[3] else float("nan")
        print(f"[coldstart] DIAG cond_z[{feat}]: min={s[0]:.2f} mean={mean:.2f} max={s[1]:.2f} n={s[3]}")

    return ColdStartResult(
        t0=t0, resting_orders=resting_orders, orders_raw=orders_raw, lob_raw=lob_raw,
        cond_orders=cond_orders, cond_lob=cond_lob, cond_stats=cond_stats, price_anchor=price_anchor,
    )


def seed_exchange_book(exchange, symbol: str, resting_orders: dict, session_date):
    """Load reconstructed resting orders directly into the ABIDES exchange's
    OrderBook via the existing OrderBook.enterOrder (no matching -- these are
    already-resting, non-crossing orders by construction) -- avoids the
    ~15-minute real-order-flow replay through the full ABIDES kernel entirely.

    exchange: an ABIDES ExchangeAgent instance (already constructed, book empty).
    session_date: pandas.Timestamp for the trading day (midnight), used to
        convert each order's LOBSTER seconds-since-midnight entry_time into an
        absolute timestamp for LimitOrder.time_placed.
    """
    from util.order.LimitOrder import LimitOrder

    book = exchange.order_books[symbol]
    ordered = sorted(resting_orders.values(), key=lambda o: o.entry_time)
    for o in ordered:
        limit_order = LimitOrder(
            agent_id=exchange.id,
            time_placed=session_date + pd.Timedelta(seconds=o.entry_time),
            symbol=symbol,
            quantity=o.size,
            is_buy_order=(o.side == "buy"),
            limit_price=o.price,
            order_id=o.order_id,
        )
        book.enterOrder(limit_order)
    return book
