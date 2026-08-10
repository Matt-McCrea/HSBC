"""Order-id bands must stay disjoint: real LOBSTER ids, world-agent ids, execution-agent ids."""
import os, sys
_ROOT = "/Users/Matthew/HSBC"
sys.path.insert(0, _ROOT); sys.path.insert(0, os.path.join(_ROOT, "ABIDES"))

from rl_execution.rl_world_agent import (WORLD_ORDER_ID_BASE, WORLD_ORDER_ID_POOL,
                                          fresh_order_id_pool)
from rl_execution.execution_agent import EXEC_ORDER_ID_BASE

# Measured across the 20 INTC days actually in data/ (usecols=range(6); the files carry a
# 7th ticker column that silently shifts every column if it is not excluded).
REAL_ID_MAX = 410_586_408


def test_world_pool_sits_above_every_real_id():
    assert WORLD_ORDER_ID_BASE > REAL_ID_MAX


def test_world_pool_cannot_reach_the_execution_agent_band():
    assert WORLD_ORDER_ID_BASE + WORLD_ORDER_ID_POOL <= EXEC_ORDER_ID_BASE


def test_pool_is_large_enough_for_an_episode():
    # A 5-minute episode carries ~12k real messages; the world agent consumes one id per
    # order it creates. 1e6 leaves ~80x headroom.
    assert WORLD_ORDER_ID_POOL >= 100_000


def test_pool_is_contiguous_and_starts_at_the_base():
    pool = fresh_order_id_pool()
    assert pool[0] == WORLD_ORDER_ID_BASE
    assert len(pool) == WORLD_ORDER_ID_POOL
