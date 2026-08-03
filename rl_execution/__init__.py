"""RL optimal-execution environment on top of TRADES/ABIDES.

This package imports ABIDES-internal modules (Kernel, agent.*, util.*,
message.*) using their bare names (e.g. `from Kernel import Kernel`), matching
the convention used throughout ABIDES/ itself. Normally that only resolves
when going through ABIDES/abides.py's own bootstrap (`sys.path.insert(0,
os.getcwd())`, run with cwd=repo root) *combined with* an old-style editable
install that puts the whole ABIDES/ directory on sys.path. rl_execution is
imported directly (`python -m rl_execution.train`, etc.), bypassing that
bootstrap entirely -- so it's done here instead, once, at package-import
time, for every entry point in this package.
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ABIDES_DIR = os.path.join(_REPO_ROOT, "ABIDES")

for _p in (_REPO_ROOT, _ABIDES_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
