#!/bin/bash
# fix_visualisations.sh
# Fixes all remaining visualisation issues in DeepMarket
# Usage: bash fix_visualisations.sh /path/to/DeepMarket

REPO=${1:-"."}

python3 - << EOF
import os
import re

repo = "$REPO"

# ── Fix 1: ORDER_EXECUTED string → integer 3 ─────────────────────────────────
order_executed_files = [
    "ABIDES/realism/impact_single_day_pov.py",
    "ABIDES/realism/market_impact/abm_market_impact.py",
    "ABIDES/realism/market_impact/marketreplay_market_impact.py",
    "ABIDES/realism/realism_utils.py",
]
for rel in order_executed_files:
    path = os.path.join(repo, rel)
    if not os.path.exists(path):
        print(f"SKIP (not found): {rel}")
        continue
    with open(path, 'r') as f:
        content = f.read()
    new = content.replace('"ORDER_EXECUTED"', '3').replace("'ORDER_EXECUTED'", '3')
    if new != content:
        with open(path, 'w') as f:
            f.write(new)
        print(f"✅ Fixed ORDER_EXECUTED: {rel}")
    else:
        print(f"ℹ️  No ORDER_EXECUTED found: {rel}")

# ── Fix 2: add Agg backend to files missing it ───────────────────────────────
needs_backend = [
    "ABIDES/realism/realism_utils.py",
    "ABIDES/util/plotting/chart_fundamental.py",
    "ABIDES/util/simulation_run_stats.py",
]
for rel in needs_backend:
    path = os.path.join(repo, rel)
    if not os.path.exists(path):
        print(f"SKIP (not found): {rel}")
        continue
    with open(path, 'r') as f:
        content = f.read()
    if 'matplotlib.use' in content:
        print(f"ℹ️  Already has backend: {rel}")
        continue
    # Insert after first matplotlib import line
    new = re.sub(
        r'(import matplotlib(?:\.[^\n]*)?\n)',
        r'\1matplotlib.use("Agg")  # headless backend - no display needed\n',
        content,
        count=1
    )
    if new != content:
        with open(path, 'w') as f:
            f.write(new)
        print(f"✅ Added Agg backend: {rel}")
    else:
        print(f"⚠️  Could not add backend (no import matplotlib line): {rel}")

# ── Fix 3: DataFrame.append() in marketreplay_market_impact.py ───────────────
path = os.path.join(repo, "ABIDES/realism/market_impact/marketreplay_market_impact.py")
if os.path.exists(path):
    with open(path, 'r') as f:
        content = f.read()
    # Replace df.append(pd.Series(...), ignore_index=True) with pd.concat
    new = re.sub(
        r'(\w+)\s*=\s*\1\.append\(pd\.Series\(data=(.*?)\),\s*ignore_index=True\)',
        r'\1 = pd.concat([\1, pd.DataFrame([{\2}])], ignore_index=True)',
        content,
        flags=re.DOTALL
    )
    if new != content:
        with open(path, 'w') as f:
            f.write(new)
        print(f"✅ Fixed DataFrame.append(): marketreplay_market_impact.py")
    else:
        print(f"ℹ️  No DataFrame.append() pattern matched in marketreplay_market_impact.py")

print("\nAll visualisation fixes applied!")
EOF
