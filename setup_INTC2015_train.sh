#!/bin/bash
# setup_INTC2015_train.sh — link INTC 2015 LOB data, restore checkpoints, and launch training.
#
# USAGE (from inside repo after sourcing env.sh):
#   bash setup_INTC2015_train.sh [/path/to/raw/LOBSTER/data]
#
# Raw LOBSTER CSVs are expected to contain message + orderbook pairs for
# the date range 2015-01-02 to 2015-01-30.
#
# Default data source: ~/INTC_data/INTC_2015-01-02_2015-01-30/
# Default checkpoint backup location: ~/TRADES_checkpoints/
# (home survives session end; scratch does not)

set -e

SCRATCH="${SCRATCH:-/scratch0/$(id -un)}"
REPO_DIR="$SCRATCH/HSBC"
DATA_SOURCE="${1:-$HOME/INTC_data/INTC_2015-01-02_2015-01-30}"
DATA_DEST="$REPO_DIR/data/INTC/INTC_2015-01-02_2015-01-30"
CKPT_DIR="$REPO_DIR/data/checkpoints/TRADES"
CKPT_HOME="$HOME/TRADES_checkpoints"

echo "=== INTC 2015 training setup ==="
echo "Branch: $(git -C "$REPO_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"

# 1. Verify source data exists
if [ ! -d "$DATA_SOURCE" ]; then
    echo ""
    echo "ERROR: Raw LOB data not found at:"
    echo "  $DATA_SOURCE"
    echo ""
    echo "Copy your LOBSTER CSVs there first:"
    echo "  mkdir -p $DATA_SOURCE"
    echo "  scp yourlocal/*.csv <host>:$DATA_SOURCE/"
    exit 1
fi

# 2. Link raw data into the repo data directory
mkdir -p "$DATA_DEST"
if [ -z "$(ls -A "$DATA_DEST" 2>/dev/null)" ]; then
    echo "Linking data: $DATA_SOURCE -> $DATA_DEST"
    ln -sf "$DATA_SOURCE"/*.csv "$DATA_DEST/"
else
    echo "Data already present in $DATA_DEST"
fi

# 3. Restore any checkpoints previously backed up to home
mkdir -p "$CKPT_DIR"
if ls "$CKPT_HOME"/*.ckpt 1>/dev/null 2>&1; then
    echo "Restoring checkpoints from $CKPT_HOME ..."
    cp "$CKPT_HOME"/*.ckpt "$CKPT_DIR/"
else
    echo "No prior checkpoints found in $CKPT_HOME, starting from scratch."
fi

# 4. Train
echo ""
echo "Starting training (IS_DATA_PREPROCESSED=False -> will preprocess on first run)..."
cd "$REPO_DIR"
python main.py

# 5. Back up checkpoints to home before session ends
mkdir -p "$CKPT_HOME"
if ls "$CKPT_DIR"/*.ckpt 1>/dev/null 2>&1; then
    echo ""
    echo "Backing up checkpoints to $CKPT_HOME ..."
    cp "$CKPT_DIR"/*.ckpt "$CKPT_HOME/"
    echo "Done. Safe to log off — checkpoints are in home."
else
    echo "No checkpoints found to back up."
fi
