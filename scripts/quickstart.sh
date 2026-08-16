#!/bin/bash
# quickstart.sh — one-command setup / data / train / simulate for TRADES (HSBC reviewers).
#
# It uses a local virtualenv at ./env and calls it directly, so you never have to
# "activate" anything. Run the four steps in order:
#
#   bash scripts/quickstart.sh setup                 # create ./env and install deps
#   bash scripts/quickstart.sh data <path-to.zip>    # unpack LOBSTER data into data/INTC/
#   bash scripts/quickstart.sh train                 # preprocess + train a TRADES checkpoint
#   bash scripts/quickstart.sh simulate              # run a market simulation (DDPM, 100 steps)
#
# Defaults target INTC, 2015-01-30, 09:30-10:30 (matches the repo's config defaults, so no
# editing configuration.py is needed for INTC 2015 data). Override the sampler/steps:
#   bash scripts/quickstart.sh simulate DDPM 100
#   bash scripts/quickstart.sh simulate DDIM 10
#
# GPU strongly recommended: training on CPU is impractical; simulation runs on CPU but slowly.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Use the project venv's python if it exists, else the system python.
PY="python3"; [ -x env/bin/python ] && PY="env/bin/python"

STOCK="INTC"
DATE="20150130"
ST="09:30:00"
ET="10:30:00"

CMD="${1:-help}"; shift || true

case "$CMD" in

  setup)
    echo "Creating virtualenv at ./env and installing dependencies…"
    python3 -m venv env
    env/bin/pip install --upgrade pip
    env/bin/pip install -r requirements.txt
    echo ""
    echo "Setup complete. NOTE: requirements.txt pins a CUDA 11.8 build of torch."
    echo "  - Different CUDA / CPU-only: reinstall torch for your platform, e.g."
    echo "      env/bin/pip install torch torchvision   # CPU or default CUDA"
    echo "  - Verify GPU:  env/bin/python -c 'import torch; print(torch.cuda.is_available())'"
    ;;

  data)
    ZIP="${1:?usage: quickstart.sh data <path-to-data.zip>}"
    [ -f "$ZIP" ] || { echo "No such file: $ZIP" >&2; exit 1; }
    mkdir -p "data/$STOCK"
    echo "Unpacking $ZIP into data/$STOCK/ …"
    unzip -o "$ZIP" -d "data/$STOCK/" >/dev/null
    echo "Contents of data/$STOCK/:"
    ls -1 "data/$STOCK/"
    echo ""
    echo "Expected result: a folder  data/$STOCK/${STOCK}_2015-01-02_2015-01-30/  containing"
    echo "LOBSTER files named like  2015-01-30_34200000_57600000_message_10.csv  (message + orderbook)."
    echo "If the files landed one level too deep or shallow, move them so that path matches."
    ;;

  preprocess)
    echo "Preprocessing only (writes data/$STOCK/{train,val,test}.npy + normalization_stats.json)…"
    echo "This happens automatically on the first 'train' too; run this only to do it separately."
    $PY main.py
    ;;

  train)
    echo "Training a TRADES model (config defaults: stock=$STOCK, 50 epochs)."
    echo "On the FIRST run, configuration.py has IS_DATA_PREPROCESSED=False, so it will preprocess"
    echo "the raw LOBSTER data first (slow), then train. To skip preprocessing on later runs, set"
    echo "IS_DATA_PREPROCESSED=True in configuration.py."
    echo ""
    $PY main.py
    echo ""
    echo "Best checkpoint saved under data/checkpoints/TRADES/  (named val_ema=<loss>_epoch=<n>_…ckpt)."
    ;;

  simulate)
    TYPE="${1:-DDPM}"      # DDPM (recommended) | DDIM | DPM_SOLVER_PP | ...
    NSTEPS="${2:-100}"     # DDPM ignores this and always runs 100; DDIM/DPM use it
    ID="${3:-}"            # optional: checkpoint val-loss id, e.g. 0.681. Empty = best available.
    [ -d data/checkpoints/TRADES ] && ls data/checkpoints/TRADES/*.ckpt >/dev/null 2>&1 || {
        echo "No checkpoint in data/checkpoints/TRADES/. Train one first (quickstart.sh train)," >&2
        echo "or drop a TRADES checkpoint there." >&2; exit 1; }
    ARGS=(-u ABIDES/abides.py -c world_agent_sim -t "$STOCK" -date "$DATE"
          -st "$ST" -et "$ET" -d True -m TRADES -type "$TYPE" -nsteps "$NSTEPS")
    [ -n "$ID" ] && ARGS+=(-id "$ID")
    echo "Running simulation: $TYPE, $NSTEPS steps…"
    echo "  $PY ${ARGS[*]}"
    $PY "${ARGS[@]}"
    echo ""
    echo "Output (orders CSV + stylized-fact plots) is under ABIDES/log/world_agent_${STOCK}_…/"
    ;;

  help|*)
    sed -n '2,25p' "$0"
    ;;
esac
