#!/bin/bash
# ============================================================
#
# cd /scratch0/$USER
# curl -O https://raw.githubusercontent.com/Matt-McCrea/HSBC/main/setup_scratch.sh
#
# setup_scratch.sh — provision a fresh UCL TSG GPU workstation
# for DeepMarket/TRADES training.
#
# Scratch (/scratch0/$USER) is WIPED at session end and on the
# first Wednesday of each month. This script rebuilds everything
# that lives on scratch (venv, caches) and clones your repo fresh.
#
# USAGE (from anywhere):
#   bash /path/to/setup_scratch.sh
#
# Then activate the env with the line it prints at the end.
# ============================================================

set -e  # stop on first error

# ---- EDIT THESE TWO LINES IF NEEDED -------------------------
REPO_URL="https://github.com/Matt-McCrea/HSBC.git"
BRANCH="${1:-main}"   # pass branch as first arg, e.g. bash setup_scratch.sh baseline
# -------------------------------------------------------------

SCRATCH="/scratch0/mmccrea"
REPO_DIR="$SCRATCH/HSBC"
VENV_DIR="$SCRATCH/dmenv"

echo "=== DeepMarket scratch setup for $USER ==="
echo "Scratch base: $SCRATCH"

# 1. Make scratch dirs for venv + redirected caches
mkdir -p "$SCRATCH"
mkdir -p "$SCRATCH/pip-cache" "$SCRATCH/mplconfig" "$SCRATCH/hf" "$SCRATCH/torch" "$SCRATCH/.cache"

# 2. Clone the repo fresh (your data is committed, so it comes too)
if [ -d "$REPO_DIR/.git" ]; then
    echo "Repo already present, pulling latest..."
    git -C "$REPO_DIR" fetch origin
    git -C "$REPO_DIR" checkout "$BRANCH"
    git -C "$REPO_DIR" pull origin "$BRANCH"
else
    echo "Cloning $REPO_URL ($BRANCH)..."
    git clone --branch "$BRANCH" "$REPO_URL" "$REPO_DIR"
fi

# 3. Build the virtualenv on scratch
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating venv at $VENV_DIR..."
    python -m venv "$VENV_DIR"
fi

# 4. Install requirements (cache on scratch, never touch home quota)
echo "Installing requirements..."
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
export PIP_CACHE_DIR="$SCRATCH/pip-cache"
pip install --upgrade pip
pip install --no-cache-dir -r "$REPO_DIR/requirements.txt"

# 5. Write an env file you can source in every new terminal
ENVFILE="$SCRATCH/env.sh"
cat > "$ENVFILE" <<EOF
# Source this in any new bash terminal: source $ENVFILE
source $VENV_DIR/bin/activate
export PIP_CACHE_DIR=$SCRATCH/pip-cache
export MPLCONFIGDIR=$SCRATCH/mplconfig
export HF_HOME=$SCRATCH/hf
export TORCH_HOME=$SCRATCH/torch
export XDG_CACHE_HOME=$SCRATCH/.cache
cd $REPO_DIR
EOF

# csh/tcsh version too, since the TSG login shell is tcsh
ENVFILE_CSH="$SCRATCH/env.csh"
cat > "$ENVFILE_CSH" <<EOF
# Source this in any new tcsh terminal: source $ENVFILE_CSH
source $VENV_DIR/bin/activate.csh
setenv PIP_CACHE_DIR $SCRATCH/pip-cache
setenv MPLCONFIGDIR $SCRATCH/mplconfig
setenv HF_HOME $SCRATCH/hf
setenv TORCH_HOME $SCRATCH/torch
setenv XDG_CACHE_HOME $SCRATCH/.cache
cd $REPO_DIR
EOF

echo ""
echo "=== Setup complete ==="
echo ""
echo "GPU check:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "  (nvidia-smi not available)"
echo ""
echo "To start working in THIS or any new terminal:"
echo "    bash:        source $ENVFILE"
echo "    tcsh:        source $ENVFILE_CSH"
echo ""
echo "Then run training with:   python main.py"
echo ""
echo "!! REMEMBER: scratch is wiped at session end. Before you log off:"
echo "   - git commit + push any code changes"
echo "   - copy checkpoints you want to keep to ~/  (home survives)"
echo ""
