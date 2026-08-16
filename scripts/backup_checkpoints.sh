#!/bin/bash
# backup_checkpoints.sh — watch data/checkpoints/TRADES during a training run and back
# up every new checkpoint (+ a quick flow-mix health check) to a persistent location,
# since the SSH session won't be around to babysit an overnight run.
#
# WHY THIS EXISTS: diffusion_engine.py's model_checkpointing() (models/diffusers/
# diffusion_engine.py:317-328) DELETES the previous best checkpoint every time a new
# epoch improves on val_ema_loss (os.remove(self.last_path_ckpt_ema)), then saves the
# new one. Only ONE .ckpt file ever exists in data/checkpoints/TRADES at a time. If a
# later epoch happens to look worse afterwards (as observed: loss just fluctuates after
# epoch 1), you can't go back and inspect the earlier "best" once it's overwritten.
# This script polls the directory and copies out every new checkpoint the instant it
# appears, before the next improvement can delete it.
#
# For each new checkpoint it also runs a short evaluation sim + flow_mix.py so you can
# judge checkpoints by simulation-level realism (flow mix, unique mid count) rather
# than val_ema_loss alone, which we've found doesn't track simulation quality.
#
# Usage (run from anywhere; survives SSH logout):
#   source /scratch0/mmccrea/env.sh   # activate venv, cd into repo (if you use this pattern)
#   nohup bash scripts/backup_checkpoints.sh > /tmp/backup_checkpoints.log 2>&1 &
#   disown
#   tail -f /tmp/backup_checkpoints.log
#
# Stop early:
#   pkill -f backup_checkpoints.sh

set -uo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT_DIR="$REPO_DIR/data/checkpoints/TRADES"

DEST_DIR="/cs/student/msc/cf/2025/mmccrea/Documents/trades_checkpoint_backups"
CKPT_DEST="$DEST_DIR/checkpoints"
REPORT_DEST="$DEST_DIR/flow_mix_reports"
SEEN_FILE="$DEST_DIR/.seen_checkpoints.txt"

POLL_SECONDS=60          # how often to check for a new checkpoint
MAX_RUNTIME_HOURS=14     # safety net so this doesn't run forever unattended

# Quick eval-sim settings: kept short (15min replay seed + 5min generation) so the
# analysis step doesn't lag far behind and risk missing the next checkpoint rotation.
TICKER="INTC"
DATE="20150130"
EVAL_START="09:30:00"
EVAL_END="09:50:00"
SAMPLER_TYPE="DDIM"
SAMPLER_NSTEPS=10
REAL_DATA_PATH="$REPO_DIR/ABIDES/log/market_replay_INTC_2015-01-30_10-30-00_30/processed_orders.csv"

# ── Setup ─────────────────────────────────────────────────────────────────────
cd "$REPO_DIR"

if ! mkdir -p "$CKPT_DEST" "$REPORT_DEST"; then
    echo "[FATAL] Could not create $DEST_DIR — is it mounted/writable from here?" >&2
    exit 1
fi
if ! touch "$DEST_DIR/.write_test" 2>/dev/null; then
    echo "[FATAL] $DEST_DIR is not writable." >&2
    exit 1
fi
rm -f "$DEST_DIR/.write_test"
touch "$SEEN_FILE"

echo "=== TRADES checkpoint backup monitor ==="
echo "PID          : $$"
echo "Watching     : $CKPT_DIR"
echo "Backing up to: $DEST_DIR"
echo "Poll interval: ${POLL_SECONDS}s | Max runtime: ${MAX_RUNTIME_HOURS}h"
echo "Stop with    : pkill -f backup_checkpoints.sh"
echo ""

START_TS=$(date +%s)
MAX_RUNTIME_SECONDS=$((MAX_RUNTIME_HOURS * 3600))

# ── Main loop ─────────────────────────────────────────────────────────────────
while true; do
    NOW_TS=$(date +%s)
    if (( NOW_TS - START_TS > MAX_RUNTIME_SECONDS )); then
        echo "[$(date '+%F %T')] Max runtime (${MAX_RUNTIME_HOURS}h) reached, exiting."
        break
    fi

    if [[ -d "$CKPT_DIR" ]]; then
        for ckpt in "$CKPT_DIR"/*.ckpt; do
            [[ -e "$ckpt" ]] || continue
            fname=$(basename "$ckpt")

            if ! grep -qxF "$fname" "$SEEN_FILE" 2>/dev/null; then
                echo "[$(date '+%F %T')] New checkpoint: $fname"

                # 1. Back up immediately — this file can be deleted the instant the
                #    NEXT epoch improves, so copy before doing anything else.
                if cp "$ckpt" "$CKPT_DEST/$fname" 2>/dev/null; then
                    echo "  -> backed up to $CKPT_DEST/$fname"
                    echo "$fname" >> "$SEEN_FILE"
                else
                    echo "  -> [WARN] copy failed (checkpoint may have rotated out already), skipping"
                    continue
                fi

                # 2. Extract val_ema loss / epoch from the filename
                #    (format: val_ema=<loss>_epoch=<n>_<rest>.ckpt)
                val_id=$(echo "$fname" | sed -E 's/^val_ema=([0-9.]+)_.*/\1/')
                epoch=$(echo "$fname" | sed -E 's/.*_epoch=([0-9]+)_.*/\1/')

                # 3. Short eval sim + flow_mix.py (best-effort: if the live checkpoint
                #    dir has already rotated past this one, the sim will fail to find
                #    it — that's fine, the .ckpt backup from step 1 is already safe).
                report="$REPORT_DEST/epoch${epoch}_val${val_id}_flow_mix.txt"
                echo "  -> running quick eval sim (epoch=$epoch, val_ema=$val_id)..."
                {
                    echo "Checkpoint : $fname"
                    echo "Epoch      : $epoch"
                    echo "val_ema    : $val_id"
                    echo "Sim window : $EVAL_START - $EVAL_END, sampler $SAMPLER_TYPE nsteps=$SAMPLER_NSTEPS"
                    echo ""
                } > "$report"

                SENTINEL=$(mktemp)
                if python ABIDES/abides.py -c world_agent_sim -t "$TICKER" -date "$DATE" \
                        -st "$EVAL_START" -et "$EVAL_END" -d True \
                        -type "$SAMPLER_TYPE" -nsteps "$SAMPLER_NSTEPS" -id "$val_id" \
                        >> "$report" 2>&1; then

                    gen_csv=$(find ABIDES/log -name "processed_orders.csv" -newer "$SENTINEL" \
                                   ! -path "*/paper/*" 2>/dev/null | sort | tail -1)
                    if [[ -n "$gen_csv" ]]; then
                        {
                            echo ""
                            echo "=== flow_mix ==="
                        } >> "$report"
                        if [[ -f "$REAL_DATA_PATH" ]]; then
                            python -m evaluation.quantitative_eval.flow_mix --real "$REAL_DATA_PATH" --gen "$gen_csv" >> "$report" 2>&1
                        else
                            python -m evaluation.quantitative_eval.flow_mix --gen "$gen_csv" >> "$report" 2>&1
                        fi
                    else
                        echo "[WARN] no generated CSV found after sim" >> "$report"
                    fi
                else
                    echo "[ERROR] eval sim failed (checkpoint may have rotated out mid-run — .ckpt backup is still safe)" >> "$report"
                fi
                rm -f "$SENTINEL"

                echo "  -> report saved: $report"
            fi
        done
    fi

    sleep "$POLL_SECONDS"
done

echo "[$(date '+%F %T')] Monitor stopped."
