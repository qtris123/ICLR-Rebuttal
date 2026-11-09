#!/bin/bash
set -euo pipefail

SESSION="diag_2x2"
LOG_DIR="/home/voquangtri2025/ICLR-Rebuttal/outputs/diag/2x2_cross_read"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/run.log"

# Kill existing session if present
tmux kill-session -t "$SESSION" 2>/dev/null || true

# Start new detached tmux session
tmux new-session -d -s "$SESSION" -c "/home/voquangtri2025/ICLR-Rebuttal"

# Send command to run the 2x2 investigation
tmux send-keys -t "$SESSION" "conda activate cartridges" C-m
tmux send-keys -t "$SESSION" "CUDA_VISIBLE_DEVICES=0 python3 /home/voquangtri2025/ICLR-Rebuttal/run_2x2_cross_read.py 2>&1 | tee '$LOG_FILE'" C-m

echo "Started tmux session: $SESSION"
echo "Log file: $LOG_FILE"

