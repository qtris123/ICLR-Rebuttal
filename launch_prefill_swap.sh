#!/bin/bash
set -euo pipefail

SESSION="diag_prefill_swap"
LOG_DIR="/home/voquangtri2025/ICLR-Rebuttal/outputs/diag/prefill_swap_ablation"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/run.log"

# Kill existing session if present
tmux kill-session -t "$SESSION" 2>/dev/null || true

# Start new detached tmux session
tmux new-session -d -s "$SESSION" -c "/home/voquangtri2025/ICLR-Rebuttal"

# Send command to run the ablation
tmux send-keys -t "$SESSION" "conda activate cartridges" C-m
tmux send-keys -t "$SESSION" "CUDA_VISIBLE_DEVICES=0 python3 /home/voquangtri2025/ICLR-Rebuttal/run_prefill_swap_ablation.py 2>&1 | tee '$LOG_FILE'" C-m

echo "Started tmux session: $SESSION"
echo "Log file: $LOG_FILE"

