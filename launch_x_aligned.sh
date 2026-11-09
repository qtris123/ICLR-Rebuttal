#!/usr/bin/env bash
set -e

SESSION_NAME="diag_x_aligned"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/outputs/diag/x_aligned_ablation"
mkdir -p "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/run.log"

echo "=== Launching X_aligned RoPE Ablation ==="
echo "Tmux session: ${SESSION_NAME}"
echo "Log file:     ${LOG_FILE}"

tmux kill-session -t "${SESSION_NAME}" 2>/dev/null || true
tmux new-session -d -s "${SESSION_NAME}" -c "${SCRIPT_DIR}"

tmux send-keys -t "${SESSION_NAME}" "source /home/voquangtri2025/miniconda3/etc/profile.d/conda.sh" C-m
tmux send-keys -t "${SESSION_NAME}" "conda activate cartridges" C-m
tmux send-keys -t "${SESSION_NAME}" "CUDA_VISIBLE_DEVICES=0 python3 ${SCRIPT_DIR}/run_x_aligned_ablation.py 2>&1 | tee '${LOG_FILE}'" C-m

echo "Session '${SESSION_NAME}' started successfully."
echo "Monitor with: tmux attach -t ${SESSION_NAME}"
echo "Or check log: tail -f ${LOG_FILE}"

