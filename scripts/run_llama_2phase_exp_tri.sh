#!/usr/bin/env bash
set -e

# Activate conda environment
source /home/voquangtri2025/miniconda3/etc/profile.d/conda.sh
conda activate submem

cd /home/voquangtri2025/ICLR-Rebuttal

echo "=========================================================="
echo "Starting 2-Phase Validation Run: exp/tri Keys & Repositioning"
echo "Model: meta-llama/Llama-3.2-3B-Instruct"
echo "Stream: data/streams/longhealth_2phase.json"
echo "Batch size: 16"
echo "Date: $(date)"
echo "=========================================================="

python3 launch_experiment.py \
  --repo b \
  --name "llama_2phase_exp_tri_keys" \
  -- \
  python3 run_continual.py \
    --config configs/monolithic_update.yaml \
    --stream data/streams/longhealth_2phase.json \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --budget 1024 \
    --write-slots 256 \
    --generate-batch 16 \
    --vllm-mem 0.36 \
    --value-prior-lambda 0.01 \
    --c2-ridge-lambda 1e-4 \
    --c2-ridge-scale spectral \
    --key-mode exp-tri \
    --out results/ablation/exp_tri_keys_2phase/r.json \
    --generate-out results/ablation/exp_tri_keys_2phase/r.generations.jsonl

echo "=========================================================="
echo "Run complete! Scoring Phase 0 and Phase 1 generations..."
echo "=========================================================="

cd /home/voquangtri2025/ICLR-Rebuttal/farid/gated-continual-cartridges

if [ -f results/ablation/exp_tri_keys_2phase/r.generations_phase0.jsonl ]; then
  python3 scripts/score/score_generations.py \
    --stream data/streams/longhealth_2phase.json \
    results/ablation/exp_tri_keys_2phase/r.generations_phase0.jsonl
fi

if [ -f results/ablation/exp_tri_keys_2phase/r.generations_phase1.jsonl ]; then
  python3 scripts/score/score_generations.py \
    --stream data/streams/longhealth_2phase.json \
    results/ablation/exp_tri_keys_2phase/r.generations_phase1.jsonl
fi

echo "=========================================================="
echo "Scoring completed! All tasks finished successfully."
echo "=========================================================="

