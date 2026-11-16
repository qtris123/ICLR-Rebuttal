#!/bin/bash
cd /home/voquangtri2025/ICLR-Rebuttal/farid/gated-continual-cartridges
python3 run_continual.py --config configs/monolithic_update.yaml --stream data/streams/longhealth_5phase.json --model Qwen/Qwen3-4B-Instruct-2507 --budget 1024 --write-slots 256 --vllm-mem 0.46 --value-prior-lambda 0.01 --rope-mode cache_len --generate-batch 16 --generate-per-phase --save-memory-per-phase --out results/ablation/rope_only_cache_len_fixed_no_ridge/longhealth_qwen/r.json --generate-out results/ablation/rope_only_cache_len_fixed_no_ridge/longhealth_qwen/r.generations.jsonl
