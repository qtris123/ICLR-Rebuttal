#!/bin/bash
cd /home/voquangtri2025/ICLR-Rebuttal/farid/gated-continual-cartridges
python3 run_continual.py --config configs/monolithic_update.yaml --stream data/streams/longhealth_5phase.json --model meta-llama/Llama-3.2-3B-Instruct --budget 1024 --write-slots 256 --vllm-mem 0.36 --value-prior-lambda 0 --c2-ridge-lambda 1e-4 --c2-ridge-scale spectral --out results/continual/five_phase/monolithic_update_1k_ridge_only/longhealth_llama/r.json --generate-out results/continual/five_phase/monolithic_update_1k_ridge_only/longhealth_llama/r.generations.jsonl
