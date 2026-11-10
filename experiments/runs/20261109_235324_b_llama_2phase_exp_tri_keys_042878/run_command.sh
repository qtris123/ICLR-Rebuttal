#!/bin/bash
cd /home/voquangtri2025/ICLR-Rebuttal/farid/gated-continual-cartridges
python3 run_continual.py --config configs/monolithic_update.yaml --stream data/streams/longhealth_2phase.json --model meta-llama/Llama-3.2-3B-Instruct --budget 1024 --write-slots 256 --generate-batch 16 --vllm-mem 0.36 --value-prior-lambda 0.01 --c2-ridge-lambda 1e-4 --c2-ridge-scale spectral --key-mode exp-tri --out results/ablation/exp_tri_keys_2phase/r.json --generate-out results/ablation/exp_tri_keys_2phase/r.generations.jsonl
