#!/bin/bash
cd /home/voquangtri2025/ICLR-Rebuttal/farid/gated-continual-cartridges
python3 run_continual.py --config configs/monolithic_update.yaml --stream data/streams/longhealth_5phase.json --model meta-llama/Llama-3.2-3B-Instruct --budget 1024 --write-slots 256 --vllm-mem 0.36 --resume-from results/continual/five_phase/monolithic_update_1k/longhealth_llama/r_phase1.memory.pt --out results/continual/five_phase/monolithic_update_1k/longhealth_llama/r.json --generate-out results/continual/five_phase/monolithic_update_1k/longhealth_llama/r.generations.jsonl
