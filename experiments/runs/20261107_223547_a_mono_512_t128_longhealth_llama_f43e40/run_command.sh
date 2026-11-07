#!/bin/bash
cd /home/voquangtri2025/ICLR-Rebuttal/gated-continual-cartridges
bash examples/e2e_subkv_sweep/run_sweep.sh --model meta-llama/Llama-3.2-3B-Instruct --dataset longhealth --budget 512 --top-t 128 --delta-weight 0.01 --gpu 0 --eval-accuracy
