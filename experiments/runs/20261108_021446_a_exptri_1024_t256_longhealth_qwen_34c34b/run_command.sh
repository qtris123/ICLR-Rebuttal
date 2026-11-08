#!/bin/bash
cd /home/voquangtri2025/ICLR-Rebuttal/gated-continual-cartridges
bash examples/e2e_subkv_sweep/run_sweep.sh --model Qwen/Qwen3-4B-Instruct-2507 --dataset longhealth --budget 1024 --top-t 256 --delta-weight 0.01 --gpu 0 --eval-accuracy
