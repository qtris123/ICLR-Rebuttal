# Diagnostic Ablations & Qwen Phase 1 Collapse Investigation

This directory contains the experimental logs, result matrices, and full generation outputs from the systematic investigation of the Phase 1 collapse of `Qwen/Qwen3-4B-Instruct-2507` on LongHealth (512 slots) in the `exp/tri` branch.

For the full detailed report, see:
[`../../DEBUGGING_REPORT_QWEN_PHASE1_COLLAPSE.md`](../../DEBUGGING_REPORT_QWEN_PHASE1_COLLAPSE.md)

---

## Directory Overview

1. **`2x2_cross_read/`**:
   - `2x2_matrix_results.json`: 2×2 cross-read accuracy and option distributions for:
     - `exp/tri cache` × `exp/tri prompt` (10/80, 100% A)
     - `exp/tri cache` × `main prompt` (0/80, 100% parsed are A)
     - `main cache` × `exp/tri prompt` (34/80, 42.50%, healthy balanced)
     - `main cache` × `main prompt` (2/80, 2.50%, format mismatch)
   - Subdirectories containing raw generations for each cell.
   - `run.log`: Full evaluation console output.

2. **`prefill_swap_ablation/`**:
   - `results.json`: Accuracy (10/80, 12.50%), 100% Option A collapse reproduced after swapping `main`'s document prefill helper.
   - `generations.jsonl`: Full generated text for all 80 questions.
   - `run.log`: Compaction and evaluation logs.

3. **`x_aligned_ablation/`**:
   - `results.json`: Accuracy (**31/80, 38.75%**), balanced option distribution `{'A': 11, 'B': 15, 'C': 30, 'D': 16, 'E': 8}`.
   - `generations.jsonl`: Full generated text and rationales for all 80 questions.
   - `run.log`: Compaction and evaluation logs.

---

## Standalone Reproduction Scripts

Located in the repo root:
- `run_2x2_cross_read.py` / `launch_2x2.sh`
- `run_prefill_swap_ablation.py` / `launch_prefill_swap.sh`
- `run_x_aligned_ablation.py` / `launch_x_aligned.sh`

