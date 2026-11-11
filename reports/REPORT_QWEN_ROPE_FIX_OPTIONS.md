# Investigation Report: Isolating Continual RoPE Fix Options on Qwen3-4B-Instruct

**Repository / Branch**: `farid/gated-continual-cartridges` (`moe-am`, replica fork `qtris123/gated-continual-cartridge-moe-am-branch` @ [`7f482552`](https://github.com/qtris123/gated-continual-cartridge-moe-am-branch/commit/7f4825527f73fb2a634a7a8cb7075f36255f6c57))  
**Model**: `Qwen/Qwen3-4B-Instruct-2507` (native `max_position_embeddings = 262,144`, `rope_theta = 5,000,000`)  
**Benchmark**: LongHealth 5-phase continual stream ([`longhealth_5phase.json`](file:///home/voquangtri2025/ICLR-Rebuttal/farid/gated-continual-cartridges/data/streams/longhealth_5phase.json), 80 QA/phase, 400 total at Phase 4; ~236k stream tokens)  
**Shared Config**: `--config configs/monolithic_update.yaml --budget 1024 --write-slots 256 --value-prior-lambda 0.01` (**no** `C2` spectral ridge penalty)  
**Date**: November 11, 2026

---

## 1. Executive Summary

Following the LLaMA-3.2-3B RoPE study ([`REPORT_LLAMA_CONTEXT_OVERFLOW_VS_EXP_TRI.md`](file:///home/voquangtri2025/ICLR-Rebuttal/reports/REPORT_LLAMA_CONTEXT_OVERFLOW_VS_EXP_TRI.md)), we evaluated `--rope-mode` (`cumulative`, `cache_len`, `window_cap` at native 262k, and `window_cap` capped at LLaMA's 131k via a new `--rope-max-context` flag) on `Qwen/Qwen3-4B-Instruct-2507`.

| Condition | Effective RoPE Ceiling | `L_logical` Trajectory (Ph0 -> Ph4) | Ph4 Acc (Unp%) | `avg_final` log-ppl | `BWT` | Status vs. Baseline |
|---|---|---|---|---|---|---|
| `cumulative` (baseline) | `262,144` | `49.5k -> 92.1k -> 140.0k -> 186.8k -> 236.2k` | **36.00%** (4.00%) | **1.5319** | **+0.1760** | Healthy (never overflows 262k) |
| `cache_len` (fixed, no ridge) | `1,024` (physical) | `1045 -> 1024 -> 1024 -> 1024 -> 1024` | 16.50% (**14.75%**) | 2.0360 | -0.1036 | Monotonic degradation |
| `window_cap` (native 262k) | `262,144` | `49.5k -> 92.1k -> 140.0k -> 186.8k -> 236.2k` | **36.00%** (3.00%) | 1.6147 | +0.0931 | No-op (cap never triggered) |
| `window_cap` (forced 131k) | `131,072` | `49.5k -> 92.1k -> 131.1k -> 131.1k -> 131.1k` | 29.50% (2.00%) | 2.1705 | -0.2470 | Collapses at Ph3 once overflow repeats |

### Key Conclusions
1. **Why baseline Qwen did not need a RoPE overflow fix:** 5-phase LongHealth accumulates `236,249` logical tokens, remaining below Qwen3's native `262,144` context window. Baseline `cumulative` completes cleanly at **36.00%** Ph4 accuracy and positive **`BWT = +0.1760`**.
2. **No RoPE wiring bug in `cache_len` or `window_cap`:** Training (`C2` least squares) and inference (`probe_ppl` / vLLM/HF generation) use identical student RoPE positions on Qwen3. Instead, `cache_len` degrades Qwen because packing keys scattered across `0..236k` into `0..1023` shifts student softmax attention away from the uncompressed teacher target `Y` (`L06`/`L30` chunk error increases ~`2.5–2.7x` even at Phase 0), and without ridge, `inplace_frozen=True` preserves `75.0%` (`768/1024`) of `C1`/`C2` slots bit-identical with unregularized value outliers across Phases 1–4.
3. **Why `window_cap` looked fine at 262k but broke at 131k:** At native `262,144`, `window_cap` never triggered. When forced to `131,072` (`--rope-max-context 131072`), Phase 2 caps slightly (`L_before = 83,174 < 92,135`) and still scores `33.75%`, but Phases 3–4 repeatedly pull `L_before` back to `~82–84k` while `M` already contains keys rotated up to `131,071`. That places new-document prefill and memory-reader queries **~47k positions behind** previously compacted keys, spiking `C2` norms (`14.25 -> 17.33`) and crashing Phase 3 accuracy to **19.69%** (`190/320` predictions collapsing to leading `'A'`).

---

## 2. Step-by-Step Timeline: What Was Tried & Observed

### Step 0: Baseline `cumulative` Reference
- **Run**: [`20261107_121513_b_mono_1k_longhealth_qwen_0bfc62`](file:///home/voquangtri2025/ICLR-Rebuttal/experiments/runs/20261107_121513_b_mono_1k_longhealth_qwen_0bfc62) (Ph0) -> resumed in [`20261107_153018_b_mono_1k_longhealth_qwen_resumed_15d721`](file:///home/voquangtri2025/ICLR-Rebuttal/experiments/runs/20261107_153018_b_mono_1k_longhealth_qwen_resumed_15d721) (`57c67c8f`, `17,391s`).
- **Result**: `L_logical` progressed `[49501, 92135, 140033, 186846, 236249]`. Accuracy remained stable (`41.25% -> 30.63% -> 35.42% -> 33.44% -> 36.00%`, `4.00%` unparsed) with `avg_final = 1.5319`, `BWT = +0.1760`.
- **Artifacts**: [`r.json`](file:///home/voquangtri2025/ICLR-Rebuttal/farid/gated-continual-cartridges/results/continual/five_phase/monolithic_update_1k/longhealth_qwen/r.json).

---

### Step 1: `--rope-mode cache_len` (Fixed Coordinate Alignment, No Ridge)
- **What was tried**: Ran 5-phase Qwen on commit `941a0b37` (`cache_len` with key rebasing to `[0..target-1]`, eval-aligned student queries, dual `onpolicy_layers`, and pre-write `Y_mem` isolation), with `--generate-batch 16` and no ridge (`--c2-ridge-lambda 0`).
- **Execution hurdles resolved**:
  1. [`20261111_132947_*`](file:///home/voquangtri2025/ICLR-Rebuttal/experiments/runs/20261111_132947_b_qwen_5phase_rope_cache_len_fixed_no_ridge_eced17): Failed (`AssertionError`) because `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is incompatible with vLLM's `CuMemAllocator` memory pool.
  2. [`20261111_134700_*`](file:///home/voquangtri2025/ICLR-Rebuttal/experiments/runs/20261111_134700_b_qwen_5phase_rope_cache_len_fixed_no_ridge_eced17): Failed with `torch.OutOfMemoryError` in `torch._inductor` during vLLM engine init (`--vllm-mem 0.46`, default `--vllm-max-len 65536`).
  3. [`20261111_135657_b_qwen_5phase_rope_cache_len_fixed_no_ridge_dcc6e6`](file:///home/voquangtri2025/ICLR-Rebuttal/experiments/runs/20261111_135657_b_qwen_5phase_rope_cache_len_fixed_no_ridge_dcc6e6): **Succeeded** (`9,596.8s`) after unsetting `PYTORCH_CUDA_ALLOC_CONF` and using `--vllm-mem 0.36 --vllm-max-len 28672`.
- **Result**: Monotonic degradation over phases — Accuracy `37.50% -> 25.00% -> 17.50% -> 17.19% -> 16.50%`, unparsed rate climbing `2.50% -> 8.13% -> 3.75% -> 13.44% -> 14.75%` (`59/400`), `avg_final = 2.0360`, `BWT = -0.1036`. Mean generation length dropped to `397.7` chars at Phase 4 (`130/400` abstentions).
- **Artifacts**: [`r.json`](file:///home/voquangtri2025/ICLR-Rebuttal/farid/gated-continual-cartridges/results/ablation/rope_only_cache_len_fixed_no_ridge/longhealth_qwen/r.json), [`scores_phase4.json`](file:///home/voquangtri2025/ICLR-Rebuttal/farid/gated-continual-cartridges/results/ablation/rope_only_cache_len_fixed_no_ridge/longhealth_qwen/scores_phase4.json).

---

### Step 2: RoPE Wiring Audit & Root-Cause Investigation of `cache_len` Degradation
Because `cache_len` improved LLaMA-3.2-3B (`9.75% -> 24.00%`) yet hurt Qwen3-4B (`36.00% -> 16.50%`), we audited both wiring correctness and the unregularized fit:

1. **Verified Training-vs-Inference Wiring Is Exact** ([`audit_qwen_wiring.py`](file:///home/voquangtri2025/.gemini/antigravity/brain/df662512-b7cc-4560-8c18-ea78c22c8b6d/scratch/audit_qwen_wiring.py), [`audit_chunked_and_wcap.py`](file:///home/voquangtri2025/.gemini/antigravity/brain/df662512-b7cc-4560-8c18-ea78c22c8b6d/scratch/audit_chunked_and_wcap.py)):
   - `SelfStudyQueryGenerator` post-`q_norm`+RoPE query extraction matches `Qwen3Attention` (`max diff = 0.0`).
   - Rotating compacted/document queries from union positions to eval `[1024..]` matches live `Qwen3Model` eval queries (`max diff = 1.69e-5` in `bfloat16`).
   - Rebased `M` leaves `frozen` slots bit-identical when `inplace_frozen=True` and rotates writable slots cleanly.
   - `ChunkedCompaction` (`prefix=6, m_len=60`) and `window_cap` unit checks both pass.
2. **Identified the True Cause: Teacher-vs-Student Softmax Distortion + Unregularized `inplace_frozen` Drift** ([`diag_part2.py`](file:///home/voquangtri2025/.gemini/antigravity/brain/df662512-b7cc-4560-8c18-ea78c22c8b6d/scratch/diag_part2.py)):
   - Even with 100% aligned student RoPE positions, teacher targets $Y$ are computed over keys spread across `0..49,500`, whereas student attention $X_s$ compresses keys into `[23..1044]` (`0..1023`). On Phase-0 Chunk 1 (`[23:12,358)` -> `255` slots):
     - `L00 H0`: condition number `15 -> 22`, test `rel_err` rises `0.3931 -> 0.5386`.
     - `L06 H0`: condition number `1,578 -> 8,407`, test `rel_err` rises `0.2179 -> 0.5931` (`2.7x`).
     - `L30 H0`: `s_min` drops `0.00460 -> 0.00094`, condition number `14,237 -> 55,410`, test `rel_err` rises `0.1958 -> 0.4883` (`2.5x`).
   - Across continual updates (Phases 1–4), [`_compact_union`](file:///home/voquangtri2025/ICLR-Rebuttal/farid/gated-continual-cartridges/submem/continual/memory_ops.py#L576-L585) sets `inplace_frozen=True`, preserving **`75.0%` (`221,184 / 294,912`)** of `C1` and `C2` slots bit-identical at their exact slot indices from Phase 1 onward. Without `--c2-ridge-lambda`, unregularized Phase 0/1 `C2` value spikes (`> 200` norm count: `2,018` at Ph0, `1,507` at Ph4 vs. `746 -> 580` in `262k`) stay locked in `M`, compounding `log_ppl` (`1.3909 -> avg_final 2.0360`).
   - *Open gap*: A full 5-phase Qwen `cache_len` run **with** `--c2-ridge-lambda 1e-4 --c2-ridge-scale spectral` was not run (`diag_part2.py` noted spectral ridge cuts Chunk-1 `L30` `C2_max` from `23.5 -> 7.2` on `clen`).

---

### Step 3: `--rope-mode window_cap` at Native Context (`262,144`)
- **What was tried**: Tested `--rope-mode window_cap` on Qwen (no ridge) with `--vllm-mem 0.36`.
- **Execution hurdle resolved**:
  1. [`20261111_163714_*`](file:///home/voquangtri2025/ICLR-Rebuttal/experiments/runs/20261111_163714_b_qwen_5phase_rope_window_cap_no_ridge_e83e54): Failed on vLLM init because default `--vllm-max-len 65536` required `9.00 GiB` KV cache vs `5.15 GiB` available (`estimated max len = 37,472`).
  2. [`20261111_170232_b_qwen_5phase_rope_window_cap_no_ridge_c59d7e`](file:///home/voquangtri2025/ICLR-Rebuttal/experiments/runs/20261111_170232_b_qwen_5phase_rope_window_cap_no_ridge_c59d7e): **Succeeded** (`9,442.5s`) with `--vllm-max-len 28672`.
- **Result**: Because Qwen3's `max_position_embeddings = 262,144 > 236,249`, `window_cap` **never triggered** in [`KVRecursiveMemory._refit`](file:///home/voquangtri2025/ICLR-Rebuttal/farid/gated-continual-cartridges/submem/continual/memory_policies.py#L272-L296). It tracked the baseline closely (`36.00%` Ph4 Acc, `3.00%` unp, `avg_final = 1.6147`, `BWT = +0.0931`).
- **Artifacts**: [`r.json`](file:///home/voquangtri2025/ICLR-Rebuttal/farid/gated-continual-cartridges/results/ablation/rope_only_window_cap_no_ridge/longhealth_qwen/r.json), [`scores_phase4.json`](file:///home/voquangtri2025/ICLR-Rebuttal/farid/gated-continual-cartridges/results/ablation/rope_only_window_cap_no_ridge/longhealth_qwen/scores_phase4.json).

---

### Step 4: `--rope-mode window_cap --rope-max-context 131072` (LLaMA Context Ceiling)
To test whether `window_cap` survives on Qwen once overflowing occurs (apples-to-apples with LLaMA's `131,072` ceiling):

- **Code changes** (commit [`7f482552`](https://github.com/qtris123/gated-continual-cartridge-moe-am-branch/commit/7f4825527f73fb2a634a7a8cb7075f36255f6c57) on `review-fork/moe-am`, `146` tests passing in `tests/`):
  - Added `--rope-max-context` to [`cli.py`](file:///home/voquangtri2025/ICLR-Rebuttal/farid/gated-continual-cartridges/submem/continual/cli.py#L331-L339) (requires `--rope-mode window_cap` when `> 0`), [`KVRecursiveMemory._max_context`](file:///home/voquangtri2025/ICLR-Rebuttal/farid/gated-continual-cartridges/submem/continual/memory_policies.py#L171-L176), [`run_continual.py`](file:///home/voquangtri2025/ICLR-Rebuttal/farid/gated-continual-cartridges/run_continual.py#L241), and [`test_recompaction.py`](file:///home/voquangtri2025/ICLR-Rebuttal/farid/gated-continual-cartridges/tests/test_recompaction.py#L1060-L1160).
- **Run**: Initial attempt [`20261111_195337_*`](file:///home/voquangtri2025/ICLR-Rebuttal/experiments/runs/20261111_195337_b_qwen_5phase_rope_window_cap_131k_no_ridge_7e07a4) hit the `PYTORCH_CUDA_ALLOC_CONF` assertion; relaunched cleanly in [`20261111_195433_b_qwen_5phase_rope_window_cap_131k_no_ridge_7e07a4`](file:///home/voquangtri2025/ICLR-Rebuttal/experiments/runs/20261111_195433_b_qwen_5phase_rope_window_cap_131k_no_ridge_7e07a4) (**Succeeded**, `10,572.5s`).
- **Result & Phase-by-Phase Behavior**:
  - **Ph0 (`49,501`) & Ph1 (`92,135`)**: Under ceiling — identical to `window_cap_262k` (`43.75%`, `30.00%`).
  - **Ph2 (first overflow, `92,135 + 47,898 = 140,033 > 131,072`)**: Re-anchors `L_before = 131,072 - 47,898 = 83,174` (~9k overlap with `M1` keys ending at `92,134`). Accuracy holds at **`33.75%`**, though `log_ppl` ticks up (`1.5568 -> 1.7479` on Ph3).
  - **Ph3 (second overflow, `131,072 + 46,813 > 131,072`)**: Pulls `L_before` back to `84,259`, **46.8k tokens behind `M2`'s rightmost compacted keys** (`~131,071`), because `window_cap` does not rebase `M2`'s keys. Both `Doc3` prefill and `mem_q` evaluation now sit inside/behind `M2`'s RoPE span (creating negative relative distances). Accuracy collapses to **`19.69%`** (`190/320` generations start with `'A'`), `C2` mean norm jumps `14.25 -> 16.20`, and `avg log_ppl` jumps (`2.09` across P1–P4).
  - **Ph4 (third overflow, `L_before = 81,669`)**: `C2` mean reaches **`17.33`** (`> 200` outlier slots `591 -> 925 -> 1,074`), ending at **`29.50%`** Ph4 accuracy, **`avg_final = 2.1705`**, and **`BWT = -0.2470`**.
- **Artifacts**: [`r.json`](file:///home/voquangtri2025/ICLR-Rebuttal/farid/gated-continual-cartridges/results/ablation/rope_only_window_cap_131k_no_ridge/longhealth_qwen/r.json), [`scores_phase4.json`](file:///home/voquangtri2025/ICLR-Rebuttal/farid/gated-continual-cartridges/results/ablation/rope_only_window_cap_131k_no_ridge/longhealth_qwen/scores_phase4.json) (`scores_phase0..4.json`).

---

## 3. Full Per-Phase Metrics Comparison

### Accuracy (Unparsed Rate %) Across Continual Checkpoints

| Condition | Ph0 (`n=80`) | Ph1 (`n=160`) | Ph2 (`n=240`) | Ph3 (`n=320`) | Ph4 (`n=400`) |
|---|---|---|---|---|---|
| `cumulative` (baseline) | 41.25% (7.50%) | 30.63% (1.87%) | **35.42%** (5.00%) | **33.44%** (5.31%) | **36.00%** (4.00%) |
| `cache_len` (fixed, no ridge) | 37.50% (2.50%) | 25.00% (8.13%) | 17.50% (3.75%) | 17.19% (13.44%) | 16.50% (**14.75%**) |
| `window_cap` (native 262k) | **43.75%** (6.25%) | **30.00%** (5.00%) | 33.75% (5.83%) | 31.25% (9.38%) | **36.00%** (3.00%) |
| `window_cap` (forced 131k) | **43.75%** (6.25%) | **30.00%** (5.00%) | 33.75% (9.58%) | **19.69%** (6.56%) | 29.50% (2.00%) |

### Final Phase-4 Per-Task `log_ppl` Breakdown

| Condition | Ph1 (`base 2.2421`) | Ph2 (`base 3.6222`) | Ph3 (`base 3.2388`) | Ph4 (`base 3.2881`) | Ph5 (`base 3.7995`) | `avg_final` | `avg_forget` | `BWT` |
|---|---|---|---|---|---|---|---|---|
| `cumulative` (baseline) | **1.0631** | **1.7279** | **1.5666** | **1.5079** | **1.7941** | **1.5319** | **-0.1760** | **+0.1760** |
| `cache_len` (fixed, no ridge) | 1.2758 | 2.3098 | 2.0996 | 2.0113 | 2.4835 | 2.0360 | +0.1036 | -0.1036 |
| `window_cap` (native 262k) | 1.1268 | 1.8062 | 1.6730 | 1.5728 | 1.8948 | 1.6147 | -0.0931 | +0.0931 |
| `window_cap` (forced 131k) | 1.5652 | 2.3617 | 2.1853 | 2.2292 | 2.5111 | 2.1705 | +0.2470 | -0.2470 |

### Model Contrast: Qwen3-4B vs. LLaMA-3.2-3B
- **LLaMA-3.2-3B** (131k native window, `--c2-ridge-lambda 1e-4`; [`REPORT_LLAMA_CONTEXT_OVERFLOW_VS_EXP_TRI.md`](file:///home/voquangtri2025/ICLR-Rebuttal/reports/REPORT_LLAMA_CONTEXT_OVERFLOW_VS_EXP_TRI.md)): `cumulative` overflows past 131k at Ph3 (`0.94%`) and Ph4 (`9.75%`); `window_cap` degrades at Ph4 (`7.50%`, `64.50%` unparsed); **`cache_len` (fixed + ridge) solves it (`24.00%` Ph4 Acc, `0.50%` unp, `BWT = +0.0102`)**.
- **Qwen3-4B** (262k native window, `rope_theta = 5e6`, tested without ridge): `cumulative` never overflows and leads (`36.00%`, `BWT = +0.1760`); **`window_cap` still fails whenever overflow actually occurs (`131k` cap: Ph3 `19.69%`, Ph4 `29.50%`)**, while unregularized **`cache_len` hurts Qwen (`16.50%`)** due to `0..236k -> 0..1023` attention-matching conditioning loss and locked Phase 0/1 outliers.

---

## 4. Reference Links for Follow-Up Investigation

### Code & Tests
- CLI flags & validation (`--rope-mode`, `--rope-max-context`): [`submem/continual/cli.py`](file:///home/voquangtri2025/ICLR-Rebuttal/farid/gated-continual-cartridges/submem/continual/cli.py#L318-L340)
- `window_cap` / `cache_len` logical tracking & `_max_context`: [`KVRecursiveMemory`](file:///home/voquangtri2025/ICLR-Rebuttal/farid/gated-continual-cartridges/submem/continual/memory_policies.py#L171-L300)
- Union recompression & `inplace_frozen` alignment spec: [`recompress_union` / `_compact_union`](file:///home/voquangtri2025/ICLR-Rebuttal/farid/gated-continual-cartridges/submem/continual/memory_ops.py#L390-L609)
- RoPE rotation & `frozen_indices` solver: [`highest_attention_keys.py`](file:///home/voquangtri2025/ICLR-Rebuttal/farid/gated-continual-cartridges/submem/compaction/algorithms/highest_attention_keys.py#L446-L509)
- Unit & CLI tests: [`tests/test_recompaction.py`](file:///home/voquangtri2025/ICLR-Rebuttal/farid/gated-continual-cartridges/tests/test_recompaction.py#L1060-L1160)

### Experiment Runs, Results, & Diagnostic Scripts
- Master registry: [`experiments/experiments_registry.tsv`](file:///home/voquangtri2025/ICLR-Rebuttal/experiments/experiments_registry.tsv)
- Result directories:
  - [`results/continual/five_phase/monolithic_update_1k/longhealth_qwen`](file:///home/voquangtri2025/ICLR-Rebuttal/farid/gated-continual-cartridges/results/continual/five_phase/monolithic_update_1k/longhealth_qwen)
  - [`results/ablation/rope_only_cache_len_fixed_no_ridge/longhealth_qwen`](file:///home/voquangtri2025/ICLR-Rebuttal/farid/gated-continual-cartridges/results/ablation/rope_only_cache_len_fixed_no_ridge/longhealth_qwen)
  - [`results/ablation/rope_only_window_cap_no_ridge/longhealth_qwen`](file:///home/voquangtri2025/ICLR-Rebuttal/farid/gated-continual-cartridges/results/ablation/rope_only_window_cap_no_ridge/longhealth_qwen)
  - [`results/ablation/rope_only_window_cap_131k_no_ridge/longhealth_qwen`](file:///home/voquangtri2025/ICLR-Rebuttal/farid/gated-continual-cartridges/results/ablation/rope_only_window_cap_131k_no_ridge/longhealth_qwen)
- Scratch audit & diagnostic scripts:
  - [`audit_qwen_wiring.py`](file:///home/voquangtri2025/.gemini/antigravity/brain/df662512-b7cc-4560-8c18-ea78c22c8b6d/scratch/audit_qwen_wiring.py), [`audit_chunked_and_wcap.py`](file:///home/voquangtri2025/.gemini/antigravity/brain/df662512-b7cc-4560-8c18-ea78c22c8b6d/scratch/audit_chunked_and_wcap.py)
  - [`diag_qwen_cache_len.py`](file:///home/voquangtri2025/.gemini/antigravity/brain/df662512-b7cc-4560-8c18-ea78c22c8b6d/scratch/diag_qwen_cache_len.py), [`diag_part2.py`](file:///home/voquangtri2025/.gemini/antigravity/brain/df662512-b7cc-4560-8c18-ea78c22c8b6d/scratch/diag_part2.py)
- Companion reports:
  - [`REPORT_LLAMA_CONTEXT_OVERFLOW_VS_EXP_TRI.md`](file:///home/voquangtri2025/ICLR-Rebuttal/reports/REPORT_LLAMA_CONTEXT_OVERFLOW_VS_EXP_TRI.md)
  - [`DEBUGGING_REPORT_QWEN_PHASE1_COLLAPSE.md`](file:///home/voquangtri2025/ICLR-Rebuttal/reports/DEBUGGING_REPORT_QWEN_PHASE1_COLLAPSE.md)
  - [`REPORT_LLAMA_RIDGE_MAGNITUDE_PENALTY.md`](file:///home/voquangtri2025/ICLR-Rebuttal/reports/REPORT_LLAMA_RIDGE_MAGNITUDE_PENALTY.md)
