# Inference Ablations & Diagnostic Results

This directory contains the full evaluation outputs, generation logs (`.jsonl`), and per-phase official score summaries (`.scores.json`) comparing the baseline behavior against the prompt instruction and RoPE headroom safeguards on **LLaMA-3.2-3B-Instruct**.

---

## 1. Executive Summary

| Benchmark | Budget | Phase | HF Baseline Acc | Interim Fix (`"Please think very briefly..."`) | **Final Fix (`"Respond with **only**..."`)** | Baseline Unparsed | **Final Fix Unparsed** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **QuALITY** | 1,024 | **Phase 2** | 25.40% | 28.34% | **27.54%** | 1.60% (6/374) | **0.00% (0/374)** |
| **QuALITY** | 1,024 | **Phase 3** | 5.80% | 15.80% | **19.60%** | 77.40% (387/500) | **32.20% (161/500)** |
| **QuALITY** | 1,024 | **Phase 4** | 0.00% | 2.25% | **5.79%** | 99.84% (621/622) | **75.56% (470/622)** |
| **LongHealth** | 1,024 | **Phase 2** | 29.58% | 30.83% | **32.50%** | 0.83% (2/240) | **0.00% (0/240)** |
| **LongHealth** | 1,024 | **Phase 3** | 27.50% | 32.50% | **33.12%** | 1.25% (4/320) | **0.00% (0/320)** |
| **LongHealth** | 1,024 | **Phase 4** | 17.75% | 22.75% | **25.50%** | 0.25% (1/400) | **1.25% (5/400)** |
| **LongHealth** | 16,384 | **Phase 2** | 36.67% | - | **40.83%** | 2.92% (7/240) | **0.00% (0/240)** |
| **LongHealth** | 16,384 | **Phase 3** | 34.69% | - | **37.19%** | 0.94% (3/320) | **0.00% (0/320)** |
| **LongHealth** | 16,384 | **Phase 4** | 36.00% | - | **36.25%** | 0.00% (0/400) | **0.00% (0/400)** |

---

## 2. Directory Layout

- `respond_only_safe_rope/`:
  - `quality_llama_1k/`: Generations & scores using `"Respond with **only** the letter ({option_list}) of the correct option."` with safe RoPE headroom ($L=129,024$).
  - `longhealth_llama_1k/`: Generations & scores using the same concise instruction and safe headroom.
- `think_briefly_safe_rope/`:
  - `quality_llama_1k/`: Generations & scores using `"Please think very briefly and then respond with **only** the letter..."` with safe RoPE headroom ($L=129,024$).
  - `longhealth_llama_1k/`: Generations & scores under the interim instruction.
- `rope_headroom_comparison/`:
  - `longhealth_1k_rl131072/`: LongHealth 1k evaluated at $L=131,072$ (demonstrates prompt token overflow effects).
  - `longhealth_1k_rl129024/`: LongHealth 1k evaluated at $L=129,024$ (safe headroom).
- `longhealth_llama_16k_budget/`:
  - `rl_131072/`: LongHealth 16k evaluated at native length.
  - `rl_129024/`: LongHealth 16k evaluated with safe headroom.
- `summary_all_ablations.json`: Master JSON holding raw metrics, question counts, and per-document accuracy tables.

---

## 3. Key Findings

1. **Root Cause of Phase 3/4 Failure**:
   - The memory update rule pushed logical context to $L = 131,072$, leaving **0 tokens of headroom** for the evaluation question prompt ($L_p \approx 100-250$ tokens) and answer generation.
   - For LLaMA 3.2-3B, any token beyond position 131,072 receives out-of-distribution RoPE embeddings, breaking attention to the prompt and causing fallback to safety/refusal conversational priors (`"I'll do my best..."`, `"I can't help you."`).
2. **Impact of Prompt Instruction**:
   - Instructions prompting the model to `"think"` trigger conversational thought generation before outputting the letter.
   - Changing the prompt to strictly `"Respond with **only** the letter ({option_list}) of the correct option."` forces direct emission of the option letter (e.g., `A`, `B`, `C`), reducing decode lengths, eliminating refusal triggers, and boosting accuracy across all evaluated phases.
