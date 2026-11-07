# LongHealth Continual Compaction Results

This directory contains evaluation matrices, generation outputs, and slot frequency diagnostics for the monolithic continual updates on the 5-phase LongHealth benchmark across both models reported in the paper.

## Directory Structure

```
outputs/mine/
├── longhealth/
│   ├── e2e_budget512_topt128/            # Qwen-2.5-3B / Qwen3-4B-Instruct-2507
│   │   ├── teacher-forced-logppl-v1/     # 5x5 Teacher-forced Log-Perplexity Matrix
│   │   ├── accuracy-freeform-mc-options-primeAnswer-v1/ # 5x5 MCQ Generation Accuracy Matrix
│   │   └── slot_frequency/               # Slot coverage & consecutive Jaccard overlap
│   └── llama3_2_3b_budget512_topt128/    # meta-llama/Llama-3.2-3B-Instruct
│       ├── teacher-forced-logppl-v1/     # 5x5 Teacher-forced Log-Perplexity Matrix
│       ├── accuracy-freeform-mc-options-primeAnswer-v1/ # 5x5 MCQ Generation Accuracy Matrix
│       └── slot_frequency/               # Slot coverage & consecutive Jaccard overlap
└── recipes/
    ├── e2e_budget512_topt128.yaml        # Compaction & AM write recipe for Qwen
    └── llama3_2_3b_budget512_topt128.yaml# Compaction & AM write recipe for LLaMA
```

## Summary of Results

### 1. Qwen (`Qwen/Qwen3-4B-Instruct-2507`)
- **Budget / Top-t**: 512 / 128 (25% update ratio, $\lambda_\Delta = 0.01$)
- **Perplexity**:
  - Final Stage (p05) Seen Loss: 2.00
  - Mean Forgetting: -0.442
- **Generation Accuracy**:
  - Stage 1 (p01) on Task 1: 31.25%
  - Final Stage (p05) Average Accuracy: 19.00%
- **Slot Coverage**: 36.13% | **Consecutive Jaccard**: 0.902

### 2. LLaMA (`meta-llama/Llama-3.2-3B-Instruct`)
- **Budget / Top-t**: 512 / 128 (25% update ratio, $\lambda_\Delta = 0.01$)
- **Perplexity**:
  - Initial Stage (p01) Loss: 7.06
  - Final Stage (p05) Seen Loss: 8.16
  - Mean Forgetting: +0.531
- **Slot Coverage**: 45.98% | **Consecutive Jaccard**: 0.864
