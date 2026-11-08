# LongHealth Continual Compaction Results (exp/tri)

This directory contains evaluation matrices, generation outputs, slot frequency diagnostics, and write recipes for monolithic continual updates on the 5-phase LongHealth benchmark across both Qwen and LLaMA, under budgets $B \in \{512, 1024\}$ and update ratios $t / B = 25\%$.

## Directory Structure

```
outputs/mine/
├── README.md
├── longhealth/
│   ├── e2e_budget512_topt128/            # Qwen-2.5-3B / Qwen3-4B (Budget 512, top-t 128)
│   ├── llama3_2_3b_budget512_topt128/    # LLaMA-3.2-3B (Budget 512, top-t 128)
│   ├── e2e_budget1024_topt256/           # Qwen-2.5-3B / Qwen3-4B (Budget 1024, top-t 256)
│   └── llama3_2_3b_budget1024_topt256/   # LLaMA-3.2-3B (Budget 1024, top-t 256)
└── recipes/
    ├── e2e_budget512_topt128.yaml
    ├── llama3_2_3b_budget512_topt128.yaml
    ├── e2e_budget1024_topt256.yaml
    └── llama3_2_3b_budget1024_topt256.yaml
```

Each experiment subdirectory contains:
- `teacher-forced-logppl-v1/`: 5×5 teacher-forced log-perplexity matrix (`matrix.json`, `matrix.csv`)
- `accuracy-freeform-mc-options-primeAnswer-v1/`: 5×5 held-out MCQ generation accuracy matrix (`matrix.json`, `matrix.csv`), plan metadata (`plan.json`), and individual cell response logs (`cells/*/generations.jsonl`)
- `slot_frequency/`: slot coverage, consecutive Jaccard overlap matrix (`jaccard_overlap.csv`), and per-layer stats (`per_layer_stats.csv`)

---

## Results Summary

| Model | Budget ($B$) | Top-$t$ | Final Stage Seen Loss | Mean Forgetting | Final Stage MCQ Accuracy | Consecutive Jaccard |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **LLaMA-3.2-3B** | 512 | 128 | 8.20 | +0.53 | **30.00%** | 0.864 |
| **LLaMA-3.2-3B** | 1024 | 256 | 7.91 | +0.48 | **30.50%** | 0.892 |
| **Qwen3-4B** | 512 | 128 | 1.95 | -0.44 | **4.25%** | 0.902 |
| **Qwen3-4B** | 1024 | 256 | 1.91 | -0.42 | **4.75%** | 0.923 |
