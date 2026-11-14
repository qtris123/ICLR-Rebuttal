# write-rule least, budget 1024 (set1-4): results

Runs of `scripts/farid_handoff/write-rule-least/write_rule_least.sh` set1-set4 on
branch `moe-am` (commit 05ee8ff6), one NVIDIA H100 80GB per run, through Slurm,
2026-11-14. All four finished cleanly: `complete: true`, 5 phases, `write_rule: least`,
`budget: 1024`, `write_slots: 256`. Each `r.json` records the exact command under `command`.

## Differences from the handoff script

1. **`--vllm-mem 0.35`** instead of the config's 0.18. On an 80 GB card, 0.18 gives
   vLLM ~14.3 GiB. After Qwen's weights (7.6 GiB) and vLLM overhead, the KV cache got
   0.88 GiB against the 9.00 GiB that `vllm-max-len 65536` needs, so vLLM refused to start.
   0.35 gives 14.34 GiB of KV cache, about what 0.18 gives on a ~180 GB B200. It only
   changes how much memory vLLM reserves; the method is unchanged.
2. **`COMPACTION_CACHE=0`** and `OMP/MKL_NUM_THREADS=8`, as in the paper's
   `scripts/launch/dgx/w30` job specs.
3. **Environment:** Python 3.12 venv with torch 2.8.0+cu128, transformers 4.57.1,
   vllm 0.11.0, the versions in the top-level README. Note that `requirements.txt`
   and `pyproject.toml` pin torch 2.14.1 / transformers 5.19.0 instead, which disagree
   with the README and with vllm 0.11.0.
4. A venv instead of conda, and Hugging Face offline mode (models from the local cache).

## Results (log-perplexity of reference answers, lower is better)

| Set | Run | avg_final | avg_forget | bwt | forgetting_ratio | runtime |
|---|---|---|---|---|---|---|
| set1 | longhealth_qwen | 1.7930 | -0.1410 | +0.1410 | -0.1189 | 03:02:57 |
| set2 | longhealth_llama | 1.2192 | -0.0591 | +0.0591 | -0.1903 | 00:46:16 |
| set3 | quality_qwen | 1.9407 | +0.0998 | -0.0998 | +0.0506 | 03:45:15 |
| set4 | quality_llama | 1.4124 | +0.1389 | -0.1389 | +0.6657 | 01:12:12 |

Per phase (phase 1 / 2 / 3 / 4 / 5); forget = final minus right-after-learning, negative = improved:

| Set | final_logppl | forget (phases 1-4) |
|---|---|---|
| set1 | 1.200 / 1.984 / 1.878 / 1.772 / 2.132 | -0.132 / -0.228 / -0.178 / -0.026 |
| set2 | 0.843 / 1.319 / 1.265 / 1.244 / 1.425 | -0.167 / -0.004 / +0.037 / -0.103 |
| set3 | 1.745 / 2.247 / 1.818 / 1.617 / 2.276 | -0.032 / +0.178 / +0.036 / +0.217 |
| set4 | 1.300 / 1.669 / 1.311 / 1.244 / 1.537 | +0.090 / +0.098 / +0.358 / +0.011 |

## Accuracy (the paper's Acc / Fgt / Unp)

`run_continual.py` writes the answers but does not score them; scoring is the separate step
`scripts/score/run_metrics.py <run_dir> ...`, which writes `r.generations_phase<k>.scores.json`
next to each answer file (included). Percent; Acc = mean accuracy over the five phases after
phase 5, Fgt = best accuracy on phases 1-4 minus their accuracy after phase 5, averaged,
Unp = share of replies after phase 5 with no parsable option (counted as wrong).

| Set | Run | Acc | Fgt | Unp | per phase after phase 5 (1 / 2 / 3 / 4 / 5) |
|---|---|---|---|---|---|
| set1 | longhealth_qwen | 27.7 | 3.8 | 2.0 | 31.2 / 23.8 / 30.0 / 27.5 / 26.2 |
| set2 | longhealth_llama | 17.8 | 17.5 | 0.2 | 20.0 / 12.5 / 13.8 / 21.2 / 21.2 |
| set3 | quality_qwen | 39.1 | 0.8 | 1.9 | 42.7 / 35.9 / 36.9 / 44.4 / 35.2 |
| set4 | quality_llama | 0.0 | 23.8 | 99.8 | 0.0 / 0.0 / 0.0 / 0.0 / 0.0 |

**set4 (QuALITY, Llama) degenerates.** Accuracy over all phases seen so far goes 36.3 -> 34.1 ->
25.4 -> 5.8 -> 0.0 after phases 1-5. After phase 4 most replies are "I'll do my best to help
you." and after phase 5 almost all are "I can't help you." (318/622) or "I can help you."
(220/622), with no option letter; after phase 3 the replies were still normal ("A.", "The
correct answer is A"). Its log-perplexity forgetting is also the worst of the four. Worth
comparing with the `most` run of the same set.

## Files

- `results/continual/five_phase/monolithic_update_1k_wincap_least/<run>/`: `r.json`,
  `r.generations.jsonl`, `r.generations_phase{0..4}.jsonl` and their `.scores.json`
  (same layout as the repo).
- `logs/`: the run logs teed by the script.
- Not included: the per-phase memory checkpoints `r_phase{0..4}.memory.pt`
  (~150 MB each for Qwen, ~120 MB for Llama, 2.6 GB total). Available on request.

The 16k sets (set5-8) are running on the same setup and will follow.
