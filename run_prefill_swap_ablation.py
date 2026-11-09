#!/usr/bin/env python3
"""
Prefill Helper Swap Ablation on exp/tri branch
Tests whether swapping main's prefill_document_kv_cache back into exp/tri
resolves the Phase 1 collapse on Qwen3-4B-Instruct-2507.
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from collections import Counter
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer

REPO_ROOT = Path("/home/voquangtri2025/ICLR-Rebuttal/gated-continual-cartridges")
OUTPUT_DIR = Path("/home/voquangtri2025/ICLR-Rebuttal/outputs/diag/prefill_swap_ablation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NEW_P1_ROOT = REPO_ROOT / "outputs/experiments/subkv_sweep_longhealth/budget512_prefill_swap/p01"
NEW_P1_ROOT.mkdir(parents=True, exist_ok=True)

RECIPE = REPO_ROOT / "outputs/recipes/subkv_sweep/e2e_budget512_topt128.yaml"

os.environ["CARTRIDGES_DIR"] = str(REPO_ROOT)
os.environ["CARTRIDGES_OUTPUT_DIR"] = str(REPO_ROOT / "outputs")
os.environ["CARTRIDGES_EAGER_GENERATE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

from cartridges.benchmark.scorers import mc_options, resolve_mc_option
from cartridges.cache import TrainableCache
from cartridges.generation import flex_generate
from cartridges.models import FlexQwen3ForCausalLM, HFModelConfig
from cartridges.initialization.tokenization_utils import (
    MODELS_WITH_THINKING,
    MODEL_TO_CHAT_TEMPLATE,
)

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
MAX_NEW_TOKENS = 32


def build_ablated_cache():
    """Locate existing newly built cache or build if not present."""
    matches = list(NEW_P1_ROOT.glob("**/cache-step0.pt")) or list(NEW_P1_ROOT.glob("**/cache_last.pt"))
    if matches:
        print(f"[OK] Found existing newly built cache: {matches[0]}")
        return matches[0]

    print("=" * 70)
    print("STEP 1: Compacting Phase 1 cache with main's prefill helper...")
    print("=" * 70)

    build_cmd = [
        sys.executable,
        str(REPO_ROOT / "examples/shared/am/build_p01.py"),
        "--dataset", "longhealth",
        "--recipe-config", str(RECIPE),
        "--p1-root", str(NEW_P1_ROOT),
        "--gpu", "0",
        "--force",
    ]

    env = dict(os.environ)
    env.pop("TORCH_COMPILE_DISABLE", None)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["CARTRIDGES_DIR"] = str(REPO_ROOT)
    env["CARTRIDGES_OUTPUT_DIR"] = str(REPO_ROOT / "outputs")
    env["CARTRIDGES_PYTHON"] = sys.executable

    t0 = time.time()
    subprocess.run(build_cmd, cwd=str(REPO_ROOT), env=env, check=True)
    print(f"Build completed in {time.time() - t0:.1f}s.")

    # Locate generated cache
    matches = list(NEW_P1_ROOT.glob("**/cache-step0.pt")) or list(NEW_P1_ROOT.glob("**/cache_last.pt"))
    if not matches:
        raise FileNotFoundError(f"No cache found under {NEW_P1_ROOT}")
    cache_path = matches[0]
    print(f"[OK] Found newly built cache: {cache_path}")
    return cache_path


def load_eval_examples():
    """Load the standardized LongHealth Phase 1 eval set."""
    parquet_path = REPO_ROOT / "data/longhealth/phases/phase1_eval.parquet"
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} evaluation questions from {parquet_path.name}")

    examples = []
    for idx, row in df.iterrows():
        msgs = [{"role": m["role"], "content": m["content"]} for m in row["messages"]]
        meta = dict(row["metadata"])
        examples.append({
            "idx": idx,
            "question_id": meta.get("question_id", f"p1_{idx}"),
            "prompt_messages": msgs[:-1],
            "reference_answer": msgs[-1]["content"],
            "options": [str(opt) for opt in meta.get("options", [])],
            "patient_id": meta.get("patient_id", ""),
        })
    return examples


def evaluate_cache(model, tokenizer, cache_path: Path, examples: List[Dict[str, Any]]):
    print("=" * 70)
    print("STEP 2: Evaluating newly built cache on 80 LongHealth questions...")
    print("=" * 70)

    cache = TrainableCache.from_pretrained(str(cache_path), device=DEVICE).to(DEVICE)
    answer_prime = " Answer:"
    prime_ids = tokenizer(answer_prime, add_special_tokens=False)["input_ids"]

    # Option tokens
    option_letters = ["A", "B", "C", "D", "E"]
    option_token_ids = []
    option_id_to_letter = {}
    for letter in option_letters:
        for fmt in [f" ({letter.lower()})", f" ({letter})", f"({letter.lower()})", f"({letter})", f" {letter}", letter]:
            ids = tokenizer(fmt, add_special_tokens=False)["input_ids"]
            if len(ids) == 1:
                option_token_ids.append(ids[0])
                option_id_to_letter[ids[0]] = letter

    records = []
    option_counts = Counter()
    n_correct = 0

    pbar = tqdm(range(0, len(examples), BATCH_SIZE), desc="evaluating_prefill_swap")
    for offset in pbar:
        batch = examples[offset : offset + BATCH_SIZE]
        input_ids_list, seq_ids_list, pos_ids_list = [], [], []

        kwargs = {}
        if MODEL_NAME in MODELS_WITH_THINKING:
            kwargs["enable_thinking"] = False

        for seq_idx, ex in enumerate(batch):
            ids = tokenizer.apply_chat_template(
                ex["prompt_messages"],
                add_generation_prompt=True,
                return_tensors="pt",
                chat_template=MODEL_TO_CHAT_TEMPLATE.get(tokenizer.name_or_path),
                **kwargs,
            ).flatten().to(DEVICE)

            ids = torch.cat([ids, torch.tensor(prime_ids, device=DEVICE, dtype=ids.dtype)])
            input_ids_list.append(ids)
            seq_ids_list.append(torch.full_like(ids, seq_idx))
            pos_ids_list.append(torch.arange(ids.numel(), device=DEVICE))

        batch_input_ids = torch.cat(input_ids_list)
        batch_seq_ids = torch.cat(seq_ids_list)
        batch_pos_ids = torch.cat(pos_ids_list)

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            gen_res = flex_generate(
                model=model,
                tokenizer=tokenizer,
                input_ids=batch_input_ids,
                seq_ids=batch_seq_ids,
                position_ids=batch_pos_ids,
                cache=cache,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.0,
                collect_token_ids=option_token_ids,
                collect_logits=True,
            )

        if isinstance(gen_res, tuple):
            generated_tokens, collected_logprobs = gen_res
        else:
            generated_tokens = gen_res
            collected_logprobs = None

        for seq_idx, ex in enumerate(batch):
            gen_toks = generated_tokens.get(seq_idx, [])
            gen_text = tokenizer.decode(gen_toks, skip_special_tokens=True)
            gen_text = answer_prime + gen_text

            options = ex["options"]
            ref_ans = ex["reference_answer"]

            score = mc_options(gen_text, ref_ans, metadata={"options": options})
            correct = bool(score == 1.0)
            if correct:
                n_correct += 1

            resolved = resolve_mc_option(gen_text, options)
            letter = None
            if resolved in options:
                letter = chr(ord("A") + options.index(resolved))
            elif gen_text.strip():
                first_char = gen_text.strip()[0].upper()
                if first_char in ["A", "B", "C", "D", "E"]:
                    letter = first_char

            if letter:
                option_counts[letter] += 1
            else:
                option_counts["Unparsed"] += 1

            step0_lps = {}
            if collected_logprobs and seq_idx in collected_logprobs and collected_logprobs[seq_idx]:
                step0 = collected_logprobs[seq_idx][0]
                for tid, l_data in step0.items():
                    opt_letter = option_id_to_letter.get(tid)
                    if opt_letter:
                        lp = l_data["logprob"]
                        if opt_letter not in step0_lps or lp > step0_lps[opt_letter]:
                            step0_lps[opt_letter] = float(lp)

            records.append({
                "question_id": ex["question_id"],
                "generated_text": gen_text,
                "resolved_option": resolved,
                "assigned_letter": letter,
                "reference_answer": ref_ans,
                "correct": correct,
                "choice_step0_logprobs": step0_lps,
            })

    total = len(examples)
    acc = n_correct / total
    pct_a = (option_counts["A"] / total) * 100.0

    print("\n" + "=" * 70)
    print("ABLATION RESULTS (exp/tri with main's prefill helper)")
    print("=" * 70)
    print(f"Accuracy: {n_correct}/{total} ({acc * 100:.2f}%)")
    print(f"Option Distribution: {dict(option_counts)}")
    print(f"Percentage Predicted 'A': {pct_a:.1f}%")

    # Mean logprobs
    avg_lps = {}
    for l in option_letters:
        vals = [r["choice_step0_logprobs"].get(l) for r in records if l in r["choice_step0_logprobs"]]
        avg_lps[l] = float(np.mean(vals)) if vals else None
    print(f"Mean Option Logprobs at step 0: {avg_lps}")

    summary = {
        "num_total": total,
        "num_correct": n_correct,
        "accuracy": acc,
        "option_distribution": dict(option_counts),
        "pct_A": pct_a,
        "mean_step0_logprobs": avg_lps,
        "cache_path": str(cache_path),
    }

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(OUTPUT_DIR / "generations.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"Saved results to {OUTPUT_DIR / 'results.json'}")
    return summary


def main():
    cache_path = build_ablated_cache()

    examples = load_eval_examples()

    print(f"\nLoading model {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = HFModelConfig(
        pretrained_model_name_or_path=MODEL_NAME,
        model_cls=FlexQwen3ForCausalLM,
    ).instantiate().to(DEVICE).to(torch.bfloat16)
    model.eval()

    summary = evaluate_cache(model, tokenizer, cache_path, examples)


if __name__ == "__main__":
    main()

