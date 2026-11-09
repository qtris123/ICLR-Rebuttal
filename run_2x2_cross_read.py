#!/usr/bin/env python3
"""
2x2 Cross-Read Investigation Script for Qwen P1 Collapse
Tests:
  - Cache: [exp/tri, main]
  - Prompt: [exp/tri, main]
Under identical reader, cap, and scorer, recording:
  - Raw generations
  - Per-choice label log-probabilities (A, B, C, D, E)
  - Accuracy and option distribution (detecting all-A collapse)
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from collections import Counter, defaultdict
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer

# Environment setup
REPO_ROOT = Path("/home/voquangtri2025/ICLR-Rebuttal/gated-continual-cartridges")
GATED_MAIN_ROOT = Path("/home/voquangtri2025/ICLR-Rebuttal/gated-main")
OUTPUT_BASE = Path("/home/voquangtri2025/ICLR-Rebuttal/outputs/diag/2x2_cross_read")
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

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

CACHE_EXPTRI = Path(
    "/home/voquangtri2025/ICLR-Rebuttal/gated-continual-cartridges/outputs/experiments/"
    "subkv_sweep_longhealth/budget512/p01/2026-11-08-00-19-26-initial_am_compaction/"
    "a823a676-e132-45b4-a904-114e1a425fd9/cache-step0.pt"
)

CACHE_MAIN_DIR = GATED_MAIN_ROOT / "outputs/experiments/subkv_sweep_longhealth/budget512_main/p01"
CACHE_MAIN = CACHE_MAIN_DIR / "cache-step0.pt"


def ensure_main_cache_built():
    """Build the main branch 512-slot P1 cache if not already present."""
    if CACHE_MAIN.exists():
        print(f"[OK] Found existing main P1 cache: {CACHE_MAIN}")
        return

    # Check if cache_last.pt exists under CACHE_MAIN_DIR
    found = list(CACHE_MAIN_DIR.glob("**/cache-step0.pt")) or list(CACHE_MAIN_DIR.glob("**/cache_last.pt"))
    if found:
        print(f"[OK] Found existing main P1 cache: {found[0]}")
        return

    print("\n" + "=" * 70)
    print("Building Phase 1 cache on `main` branch (~2 min)...")
    print("=" * 70)
    build_cmd = [
        sys.executable,
        str(GATED_MAIN_ROOT / "examples/shared/am/build_p01.py"),
        "--dataset", "longhealth",
        "--recipe-config", str(GATED_MAIN_ROOT / "outputs/recipes/subkv_sweep/e2e_budget512_topt128.yaml"),
        "--p1-root", str(CACHE_MAIN_DIR),
        "--gpu", "0",
        "--force",
    ]
    env = dict(os.environ)
    env.pop("TORCH_COMPILE_DISABLE", None)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["CARTRIDGES_DIR"] = str(GATED_MAIN_ROOT)
    env["CARTRIDGES_OUTPUT_DIR"] = str(GATED_MAIN_ROOT / "outputs")
    env["CARTRIDGES_PYTHON"] = sys.executable

    res = subprocess.run(build_cmd, cwd=str(GATED_MAIN_ROOT), env=env, check=True)
    print("Main cache build complete.")


def load_eval_examples():
    """Load both prompt variants for the 80 Phase 1 questions."""
    # 1. exp/tri prompt (standardized lettered (a)-(e) with Answer: prime)
    exptri_parquet = REPO_ROOT / "data/longhealth/phases/phase1_eval.parquet"
    df_exptri = pd.read_parquet(exptri_parquet)
    print(f"Loaded {len(df_exptri)} examples for exp/tri prompt from {exptri_parquet.name}")

    exptri_examples = []
    for idx, row in df_exptri.iterrows():
        msgs = [{"role": m["role"], "content": m["content"]} for m in row["messages"]]
        meta = dict(row["metadata"])
        exptri_examples.append({
            "idx": idx,
            "question_id": meta.get("question_id", f"p1_{idx}"),
            "prompt_messages": msgs[:-1],
            "reference_answer": msgs[-1]["content"],
            "options": [str(opt) for opt in meta.get("options", [])],
            "patient_id": meta.get("patient_id", ""),
        })

    # 2. main prompt (original XML <options> and <answer> format from benchmark source)
    with open(REPO_ROOT / "data/longhealth/benchmark_v5.json") as f:
        bench_data = json.load(f)

    main_parquet = REPO_ROOT / "data/longhealth/eval/patients_01_to_10.parquet"
    df_main_raw = pd.read_parquet(main_parquet)
    # Filter to patients 1-4 (80 questions)
    p1_pids = {"patient_01", "patient_02", "patient_03", "patient_04"}
    df_main = df_main_raw[df_main_raw["metadata"].apply(lambda m: m.get("patient_id") in p1_pids)].copy()
    print(f"Loaded {len(df_main)} examples for main prompt from {main_parquet.name}")

    main_examples = []
    for idx, (_, row) in enumerate(df_main.iterrows()):
        msgs = [{"role": m["role"], "content": m["content"]} for m in row["messages"]]
        meta = dict(row["metadata"])
        pid = meta.get("patient_id")
        qid = meta.get("question_id")
        # Extract options from benchmark_v5.json
        qnum = int(qid.split("_")[-1])
        q_bench = bench_data[pid]["questions"][qnum]
        options = [
            q_bench["answer_a"],
            q_bench["answer_b"],
            q_bench["answer_c"],
            q_bench["answer_d"],
            q_bench["answer_e"],
        ]
        main_examples.append({
            "idx": idx,
            "question_id": qid,
            "prompt_messages": msgs[:-1],
            "reference_answer": msgs[-1]["content"],
            "options": options,
            "patient_id": pid,
        })

    return exptri_examples, main_examples


def find_resolved_main_cache_path():
    """Return the exact path to cache-step0.pt or cache_last.pt for main."""
    for cand in [
        CACHE_MAIN,
        CACHE_MAIN_DIR / "cache_last.pt",
    ]:
        if cand.exists():
            return cand
    matches = list(CACHE_MAIN_DIR.glob("**/cache-step0.pt")) or list(CACHE_MAIN_DIR.glob("**/cache_last.pt"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not find main cache under {CACHE_MAIN_DIR}")


def run_cell_evaluation(
    model,
    tokenizer,
    cache_path: Path,
    cache_label: str,
    examples: List[Dict[str, Any]],
    prompt_label: str,
    answer_prime: str,
    option_token_ids: List[int],
    option_id_to_letter: Dict[int, str],
):
    print(f"\nEvaluating Cell: [{cache_label}] x [{prompt_label}] (80 examples)...")
    cache = TrainableCache.from_pretrained(str(cache_path), device=DEVICE).to(DEVICE)

    prime_ids = (
        tokenizer(answer_prime, add_special_tokens=False)["input_ids"]
        if answer_prime
        else None
    )

    records = []
    option_counts = Counter()
    n_correct = 0

    pbar = tqdm(range(0, len(examples), BATCH_SIZE), desc=f"{cache_label}_x_{prompt_label}")
    for offset in pbar:
        batch = examples[offset : offset + BATCH_SIZE]
        
        # Tokenize batch
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

            if prime_ids:
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
            if answer_prime:
                gen_text = answer_prime + gen_text

            options = ex["options"]
            ref_ans = ex["reference_answer"]

            # Score using standard benchmark scorer
            score = mc_options(gen_text, ref_ans, metadata={"options": options})
            correct = bool(score == 1.0)
            if correct:
                n_correct += 1

            # Resolve option letter (A, B, C, D, E)
            resolved = resolve_mc_option(gen_text, options)
            letter = None
            if resolved in options:
                letter = chr(ord("A") + options.index(resolved))
            elif gen_text.strip():
                # check if begins with letter
                first_char = gen_text.strip()[0].upper()
                if first_char in ["A", "B", "C", "D", "E"]:
                    letter = first_char

            if letter:
                option_counts[letter] += 1
            else:
                option_counts["Unparsed"] += 1

            # Extract logprobs of choice options at the first generation step
            step0_lps = {}
            if collected_logprobs and seq_idx in collected_logprobs and collected_logprobs[seq_idx]:
                step0 = collected_logprobs[seq_idx][0]  # token_id -> {logit, logprob}
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

    accuracy = n_correct / len(examples)
    summary = {
        "cache": cache_label,
        "prompt": prompt_label,
        "num_total": len(examples),
        "num_correct": n_correct,
        "accuracy": accuracy,
        "option_distribution": dict(option_counts),
        "pct_A": (option_counts["A"] / len(examples)) * 100,
    }

    # Save cell results
    cell_id = f"{cache_label}__x__{prompt_label}"
    cell_dir = OUTPUT_BASE / cell_id
    cell_dir.mkdir(parents=True, exist_ok=True)
    with (cell_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    with (cell_dir / "generations.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"  -> Accuracy: {n_correct}/{len(examples)} ({accuracy * 100:.2f}%)")
    print(f"  -> Choice Distribution: {dict(option_counts)} (% Option A: {summary['pct_A']:.1f}%)")
    return summary


def main():
    print("=" * 70)
    print("QWEN P1 2x2 CROSS-READ INVESTIGATION")
    print(f"Model: {MODEL_NAME} | Device: {DEVICE}")
    print("=" * 70)

    # 1. Ensure main cache is built
    ensure_main_cache_built()
    main_cache_path = find_resolved_main_cache_path()
    print(f"Using exp/tri cache: {CACHE_EXPTRI}")
    print(f"Using main cache:    {main_cache_path}")

    # 2. Load eval examples for both prompts
    exptri_examples, main_examples = load_eval_examples()

    # 3. Load model and tokenizer once
    print(f"\nLoading model {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = HFModelConfig(
        pretrained_model_name_or_path=MODEL_NAME,
        model_cls=FlexQwen3ForCausalLM,
    ).instantiate().to(DEVICE).to(torch.bfloat16)
    model.eval()

    # Build option token map
    option_id_to_letter = {}
    letters = ["A", "B", "C", "D", "E"]
    option_token_ids = []
    for L in letters:
        for variant in [L, f" {L}", f"({L})", L.lower(), f"({L.lower()})"]:
            enc = tokenizer.encode(variant, add_special_tokens=False)
            if enc:
                tid = enc[0]
                option_token_ids.append(tid)
                option_id_to_letter[tid] = L
    option_token_ids = list(set(option_token_ids))

    # 4. Evaluate 2x2 grid
    grid_configs = [
        # (cache_path, cache_label, examples, prompt_label, answer_prime)
        (CACHE_EXPTRI, "exptri_cache", exptri_examples, "exptri_prompt", " Answer:"),
        (CACHE_EXPTRI, "exptri_cache", main_examples, "main_prompt", ""),
        (main_cache_path, "main_cache", exptri_examples, "exptri_prompt", " Answer:"),
        (main_cache_path, "main_cache", main_examples, "main_prompt", ""),
    ]

    all_summaries = []
    for c_path, c_label, exs, p_label, prime in grid_configs:
        s = run_cell_evaluation(
            model=model,
            tokenizer=tokenizer,
            cache_path=c_path,
            cache_label=c_label,
            examples=exs,
            prompt_label=p_label,
            answer_prime=prime,
            option_token_ids=option_token_ids,
            option_id_to_letter=option_id_to_letter,
        )
        all_summaries.append(s)

    # 5. Print final 2x2 summary table
    print("\n" + "=" * 78)
    print("2x2 CROSS-READ MATRIX RESULTS SUMMARY (80 Phase 1 LongHealth questions)")
    print("=" * 78)
    header = f"{'Cache':<15} | {'Prompt':<15} | {'Correct':<10} | {'Accuracy':<10} | {'% Pred A':<10} | {'Distribution'}"
    print(header)
    print("-" * 78)
    for s in all_summaries:
        dist_str = " ".join(f"{k}:{v}" for k, v in sorted(s['option_distribution'].items()))
        print(f"{s['cache']:<15} | {s['prompt']:<15} | {s['num_correct']:>2}/{s['num_total']}     | {s['accuracy']*100:>5.2f}%    | {s['pct_A']:>6.1f}%    | {dist_str}")
    print("=" * 78)

    # Save final matrix JSON
    with (OUTPUT_BASE / "2x2_matrix_results.json").open("w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nWrote full 2x2 diagnostic results to: {OUTPUT_BASE / '2x2_matrix_results.json'}")


if __name__ == "__main__":
    main()
