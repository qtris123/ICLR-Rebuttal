#!/usr/bin/env python3
"""Five-phase progression of the monolithic update under write-least vs write-most.

For each (dataset, model) and each budget (1,024 and 16,384 slots) it plots, over
the five update phases:
  - solid line: accuracy on phase 1, read after each phase (phase-1 retention);
  - dashed line: accuracy on the current (newest) phase at each phase (the diagonal).

Write-least and write-most are drawn in two colors so the two writable-slot rules
can be compared directly. The write-least runs are these ablation folders; the
write-most runs are the main-text monolithic update at the matching budget.

Output: write_rule_progression.png next to this script.

Run:
  python plot_write_rule_progression.py
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
# results/ root, four levels up from this ablation folder.
RESULTS = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))

LEAST = os.path.join(RESULTS, "submissions", "ablation", "random-write_or_write_least")
MOST = os.path.join(RESULTS, "submissions", "main_text", "farid_5_phase")

BUDGETS = [
    ("1,024 slots", "monolithic_update_1k_wincap_least", "monolithic_update_1k_wincap"),
    ("16,384 slots", "monolithic_update_16k_wincap_least", "monolithic_update_16k_wincap"),
]
COMBOS = [
    ("longhealth_qwen", "LongHealth, Qwen3-4B"),
    ("longhealth_llama", "LongHealth, Llama-3.2-3B"),
    ("quality_qwen", "QuALITY, Qwen3-4B"),
    ("quality_llama", "QuALITY, Llama-3.2-3B"),
]
PHASES = [1, 2, 3, 4, 5]

COLOR_LEAST = "#d62728"  # write-least
COLOR_MOST = "#1f77b4"   # write-most


def series(run_dir):
    """(phase-1 accuracy per phase, current-phase accuracy per phase), in %."""
    phase1, diagonal = [], []
    for p in range(5):
        with open(os.path.join(run_dir, f"r.generations_phase{p}.scores.json")) as f:
            s = json.load(f)
        keys = sorted(
            (k for k in s if not k.startswith("_")),
            key=lambda k: int(k.rsplit(" ", 1)[1]),
        )
        phase1.append(100 * s[keys[0]]["score"])
        diagonal.append(100 * s[keys[-1]]["score"])
    return phase1, diagonal


def main():
    fig, axes = plt.subplots(
        len(BUDGETS), len(COMBOS), figsize=(16, 7), sharex=True, sharey=True
    )
    for r, (budget_label, least_name, most_name) in enumerate(BUDGETS):
        for c, (combo, combo_label) in enumerate(COMBOS):
            ax = axes[r][c]
            l_p1, l_diag = series(os.path.join(LEAST, least_name, combo))
            m_p1, m_diag = series(os.path.join(MOST, most_name, combo))
            ax.plot(PHASES, l_p1, "-", color=COLOR_LEAST, marker="o", label="write-least, phase 1")
            ax.plot(PHASES, l_diag, "--", color=COLOR_LEAST, marker="o", label="write-least, current phase")
            ax.plot(PHASES, m_p1, "-", color=COLOR_MOST, marker="s", label="write-most, phase 1")
            ax.plot(PHASES, m_diag, "--", color=COLOR_MOST, marker="s", label="write-most, current phase")
            ax.set_title(f"{combo_label}\n{budget_label}", fontsize=9)
            ax.set_xticks(PHASES)
            ax.set_ylim(0, 80)
            ax.grid(True, alpha=0.3)
            if c == 0:
                ax.set_ylabel("Accuracy (%)")
            if r == len(BUDGETS) - 1:
                ax.set_xlabel("Phase")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(HERE, "write_rule_progression.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
