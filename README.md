# ICLR Rebuttal: Continual Learning & Cartridge Evaluation

This repository houses the unified experimental harness, comparative evaluation code, experiment tracking registry, and analysis logs for our **ICLR Rebuttal**.

It provides a head-to-head empirical comparison between:
- **Repo A (Cartridges)**: [`gated-continual-cartridges`](https://github.com/qtris123/gated-continual-cartridges) (`main` branch) &mdash; continual learning via sparse, gated KV cartridges.
- **Repo B (Baseline / SubMem)**: [`farid/gated-continual-cartridges`](https://github.com/faridlazuarda/gated-continual-cartridges) (`moe-am` branch) &mdash; monolithic update and compaction memory baselines.

---

## 📁 Repository Structure

```text
ICLR-Rebuttal/
├── .gitmodules                        # Submodule configuration tracking Repo A & Repo B
├── .gitignore                         # Excludes large checkpoint tensors (*.pt) & caches
├── README.md                          # Project documentation and reproduction guide
├── install_driver.sh                  # Host VM NVIDIA driver & CUDA installer
├── launch_experiment.py               # 100% reproducible experiment tracker & runner
│
├── environments/                      # Conda environment specifications & locks
│   ├── cartridges.yml                 # Standalone Conda environment for Repo A (Cartridges)
│   ├── submem.yml                     # Standalone Conda environment for Repo B (SubMem)
│   ├── requirements_cartridges.txt    # Exact pinned pip dependencies for Repo A
│   ├── requirements_submem.txt        # Exact pinned pip dependencies for Repo B
│   └── setup_envs.sh                  # Automated script to build & verify both envs
│
├── gated-continual-cartridges/        # [Submodule] Repo A (Tri): Cartridges (main)
│   ├── cartridges/                    # Core cartridge models, clients, & trainers
│   ├── configs/                       # Experiment recipes & manifests
│   └── environment.yml                # Conda environment definition for cartridges
│
├── farid/
│   └── gated-continual-cartridges/    # [Submodule] Repo B (Farid): MoE-AM / SubMem (moe-am)
│       ├── submem/                    # Core SubMem compaction & continual memory
│       ├── configs/                   # Monolithic update & memory configs
│       └── requirements.txt           # Dependencies for SubMem
│
├── patches/                           # Essential operational patches
│   └── farid_vllm_mem_and_dir_fixes.patch # vLLM GPU memory release & safe dir creation
│
└── experiments/                       # Persistent experimental records
    ├── experiments_registry.tsv       # Master TSV log of all runs with metrics
    └── runs/                          # Granular per-run audit trails
        └── <run_id>/                  # Isolated directory per experiment execution
            ├── run_command.sh         # Exact resolved execution command
            ├── run.log                # Complete stdout/stderr execution log
            ├── env.json               # Host, OS, PyTorch, CUDA, and GPU hardware info
            ├── git_info.json          # Git commit SHA, branch, and status
            ├── patch.diff             # Exact uncommitted git diff at run time
            └── metrics.json           # Extracted evaluation metrics and scores
```

---

## 🚀 Getting Started

### 1. Clone the Repository (with Submodules)

Clone the repository and recursively fetch both nested codebases:

```bash
git clone --recursive https://github.com/qtris123/ICLR-Rebuttal.git
cd ICLR-Rebuttal
```

If already cloned without `--recursive`, initialize submodules with:

```bash
git submodule update --init --recursive
```

### 2. Environment Setup

To ensure 100% reproducible execution, full standalone Conda environment definitions and pinned requirement locks are provided in `environments/`.

| Repository | Conda Env Name | Python | Core Stack | Config File |
| :--- | :--- | :--- | :--- | :--- |
| **Repo A (Cartridges)** | `cartridges` | 3.12 | PyTorch 2.6 (`cu124`), Transformers 4.53, PEFT 0.18, FlashInfer | `environments/cartridges.yml` |
| **Repo B (SubMem)** | `submem` | 3.12 | PyTorch 2.8 (`cu128`), vLLM 0.11, Transformers 4.57, Accelerate 1.12 | `environments/submem.yml` |

#### Option 1: Automated Setup (Recommended)
Run the automated setup script to build both environments and link local packages in editable mode:

```bash
# Set up both conda environments and install editable packages
bash environments/setup_envs.sh all

# Or set up individually:
bash environments/setup_envs.sh cartridges
bash environments/setup_envs.sh submem
```

#### Option 2: Manual Setup

##### Repo A (`cartridges`):
```bash
conda env create -f environments/cartridges.yml
conda activate cartridges
pip install -e gated-continual-cartridges
```

*Verification:*
```bash
python -c "import torch, transformers, peft; print('Cartridges OK: PyTorch', torch.__version__, 'CUDA:', torch.cuda.is_available())"
```

##### Repo B (`submem`):
```bash
conda env create -f environments/submem.yml
conda activate submem
pip install -e farid/gated-continual-cartridges
```

*Verification:*
```bash
python -c "import torch, transformers, vllm; print('SubMem OK: PyTorch', torch.__version__, 'vLLM:', vllm.__version__, 'CUDA:', torch.cuda.is_available())"
```

*Alternative pip-only setup (from clean Python 3.12 environment):*
```bash
conda create -n submem python=3.12 -y && conda activate submem
pip install -r environments/requirements_submem.txt
pip install -e farid/gated-continual-cartridges
```

#### Applying Patches to Repo B:
To ensure stable GPU execution on A100/H100 GPUs (preventing vLLM VRAM lock during query extraction) and auto-creating nested output directories:
```bash
git -C farid/gated-continual-cartridges apply ../../patches/farid_vllm_mem_and_dir_fixes.patch
```

---

## 🔬 Reproducible Experiment Execution

All experiments are dispatched through `launch_experiment.py`, guaranteeing 100% auditability and reproducibility:

```bash
python3 launch_experiment.py --repo {a|b} --name <EXPERIMENT_NAME> -- <COMMAND...>
```

### What `launch_experiment.py` Captures Automatically:
1. **Directory Isolation**: Creates `experiments/runs/<timestamp>_<repo>_<name>_<hash>/`.
2. **Git Snapshot**: Records exact commit SHA, branch name, and uncommitted diff in `patch.diff`.
3. **Hardware & Environment**: Logs GPU model, driver, VRAM, PyTorch version, and CUDA version into `env.json`.
4. **Log Streaming**: Live-tees standard output and standard error to `run.log`.
5. **Results Extraction**: Automatically parses `r.json` and records metrics (`final_acc`, `final_loss`, `forgetting`).
6. **Master Registry**: Appends the summary row to `experiments/experiments_registry.tsv`.

---

## 📊 Example Experiment Commands

### 1. Running Repo B (Farid Monolithic Update Baseline)

Run the 5-phase LongHealth continual benchmark under a 1,024 memory token budget on LLaMA 3.2 3B:

```bash
conda activate submem

python3 launch_experiment.py \
  --repo b \
  --name "mono_1k_longhealth_llama" \
  -- python3 run_continual.py \
       --config configs/monolithic_update.yaml \
       --stream data/streams/longhealth_5phase.json \
       --model meta-llama/Llama-3.2-3B-Instruct \
       --budget 1024 \
       --write-slots 256 \
       --vllm-mem 0.36 \
       --out results/continual/five_phase/monolithic_update_1k/longhealth_llama/r.json \
       --generate-out results/continual/five_phase/monolithic_update_1k/longhealth_llama/r.generations.jsonl
```

### 2. Running Repo A (Gated Cartridges Evaluation)

Run cartridge continual evaluation on LongHealth:

```bash
conda activate cartridges

python3 launch_experiment.py \
  --repo a \
  --name "cartridges_longhealth_llama" \
  -- python3 -m cartridges.evaluate \
       --config configs/longhealth.yaml
```

---

## 📈 Experiments Registry

Review the master experiment registry anytime:

```bash
column -t -s $'\t' experiments/experiments_registry.tsv | less -S
```

