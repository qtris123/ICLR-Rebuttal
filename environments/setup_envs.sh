#!/usr/bin/env bash
# ==============================================================================
# ICLR Rebuttal: Automated Conda Environment Setup & Verification
# ==============================================================================
# Sets up and verifies the Conda environments for both experimental codebases:
#   - Repo A: Cartridges (conda env: cartridges, Python 3.12, PyTorch 2.6)
#   - Repo B: Farid SubMem (conda env: submem, Python 3.12, PyTorch 2.8, vLLM 0.11)
#
# Usage:
#   bash environments/setup_envs.sh all          # Set up both environments
#   bash environments/setup_envs.sh cartridges   # Set up cartridges environment only
#   bash environments/setup_envs.sh submem       # Set up submem environment only
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

TARGET="${1:-all}"

# Ensure conda command is accessible
if ! command -v conda &>/dev/null; then
    if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
        source "${HOME}/miniconda3/etc/profile.d/conda.sh"
    elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
        source "${HOME}/anaconda3/etc/profile.d/conda.sh"
    elif [[ -f "/opt/conda/etc/profile.d/conda.sh" ]]; then
        source "/opt/conda/etc/profile.d/conda.sh"
    else
        echo "Error: conda could not be found in PATH or standard install locations."
        exit 1
    fi
fi

UPDATE_EXISTING=0
for arg in "$@"; do
    if [[ "$arg" == "--update" || "$arg" == "-u" ]]; then
        UPDATE_EXISTING=1
    fi
done

setup_cartridges() {
    echo "=========================================================="
    echo " Setting up Conda environment: cartridges (Repo A)"
    echo "=========================================================="
    if conda info --envs | grep -E '^cartridges\s' >/dev/null 2>&1; then
        if [[ $UPDATE_EXISTING -eq 1 ]]; then
            echo "Environment 'cartridges' exists. Updating packages (--update requested)..."
            conda env update -n cartridges -f "${SCRIPT_DIR}/cartridges.yml" --prune
        else
            echo "Environment 'cartridges' already exists. Verifying packages (use --update to re-sync)..."
        fi
    else
        echo "Creating 'cartridges' environment from ${SCRIPT_DIR}/cartridges.yml..."
        conda env create -f "${SCRIPT_DIR}/cartridges.yml"
    fi

    echo "Installing gated-continual-cartridges in editable mode..."
    conda run -n cartridges pip install -e "${ROOT_DIR}/gated-continual-cartridges"

    echo "Verifying cartridges environment..."
    conda run -n cartridges python -c "
import torch, transformers, peft, accelerate
print('✓ Cartridges Environment Verified:')
print(f'    PyTorch:      {torch.__version__} (CUDA available: {torch.cuda.is_available()})')
print(f'    Transformers: {transformers.__version__}')
print(f'    PEFT:         {peft.__version__}')
print(f'    Accelerate:   {accelerate.__version__}')
"
    echo "Cartridges setup complete!"
}

setup_submem() {
    echo "=========================================================="
    echo " Setting up Conda environment: submem (Repo B)"
    echo "=========================================================="
    if conda info --envs | grep -E '^submem\s' >/dev/null 2>&1; then
        if [[ $UPDATE_EXISTING -eq 1 ]]; then
            echo "Environment 'submem' exists. Updating packages (--update requested)..."
            conda env update -n submem -f "${SCRIPT_DIR}/submem.yml" --prune
        else
            echo "Environment 'submem' already exists. Verifying packages (use --update to re-sync)..."
        fi
    else
        echo "Creating 'submem' environment from ${SCRIPT_DIR}/submem.yml..."
        conda env create -f "${SCRIPT_DIR}/submem.yml"
    fi

    echo "Installing farid/gated-continual-cartridges (submem) in editable mode..."
    conda run -n submem pip install -e "${ROOT_DIR}/farid/gated-continual-cartridges"

    echo "Verifying submem environment..."
    conda run -n submem python -c "
import torch, transformers, vllm, accelerate
print('✓ SubMem Environment Verified:')
print(f'    PyTorch:      {torch.__version__} (CUDA available: {torch.cuda.is_available()})')
print(f'    Transformers: {transformers.__version__}')
print(f'    vLLM:         {vllm.__version__}')
print(f'    Accelerate:   {accelerate.__version__}')
"
    echo "SubMem setup complete!"
}

case "$TARGET" in
    cartridges)
        setup_cartridges
        ;;
    submem)
        setup_submem
        ;;
    all)
        setup_cartridges
        echo ""
        setup_submem
        ;;
    *)
        echo "Usage: $0 [cartridges|submem|all]"
        exit 1
        ;;
esac
