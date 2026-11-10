#!/bin/bash
# Stop on any error
set -e

# --- Check if driver is already working ---
if nvidia-smi &> /dev/null; then
    echo "NVIDIA driver is already installed and working."
    nvidia-smi
    exit 0
fi

echo "--- Starting NVIDIA Driver and CUDA Toolkit Installation ---"

# --- 1. Setup Installer Directory ---
echo "Setting up installer directory..."
mkdir -p /opt/google/cuda-installer
cd /opt/google/cuda-installer/

# --- 2. Download the GCP Installer ---
echo "Downloading the installer..."
if [ ! -f "cuda_installer.pyz" ]; then
    curl -fSsL -O https://storage.googleapis.com/compute-gpu-installation-us/installer/latest/cuda_installer.pyz
else
    echo "Installer already downloaded."
fi

# --- 3. Install the NVIDIA Driver ---
# This is the step that failed before. We are adding --installation-mode=binary
echo "Installing the NVIDIA driver..."
sudo python3 cuda_installer.pyz install_driver --installation-mode=binary

# --- 4. Install the CUDA Toolkit ---
# This installs the libraries needed for development (nvcc, etc.)
echo "Installing the CUDA Toolkit..."
sudo python3 cuda_installer.pyz install_cuda

# --- 5. Done ---
echo "Installation complete."
echo "If the driver does not immediately appear in nvidia-smi, you may manually reboot with 'sudo reboot'."

