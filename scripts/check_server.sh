#!/usr/bin/env bash
set -u

echo "System"
uname -a

echo
echo "Python"
python3 --version

echo
echo "NVIDIA GPUs"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi is not available"
fi

echo
echo "CUDA compiler"
if command -v nvcc >/dev/null 2>&1; then
  nvcc --version
else
  echo "nvcc is not available in PATH"
fi

echo
echo "Data directory"
receipt_data_root="${RECEIPT_DATA_ROOT:-$HOME/datos/receipt-benchmark}"
echo "$receipt_data_root"
if [[ -d "$receipt_data_root" ]]; then
  df -h "$receipt_data_root"
else
  echo "Directory does not exist yet"
fi

