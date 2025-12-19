#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Hunyuan3D 2D→3D Gradio UI Launcher
#
# Multi-System Configuration (searidge cluster):
#   - searidge02 (CORE): Primary UI host - arkrunr02
#   - searidge01 (WORKER01): Worker node - arkrunr01
#   - searidge03 (WORKER02): Worker node - arkrunr03
#
# This script launches the Gradio web interface on the current node.
# All paths point to NFS shared storage at /srv/searidge_share
# =============================================================================

# Shared storage paths
SEARIDGE_SHARE="/srv/searidge_share"
PROJECT_DIR="${SEARIDGE_SHARE}/projects/Hunyuan3D-2-Fork"

# Conda environment - use gen3c-rocm310 (Python 3.10, matches NVIDIA stack)
# gen3c-rocm310: Python 3.10, NumPy 1.26.4 - compatible with GEN3C/Lyra/TRELLIS.2/SHARP
# gen3c-rocm: Python 3.12, NumPy 2.3.3 - NOT compatible with RunPod unified image
ENV_NAME="${ENV_NAME:-gen3c-rocm310}"

# ROCm configuration for AMD RX 6900 XT (gfx1030)
export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-10.3.0}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-max_split_size_mb:512}"

# HuggingFace cache on shared storage
export HF_HOME="${SEARIDGE_SHARE}/checkpoints/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}"

# Ensure config module is importable
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

# =============================================================================
# CONDA INITIALIZATION
# =============================================================================
# Find and source conda
CONDA_PATHS=(
  "$HOME/mambaforge/etc/profile.d/conda.sh"
  "$HOME/opt/miniconda3/etc/profile.d/conda.sh"
  "$HOME/miniconda3/etc/profile.d/conda.sh"
  "/opt/conda/etc/profile.d/conda.sh"
)

for conda_path in "${CONDA_PATHS[@]}"; do
  if [[ -f "$conda_path" ]]; then
    source "$conda_path"
    break
  fi
done

# Verify conda is available
if ! command -v conda &> /dev/null; then
  echo "ERROR: conda not found. Please install Mambaforge." >&2
  exit 1
fi

# Get conda prefix for library paths
CONDA_PREFIX="$(conda run -n "$ENV_NAME" printenv CONDA_PREFIX 2>/dev/null || echo "")"
if [[ -n "$CONDA_PREFIX" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${CONDA_PREFIX}/lib64:${LD_LIBRARY_PATH:-}"
fi

# =============================================================================
# LAUNCH
# =============================================================================
echo "=============================================="
echo "Hunyuan3D 2D→3D Launcher - searidge cluster"
echo "=============================================="
echo "Node: $(hostname)"
echo "User: $(whoami)"
echo "Project: $PROJECT_DIR"
echo "Conda Env: $ENV_NAME"
echo "GPU Device: $HIP_VISIBLE_DEVICES"
echo "HF Cache: $HF_HOME"
echo "=============================================="

cd "$PROJECT_DIR"
exec conda run -n "$ENV_NAME" python "${PROJECT_DIR}/2d3d.py"
