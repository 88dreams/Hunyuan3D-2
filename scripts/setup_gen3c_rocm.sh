#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# GEN3C ROCm Environment Setup Script
# 
# Multi-System Configuration (searidge cluster):
#   - searidge02 (CORE): Primary host, arkrunr02
#   - searidge01 (WORKER01): Worker node, arkrunr01
#   - searidge03 (WORKER02): Worker node, arkrunr03
#
# This script sets up the gen3c-rocm Conda environment with:
#   - Python 3.10
#   - ROCm PyTorch wheels
#   - GEN3C dependencies (excluding CUDA-only packages)
#   - ROCm Apex fork
# =============================================================================

# Shared storage paths
SEARIDGE_SHARE="/srv/searidge_share"

# Default locations - can be overridden via environment variables
GEN3C_DIR="${GEN3C_DIR:-${SEARIDGE_SHARE}/projects/GEN3C}"
ENV_NAME="${ENV_NAME:-gen3c-rocm}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
ROCM_CHANNEL="${ROCM_CHANNEL:-https://download.pytorch.org/whl/rocm6.1}"

echo "=============================================="
echo "GEN3C ROCm Setup - searidge cluster"
echo "=============================================="
echo "Node: $(hostname)"
echo "User: $(whoami)"
echo "GEN3C Dir: $GEN3C_DIR"
echo "Conda Env: $ENV_NAME"
echo "Python: $PYTHON_VERSION"
echo "=============================================="

# Validate GEN3C directory exists on shared storage
if [[ ! -d "$GEN3C_DIR" ]]; then
  echo "ERROR: GEN3C directory not found at $GEN3C_DIR" >&2
  echo "       Expected on NFS share. Check mount: ls ${SEARIDGE_SHARE}/projects/" >&2
  exit 1
fi

# =============================================================================
# CONDA INITIALIZATION
# =============================================================================
# Ensure conda is available in the current shell
if [[ -z "${CONDA_EXE:-}" ]]; then
  # Try common Mambaforge/Miniconda locations
  CONDA_PATHS=(
    "$HOME/mambaforge/etc/profile.d/conda.sh"
    "$HOME/opt/miniconda3/etc/profile.d/conda.sh"
    "$HOME/miniconda3/etc/profile.d/conda.sh"
    "/opt/conda/etc/profile.d/conda.sh"
  )
  
  CONDA_FOUND=false
  for conda_path in "${CONDA_PATHS[@]}"; do
    if [[ -f "$conda_path" ]]; then
      echo "Found conda at: $conda_path"
      # shellcheck source=/dev/null
      source "$conda_path"
      CONDA_FOUND=true
      break
    fi
  done
  
  if [[ "$CONDA_FOUND" == "false" ]]; then
    if command -v conda >/dev/null 2>&1; then
      eval "$(command conda 'shell.bash' hook 2>/dev/null)"
    else
      echo "ERROR: conda not found. Please install Mambaforge first." >&2
      echo "       See docs/localai-multi-system-setup.md section 5.1" >&2
      exit 1
    fi
  fi
fi

# =============================================================================
# ENVIRONMENT CREATION
# =============================================================================
if ! conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  echo "Creating Conda environment $ENV_NAME (Python $PYTHON_VERSION)..."
  conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
else
  echo "Conda environment $ENV_NAME already exists (skipping creation)."
fi

PIP_CMD=(conda run -n "$ENV_NAME" python -m pip)

echo "Upgrading pip inside $ENV_NAME..."
"${PIP_CMD[@]}" install --upgrade pip

# =============================================================================
# PYTORCH ROCm INSTALLATION
# =============================================================================
echo "Installing ROCm builds of PyTorch (torch/torchvision)..."
"${PIP_CMD[@]}" install \
  --extra-index-url "$ROCM_CHANNEL" \
  "torch==2.4.1+rocm6.1" \
  "torchvision==0.19.1+rocm6.1"

# =============================================================================
# GEN3C DEPENDENCIES
# =============================================================================
# Filter out CUDA-only wheels that have no ROCm equivalent
FILTERED_REQ_FILE=$(mktemp)
grep -Ev '^(torch==|torchvision==|triton==|warp-lang==|nvidia-)' \
  "$GEN3C_DIR/requirements.txt" > "$FILTERED_REQ_FILE"

echo "Installing remaining GEN3C Python requirements (ROCm path)..."
"${PIP_CMD[@]}" install -r "$FILTERED_REQ_FILE"
rm -f "$FILTERED_REQ_FILE"

# Remove CUDA-only helper packages that break ROCm dependency resolution
if "${PIP_CMD[@]}" show nvidia-modelopt >/dev/null 2>&1; then
  echo "Removing nvidia-modelopt (CUDA-only) from ROCm environment..."
  "${PIP_CMD[@]}" uninstall -y nvidia-modelopt
fi

# Some dependencies reinstall CUDA wheels. Force ROCm builds back into place.
echo "Re-installing ROCm torch packages to override CUDA-only wheels..."
"${PIP_CMD[@]}" install --upgrade --force-reinstall \
  --extra-index-url "$ROCM_CHANNEL" \
  "torch==2.4.1+rocm6.1" \
  "torchvision==0.19.1+rocm6.1"

# =============================================================================
# APEX INSTALLATION (ROCm Fork)
# =============================================================================
# Use a local extensions directory within the GEN3C project
EXT_DIR="$GEN3C_DIR/.ext"
APEX_DIR="$EXT_DIR/apex-rocm"
mkdir -p "$EXT_DIR"

if [[ ! -d "$APEX_DIR/.git" ]]; then
  echo "Cloning ROCm Apex fork..."
  git clone https://github.com/ROCmSoftwarePlatform/apex "$APEX_DIR"
else
  echo "Updating existing ROCm Apex checkout..."
  git -C "$APEX_DIR" fetch --all --prune
  git -C "$APEX_DIR" reset --hard origin/master
fi

echo "Building & installing ROCm Apex..."
if ! conda run -n "$ENV_NAME" bash -c "cd \"$APEX_DIR\" && python setup.py install --cpp_ext --cuda_ext"; then
  echo "Apex extension build failed, falling back to python-only install..." >&2
  conda run -n "$ENV_NAME" bash -c "cd \"$APEX_DIR\" && python setup.py install"
fi

# =============================================================================
# COMPLETION
# =============================================================================
cat <<MSG

==============================================
ROCm setup complete on $(hostname)!
==============================================

Environment: $ENV_NAME
GEN3C Dir: $GEN3C_DIR
Checkpoints: ${SEARIDGE_SHARE}/checkpoints/gen3c

Next steps:

1. Set ROCm environment variables:
   export HIP_VISIBLE_DEVICES=0
   export PYTORCH_ALLOC_CONF=max_split_size_mb:512
   export HSA_OVERRIDE_GFX_VERSION=10.3.0

2. Activate the environment:
   conda activate $ENV_NAME

3. Test the environment:
   python -c "import torch; print('ROCm:', torch.cuda.is_available())"

4. Run GEN3C:
   cd ${SEARIDGE_SHARE}/projects/Hunyuan3D-2-Fork
   ./scripts/run_gen3c.sh --input /path/to/image.png --video-name test

For multi-node setup, repeat this script on:
  - searidge01 (arkrunr01)
  - searidge02 (arkrunr02) 
  - searidge03 (arkrunr03)

MSG
