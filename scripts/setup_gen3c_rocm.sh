#!/usr/bin/env bash
set -euo pipefail

# Default locations can be overridden via environment variables.
GEN3C_DIR=${GEN3C_DIR:-"$HOME/GEN3C"}
ENV_NAME=${ENV_NAME:-"gen3c-rocm"}
PYTHON_VERSION=${PYTHON_VERSION:-"3.10"}
ROCM_CHANNEL=${ROCM_CHANNEL:-"https://download.pytorch.org/whl/rocm6.1"}

if [[ ! -d "$GEN3C_DIR" ]]; then
  echo "ERROR: GEN3C directory not found at $GEN3C_DIR" >&2
  exit 1
fi

# Ensure conda is available in the current shell.
if [[ -z "${CONDA_EXE:-}" ]]; then
  if [[ -f "$HOME/opt/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck source=/dev/null
    source "$HOME/opt/miniconda3/etc/profile.d/conda.sh"
  elif command -v conda >/dev/null 2>&1; then
    eval "$(command conda 'shell.bash' hook 2>/dev/null)"
  else
    echo "ERROR: conda not found. Please install Miniconda/Anaconda first." >&2
    exit 1
  fi
fi

if ! conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  echo "Creating Conda environment $ENV_NAME (Python $PYTHON_VERSION)..."
  conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
else
  echo "Conda environment $ENV_NAME already exists (skipping creation)."
fi

PIP_CMD=(conda run -n "$ENV_NAME" python -m pip)

echo "Upgrading pip inside $ENV_NAME..."
"${PIP_CMD[@]}" install --upgrade pip

echo "Installing ROCm builds of PyTorch (torch/torchvision)..."
"${PIP_CMD[@]}" install \
  --extra-index-url "$ROCM_CHANNEL" \
  "torch==2.4.1+rocm6.1" \
  "torchvision==0.19.1+rocm6.1"

# Filter out CUDA-only wheels that have no ROCm equivalent.
FILTERED_REQ_FILE=$(mktemp)
grep -Ev '^(torch==|torchvision==|triton==|warp-lang==|nvidia-)' \
  "$GEN3C_DIR/requirements.txt" > "$FILTERED_REQ_FILE"

echo "Installing remaining GEN3C Python requirements (ROCm path)..."
"${PIP_CMD[@]}" install -r "$FILTERED_REQ_FILE"
rm -f "$FILTERED_REQ_FILE"

# Remove CUDA-only helper packages that break ROCm dependency resolution.
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

cat <<'MSG'

ROCm setup complete!

Next steps:
  1. export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}
  2. export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:512
  3. conda run -n gen3c-rocm python scripts/test_environment.py

MSG

