#!/bin/bash
# =============================================================================
# GEN3C Container Startup Script for RunPod
# =============================================================================
# This script runs when the container starts on RunPod.
# It sets up the environment, creates necessary symlinks, and starts the server.
# =============================================================================

set -e

echo "=============================================="
echo "GEN3C RunPod Container Starting"
echo "=============================================="
echo "Timestamp: $(date)"
echo "Hostname: $(hostname)"

# =============================================================================
# Activate Conda Environment
# =============================================================================
echo ""
echo "Activating conda environment..."
source ~/miniforge3/bin/activate cosmos-predict1

echo "  ✓ Conda environment: $CONDA_DEFAULT_ENV"
echo "  ✓ Python: $(which python)"
echo "  ✓ PyTorch version: $(python -c 'import torch; print(torch.__version__)')"

# =============================================================================
# GPU Detection
# =============================================================================
echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "  ⚠ No GPU detected via nvidia-smi"

# Verify CUDA is available to PyTorch
python -c "import torch; print(f'  ✓ CUDA available: {torch.cuda.is_available()}'); print(f'  ✓ GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# =============================================================================
# Environment Setup
# =============================================================================
echo ""
echo "Setting up environment..."

export PYTHONPATH="/workspace/GEN3C:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/workspace/checkpoints/huggingface}"
export HF_HOME="${HF_HOME:-/workspace/checkpoints/huggingface}"

# CRITICAL: Set checkpoint dir to parent (GEN3C appends model name internally)
export GEN3C_CHECKPOINT_DIR="${GEN3C_CHECKPOINT_DIR:-/workspace/checkpoints}"

echo "  ✓ PYTHONPATH set"
echo "  ✓ GEN3C_CHECKPOINT_DIR: $GEN3C_CHECKPOINT_DIR"

# Create directories
mkdir -p /workspace/outputs
mkdir -p /tmp/gen3c

# =============================================================================
# Fix huggingface-hub Version (Required for transformers compatibility)
# =============================================================================
echo ""
echo "Checking huggingface-hub version..."
HF_VERSION=$(pip show huggingface-hub 2>/dev/null | grep Version | cut -d' ' -f2)
if [[ "$HF_VERSION" == 1.* ]]; then
    echo "  ⚠ huggingface-hub $HF_VERSION detected, downgrading for compatibility..."
    pip install -q "huggingface-hub>=0.26.0,<1.0"
    echo "  ✓ huggingface-hub downgraded"
else
    echo "  ✓ huggingface-hub version OK: $HF_VERSION"
fi

# =============================================================================
# Checkpoint Symlink Setup
# =============================================================================
echo ""
echo "Setting up checkpoint symlinks..."

# Handle nested checkpoint directory (from network volume download)
if [ ! -d "/workspace/checkpoints/Gen3C-Cosmos-7B" ] && [ -d "/workspace/checkpoints/checkpoints/Gen3C-Cosmos-7B" ]; then
    echo "  Creating symlink for nested Gen3C-Cosmos-7B..."
    ln -sf /workspace/checkpoints/checkpoints/Gen3C-Cosmos-7B /workspace/checkpoints/Gen3C-Cosmos-7B
    echo "  ✓ Gen3C-Cosmos-7B symlink created"
fi

# Create symlinks for tokenizer and T5 at parent level (GEN3C expects these paths)
if [ -d "/workspace/checkpoints/Gen3C-Cosmos-7B/Cosmos-Tokenize1-CV8x8x8-720p" ]; then
    if [ ! -e "/workspace/checkpoints/Cosmos-Tokenize1-CV8x8x8-720p" ]; then
        ln -sf /workspace/checkpoints/Gen3C-Cosmos-7B/Cosmos-Tokenize1-CV8x8x8-720p /workspace/checkpoints/Cosmos-Tokenize1-CV8x8x8-720p
        echo "  ✓ Cosmos-Tokenize1 symlink created"
    else
        echo "  ✓ Cosmos-Tokenize1 symlink exists"
    fi
fi

if [ -d "/workspace/checkpoints/Gen3C-Cosmos-7B/google-t5" ]; then
    if [ ! -e "/workspace/checkpoints/google-t5" ]; then
        ln -sf /workspace/checkpoints/Gen3C-Cosmos-7B/google-t5 /workspace/checkpoints/google-t5
        echo "  ✓ google-t5 symlink created"
    else
        echo "  ✓ google-t5 symlink exists"
    fi
fi

# =============================================================================
# Verify Checkpoints
# =============================================================================
echo ""
echo "Verifying checkpoints..."

CHECKPOINT_OK=true

if [ -f "/workspace/checkpoints/Gen3C-Cosmos-7B/model.pt" ]; then
    echo "  ✓ Main model: model.pt found"
else
    echo "  ✗ Main model: model.pt NOT FOUND"
    CHECKPOINT_OK=false
fi

if [ -f "/workspace/checkpoints/Cosmos-Tokenize1-CV8x8x8-720p/mean_std.pt" ]; then
    echo "  ✓ Tokenizer: mean_std.pt found"
else
    echo "  ✗ Tokenizer: mean_std.pt NOT FOUND"
    CHECKPOINT_OK=false
fi

if [ -d "/workspace/checkpoints/google-t5/t5-11b" ]; then
    echo "  ✓ T5 model: t5-11b directory found"
else
    echo "  ✗ T5 model: t5-11b directory NOT FOUND"
    CHECKPOINT_OK=false
fi

if [ "$CHECKPOINT_OK" = false ]; then
    echo ""
    echo "⚠ WARNING: Some checkpoints are missing!"
    echo "  Please ensure the network volume is mounted at /workspace/checkpoints"
    echo "  The server will start but inference will fail."
fi

# =============================================================================
# Start Server
# =============================================================================
echo ""
echo "=============================================="
echo "Starting GEN3C API Server on port 8000..."
echo "=============================================="

exec python /workspace/server.py
