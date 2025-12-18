#!/bin/bash
# =============================================================================
# GEN3C Container Startup Script for RunPod
# =============================================================================
# This script runs when the container starts on RunPod.
# It sets up the environment, creates necessary symlinks, and starts either:
#   - API Server (for GPU Pods) - server.py
#   - Serverless Handler (for Serverless Endpoints) - handler.py
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
# GPU Detection with CUDA Initialization Retry
# =============================================================================
echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "  ⚠ No GPU detected via nvidia-smi"

# CUDA initialization can fail on cold starts - retry up to 5 times
echo ""
echo "Initializing CUDA..."
MAX_CUDA_RETRIES=5
CUDA_RETRY_DELAY=5

for i in $(seq 1 $MAX_CUDA_RETRIES); do
    echo "  Attempt $i/$MAX_CUDA_RETRIES..."
    
    # Try to initialize CUDA in a fresh Python process
    CUDA_STATUS=$(python -c "
import os
import sys
# Clear any stale CUDA state
os.environ.pop('CUDA_VISIBLE_DEVICES', None)
import torch
try:
    if torch.cuda.is_available():
        # Force CUDA initialization
        torch.cuda.init()
        device_name = torch.cuda.get_device_name(0)
        print(f'SUCCESS:{device_name}')
    else:
        print('FAILED:CUDA not available')
except Exception as e:
    print(f'FAILED:{e}')
" 2>/dev/null)
    
    if [[ "$CUDA_STATUS" == SUCCESS:* ]]; then
        GPU_NAME="${CUDA_STATUS#SUCCESS:}"
        echo "  ✓ CUDA initialized successfully!"
        echo "  ✓ GPU: $GPU_NAME"
        break
    else
        ERROR_MSG="${CUDA_STATUS#FAILED:}"
        echo "  ✗ CUDA init failed: $ERROR_MSG"
        if [ $i -lt $MAX_CUDA_RETRIES ]; then
            echo "  Waiting ${CUDA_RETRY_DELAY}s before retry..."
            sleep $CUDA_RETRY_DELAY
        fi
    fi
done

# Final check - if CUDA still not working, warn but continue (job will fail gracefully)
FINAL_CUDA_CHECK=$(python -c "import torch; print('OK' if torch.cuda.is_available() else 'FAIL')" 2>/dev/null)
if [ "$FINAL_CUDA_CHECK" != "OK" ]; then
    echo ""
    echo "⚠ WARNING: CUDA initialization failed after $MAX_CUDA_RETRIES attempts!"
    echo "  The job will likely fail. Consider terminating and retrying."
    echo ""
fi

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
# Network Volume Detection (Serverless vs Pod)
# =============================================================================
echo ""
echo "Detecting network volume mount..."

# RunPod Serverless mounts network volumes at /runpod-volume
# Pods mount at /workspace/checkpoints directly
# We need checkpoints at /workspace/checkpoints/Gen3C-Cosmos-7B

VOLUME_LINKED=false
MODEL_PATH=""

# Ensure /workspace/checkpoints exists
mkdir -p /workspace/checkpoints

# Check for serverless volume mount at /runpod-volume
if [ -d "/runpod-volume" ]; then
    echo "  Found /runpod-volume (serverless mount)"
    echo "  Contents:"
    ls -la /runpod-volume/ 2>/dev/null | head -10
    
    # Priority 1: Check nested /runpod-volume/checkpoints/Gen3C-Cosmos-7B (most common)
    if [ -d "/runpod-volume/checkpoints/Gen3C-Cosmos-7B" ] && [ -f "/runpod-volume/checkpoints/Gen3C-Cosmos-7B/model.pt" ]; then
        MODEL_PATH="/runpod-volume/checkpoints/Gen3C-Cosmos-7B"
        echo "  ✓ Found model at $MODEL_PATH"
    # Priority 2: Direct /runpod-volume/Gen3C-Cosmos-7B (if it's a real directory, not symlink)
    elif [ -d "/runpod-volume/Gen3C-Cosmos-7B" ] && [ ! -L "/runpod-volume/Gen3C-Cosmos-7B" ] && [ -f "/runpod-volume/Gen3C-Cosmos-7B/model.pt" ]; then
        MODEL_PATH="/runpod-volume/Gen3C-Cosmos-7B"
        echo "  ✓ Found model at $MODEL_PATH"
    # Priority 3: Search for model.pt
    else
        echo "  Searching for model.pt in /runpod-volume..."
        FOUND_MODEL=$(find /runpod-volume -maxdepth 4 -name "model.pt" -type f 2>/dev/null | head -1)
        if [ -n "$FOUND_MODEL" ]; then
            MODEL_PATH=$(dirname "$FOUND_MODEL")
            echo "  ✓ Found model at $MODEL_PATH"
        fi
    fi
    
    # Create symlink if we found the model
    if [ -n "$MODEL_PATH" ]; then
        # Remove any existing broken symlink
        rm -f /workspace/checkpoints/Gen3C-Cosmos-7B 2>/dev/null
        ln -sf "$MODEL_PATH" /workspace/checkpoints/Gen3C-Cosmos-7B
        echo "  ✓ Linked $MODEL_PATH -> /workspace/checkpoints/Gen3C-Cosmos-7B"
        VOLUME_LINKED=true
    else
        echo "  ⚠ Could not find Gen3C-Cosmos-7B model in /runpod-volume"
    fi
fi

# Also check /workspace/volume (another possible mount point)
if [ "$VOLUME_LINKED" = false ] && [ -d "/workspace/volume" ]; then
    echo "  Found /workspace/volume"
    if [ -d "/workspace/volume/Gen3C-Cosmos-7B" ] && [ -f "/workspace/volume/Gen3C-Cosmos-7B/model.pt" ]; then
        rm -f /workspace/checkpoints/Gen3C-Cosmos-7B 2>/dev/null
        ln -sf /workspace/volume/Gen3C-Cosmos-7B /workspace/checkpoints/Gen3C-Cosmos-7B
        echo "  ✓ Linked /workspace/volume/Gen3C-Cosmos-7B -> /workspace/checkpoints/Gen3C-Cosmos-7B"
        VOLUME_LINKED=true
    elif [ -d "/workspace/volume/checkpoints/Gen3C-Cosmos-7B" ] && [ -f "/workspace/volume/checkpoints/Gen3C-Cosmos-7B/model.pt" ]; then
        rm -f /workspace/checkpoints/Gen3C-Cosmos-7B 2>/dev/null
        ln -sf /workspace/volume/checkpoints/Gen3C-Cosmos-7B /workspace/checkpoints/Gen3C-Cosmos-7B
        echo "  ✓ Linked /workspace/volume/checkpoints/Gen3C-Cosmos-7B -> /workspace/checkpoints/Gen3C-Cosmos-7B"
        VOLUME_LINKED=true
    fi
fi

# Check if /workspace/checkpoints/Gen3C-Cosmos-7B already exists (Pod mode)
if [ "$VOLUME_LINKED" = false ] && [ -d "/workspace/checkpoints/Gen3C-Cosmos-7B" ] && [ -f "/workspace/checkpoints/Gen3C-Cosmos-7B/model.pt" ]; then
    echo "  ✓ Model already at /workspace/checkpoints/Gen3C-Cosmos-7B (Pod mode)"
    VOLUME_LINKED=true
fi

if [ "$VOLUME_LINKED" = false ]; then
    echo "  ⚠ WARNING: Could not locate Gen3C-Cosmos-7B model!"
    echo "    Checked: /runpod-volume, /workspace/volume, /workspace/checkpoints"
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
# Determine Run Mode and Start
# =============================================================================
echo ""
echo "=============================================="

# Detect serverless mode:
# - RUNPOD_ENDPOINT_ID is set for serverless workers
# - RUNPOD_SERVERLESS=true (sometimes set)
# - /runpod-volume exists (serverless mount point)
# For Pods, typically only RUNPOD_POD_ID is set without RUNPOD_ENDPOINT_ID

echo "Mode detection:"
echo "  RUNPOD_ENDPOINT_ID: ${RUNPOD_ENDPOINT_ID:-not set}"
echo "  RUNPOD_SERVERLESS: ${RUNPOD_SERVERLESS:-not set}"
echo "  RUNPOD_POD_ID: ${RUNPOD_POD_ID:-not set}"

if [ -n "$RUNPOD_ENDPOINT_ID" ]; then
    echo ""
    echo "Starting in SERVERLESS mode (handler.py)..."
    echo "  Endpoint: $RUNPOD_ENDPOINT_ID"
    echo "=============================================="
    exec python /workspace/handler.py
else
    echo ""
    echo "Starting in POD mode (API Server on port 8000)..."
    echo "=============================================="
    exec python /workspace/server.py
fi
