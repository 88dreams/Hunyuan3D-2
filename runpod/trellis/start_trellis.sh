#!/bin/bash
# TRELLIS.2 RunPod Startup Script
# Initializes environment and starts the serverless handler

set -e

echo "=============================================="
echo "TRELLIS.2 RunPod Serverless Worker"
echo "=============================================="

# Activate conda environment
source /root/miniforge3/bin/activate trellis2

# Check for network volume
echo "Detecting network volume mount..."
if [ -d "/runpod-volume" ]; then
    echo "Found /runpod-volume (serverless mount)"
    ls -la /runpod-volume/
    
    # Check for TRELLIS.2 checkpoints
    TRELLIS_CKPT_DIR="/runpod-volume/checkpoints/trellis"
    if [ -d "$TRELLIS_CKPT_DIR" ]; then
        echo "✓ TRELLIS.2 checkpoints found at $TRELLIS_CKPT_DIR"
        ls -la "$TRELLIS_CKPT_DIR/"
        export TRELLIS_CHECKPOINT_DIR="$TRELLIS_CKPT_DIR"
    else
        echo "⚠ TRELLIS.2 checkpoints not found at $TRELLIS_CKPT_DIR"
        echo "  Model will download from HuggingFace on first run"
    fi
    
    # Set HuggingFace cache to network volume
    export HF_HOME="/runpod-volume/huggingface"
    mkdir -p "$HF_HOME"
    
    # Create output directory
    mkdir -p /runpod-volume/outputs/trellis
    
elif [ -d "/workspace" ]; then
    echo "Found /workspace (pod mount)"
    ls -la /workspace/
    
    # Check for TRELLIS.2 checkpoints
    TRELLIS_CKPT_DIR="/workspace/checkpoints/trellis"
    if [ -d "$TRELLIS_CKPT_DIR" ]; then
        echo "✓ TRELLIS.2 checkpoints found at $TRELLIS_CKPT_DIR"
        export TRELLIS_CHECKPOINT_DIR="$TRELLIS_CKPT_DIR"
    fi
    
    export HF_HOME="/workspace/huggingface"
    mkdir -p "$HF_HOME"
else
    echo "No network volume detected, using container storage"
fi

# Set Python path
export PYTHONPATH="/workspace/TRELLIS2:${PYTHONPATH}"

# Verify TRELLIS.2 installation
echo ""
echo "Verifying TRELLIS.2 installation..."
if [ -d "/workspace/TRELLIS2" ]; then
    echo "✓ TRELLIS.2 repository found"
    ls -la /workspace/TRELLIS2/
else
    echo "✗ TRELLIS.2 repository NOT found"
fi

# Check Python environment
echo ""
echo "Python environment:"
which python
python --version
pip list | grep -E "torch|trellis|nvdiffrast|flash-attn" || echo "Some packages not found"

# Check GPU
echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv 2>/dev/null || echo "No GPU detected"

# Mode detection
echo ""
echo "=============================================="
echo "Mode detection:"
if [ -n "$RUNPOD_ENDPOINT_ID" ]; then
    echo "RUNPOD_ENDPOINT_ID: $RUNPOD_ENDPOINT_ID"
    echo "Starting in SERVERLESS mode..."
    echo "=============================================="
    python /workspace/handler.py
else
    echo "No endpoint ID detected"
    echo "Starting in POD mode (API server)..."
    echo "=============================================="
    python /workspace/handler.py
fi

