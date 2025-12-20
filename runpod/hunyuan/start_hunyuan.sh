#!/bin/bash
# Hunyuan3D RunPod Startup Script

set -e

echo "=============================================="
echo "Hunyuan3D RunPod Serverless Worker"
echo "=============================================="

# Check for network volume
echo "Detecting network volume mount..."
if [ -d "/runpod-volume" ]; then
    echo "Found /runpod-volume (serverless mount)"
    ls -la /runpod-volume/
    
    # Check for Hunyuan checkpoints
    HUNYUAN_CKPT_DIR="/runpod-volume/checkpoints/hunyuan"
    if [ -d "$HUNYUAN_CKPT_DIR" ]; then
        echo "✓ Hunyuan3D checkpoints found at $HUNYUAN_CKPT_DIR"
        ls -la "$HUNYUAN_CKPT_DIR/"
        export HUNYUAN_CHECKPOINT_DIR="$HUNYUAN_CKPT_DIR"
    else
        echo "⚠ Hunyuan3D checkpoints not found at $HUNYUAN_CKPT_DIR"
        echo "  Model will download from HuggingFace on first run"
    fi
    
    # Set HuggingFace cache to network volume
    export HF_HOME="/runpod-volume/huggingface"
    mkdir -p "$HF_HOME"
    
    # Create output directory
    mkdir -p /runpod-volume/outputs/hunyuan
    
elif [ -d "/workspace" ]; then
    echo "Found /workspace (pod mount)"
    ls -la /workspace/
    
    HUNYUAN_CKPT_DIR="/workspace/checkpoints/hunyuan"
    if [ -d "$HUNYUAN_CKPT_DIR" ]; then
        echo "✓ Hunyuan3D checkpoints found at $HUNYUAN_CKPT_DIR"
        export HUNYUAN_CHECKPOINT_DIR="$HUNYUAN_CKPT_DIR"
    fi
    
    export HF_HOME="/workspace/huggingface"
    mkdir -p "$HF_HOME"
else
    echo "No network volume detected, using container storage"
fi

# Set Python path
export PYTHONPATH="/workspace/Hunyuan3D-2:${PYTHONPATH}"

# Verify Hunyuan3D installation
echo ""
echo "Verifying Hunyuan3D installation..."
python -c "from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline; print('✓ Hunyuan3D shapegen available')" 2>/dev/null || echo "✗ Hunyuan3D shapegen NOT available"
python -c "from hy3dgen.rembg import BackgroundRemover; print('✓ Background remover available')" 2>/dev/null || echo "✗ Background remover NOT available"

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
    echo "Starting in POD mode..."
    echo "=============================================="
    python /workspace/handler.py
fi

