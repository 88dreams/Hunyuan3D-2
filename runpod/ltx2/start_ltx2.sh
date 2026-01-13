#!/bin/bash
# LTX-2 Startup Script for RunPod Serverless
#
# This script:
# 1. Sets up HuggingFace authentication (if token provided)
# 2. Creates necessary directories
# 3. Optionally pre-downloads model weights
# 4. Starts the serverless handler

set -e

echo "=========================================="
echo "LTX-2 RunPod Serverless Startup"
echo "=========================================="

# HuggingFace authentication (optional, but recommended for faster downloads)
if [ -n "$HF_TOKEN" ]; then
    echo "Setting up HuggingFace authentication..."
    python -c "from huggingface_hub import login; login(token='$HF_TOKEN')"
    echo "✓ HuggingFace authentication configured"
else
    echo "⚠ No HF_TOKEN provided - using anonymous access"
    echo "  For faster downloads, set HF_TOKEN environment variable"
fi

# Create output directories
echo "Creating directories..."
mkdir -p ${OUTPUT_DIR:-/runpod-volume/outputs/ltx2}
mkdir -p ${MODEL_CACHE_DIR:-/runpod-volume/models/ltx2}
mkdir -p ${HF_HOME:-/runpod-volume/huggingface}
echo "✓ Directories created"

# GPU check
echo "Checking GPU..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    props = torch.cuda.get_device_properties(0)
    print(f'VRAM: {props.total_memory / 1024**3:.1f} GB')
"

# Check diffusers version
echo "Checking diffusers..."
python -c "
import diffusers
print(f'Diffusers version: {diffusers.__version__}')
"

# Optionally pre-download model on first startup
if [ "$PRELOAD_MODEL" = "true" ]; then
    echo "Pre-downloading LTX-2 model..."
    python -c "
from diffusers import LTXPipeline
import torch
import os

model_id = 'Lightricks/LTX-Video'
variant = os.environ.get('LTX2_MODEL_VARIANT', 'ltx-2-19b-distilled')

print(f'Loading model: {model_id} ({variant})')
# Just load to cache, then free memory
pipe = LTXPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
)
print('✓ Model cached')
del pipe
torch.cuda.empty_cache()
"
fi

echo "=========================================="
echo "Starting LTX-2 handler..."
echo "=========================================="

# Start the handler
exec python -u /workspace/handler.py
