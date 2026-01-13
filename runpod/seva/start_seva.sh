#!/bin/bash
# Startup script for SEVA (Stable Virtual Camera) serverless handler

set -e

echo "=========================================="
echo "SEVA Serverless Handler Startup"
echo "=========================================="

# Display environment info
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())')"
if python -c 'import torch; torch.cuda.is_available()' 2>/dev/null | grep -q True; then
    echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
fi

# Check HuggingFace authentication
if [ -n "$HF_TOKEN" ]; then
    echo "HF_TOKEN: Set (length=${#HF_TOKEN})"
    # Login to HuggingFace
    huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
    echo "HuggingFace login complete"
else
    echo "WARNING: HF_TOKEN not set - model download may fail!"
fi

# Ensure output directories exist
mkdir -p /runpod-volume/outputs/seva
mkdir -p /workspace/checkpoints/huggingface

# Pre-download model if not present (optional - can be slow)
# This is commented out because model access requires approval
# python -c "from seva import SEVAModel; SEVAModel.from_pretrained('stabilityai/stable-virtual-camera')" || true

echo "=========================================="
echo "Starting SEVA handler..."
echo "=========================================="

# Start the handler
cd /workspace
python -u handler.py
