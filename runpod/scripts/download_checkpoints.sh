#!/bin/bash
# =============================================================================
# Download GEN3C Checkpoints to RunPod Network Volume
# =============================================================================
# Run this script on a temporary RunPod pod with the network volume mounted
# at /workspace/checkpoints
#
# Usage:
#   1. Create a temporary pod with network volume mounted
#   2. SSH into the pod
#   3. Run: bash download_checkpoints.sh
#   4. Wait for downloads to complete (~30-60 minutes)
#   5. Stop the pod (checkpoints persist on network volume)
# =============================================================================

set -e

echo "=============================================="
echo "GEN3C Checkpoint Downloader"
echo "=============================================="

# Configuration
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/workspace/checkpoints}"
HF_TOKEN="${HF_TOKEN:-}"  # Set this if you have a HuggingFace token

# Create directories
mkdir -p "$CHECKPOINT_DIR"
cd "$CHECKPOINT_DIR"

echo "Download directory: $CHECKPOINT_DIR"
echo "Available disk space:"
df -h "$CHECKPOINT_DIR"
echo ""

# =============================================================================
# Install huggingface-cli if not present
# =============================================================================
if ! command -v huggingface-cli &> /dev/null; then
    echo "Installing huggingface-cli..."
    pip install -q huggingface_hub[cli]
fi

# Login if token provided
if [ -n "$HF_TOKEN" ]; then
    echo "Logging in to HuggingFace..."
    huggingface-cli login --token "$HF_TOKEN"
fi

# =============================================================================
# Download Gen3C-Cosmos-7B
# =============================================================================
echo ""
echo "=============================================="
echo "Downloading Gen3C-Cosmos-7B (~28GB)"
echo "=============================================="

huggingface-cli download nvidia/Gen3C-Cosmos-7B \
    --local-dir "$CHECKPOINT_DIR/Gen3C-Cosmos-7B" \
    --local-dir-use-symlinks False

# =============================================================================
# Download Cosmos Tokenizer
# =============================================================================
echo ""
echo "=============================================="
echo "Downloading Cosmos-Tokenize1-CV8x8x8-720p (~2GB)"
echo "=============================================="

huggingface-cli download nvidia/Cosmos-Tokenize1-CV8x8x8-720p \
    --local-dir "$CHECKPOINT_DIR/Gen3C-Cosmos-7B/Cosmos-Tokenize1-CV8x8x8-720p" \
    --local-dir-use-symlinks False

# =============================================================================
# Download T5-11B Text Encoder
# =============================================================================
echo ""
echo "=============================================="
echo "Downloading google-t5/t5-11b (~42GB)"
echo "=============================================="

mkdir -p "$CHECKPOINT_DIR/Gen3C-Cosmos-7B/google-t5"
huggingface-cli download google-t5/t5-11b \
    --local-dir "$CHECKPOINT_DIR/Gen3C-Cosmos-7B/google-t5/t5-11b" \
    --local-dir-use-symlinks False

# =============================================================================
# Download Aegis Guardrail (optional but recommended)
# =============================================================================
echo ""
echo "=============================================="
echo "Downloading nvidia/Aegis-AI-Content-Safety-LlamaGuard-Defensive-1.0 (~2GB)"
echo "=============================================="

huggingface-cli download nvidia/Aegis-AI-Content-Safety-LlamaGuard-Defensive-1.0 \
    --local-dir "$CHECKPOINT_DIR/nvidia/Aegis-AI-Content-Safety-LlamaGuard-Defensive-1.0" \
    --local-dir-use-symlinks False || echo "Guardrail download failed (may require access request)"

# =============================================================================
# Create HuggingFace cache directory
# =============================================================================
mkdir -p "$CHECKPOINT_DIR/huggingface"

# =============================================================================
# SUMMARY
# =============================================================================
echo ""
echo "=============================================="
echo "Download Complete!"
echo "=============================================="
echo ""
echo "Checkpoint directory structure:"
find "$CHECKPOINT_DIR" -maxdepth 3 -type d | head -30
echo ""
echo "Total size:"
du -sh "$CHECKPOINT_DIR"
echo ""
echo "Disk space remaining:"
df -h "$CHECKPOINT_DIR"
echo ""
echo "You can now stop this pod. The checkpoints will persist on the network volume."

