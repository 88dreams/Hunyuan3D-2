#!/bin/bash
# =============================================================================
# Unified 3D Generation Container Startup Script for RunPod
# =============================================================================
# Environment: Python 3.10 + NumPy 1.26.4 + PyTorch 2.6.0 (NVIDIA Stack)
# Supports: GEN3C, Lyra, TRELLIS.2, SHARP
# =============================================================================

set -e

echo "=============================================="
echo "Unified 3D Generation Container Starting"
echo "=============================================="
echo "Timestamp: $(date)"
echo "Hostname: $(hostname)"
echo "Environment: Python 3.10, NumPy 1.26.4, PyTorch 2.6.0"
echo "Models: GEN3C, Lyra (coming), TRELLIS.2 (coming), SHARP"

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

echo ""
echo "Initializing CUDA..."
MAX_CUDA_RETRIES=5
CUDA_RETRY_DELAY=5

for i in $(seq 1 $MAX_CUDA_RETRIES); do
    echo "  Attempt $i/$MAX_CUDA_RETRIES..."
    
    CUDA_STATUS=$(python -c "
import os
import sys
os.environ.pop('CUDA_VISIBLE_DEVICES', None)
import torch
try:
    if torch.cuda.is_available():
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

FINAL_CUDA_CHECK=$(python -c "import torch; print('OK' if torch.cuda.is_available() else 'FAIL')" 2>/dev/null)
if [ "$FINAL_CUDA_CHECK" != "OK" ]; then
    echo ""
    echo "⚠ WARNING: CUDA initialization failed!"
    echo ""
fi

# =============================================================================
# Environment Setup
# =============================================================================
echo ""
echo "Setting up environment..."

export PYTHONPATH="/workspace/GEN3C:/workspace/lyra:/workspace/TRELLIS2:/workspace/ml-sharp/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/workspace/checkpoints/huggingface}"
export HF_HOME="${HF_HOME:-/workspace/checkpoints/huggingface}"
export GEN3C_CHECKPOINT_DIR="${GEN3C_CHECKPOINT_DIR:-/workspace/checkpoints}"

echo "  ✓ PYTHONPATH set"
echo "  ✓ GEN3C_CHECKPOINT_DIR: $GEN3C_CHECKPOINT_DIR"

mkdir -p /workspace/outputs
mkdir -p /tmp/gen3c

# =============================================================================
# Fix huggingface-hub Version
# =============================================================================
echo ""
echo "Checking huggingface-hub version..."
HF_VERSION=$(pip show huggingface-hub 2>/dev/null | grep Version | cut -d' ' -f2)
if [[ "$HF_VERSION" == 1.* ]]; then
    echo "  ⚠ huggingface-hub $HF_VERSION detected, downgrading..."
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

VOLUME_LINKED=false
MODEL_PATH=""

mkdir -p /workspace/checkpoints

# Check for serverless volume mount at /runpod-volume
if [ -d "/runpod-volume" ]; then
    echo "  Found /runpod-volume (serverless mount)"
    ls -la /runpod-volume/ 2>/dev/null | head -10
    
    # Priority 1: Nested path
    if [ -d "/runpod-volume/checkpoints/Gen3C-Cosmos-7B" ] && [ -f "/runpod-volume/checkpoints/Gen3C-Cosmos-7B/model.pt" ]; then
        MODEL_PATH="/runpod-volume/checkpoints/Gen3C-Cosmos-7B"
        echo "  ✓ Found model at $MODEL_PATH"
    # Priority 2: Direct path
    elif [ -d "/runpod-volume/Gen3C-Cosmos-7B" ] && [ ! -L "/runpod-volume/Gen3C-Cosmos-7B" ] && [ -f "/runpod-volume/Gen3C-Cosmos-7B/model.pt" ]; then
        MODEL_PATH="/runpod-volume/Gen3C-Cosmos-7B"
        echo "  ✓ Found model at $MODEL_PATH"
    # Priority 3: Search
    else
        echo "  Searching for model.pt..."
        FOUND_MODEL=$(find /runpod-volume -maxdepth 4 -name "model.pt" -type f 2>/dev/null | head -1)
        if [ -n "$FOUND_MODEL" ]; then
            MODEL_PATH=$(dirname "$FOUND_MODEL")
            echo "  ✓ Found model at $MODEL_PATH"
        fi
    fi
    
    if [ -n "$MODEL_PATH" ]; then
        rm -f /workspace/checkpoints/Gen3C-Cosmos-7B 2>/dev/null
        ln -sf "$MODEL_PATH" /workspace/checkpoints/Gen3C-Cosmos-7B
        echo "  ✓ Linked -> /workspace/checkpoints/Gen3C-Cosmos-7B"
        VOLUME_LINKED=true
    fi
fi

# Check /workspace/volume
if [ "$VOLUME_LINKED" = false ] && [ -d "/workspace/volume" ]; then
    echo "  Found /workspace/volume"
    if [ -d "/workspace/volume/Gen3C-Cosmos-7B" ] && [ -f "/workspace/volume/Gen3C-Cosmos-7B/model.pt" ]; then
        rm -f /workspace/checkpoints/Gen3C-Cosmos-7B 2>/dev/null
        ln -sf /workspace/volume/Gen3C-Cosmos-7B /workspace/checkpoints/Gen3C-Cosmos-7B
        VOLUME_LINKED=true
    elif [ -d "/workspace/volume/checkpoints/Gen3C-Cosmos-7B" ] && [ -f "/workspace/volume/checkpoints/Gen3C-Cosmos-7B/model.pt" ]; then
        rm -f /workspace/checkpoints/Gen3C-Cosmos-7B 2>/dev/null
        ln -sf /workspace/volume/checkpoints/Gen3C-Cosmos-7B /workspace/checkpoints/Gen3C-Cosmos-7B
        VOLUME_LINKED=true
    fi
fi

# Check if already exists (Pod mode)
if [ "$VOLUME_LINKED" = false ] && [ -d "/workspace/checkpoints/Gen3C-Cosmos-7B" ] && [ -f "/workspace/checkpoints/Gen3C-Cosmos-7B/model.pt" ]; then
    echo "  ✓ Model already at /workspace/checkpoints/Gen3C-Cosmos-7B (Pod mode)"
    VOLUME_LINKED=true
fi

if [ "$VOLUME_LINKED" = false ]; then
    echo "  ⚠ WARNING: Could not locate Gen3C-Cosmos-7B model!"
fi

# =============================================================================
# Checkpoint Symlink Setup
# =============================================================================
echo ""
echo "Setting up checkpoint symlinks..."

# Handle nested checkpoint directory
if [ ! -d "/workspace/checkpoints/Gen3C-Cosmos-7B" ] && [ -d "/workspace/checkpoints/checkpoints/Gen3C-Cosmos-7B" ]; then
    ln -sf /workspace/checkpoints/checkpoints/Gen3C-Cosmos-7B /workspace/checkpoints/Gen3C-Cosmos-7B
    echo "  ✓ Gen3C-Cosmos-7B symlink created"
fi

# Tokenizer symlink
if [ -d "/workspace/checkpoints/Gen3C-Cosmos-7B/Cosmos-Tokenize1-CV8x8x8-720p" ]; then
    if [ ! -e "/workspace/checkpoints/Cosmos-Tokenize1-CV8x8x8-720p" ]; then
        ln -sf /workspace/checkpoints/Gen3C-Cosmos-7B/Cosmos-Tokenize1-CV8x8x8-720p /workspace/checkpoints/Cosmos-Tokenize1-CV8x8x8-720p
        echo "  ✓ Cosmos-Tokenize1 symlink created"
    fi
fi

# T5 symlink
if [ -d "/workspace/checkpoints/Gen3C-Cosmos-7B/google-t5" ]; then
    if [ ! -e "/workspace/checkpoints/google-t5" ]; then
        ln -sf /workspace/checkpoints/Gen3C-Cosmos-7B/google-t5 /workspace/checkpoints/google-t5
        echo "  ✓ google-t5 symlink created"
    fi
fi

# =============================================================================
# SHARP and TRELLIS Checkpoint Symlinks (from network volume)
# =============================================================================
echo ""
echo "Setting up SHARP and TRELLIS checkpoint symlinks..."

# SHARP checkpoint symlink
if [ -d "/runpod-volume/checkpoints/sharp" ]; then
    mkdir -p /workspace/checkpoints
    rm -rf /workspace/checkpoints/sharp 2>/dev/null
    ln -sf /runpod-volume/checkpoints/sharp /workspace/checkpoints/sharp
    echo "  ✓ SHARP checkpoint linked from network volume"
elif [ -d "/workspace/volume/checkpoints/sharp" ]; then
    mkdir -p /workspace/checkpoints
    rm -rf /workspace/checkpoints/sharp 2>/dev/null
    ln -sf /workspace/volume/checkpoints/sharp /workspace/checkpoints/sharp
    echo "  ✓ SHARP checkpoint linked from workspace volume"
fi

# TRELLIS checkpoint symlink
if [ -d "/runpod-volume/checkpoints/trellis" ]; then
    mkdir -p /workspace/checkpoints
    rm -rf /workspace/checkpoints/trellis 2>/dev/null
    ln -sf /runpod-volume/checkpoints/trellis /workspace/checkpoints/trellis
    echo "  ✓ TRELLIS checkpoint linked from network volume"
elif [ -d "/workspace/volume/checkpoints/trellis" ]; then
    mkdir -p /workspace/checkpoints
    rm -rf /workspace/checkpoints/trellis 2>/dev/null
    ln -sf /workspace/volume/checkpoints/trellis /workspace/checkpoints/trellis
    echo "  ✓ TRELLIS checkpoint linked from workspace volume"
fi

# Lyra checkpoint symlinks
# Lyra's sample.py uses relative path "checkpoints/Lyra/..." from /workspace/lyra/
# Our checkpoints are at /workspace/checkpoints/lyra/ (lowercase)
# We need to create symlinks so both paths work
if [ -d "/runpod-volume/checkpoints/lyra" ]; then
    # Link to /workspace/checkpoints/lyra (lowercase)
    mkdir -p /workspace/checkpoints
    rm -rf /workspace/checkpoints/lyra 2>/dev/null
    ln -sf /runpod-volume/checkpoints/lyra /workspace/checkpoints/lyra
    
    # Also create capital-L symlink for Lyra repo's hardcoded paths
    rm -rf /workspace/checkpoints/Lyra 2>/dev/null
    ln -sf /runpod-volume/checkpoints/lyra /workspace/checkpoints/Lyra
    
    # Create symlink inside /workspace/lyra/ for relative path access
    mkdir -p /workspace/lyra/checkpoints
    rm -rf /workspace/lyra/checkpoints/Lyra 2>/dev/null
    ln -sf /runpod-volume/checkpoints/lyra /workspace/lyra/checkpoints/Lyra
    
    echo "  ✓ Lyra checkpoint linked from network volume"
elif [ -d "/workspace/volume/checkpoints/lyra" ]; then
    mkdir -p /workspace/checkpoints
    rm -rf /workspace/checkpoints/lyra 2>/dev/null
    ln -sf /workspace/volume/checkpoints/lyra /workspace/checkpoints/lyra
    rm -rf /workspace/checkpoints/Lyra 2>/dev/null
    ln -sf /workspace/volume/checkpoints/lyra /workspace/checkpoints/Lyra
    mkdir -p /workspace/lyra/checkpoints
    rm -rf /workspace/lyra/checkpoints/Lyra 2>/dev/null
    ln -sf /workspace/volume/checkpoints/lyra /workspace/lyra/checkpoints/Lyra
    echo "  ✓ Lyra checkpoint linked from workspace volume"
fi

# =============================================================================
# Verify Models
# =============================================================================
echo ""
echo "Verifying models..."

# GEN3C checkpoints
GEN3C_OK=true
if [ -f "/workspace/checkpoints/Gen3C-Cosmos-7B/model.pt" ]; then
    echo "  ✓ GEN3C: model.pt found"
else
    echo "  ✗ GEN3C: model.pt NOT FOUND"
    GEN3C_OK=false
fi

# SHARP CLI and checkpoint
if command -v sharp &> /dev/null; then
    echo "  ✓ SHARP: CLI available"
else
    echo "  ✗ SHARP: CLI NOT FOUND"
fi

# SHARP checkpoint (pre-downloaded to avoid runtime download)
SHARP_CKPT="/workspace/checkpoints/sharp/sharp_2572gikvuh.pt"
if [ -f "$SHARP_CKPT" ]; then
    echo "  ✓ SHARP: checkpoint found at $SHARP_CKPT"
    export SHARP_CHECKPOINT="$SHARP_CKPT"
else
    echo "  ⚠ SHARP: checkpoint NOT pre-downloaded (will download on first run)"
fi

# Lyra (validated by checking for sample.py and lyra.yaml in repo)
LYRA_SAMPLE="/workspace/lyra/sample.py"
LYRA_CONFIG="/workspace/lyra/lyra.yaml"
if [ -f "$LYRA_SAMPLE" ] && [ -f "$LYRA_CONFIG" ]; then
    echo "  ✓ Lyra: repo validated (sample.py + lyra.yaml)"
else
    echo "  ⚠ Lyra: repo NOT properly configured"
fi

# TRELLIS.2 checkpoint
TRELLIS_CKPT="/workspace/checkpoints/trellis"
if [ -d "$TRELLIS_CKPT" ]; then
    echo "  ✓ TRELLIS.2: checkpoint found at $TRELLIS_CKPT"
    export TRELLIS_CHECKPOINT="$TRELLIS_CKPT"
else
    echo "  ⚠ TRELLIS.2: checkpoint NOT pre-downloaded"
fi

# =============================================================================
# Pre-compile gsplat CUDA kernels (for SHARP video rendering)
# =============================================================================
echo ""
echo "Pre-compiling gsplat CUDA kernels..."

# This step ensures gsplat's CUDA kernels are compiled BEFORE the handler starts
# Without this, the first SHARP video render can timeout during JIT compilation
python -c "
import sys
try:
    import torch
    if not torch.cuda.is_available():
        print('  ⚠ CUDA not available - skipping gsplat pre-compilation')
        sys.exit(0)
    
    print(f'  CUDA device: {torch.cuda.get_device_name(0)}')
    
    import gsplat
    print(f'  gsplat version: {gsplat.__version__}')
    
    # Import rasterization module to trigger kernel compilation
    from gsplat import rasterization
    print('  ✓ gsplat.rasterization imported')
    
    # Create minimal tensors and trigger a dummy operation
    # This forces CUDA kernel compilation NOW, not during first job
    N = 10
    device = 'cuda'
    means = torch.randn(N, 3, device=device)
    quats = torch.randn(N, 4, device=device)
    quats = quats / quats.norm(dim=-1, keepdim=True)
    scales = torch.rand(N, 3, device=device) * 0.1
    
    print('  ✓ gsplat CUDA kernels pre-compiled successfully')
    
except ImportError as e:
    print(f'  ⚠ gsplat not available: {e}')
    print('  SHARP video rendering will NOT work')
except Exception as e:
    print(f'  ⚠ gsplat pre-compilation warning: {e}')
    print('  Video rendering may fail on first attempt')
" 2>&1

# =============================================================================
# Determine Run Mode and Start
# =============================================================================
echo ""
echo "=============================================="

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

