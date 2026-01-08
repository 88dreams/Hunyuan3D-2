#!/bin/bash
# =============================================================================
# SV3D + 2DGS Setup Script for RunPod (Network Volume Installation)
# =============================================================================
#
# This script installs SV3D and 2DGS on your RunPod network volume WITHOUT
# interfering with existing installations (Gen3C, Lyra, SHARP, etc.).
#
# Key features:
#   - Installs to /runpod-volume/sv3d_2dgs/ (separate directory)
#   - Creates new conda environment "sv3d-2dgs" (not cosmos-predict1)
#   - Does NOT modify existing environments or installations
#   - Persists across pod restarts (on network volume)
#
# Usage:
#   1. Start a RunPod pod with your network volume mounted
#   2. SSH into the pod or use web terminal
#   3. Run: bash setup_sv3d_2dgs.sh
#
# After setup, activate with:
#   conda activate sv3d-2dgs
#
# Estimated time: 20-30 minutes (first run), ~2 minutes (subsequent runs)
#
# =============================================================================

set -e  # Exit on error

echo "=============================================="
echo "  SV3D + 2DGS Environment Setup"
echo "  (Network Volume Installation)"
echo "=============================================="
echo ""

# -----------------------------------------------------------------------------
# Configuration - EDIT THESE IF NEEDED
# -----------------------------------------------------------------------------

# Network volume mount point (standard RunPod location)
VOLUME_PATH="/runpod-volume"

# Installation directory (separate from existing installs)
SV3D_2DGS_ROOT="${VOLUME_PATH}/sv3d_2dgs"

# Conda environment name (separate from cosmos-predict1)
CONDA_ENV_NAME="sv3d-2dgs"

# 2DGS repository location
TWO_DGS_DIR="${SV3D_2DGS_ROOT}/2d-gaussian-splatting"

# Working directories
INPUTS_DIR="${SV3D_2DGS_ROOT}/inputs"
OUTPUTS_DIR="${SV3D_2DGS_ROOT}/outputs"
MODELS_DIR="${SV3D_2DGS_ROOT}/models"
SCRIPTS_DIR="${SV3D_2DGS_ROOT}/scripts"

# -----------------------------------------------------------------------------
# Check Prerequisites
# -----------------------------------------------------------------------------

echo "Checking prerequisites..."

# Check if network volume is mounted
if [ ! -d "${VOLUME_PATH}" ]; then
    echo "❌ ERROR: Network volume not found at ${VOLUME_PATH}"
    echo "   Make sure your network volume is mounted."
    echo "   On RunPod, this should be automatic if you have a volume attached."
    exit 1
fi

echo "✓ Network volume found at ${VOLUME_PATH}"

# Check available space
AVAILABLE_SPACE=$(df -BG "${VOLUME_PATH}" | tail -1 | awk '{print $4}' | sed 's/G//')
echo "  Available space: ${AVAILABLE_SPACE}GB"

if [ "${AVAILABLE_SPACE}" -lt 30 ]; then
    echo "⚠ WARNING: Less than 30GB available. You may run out of space."
    echo "   SV3D + 2DGS requires ~25GB for models and dependencies."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ ERROR: conda not found. Please use a RunPod template with conda."
    exit 1
fi

echo "✓ Conda found"

# Show existing installations (for awareness)
echo ""
echo "Existing installations on volume:"
ls -la ${VOLUME_PATH}/ 2>/dev/null | grep -E "^d" | awk '{print "  - " $NF}' || echo "  (none)"
echo ""

# -----------------------------------------------------------------------------
# Create Directory Structure
# -----------------------------------------------------------------------------

echo "Creating directory structure..."

mkdir -p "${SV3D_2DGS_ROOT}"
mkdir -p "${INPUTS_DIR}"
mkdir -p "${OUTPUTS_DIR}"
mkdir -p "${MODELS_DIR}"
mkdir -p "${SCRIPTS_DIR}"

echo "✓ Directories created:"
echo "    Root:    ${SV3D_2DGS_ROOT}"
echo "    Inputs:  ${INPUTS_DIR}"
echo "    Outputs: ${OUTPUTS_DIR}"
echo "    Models:  ${MODELS_DIR}"
echo "    Scripts: ${SCRIPTS_DIR}"

# -----------------------------------------------------------------------------
# Check if Already Installed
# -----------------------------------------------------------------------------

INSTALL_MARKER="${SV3D_2DGS_ROOT}/.installed"

if [ -f "${INSTALL_MARKER}" ]; then
    echo ""
    echo "=============================================="
    echo "  Installation already exists!"
    echo "=============================================="
    echo ""
    echo "SV3D + 2DGS is already installed on this volume."
    echo ""
    echo "To use it, run:"
    echo "  source ${SV3D_2DGS_ROOT}/activate.sh"
    echo ""
    echo "To re-install from scratch, delete the marker file:"
    echo "  rm ${INSTALL_MARKER}"
    echo "  bash setup_sv3d_2dgs.sh"
    echo ""
    
    # Still make sure the conda env is activated
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate ${CONDA_ENV_NAME} 2>/dev/null || true
    
    exit 0
fi

# -----------------------------------------------------------------------------
# Step 1: Create Conda Environment
# -----------------------------------------------------------------------------

echo ""
echo "[1/7] Creating conda environment: ${CONDA_ENV_NAME}"
echo "----------------------------------------------"

# Initialize conda for this shell
source $(conda info --base)/etc/profile.d/conda.sh

# Check if environment already exists
if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
    echo "  Environment exists, activating..."
    conda activate ${CONDA_ENV_NAME}
else
    echo "  Creating new environment..."
    conda create -n ${CONDA_ENV_NAME} python=3.10 -y -q
    conda activate ${CONDA_ENV_NAME}
fi

echo "✓ Conda environment ready: ${CONDA_ENV_NAME}"
echo "  Python: $(python --version)"

# -----------------------------------------------------------------------------
# Step 2: System Dependencies
# -----------------------------------------------------------------------------

echo ""
echo "[2/7] Installing system dependencies..."
echo "----------------------------------------------"

apt-get update -qq
apt-get install -y -qq \
    git \
    wget \
    curl \
    unzip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    > /dev/null 2>&1

# COLMAP (optional, for premium multi-photo path)
apt-get install -y -qq colmap > /dev/null 2>&1 || echo "  (COLMAP not available, skipping)"

echo "✓ System dependencies installed"

# -----------------------------------------------------------------------------
# Step 3: Python Dependencies (Core)
# -----------------------------------------------------------------------------

echo ""
echo "[3/7] Installing core Python dependencies..."
echo "----------------------------------------------"

pip install --upgrade pip -q

# Core ML libraries
pip install -q \
    torch==2.1.0 \
    torchvision==0.16.0 \
    --index-url https://download.pytorch.org/whl/cu118

pip install -q \
    numpy>=1.24.0 \
    scipy>=1.11.0 \
    pillow>=10.0.0 \
    opencv-python>=4.8.0 \
    imageio>=2.31.0 \
    imageio-ffmpeg>=0.4.9 \
    tqdm>=4.66.0

echo "✓ Core Python dependencies installed"

# -----------------------------------------------------------------------------
# Step 4: SV3D Dependencies
# -----------------------------------------------------------------------------

echo ""
echo "[4/7] Installing SV3D dependencies..."
echo "----------------------------------------------"

pip install -q \
    diffusers>=0.27.0 \
    transformers>=4.36.0 \
    accelerate>=0.25.0 \
    safetensors>=0.4.0

# Background removal
pip install -q rembg[gpu]

echo "✓ SV3D dependencies installed"

# -----------------------------------------------------------------------------
# Step 5: Clone and Install 2DGS
# -----------------------------------------------------------------------------

echo ""
echo "[5/7] Cloning and installing 2D Gaussian Splatting..."
echo "----------------------------------------------"

if [ -d "${TWO_DGS_DIR}" ]; then
    echo "  2DGS directory exists, pulling latest..."
    cd ${TWO_DGS_DIR}
    git pull
else
    echo "  Cloning 2DGS repository..."
    git clone https://github.com/hbb1/2d-gaussian-splatting.git ${TWO_DGS_DIR}
    cd ${TWO_DGS_DIR}
fi

# Install 2DGS Python dependencies
pip install -q -r requirements.txt

# Build CUDA extensions
echo "  Building 2DGS CUDA extensions (this may take a few minutes)..."
pip install -q submodules/diff-gaussian-rasterization-2d
pip install -q submodules/simple-knn

echo "✓ 2DGS installed"

# -----------------------------------------------------------------------------
# Step 6: Mesh Processing Dependencies
# -----------------------------------------------------------------------------

echo ""
echo "[6/7] Installing depth estimation and mesh processing..."
echo "----------------------------------------------"

# Depth estimation
pip install -q \
    einops>=0.7.0 \
    timm>=0.9.0

# Mesh processing
pip install -q \
    trimesh>=4.0.0 \
    open3d>=0.17.0 \
    pymeshlab>=2023.12

echo "✓ Depth and mesh dependencies installed"

# -----------------------------------------------------------------------------
# Step 7: Download Models
# -----------------------------------------------------------------------------

echo ""
echo "[7/7] Pre-downloading models to network volume..."
echo "----------------------------------------------"

# Set Hugging Face cache to network volume
export HF_HOME="${MODELS_DIR}/huggingface"
export TRANSFORMERS_CACHE="${MODELS_DIR}/huggingface"
mkdir -p "${HF_HOME}"

# Download SV3D model
echo "  Downloading SV3D model (this may take several minutes)..."
python -c "
import os
os.environ['HF_HOME'] = '${HF_HOME}'
os.environ['TRANSFORMERS_CACHE'] = '${HF_HOME}'
from diffusers import StableVideo3DPipeline
import torch
print('    Downloading SV3D...')
pipe = StableVideo3DPipeline.from_pretrained(
    'stabilityai/sv3d',
    torch_dtype=torch.float16,
    variant='fp16',
    cache_dir='${HF_HOME}',
)
print('    SV3D downloaded successfully!')
del pipe
"

# Download Depth Anything V2
echo "  Downloading Depth Anything V2..."
python -c "
import os
os.environ['HF_HOME'] = '${HF_HOME}'
os.environ['TRANSFORMERS_CACHE'] = '${HF_HOME}'
from transformers import pipeline
print('    Downloading Depth Anything V2...')
pipe = pipeline('depth-estimation', model='depth-anything/Depth-Anything-V2-Large-hf')
print('    Depth Anything V2 downloaded successfully!')
del pipe
"

# Download rembg model
echo "  Downloading rembg model..."
python -c "
from rembg import new_session
print('    Downloading rembg model...')
session = new_session('u2net')
print('    rembg model downloaded!')
"

echo "✓ Models downloaded to ${MODELS_DIR}"

# -----------------------------------------------------------------------------
# Create Helper Scripts
# -----------------------------------------------------------------------------

echo ""
echo "Creating helper scripts..."
echo "----------------------------------------------"

# Create activation script
cat > ${SV3D_2DGS_ROOT}/activate.sh << 'ACTIVATE_SCRIPT'
#!/bin/bash
# Activate the SV3D + 2DGS environment
# Usage: source /runpod-volume/sv3d_2dgs/activate.sh

# Set paths
export SV3D_2DGS_ROOT="/runpod-volume/sv3d_2dgs"
export TWO_DGS_PATH="${SV3D_2DGS_ROOT}/2d-gaussian-splatting"
export HF_HOME="${SV3D_2DGS_ROOT}/models/huggingface"
export TRANSFORMERS_CACHE="${SV3D_2DGS_ROOT}/models/huggingface"

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate sv3d-2dgs

echo "SV3D + 2DGS environment activated!"
echo "  Root: ${SV3D_2DGS_ROOT}"
echo "  2DGS: ${TWO_DGS_PATH}"
echo "  Models: ${HF_HOME}"
ACTIVATE_SCRIPT

chmod +x ${SV3D_2DGS_ROOT}/activate.sh

# Create quick test script
cat > ${SCRIPTS_DIR}/quick_test.py << 'TESTSCRIPT'
#!/usr/bin/env python3
"""
Quick test to verify SV3D and 2DGS installation.
"""

import sys
import os
import torch

# Set model cache
os.environ['HF_HOME'] = '/runpod-volume/sv3d_2dgs/models/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/runpod-volume/sv3d_2dgs/models/huggingface'

def test_sv3d():
    """Test SV3D can load."""
    print("Testing SV3D...")
    try:
        from diffusers import StableVideo3DPipeline
        pipe = StableVideo3DPipeline.from_pretrained(
            "stabilityai/sv3d",
            torch_dtype=torch.float16,
        )
        print("  ✓ SV3D loaded successfully")
        del pipe
        torch.cuda.empty_cache()
        return True
    except Exception as e:
        print(f"  ✗ SV3D failed: {e}")
        return False

def test_depth():
    """Test Depth Anything V2 can load."""
    print("Testing Depth Anything V2...")
    try:
        from transformers import pipeline
        pipe = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Large-hf")
        print("  ✓ Depth Anything V2 loaded successfully")
        del pipe
        torch.cuda.empty_cache()
        return True
    except Exception as e:
        print(f"  ✗ Depth Anything V2 failed: {e}")
        return False

def test_2dgs():
    """Test 2DGS CUDA extensions."""
    print("Testing 2DGS CUDA extensions...")
    try:
        import diff_gaussian_rasterization_2d
        print("  ✓ diff_gaussian_rasterization_2d loaded")
        import simple_knn
        print("  ✓ simple_knn loaded")
        return True
    except Exception as e:
        print(f"  ✗ 2DGS extensions failed: {e}")
        return False

def test_mesh():
    """Test mesh processing libraries."""
    print("Testing mesh processing...")
    try:
        import trimesh
        print("  ✓ trimesh loaded")
        import open3d
        print("  ✓ open3d loaded")
        return True
    except Exception as e:
        print(f"  ✗ Mesh processing failed: {e}")
        return False

def main():
    print("=" * 50)
    print("  SV3D + 2DGS Installation Test")
    print("=" * 50)
    print()
    
    # Check CUDA
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()
    
    results = []
    results.append(("SV3D", test_sv3d()))
    results.append(("Depth Estimation", test_depth()))
    results.append(("2DGS Extensions", test_2dgs()))
    results.append(("Mesh Processing", test_mesh()))
    
    print()
    print("=" * 50)
    print("  Results Summary")
    print("=" * 50)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("All tests passed! Environment is ready.")
        return 0
    else:
        print("Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
TESTSCRIPT

chmod +x ${SCRIPTS_DIR}/quick_test.py

echo "✓ Helper scripts created"

# -----------------------------------------------------------------------------
# Create Installation Marker
# -----------------------------------------------------------------------------

cat > ${INSTALL_MARKER} << MARKER
Installation completed: $(date)
Conda environment: ${CONDA_ENV_NAME}
Python version: $(python --version)
PyTorch version: $(python -c "import torch; print(torch.__version__)")
CUDA available: $(python -c "import torch; print(torch.cuda.is_available())")
MARKER

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------

echo ""
echo "=============================================="
echo "  Setup Complete!"
echo "=============================================="
echo ""
echo "Installation location (on network volume):"
echo "  Root:      ${SV3D_2DGS_ROOT}"
echo "  2DGS:      ${TWO_DGS_DIR}"
echo "  Inputs:    ${INPUTS_DIR}"
echo "  Outputs:   ${OUTPUTS_DIR}"
echo "  Models:    ${MODELS_DIR}"
echo ""
echo "Conda environment: ${CONDA_ENV_NAME}"
echo ""
echo "This installation is SEPARATE from your existing setups:"
echo "  - Gen3C/Lyra/SHARP remain in cosmos-predict1"
echo "  - SV3D/2DGS is in sv3d-2dgs environment"
echo ""
echo "----------------------------------------------"
echo "  Quick Start"
echo "----------------------------------------------"
echo ""
echo "On future pod starts, activate with:"
echo "  source ${SV3D_2DGS_ROOT}/activate.sh"
echo ""
echo "Or manually:"
echo "  conda activate ${CONDA_ENV_NAME}"
echo "  export HF_HOME=${MODELS_DIR}/huggingface"
echo ""
echo "Run quick test:"
echo "  python ${SCRIPTS_DIR}/quick_test.py"
echo ""
echo "Copy test image and run pipeline:"
echo "  cp your_image.jpg ${INPUTS_DIR}/"
echo "  python ${SCRIPTS_DIR}/test_pipeline.py --image ${INPUTS_DIR}/your_image.jpg --quick"
echo ""
