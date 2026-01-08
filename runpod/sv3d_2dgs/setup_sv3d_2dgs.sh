#!/bin/bash
# =============================================================================
# SV3D + 2DGS Setup Script for RunPod
# =============================================================================
#
# This script sets up the environment for SV3D multi-view generation and
# 2DGS training on a RunPod pod.
#
# Usage:
#   1. Start a RunPod pod with PyTorch template (A6000 or A100 recommended)
#   2. SSH into the pod or use web terminal
#   3. Run: bash setup_sv3d_2dgs.sh
#
# Estimated time: 15-20 minutes
#
# =============================================================================

set -e  # Exit on error

echo "=============================================="
echo "  SV3D + 2DGS Environment Setup"
echo "=============================================="
echo ""
echo "This will install:"
echo "  - SV3D (Stable Video 3D) for multi-view generation"
echo "  - 2DGS (2D Gaussian Splatting) for 3D reconstruction"
echo "  - Depth Anything V2 for depth estimation"
echo "  - Supporting libraries for mesh extraction"
echo ""
echo "Estimated time: 15-20 minutes"
echo ""

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

WORKSPACE="/workspace"
SV3D_2DGS_DIR="${WORKSPACE}/sv3d_2dgs"
TWO_DGS_DIR="${WORKSPACE}/2d-gaussian-splatting"

# Create workspace directories
mkdir -p ${SV3D_2DGS_DIR}
mkdir -p ${SV3D_2DGS_DIR}/inputs
mkdir -p ${SV3D_2DGS_DIR}/outputs
mkdir -p ${SV3D_2DGS_DIR}/models

cd ${WORKSPACE}

# -----------------------------------------------------------------------------
# Step 1: System Dependencies
# -----------------------------------------------------------------------------

echo ""
echo "[1/7] Installing system dependencies..."
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
    colmap \
    > /dev/null 2>&1

echo "✓ System dependencies installed"

# -----------------------------------------------------------------------------
# Step 2: Python Dependencies (Core)
# -----------------------------------------------------------------------------

echo ""
echo "[2/7] Installing core Python dependencies..."
echo "----------------------------------------------"

pip install --upgrade pip -q

# Core ML libraries
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
# Step 3: SV3D Dependencies
# -----------------------------------------------------------------------------

echo ""
echo "[3/7] Installing SV3D dependencies..."
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
# Step 4: Clone and Install 2DGS
# -----------------------------------------------------------------------------

echo ""
echo "[4/7] Cloning and installing 2D Gaussian Splatting..."
echo "----------------------------------------------"

cd ${WORKSPACE}

if [ -d "${TWO_DGS_DIR}" ]; then
    echo "2DGS directory exists, pulling latest..."
    cd ${TWO_DGS_DIR}
    git pull
else
    echo "Cloning 2DGS repository..."
    git clone https://github.com/hbb1/2d-gaussian-splatting.git ${TWO_DGS_DIR}
    cd ${TWO_DGS_DIR}
fi

# Install 2DGS Python dependencies
pip install -q -r requirements.txt

# Build CUDA extensions
echo "Building 2DGS CUDA extensions (this may take a few minutes)..."
pip install -q submodules/diff-gaussian-rasterization-2d
pip install -q submodules/simple-knn

echo "✓ 2DGS installed"

# -----------------------------------------------------------------------------
# Step 5: Depth Estimation Dependencies
# -----------------------------------------------------------------------------

echo ""
echo "[5/7] Installing depth estimation dependencies..."
echo "----------------------------------------------"

pip install -q \
    einops>=0.7.0 \
    timm>=0.9.0

echo "✓ Depth estimation dependencies installed"

# -----------------------------------------------------------------------------
# Step 6: Mesh Processing Dependencies
# -----------------------------------------------------------------------------

echo ""
echo "[6/7] Installing mesh processing dependencies..."
echo "----------------------------------------------"

pip install -q \
    trimesh>=4.0.0 \
    open3d>=0.17.0 \
    pymeshlab>=2023.12

echo "✓ Mesh processing dependencies installed"

# -----------------------------------------------------------------------------
# Step 7: Download Models (Optional - uncomment to pre-download)
# -----------------------------------------------------------------------------

echo ""
echo "[7/7] Pre-downloading models..."
echo "----------------------------------------------"

# Pre-download SV3D model
echo "Downloading SV3D model (this may take a few minutes)..."
python -c "
from diffusers import StableVideo3DPipeline
import torch
print('Downloading SV3D...')
pipe = StableVideo3DPipeline.from_pretrained(
    'stabilityai/sv3d',
    torch_dtype=torch.float16,
    variant='fp16',
)
print('SV3D downloaded successfully!')
"

# Pre-download Depth Anything V2
echo "Downloading Depth Anything V2..."
python -c "
from transformers import pipeline
print('Downloading Depth Anything V2...')
pipe = pipeline('depth-estimation', model='depth-anything/Depth-Anything-V2-Large-hf')
print('Depth Anything V2 downloaded successfully!')
"

echo "✓ Models downloaded"

# -----------------------------------------------------------------------------
# Create Helper Scripts
# -----------------------------------------------------------------------------

echo ""
echo "Creating helper scripts..."
echo "----------------------------------------------"

# Create a quick test script
cat > ${SV3D_2DGS_DIR}/quick_test.py << 'TESTSCRIPT'
#!/usr/bin/env python3
"""
Quick test to verify SV3D and 2DGS installation.
"""

import sys
import torch

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

chmod +x ${SV3D_2DGS_DIR}/quick_test.py

echo "✓ Helper scripts created"

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------

echo ""
echo "=============================================="
echo "  Setup Complete!"
echo "=============================================="
echo ""
echo "Installation locations:"
echo "  - Workspace:     ${SV3D_2DGS_DIR}"
echo "  - 2DGS:          ${TWO_DGS_DIR}"
echo "  - Inputs:        ${SV3D_2DGS_DIR}/inputs"
echo "  - Outputs:       ${SV3D_2DGS_DIR}/outputs"
echo ""
echo "Next steps:"
echo "  1. Run quick test:  python ${SV3D_2DGS_DIR}/quick_test.py"
echo "  2. Copy test image: cp your_image.jpg ${SV3D_2DGS_DIR}/inputs/"
echo "  3. Run pipeline:    python ${SV3D_2DGS_DIR}/test_pipeline.py"
echo ""
echo "VRAM usage (estimated):"
echo "  - SV3D inference:    ~20GB"
echo "  - 2DGS training:     ~15GB"
echo "  - Total (sequential): ~20GB peak"
echo ""
echo "Recommended GPU: A6000 (48GB) or A100 (40GB)"
echo ""

