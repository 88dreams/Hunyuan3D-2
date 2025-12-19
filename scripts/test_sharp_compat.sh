#!/bin/bash
# =============================================================================
# Test SHARP Compatibility with NVIDIA Stack (GEN3C/Lyra/TRELLIS.2)
# =============================================================================
# This script tests if SHARP can run in the same Python 3.10 environment
# that GEN3C, Lyra, and TRELLIS.2 use (NumPy 1.x instead of 2.x)
#
# Based on dependency analysis:
# - GEN3C/Lyra/TRELLIS.2: Python 3.10, NumPy 1.26.4, PyTorch 2.6.0
# - SHARP official: Python 3.13, NumPy 2.3.3, PyTorch 2.8.0
# =============================================================================

set -e

echo "=============================================="
echo "SHARP Compatibility Test with NVIDIA Stack"
echo "=============================================="
echo ""
echo "Testing if SHARP can work with:"
echo "  - Python 3.10 (instead of 3.13)"
echo "  - NumPy 1.26.4 (instead of 2.3.3)"
echo "  - PyTorch 2.6.0 (instead of 2.8.0)"
echo ""

# Activate the GEN3C environment (gen3c-rocm310 has Python 3.10 + NumPy 1.x)
# gen3c-rocm has Python 3.12 + NumPy 2.x which is NOT compatible
ENV_NAME="${CONDA_ENV:-gen3c-rocm310}"
echo "Using conda environment: $ENV_NAME"

if [ -f ~/opt/miniconda3/bin/activate ]; then
    source ~/opt/miniconda3/bin/activate "$ENV_NAME"
elif [ -f ~/miniconda3/bin/activate ]; then
    source ~/miniconda3/bin/activate "$ENV_NAME"
else
    echo "❌ Cannot find conda. Please activate $ENV_NAME manually."
    exit 1
fi

echo "=============================================="
echo "Current Environment Info"
echo "=============================================="
echo "  Python: $(python --version 2>&1)"
echo "  NumPy: $(python -c 'import numpy; print(numpy.__version__)' 2>&1)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>&1)"
echo ""

# Check Python version
PYTHON_VERSION=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [ "$PYTHON_VERSION" != "3.10" ] && [ "$PYTHON_VERSION" != "3.12" ]; then
    echo "⚠️  Warning: Python $PYTHON_VERSION detected. NVIDIA stack uses 3.10."
fi

# Check NumPy version
NUMPY_VERSION=$(python -c 'import numpy; print(numpy.__version__)')
if [[ "$NUMPY_VERSION" == 2.* ]]; then
    echo "❌ NumPy 2.x detected. This test requires NumPy 1.x."
    echo "   Run: pip install numpy==1.26.4"
    exit 1
fi
echo "✓ NumPy version OK: $NUMPY_VERSION"

echo ""
echo "=============================================="
echo "Installing SHARP Dependencies"
echo "=============================================="

# Install SHARP's key dependencies without version constraints
echo "Installing gsplat..."
pip install gsplat --quiet 2>/dev/null || {
    echo "⚠️  gsplat pip install failed. Trying from git..."
    pip install git+https://github.com/nerfstudio-project/gsplat.git --quiet 2>/dev/null || {
        echo "⚠️  gsplat install failed (may need CUDA compilation)"
    }
}

echo "Installing other SHARP dependencies..."
pip install plyfile pillow-heif scipy click rich matplotlib --quiet 2>/dev/null || true

echo ""
echo "=============================================="
echo "Installing SHARP"
echo "=============================================="

SHARP_DIR="/tmp/ml-sharp-test-$$"
rm -rf "$SHARP_DIR"

echo "Cloning SHARP repository..."
git clone --depth 1 https://github.com/apple/ml-sharp.git "$SHARP_DIR" 2>/dev/null

cd "$SHARP_DIR"

echo "Attempting SHARP installation (editable, no deps)..."
pip install -e . --no-deps 2>&1 | head -20 || {
    echo ""
    echo "Trying regular install..."
    pip install . --no-deps 2>&1 | head -20 || {
        echo ""
        echo "❌ SHARP installation failed."
        echo ""
        echo "This likely means SHARP requires Python 3.13 features."
        echo "Recommendation: Use separate Docker images for SHARP."
        cd /
        rm -rf "$SHARP_DIR"
        exit 1
    }
}

echo ""
echo "=============================================="
echo "Testing SHARP Import"
echo "=============================================="

python -c "
import sys
print(f'Python version: {sys.version}')
print()

# Test core imports
errors = []

try:
    import numpy as np
    print(f'✓ NumPy {np.__version__}')
except ImportError as e:
    errors.append(f'NumPy: {e}')

try:
    import torch
    print(f'✓ PyTorch {torch.__version__}')
    print(f'  CUDA available: {torch.cuda.is_available()}')
except ImportError as e:
    errors.append(f'PyTorch: {e}')

try:
    import gsplat
    print('✓ gsplat imported')
except ImportError as e:
    print(f'⚠ gsplat not available (expected on ROCm): {e}')

try:
    import plyfile
    print('✓ plyfile imported')
except ImportError as e:
    errors.append(f'plyfile: {e}')

# Try importing SHARP
print()
print('Testing SHARP import...')
try:
    import sharp
    print('✓ sharp module imported successfully!')
    
    # Check for key functions
    if hasattr(sharp, 'main'):
        print('✓ sharp.main found')
    
except ImportError as e:
    errors.append(f'sharp: {e}')
    print(f'✗ sharp import failed: {e}')

except Exception as e:
    errors.append(f'sharp (other): {e}')
    print(f'✗ sharp import error: {e}')

print()
if errors:
    print('=== ERRORS ===')
    for err in errors:
        print(f'  ✗ {err}')
    print()
    print('SHARP may not be fully compatible with NumPy 1.x')
    sys.exit(1)
else:
    print('=== ALL IMPORTS SUCCESSFUL ===')
    print('SHARP appears compatible with the NVIDIA stack!')
"

IMPORT_RESULT=$?

echo ""
echo "=============================================="
echo "Testing SHARP CLI"
echo "=============================================="

if command -v sharp &> /dev/null; then
    echo "Testing 'sharp --help'..."
    sharp --help 2>&1 | head -20 || echo "⚠️  CLI help failed"
    CLI_OK=true
else
    echo "⚠️  SHARP CLI not in PATH"
    CLI_OK=false
fi

# Cleanup
cd /
rm -rf "$SHARP_DIR"

echo ""
echo "=============================================="
if [ $IMPORT_RESULT -eq 0 ]; then
    echo "✅ SHARP COMPATIBILITY TEST PASSED!"
    echo "=============================================="
    echo ""
    echo "SHARP can be installed alongside GEN3C/Lyra/TRELLIS.2!"
    echo ""
    echo "Recommended next steps:"
    echo "  1. Build unified Docker image with all models"
    echo "  2. Use single RunPod serverless endpoint"
    echo ""
    echo "Unified Dockerfile should include:"
    echo "  - Python 3.10"
    echo "  - NumPy 1.26.4"
    echo "  - PyTorch 2.6.0+cu124"
    echo "  - All model dependencies"
else
    echo "❌ SHARP COMPATIBILITY TEST FAILED"
    echo "=============================================="
    echo ""
    echo "SHARP requires Python 3.13 or NumPy 2.x features."
    echo ""
    echo "Recommended approach:"
    echo "  1. Create separate Docker image for SHARP"
    echo "  2. Use separate RunPod endpoint for SHARP"
    echo "  3. Keep GEN3C/Lyra/TRELLIS.2 in unified image"
    exit 1
fi
