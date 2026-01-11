#!/bin/bash
# ViPE Setup Script for RunPod
# Repository: https://github.com/nv-tlabs/vipe
#
# IMPORTANT: ViPE requires a SEPARATE conda environment from Gen3C
# due to dependency conflicts.

set -e

echo "=============================================="
echo "  ViPE Setup for Gen3C → 2DGS Pipeline"
echo "=============================================="

# Configuration
WORKSPACE="/workspace"
VIPE_DIR="${WORKSPACE}/vipe"
CONDA_ENV_NAME="vipe"

# Check for conda
if ! command -v conda &> /dev/null; then
    echo "❌ conda not found. Checking for miniconda..."
    
    if [ -f "${WORKSPACE}/miniconda3/bin/conda" ]; then
        export PATH="${WORKSPACE}/miniconda3/bin:$PATH"
        eval "$(${WORKSPACE}/miniconda3/bin/conda shell.bash hook)"
        echo "✓ Using miniconda from ${WORKSPACE}/miniconda3"
    else
        echo "Installing Miniconda..."
        wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
        bash /tmp/miniconda.sh -b -p ${WORKSPACE}/miniconda3
        rm /tmp/miniconda.sh
        export PATH="${WORKSPACE}/miniconda3/bin:$PATH"
        eval "$(${WORKSPACE}/miniconda3/bin/conda shell.bash hook)"
        echo "✓ Miniconda installed"
    fi
fi

echo ""
echo "[1/5] Creating conda environment: ${CONDA_ENV_NAME}"
echo "----------------------------------------------"

# Check if environment already exists
if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
    echo "Environment ${CONDA_ENV_NAME} already exists"
    read -p "Recreate it? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        conda env remove -n ${CONDA_ENV_NAME} -y
        conda create -n ${CONDA_ENV_NAME} python=3.10 -y
    fi
else
    conda create -n ${CONDA_ENV_NAME} python=3.10 -y
fi

# Activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${CONDA_ENV_NAME}
echo "✓ Environment activated: ${CONDA_ENV_NAME}"

echo ""
echo "[2/5] Cloning ViPE repository"
echo "----------------------------------------------"

if [ -d "${VIPE_DIR}" ]; then
    echo "ViPE directory already exists at ${VIPE_DIR}"
    cd ${VIPE_DIR}
    git pull origin main || echo "Could not pull updates"
else
    cd ${WORKSPACE}
    git clone https://github.com/nv-tlabs/vipe.git
    cd ${VIPE_DIR}
fi

echo "✓ ViPE repository ready"

echo ""
echo "[3/5] Installing dependencies"
echo "----------------------------------------------"

# Install PyTorch first (CUDA 12.1 compatible)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install ViPE dependencies
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "Warning: requirements.txt not found"
    # Install common dependencies
    pip install numpy scipy opencv-python pillow tqdm
fi

# Install ViPE as editable package if setup.py exists
if [ -f "setup.py" ] || [ -f "pyproject.toml" ]; then
    pip install -e .
fi

echo "✓ Dependencies installed"

echo ""
echo "[4/5] Verifying installation"
echo "----------------------------------------------"

# Try to import and run vipe
python -c "
import sys
try:
    # Try importing vipe modules
    print('Checking ViPE installation...')
    
    # Check if vipe command is available
    import subprocess
    result = subprocess.run(['vipe', '--help'], capture_output=True, text=True)
    if result.returncode == 0:
        print('✓ vipe command available')
    else:
        print('⚠ vipe command not in PATH')
        print('  Try: pip install -e .')
except Exception as e:
    print(f'⚠ ViPE check failed: {e}')
    sys.exit(1)
"

echo ""
echo "[5/5] Creating activation script"
echo "----------------------------------------------"

# Create activation script
cat > ${WORKSPACE}/activate_vipe.sh << 'EOF'
#!/bin/bash
# Activate ViPE environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate vipe

export VIPE_DIR="/workspace/vipe"
export PATH="${VIPE_DIR}:$PATH"

echo "ViPE environment activated!"
echo "  Python: $(python --version)"
echo "  Conda env: vipe"
echo ""
echo "Usage:"
echo "  vipe infer video.mp4 --output results/"
echo "  vipe visualize results/"
EOF

chmod +x ${WORKSPACE}/activate_vipe.sh

echo "✓ Activation script created: ${WORKSPACE}/activate_vipe.sh"

echo ""
echo "=============================================="
echo "  ViPE Setup Complete!"
echo "=============================================="
echo ""
echo "To activate ViPE environment:"
echo "  source /workspace/activate_vipe.sh"
echo ""
echo "To process a Gen3C video:"
echo "  vipe infer /path/to/gen3c_video.mp4 --output vipe_results/"
echo ""
echo "Output will include:"
echo "  - poses.npy      (N × 4 × 4 camera poses)"
echo "  - intrinsics.npy (camera intrinsics)"
echo "  - depth/*.npy    (dense depth maps)"
echo ""
echo "Note: ViPE runs at 3-5 FPS on a single GPU"
echo "      A 121-frame video takes ~25-40 seconds"
