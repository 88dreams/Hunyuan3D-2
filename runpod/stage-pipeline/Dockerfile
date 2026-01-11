# Stage Pipeline: ViPE + 2DGS Combined
# Converts Gen3C video → 3D Mesh in one endpoint

FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libeigen3-dev \
    build-essential \
    cmake \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

ENV PATH="/opt/conda/bin:$PATH"

# Accept Conda TOS (required for non-interactive builds)
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

# Create environment with Python 3.10
RUN conda create -n pipeline python=3.10 -y
SHELL ["conda", "run", "-n", "pipeline", "/bin/bash", "-c"]

# Install PyTorch with CUDA 12.4
RUN pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124

# ============================================
# VIPE INSTALLATION
# ============================================
WORKDIR /opt

# Clone ViPE
RUN git clone https://github.com/nv-tlabs/vipe.git

WORKDIR /opt/vipe

# Install ViPE dependencies (from pyproject.toml + extras)
RUN pip install \
    numpy==1.26.4 \
    opencv-python-headless \
    einops \
    scipy \
    tqdm \
    plyfile \
    trimesh \
    imageio \
    imageio-ffmpeg \
    PyYAML \
    matplotlib \
    open3d \
    OpenEXR \
    ninja \
    omegaconf \
    hydra-core \
    rerun-sdk \
    python-pycg \
    timm \
    transformers \
    gdown \
    kornia \
    viser \
    click

# Build ViPE CUDA extensions (includes lietorch, correlation, etc.)
# CRITICAL: Set ABI flag to match PyTorch (PyTorch wheels use old ABI)
# GLIBCXX_USE_CXX11_ABI is what PyTorch's cpp_extension module checks
ENV CPLUS_INCLUDE_PATH="/usr/include/eigen3:$CPLUS_INCLUDE_PATH"
ENV GLIBCXX_USE_CXX11_ABI="0"
ENV _GLIBCXX_USE_CXX11_ABI="0"

# CRITICAL: Remove precompiled .so files from repo - they have wrong ABI
RUN find . -name "*.so" -delete && \
    find . -name "*.egg-info" -type d -exec rm -rf {} + 2>/dev/null || true

# Set compiler flags for correct ABI (PyTorch wheels use old ABI)
ENV CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=0"
ENV CFLAGS="-D_GLIBCXX_USE_CXX11_ABI=0"

# Explicitly build the CUDA extension first (pip install -e . may skip it)
RUN python setup.py build_ext --inplace 2>&1 | tail -50

# Verify extension was built
RUN ls -la /opt/vipe/*.so && echo "✓ Extension built successfully"

# Install package
RUN pip install -e . --no-build-isolation

# ============================================
# 2DGS INSTALLATION  
# ============================================
WORKDIR /opt

# Clone 2D Gaussian Splatting
RUN git clone https://github.com/hbb1/2d-gaussian-splatting.git

WORKDIR /opt/2d-gaussian-splatting

# Initialize submodules
RUN git submodule update --init --recursive

# Install 2DGS dependencies
RUN pip install \
    plyfile \
    tqdm \
    lpips \
    tensorboard \
    mediapy \
    scikit-image

# Build CUDA rasterizer
RUN pip install --no-build-isolation submodules/diff-surfel-rasterization
RUN pip install --no-build-isolation submodules/simple-knn

# ============================================
# PIPELINE SCRIPTS
# ============================================
WORKDIR /opt/pipeline

# Copy handler and helper scripts
COPY handler.py /opt/pipeline/
COPY vipe_to_2dgs.py /opt/pipeline/
COPY init_points_from_depth.py /opt/pipeline/

# Install RunPod
RUN pip install runpod boto3 requests

# Verify installations
RUN python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
RUN python -c "import vipe_ext; print('vipe_ext loaded successfully')" || echo "WARNING: vipe_ext failed to load"

# Set paths
ENV VIPE_PATH="/opt/vipe"
ENV TDGS_PATH="/opt/2d-gaussian-splatting"
ENV PYTHONPATH="/opt/vipe:/opt/2d-gaussian-splatting:$PYTHONPATH"

# Entry point
CMD ["conda", "run", "--no-capture-output", "-n", "pipeline", "python", "-u", "/opt/pipeline/handler.py"]
