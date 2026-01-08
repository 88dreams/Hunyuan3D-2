# =============================================================================
# Dockerfile for SV3D + 2DGS Pipeline
# =============================================================================
#
# This Dockerfile creates a RunPod-ready image with:
# - SV3D (Stable Video 3D) for multi-view generation
# - 2DGS (2D Gaussian Splatting) for 3D reconstruction
# - Depth Anything V2 for depth estimation
# - Mesh processing libraries
#
# Build:
#   docker build -t 88dreams/sv3d-2dgs-runpod:v1 -f Dockerfile .
#
# Push:
#   docker push 88dreams/sv3d-2dgs-runpod:v1
#
# =============================================================================

# Base image with PyTorch and CUDA
FROM nvcr.io/nvidia/pytorch:24.01-py3

# Environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0"

# Workspace setup
WORKDIR /workspace

# -----------------------------------------------------------------------------
# System Dependencies
# -----------------------------------------------------------------------------

RUN apt-get update && apt-get install -y --no-install-recommends \
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
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# Python Dependencies
# -----------------------------------------------------------------------------

# Upgrade pip
RUN pip install --upgrade pip

# Core dependencies
RUN pip install --no-cache-dir \
    numpy>=1.24.0 \
    scipy>=1.11.0 \
    pillow>=10.0.0 \
    opencv-python>=4.8.0 \
    imageio>=2.31.0 \
    imageio-ffmpeg>=0.4.9 \
    tqdm>=4.66.0

# SV3D dependencies
RUN pip install --no-cache-dir \
    diffusers>=0.27.0 \
    transformers>=4.36.0 \
    accelerate>=0.25.0 \
    safetensors>=0.4.0

# Background removal
RUN pip install --no-cache-dir rembg[gpu]

# Depth estimation dependencies
RUN pip install --no-cache-dir \
    einops>=0.7.0 \
    timm>=0.9.0

# Mesh processing
RUN pip install --no-cache-dir \
    trimesh>=4.0.0 \
    open3d>=0.17.0 \
    pymeshlab>=2023.12

# RunPod and AWS
RUN pip install --no-cache-dir \
    runpod>=1.6.0 \
    boto3>=1.34.0

# -----------------------------------------------------------------------------
# 2DGS Installation
# -----------------------------------------------------------------------------

# Clone 2DGS repository
RUN git clone https://github.com/hbb1/2d-gaussian-splatting.git /workspace/2d-gaussian-splatting

# Install 2DGS Python dependencies
WORKDIR /workspace/2d-gaussian-splatting
RUN pip install --no-cache-dir -r requirements.txt

# Build CUDA extensions
RUN pip install --no-cache-dir submodules/diff-gaussian-rasterization-2d
RUN pip install --no-cache-dir submodules/simple-knn

WORKDIR /workspace

# -----------------------------------------------------------------------------
# Pre-download Models (speeds up first run significantly)
# -----------------------------------------------------------------------------

# Download SV3D model
RUN python -c "\
from diffusers import StableVideo3DPipeline; \
import torch; \
print('Downloading SV3D...'); \
pipe = StableVideo3DPipeline.from_pretrained('stabilityai/sv3d', torch_dtype=torch.float16, variant='fp16'); \
print('SV3D downloaded!'); \
del pipe; \
"

# Download Depth Anything V2
RUN python -c "\
from transformers import pipeline; \
print('Downloading Depth Anything V2...'); \
pipe = pipeline('depth-estimation', model='depth-anything/Depth-Anything-V2-Large-hf'); \
print('Depth Anything V2 downloaded!'); \
del pipe; \
"

# Download rembg model
RUN python -c "\
from rembg import new_session; \
print('Downloading rembg model...'); \
session = new_session('u2net'); \
print('rembg model downloaded!'); \
"

# -----------------------------------------------------------------------------
# Create Directory Structure
# -----------------------------------------------------------------------------

RUN mkdir -p /workspace/sv3d_2dgs/inputs \
    && mkdir -p /workspace/sv3d_2dgs/outputs \
    && mkdir -p /workspace/sv3d_2dgs/models

# -----------------------------------------------------------------------------
# Copy Handler Files (will be added later)
# -----------------------------------------------------------------------------

# Copy generator modules
# COPY generators/ /workspace/generators/

# Copy handler
# COPY handler_sv3d_2dgs.py /workspace/

# -----------------------------------------------------------------------------
# Health Check Script
# -----------------------------------------------------------------------------

RUN echo '#!/bin/bash\n\
python -c "import torch; print(f\"CUDA: {torch.cuda.is_available()}\")" && \
python -c "from diffusers import StableVideo3DPipeline; print(\"SV3D: OK\")" && \
python -c "import diff_gaussian_rasterization_2d; print(\"2DGS: OK\")" && \
python -c "import trimesh; print(\"Trimesh: OK\")"\n\
' > /workspace/health_check.sh && chmod +x /workspace/health_check.sh

# -----------------------------------------------------------------------------
# Default Command
# -----------------------------------------------------------------------------

# For interactive development
CMD ["/bin/bash"]

# For serverless (uncomment when handler is ready)
# CMD ["python", "-u", "handler_sv3d_2dgs.py"]

