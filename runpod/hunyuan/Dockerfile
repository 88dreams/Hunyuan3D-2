# Hunyuan3D RunPod Serverless Docker Image
# Tencent's Hunyuan3D: Image → High-Quality GLB Mesh
#
# Base: NVIDIA PyTorch with CUDA 12.4
# Models: Mini (9GB, faster) and Full (23GB, higher quality)
#
# v8: Use runpod base image instead of NVIDIA (smaller, better RunPod compatibility)

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies (including OpenGL for pymeshlab)
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    wget \
    curl \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libopengl0 \
    libglx0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Create pip constraints file to PREVENT numpy upgrades
RUN echo "numpy<2.0" > /tmp/constraints.txt

# Install numpy and opencv first (runpod base should be cleaner)
RUN pip install --no-cache-dir "numpy==1.26.4" && \
    pip install --no-cache-dir "opencv-python==4.8.1.78"

# Clone Hunyuan3D-2 repository
RUN git clone https://github.com/Tencent/Hunyuan3D-2.git /workspace/Hunyuan3D-2

# Install Hunyuan3D dependencies WITH CONSTRAINTS to prevent numpy upgrade
RUN cd /workspace/Hunyuan3D-2 && \
    pip install --no-cache-dir -c /tmp/constraints.txt -e . && \
    pip install --no-cache-dir -c /tmp/constraints.txt \
        trimesh \
        pygltflib \
        plyfile \
        pillow \
        scipy \
        tqdm \
        einops \
        transformers \
        accelerate \
        safetensors \
        huggingface_hub

# Install rembg WITH CONSTRAINTS - this is likely what was upgrading numpy
RUN pip install --no-cache-dir -c /tmp/constraints.txt rembg[gpu] onnxruntime-gpu

# Install RunPod SDK and boto3 WITH CONSTRAINTS
RUN pip install --no-cache-dir -c /tmp/constraints.txt runpod boto3 requests

# FINAL CLEANUP: Ensure correct versions
RUN pip install --no-cache-dir "numpy==1.26.4" && \
    python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"

# Copy handler and scripts
COPY handler_hunyuan.py /workspace/handler.py
COPY start_hunyuan.sh /workspace/start.sh
RUN chmod +x /workspace/start.sh

# Create output directory
RUN mkdir -p /runpod-volume/outputs/hunyuan

# Environment
ENV PYTHONPATH="/workspace/Hunyuan3D-2"
ENV HF_HOME="/runpod-volume/huggingface"
ENV HUNYUAN_CHECKPOINT_DIR="/runpod-volume/checkpoints/hunyuan"

# Default command
CMD ["/workspace/start.sh"]

