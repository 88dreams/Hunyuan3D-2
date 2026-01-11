# ViPE Serverless Endpoint Dockerfile
# NVIDIA's Video Pose Engine for camera pose + depth extraction
#
# Build: docker build -t your-dockerhub/vipe-serverless:v1 .
# Push:  docker push your-dockerhub/vipe-serverless:v1

FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    ninja-build \
    libeigen3-dev \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.4
RUN pip install --no-cache-dir \
    torch==2.4.0 \
    torchvision==0.19.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Clone and install ViPE
WORKDIR /workspace
RUN git clone https://github.com/nv-tlabs/vipe.git

WORKDIR /workspace/vipe
RUN pip install --no-cache-dir -e .

# Install additional dependencies
RUN pip install --no-cache-dir \
    runpod \
    requests \
    boto3 \
    opencv-python-headless \
    numpy \
    scipy \
    hydra-core \
    omegaconf

# Copy handler script
COPY handler_vipe.py /workspace/handler_vipe.py

# Set working directory
WORKDIR /workspace

# Set environment variables
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"

# RunPod serverless entry point
CMD ["python", "-u", "/workspace/handler_vipe.py"]
