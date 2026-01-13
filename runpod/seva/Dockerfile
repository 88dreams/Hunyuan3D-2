# Stable Virtual Camera (SEVA) - RunPod Serverless Dockerfile
# 
# Model: Stability AI Stable Virtual Camera (1.3B parameters)
# Purpose: Novel View Synthesis with precise camera control
# Output: Video (MP4) from single image input
#
# Build:
#   docker build -t 88dreams/seva-runpod:v1 .
#   docker push 88dreams/seva-runpod:v1

FROM nvcr.io/nvidia/pytorch:24.10-py3

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_HOME=/workspace/checkpoints
ENV HF_HOME=/workspace/checkpoints/huggingface

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set up workspace
WORKDIR /workspace

# Upgrade PyTorch to 2.6.0+ (SEVA requirement)
RUN pip install --upgrade pip && \
    pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# Clone SEVA repository
RUN git clone --recursive https://github.com/Stability-AI/stable-virtual-camera /workspace/seva

# Install SEVA and dependencies
WORKDIR /workspace/seva
RUN pip install -e . || pip install -r requirements.txt || echo "SEVA install - may need manual setup"

# Install additional dependencies
RUN pip install boto3 runpod requests pillow imageio imageio-ffmpeg \
    einops transformers accelerate safetensors diffusers huggingface_hub

# Create output directory
RUN mkdir -p /runpod-volume/outputs/seva

# Copy handler and startup script
COPY handler_seva.py /workspace/handler.py
COPY start_seva.sh /workspace/start.sh
RUN chmod +x /workspace/start.sh

# Set working directory for runtime
WORKDIR /workspace

# Environment for HuggingFace authentication
# Set HF_TOKEN at runtime or in RunPod secrets
ENV HF_TOKEN=""

# Entrypoint
CMD ["/workspace/start.sh"]
