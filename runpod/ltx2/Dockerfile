# LTX-2 RunPod Serverless Docker Image
# 
# Lightricks LTX-2: 19B parameter video generation model
# Supports image-to-video with camera control LoRAs
#
# Build:
#   docker build -t 88dreams/ltx2-runpod:v1 .
#   docker push 88dreams/ltx2-runpod:v1

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4-devel-ubuntu22.04

# System dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git-lfs \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Install Python dependencies
# Note: LTX-2 requires diffusers >= 0.32.0
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
    diffusers>=0.32.0 \
    transformers>=4.45.0 \
    accelerate>=0.34.0 \
    torch>=2.4.0 \
    torchvision \
    safetensors \
    boto3 \
    runpod \
    imageio[ffmpeg] \
    imageio-ffmpeg \
    pillow \
    numpy \
    requests \
    sentencepiece

# Clone LTX-2 repository for reference/utilities
RUN git clone --depth 1 https://github.com/Lightricks/LTX-2.git ltx2-repo || true

# Create directories
RUN mkdir -p /workspace/outputs /workspace/models

# Copy handler and startup script
COPY handler_ltx2.py /workspace/handler.py
COPY start_ltx2.sh /start.sh
RUN chmod +x /start.sh

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/runpod-volume/huggingface
ENV TRANSFORMERS_CACHE=/runpod-volume/huggingface
ENV HF_HUB_CACHE=/runpod-volume/huggingface/hub
ENV OUTPUT_DIR=/runpod-volume/outputs/ltx2
ENV MODEL_CACHE_DIR=/runpod-volume/models/ltx2

# Default to distilled model for faster inference
ENV LTX2_MODEL_VARIANT=ltx-2-19b-distilled

# RunPod serverless
CMD ["/start.sh"]
