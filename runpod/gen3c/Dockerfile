# =============================================================================
# GEN3C Inference Docker Image for RunPod
# =============================================================================
# Base: NVIDIA PyTorch with CUDA 12.1
# Target: RunPod GPU Pods (A100 80GB recommended)
# =============================================================================

FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# =============================================================================
# SYSTEM DEPENDENCIES
# =============================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
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
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

# =============================================================================
# WORKING DIRECTORY
# =============================================================================
WORKDIR /workspace

# =============================================================================
# PYTHON DEPENDENCIES (Base)
# =============================================================================
# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Install core ML dependencies
RUN pip install --no-cache-dir \
    torch==2.2.0 \
    torchvision==0.17.0 \
    torchaudio==2.2.0 \
    --index-url https://download.pytorch.org/whl/cu121

# =============================================================================
# CLONE GEN3C REPOSITORY
# =============================================================================
RUN git clone https://github.com/nv-tlabs/GEN3C.git /workspace/GEN3C

WORKDIR /workspace/GEN3C

# =============================================================================
# GEN3C DEPENDENCIES
# =============================================================================
# Install GEN3C requirements (excluding torch which we already have)
COPY requirements-gen3c.txt /workspace/requirements-gen3c.txt
RUN pip install --no-cache-dir -r /workspace/requirements-gen3c.txt

# =============================================================================
# NVIDIA APEX (with CUDA extensions)
# =============================================================================
RUN git clone https://github.com/NVIDIA/apex /tmp/apex && \
    cd /tmp/apex && \
    pip install -v --disable-pip-version-check --no-cache-dir \
        --no-build-isolation \
        --config-settings "--build-option=--cpp_ext" \
        --config-settings "--build-option=--cuda_ext" \
        . && \
    rm -rf /tmp/apex

# =============================================================================
# TRANSFORMER ENGINE
# =============================================================================
RUN pip install --no-cache-dir transformer-engine[pytorch]==1.12.0

# =============================================================================
# MOGE (Microsoft Monocular Geometry Estimation)
# =============================================================================
RUN pip install --no-cache-dir git+https://github.com/microsoft/MoGe.git

# =============================================================================
# NVIDIA WARP (for ray-triangle intersection)
# =============================================================================
RUN pip install --no-cache-dir warp-lang

# =============================================================================
# HUGGINGFACE CLI (for checkpoint downloads)
# =============================================================================
RUN pip install --no-cache-dir huggingface_hub[cli]

# =============================================================================
# COPY SERVER/HANDLER SCRIPTS
# =============================================================================
COPY server.py /workspace/server.py
COPY handler.py /workspace/handler.py
COPY start.sh /workspace/start.sh
RUN chmod +x /workspace/start.sh

# =============================================================================
# ENVIRONMENT VARIABLES
# =============================================================================
ENV PYTHONPATH="/workspace/GEN3C:${PYTHONPATH}"
ENV CUDA_HOME="/usr/local/cuda"
ENV TRANSFORMERS_CACHE="/workspace/checkpoints/huggingface"
ENV HF_HOME="/workspace/checkpoints/huggingface"

# Checkpoint directory (mounted from network volume)
ENV GEN3C_CHECKPOINT_DIR="/workspace/checkpoints/Gen3C-Cosmos-7B"

# =============================================================================
# EXPOSE PORTS
# =============================================================================
# API server port
EXPOSE 8000
# Gradio port (if running embedded UI)
EXPOSE 7860

# =============================================================================
# DEFAULT COMMAND
# =============================================================================
# For GPU Pods: Run the API server
# For Serverless: RunPod will call handler.py directly
CMD ["/workspace/start.sh"]

