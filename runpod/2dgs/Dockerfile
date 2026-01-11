# 2DGS Serverless Endpoint Dockerfile
# 2D Gaussian Splatting training and mesh extraction
#
# Build: docker build -t your-dockerhub/2dgs-serverless:v1 .
# Push:  docker push your-dockerhub/2dgs-serverless:v1

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
    cmake \
    libglew-dev \
    libassimp-dev \
    libboost-all-dev \
    libgtk-3-dev \
    libopencv-dev \
    libglfw3-dev \
    libavdevice-dev \
    libavcodec-dev \
    libeigen3-dev \
    libxxf86vm-dev \
    libembree-dev \
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

# Install base dependencies
RUN pip install --no-cache-dir \
    runpod \
    requests \
    boto3 \
    numpy==1.26.4 \
    scipy \
    opencv-python-headless \
    plyfile \
    tqdm \
    trimesh \
    open3d \
    mediapy

# Clone 2DGS repository
WORKDIR /workspace
RUN git clone https://github.com/hbb1/2d-gaussian-splatting.git --recursive

# Install 2DGS CUDA extensions
WORKDIR /workspace/2d-gaussian-splatting
RUN pip install --no-cache-dir submodules/diff-surfel-rasterization
RUN pip install --no-cache-dir submodules/simple-knn

# Install remaining 2DGS dependencies
RUN pip install --no-cache-dir \
    lpips \
    pytorch-msssim \
    tensorboard

# Copy handler and utility scripts
COPY handler_2dgs.py /workspace/handler_2dgs.py
COPY vipe_to_2dgs.py /workspace/vipe_to_2dgs.py
COPY init_points_from_depth.py /workspace/init_points_from_depth.py

# Patch mesh_utils.py for trimesh 4.x compatibility
RUN sed -i 's/mesh = mesh.as_open3d/o3d_mesh = o3d.geometry.TriangleMesh(); o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices); o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces); mesh = o3d_mesh/' \
    /workspace/2d-gaussian-splatting/utils/mesh_utils.py

# Set working directory
WORKDIR /workspace

# Set environment variables
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"

# RunPod serverless entry point
CMD ["python", "-u", "/workspace/handler_2dgs.py"]
