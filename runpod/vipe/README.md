# ViPE Serverless Deployment Guide

This folder contains everything needed to deploy **NVIDIA's ViPE** (Video Pose Engine) as a serverless endpoint on RunPod.

## Overview

ViPE extracts from video:
- **Camera poses** (4×4 matrices per frame)
- **Camera intrinsics** (fx, fy, cx, cy)
- **Dense depth maps** (EXR format)
- **RGB frames**

This is the first step in the stage reconstruction pipeline:
`Gen3C Video → **ViPE** → 2DGS Training → Mesh`

## Files

| File | Description |
|------|-------------|
| `Dockerfile` | Docker image definition |
| `handler_vipe.py` | RunPod serverless handler |
| `setup_vipe.sh` | Pod installation script (for testing) |

## Deployment Steps

### Step 1: Build Docker Image

```bash
cd /path/to/Hunyuan3D-2-Fork/runpod/vipe

# Build the image (takes ~20-30 minutes due to CUDA extensions)
docker build -t YOUR_DOCKERHUB_USERNAME/vipe-serverless:v1 .

# Test locally (optional)
docker run --gpus all -it YOUR_DOCKERHUB_USERNAME/vipe-serverless:v1 bash

# Push to Docker Hub
docker push YOUR_DOCKERHUB_USERNAME/vipe-serverless:v1
```

### Step 2: Create RunPod Serverless Endpoint

1. Go to [RunPod Serverless](https://www.runpod.io/console/serverless)
2. Click **"+ New Endpoint"**
3. Configure:

| Setting | Value |
|---------|-------|
| **Name** | `vipe-serverless` |
| **Docker Image** | `YOUR_DOCKERHUB_USERNAME/vipe-serverless:v1` |
| **GPU Type** | RTX 4090 or A100 (needs VRAM for SLAM) |
| **Min Workers** | 0 |
| **Max Workers** | 3 |
| **GPU Count** | 1 |
| **Container Disk** | 20 GB |
| **Volume Disk** | 0 GB (stateless) |
| **Idle Timeout** | 60 seconds |

4. Click **"Create Endpoint"**
5. Copy the **Endpoint ID** and **API Key**

### Step 3: Test the Endpoint

```python
import runpod

runpod.api_key = "YOUR_API_KEY"

job = runpod.run_sync(
    endpoint_id="YOUR_ENDPOINT_ID",
    input={
        "video_url": "https://your-bucket.s3.amazonaws.com/gen3c_video.mp4",
        "output_s3": {
            "bucket": "your-bucket",
            "prefix": "vipe_results/"
        }
    }
)

print(job)
```

## Input/Output Format

### Input JSON

```json
{
    "input": {
        "video_url": "https://...",
        "output_s3": {
            "bucket": "your-bucket",
            "prefix": "vipe_results/"
        }
    }
}
```

**Alternative inputs:**
- `video_base64`: Base64-encoded video (for small files)
- `s3_input`: `{"bucket": "...", "key": "path/to/video.mp4"}`

### Output JSON

```json
{
    "status": "success",
    "num_frames": 241,
    "elapsed_seconds": 180,
    "poses_url": "https://presigned-s3-url.../poses.npz",
    "intrinsics_url": "https://presigned-s3-url.../intrinsics.npz",
    "depth_url": "https://presigned-s3-url.../depth.zip",
    "rgb_url": "https://presigned-s3-url.../rgb.mp4"
}
```

## Output Data Format

### poses.npz
```python
import numpy as np
data = np.load("poses.npz")
poses = data["data"]  # Shape: (N, 4, 4) - camera-to-world matrices
```

### intrinsics.npz
```python
data = np.load("intrinsics.npz")
intrinsics = data["data"]  # Shape: (N, 4) - [fx, fy, cx, cy] per frame
```

### depth.zip
Contains N `.exr` files with dense depth maps (float32).

## Performance

| Video Length | Frames | Processing Time |
|--------------|--------|-----------------|
| 5 sec | 121 | ~2-3 min |
| 10 sec | 241 | ~3-5 min |
| 15 sec | 361 | ~5-7 min |
| 20 sec | 481 | ~7-10 min |

ViPE runs two SLAM passes plus depth estimation, so it's moderately compute-intensive.

## Troubleshooting

### CUDA Extension Build Fails
- Ensure Eigen3 is installed: `apt install libeigen3-dev`
- Ensure ninja is installed: `pip install ninja`

### Out of Memory
- Use a larger GPU (A100 recommended for >360 frames)
- Process shorter videos

### Slow Processing
- ViPE is CPU-bound for some operations
- Using better GPU won't help much beyond A100

## Cost Estimate

| GPU | $/hr | Per Job (5 min avg) |
|-----|------|---------------------|
| RTX 4090 | ~$0.74 | ~$0.06 |
| A100 80GB | ~$1.99 | ~$0.17 |
