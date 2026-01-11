# 2DGS Serverless Deployment Guide

This folder contains everything needed to deploy a **2D Gaussian Splatting** serverless endpoint on RunPod.

## Overview

The 2DGS endpoint:
1. Receives ViPE output (poses, intrinsics, depth, RGB)
2. Converts to 2DGS training format
3. Trains a 2D Gaussian Splatting model
4. Extracts a textured mesh
5. Returns GLB/OBJ/PLY mesh file

## Files

| File | Description |
|------|-------------|
| `Dockerfile` | Docker image definition |
| `handler_2dgs.py` | RunPod serverless handler |
| `vipe_to_2dgs.py` | ViPE → COLMAP format converter |
| `init_points_from_depth.py` | Depth → initial 3D points |

## Deployment Steps

### Step 1: Build Docker Image

```bash
cd /path/to/Hunyuan3D-2-Fork/runpod/2dgs

# Build the image (takes ~15-20 minutes)
docker build -t YOUR_DOCKERHUB_USERNAME/2dgs-serverless:v1 .

# Test locally (optional)
docker run --gpus all -it YOUR_DOCKERHUB_USERNAME/2dgs-serverless:v1 bash

# Push to Docker Hub
docker push YOUR_DOCKERHUB_USERNAME/2dgs-serverless:v1
```

### Step 2: Create RunPod Serverless Endpoint

1. Go to [RunPod Serverless](https://www.runpod.io/console/serverless)
2. Click **"+ New Endpoint"**
3. Configure:

| Setting | Value |
|---------|-------|
| **Name** | `2dgs-serverless` |
| **Docker Image** | `YOUR_DOCKERHUB_USERNAME/2dgs-serverless:v1` |
| **GPU Type** | RTX 4090 or A100 (recommended) |
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
import time

runpod.api_key = "YOUR_API_KEY"

# Submit job
job = runpod.run_sync(
    endpoint_id="YOUR_ENDPOINT_ID",
    input={
        "vipe_results_url": "https://your-s3-bucket.s3.amazonaws.com/vipe_results.zip",
        "iterations": 5000,
        "mesh_resolution": 512,
        "output_format": "glb"
    }
)

print(job)
```

## Input/Output Format

### Input JSON

```json
{
    "input": {
        "vipe_results_url": "https://...",
        "iterations": 5000,
        "mesh_resolution": 512,
        "output_format": "glb",
        "output_s3": {
            "bucket": "your-bucket",
            "prefix": "results/"
        }
    }
}
```

### Output JSON

```json
{
    "status": "success",
    "mesh_url": "https://presigned-s3-url...",
    "stats": {
        "num_frames": 241,
        "initial_points": 100000,
        "final_loss": 0.029,
        "num_points": 156032,
        "training_seconds": 150,
        "vertices": 830682,
        "faces": 1661624
    }
}
```

## Performance

| Input | Time | Output |
|-------|------|--------|
| 241 frames | ~5-10 min | 42 MB GLB |

Training at 5000 iterations on RTX 4090 takes ~2.5 minutes.
Mesh extraction adds ~3-5 minutes depending on resolution.

## Troubleshooting

### CUDA Out of Memory
- Reduce `mesh_resolution` to 256
- Reduce `iterations` to 3000

### Mesh Extraction Fails
- The Dockerfile includes the trimesh 4.x compatibility patch
- If still failing, check CUDA compute capability matches GPU

### Slow Cold Start
- First request takes longer (~60s) as container initializes
- Subsequent requests are faster (~5-10 min total)

## Cost Estimate

| GPU | $/hr | Per Job (10 min) |
|-----|------|------------------|
| RTX 4090 | ~$0.74 | ~$0.12 |
| A100 80GB | ~$1.99 | ~$0.33 |

With 0 min workers, you only pay when processing.
