# 2DGS Pipeline Serverless Deployment

**One endpoint**: Gen3C Video → 3D Mesh

---

## What's Ready

```
runpod/2dgs-pipeline/
├── Dockerfile              ← Docker image
├── handler.py              ← Serverless handler
├── vipe_to_2dgs.py         ← Format converter
├── init_points_from_depth.py
└── README.md
```

---

## Deployment Steps

### Step 1: Build Docker Image

You can build this locally or on a RunPod GPU pod.

**Option A: Build Locally** (if you have Docker + NVIDIA GPU)
    
```bash
cd ~/Hunyuan3D-2-Fork/runpod/2dgs-pipeline

# Build (~30 min for CUDA extensions)
docker build -t 88dreams/2dgs-pipeline:v1 .

# Push to Docker Hub
docker login
docker push 88dreams/2dgs-pipeline:v1
```

**Option B: Build on RunPod Pod**

1. Start a GPU pod with the PyTorch template
2. SSH in and run:

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Clone repo
cd /workspace
git clone https://github.com/88dreams/Hunyuan3D-2-Fork.git
cd Hunyuan3D-2-Fork/runpod/2dgs-pipeline

# Build
docker build -t 88dreams/2dgs-pipeline:v1 .

# Push
docker login
docker push 88dreams/2dgs-pipeline:v1
```

---

### Step 2: Create RunPod Serverless Endpoint

1. Go to: https://www.runpod.io/console/serverless
2. Click **"+ New Endpoint"**
3. Configure:

| Field | Value |
|-------|-------|
| **Name** | `2dgs-pipeline` |
| **Docker Image** | `88dreams/2dgs-pipeline:v1` |
| **GPU Type** | RTX 4090 (or A100) |
| **Min Workers** | 0 |
| **Max Workers** | 2 |
| **GPU Count** | 1 |
| **Container Disk** | 30 GB |
| **Idle Timeout** | 60 sec |
| **Execution Timeout** | 900 sec (15 min) |

4. Click **Create**
5. Copy the **Endpoint ID**: `_________________`

---

### Step 3: Test the Endpoint

```python
import runpod

runpod.api_key = "YOUR_RUNPOD_API_KEY"

result = runpod.run_sync(
    endpoint_id="YOUR_ENDPOINT_ID",
    input={
        "video_url": "https://your-s3-bucket.s3.amazonaws.com/gen3c_video.mp4",
        "iterations": 5000,
        "output_format": "glb"
    },
    timeout=900
)

print(result)
```

---

### Step 4: Integrate with Gradio (Optional)

Update the 2DGS tab (`ui/tabs/create_tab.py`) to call this endpoint instead of separate ViPE/2DGS calls.

---

## API Reference

### Input

```json
{
    "video_url": "https://...",
    "iterations": 5000,
    "mesh_resolution": 512,
    "output_format": "glb",
    "output_s3": {
        "bucket": "your-bucket",
        "prefix": "meshes/"
    }
}
```

### Output

```json
{
    "status": "success",
    "mesh_url": "https://presigned-url...",
    "num_frames": 241,
    "elapsed_seconds": 480
}
```

---

## Expected Timing & Cost

| Video Length | Processing Time | Cost (4090) |
|--------------|-----------------|-------------|
| 5 sec (121 frames) | ~5 min | ~$0.06 |
| 10 sec (241 frames) | ~8 min | ~$0.10 |
| 15 sec (361 frames) | ~12 min | ~$0.15 |

---

## Troubleshooting

**Docker build fails**
- Ensure you're building on a machine with NVIDIA GPU
- CUDA 12.4 required

**Endpoint times out**
- Increase execution timeout to 900+ seconds
- Reduce `iterations` to 3000 for faster results

**Mesh quality is poor**
- Increase `iterations` to 7000-10000
- Ensure Gen3C video has good parallax/camera motion
