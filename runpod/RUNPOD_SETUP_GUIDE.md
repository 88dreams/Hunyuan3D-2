# GEN3C RunPod Deployment Guide

This guide documents the complete setup for running GEN3C on RunPod with NVIDIA A100 GPUs.

**Current Docker Image:** `88dreams/gen3c-runpod:v9`

---

## Table of Contents

1. [Quick Start - Pod Mode](#quick-start)
2. [Serverless Deployment](#serverless-deployment)
3. [Updating Docker Image](#updating-docker-image)
4. [Useful Commands Reference](#useful-commands-reference)
5. [Troubleshooting](#troubleshooting)
6. [Technical Details](#technical-details)

---

## Quick Start

### 1. Create Network Volume (One-time setup)

1. Go to RunPod → Storage → Create Network Volume
2. Settings:
   - Region: Choose one with A100 availability (e.g., CA-MTL-1)
   - Size: 150GB minimum
   - Name: `gen3c-checkpoints`

### 2. Download Checkpoints to Network Volume (One-time setup)

Start a cheap pod (e.g., RTX 2000 Ada) with the network volume mounted at `/workspace/checkpoints`, then run:

```bash
# Install huggingface CLI
pip install "huggingface-hub[cli]"

# Login (required for gated models)
python -c "from huggingface_hub import login; login()"

# Download checkpoints
huggingface-cli download nvidia/GEN3C \
    --local-dir /workspace/checkpoints/Gen3C-Cosmos-7B \
    --local-dir-use-symlinks False

huggingface-cli download nvidia/Cosmos-Tokenize1-CV8x8x8-720p \
    --local-dir /workspace/checkpoints/Gen3C-Cosmos-7B/Cosmos-Tokenize1-CV8x8x8-720p \
    --local-dir-use-symlinks False

huggingface-cli download google-t5/t5-11b \
    --local-dir /workspace/checkpoints/Gen3C-Cosmos-7B/google-t5/t5-11b \
    --include "pytorch_model.bin" "config.json" "spiece.model" "tokenizer.json" \
    --local-dir-use-symlinks False
```

Terminate the cheap pod after downloads complete.

### 3. Create Pod Template

1. Go to RunPod → Templates → New Template
2. Settings:
   - Name: `GEN3C-A100`
   - Container Image: `88dreams/gen3c-runpod:v2`
   - Container Disk: 30GB
   - Volume Mount Path: `/workspace/checkpoints`
   - Exposed HTTP Ports: `8000`
   - Docker Command: (leave empty - uses default)

### 4. Start a Pod

1. Go to RunPod → Pods → Deploy
2. Select your `GEN3C-A100` template
3. Choose GPU: A100 80GB recommended
4. Attach your `gen3c-checkpoints` network volume
5. Deploy

### 5. Use the API

Once the pod is running, access the API at:
```
https://{POD_ID}-8000.proxy.runpod.net/
```

#### Health Check
```bash
curl https://{POD_ID}-8000.proxy.runpod.net/health
```

#### Generate Video
```python
import base64
import requests
import time

# Your pod's URL
API_URL = "https://{POD_ID}-8000.proxy.runpod.net"

# Read and encode image
with open("input_image.png", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

# Submit job
response = requests.post(f"{API_URL}/generate", json={
    "image_base64": image_b64,
    "video_name": "my_video",
    "num_frames": 121,        # Options: 121, 241, 361, 481 (N*120+1)
    "trajectory": "left",     # Options: left, right, zoom_in, zoom_out, orbit
    "guidance": 1.0,
    "foreground_masking": True
})
job_id = response.json()["job_id"]
print(f"Job submitted: {job_id}")

# Poll for completion
while True:
    status = requests.get(f"{API_URL}/status/{job_id}").json()
    print(f"Status: {status['status']}")
    
    if status["status"] == "completed":
        # Download video
        video_data = base64.b64decode(status["video_base64"])
        with open("output.mp4", "wb") as f:
            f.write(video_data)
        print("Video saved to output.mp4")
        break
    elif status["status"] == "failed":
        print(f"Error: {status['error']}")
        break
    
    time.sleep(30)  # Check every 30 seconds
```

---

## Technical Details

### Docker Image Contents

The `88dreams/gen3c-runpod:v2` image includes:

- Base: `nvcr.io/nvidia/pytorch:24.10-py3`
- Conda environment: `cosmos-predict1` (Python 3.10)
- GEN3C repository and all dependencies
- Apex with CUDA extensions
- MoGe depth estimation
- Transformer Engine
- FastAPI server for REST API

### Key Fixes Included

1. **Checkpoint Path**: Set to `/workspace/checkpoints` (parent directory). GEN3C internally appends `Gen3C-Cosmos-7B/model.pt`.

2. **Symlinks**: The startup script automatically creates symlinks for:
   - `/workspace/checkpoints/Cosmos-Tokenize1-CV8x8x8-720p` → `Gen3C-Cosmos-7B/Cosmos-Tokenize1-CV8x8x8-720p`
   - `/workspace/checkpoints/google-t5` → `Gen3C-Cosmos-7B/google-t5`

3. **huggingface-hub Version**: Downgraded to `<1.0` for transformers compatibility.

4. **Output Path**: Server checks both `/workspace/GEN3C/outputs/` and `/workspace/GEN3C/videos/` for generated videos.

5. **GPU Detection**: Uses `torch.cuda.is_available()` instead of checking `/dev/nvidia0`.

6. **Non-blocking API**: Job processing runs in a separate thread so status checks work during generation.

### Expected Performance

| GPU | 121 Frames | 241 Frames |
|-----|------------|------------|
| A100 80GB | ~15-20 min | ~30-40 min |
| A100 40GB | ~20-25 min | ~40-50 min |

### Network Volume Structure

After checkpoint download, your network volume should contain:
```
/workspace/checkpoints/
├── Gen3C-Cosmos-7B/
│   ├── model.pt                    # 27GB - Main model
│   ├── config.json
│   ├── Cosmos-Tokenize1-CV8x8x8-720p/
│   │   ├── mean_std.pt
│   │   ├── encoder.jit
│   │   ├── decoder.jit
│   │   └── ...
│   └── google-t5/
│       └── t5-11b/
│           ├── pytorch_model.bin   # 42GB - T5 weights
│           ├── config.json
│           └── ...
├── Cosmos-Tokenize1-CV8x8x8-720p -> Gen3C-Cosmos-7B/Cosmos-Tokenize1-CV8x8x8-720p  (symlink)
└── google-t5 -> Gen3C-Cosmos-7B/google-t5  (symlink)
```

### Common Issues

#### "Checkpoint not found" errors
- **Pod mode:** Ensure network volume is mounted at `/workspace/checkpoints`
- **Serverless mode:** Volume is at `/runpod-volume`, script creates symlinks automatically
- Check symlinks exist: `ls -la /workspace/checkpoints/`
- Verify `model.pt` exists: `ls -la /workspace/checkpoints/Gen3C-Cosmos-7B/model.pt`

#### Serverless starts in Pod mode (wrong handler)
- Check if `RUNPOD_ENDPOINT_ID` is set: `echo $RUNPOD_ENDPOINT_ID`
- If using old Docker image, update to `v9` or later
- Manual fix: `pkill -f server.py && python /workspace/handler.py &`

#### "huggingface-hub version" errors
- The startup script auto-fixes this, but you can manually run:
  ```bash
  pip install "huggingface-hub>=0.26.0,<1.0"
  ```

#### Server not responding
- Check if a job is running: `ps aux | grep gen3c`
- The server may be blocked during inference (status checks should still work)

#### GPU not detected
- Verify with `nvidia-smi`
- Check PyTorch: `python -c "import torch; print(torch.cuda.is_available())"`

#### CUDA initialization failed (H100 cold start)
- This is a known issue with H100 GPUs on serverless cold starts
- **Solution:** Use A100 instead of H100
- The `start.sh` includes a retry loop, but it may not help if CUDA fails at system level

#### Docker image not updating
- RunPod may cache images; try a new tag (e.g., `v9` → `v10`)
- Terminate ALL workers after updating endpoint
- Verify image version: `grep "Detecting network volume" /workspace/start.sh`

#### Job stuck / no progress
- Check worker web terminal for errors
- Verify handler is running: `ps aux | grep handler`
- Check GPU is being used: `nvidia-smi`

#### Network volume not found in serverless
- Serverless mounts at `/runpod-volume`, not `/workspace/checkpoints`
- Check: `ls -la /runpod-volume/`
- The `start.sh` (v9+) handles this automatically

---

## Serverless Deployment

Serverless is ideal for production use - you only pay for actual compute time.

### 1. Create Serverless Endpoint

1. Go to RunPod → Serverless → New Endpoint
2. Settings:
   - **Name:** `GEN3C-Serverless`
   - **Container Image:** `88dreams/gen3c-runpod:v9`
   - **GPU Type:** A100 80GB (recommended) - **Avoid H100 due to cold start CUDA issues**
   - **Min Workers:** 0 (scale to zero when idle)
   - **Max Workers:** 1-3 (based on expected load)
   - **Idle Timeout:** 30 seconds (workers spin down when idle)
   - **Network Volume:** Attach your `gen3c-checkpoints` volume
   - **Volume Mount Path:** (RunPod serverless mounts at `/runpod-volume` automatically)
   - **Min CUDA Version:** 12.0

3. Note your **Endpoint ID** (shown after creation)

### Important Notes for Serverless

- **Network Volume Mount:** Serverless mounts volumes at `/runpod-volume`, NOT `/workspace/checkpoints`
- The `start.sh` script automatically detects this and creates symlinks
- **GPU Recommendation:** Use A100 80GB. H100 has CUDA initialization issues on cold starts
- **Cold Start:** First job after idle takes 30-120s to spin up a worker

### 2. Get Your API Key

1. Go to RunPod → Settings → API Keys
2. Create a new key or copy existing one (starts with `rp_`)

### 3. Use Serverless API

The serverless API uses RunPod's managed endpoints:

```python
import base64
import requests
import time

# Your credentials
ENDPOINT_ID = "your-endpoint-id"
API_KEY = "rp_your_api_key"

# RunPod API base
API_BASE = f"https://api.runpod.ai/v2/{ENDPOINT_ID}"
headers = {"Authorization": f"Bearer {API_KEY}"}

# Read and encode image
with open("input_image.png", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

# Submit job
response = requests.post(f"{API_BASE}/run", headers=headers, json={
    "input": {
        "image_base64": image_b64,
        "video_name": "my_video",
        "num_frames": 121,
        "trajectory": "left",
        "guidance": 1.0,
        "foreground_masking": True,
        "return_base64": True
    }
})
job_id = response.json()["id"]
print(f"Job submitted: {job_id}")

# Poll for completion
while True:
    status = requests.get(f"{API_BASE}/status/{job_id}", headers=headers).json()
    print(f"Status: {status['status']}")
    
    if status["status"] == "COMPLETED":
        output = status["output"]
        if output.get("status") == "success":
            video_data = base64.b64decode(output["video_base64"])
            with open("output.mp4", "wb") as f:
                f.write(video_data)
            print("Video saved!")
        else:
            print(f"Error: {output.get('message')}")
        break
    elif status["status"] == "FAILED":
        print(f"Error: {status.get('error')}")
        break
    
    time.sleep(30)
```

### 4. Serverless vs Pod Comparison

| Feature | GPU Pod | Serverless |
|---------|---------|------------|
| **Billing** | Per hour (running) | Per second (processing) |
| **Cold Start** | None | 30-120s if no warm workers |
| **Scaling** | Manual | Automatic |
| **Cost (idle)** | ~$2/hr | $0 |
| **Cost (active)** | ~$2/hr | ~$2/hr + overhead |
| **API** | Custom FastAPI | RunPod managed |
| **Best for** | Testing, frequent use | Production, sporadic use |

### 5. Gradio UI Integration

In the Gradio UI, select "RunPod Serverless" mode and enter:
- **Endpoint ID**: Your serverless endpoint ID
- **API Key**: Your RunPod API key

The UI will handle job submission and polling automatically.

---

## Cost Optimization

### GPU Pod vs Serverless

For sporadic usage (few times per day), consider:

| Usage Pattern | GPU Pod Cost | Serverless Cost |
|--------------|--------------|-----------------|
| 3 jobs/day (15 min each) | $45/day (always on) | ~$3.78/day |
| 1 job/day | $45/day | ~$1.26/day |
| No usage | $45/day | $0/day |

**Recommendation**: Use GPU Pods for testing, switch to Serverless for production.

### Stopping vs Terminating

- **Stop**: Not available for GPU pods
- **Terminate**: Deletes the pod, image must re-download on next start (~5-10 min)
- **Network Volume**: Persists across pod terminations, ~$0.07/GB/month

---

## API Reference

### `GET /health`
Returns server health and checkpoint status.

### `POST /generate`
Submit a video generation job.

**Request Body:**
```json
{
    "image_base64": "base64_encoded_image",
    "video_name": "output_name",
    "num_frames": 121,
    "trajectory": "left",
    "guidance": 1.0,
    "foreground_masking": true,
    "seed": null
}
```

**Response:**
```json
{
    "job_id": "abc12345",
    "status": "pending",
    "message": "Job submitted successfully"
}
```

### `GET /status/{job_id}`
Get job status and results.

**Response (completed):**
```json
{
    "job_id": "abc12345",
    "status": "completed",
    "video_base64": "base64_encoded_video",
    "video_url": "/workspace/outputs/abc12345_output.mp4"
}
```

### `GET /download/{job_id}`
Download the generated video file directly.

### `GET /jobs`
List all jobs and their statuses.

---

## Updating Docker Image

When you need to update the Docker image (e.g., after fixing bugs in `start.sh`, `handler.py`, or `server.py`):

### Step-by-Step Update Process

```bash
# 1. Navigate to the runpod directory
cd ~/Hunyuan3D-2-Fork/runpod/gen3c

# 2. Build the new image (increment version number)
docker build -f Dockerfile.gen3c -t 88dreams/gen3c-runpod:v9 .

# For a completely fresh build (slower, but ensures no cached layers):
docker build --no-cache -f Dockerfile.gen3c -t 88dreams/gen3c-runpod:v9 .

# 3. Push to Docker Hub
docker push 88dreams/gen3c-runpod:v9

# 4. Update RunPod endpoint to use the new image tag
```

### After Pushing

1. **For Pods:** Terminate and redeploy with new image
2. **For Serverless:**
   - Go to your endpoint settings
   - Update the Container Image to the new tag (e.g., `v9`)
   - Save
   - **Terminate all existing workers** (they won't auto-update)
   - New workers will use the updated image

### Fixing Typos in Docker Tag

```bash
# If you accidentally created the wrong tag:
docker tag 888dreams/gen3c-runpod:v7 88dreams/gen3c-runpod:v7
docker push 88dreams/gen3c-runpod:v7

# Remove the wrong tag locally:
docker rmi 888dreams/gen3c-runpod:v7
```

### Build Time Expectations

- **Full build (with --no-cache):** 30-45 minutes (Apex CUDA compilation is slow)
- **Incremental build:** 1-2 minutes (if only COPY commands changed)

---

## Useful Commands Reference

### RunPod Worker Commands (Web Terminal)

#### System Status

```bash
# GPU status and memory usage
nvidia-smi

# Continuous GPU monitoring (updates every 1 second)
watch -n 1 nvidia-smi

# CPU and memory usage
top -b -n 1 | head -20

# Disk usage
df -h
```

#### Process Management

```bash
# Check running processes
ps aux | grep -E "server.py|handler.py|gen3c|python"

# Kill a specific process
pkill -f server.py
pkill -f handler.py

# Kill all Python processes (use carefully!)
pkill -f python
```

#### Server/Handler Control

```bash
# Start the Pod server (HTTP API)
python /workspace/server.py &

# Start the Serverless handler
python /workspace/handler.py &

# Check server health (Pod mode only)
curl -s http://localhost:8000/health

# Check what's listening on port 8000
netstat -tlnp 2>/dev/null | grep 8000
```

#### Checkpoint Verification

```bash
# Check if checkpoints exist
ls -la /workspace/checkpoints/Gen3C-Cosmos-7B/model.pt
ls -la /workspace/checkpoints/Cosmos-Tokenize1-CV8x8x8-720p/mean_std.pt
ls -la /workspace/checkpoints/google-t5/t5-11b/

# Check symlinks
ls -la /workspace/checkpoints/

# Check network volume (serverless)
ls -la /runpod-volume/
ls -la /runpod-volume/checkpoints/
```

#### Docker Image Verification

```bash
# Check which start.sh version is running
grep -n "Detecting network volume" /workspace/start.sh || echo "OLD IMAGE - missing new code"

# Check file modification time
ls -la /workspace/start.sh

# View start.sh contents
head -100 /workspace/start.sh
```

#### Environment Variables

```bash
# Check serverless detection variables
echo "RUNPOD_ENDPOINT_ID: $RUNPOD_ENDPOINT_ID"
echo "RUNPOD_SERVERLESS: $RUNPOD_SERVERLESS"
echo "RUNPOD_POD_ID: $RUNPOD_POD_ID"

# Check checkpoint directory
echo "GEN3C_CHECKPOINT_DIR: $GEN3C_CHECKPOINT_DIR"

# All RunPod variables
env | grep -i runpod

# All volume-related variables
env | grep -iE "volume|checkpoint|mount"
```

#### Network Volume Discovery

```bash
# Find where volumes are mounted
df -h | grep -E "runpod|volume|workspace|nfs"

# Search for model files
find / -maxdepth 5 -name "model.pt" -type f 2>/dev/null

# Search for Gen3C directory
find / -maxdepth 5 -type d -name "Gen3C-Cosmos-7B" 2>/dev/null
```

#### Log Viewing

```bash
# View recent server logs
tail -50 /tmp/gen3c/*.log 2>/dev/null

# Follow logs in real-time
tail -f /tmp/gen3c/*.log 2>/dev/null
```

### Local Machine Commands (searidge02)

#### Docker Build & Push

```bash
# Navigate to runpod directory
cd ~/Hunyuan3D-2-Fork/runpod/gen3c

# Build Docker image
docker build -f Dockerfile.gen3c -t 88dreams/gen3c-runpod:v9 .

# Build without cache (fresh build)
docker build --no-cache -f Dockerfile.gen3c -t 88dreams/gen3c-runpod:v9 .

# Push to Docker Hub
docker push 88dreams/gen3c-runpod:v9

# List local images
docker images | grep gen3c

# Remove old images
docker rmi 88dreams/gen3c-runpod:v8
```

#### Monitoring Docker Build Progress

```bash
# Check if build is still running
ps aux | grep -E "docker|pip|nvcc|gcc"

# Check CPU usage during build
top -b -n 1 | head -20
```

#### Gradio UI

```bash
# Start Gradio UI
cd /srv/searidge_share/projects/Hunyuan3D-2-Fork
conda activate gen3c-rocm
python 2d3d.py

# Or use the wrapper script
./run2d3d.sh
```

#### Copy Updated Files to NFS Share

```bash
# Copy updated 2d3d.py
cp ~/Hunyuan3D-2-Fork/2d3d.py /srv/searidge_share/projects/Hunyuan3D-2-Fork/

# Copy entire runpod directory
cp -r ~/Hunyuan3D-2-Fork/runpod /srv/searidge_share/projects/Hunyuan3D-2-Fork/
```

---

## Troubleshooting

### Common Issues

