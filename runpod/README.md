# RunPod Deployment for GEN3C + Hunyuan3D

This directory contains Docker configurations and scripts for deploying GEN3C inference to RunPod's GPU cloud.

## Directory Structure

```
runpod/
├── gen3c/
│   ├── Dockerfile              # GEN3C container image
│   ├── requirements-gen3c.txt  # Python dependencies
│   ├── server.py               # REST API server (for GPU Pods)
│   ├── handler.py              # Serverless handler (for Endpoints)
│   └── start.sh                # Container startup script
├── hunyuan3d/                  # (Future: Hunyuan3D container)
├── shared/
│   └── runpod_client.py        # Python client for Gradio integration
├── scripts/
│   └── download_checkpoints.sh # Download models to network volume
└── README.md
```

## Quick Start

### 1. Create RunPod Network Volume

1. Go to https://runpod.io → Storage → Network Volumes
2. Create a 100GB volume named `gen3c-checkpoints`
3. Note the volume ID

### 2. Download Checkpoints

Create a temporary pod with the network volume mounted, then run:

```bash
bash /workspace/download_checkpoints.sh
```

This downloads ~75GB of model checkpoints.

### 3. Build Docker Image

```bash
cd runpod/gen3c
docker build -t your-dockerhub-username/gen3c-runpod:latest .
docker push your-dockerhub-username/gen3c-runpod:latest
```

### 4. Create Pod Template

In RunPod:
1. Go to Pods → Templates → New Template
2. Set image: `your-dockerhub-username/gen3c-runpod:latest`
3. Expose port: `8000`
4. Mount network volume at `/workspace/checkpoints`

### 5. Launch Pod

Select A100 80GB GPU and start the pod.

## API Usage

### Health Check

```bash
curl http://POD_URL:8000/health
```

### Generate Video

```bash
# Encode image
IMAGE_B64=$(base64 -w0 input.png)

# Submit job
curl -X POST http://POD_URL:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "image_base64": "'$IMAGE_B64'",
    "video_name": "my_video",
    "guidance": 1.0,
    "num_frames": 121,
    "trajectory": "left"
  }'

# Returns: {"job_id": "abc123", "status": "pending"}
```

### Check Status

```bash
curl http://POD_URL:8000/status/abc123
```

### Download Video

```bash
curl http://POD_URL:8000/download/abc123 -o output.mp4
```

## Python Client

```python
from runpod_client import create_pod_client

client = create_pod_client("http://POD_URL:8000")

result = client.generate_video(
    image_path="input.png",
    output_path="output.mp4",
    num_frames=121,
    trajectory="left"
)

if result.success:
    print(f"Video saved to: {result.video_path}")
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GEN3C_DIR` | `/workspace/GEN3C` | GEN3C repository path |
| `GEN3C_CHECKPOINT_DIR` | `/workspace/checkpoints/Gen3C-Cosmos-7B` | Model checkpoints |
| `OUTPUT_DIR` | `/workspace/outputs` | Generated video output |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU device index |

## Estimated Costs

| GPU | Price/Hour | GEN3C 121 frames |
|-----|------------|------------------|
| A100 80GB | $1.89 | ~15-30 min = $0.50-1.00 |
| A100 40GB | $1.64 | ~20-40 min = $0.55-1.10 |
| H100 80GB | $3.89 | ~10-20 min = $0.65-1.30 |

## Troubleshooting

### Checkpoint not found
Ensure network volume is mounted at `/workspace/checkpoints` and contains:
- `Gen3C-Cosmos-7B/model.pt`
- `Gen3C-Cosmos-7B/Cosmos-Tokenize1-CV8x8x8-720p/`
- `Gen3C-Cosmos-7B/google-t5/t5-11b/`

### Out of memory
- Use A100 80GB (recommended)
- Ensure offloading flags are enabled (default in server.py)

### Slow inference
- First run downloads additional models to HF cache
- Subsequent runs should be faster

