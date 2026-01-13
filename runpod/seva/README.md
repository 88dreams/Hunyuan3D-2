# SEVA (Stable Virtual Camera) - RunPod Serverless

Generates novel view videos from single images with precise camera control using Stability AI's Stable Virtual Camera (1.3B parameter diffusion model).

## Features

- **Single image input** → multi-frame video output
- **14+ preset trajectories** (orbit, pan, tilt, spiral, dolly zoom, etc.)
- **Custom trajectories** via camera-to-world (C2W) matrices
- **3D-consistent** output without NeRF distillation
- **Loop closure** for seamless videos
- **Up to 1,000 frames** per generation

## Prerequisites

1. **HuggingFace Access**
   - Request access at: https://huggingface.co/stabilityai/stable-virtual-camera
   - Generate HF token at: https://huggingface.co/settings/tokens

2. **AWS S3 Credentials** (for file transfer)
   - `AWS_ACCESS_KEY_ID`
   - `AWS_SECRET_ACCESS_KEY`

## Build & Deploy

### Build Docker Image

```bash
cd /home/arkrunr02/Hunyuan3D-2-Fork/runpod/seva
docker build -t 88dreams/seva-runpod:v1 .
docker push 88dreams/seva-runpod:v1
```

### Create RunPod Endpoint

1. Go to RunPod Console → Serverless → New Endpoint
2. **Name**: `seva-serverless`
3. **Docker Image**: `88dreams/seva-runpod:v1`
4. **GPU**: RTX A5000 (24GB) or better
5. **Environment Variables**:
   - `HF_TOKEN`: Your HuggingFace token
   - `AWS_ACCESS_KEY_ID`: Your AWS key
   - `AWS_SECRET_ACCESS_KEY`: Your AWS secret
   - `S3_BUCKET`: `arkrunr`
   - `S3_REGION`: `us-west-1`
6. **Network Volume**: 10GB (for model weights)
7. **Active Workers**: 0 (scale to zero)
8. **Max Workers**: 2

## API Usage

### Submit Job

```json
POST /run
{
  "input": {
    "image_url": "https://arkrunr.s3.us-west-1.amazonaws.com/inputs/image.png",
    "trajectory": "orbit",
    "duration": 5.0,
    "fps": 24,
    "output_name": "my_video"
  }
}
```

### Response

```json
{
  "status": "success",
  "video_url": "https://arkrunr.s3.us-west-1.amazonaws.com/seva/my_video.mp4",
  "video_path": "/runpod-volume/outputs/seva/my_video.mp4",
  "duration": 5.0,
  "fps": 24,
  "frame_count": 120,
  "trajectory": "orbit"
}
```

## Trajectories

| Trajectory | Description |
|------------|-------------|
| `orbit` | 360° rotation around subject |
| `pan` | Horizontal camera movement |
| `tilt` | Vertical camera angle change |
| `spiral` | Spiral path around subject |
| `zoom-out` | Camera moves backward |
| `dolly-zoom-out` | Vertigo/Hitchcock effect |
| `arc` | Curved path |
| `crane` | Vertical + horizontal movement |
| `left` | Move camera left |
| `right` | Move camera right |
| `up` | Move camera up |
| `down` | Move camera down |
| `custom` | User-defined C2W matrices |

## Custom Trajectories

For `custom` trajectory, provide a list of 4×4 camera-to-world matrices:

```json
{
  "input": {
    "image_url": "...",
    "trajectory": "custom",
    "custom_poses": [
      [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]],
      [[1,0,0,0], [0,1,0,0.1], [0,0,1,0], [0,0,0,1]],
      ...
    ],
    "fps": 24
  }
}
```

## Python Client

```python
from runpod.runpod_client import SEVAServerlessClient

client = SEVAServerlessClient(
    endpoint_id="your_endpoint_id",
    api_key="your_runpod_api_key"
)

# Synchronous generation
result = client.generate_sync(
    image_path="input.png",
    output_dir="./outputs/seva",
    trajectory="orbit",
    duration=5.0
)

if result.success:
    print(f"Video: {result.video_path}")
else:
    print(f"Error: {result.error}")
```

## Generator Module

```python
from generators.seva import run_seva_runpod, create_custom_trajectory

# Using preset trajectory
result = run_seva_runpod(
    image_path="input.png",
    output_dir="./outputs/seva",
    trajectory="pan",
    duration=5.0,
    api_key="your_key",
    endpoint_id="your_endpoint"
)

# Custom trajectory: move up 0.5 units, tilt down 15 degrees
poses = create_custom_trajectory(
    start_position=(0, 0, 0),
    end_position=(0, 0.5, 0),
    start_rotation=(0, 0, 0),
    end_rotation=(-15, 0, 0),
    num_frames=120
)

result = run_seva_runpod(
    image_path="input.png",
    trajectory="custom",
    custom_poses=poses,
    ...
)
```

## Integration with 2DGS Pipeline

SEVA is designed to work with the 2DGS pipeline for video → 3D mesh conversion:

```
Image → SEVA (camera video) → 2DGS Pipeline → 3D Mesh (GLB)
```

This provides better camera control than Gen3C for 3D reconstruction.

## Troubleshooting

### Model Download Fails
- Ensure `HF_TOKEN` is set correctly
- Verify you have access to the model on HuggingFace

### Out of Memory
- Use A6000 (48GB) instead of A5000 (24GB)
- Reduce frame count or duration

### Slow Cold Start
- First request downloads model weights (~2-3GB)
- Subsequent requests use cached weights

## Files

| File | Purpose |
|------|---------|
| `Dockerfile` | Docker image definition |
| `handler_seva.py` | Serverless handler |
| `start_seva.sh` | Container startup script |
| `README.md` | This file |

## License

SEVA model is under Stability AI's **Non-Commercial License**. 
This integration is for research and non-commercial use only.

## References

- [SEVA GitHub](https://github.com/Stability-AI/stable-virtual-camera)
- [SEVA HuggingFace](https://huggingface.co/stabilityai/stable-virtual-camera)
- [Stability AI Announcement](https://stability.ai/news/introducing-stable-virtual-camera-multi-view-video-generation-with-3d-camera-control)
