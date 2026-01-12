# 2DGS Pipeline Serverless

**One endpoint** that converts Gen3C video directly to 3D mesh.

```
Gen3C Video → [This Endpoint] → GLB/OBJ Mesh
                    │
                    ├── ViPE (pose + depth extraction)
                    ├── Format conversion
                    ├── 2DGS training
                    └── Mesh extraction
```

## Quick Start

### 1. Build the Docker Image

```bash
cd runpod/2dgs-pipeline

# Build (takes ~30 minutes due to CUDA extensions)
docker build -t 88dreams/2dgs-pipeline:v1 .

# Push to Docker Hub
docker push 88dreams/2dgs-pipeline:v1
```

### 2. Create RunPod Serverless Endpoint

1. Go to [RunPod Serverless](https://www.runpod.io/console/serverless)
2. Click **"+ New Endpoint"**
3. Configure:

| Setting | Value |
|---------|-------|
| Name | `2dgs-pipeline` |
| Docker Image | `88dreams/2dgs-pipeline:v1` |
| GPU Type | RTX 4090 or A100 |
| Min Workers | 0 |
| Max Workers | 2 |
| GPU Count | 1 |
| Container Disk | 30 GB |
| Idle Timeout | 60 sec |
| **Execution Timeout** | **900 sec** (15 min) |

4. Save the **Endpoint ID**

### 3. Test

```python
import runpod

runpod.api_key = "YOUR_API_KEY"

result = runpod.run_sync(
    endpoint_id="YOUR_ENDPOINT_ID",
    input={
        "video_url": "https://your-bucket.s3.amazonaws.com/gen3c_video.mp4",
        "iterations": 5000,
        "mesh_resolution": 512,
        "output_format": "glb",
        "output_s3": {
            "bucket": "your-bucket",
            "prefix": "meshes/"
        }
    },
    timeout=900  # 15 minutes
)

print(result)
# {"status": "success", "mesh_url": "https://...", "num_frames": 241, "elapsed_seconds": 480}
```

## Input Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `video_url` | string | required* | URL to Gen3C video |
| `video_base64` | string | required* | Base64-encoded video |
| `s3_input` | object | required* | `{"bucket": "...", "key": "..."}` |
| `iterations` | int | 5000 | 2DGS training iterations |
| `mesh_resolution` | int | 512 | TSDF mesh resolution |
| `output_format` | string | "glb" | "glb", "obj", or "ply" |
| `output_s3` | object | - | S3 config for output upload |

*One of `video_url`, `video_base64`, or `s3_input` is required.

## Output

```json
{
    "status": "success",
    "mesh_url": "https://presigned-s3-url...",
    "num_frames": 241,
    "iterations": 5000,
    "elapsed_seconds": 480
}
```

Or on error:
```json
{
    "status": "error",
    "error": "ErrorType: error message",
    "elapsed_seconds": 120
}
```

## Processing Time

| Video Frames | Training Iters | Approx. Time |
|--------------|----------------|--------------|
| 121 (5 sec) | 3000 | ~5 min |
| 241 (10 sec) | 5000 | ~8 min |
| 361 (15 sec) | 5000 | ~12 min |

**Cost estimate**: ~$0.15-0.25 per job on RTX 4090

## Pipeline Steps

1. **Download video** - From URL, S3, or base64
2. **ViPE inference** - Extract poses, intrinsics, depth (~2-3 min)
3. **Format conversion** - Convert to COLMAP format for 2DGS
4. **Point cloud init** - Generate initial 3D points from depth
5. **2DGS training** - Train Gaussian splatting model (~3-5 min)
6. **Mesh extraction** - TSDF fusion to mesh (~1 min)
7. **Upload** - Return presigned URL or base64

## Files

```
runpod/2dgs-pipeline/
├── Dockerfile              # Docker image definition
├── handler.py              # Main serverless handler
├── vipe_to_2dgs.py         # ViPE → COLMAP converter
├── init_points_from_depth.py  # Point cloud generator
└── README.md               # This file
```

## Troubleshooting

### Build takes too long
- CUDA extension compilation is slow (~20 min)
- Consider building on a GPU instance

### Out of memory during training
- Reduce `iterations` to 3000
- Use larger GPU (A100 recommended for long videos)

### Mesh looks bad
- Increase `iterations` to 7000-10000
- Increase `mesh_resolution` to 1024
- Ensure Gen3C video has good camera motion

### Timeout errors
- Increase endpoint execution timeout to 900-1200 seconds
- Use fewer training iterations
