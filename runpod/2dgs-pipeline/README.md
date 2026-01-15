# 2DGS Pipeline Serverless

**One endpoint** that converts Gen3C/LTX-2 video(s) directly to 3D mesh.

Supports both **single-video** and **multi-video** modes. Multi-video mode combines multiple camera angles for improved 3D reconstruction quality using frame 0 pose alignment.

```
Single Video:
Gen3C Video → [This Endpoint] → GLB/OBJ Mesh
                    │
                    ├── ViPE (pose + depth extraction)
                    ├── Format conversion
                    ├── 2DGS training
                    └── Mesh extraction

Multi-Video (NEW):
Multiple Videos → [This Endpoint] → GLB/OBJ Mesh
       │                 │
       │                 ├── ViPE on each video
       │                 ├── Depth coverage filtering
       │                 ├── Frame 0 pose alignment
       │                 ├── Merge into single COLMAP dataset
       │                 ├── 2DGS training
       │                 └── Mesh extraction
       │
       └── LTX-2 generates videos from different camera motions
           (dolly_out, dolly_left, dolly_right, orbit, etc.)
```

## Quick Start

### 1. Build the Docker Image

```bash
cd runpod/2dgs-pipeline

# Build (takes ~30 minutes due to CUDA extensions)
docker build -t 88dreams/2dgs-pipeline:v2 .

# Push to Docker Hub
docker push 88dreams/2dgs-pipeline:v2
```

**Version History:**
- `v1`: Single-video support only
- `v2`: **Multi-video support** with frame 0 alignment, depth filtering

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

**Single Video (Legacy):**
```python
import runpod

runpod.api_key = "YOUR_API_KEY"

result = runpod.run_sync(
    endpoint_id="YOUR_ENDPOINT_ID",
    input={
        "video_url": "https://your-bucket.s3.amazonaws.com/gen3c_video.mp4",
        "iterations": 5000,
        "mesh_quality": "high",
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

**Multi-Video (NEW - Better Quality):**
```python
import runpod

runpod.api_key = "YOUR_API_KEY"

result = runpod.run_sync(
    endpoint_id="YOUR_ENDPOINT_ID",
    input={
        "video_urls": [
            "https://your-bucket.s3.amazonaws.com/ltx2_dolly_out.mp4",
            "https://your-bucket.s3.amazonaws.com/ltx2_dolly_left.mp4",
            "https://your-bucket.s3.amazonaws.com/ltx2_dolly_right.mp4",
            "https://your-bucket.s3.amazonaws.com/ltx2_orbit.mp4"
        ],
        "iterations": 5000,
        "mesh_quality": "high",
        "output_format": "glb",
        "depth_threshold": 0.5,  # Filter frames with <50% depth coverage
        "output_s3": {
            "bucket": "your-bucket",
            "prefix": "meshes/"
        }
    },
    timeout=1800  # 30 minutes for multi-video
)

print(result)
# {
#     "status": "success",
#     "mesh_url": "https://...",
#     "num_frames": 385,
#     "elapsed_seconds": 720,
#     "quality_stats": {
#         "total_frames": 400,
#         "valid_frames": 385,
#         "skipped_frames": 15,
#         "avg_depth_coverage": 0.78,
#         "videos_processed": 4
#     }
# }
```

## Input Parameters

### Video Input (one required)

| Parameter | Type | Description |
|-----------|------|-------------|
| `video_urls` | array | **Preferred**: Array of 1-8 video URLs (multi-video mode) |
| `video_url` | string | Single video URL (legacy single-video mode) |
| `video_base64` | string | Base64-encoded video |
| `s3_input` | object | `{"bucket": "...", "key": "..."}` |

### Processing Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `iterations` | int | 5000 | 2DGS training iterations |
| `mesh_quality` | string | "high" | "fast", "balanced", "high", "ultra" |
| `mesh_resolution` | int | 512 | TSDF mesh resolution |
| `output_format` | string | "glb" | "glb", "obj", or "ply" |
| `depth_threshold` | float | 0.5 | Min depth coverage to keep frame (0.0-1.0) |
| `output_s3` | object | - | S3 config for output upload |

## Output

**Single-Video Mode:**
```json
{
    "status": "success",
    "mesh_url": "https://presigned-s3-url...",
    "num_frames": 241,
    "iterations": 5000,
    "elapsed_seconds": 480
}
```

**Multi-Video Mode (includes quality_stats):**
```json
{
    "status": "success",
    "mesh_url": "https://presigned-s3-url...",
    "num_frames": 385,
    "iterations": 5000,
    "elapsed_seconds": 720,
    "quality_stats": {
        "total_frames": 400,
        "valid_frames": 385,
        "skipped_frames": 15,
        "avg_depth_coverage": 0.78,
        "videos_processed": 4
    }
}
```

**On error:**
```json
{
    "status": "error",
    "error": "ErrorType: error message",
    "elapsed_seconds": 120
}
```

## Processing Time

**Single-Video Mode:**
| Video Frames | Training Iters | Approx. Time |
|--------------|----------------|--------------|
| 121 (5 sec) | 3000 | ~5 min |
| 241 (10 sec) | 5000 | ~8 min |
| 361 (15 sec) | 5000 | ~12 min |

**Multi-Video Mode:**
| Videos | Total Frames | Training Iters | Approx. Time |
|--------|--------------|----------------|--------------|
| 2 | ~200 | 5000 | ~12 min |
| 4 | ~400 | 5000 | ~18 min |
| 8 | ~800 | 7000 | ~30 min |

**Cost estimate**:
- Single-video: ~$0.15-0.25 per job on RTX 4090
- Multi-video: ~$0.30-0.60 per job (more ViPE passes + more frames)

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
├── handler.py              # Main serverless handler (single + multi-video)
├── multi_video_merge.py    # Multi-video pose alignment & merging utilities
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
- Increase endpoint execution timeout to 900-1200 seconds (single) or 1800+ seconds (multi-video)
- Use fewer training iterations

### Multi-video: Too many frames skipped
- Lower `depth_threshold` from 0.5 to 0.3 to keep more frames
- Check video quality - blurry/motion blur frames have poor depth

### Multi-video: Mesh artifacts at seams
- Ensure all videos start from the same image (frame 0 alignment depends on this)
- Try using videos with overlapping camera coverage (e.g., dolly_out + orbit)

### Multi-video: Out of VRAM
- Reduce number of videos (4 max recommended on RTX 4090)
- Use A100 80GB for 8+ videos
