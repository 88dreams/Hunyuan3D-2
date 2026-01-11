# ARKRUNR WORLDS - Conversation Handoff Document

This document summarizes the work completed and provides context for new conversation instances to continue development.

---

## Project Overview

**ARKRUNR WORLDS** is a multi-model 3D generation system built as a fork of Hunyuan3D-2. It provides a unified Gradio UI for multiple AI models that generate 3D content from images and video.

### Supported Models

| Model | Source | Output | Status |
|-------|--------|--------|--------|
| **SHARP** | Apple | 3DGS PLY + Video | ✅ Working |
| **Gen3C** | NVIDIA | Video (MP4) | ✅ Working |
| **Lyra** | NVIDIA | 3D/4D Gaussian Splatting | ✅ Working |
| **TRELLIS.2** | Microsoft | 3D (GLB) | ✅ Working |
| **Hunyuan3D** | Tencent | 3D Mesh (GLB) | ✅ Working |
| **Stage Pipeline** | ViPE + 2DGS | Video → Mesh (GLB) | 🔄 Testing (v15) |

### Architecture

- **Local**: Gradio UI (`app_sidebar.py`) running on user's machine
- **Remote**: RunPod Serverless endpoints for GPU inference
- **Storage**: AWS S3 for large file transfer, RunPod network volume for persistence

---

## Current Work In Progress

### Stage Pipeline: ViPE + 2DGS (January 11, 2026)

**Goal**: Convert Gen3C video output into 3D mesh using video pose estimation and 2D Gaussian Splatting.

**Pipeline Flow**:
1. **ViPE** (NVIDIA) extracts camera poses, depth maps, and intrinsics from video
2. **Converter** transforms ViPE output to COLMAP format for 2DGS
3. **Point Cloud Generator** creates initial 3D points from depth maps
4. **2DGS Training** produces Gaussian splats from video frames
5. **Mesh Extraction** via marching cubes

**Current Status**: Testing v15 - waiting for RunPod endpoint update

**Issues Fixed (v1-v15)**:
| Version | Issue | Fix |
|---------|-------|-----|
| v1-v6 | Conda TOS acceptance | Added `conda tos accept` commands |
| v7-v8 | ViPE ABI mismatch | Set `_GLIBCXX_USE_CXX11_ABI=0` flag |
| v9-v10 | ViPE module invocation | Changed to `python -c "from vipe.cli.main import main; main([...])"` |
| v11-v12 | ViPE output paths | Fixed to use `pose/input.npz` structure |
| v12 | Converter signature | Fixed `vipe_to_2dgs.py` to accept `vipe_dir` |
| v13 | PLY format | Added normals (`nx`, `ny`, `nz`) for 2DGS compatibility |
| v14 | Missing dependency | Added `mediapy` |
| v15 | Missing dependency | Added `scikit-image` for marching cubes |

**Key Files**:
- `runpod/stage-pipeline/Dockerfile` - Combined ViPE + 2DGS image
- `runpod/stage-pipeline/handler.py` - Serverless handler
- `runpod/stage-pipeline/vipe_to_2dgs.py` - Format converter
- `runpod/stage-pipeline/init_points_from_depth.py` - Point cloud generator
- `scripts/test_stage_pipeline.py` - Test script

**Docker Image**: `88dreams/stage-pipeline:v15`

**RunPod Endpoint**: `2dgs-serverless` (ID: `s9txp6edtf2vg4`)

---

## Recent Work Completed

### 1. Stage Pipeline Development (Current Session - Jan 11, 2026)

Created combined ViPE + 2DGS serverless endpoint for video-to-mesh reconstruction:

- Built Dockerfile with ViPE and 2DGS CUDA extensions
- Resolved ABI compatibility issues between PyTorch and ViPE extensions
- Fixed PLY format to include normal vectors required by 2DGS
- Added missing dependencies (mediapy, scikit-image)
- Created test script with S3 upload support

### 2. SHARP Video Rendering (Previous Session)

**Problem**: SHARP's `--render` flag for video generation was intermittently failing.

**Solutions Implemented**:
1. **gsplat Pre-compilation** in `start_unified.sh`
2. **Always Upload to S3** via `upload_file_to_s3_always()`
3. **Extended Timeout** (30 minutes for video)
4. **Enhanced Logging** via `check_environment()`

**Docker Version**: v50 (`88dreams/gen3c-runpod:v50`)

### 3. UI Modernization

- Sidebar navigation layout with INPUT, CREATE, TOOLS sections
- Custom dark theme color palette
- System metrics display (CPU, Memory, GPU)
- Experiment logging to CSV

---

## Docker Images

| Image | Models | Current Version |
|-------|--------|-----------------|
| `88dreams/gen3c-runpod` | Gen3C, Lyra, SHARP, SuGaR | v50 |
| `88dreams/trellis-runpod` | TRELLIS.2 | v10 |
| `88dreams/hunyuan-runpod` | Hunyuan3D | v10 |
| `88dreams/stage-pipeline` | ViPE + 2DGS | v15 |

### Build Commands

```bash
# Unified Image
cd runpod/gen3c
docker build -f Dockerfile.patch -t 88dreams/gen3c-runpod:vXX .
docker push 88dreams/gen3c-runpod:vXX

# Stage Pipeline
cd runpod/stage-pipeline
docker build -t 88dreams/stage-pipeline:vXX .
docker push 88dreams/stage-pipeline:vXX
```

---

## RunPod Serverless Endpoints

| Endpoint Name | Image | Purpose |
|---------------|-------|---------|
| `gen3c-serverless` | gen3c-runpod:v50 | Multi-model (Gen3C, Lyra, SHARP) |
| `trellis-serverless` | trellis-runpod:v10 | TRELLIS.2 3D generation |
| `hunyuan-serverless` | hunyuan-runpod:v10 | Hunyuan3D mesh generation |
| `2dgs-serverless` | stage-pipeline:v15 | Video to mesh (ViPE + 2DGS) |

---

## Key Files

### Stage Pipeline
| File | Purpose |
|------|---------|
| `runpod/stage-pipeline/Dockerfile` | Combined ViPE + 2DGS Docker image |
| `runpod/stage-pipeline/handler.py` | Serverless handler orchestrating pipeline |
| `runpod/stage-pipeline/vipe_to_2dgs.py` | ViPE → COLMAP format converter |
| `runpod/stage-pipeline/init_points_from_depth.py` | Point cloud from depth maps |
| `scripts/test_stage_pipeline.py` | Test script with S3 upload |

### Unified Handler
| File | Purpose |
|------|---------|
| `runpod/gen3c/handler_unified.py` | Multi-model serverless handler |
| `runpod/gen3c/start_unified.sh` | Container startup with gsplat precompile |
| `runpod/runpod_client.py` | Python client for RunPod API |

### UI
| File | Purpose |
|------|---------|
| `app_sidebar.py` | Main Gradio application |
| `handlers/generation_handlers.py` | Business logic for generation |
| `generators/*.py` | Model-specific logic |

---

## Environment Setup

### Local Requirements

```bash
pip install gradio==6.0.1 requests boto3 python-dotenv trimesh runpod
```

### AWS Credentials

Create `.env` file (not in repo):
```
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
```

### RunPod API Key

For testing scripts:
```bash
export RUNPOD_API_KEY="your_key"
```

Or set in UI Settings page.

---

## Testing Stage Pipeline

```bash
# Test with local video (uploads to S3 automatically)
cd /home/arkrunr02/Hunyuan3D-2-Fork
RUNPOD_API_KEY="your_key" python scripts/test_stage_pipeline.py \
    --video /path/to/video.mp4 \
    --iterations 1000

# Check logs
cat /tmp/test-v15.log

# Health check
RUNPOD_API_KEY="your_key" python scripts/test_stage_pipeline.py --health
```

---

## Known Issues / Next Steps

### 1. Stage Pipeline (v15)
- Currently testing - awaiting RunPod endpoint update
- If v15 fails, check logs for next missing dependency

### 2. Lyra Integration
- SDG step takes 60-90 minutes
- Consider adding progress callbacks

### 3. SuGaR Mesh Extraction
- Full training pipeline is complex
- Currently using Poisson reconstruction alternative

---

## Debugging Tips

### Stage Pipeline Logs
1. Go to RunPod Console → Serverless → `2dgs-serverless`
2. Click job ID to see logs
3. Look for:
   - "STEP 1: Running ViPE" - Camera pose extraction
   - "STEP 2: Converting to 2DGS format" - COLMAP conversion
   - "STEP 3: Generating initial point cloud" - Depth → points
   - "STEP 4: Training 2DGS" - Gaussian splatting
   - "STEP 5: Extracting mesh" - Marching cubes

### Common Issues
1. **Missing dependency**: Add to Dockerfile, rebuild, push
2. **ABI mismatch**: Ensure `_GLIBCXX_USE_CXX11_ABI=0` is set
3. **PLY format errors**: Check normals are included
4. **Timeout**: Increase RunPod execution timeout or reduce iterations

---

## Contact / Resources

- **RunPod Console**: https://www.runpod.io/console
- **Docker Hub**: https://hub.docker.com/u/88dreams
- **S3 Bucket**: arkrunr (us-west-1)
- **Git Branch**: `2dgs`

---

*Last Updated: January 11, 2026*

