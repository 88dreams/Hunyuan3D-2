# ARKRUNR WORLDS - Conversation Handoff Document

This document summarizes the work completed and provides context for new conversation instances to continue development.

---

## Project Overview

**ARKRUNR WORLDS** is a multi-model 3D generation system built as a fork of Hunyuan3D-2. It provides a unified Gradio UI for multiple AI models that generate 3D content from images.

### Supported Models

| Model | Source | Output | Status |
|-------|--------|--------|--------|
| **SHARP** | Apple | 3DGS PLY + Video | ✅ Working |
| **Gen3C** | NVIDIA | Video (MP4) | ✅ Working |
| **Lyra** | NVIDIA | 3D/4D Gaussian Splatting | ✅ Working |
| **TRELLIS.2** | Microsoft | 3D (GLB) | ✅ Working |
| **Hunyuan3D** | Tencent | 3D Mesh (GLB) | ✅ Working |

### Architecture

- **Local**: Gradio UI (`app_sidebar.py`) running on user's machine
- **Remote**: RunPod Serverless endpoints for GPU inference
- **Storage**: AWS S3 for large file transfer, RunPod network volume for persistence

---

## Recent Work Completed

### 1. SHARP Video Rendering (Current Session)

**Problem**: SHARP's `--render` flag for video generation was intermittently failing.

**Root Cause**: `gsplat` CUDA kernels need JIT compilation on first run, which can timeout on cold workers.

**Solutions Implemented**:

1. **gsplat Pre-compilation** (`start_unified.sh`):
   - Added startup step that imports `gsplat` and triggers kernel compilation
   - Runs before handler starts accepting jobs
   - Ensures video rendering works on first job

2. **Always Upload to S3** (`handler_unified.py`):
   - New function `upload_file_to_s3_always()` 
   - SHARP now uploads both PLY and video to S3 regardless of size
   - Ensures files are available even if API response times out

3. **Extended Timeout** (`runpod_client.py`):
   - SHARP with video: 30 minutes (was 10 minutes)
   - SHARP without video: 10 minutes

4. **Enhanced Logging** (`handler_unified.py`):
   - `check_environment()` function at startup
   - Logs Python path, conda env, PyTorch/CUDA status, gsplat availability
   - Better subprocess environment logging

**Docker Version**: v50 (`88dreams/gen3c-runpod:v50`)

### 2. SHARP UI Enhancements

- Added tabbed interface: "Generate PLY" and "Render Video"
- Exposed all video trajectory parameters:
  - `trajectory_type`: rotate_forward, rotate, swipe, shake
  - `num_steps`: Number of video frames
  - `num_repeats`: Trajectory loops
  - `max_disparity`: Lateral camera offset
  - `max_zoom`: Forward camera movement
  - `lookat_mode`: point, ahead

### 3. UI Modernization (Previous Sessions)

- Implemented sidebar navigation layout
- Custom color palette (dark theme)
- Section categories: INPUT, CREATE, TOOLS
- System metrics display (CPU, Memory, GPU)
- Persistent button highlighting
- Image preview persistence

### 4. Experiment Logging

- CSV logging for all models (`scripts/experiment_logger.py`)
- Logs saved to `/srv/searidge_share/outputs/logs/`
- Optional parameter encoding in filenames

### 5. Mesh Cleanup Tab

- Integrated `trimesh` and `fast_simplification`
- Before/after 3D viewers
- Decimation and smoothing options

### 6. Update Tracker

- GitHub API integration for upstream version checking
- Local version tracking via `.version_tracking.json`
- "Query Deployed Version" to check RunPod instances

---

## Docker Images

### Unified Image (Gen3C, Lyra, SHARP, SuGaR)

```bash
# Build incremental update
cd runpod/gen3c
docker build -f Dockerfile.v49-patch -t 88dreams/gen3c-runpod:v50 .
docker push 88dreams/gen3c-runpod:v50
```

**Current Version**: v50
- v49: gsplat pre-compilation, environment checks
- v50: Always upload to S3

### Trellis Image

```bash
cd runpod/trellis
docker build -f Dockerfile -t 88dreams/trellis-runpod:v10 .
docker push 88dreams/trellis-runpod:v10
```

**Current Version**: v10

### Hunyuan Image

```bash
cd runpod/hunyuan
docker build -f Dockerfile -t 88dreams/hunyuan-runpod:v10 .
docker push 88dreams/hunyuan-runpod:v10
```

**Current Version**: v10

---

## Key Files Modified

### Handler (`runpod/gen3c/handler_unified.py`)

- `check_environment()` - Startup diagnostics
- `upload_file_to_s3_always()` - Always upload to S3
- `handle_sharp()` - Updated to always upload PLY and video
- `run_sharp()` - Enhanced logging for video rendering

### Startup Script (`runpod/gen3c/start_unified.sh`)

- Added gsplat pre-compilation section before handler starts

### Client (`runpod/runpod_client.py`)

- `generate_sharp_sync()` - Dynamic timeout based on `render_video`

### Main UI (`app_sidebar.py`)

- SHARP tab with video rendering options
- All trajectory parameters exposed

---

## Known Issues / Future Work

### 1. Video Trajectory Parameters
SHARP's CLI doesn't expose trajectory parameters directly. The parameters are passed to the handler but may not affect the actual trajectory until SHARP's Python API is used instead of CLI.

### 2. Lyra Integration
Lyra is integrated but the SDG (Synthetic Data Generation) step can take 60-90 minutes. Consider adding progress callbacks.

### 3. SuGaR Mesh Extraction
SuGaR is installed but the full training pipeline is complex. Current implementation uses Poisson reconstruction as a simpler alternative.

---

## Environment Setup

### Local Requirements

```bash
pip install gradio==6.0.1 requests boto3 python-dotenv trimesh
```

### AWS Credentials

Create `.env` file (not in repo):
```
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
```

### RunPod Credentials

Set in the UI's Settings page:
- Endpoint IDs for each model
- RunPod API Key

---

## Useful Commands

### Start the UI
```bash
python app_sidebar.py
```

### Quick Docker Rebuild
```bash
cd runpod/gen3c
docker build -f Dockerfile.v49-patch -t 88dreams/gen3c-runpod:vXX .
docker push 88dreams/gen3c-runpod:vXX
```

### Download from RunPod Storage Pod
```bash
scp -P <PORT> -i ~/.ssh/id_ed25519 root@<IP>:/workspace/outputs/sharp/file.mp4 ~/Downloads/
```

### Check S3 Files
```bash
curl -I "https://arkrunr.s3.us-west-1.amazonaws.com/MediaContent/outputs/sharp/filename.ply"
```

---

## File Locations

| Type | Local Path | RunPod Path | S3 Path |
|------|------------|-------------|---------|
| SHARP outputs | `outputs/sharp/` | `/runpod-volume/outputs/sharp/` | `MediaContent/outputs/sharp/` |
| Gen3C outputs | `outputs/gen3c/` | `/runpod-volume/outputs/gen3c/` | `MediaContent/outputs/gen3c/` |
| Logs | `/srv/searidge_share/outputs/logs/` | N/A | N/A |

---

## Debugging Tips

### Check RunPod Logs
1. Go to RunPod Console → Serverless → Your Endpoint
2. Click on a job ID to see logs
3. Look for:
   - "ENVIRONMENT CHECK" section at startup
   - "gsplat version" confirmation
   - "SHARP stdout/stderr" for model output
   - "S3 upload complete" for file transfers

### Common Issues

1. **UI Timeout**: Increase `max_wait` in client or check RunPod logs for actual completion
2. **Missing Video**: Check if gsplat pre-compilation succeeded in startup logs
3. **S3 Upload Fails**: Verify AWS credentials are set in RunPod environment variables
4. **Model Not Found**: Check network volume mount in startup logs

---

## Contact / Resources

- **RunPod Console**: https://www.runpod.io/console
- **Docker Hub**: https://hub.docker.com/u/88dreams
- **S3 Bucket**: arkrunr (us-west-1)

---

*Last Updated: January 8, 2026*

