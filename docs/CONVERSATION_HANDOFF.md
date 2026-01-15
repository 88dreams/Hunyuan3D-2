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
| **2DGS Pipeline** | ViPE + 2DGS | Video → Mesh (GLB) | ✅ Working (v20) |
| **LTX-2** | Lightricks | Video (official API) | ✅ Working |
| **SEVA** | Stability AI | Video (camera control) | ⏸️ Blocked (HF access) |

### Architecture

- **Local**: Gradio UI (`app_sidebar.py`) running on user's machine
- **Remote**: RunPod Serverless endpoints for GPU inference
- **Storage**: AWS S3 for large file transfer, RunPod network volume for persistence

---

## Current Work In Progress

### LTX-2 Integration (January 15, 2026) ✅ COMPLETE

**Goal**: High-quality video generation from images with camera control.

**Use Case**: Generate camera movement videos (dolly, orbit, jib) that can be fed into the 2DGS pipeline for 3D reconstruction.

#### Final Solution: Official LTX API ✅

After exploring multiple approaches, we implemented the **official Lightricks API** at `https://api.ltx.video`.

| Approach | Status | Notes |
|----------|--------|-------|
| Native Pipeline (`ltx-native`) | ❌ Abandoned | OOM - 19B model exceeds 140GB VRAM |
| Diffusers Pipeline (`ltx-diffuser`) | ❌ Abandoned | Camera LoRAs incompatible (2B vs 19B) |
| **Official LTX API** | ✅ **IMPLEMENTED** | Direct API calls, no local GPU needed |

#### LTX API Features

- **Models**: `ltx-2-pro` (best quality), `ltx-2-fast` (quick generation)
- **Resolutions**: 1080p, 1440p, 4K
- **Durations**: 6, 8, 10 seconds
- **FPS**: 25 or 50
- **Camera Control**: Via prompt description (not LoRAs)
- **Audio**: Optional AI-generated audio

#### Camera Motion via Prompt

Camera motion is controlled through the prompt text:
- "Camera slowly pulls back from the subject" → dolly out
- "Camera orbits around the subject" → orbit
- "Camera rises vertically" → jib up

Preset prompts are available in the UI dropdown.

#### Files Modified

| File | Changes |
|------|---------|
| `runpod/runpod_client.py` | Added `LTXAPIClient` class for official API |
| `generators/ltx2.py` | New `run_ltx2_api()` function, kept legacy RunPod support |
| `app_sidebar.py` | New LTX-2 UI with model/resolution/duration selectors, LTX API key in Settings |

#### API Configuration

1. Get API key at https://ltx.video
2. Add to Settings → External APIs → LTX-2

No RunPod endpoint needed - API calls go directly to Lightricks.

#### Key Technical Learnings (Historical)

1. **Model Size Mismatch**: HuggingFace `Lightricks/LTX-Video` (2B) ≠ `LTX-2` (19B)
2. **FP8 Upcasting**: Native pipeline upcasts FP8 to BF16 during inference
3. **API is Best**: Official API avoids all local hosting complexity

---

### TRELLIS.2 Consolidated into Unified (January 13, 2026)

**Change**: TRELLIS.2 moved from separate endpoint to unified handler.

**Why**: The dependency issue (transformers 4.48.0) is now resolved in unified v52.

**Before**: Separate `trellis-serverless` endpoint with `trellis-runpod:v10` image  
**After**: Use `gen3c-serverless` with `"model": "trellis"`

---

### SEVA (Stable Virtual Camera) Integration

**Status**: ⏸️ **Blocked** - Awaiting HuggingFace Model Access

**Completed Infrastructure**:
- `runpod/seva/Dockerfile` - Docker image definition
- `runpod/seva/handler_seva.py` - Serverless handler
- `generators/seva.py` - Generator module
- Docker image: `88dreams/seva-runpod:v1`

**Blocker**: Requires HuggingFace access to `stabilityai/stable-virtual-camera`

---

### 2DGS Pipeline: ViPE + 2DGS

**Status**: ✅ **WORKING** (v20)

**Pipeline Flow**:
1. ViPE extracts camera poses, depth maps, intrinsics
2. Converter transforms to COLMAP format
3. Point cloud generated from depth maps
4. 2DGS training produces Gaussian splats
5. Mesh extraction via marching cubes
6. 180° X-axis rotation correction applied

**Docker Image**: `88dreams/2dgs-pipeline:v20`
**RunPod Endpoint**: `2dgs-serverless` (ID: `s9txp6edtf2vg4`)

---

## Docker Images

| Image | Models | Current Version |
|-------|--------|-----------------|
| `88dreams/gen3c-runpod` | Gen3C, Lyra, SHARP, SuGaR, TRELLIS.2, LTX-2 | **v55g** |
| `88dreams/hunyuan-runpod` | Hunyuan3D | v10 |
| `88dreams/2dgs-pipeline` | ViPE + 2DGS | v20 |
| `88dreams/seva-runpod` | SEVA (camera control) | v1 (blocked) |

### Version History (LTX-2 Development)

| Version | Changes |
|---------|---------|
| v55 | Initial diffusers implementation |
| v55a | Auto-detect LoRA paths, HuggingFace cache on network volume |
| v55b | Set HF_HOME before imports |
| v55c | Fixed path detection priority |
| v55d | Runtime path detection |
| v55e | Forced rebuild with correct paths |
| v55f | Runtime `get_ltx2_checkpoint_dir()` function |
| **v55g** | Graceful handling of incompatible LoRAs |

### Build Commands

```bash
# Unified Image (LTX-2 diffusers)
cd runpod/gen3c
docker build --no-cache -f Dockerfile.ltx2.diffusers -t 88dreams/gen3c-runpod:v55g .
docker push 88dreams/gen3c-runpod:v55g

# 2DGS Pipeline
cd runpod/2dgs-pipeline
docker build -t 88dreams/2dgs-pipeline:vXX .
docker push 88dreams/2dgs-pipeline:vXX
```

---

## RunPod Serverless Endpoints

| Endpoint Name | Image | Purpose |
|---------------|-------|---------|
| `gen3c-serverless` | gen3c-runpod:**v55g** | Multi-model (Gen3C, Lyra, SHARP, TRELLIS.2, LTX-2) |
| `hunyuan-serverless` | hunyuan-runpod:v10 | Hunyuan3D mesh generation |
| `2dgs-serverless` | 2dgs-pipeline:v20 | Video to mesh (ViPE + 2DGS) |

---

## Key Files

### LTX-2 Integration (API-based)

| File | Purpose |
|------|---------|
| `runpod/runpod_client.py` | `LTXAPIClient` class for official API |
| `generators/ltx2.py` | Generator module with `run_ltx2_api()` |
| `app_sidebar.py` | UI tab with model/resolution/duration selectors |

### LTX API Client Functions

```python
# Main client class
class LTXAPIClient:
    def generate_video(image_path, prompt, model, resolution, duration, fps, ...) -> LTX2Result
    def text_to_video(prompt, model, resolution, duration, fps, ...) -> LTX2Result

# Generator functions
def run_ltx2_api(image_path, prompt, model, resolution, ...) -> LTX2Result
def generate_video_for_3d(image_path, camera_motion, ...) -> LTX2Result
def text_to_video(prompt, ...) -> LTX2Result
```

---

## Environment Setup

### AWS Credentials (`.env` file)
```
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
```

### RunPod API Key
```bash
export RUNPOD_API_KEY="your_key"
```

### Local Output Directory
```
/srv/searidge_share/outputs/
├── gen3c/      # Gen3C video outputs
├── ltx2/       # LTX-2 video outputs
├── sharp/      # SHARP PLY outputs
├── mesh_2dgs/  # 2DGS mesh outputs
└── logs/       # Experiment logs
```

---

## Known Issues / Next Steps

### 1. LTX-2 ✅ COMPLETE
- **Solution**: Using official Lightricks API at https://api.ltx.video
- **Camera Control**: Via prompt descriptions (built-in presets in UI)
- **Quality**: Up to 4K @ 50fps with Pro model

### 2. SEVA Integration ⏸️ BLOCKED
- **Issue**: HuggingFace model access required
- **Action**: Request access at https://huggingface.co/stabilityai/stable-virtual-camera

### 3. 2DGS Pipeline ✅ COMPLETE
- Successfully generates meshes from Gen3C video
- 180° X-axis rotation applied automatically

---

## Debugging Tips

### LTX-2 API Issues

1. **API key not found**: Configure in Settings → External APIs → LTX-2
2. **401 Unauthorized**: Check API key is valid at https://ltx.video
3. **Timeout**: API may take 30-120s for video generation
4. **S3 upload fails**: Check AWS credentials in `.env`, or disable "Use S3" option to use base64

### Logs to Check

```bash
# In app logs, look for:
[LTX-API] Starting video generation
[LTX-API] Model: ltx-2-pro, Resolution: 1920x1080, Duration: 6s
[LTX-API] Response status: 200
[LTX-API] Video saved: X.XX MB
```

---

## Contact / Resources

- **RunPod Console**: https://www.runpod.io/console
- **Docker Hub**: https://hub.docker.com/u/88dreams
- **S3 Bucket**: arkrunr (us-west-1)
- **LTX API**: https://docs.ltx.video/welcome
- **Git Branch**: `main` (LTX API integrated)

---

*Last Updated: January 15, 2026 (LTX-2 official API integration complete)*
