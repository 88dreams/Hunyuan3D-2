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
| **2DGS Pipeline** | ViPE + 2DGS | Multi-Video → Mesh (GLB) | ✅ Working (v2) |
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

#### UI Features

- **Multi-video generation**: Select multiple camera motions, generates all videos
- **4-video preview grid**: 2x2 preview layout with filename labels
- **Play All button**: Play/pause all previews simultaneously
- **Parameter encoding**: Filenames encode model, duration, resolution, motion

#### Files Modified

| File | Changes |
|------|---------|
| `runpod/runpod_client.py` | Added `LTXAPIClient` class for official API |
| `generators/ltx2.py` | New `run_ltx2_api()` function, kept legacy RunPod support |
| `app_sidebar.py` | LTX-2 UI with 4-video previews, Play All, multi-motion selection |
| `scripts/experiment_logger.py` | Added `ltx2_param_filename()` for parameter encoding |

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

### 2DGS Pipeline: Multi-Video to Mesh

**Status**: ✅ **WORKING** (v2 - Multi-Video Support)

**Pipeline Flow**:
1. Accept up to 4 input videos
2. ViPE extracts camera poses, depth maps, intrinsics per video
3. **Frame 0 alignment**: First frame (input image) used as anchor to align poses
4. **Depth coverage filter**: Skip frames with low depth coverage
5. Converter merges all videos to unified COLMAP format
6. 2DGS training produces Gaussian splats
7. Mesh extraction via marching cubes
8. 180° X-axis rotation correction applied

**Key Enhancement**: Multi-video input provides diverse viewpoints for better 3D reconstruction.

**Docker Image**: `88dreams/2dgs-pipeline:v2`
**RunPod Endpoint**: `2dgs-serverless`

**UI Features**:
- 4-video preview grid with filename labels
- Play All button for simultaneous playback
- Video list with sort (date/filename/model) and limit (10/20/30/50/All)
- Single-column video list layout
- Highlight currently playing video in list

---

## Docker Images

| Image | Models | Current Version |
|-------|--------|-----------------|
| `88dreams/gen3c-runpod` | Gen3C, Lyra, SHARP, SuGaR, TRELLIS.2 | **v55g** |
| `88dreams/hunyuan-runpod` | Hunyuan3D | v10 |
| `88dreams/2dgs-pipeline` | ViPE + 2DGS (multi-video) | **v2** |
| `88dreams/seva-runpod` | SEVA (camera control) | v1 (blocked) |

**Note**: LTX-2 uses official API - no Docker image needed.

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
# Unified Image
cd runpod/gen3c
docker build --no-cache -f Dockerfile.unified -t 88dreams/gen3c-runpod:v55g .
docker push 88dreams/gen3c-runpod:v55g

# 2DGS Pipeline (Multi-Video)
cd runpod/2dgs-pipeline
docker build -t 88dreams/2dgs-pipeline:v2 .
docker push 88dreams/2dgs-pipeline:v2
```

---

## UI Features (January 2026)

### Automatic Output Naming

When an image is loaded, output filenames are automatically set to `{input_name}-{model}`:
- Load `CBGB1.jpg` → Gen3C output: `CBGB1-gen3c`
- Load `CBGB1.jpg` → LTX-2 output: `CBGB1-ltx2`

User can still edit the output name before generation.

### Encode Parameters in Filename

When "Encode in Filename" is enabled in Settings, output filenames include key parameters:

| Model | Encoded Parameters | Example |
|-------|-------------------|---------|
| **SHARP (PLY)** | guidance (g), steps (s) | `sharp_CBGB1_g7.5_s50.ply` |
| **SHARP (Video)** | g, s, lateral_offset (lo), zoom_forward (zf) | `sharp_CBGB1_g7.5_s50_lo050_zf025.mp4` |
| **Gen3C** | frames (f), trajectory (t), movement_distance (md), guidance (g), seed (s) | `gen3c_CBGB1_f121_torb_md10_g10_s42.mp4` |
| **LTX-2** | model (m), duration (d), resolution, camera_motion | `ltx2_CBGB1_mpro_d6_1080p_dollyout.mp4` |
| **2DGS** | input video basenames | `2dgs_CBGB1_dolly_orbit.glb` |
| **Mesh Extract** | input name + "_processed" | `CBGB1_processed.glb` |

### Video Preview Features

Both LTX-2 and 2DGS pages have:
- **4-video preview grid** (2x2 layout)
- **Filename labels** above each preview
- **Play All button** - plays/pauses all loaded videos simultaneously
- **Resume from pause** - videos continue from where paused (not restart)

---

## RunPod Serverless Endpoints

| Endpoint Name | Image | Purpose |
|---------------|-------|---------|
| `gen3c-serverless` | gen3c-runpod:**v55g** | Multi-model (Gen3C, Lyra, SHARP, TRELLIS.2) |
| `hunyuan-serverless` | hunyuan-runpod:v10 | Hunyuan3D mesh generation |
| `2dgs-serverless` | 2dgs-pipeline:**v2** | Multi-video to mesh (ViPE + 2DGS) |

**Note**: LTX-2 uses official Lightricks API at https://api.ltx.video - no RunPod endpoint.

---

## Key Files

### LTX-2 Integration (API-based)

| File | Purpose |
|------|---------|
| `runpod/runpod_client.py` | `LTXAPIClient` class for official API |
| `generators/ltx2.py` | Generator module with `run_ltx2_api()` |
| `app_sidebar.py` | UI: 4-video preview, multi-motion selection, Play All |
| `scripts/experiment_logger.py` | `ltx2_param_filename()` for encoding |

### 2DGS Pipeline (Multi-Video)

| File | Purpose |
|------|---------|
| `runpod/2dgs-pipeline/handler.py` | Serverless handler with multi-video support |
| `runpod/2dgs-pipeline/multi_video_merge.py` | Pose alignment and frame merging |
| `runpod/runpod_client.py` | `TwoDGSPipelineClient.submit_multi_video_job()` |
| `ui/tabs/create_tab.py` | 2DGS UI components |
| `ui/styles.py` | CSS for video previews, Play All button |

### LTX API Client Functions

```python
# Main client class
class LTXAPIClient:
    def generate_video(image_path, prompt, model, resolution, duration, fps, ...) -> LTX2Result
    def text_to_video(prompt, model, resolution, duration, fps, ...) -> LTX2Result

# Generator functions
def run_ltx2_api(image_path, prompt, model, resolution, ...) -> LTX2Result
def generate_video_for_3d(image_path, camera_motion, ...) -> LTX2Result
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
- **Multi-video**: Select multiple camera motions, 4-video preview grid

### 2. 2DGS Multi-Video Pipeline ✅ COMPLETE
- Accepts up to 4 videos for diverse viewpoints
- Frame 0 alignment ensures consistent pose origin
- Depth coverage filter improves quality
- Enhanced UI with sort/filter and Play All

### 3. SEVA Integration ⏸️ BLOCKED
- **Issue**: HuggingFace model access required
- **Action**: Request access at https://huggingface.co/stabilityai/stable-virtual-camera

### 4. Potential Improvements
- LTX API camera pose extraction (asked Lightricks if available)
- Auto-select best frames based on depth coverage
- Multi-GPU parallel video generation

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

*Last Updated: January 14, 2026 (LTX-2 multi-preview, 2DGS multi-video, UI enhancements)*
