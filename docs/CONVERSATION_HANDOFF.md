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
| **2DGS Pipeline** | ViPE + 2DGS | Multi-Video → Mesh (GLB) | ✅ Working (v13) |
| **LTX-2** | Lightricks | Video (official API) | ✅ Working |
| **SEVA** | Stability AI | Video (camera control) | ⏸️ Blocked (HF access) |

### Architecture

- **Local**: Gradio UI (`app_sidebar.py`) running on user's machine
- **Remote**: RunPod Serverless endpoints for GPU inference
- **Storage**: AWS S3 for large file transfer, RunPod network volume for persistence

---

## Current Work In Progress

### Tagging System (January 17, 2026) ✅ COMPLETE

**Goal**: Unified tagging system for organizing and filtering render results across all models.

#### Features Implemented

| Feature | Description |
|---------|-------------|
| **TagManager** | Thread-safe JSON-based tag storage (`utils/tag_manager.py`) |
| **Gallery Tab** | Browse, filter, and tag outputs from all models |
| **Post-Generation Tagging** | Tag results directly after Gen3C, LTX-2, Hunyuan, Trellis generation |
| **2DGS Video Filter** | Filter videos by tag in 2DGS video selection |
| **Auto-Save** | Tags save automatically on checkbox change |
| **Delete Tag** | Removes file from disk with confirmation |

#### Predefined Tags

| Column 1 | Column 2 |
|----------|----------|
| Approved | Review |
| Favorite | Bad |
| Best-take | Delete |

Plus **Custom** tags with free-form text entry.

#### Gallery Tab UI

- **Filters**: Model filter, Tag filter, Max results slider
- **File List**: Scrollable list showing `[Model] filename [tags] (date)`
- **Preview**: Video or 3D model preview (400px height)
- **Tagging**: Two-column checkbox layout + custom tag input
- **Actions**: Clear Tags, Clear Tags (ALL displayed) with confirmation

#### Files Created/Modified

| File | Purpose |
|------|---------|
| `utils/tag_manager.py` | **NEW** - Tag storage and management |
| `ui/tabs/gallery_tab.py` | **NEW** - Gallery tab UI and logic |
| `ui/tabs/__init__.py` | Added gallery_tab export |
| `app_sidebar.py` | Gallery nav, post-generation tagging panels |
| `handlers/generation_handlers.py` | Tag filtering in `list_all_videos()` |
| `ui/tabs/create_tab.py` | Tag filter dropdown for 2DGS |

#### Tag Storage

Tags are stored in `/srv/searidge_share/outputs/tags.json`:

```json
{
  "files": {
    "/path/to/file.mp4": {
      "tags": ["approved", "favorite"],
      "model": "LTX-2",
      "added": "2026-01-17T10:30:00",
      "modified": "2026-01-17T10:35:00"
    }
  },
  "custom_tags": ["my-project", "hero-shot"],
  "predefined_tags": ["approved", "favorite", "best-take", "review", "bad", "delete"]
}
```

---

### 2DGS Pipeline v13 (January 16, 2026) ✅ COMPLETE

**Enhancements**:

1. **Checkpoint Save/Restore**: Intermediate outputs saved to S3 for job resumption
2. **Output Name Preservation**: Custom `output_name` from UI used for final S3 upload
3. **CUDA Availability Check**: Validates GPU before ViPE inference
4. **Depth Handling Fixes**: Improved depth map processing

**Docker Image**: `88dreams/2dgs-pipeline:v13`

---

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

## Docker Images

| Image | Models | Current Version |
|-------|--------|-----------------|
| `88dreams/gen3c-runpod` | Gen3C, Lyra, SHARP, SuGaR, TRELLIS.2 | **v55g** |
| `88dreams/hunyuan-runpod` | Hunyuan3D | v10 |
| `88dreams/2dgs-pipeline` | ViPE + 2DGS (multi-video) | **v13** |
| `88dreams/seva-runpod` | SEVA (camera control) | v1 (blocked) |

**Note**: LTX-2 uses official API - no Docker image needed.

### Build Commands

```bash
# Unified Image
cd runpod/gen3c
docker build --no-cache -f Dockerfile.unified -t 88dreams/gen3c-runpod:v55g .
docker push 88dreams/gen3c-runpod:v55g

# 2DGS Pipeline (Multi-Video with Checkpointing)
cd runpod/2dgs-pipeline
docker build -t 88dreams/2dgs-pipeline:v13 .
docker push 88dreams/2dgs-pipeline:v13
```

---

## UI Features (January 2026)

### Gallery Tab (NEW)

Navigate via **BROWSE** section in sidebar:
- Browse all model outputs in one place
- Filter by model (Gen3C, LTX-2, Hunyuan, Trellis, 2DGS)
- Filter by tags
- Preview videos and 3D models
- Tag/untag files with auto-save
- Delete files with confirmation

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
| `2dgs-serverless` | 2dgs-pipeline:**v13** | Multi-video to mesh (ViPE + 2DGS) |

**Note**: LTX-2 uses official Lightricks API at https://api.ltx.video - no RunPod endpoint.

---

## Key Files

### Tagging System (NEW)

| File | Purpose |
|------|---------|
| `utils/tag_manager.py` | Thread-safe tag storage in JSON |
| `ui/tabs/gallery_tab.py` | Gallery tab UI for browsing/tagging |
| `handlers/generation_handlers.py` | `list_all_videos()` with tag filtering |
| `ui/tabs/create_tab.py` | Tag filter in 2DGS video selection |

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
| `runpod/2dgs-pipeline/handler.py` | Serverless handler with checkpointing |
| `runpod/2dgs-pipeline/multi_video_merge.py` | Pose alignment and frame merging |
| `runpod/runpod_client.py` | `TwoDGSPipelineClient.submit_multi_video_job()` |
| `ui/tabs/create_tab.py` | 2DGS UI components |
| `ui/styles.py` | CSS for video previews, Play All button |

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
├── hunyuan/    # Hunyuan3D outputs
├── trellis/    # Trellis outputs
├── 2dgs/       # 2DGS mesh outputs
├── sharp/      # SHARP PLY outputs
├── logs/       # Experiment logs
└── tags.json   # Tag database (NEW)
```

---

## Known Issues / Next Steps

### 1. Tagging System ✅ COMPLETE
- Gallery tab for browsing all outputs
- Post-generation tagging for each model
- Tag filtering in 2DGS video selection
- Auto-save and delete with confirmation

### 2. LTX-2 ✅ COMPLETE
- **Solution**: Using official Lightricks API at https://api.ltx.video
- **Camera Control**: Via prompt descriptions (built-in presets in UI)
- **Quality**: Up to 4K @ 50fps with Pro model
- **Multi-video**: Select multiple camera motions, 4-video preview grid

### 3. 2DGS Pipeline v13 ✅ COMPLETE
- Checkpoint save/restore for job resumption
- Custom output naming preserved through pipeline
- CUDA availability validation
- Improved depth handling

### 4. SEVA Integration ⏸️ BLOCKED
- **Issue**: HuggingFace model access required
- **Action**: Request access at https://huggingface.co/stabilityai/stable-virtual-camera

### 5. Potential Improvements
- LTX API camera pose extraction (asked Lightricks if available)
- Auto-select best frames based on depth coverage
- Multi-GPU parallel video generation
- Tag-based batch processing workflows

---

## Debugging Tips

### Tagging System Issues

1. **Tags not saving**: Check write permissions on `/srv/searidge_share/outputs/tags.json`
2. **Files not appearing in Gallery**: Verify output directories exist and contain files
3. **Delete not working**: Check file permissions, look for OS errors in console

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

# For tagging:
[TagManager] Saving tags for: /path/to/file
[Gallery] Scanning output directories
```

---

## Contact / Resources

- **RunPod Console**: https://www.runpod.io/console
- **Docker Hub**: https://hub.docker.com/u/88dreams
- **S3 Bucket**: arkrunr (us-west-1)
- **LTX API**: https://docs.ltx.video/welcome
- **Git Branch**: `ltx-API` (main development branch)

---

*Last Updated: January 17, 2026 (Tagging system, Gallery tab, 2DGS v13)*
