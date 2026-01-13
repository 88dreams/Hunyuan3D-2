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
| **LTX-2** | Lightricks | Video (4K, camera LoRAs) | 🔧 Infrastructure Ready |
| **SEVA** | Stability AI | Video (camera control) | ⏸️ Blocked (HF access) |

### Architecture

- **Local**: Gradio UI (`app_sidebar.py`) running on user's machine
- **Remote**: RunPod Serverless endpoints for GPU inference
- **Storage**: AWS S3 for large file transfer, RunPod network volume for persistence

---

## Current Work In Progress

### LTX-2 Integration (January 13, 2026) ⭐ NEW

**Goal**: High-quality video generation from images with camera control LoRAs.

**Use Case**: Generate camera movement videos (dolly, jib) that can be fed into the 2DGS pipeline for 3D reconstruction. **No HuggingFace approval needed** - open weights!

**Status**: 🔧 Ready for Build - Added to Unified Handler

**Why LTX-2 over SEVA**:
- ✅ Open weights (no HF access wait)
- ✅ Higher quality (native 4K @ 50fps)
- ✅ Faster inference (8-step distilled model)
- ✅ Commercial-friendly license

**Approach**: Added to unified handler (same pattern as Trellis) - no separate Docker needed!

**Completed**:
- `runpod/gen3c/handler_unified.py` - Added handle_ltx2(), validate_ltx2(), run_ltx2() ✅
- `runpod/gen3c/Dockerfile.patch.v51` - Patch to add diffusers ✅
- `runpod/runpod_client.py` - Added LTX2ServerlessClient, LTX2Result ✅
- `generators/ltx2.py` - Generator module ✅
- Standalone files also available in `runpod/ltx2/` as fallback

**Camera Motion LoRAs**:
- `dolly_out` - Best for 3D reconstruction (pulls away from subject)
- `dolly_in` - Pushes toward subject
- `dolly_left` / `dolly_right` - Lateral movement
- `jib_up` - Vertical rise
- `static` - No camera movement

**Next Steps**:
1. Build v51 image: `docker build -f Dockerfile.patch.v51 -t 88dreams/gen3c-runpod:v51 .`
2. Push to Docker Hub
3. Update existing gen3c-serverless endpoint to v51
4. Add UI integration

**Usage**: Same endpoint as Gen3C/SHARP/Lyra, just use `"model": "ltx2"`

**Documentation**: `docs/LTX2_INTEGRATION_PLAN.md`

---

### TRELLIS.2 Consolidated into Unified (January 13, 2026)

**Change**: TRELLIS.2 moved from separate endpoint to unified handler.

**Why**: The dependency issue (transformers 4.48.0) is now resolved in unified v52.

**Before**: Separate `trellis-serverless` endpoint with `trellis-runpod:v10` image  
**After**: Use `gen3c-serverless` with `"model": "trellis"`

**Files Changed**:
- `runpod/gen3c/handler_unified.py` - Fixed `run_trellis()` implementation
- `runpod/gen3c/trellis_inference.py` - Copied from trellis/ directory
- `runpod/gen3c/Dockerfile.patch.v52` - Includes trellis_inference.py

**UI Note**: In Settings, users can now use the unified Gen3C endpoint ID for TRELLIS.2 generation (the separate Trellis endpoint field is for backwards compatibility only).

---

### SEVA (Stable Virtual Camera) Integration (January 13, 2026)

**Goal**: Add precise camera control video generation from single images.

**Use Case**: Generate camera movement videos (pan, tilt, orbit, etc.) that can be fed into the 2DGS pipeline for better 3D reconstruction than Gen3C provides.

**Status**: ⏸️ **Blocked** - Awaiting HuggingFace Model Access

**Completed**:
- `runpod/seva/Dockerfile` - Docker image definition ✅
- `runpod/seva/handler_seva.py` - Serverless handler with S3 support ✅
- `runpod/seva/start_seva.sh` - Startup script with HF login ✅
- `runpod/seva/README.md` - Deployment documentation ✅
- `runpod/runpod_client.py` - Added SEVAServerlessClient, SEVAResult ✅
- `generators/seva.py` - Generator module with trajectory helpers ✅
- Docker image built and pushed: `88dreams/seva-runpod:v1` ✅

**Next Steps**:
1. Request HuggingFace access: https://huggingface.co/stabilityai/stable-virtual-camera
2. Create RunPod serverless endpoint with `88dreams/seva-runpod:v1`
3. Add UI integration

**Documentation**: `docs/STABLE_VIRTUAL_CAMERA_PLAN.md`

---

### 2DGS Pipeline: ViPE + 2DGS (January 11-12, 2026)

**Goal**: Convert Gen3C video output into 3D mesh using video pose estimation and 2D Gaussian Splatting.

**Pipeline Flow**:
1. **ViPE** (NVIDIA) extracts camera poses, depth maps, and intrinsics from video
2. **Converter** transforms ViPE output to COLMAP format for 2DGS
3. **Point Cloud Generator** creates initial 3D points from depth maps
4. **2DGS Training** produces Gaussian splats from video frames
5. **Mesh Extraction** via marching cubes
6. **Post-processing**: 180° X-axis rotation to correct mesh orientation

**Current Status**: ✅ **WORKING** (v20) - Successfully generates 36MB+ meshes with 727K+ vertices

**Performance** (241-frame video):
- Total pipeline time: ~8 minutes
- ViPE (pose extraction): ~5 minutes
- 2DGS training (1000 iterations): ~1 minute
- Mesh extraction (TSDF fusion): ~1 minute
- Output: 36.76 MB GLB with 727,168 vertices

**Issues Fixed (v1-v20)**:
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
| v16 | Mesh extraction errors | Improved error handling, log stdout/stderr |
| v17 | S3 region mismatch | Added explicit `region_name` to boto3 client |
| v18 | Trimesh/Open3D compat | Patched `mesh_utils.py` for manual conversion |
| v19 | Wrong mesh path | Fixed to look in `train/ours_{iter}/` not `mesh/` |
| v20 | GeoCalib weights | Pre-downloaded 110MB GeoCalib model weights |

**Key Files**:
- `runpod/2dgs-pipeline/Dockerfile` - Combined ViPE + 2DGS image
- `runpod/2dgs-pipeline/handler.py` - Serverless handler
- `runpod/2dgs-pipeline/vipe_to_2dgs.py` - Format converter
- `runpod/2dgs-pipeline/init_points_from_depth.py` - Point cloud generator
- `scripts/test_2dgs_pipeline.py` - Test script

**Docker Image**: `88dreams/2dgs-pipeline:v20`

**RunPod Endpoint**: `2dgs-serverless` (ID: `s9txp6edtf2vg4`)

---

## Recent Work Completed

### 1. UI Improvements & Mesh Orientation Fix (Jan 12, 2026)

**UI Changes:**
- Settings page: 4-column layout with stacked Test/Save buttons
- Added 2DGS Pipeline endpoint to Settings page
- Moved Logs accordions inside left column for consistent width
- Fixed Gen3C and 2DGS to update global output path display
- Removed emoji from 2DGS Generate button
- Hidden FP16 option on Hunyuan page, defaulted to "full" model

**2DGS Mesh Orientation Fix:**
- Added 180° X-axis rotation to correct mesh orientation after download
- Meshes from 2DGS pipeline were rotated vs original video frames
- Rotation applied automatically in `TwoDGSPipelineClient.generate_sync()`
- Added `scripts/test_mesh_rotation.py` for manual testing

### 2. 2DGS Pipeline Development (Jan 11, 2026)

Created combined ViPE + 2DGS serverless endpoint for video-to-mesh reconstruction:

- Built Dockerfile with ViPE and 2DGS CUDA extensions
- Resolved ABI compatibility issues between PyTorch and ViPE extensions
- Fixed PLY format to include normal vectors required by 2DGS
- Added missing dependencies (mediapy, scikit-image)
- Created test script with S3 upload support

### 3. SHARP Video Rendering (Previous Session)

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
| `88dreams/gen3c-runpod` | Gen3C, Lyra, SHARP, SuGaR, **LTX-2**, **TRELLIS.2** ⭐ | v51 → **v52** |
| `88dreams/hunyuan-runpod` | Hunyuan3D | v10 |
| `88dreams/2dgs-pipeline` | ViPE + 2DGS | v20 ✅ |
| `88dreams/seva-runpod` | SEVA (camera control) | v1 (blocked - HF access) |
| ~~`88dreams/trellis-runpod`~~ | ~~TRELLIS.2~~ | **DEPRECATED** - Use unified |

### Build Commands

```bash
# Unified Image
cd runpod/gen3c
docker build -f Dockerfile.patch -t 88dreams/gen3c-runpod:vXX .
docker push 88dreams/gen3c-runpod:vXX

# 2DGS Pipeline
cd runpod/2dgs-pipeline
docker build -t 88dreams/2dgs-pipeline:vXX .
docker push 88dreams/2dgs-pipeline:vXX
```

---

## RunPod Serverless Endpoints

| Endpoint Name | Image | Purpose |
|---------------|-------|---------|
| `gen3c-serverless` | gen3c-runpod:**v52** | Multi-model (Gen3C, Lyra, SHARP, **TRELLIS.2**, **LTX-2**) ⭐ |
| `hunyuan-serverless` | hunyuan-runpod:v10 | Hunyuan3D mesh generation |
| `2dgs-serverless` | 2dgs-pipeline:v20 | Video to mesh (ViPE + 2DGS) ✅ |
| `seva-serverless` | seva-runpod:v1 | Novel view video - blocked (HF access) |
| ~~`trellis-serverless`~~ | ~~trellis-runpod:v10~~ | **TERMINATED** - Use gen3c-serverless |

---

## Key Files

### LTX-2 (Lightricks Video) ⭐ NEW - Added to Unified Handler
| File | Purpose |
|------|---------|
| `runpod/gen3c/handler_unified.py` | Added handle_ltx2(), validate_ltx2(), run_ltx2() |
| `runpod/gen3c/Dockerfile.patch.v51` | Patch to add diffusers to v50 |
| `generators/ltx2.py` | Generator module |
| `docs/LTX2_INTEGRATION_PLAN.md` | Integration plan |
| `runpod/ltx2/*` | Standalone fallback (if unified doesn't work) |

### SEVA (Stable Virtual Camera)
| File | Purpose |
|------|---------|
| `runpod/seva/Dockerfile` | Docker image definition |
| `runpod/seva/handler_seva.py` | Serverless handler |
| `runpod/seva/start_seva.sh` | Startup script with HF login |
| `runpod/seva/README.md` | Deployment instructions |
| `generators/seva.py` | Generator module with trajectory helpers |
| `docs/STABLE_VIRTUAL_CAMERA_PLAN.md` | Integration plan |

### 2DGS Pipeline
| File | Purpose |
|------|---------|
| `runpod/2dgs-pipeline/Dockerfile` | Combined ViPE + 2DGS Docker image |
| `runpod/2dgs-pipeline/handler.py` | Serverless handler orchestrating pipeline |
| `runpod/2dgs-pipeline/vipe_to_2dgs.py` | ViPE → COLMAP format converter |
| `runpod/2dgs-pipeline/init_points_from_depth.py` | Point cloud from depth maps |
| `scripts/test_2dgs_pipeline.py` | Test script with S3 upload |
| `scripts/test_mesh_rotation.py` | Test 180° X-axis rotation on mesh files |

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

## Testing 2DGS Pipeline

```bash
# Test with local video (uploads to S3 automatically)
cd /home/arkrunr02/Hunyuan3D-2-Fork
RUNPOD_API_KEY="your_key" python scripts/test_2dgs_pipeline.py \
    --video /path/to/video.mp4 \
    --iterations 1000

# Check logs
cat /tmp/test-v20.log

# Health check
RUNPOD_API_KEY="your_key" python scripts/test_2dgs_pipeline.py --health
```

---

## Known Issues / Next Steps

### 0. SEVA Integration 🔧 IN PROGRESS
- Infrastructure complete (Dockerfile, handler, client, generator)
- **Blocker**: Requires HuggingFace model access approval
- Next: Build Docker image, create RunPod endpoint, add UI

### 1. 2DGS Pipeline (v20) ✅ COMPLETE
- Successfully generates high-quality meshes from Gen3C video
- Auto-downloads to `/srv/searidge_share/outputs/mesh_2dgs/`
- **Mesh orientation fix**: 180° X-axis rotation applied automatically after download

### 2. Lyra Integration
- SDG step takes 60-90 minutes
- Consider adding progress callbacks

### 3. SuGaR Mesh Extraction
- Full training pipeline is complex
- Currently using Poisson reconstruction alternative

---

## Debugging Tips

### 2DGS Pipeline Logs
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

*Last Updated: January 13, 2026 (added LTX-2 integration)*

