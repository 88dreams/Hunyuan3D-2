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
| **LTX-2** | Lightricks | Video (camera LoRAs) | 🔧 Partial (see below) |
| **SEVA** | Stability AI | Video (camera control) | ⏸️ Blocked (HF access) |

### Architecture

- **Local**: Gradio UI (`app_sidebar.py`) running on user's machine
- **Remote**: RunPod Serverless endpoints for GPU inference
- **Storage**: AWS S3 for large file transfer, RunPod network volume for persistence

---

## Current Work In Progress

### LTX-2 Integration (January 15, 2026) ⭐ ACTIVE

**Goal**: High-quality video generation from images with camera control LoRAs.

**Use Case**: Generate camera movement videos (dolly, jib) that can be fed into the 2DGS pipeline for 3D reconstruction.

#### Status Summary

| Approach | Branch | Status | Issue |
|----------|--------|--------|-------|
| Native Pipeline (`ltx-pipelines`) | `ltx-native` | ❌ OOM | 19B model + Gemma 12B exceeds 140GB VRAM |
| Diffusers Pipeline | `ltx-diffuser` | ⚠️ Partial | Camera LoRAs incompatible (2B vs 19B) |
| API Integration | `ltx-API` | 🔧 In Progress | Current focus |

#### Detailed History

**Native Pipeline Attempt (`ltx-native` branch):**
- Used Lightricks' `ltx-pipelines` package with `TI2VidOneStagePipeline` and `DistilledPipeline`
- Required `gemma-3-12b-it-qat-q4_0-unquantized` text encoder (~23GB)
- Used FP8 transformer (`ltx-2-19b-dev-fp8.safetensors`, ~26GB)
- **Problem**: FP8 weights are upcasted to BF16 during inference, causing VRAM explosion
- **Result**: OOM even on H200 (140GB VRAM)
- Camera LoRAs worked correctly when model fit in memory

**Diffusers Pipeline (`ltx-diffuser` branch):**
- Switched to HuggingFace `diffusers` library with `LTXImageToVideoPipeline`
- Key feature: `enable_model_cpu_offload()` for memory efficiency
- **Discovery**: `Lightricks/LTX-Video` on HuggingFace is a **2B parameter** model
- Camera LoRAs from Lightricks are trained for the **19B parameter** model
- **Result**: Video generation works, but camera LoRAs are incompatible (size mismatch)
- Docker image: `88dreams/gen3c-runpod:v55g`

**API Integration (`ltx-API` branch) - CURRENT:**
- Exploring Lightricks official API or alternative video APIs
- Avoids local model hosting memory issues
- **Next steps**: Define API integration approach

#### Git Branches

```
main
├── ltx-native     # Native pipeline (OOM issues) - pushed
├── ltx-diffuser   # Diffusers pipeline (no LoRAs) - pushed  
└── ltx-API        # API integration - CURRENT
```

#### Key Technical Learnings

1. **Model Size Mismatch**: HuggingFace `Lightricks/LTX-Video` (2B) ≠ `LTX-2` (19B)
2. **FP8 Upcasting**: Native pipeline upcasts FP8 to BF16 during inference (~38GB for transformer alone)
3. **Memory Requirements**: Full 19B pipeline needs >150GB VRAM (transformer + Gemma + activations)
4. **Diffusers CPU Offload**: Works well but limited to 2B model currently

#### Files Modified

| File | Changes |
|------|---------|
| `runpod/gen3c/handler_unified.py` | Added LTX-2 handlers, runtime path detection, graceful LoRA error handling |
| `runpod/gen3c/start_unified.sh` | Added LTX-2 symlink creation, HuggingFace cache setup |
| `runpod/runpod_client.py` | Added `LTX2ServerlessClient`, fixed `video_s3_url` extraction |
| `generators/ltx2.py` | Generator module for Gradio integration |
| `app_sidebar.py` | LTX-2 UI tab with camera motion selection |

#### Storage Layout (RunPod Network Volume)

```
/runpod-volume/
├── ltx2/                    # LTX-2 files (NOT under checkpoints/)
│   └── loras/               # Camera control LoRAs
│       ├── LTX-2-19b-LoRA-Camera-Control-Dolly-Out.safetensors
│       ├── LTX-2-19b-LoRA-Camera-Control-Dolly-In.safetensors
│       ├── LTX-2-19b-LoRA-Camera-Control-Dolly-Left.safetensors
│       ├── LTX-2-19b-LoRA-Camera-Control-Dolly-Right.safetensors
│       ├── LTX-2-19b-LoRA-Camera-Control-Jib-Up.safetensors
│       ├── LTX-2-19b-LoRA-Camera-Control-Jib-Down.safetensors
│       └── LTX-2-19b-LoRA-Camera-Control-Static.safetensors
├── huggingface/             # HuggingFace cache (diffusers downloads here)
├── Gen3C-Cosmos-7B/         # Gen3C model
├── sharp/                   # SHARP checkpoint
├── trellis/                 # TRELLIS checkpoint
└── lyra/                    # Lyra checkpoint
```

**Note**: Native pipeline checkpoints (gemma, fp8 model) can be deleted - not used by diffusers.

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

### LTX-2 Integration

| File | Purpose |
|------|---------|
| `runpod/gen3c/handler_unified.py` | LTX-2 handlers with runtime path detection |
| `runpod/gen3c/start_unified.sh` | Symlink creation for /runpod-volume/ltx2 |
| `runpod/gen3c/Dockerfile.ltx2.diffusers` | Diffusers-based Docker build |
| `runpod/runpod_client.py` | `LTX2ServerlessClient` with S3 download |
| `generators/ltx2.py` | Generator module |
| `app_sidebar.py` | UI tab for LTX-2 |

### Handler Key Functions (LTX-2)

```python
# Runtime path detection (runs when job arrives, not at module load)
def get_ltx2_checkpoint_dir() -> str

# Camera LoRA path lookup
def get_ltx2_camera_lora_path(motion: str) -> Optional[str]

# Pipeline loading with CPU offload
def load_ltx2_model(camera_lora_path: Optional[str] = None)

# Video generation
def run_ltx2(input_image_path, output_name, prompt, ...) -> Dict

# Job handler
def handle_ltx2(job, job_input, input_path, return_base64) -> Dict
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

### 1. LTX-2 Camera LoRAs 🔧 IN PROGRESS
- **Issue**: Diffusers 2B model incompatible with 19B camera LoRAs
- **Workaround**: Video generates without camera control (prompt-driven only)
- **Next**: Explore API integration on `ltx-API` branch

### 2. SEVA Integration ⏸️ BLOCKED
- **Issue**: HuggingFace model access required
- **Action**: Request access at https://huggingface.co/stabilityai/stable-virtual-camera

### 3. 2DGS Pipeline ✅ COMPLETE
- Successfully generates meshes from Gen3C video
- 180° X-axis rotation applied automatically

---

## Debugging Tips

### LTX-2 Issues

1. **LoRA not found**: Check `/runpod-volume/ltx2/loras/` exists (not `/runpod-volume/checkpoints/ltx2/`)
2. **OOM with native pipeline**: Use diffusers branch instead
3. **LoRA size mismatch**: Expected - diffusers uses 2B model, LoRAs need 19B
4. **HuggingFace download fails**: Check HF_HOME points to network volume

### Logs to Check

```bash
# In handler logs, look for:
[LTX2] Checkpoint dir: /workspace/checkpoints/ltx2  # Should be this, NOT /runpod-volume/checkpoints/ltx2
[LTX2] Looking for camera LoRA at: ...
[LTX2] Camera LoRA incompatible with this model version  # Expected with diffusers
[LTX2] Continuing without camera control
```

---

## Contact / Resources

- **RunPod Console**: https://www.runpod.io/console
- **Docker Hub**: https://hub.docker.com/u/88dreams
- **S3 Bucket**: arkrunr (us-west-1)
- **Git Branch**: `ltx-API` (current), `ltx-diffuser`, `ltx-native`

---

*Last Updated: January 15, 2026 (LTX-2 diffusers pipeline complete, moving to API integration)*
