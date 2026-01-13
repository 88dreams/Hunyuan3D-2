# ARKRUNR WORLDS - Project Structure

**Last Updated:** January 12, 2026

This document describes the structure of the ARKRUNR WORLDS project, a multi-model 3D generation system built on top of Hunyuan3D-2.

## Overview

The project provides a unified Gradio UI for multiple 3D generation models, with execution on RunPod Serverless for GPU-intensive tasks.

```
Hunyuan3D-2-Fork/
├── app_sidebar.py          # Main Gradio application (entry point)
├── handlers/               # Generation handler functions (business logic)
├── help/                   # Help documentation strings
├── generators/             # Model-specific generation logic
├── ui/                     # UI components and tabs
├── runpod/                 # RunPod serverless infrastructure
├── scripts/                # Utility scripts
├── utils/                  # Shared utilities
├── docs/                   # Documentation
├── config/                 # Configuration files
└── outputs/                # Generated output files
```

---

## Core Application Files

### `app_sidebar.py`
**Main entry point** - The primary Gradio application with sidebar navigation (~2,380 lines).

- **Purpose**: Unified UI for all 3D generation models
- **Key Features**:
  - Sidebar navigation with INPUT, CREATE, TOOLS sections
  - Image input with auto-scaling
  - Model selection (SHARP, Gen3C, Lyra, Trellis, Hunyuan)
  - Settings page for RunPod credentials
  - Help documentation
  - Update tracker for model versions
- **Dependencies**: 
  - `handlers/*` for generation business logic
  - `help/*` for documentation strings
  - `generators/*` for model execution
  - `ui/components/*` for UI elements
  - `runpod/runpod_client.py` for API calls
  - `utils/*` for system metrics and version tracking

### `gradio_app.py`
Legacy Gradio application (older version, kept for reference).

---

## Handlers (`handlers/`)

Business logic layer between UI and generators. Extracted from `app_sidebar.py` for maintainability.

### `handlers/generation_handlers.py` (~886 lines)

| Function | Purpose |
|----------|---------|
| `handle_sharp_generation()` | SHARP 3DGS generation with video rendering |
| `handle_gen3c_generation()` | Gen3C video generation |
| `handle_lyra_generation()` | Lyra 3DGS/4DGS generation |
| `handle_trellis_generation()` | TRELLIS.2 3D generation |
| `handle_hunyuan_generation()` | Hunyuan3D mesh generation |
| `handle_mesh_extraction()` | PLY to mesh conversion |
| `handle_mesh_analyze()` | Mesh statistics analysis |
| `handle_mesh_cleanup()` | Mesh decimation and cleanup |

**Responsibilities**:
- Image scaling and preprocessing
- Parameter validation and encoding
- Calling appropriate generator (local or RunPod)
- Experiment logging to CSV
- Cleanup of temporary files

---

## Help Documentation (`help/`)

Help text and documentation strings for the UI. Extracted from `app_sidebar.py` for easier maintenance.

### `help/documentation.py` (~514 lines)

| Constant | Content |
|----------|---------|
| `SHARP_HELP` | SHARP model documentation and settings |
| `GEN3C_HELP` | Gen3C video generation documentation |
| `LYRA_HELP` | Lyra 3DGS/4DGS documentation |
| `TRELLIS_HELP` | TRELLIS.2 documentation |
| `HUNYUAN_HELP` | Hunyuan3D documentation with memory options |
| `MESH_HELP` | Mesh extraction documentation |
| `MESH_CLEANUP_HELP` | Mesh cleanup detailed documentation |
| `GENERAL_TIPS_HELP` | Workflow recommendations and tips |

---

## Generators (`generators/`)

Model-specific Python modules that handle local and RunPod execution.

| File | Model | Description |
|------|-------|-------------|
| `sharp.py` | SHARP (Apple) | Single-image to 3DGS PLY + video rendering |
| `gen3c.py` | Gen3C (NVIDIA) | Image to video generation with camera control |
| `lyra.py` | Lyra (NVIDIA) | Image/video to 3D/4D Gaussian Splatting |
| `trellis.py` | TRELLIS.2 (Microsoft) | Image to 3D with O-Voxel representation |
| `hunyuan.py` | Hunyuan3D (Tencent) | Image to 3D mesh (GLB) |
| `sugar.py` | SuGaR | 3DGS to mesh conversion (Poisson reconstruction) |
| `vipe_integration.py` | ViPE (NVIDIA) | Video pose estimation integration |

### Common Pattern
Each generator module typically contains:
```python
def run_<model>_local(...)     # Local execution (if supported)
def run_<model>_runpod(...)    # RunPod serverless execution
```

### Dependencies
- All generators import from `runpod/runpod_client.py`
- Some use `scripts/experiment_logger.py` for CSV logging

---

## UI Components (`ui/`)

### `ui/components/`

| File | Purpose |
|------|---------|
| `sidebar.py` | Sidebar navigation component |
| `credentials_manager.py` | RunPod/AWS credential management |

### `ui/tabs/`

Individual tab implementations (legacy, mostly integrated into `app_sidebar.py`):

| File | Tab |
|------|-----|
| `sharp_tab.py` | SHARP generation tab |
| `gen3c_tab.py` | Gen3C video generation tab |
| `lyra_tab.py` | Lyra 3DGS tab |
| `trellis_tab.py` | TRELLIS.2 tab |
| `hunyuan_tab.py` | Hunyuan3D tab |
| `mesh_extraction_tab.py` | Mesh cleanup/extraction tab |
| `create_tab.py` | 2DGS Pipeline tab (video → mesh) |
| `placeholder_tabs.py` | Placeholder tabs for future features |

### `ui/styles.py`
CSS styles and color palette for the Gradio UI.

---

## RunPod Infrastructure (`runpod/`)

### Client Library

#### `runpod/runpod_client.py`
**Primary client** for RunPod API interactions (~2,090 lines).

- **Classes**:
  - `RunPodGEN3CClient` - Pod-based API client
  - `RunPodServerlessClient` - Serverless endpoint client
  - `UnifiedServerlessClient` - Multi-model serverless client (recommended)
  - `TwoDGSPipelineClient` - 2DGS pipeline client (video → mesh)
- **Key Methods**:
  - `submit_*_job()` - Submit jobs for each model
  - `generate_*_sync()` - Synchronous job execution with polling
  - `wait_for_completion()` - Poll for job completion
- **Features**:
  - S3 upload/download for large files
  - Base64 encoding for small files
  - Automatic timeout handling
  - 180° X-axis mesh rotation correction for 2DGS outputs

### Docker Images

#### `runpod/gen3c/` - Unified Multi-Model Image
| File | Purpose |
|------|---------|
| `Dockerfile.unified` | Main Docker image (Gen3C, Lyra, SHARP, TRELLIS.2, SuGaR) |
| `Dockerfile.patch` | Incremental patch Dockerfile for quick updates |
| `handler_unified.py` | Serverless handler for all models |
| `start_unified.sh` | Container startup script |
| `lyra_inference.py` | Lyra-specific inference logic |
| `sugar_inference.py` | SuGaR mesh extraction logic |
| `server_unified.py` | Pod-mode API server |

#### `runpod/hunyuan/` - Hunyuan3D Image
| File | Purpose |
|------|---------|
| `Dockerfile` | Hunyuan-specific image |
| `handler_hunyuan.py` | Serverless handler |
| `start_hunyuan.sh` | Startup script |

#### `runpod/trellis/` - TRELLIS.2 Image
| File | Purpose |
|------|---------|
| `Dockerfile` | Trellis-specific image |
| `handler_trellis.py` | Serverless handler |
| `trellis_inference.py` | Inference logic |

#### `runpod/2dgs-pipeline/` - ViPE + 2DGS Pipeline (NEW)
Combined serverless endpoint for video-to-mesh reconstruction:
| File | Purpose |
|------|---------|
| `Dockerfile` | Combined ViPE + 2DGS image |
| `handler.py` | Serverless handler orchestrating full pipeline |
| `vipe_to_2dgs.py` | Converter: ViPE output → COLMAP format |
| `init_points_from_depth.py` | Generate initial point cloud from depth maps |
| `README.md` | Deployment and usage instructions |

**Pipeline Flow:**
1. ViPE extracts camera poses, depth, intrinsics from video
2. Converter transforms to COLMAP format for 2DGS
3. Point cloud generated from depth maps
4. 2DGS training produces Gaussian splats
5. Mesh extraction via marching cubes

#### `runpod/vipe/` - Standalone ViPE (Reference)
| File | Purpose |
|------|---------|
| `Dockerfile` | ViPE-only image |
| `handler_vipe.py` | Serverless handler |
| `setup_vipe.sh` | Setup script |

#### `runpod/2dgs/` - Standalone 2DGS (Reference)
| File | Purpose |
|------|---------|
| `Dockerfile` | 2DGS-only image |
| `handler_2dgs.py` | Serverless handler |
| `vipe_to_2dgs.py` | Converter script |
| `init_points_from_depth.py` | Point cloud generator |

---

## Scripts (`scripts/`)

| File | Purpose |
|------|---------|
| `experiment_logger.py` | CSV logging for model parameters and results |
| `cleanup_mesh.py` | Mesh decimation and cleanup |
| `convert_lyra_ply.py` | Convert Lyra's PyTorch PLY to standard PLY |
| `blender_import_pointcloud.py` | Blender script for PLY visualization |
| `benchmark_models.py` | Performance benchmarking |
| `init_points_from_depth.py` | Generate point cloud from ViPE depth maps |
| `vipe_to_2dgs.py` | Convert ViPE output to 2DGS COLMAP format |
| `test_2dgs_pipeline.py` | Test script for 2dgs-pipeline endpoint |
| `test_vipe.py` | Test script for ViPE endpoint |
| `test_mesh_rotation.py` | Test 180° X-axis rotation on mesh files |

---

## Utilities (`utils/`)

| File | Purpose |
|------|---------|
| `system_metrics.py` | CPU/GPU/Memory monitoring |
| `version_tracker.py` | Track upstream and deployed model versions |
| `image_utils.py` | Image processing utilities |

---

## Configuration (`config/`)

| File | Purpose |
|------|---------|
| `paths.py` | Path configuration |
| `multi_system.yaml` | Multi-system configuration |
| `monitoring/` | Grafana/Prometheus configs |

---

## Documentation (`docs/`)

| File | Purpose |
|------|---------|
| `PROJECT_STRUCTURE.md` | This file - project structure overview |
| `SERVERLESS_DEPLOYMENT_MORNING.md` | Stage-pipeline deployment guide |
| `VIPE_SETUP_GUIDE.md` | ViPE installation and usage |
| `SV3D_SETUP_GUIDE.md` | SV3D setup instructions |
| `INTERIOR_RECONSTRUCTION_APPROACHES.md` | Research on interior 3D reconstruction |
| `S3_UPLOAD_ACCESS.md` | S3 configuration for large files |
| `CONVERSATION_HANDOFF.md` | Session handoff documentation |

---

## Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        app_sidebar.py                           │
│                     (Gradio UI - Local)                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      generators/*.py                            │
│              (Model-specific logic - Local)                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   runpod/runpod_client.py                       │
│                (API Client - Local → RunPod)                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ (HTTPS API)
┌─────────────────────────────────────────────────────────────────┐
│                  RunPod Serverless Endpoint                     │
│                   (GPU Cloud - Remote)                          │
├─────────────────────────────────────────────────────────────────┤
│  handler_unified.py / handler_hunyuan.py / handler_trellis.py   │
│  handler.py (2dgs-pipeline)                                    │
│                    (Serverless Handlers)                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        S3 Bucket                                │
│              (Large file storage - arkrunr)                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key File Relationships

### UI → Handlers → Generators → RunPod
```
app_sidebar.py (UI components + event bindings)
    └── handlers/generation_handlers.py (business logic)
            └── generators/sharp.py (model-specific logic)
                    └── runpod/runpod_client.py (UnifiedServerlessClient)
                            └── RunPod API → handler_unified.py
```

### Docker Build Chain
```
Dockerfile.unified (base image)
    └── Dockerfile.patch (incremental updates)
            └── 88dreams/gen3c-runpod:v50 (Docker Hub)
```

### 2DGS Pipeline Build
```
runpod/2dgs-pipeline/Dockerfile
    └── 88dreams/2dgs-pipeline:v15 (Docker Hub)
```

### Configuration Flow
```
.env (AWS credentials - not in repo)
    └── runpod/runpod_client.py (reads for S3)
    └── handler_unified.py (reads for S3 upload)
```

---

## Running the Application

```bash
# Start the Gradio UI
python app_sidebar.py

# Access at http://localhost:7860
```

## Docker Images on Docker Hub

| Image | Models | Current Version |
|-------|--------|-----------------|
| `88dreams/gen3c-runpod` | Gen3C, Lyra, SHARP, SuGaR | v50 |
| `88dreams/trellis-runpod` | TRELLIS.2 | v10 |
| `88dreams/hunyuan-runpod` | Hunyuan3D | v10 |
| `88dreams/2dgs-pipeline` | ViPE + 2DGS (video→mesh) | v20 ✅ |

## RunPod Serverless Endpoints

| Endpoint Name | Image | Purpose |
|---------------|-------|---------|
| `gen3c-serverless` | gen3c-runpod | Multi-model (Gen3C, Lyra, SHARP) |
| `trellis-serverless` | trellis-runpod | TRELLIS.2 3D generation |
| `hunyuan-serverless` | hunyuan-runpod | Hunyuan3D mesh generation |
| `2dgs-serverless` | 2dgs-pipeline:v20 | Video to mesh (ViPE + 2DGS) ✅ |