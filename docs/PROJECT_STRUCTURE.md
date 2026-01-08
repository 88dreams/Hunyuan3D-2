# ARKRUNR WORLDS - Project Structure

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
**Main entry point** - The primary Gradio application with sidebar navigation (~2,070 lines).

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
| `placeholder_tabs.py` | Placeholder tabs for future features |

### `ui/styles.py`
CSS styles and color palette for the Gradio UI.

---

## RunPod Infrastructure (`runpod/`)

### Client Library

#### `runpod/runpod_client.py`
**Primary client** for RunPod API interactions.

- **Classes**:
  - `RunPodGEN3CClient` - Pod-based API client
  - `RunPodServerlessClient` - Serverless endpoint client
  - `UnifiedServerlessClient` - Multi-model serverless client (recommended)
- **Key Methods**:
  - `submit_*_job()` - Submit jobs for each model
  - `generate_*_sync()` - Synchronous job execution with polling
  - `wait_for_completion()` - Poll for job completion
- **Features**:
  - S3 upload/download for large files
  - Base64 encoding for small files
  - Automatic timeout handling

### Docker Images (`runpod/gen3c/`)

#### `Dockerfile.unified`
**Main Docker image** containing all models:
- Gen3C, Lyra, SHARP, TRELLIS.2, SuGaR
- Base: `nvcr.io/nvidia/pytorch:24.10-py3`
- Conda environment: `cosmos-predict1`
- Current version: v50

#### `Dockerfile.patch`
Incremental patch Dockerfile for quick updates (generic, reusable).

#### Key Files:
| File | Purpose |
|------|---------|
| `handler_unified.py` | Serverless handler for all models |
| `start_unified.sh` | Container startup script |
| `lyra_inference.py` | Lyra-specific inference logic |
| `sugar_inference.py` | SuGaR mesh extraction logic |
| `server_unified.py` | Pod-mode API server |

### Hunyuan Docker (`runpod/hunyuan/`)
Separate Docker image for Hunyuan3D:
- `Dockerfile` - Hunyuan-specific image
- `handler_hunyuan.py` - Serverless handler
- `start_hunyuan.sh` - Startup script

### Trellis Docker (`runpod/trellis/`)
Separate Docker image for TRELLIS.2:
- `Dockerfile` - Trellis-specific image
- `handler_trellis.py` - Serverless handler
- `trellis_inference.py` - Inference logic

---

## Scripts (`scripts/`)

| File | Purpose |
|------|---------|
| `experiment_logger.py` | CSV logging for model parameters and results |
| `cleanup_mesh.py` | Mesh decimation and cleanup |
| `convert_lyra_ply.py` | Convert Lyra's PyTorch PLY to standard PLY |
| `blender_import_pointcloud.py` | Blender script for PLY visualization |
| `benchmark_models.py` | Performance benchmarking |

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

