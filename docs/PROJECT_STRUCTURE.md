# ARKRUNR WORLDS - Project Structure

**Last Updated:** January 17, 2026

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
├── utils/                  # Shared utilities (including TagManager)
├── docs/                   # Documentation
├── config/                 # Configuration files
└── outputs/                # Generated output files
```

---

## Core Application Files

### `app_sidebar.py`
**Main entry point** - The primary Gradio application with sidebar navigation (~3,450 lines).

- **Purpose**: Unified UI for all 3D generation models
- **Key Features**:
  - Sidebar navigation with INPUT, CREATE, BROWSE, TOOLS sections
  - Image input with auto-scaling and automatic output naming
  - Model selection (SHARP, Gen3C, Lyra, Trellis, Hunyuan, **LTX-2**, **2DGS**)
  - **Gallery Tab**: Browse and tag outputs from all models
  - **Post-Generation Tagging**: Tag results after generation for each model
  - **LTX-2**: 4-video preview grid, multi-motion selection, Play All button
  - **2DGS**: 4-video preview, sort/filter controls, tag filtering
  - Settings page for API credentials (RunPod, LTX, AWS)
  - Help documentation
  - Update tracker for model versions
  - Encode parameters in filename option
- **Dependencies**: 
  - `handlers/*` for generation business logic
  - `help/*` for documentation strings
  - `generators/*` for model execution
  - `ui/tabs/*` for tab-specific UI components
  - `ui/styles.py` for CSS and JavaScript
  - `runpod/runpod_client.py` for API calls
  - `scripts/experiment_logger.py` for filename encoding
  - `utils/*` for system metrics, version tracking, and **tagging**

### `gradio_app.py`
Legacy Gradio application (older version, kept for reference).

---

## Handlers (`handlers/`)

Business logic layer between UI and generators. Extracted from `app_sidebar.py` for maintainability.

### `handlers/generation_handlers.py` (~1,240 lines)

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
| `list_all_videos()` | List videos with sort/filter/tag support for 2DGS |

**Note**: LTX-2 and 2DGS multi-video are handled in `app_sidebar.py`.

---

## Utilities (`utils/`)

Shared utility modules for the application.

### `utils/tag_manager.py` (~750 lines) ✅ NEW

**TagManager** - Thread-safe tag storage and management system.

| Feature | Description |
|---------|-------------|
| **Storage** | JSON file at `/srv/searidge_share/outputs/tags.json` |
| **Thread-Safe** | Uses `threading.Lock` for concurrent access |
| **Predefined Tags** | approved, favorite, best-take, review, bad, delete |
| **Custom Tags** | User-defined free-form tags |
| **Delete Handling** | Marks files for deletion, removes from disk |

**Key Methods**:
```python
class TagManager:
    get_tags(file_path) -> List[str]
    set_tags(file_path, tags, model=None)
    add_tag(file_path, tag)
    remove_tag(file_path, tag)
    clear_tags(file_path)
    list_files_by_tag(tag) -> List[str]
    list_files_by_model(model) -> List[str]
    get_all_tags_flat() -> List[str]
    is_marked_for_delete(file_path) -> bool
    clear_tags_for_files(file_paths) -> int
```

**Usage**:
```python
from utils.tag_manager import get_tag_manager

tag_manager = get_tag_manager()
tag_manager.set_tags("/path/to/video.mp4", ["approved", "favorite"], model="LTX-2")
tags = tag_manager.get_tags("/path/to/video.mp4")
```

### `utils/version_tracker.py`
Tracks model/Docker image versions for update notifications.

### `utils/system_metrics.py`
System resource monitoring (CPU, memory, GPU).

### `utils/image_utils.py`
Image processing utilities.

---

## UI Components (`ui/`)

### `ui/tabs/gallery_tab.py` (~695 lines) ✅ NEW

**Gallery Tab** - Browse, filter, and tag render outputs from all models.

| Component | Purpose |
|-----------|---------|
| **Model Filter** | Dropdown to filter by model (Gen3C, LTX-2, etc.) |
| **Tag Filter** | Multi-select dropdown to filter by tags |
| **File List** | Scrollable radio list of files with tags displayed |
| **Preview** | Video or 3D model preview (400px height) |
| **Tag Checkboxes** | Two-column layout: Approved/Favorite/Best-take, Review/Bad/Delete |
| **Custom Tag** | Text input with checkbox for custom tags |
| **Clear Tags** | Single file and bulk (all displayed) clear |
| **Delete Confirmation** | Yes/No dialog before file deletion |

**Key Functions**:
```python
def scan_output_files(model_filter, tag_filter, limit, include_deleted) -> List[Dict]
def format_file_list(files) -> List[Tuple[str, str]]
def create_gallery_tab() -> Dict[str, Any]
```

### `ui/tabs/create_tab.py` (~360 lines)

2DGS pipeline UI components.

- 4-video preview grid with filename labels
- Video list with sort (date/filename/model/tagged first), limit slider
- **Tag filter dropdown** for filtering videos by tag
- Play All button
- Output naming with parameter encoding

### `ui/tabs/__init__.py`

Exports for UI tab modules:
```python
from ui.tabs.create_tab import create_2dgs_tab
from ui.tabs.gallery_tab import create_gallery_tab

__all__ = ["create_2dgs_tab", "create_gallery_tab"]
```

### `ui/styles.py`

CSS and JavaScript for:
- Video preview grids
- Play All button functionality
- Gallery file list scrolling
- Responsive layouts

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
| `ltx2.py` | LTX-2 (Lightricks) | Video generation via official API ✅ |
| `seva.py` | SEVA (Stability AI) | Novel view synthesis (blocked - HF access) |

### Common Pattern
Each generator module typically contains:
```python
def run_<model>_local(...)     # Local execution (if supported)
def run_<model>_runpod(...)    # RunPod serverless execution
```

---

## RunPod Infrastructure (`runpod/`)

### Client Library

#### `runpod/runpod_client.py` (~3,800 lines)
**Primary client** for RunPod API interactions.

- **Classes**:
  - `RunPodGEN3CClient` - Pod-based API client
  - `RunPodServerlessClient` - Serverless endpoint client
  - `UnifiedServerlessClient` - Multi-model serverless client
  - `TwoDGSPipelineClient` - 2DGS multi-video pipeline client ✅
  - `LTXAPIClient` - Official LTX API client ✅
  - `LTX2ServerlessClient` - Legacy RunPod client (deprecated)
  - `SEVAServerlessClient` - SEVA client (blocked)

- **LTX API Client Features**:
  - Direct API calls to https://api.ltx.video
  - Image-to-video and text-to-video generation
  - Models: ltx-2-pro (best quality), ltx-2-fast
  - Up to 4K @ 50fps output
  - Camera motion via prompt descriptions

- **2DGS Pipeline Client Features**:
  - `submit_multi_video_job()` - Accept array of video URLs + output_name
  - `get_status()` with quality_stats (depth coverage, frame counts)
  - Checkpoint restore support

### Docker Images

#### `runpod/gen3c/` - Unified Multi-Model Image

| File | Purpose |
|------|---------|
| `Dockerfile.unified` | Main Docker image (base) |
| `Dockerfile.patch` | Incremental patch Dockerfile |
| `Dockerfile.ltx2.diffusers` | LTX-2 diffusers build |
| `handler_unified.py` | Serverless handler (~2,200 lines) |
| `start_unified.sh` | Container startup script |
| `lyra_inference.py` | Lyra-specific inference logic |
| `sugar_inference.py` | SuGaR mesh extraction logic |
| `trellis_inference.py` | TRELLIS inference logic |

#### `runpod/hunyuan/` - Hunyuan3D Image
| File | Purpose |
|------|---------|
| `Dockerfile` | Hunyuan-specific image |
| `handler_hunyuan.py` | Serverless handler |
| `start_hunyuan.sh` | Startup script |

#### `runpod/2dgs-pipeline/` - ViPE + 2DGS Pipeline (Multi-Video)
| File | Purpose |
|------|---------|
| `Dockerfile` | Combined ViPE + 2DGS image |
| `handler.py` | Serverless handler (~1,370 lines) with checkpointing |
| `multi_video_merge.py` | Pose alignment + frame merging |
| `vipe_to_2dgs.py` | ViPE → COLMAP converter |
| `init_points_from_depth.py` | Point cloud generator |
| `README.md` | API documentation |

**Multi-Video Features**:
- Accept array of `video_urls` (up to 4)
- Frame 0 alignment for consistent poses
- `depth_threshold` parameter for quality filtering
- `output_name` parameter for custom S3 naming
- Checkpoint save/restore to S3
- Returns `quality_stats` in response

#### `runpod/seva/` - SEVA (Blocked)
| File | Purpose |
|------|---------|
| `Dockerfile` | SEVA Docker image |
| `handler_seva.py` | Serverless handler |
| `start_seva.sh` | Startup script |

---

## Scripts (`scripts/`)

| File | Purpose |
|------|---------|
| `experiment_logger.py` | CSV logging + parameter-encoded filenames ✅ |
| `cleanup_mesh.py` | Mesh decimation and cleanup |
| `convert_lyra_ply.py` | Convert Lyra's PyTorch PLY to standard PLY |
| `blender_import_pointcloud.py` | Blender script for PLY visualization |
| `benchmark_models.py` | Performance benchmarking |
| `init_points_from_depth.py` | Generate point cloud from ViPE depth maps |
| `vipe_to_2dgs.py` | Convert ViPE output to 2DGS COLMAP format |
| `test_2dgs_pipeline.py` | Test script for 2DGS endpoint |
| `test_vipe.py` | Test script for ViPE endpoint |
| `test_mesh_rotation.py` | Test 180° X-axis rotation on mesh files |

### `experiment_logger.py` - Parameter Encoding Functions

| Function | Output Example |
|----------|----------------|
| `sharp_param_filename()` | `sharp_CBGB1_g7.5_s50.ply` |
| `sharp_video_param_filename()` | `sharp_CBGB1_g7.5_s50_lo050_zf025.mp4` |
| `gen3c_param_filename()` | `gen3c_CBGB1_f121_torb_md10_g10_s42.mp4` |
| `ltx2_param_filename()` | `ltx2_CBGB1_mpro_d6_1080p_dollyout.mp4` |
| `twodgs_param_filename()` | `2dgs_CBGB1_dolly_orbit.glb` |
| `mesh_extract_param_filename()` | `CBGB1_processed.glb` |

---

## Documentation (`docs/`)

| File | Purpose |
|------|---------|
| `PROJECT_STRUCTURE.md` | This file - project structure overview |
| `CONVERSATION_HANDOFF.md` | Session handoff documentation |
| `STABLE_VIRTUAL_CAMERA_PLAN.md` | SEVA integration plan |
| `VIPE_SETUP_GUIDE.md` | ViPE installation and usage |
| `2DGS_PLAN.md` | 2DGS pipeline architecture |
| `NERF_VS_GAUSSIAN_SPLATTING_RESEARCH.md` | Research notes |
| `GAUSSIAN_SPLAT_TO_MESH_GUIDE.md` | Mesh extraction guide |

---

## Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        app_sidebar.py                           │
│                     (Gradio UI - Local)                         │
│   Features: Gallery, Tagging, 4-video preview, Play All         │
└─────────────────────────────────────────────────────────────────┘
                    │                          │
                    ▼                          ▼
┌───────────────────────────────┐   ┌─────────────────────────────┐
│   runpod/runpod_client.py     │   │   LTX API (External)        │
│  (RunPod API - GPU inference) │   │  https://api.ltx.video      │
└───────────────────────────────┘   └─────────────────────────────┘
                    │
                    ▼ (HTTPS API)
┌─────────────────────────────────────────────────────────────────┐
│                  RunPod Serverless Endpoints                    │
│                    (GPU Cloud - Remote)                         │
├─────────────────────────────────────────────────────────────────┤
│  handler_unified.py  →  Gen3C, SHARP, Lyra, TRELLIS            │
│  handler_hunyuan.py  →  Hunyuan3D                               │
│  handler.py (2dgs)   →  ViPE + 2DGS (multi-video + checkpoint)  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        S3 Bucket (arkrunr)                      │
│                                                                 │
│  MediaContent/inputs/ltx2/   - Input images                     │
│  MediaContent/outputs/ltx2/  - LTX-2 videos                     │
│  MediaContent/outputs/gen3c/ - Gen3C videos                     │
│  MediaContent/outputs/sharp/ - SHARP outputs                    │
│  MediaContent/2dgs-pipeline/ - 2DGS checkpoints + outputs       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Docker Images on Docker Hub

| Image | Models | Current Version | Notes |
|-------|--------|-----------------|-------|
| `88dreams/gen3c-runpod` | Gen3C, Lyra, SHARP, SuGaR, TRELLIS.2 | **v55g** | Unified handler |
| `88dreams/hunyuan-runpod` | Hunyuan3D | v10 | Stable |
| `88dreams/2dgs-pipeline` | ViPE + 2DGS (multi-video→mesh) | **v13** | ✅ Checkpointing |
| `88dreams/seva-runpod` | SEVA (camera control video) | v1 | ⏸️ Blocked (HF access) |

**Note**: LTX-2 uses official API (https://api.ltx.video) - no Docker image needed.

---

## RunPod Serverless Endpoints

| Endpoint Name | Image | Purpose |
|---------------|-------|---------|
| `gen3c-serverless` | gen3c-runpod:**v55g** | Multi-model (Gen3C, Lyra, SHARP, TRELLIS.2) |
| `hunyuan-serverless` | hunyuan-runpod:v10 | Hunyuan3D mesh generation |
| `2dgs-serverless` | 2dgs-pipeline:**v13** | Multi-video to mesh (ViPE + 2DGS) |

**Note**: LTX-2 uses official Lightricks API - no RunPod endpoint.

---

## Storage Layout

### RunPod Network Volume (`/runpod-volume/`)

```
/runpod-volume/
├── ltx2/                           # LTX-2 (NOT under checkpoints!)
│   └── loras/                      # Camera control LoRAs (19B model)
├── huggingface/                    # HuggingFace cache
│   └── hub/                        # Downloaded models
├── Gen3C-Cosmos-7B/                # Gen3C model
├── sharp/                          # SHARP checkpoint
├── trellis/                        # TRELLIS checkpoint
└── lyra/                           # Lyra checkpoint
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
├── logs/       # Experiment CSV logs
└── tags.json   # Tag database (TagManager) ✅ NEW
```

---

## Running the Application

```bash
# Start the Gradio UI
python app_sidebar.py

# Access at http://localhost:5684
```

---

## Environment Setup

### AWS Credentials (`.env` file - not in repo)
```
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
```

### RunPod API Key
```bash
export RUNPOD_API_KEY="your_key"
```

Or configure in UI Settings page.

---

## Key File Relationships

### UI → Handlers → Generators → APIs
```
app_sidebar.py (UI + LTX-2 + Gallery)
    ├── utils/tag_manager.py (TagManager)
    ├── ui/tabs/gallery_tab.py (Gallery UI)
    └── handlers/generation_handlers.py (other models)
            └── generators/ltx2.py (LTX-2 logic)
                    └── runpod/runpod_client.py (LTXAPIClient)
                            └── LTX API (https://api.ltx.video)
```

### Tagging Data Flow
```
User clicks tag checkbox in Gallery/Post-generation
    └── gallery_tab.py / app_sidebar.py event handler
            └── TagManager.set_tags(file_path, tags)
                    └── /srv/searidge_share/outputs/tags.json
```

### Docker Build Chain
```
Dockerfile.unified (base image)
    └── Dockerfile.ltx2.diffusers (LTX-2 additions)
            └── 88dreams/gen3c-runpod:v55g (Docker Hub)
```

---

*Last Updated: January 17, 2026 (Tagging system, Gallery tab, 2DGS v13)*
