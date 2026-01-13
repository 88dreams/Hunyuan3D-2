# Stable Virtual Camera (SEVA) Integration Plan

**Date**: January 13, 2026  
**Status**: Infrastructure Complete - Ready for Deployment  
**Priority**: High

---

## Overview

**Stable Virtual Camera (SEVA)** is Stability AI's 1.3B parameter diffusion model for Novel View Synthesis (NVS). It generates 3D-consistent novel views from input images with precise camera trajectory control.

### Key Capabilities
- Generate 3D videos from 1-32 input images
- User-defined camera trajectories OR 14 preset paths
- Up to 1,000 frames with 3D consistency
- Seamless loop closure
- Multiple aspect ratios (1:1, 9:16, 16:9)

### Use Case for ARKRUNR WORLDS
- **Input**: Single 2D image (e.g., AI-generated image, photo)
- **Output**: Video showing camera movement (pan, tilt, orbit, dolly)
- **Goal**: Create source video for 2DGS pipeline → 3D mesh

---

## Technical Specifications

| Spec | Value |
|------|-------|
| **Model Size** | 1.3B parameters |
| **Python** | ≥ 3.10 |
| **PyTorch** | ≥ 2.6.0 |
| **VRAM (estimated)** | 16-24GB (fp16) |
| **License** | Non-Commercial |
| **Output** | Video (up to 1000 frames) |

### Repository
- **GitHub**: https://github.com/Stability-AI/stable-virtual-camera
- **HuggingFace**: https://huggingface.co/stabilityai/stable-virtual-camera
- **Model Weights**: Requires HuggingFace authentication + access request

---

## Camera Trajectory Options

### Preset Trajectories (14 options)
| Trajectory | Description |
|------------|-------------|
| `orbit` | 360° rotation around subject |
| `pan` | Horizontal camera movement |
| `tilt` | Vertical camera angle change |
| `spiral` | Spiral path around subject |
| `zoom-out` | Camera moves backward |
| `dolly zoom-out` | Vertigo/Hitchcock effect |
| `arc` | Curved path |
| `crane` | Vertical + horizontal movement |
| And more... | |

### Custom Trajectories
Define camera poses using:
- **C2W Matrix** (Camera-to-World): 4×4 transformation matrix
- **Intrinsics Matrix (K)**: Focal length, principal point

```python
# Example: Camera moves up 0.5 units, tilts down 15°
import numpy as np

# Base pose (looking forward)
c2w_base = np.eye(4)

# Translate up (Y axis)
c2w_translated = c2w_base.copy()
c2w_translated[1, 3] = 0.5  # Move up 0.5 units

# Rotate (tilt down 15°)
angle = np.radians(-15)
rotation = np.array([
    [1, 0, 0, 0],
    [0, np.cos(angle), -np.sin(angle), 0],
    [0, np.sin(angle), np.cos(angle), 0],
    [0, 0, 0, 1]
])
c2w_final = rotation @ c2w_translated
```

---

## Integration Architecture

### Option A: RunPod Serverless (Recommended)
```
┌─────────────────────┐     ┌──────────────────────┐     ┌─────────────────┐
│   Gradio UI         │────▶│  RunPod Serverless   │────▶│   S3 Bucket     │
│   (app_sidebar.py)  │     │  (SEVA Handler)      │     │   (arkrunr)     │
└─────────────────────┘     └──────────────────────┘     └─────────────────┘
         │                           │
         │                           ▼
         │                  ┌──────────────────────┐
         │                  │  Output Video (MP4)  │
         │                  └──────────────────────┘
         │                           │
         ▼                           ▼
┌─────────────────────────────────────────────────────────┐
│                    2DGS Pipeline                        │
│              (Video → 3D Mesh)                          │
└─────────────────────────────────────────────────────────┘
```

### Option B: Local Execution
- Requires RTX 4090 (24GB) or better
- Not recommended for your current setup

---

## Implementation Plan

### Phase 1: Docker Image Creation
**Goal**: Create RunPod-compatible Docker image

```dockerfile
# runpod/seva/Dockerfile
FROM runpod/pytorch:2.6.0-py3.10-cuda12.4-devel-ubuntu22.04

# Install SEVA
RUN git clone --recursive https://github.com/Stability-AI/stable-virtual-camera /app/seva
WORKDIR /app/seva
RUN pip install -e .

# Install additional dependencies
RUN pip install boto3 runpod

# Copy handler
COPY handler_seva.py /app/handler.py

# Set entrypoint
CMD ["python", "-u", "/app/handler.py"]
```

### Phase 2: Serverless Handler
**Goal**: Create handler for RunPod serverless

```python
# runpod/seva/handler_seva.py (skeleton)
import runpod
import torch
from seva import SEVAModel  # TBD: actual import path

def handler(job):
    """
    Input:
        - image_url: S3 URL or base64 of input image
        - trajectory: preset name OR custom poses
        - duration: video length in seconds
        - fps: frames per second
        
    Output:
        - video_url: S3 URL of generated video
    """
    input_data = job["input"]
    
    # Load model (cached on network volume)
    model = load_model()
    
    # Generate video
    video = model.generate(
        image=input_data["image"],
        trajectory=input_data["trajectory"],
        duration=input_data.get("duration", 5.0),
        fps=input_data.get("fps", 24)
    )
    
    # Upload to S3
    video_url = upload_to_s3(video)
    
    return {"video_url": video_url}

runpod.serverless.start({"handler": handler})
```

### Phase 3: Client Integration
**Goal**: Add SEVA to `runpod_client.py`

```python
class SEVAServerlessClient:
    """Client for Stable Virtual Camera serverless endpoint."""
    
    def __init__(self, api_key, endpoint_id):
        self.api_key = api_key
        self.endpoint_id = endpoint_id
    
    def generate_video(
        self,
        image_path: str,
        trajectory: str = "orbit",
        duration: float = 5.0,
        fps: int = 24,
        custom_poses: list = None
    ) -> str:
        """
        Generate novel view video from single image.
        
        Args:
            image_path: Path to input image
            trajectory: Preset name (orbit, pan, tilt, etc.) or "custom"
            duration: Video length in seconds
            fps: Frames per second
            custom_poses: List of 4x4 C2W matrices (if trajectory="custom")
            
        Returns:
            Path to downloaded video file
        """
        # Implementation here
        pass
```

### Phase 4: UI Integration
**Goal**: Add SEVA tab to Gradio UI

**Location in sidebar**: INPUT section (generates video for 3D pipeline)

**UI Components**:
- Image input (drag & drop)
- Trajectory dropdown (presets)
- Custom trajectory editor (advanced)
- Duration slider (1-30 seconds)
- FPS selector (12, 24, 30)
- Generate button
- Video preview output

---

## Workflow Integration

### Standalone SEVA
```
Image → SEVA → Video (novel views)
```

### SEVA + 2DGS Pipeline
```
Image → SEVA → Video → 2DGS Pipeline → 3D Mesh (GLB)
```

### Full Pipeline (Gen3C alternative)
```
Image → SEVA (camera control) → 2DGS → Mesh
  vs
Image → Gen3C (less camera control) → 2DGS → Mesh
```

---

## File Structure (Created)

```
runpod/seva/
├── Dockerfile              # Docker image definition ✅
├── handler_seva.py         # Serverless handler ✅
├── start_seva.sh           # Startup script ✅
└── README.md               # Deployment instructions ✅

generators/
└── seva.py                 # Generator module ✅
    - run_seva_runpod()
    - create_custom_trajectory()
    - create_orbit_trajectory()
    - generate_camera_video()

runpod/runpod_client.py     # Updated with:
├── SEVAServerlessClient    # Client class ✅
├── SEVAResult              # Result dataclass ✅
└── get_seva_client()       # Helper function ✅

ui/tabs/
└── seva_tab.py             # UI tab (TODO)
```

---

## Tasks Checklist

### Phase 1: Research & Setup
- [ ] Request access to SEVA model on HuggingFace ⚠️ **BLOCKER**
- [ ] Clone repo locally and test basic inference
- [ ] Determine exact VRAM requirements
- [ ] Test preset trajectories
- [ ] Test custom camera pose input

### Phase 2: Docker Image ✅ COMPLETE
- [x] Create Dockerfile (`runpod/seva/Dockerfile`)
- [x] Add HuggingFace token handling (`start_seva.sh`)
- [x] Build Docker image (27.7GB)
- [x] Push to Docker Hub (`88dreams/seva-runpod:v1`) ✅

### Phase 3: RunPod Deployment
- [ ] Create serverless endpoint
- [ ] Configure network volume for model weights
- [ ] Test endpoint health check
- [ ] Test video generation

### Phase 4: Client Integration ✅ COMPLETE
- [x] Add SEVAServerlessClient to runpod_client.py
- [x] Add generate_sync() method with SEVAResult
- [x] S3 upload/download methods
- [x] Create generators/seva.py module
- [x] Add helper functions (create_custom_trajectory, create_orbit_trajectory)

### Phase 5: UI Integration
- [ ] Add SEVA to sidebar navigation
- [ ] Create SEVA generation page
- [ ] Add trajectory presets dropdown
- [ ] Add custom trajectory editor (stretch goal)
- [ ] Connect to 2DGS pipeline

---

## Estimated Timeline

| Phase | Duration | Notes |
|-------|----------|-------|
| Research & Setup | 2-4 hours | Depends on HF access approval |
| Docker Image | 4-6 hours | May need dependency debugging |
| RunPod Deployment | 2-3 hours | Similar to other endpoints |
| Client Integration | 2-3 hours | Pattern established |
| UI Integration | 3-4 hours | Similar to existing tabs |
| **Total** | **13-20 hours** | |

---

## Hardware Requirements

### RunPod Serverless
- **GPU**: RTX A5000 (24GB) or RTX A6000 (48GB)
- **Alternative**: A100 40GB (more expensive but faster)
- **Network Volume**: 10GB for model weights

### Cost Estimate
- A5000: ~$0.22/min active
- A6000: ~$0.38/min active
- Cold start: ~30-60 seconds
- Inference: ~30-120 seconds per video

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| HuggingFace access delayed | Use SV3D as fallback |
| VRAM exceeds 24GB | Use A100 80GB or optimize with fp16 |
| Long inference times | Add progress callbacks |
| Non-commercial license | Fine for research use |

---

## References

- [SEVA GitHub](https://github.com/Stability-AI/stable-virtual-camera)
- [SEVA HuggingFace](https://huggingface.co/stabilityai/stable-virtual-camera)
- [Stability AI Announcement](https://stability.ai/news/introducing-stable-virtual-camera-multi-view-video-generation-with-3d-camera-control)
- [CLI Usage Guide](https://github.com/Stability-AI/stable-virtual-camera/blob/main/docs/CLI_USAGE.md)

---

## Next Steps

1. **Immediate**: Request HuggingFace access for SEVA model
2. **This session**: Start Phase 1 (research & local testing)
3. **Follow-up**: Build Docker image once access granted

---

*Created: January 13, 2026*
