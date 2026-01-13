# LTX-2 Integration Plan

**Date**: January 13, 2026  
**Status**: ✅ Added to Unified Handler - Ready for Build  
**Priority**: High (Open weights - no HF access needed!)

## Update (Jan 13, 2026)

**Decision**: Added LTX-2 to the unified handler instead of a standalone Docker.

**Rationale**:
- No dependency conflicts (LTX-2 works with Python 3.10, CUDA 12.4)
- Same pattern as Trellis (thin layer on unified)
- Single endpoint to manage
- Just needs `pip install diffusers>=0.32.0`

**Files Modified**:
- `runpod/gen3c/handler_unified.py` - Added handle_ltx2(), validate_ltx2(), run_ltx2()
- `runpod/gen3c/Dockerfile.patch.v51` - New patch to add diffusers

**Build Command**:
```bash
cd /home/arkrunr02/Hunyuan3D-2-Fork/runpod/gen3c
docker build -f Dockerfile.patch.v51 -t 88dreams/gen3c-runpod:v51 .
docker push 88dreams/gen3c-runpod:v51
```

---

---

## Overview

**LTX-2** is Lightricks' 19B parameter DiT-based audio-video foundation model. It generates high-quality video from images with camera control via LoRAs.

### Key Advantages Over Existing Options

| Feature | LTX-2 | Gen3C | SEVA |
|---------|-------|-------|------|
| **Weights Access** | ✅ Open | ✅ Open | ❌ Requires HF approval |
| **Resolution** | 4K @ 50fps | ~720p | ~720p |
| **Camera Control** | LoRAs | 3D depth | 6-DOF matrices |
| **Speed** | Fast (8 steps distilled) | Medium | Medium |
| **License** | Community (commercial OK) | Research | Non-commercial |

### Use Case for ARKRUNR WORLDS
- **Input**: Single image (2D)
- **Output**: High-quality video with camera movement
- **Goal**: Feed into 2DGS pipeline → 3D mesh
- **Advantage**: Available NOW (no HF access wait like SEVA)

---

## Technical Specifications

| Spec | Value |
|------|-------|
| **Model Size** | 19B parameters |
| **Python** | ≥ 3.12 |
| **PyTorch** | ~2.7 |
| **CUDA** | ≥ 12.7 |
| **VRAM (FP8)** | ~23GB |
| **VRAM (FP4)** | ~12GB |
| **Output** | Up to 4K, 50fps |

### Model Checkpoints

| Checkpoint | Description | Use Case |
|------------|-------------|----------|
| `ltx-2-19b-dev` | Full model (bf16) | Training/fine-tuning |
| `ltx-2-19b-dev-fp8` | FP8 quantized | **Production inference** |
| `ltx-2-19b-dev-fp4` | FP4 quantized | Memory-constrained |
| `ltx-2-19b-distilled` | 8-step, CFG=1 | **Fastest inference** |
| `ltx-2-spatial-upscaler-x2-1.0` | 2x resolution | High-res output |
| `ltx-2-temporal-upscaler-x2-1.0` | 2x framerate | High-fps output |

### Repository
- **GitHub**: https://github.com/Lightricks/LTX-2
- **HuggingFace**: https://huggingface.co/Lightricks/LTX-2
- **License**: `ltx-2-community-license-agreement` (commercial OK)

---

## Camera Control LoRAs

LTX-2 provides fine-tuned LoRAs for specific camera movements:

| LoRA | Description | HuggingFace |
|------|-------------|-------------|
| **Dolly Left** | Camera moves laterally left | `Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Left` |
| **Dolly Right** | Camera moves laterally right | `Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Right` |
| **Dolly In** | Camera pushes toward subject | `Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-In` |
| **Dolly Out** | Camera pulls away from subject | `Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Out` |
| **Jib Up** | Camera rises vertically | `Lightricks/LTX-2-19b-LoRA-Camera-Control-Jib-Up` |
| **Static** | No camera movement | `Lightricks/LTX-2-19b-LoRA-Camera-Control-Static` |

### Camera Movement Selection Strategy

For 3D reconstruction, we want views from multiple angles. Recommended sequence:
1. **Dolly Out** - Reveals full object
2. **Orbit-like** - Combine Dolly Left/Right with slight movement
3. **Multiple angles** - Generate multiple videos with different LoRAs

---

## Integration Architecture

```
┌─────────────────────┐     ┌──────────────────────┐     ┌─────────────────┐
│   Gradio UI         │────▶│  RunPod Serverless   │────▶│   S3 Bucket     │
│   (app_sidebar.py)  │     │  (LTX-2 Handler)     │     │   (arkrunr)     │
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

---

## Implementation Plan

### Phase 1: Docker Image

**File**: `runpod/ltx2/Dockerfile`

```dockerfile
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4-devel-ubuntu22.04

# System dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Clone LTX-2 repository
WORKDIR /workspace
RUN git clone https://github.com/Lightricks/LTX-2.git ltx2

# Install dependencies
WORKDIR /workspace/ltx2
RUN pip install -e .
RUN pip install diffusers>=0.32.0 transformers accelerate boto3 runpod imageio[ffmpeg]

# Download FP8 model checkpoint (cached on network volume)
# Model will be loaded at runtime from HuggingFace

# Copy handler
COPY handler_ltx2.py /workspace/handler.py
COPY start_ltx2.sh /start.sh
RUN chmod +x /start.sh

# Set environment
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/runpod-volume/huggingface

WORKDIR /workspace
CMD ["/start.sh"]
```

### Phase 2: Serverless Handler

**File**: `runpod/ltx2/handler_ltx2.py`

```python
#!/usr/bin/env python3
"""
LTX-2 RunPod Serverless Handler

Generates high-quality video from images with camera control.
"""

import os
import torch
import runpod
from diffusers import LTX2Pipeline
from PIL import Image

# Configuration
MODEL_ID = "Lightricks/LTX-2"
CHECKPOINT = os.environ.get("LTX2_CHECKPOINT", "ltx-2-19b-distilled")

# Camera LoRAs
CAMERA_LORAS = {
    "dolly_left": "Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Left",
    "dolly_right": "Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Right",
    "dolly_in": "Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-In",
    "dolly_out": "Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Out",
    "jib_up": "Lightricks/LTX-2-19b-LoRA-Camera-Control-Jib-Up",
    "static": "Lightricks/LTX-2-19b-LoRA-Camera-Control-Static",
}

# Global model
_pipeline = None

def load_model():
    global _pipeline
    if _pipeline is None:
        print(f"Loading LTX-2 ({CHECKPOINT})...")
        _pipeline = LTX2Pipeline.from_pretrained(
            MODEL_ID,
            subfolder=CHECKPOINT,
            torch_dtype=torch.bfloat16,
        ).to("cuda")
    return _pipeline

def handler(job):
    """
    Input:
        - image: Base64 encoded image OR image_url
        - prompt: Text description for generation
        - camera_motion: Camera LoRA to use (dolly_left, dolly_out, etc.)
        - num_frames: Number of frames (divisible by 8 + 1)
        - width: Output width (divisible by 32)
        - height: Output height (divisible by 32)
        - num_inference_steps: Diffusion steps (default: 8 for distilled)
        - seed: Random seed
        
    Output:
        - video_url: S3 URL of generated video
        - video_base64: Base64 encoded video (if small enough)
    """
    input_data = job["input"]
    
    # Load model
    pipe = load_model()
    
    # Load camera LoRA if specified
    camera_motion = input_data.get("camera_motion", "static")
    if camera_motion in CAMERA_LORAS:
        pipe.load_lora_weights(CAMERA_LORAS[camera_motion])
    
    # Load input image
    image = load_image(input_data)
    
    # Generate video
    output = pipe(
        prompt=input_data.get("prompt", ""),
        image=image,
        num_frames=input_data.get("num_frames", 97),  # 97 = 96 + 1
        width=input_data.get("width", 768),
        height=input_data.get("height", 512),
        num_inference_steps=input_data.get("num_inference_steps", 8),
        generator=torch.Generator("cuda").manual_seed(
            input_data.get("seed", 42)
        ),
    )
    
    # Save and upload video
    video_path = save_video(output.frames, input_data.get("fps", 24))
    video_url = upload_to_s3(video_path)
    
    return {
        "status": "success",
        "video_url": video_url,
        "camera_motion": camera_motion,
        "num_frames": len(output.frames),
    }

runpod.serverless.start({"handler": handler})
```

### Phase 3: Client Integration

**Add to**: `runpod/runpod_client.py`

```python
@dataclass
class LTX2Result:
    """Result from LTX-2 video generation."""
    success: bool
    video_path: Optional[str] = None
    video_url: Optional[str] = None
    error: Optional[str] = None
    camera_motion: str = "static"
    num_frames: int = 0
    duration_seconds: float = 0.0

class LTX2ServerlessClient:
    """Client for LTX-2 serverless endpoint."""
    
    VALID_CAMERA_MOTIONS = [
        "dolly_left", "dolly_right", 
        "dolly_in", "dolly_out",
        "jib_up", "static"
    ]
    
    def __init__(self, endpoint_id: str, api_key: str):
        self.endpoint_id = endpoint_id
        self.api_key = api_key
        self.base_url = f"https://api.runpod.ai/v2/{endpoint_id}"
    
    def generate_sync(
        self,
        image_path: str,
        output_dir: str,
        prompt: str = "",
        camera_motion: str = "static",
        num_frames: int = 97,
        width: int = 768,
        height: int = 512,
        fps: int = 24,
        seed: Optional[int] = None,
        **kwargs
    ) -> LTX2Result:
        """Generate video from image with camera motion."""
        # Implementation follows SEVA pattern
        pass
```

### Phase 4: Generator Module

**File**: `generators/ltx2.py`

```python
"""
LTX-2 Video Generation Module

Generates high-quality videos from images with camera control.
Uses Lightricks' LTX-2 19B parameter model.
"""

from runpod.runpod_client import LTX2ServerlessClient, LTX2Result

VALID_CAMERA_MOTIONS = [
    "dolly_left", "dolly_right",
    "dolly_in", "dolly_out", 
    "jib_up", "static"
]

def run_ltx2_runpod(
    image_path: str,
    output_dir: str,
    prompt: str = "",
    camera_motion: str = "dolly_out",
    num_frames: int = 97,
    width: int = 768,
    height: int = 512,
    fps: int = 24,
    seed: Optional[int] = None,
    api_key: str = "",
    endpoint_id: str = "",
    **kwargs
) -> LTX2Result:
    """Generate video using LTX-2 on RunPod."""
    client = LTX2ServerlessClient(endpoint_id, api_key)
    return client.generate_sync(
        image_path=image_path,
        output_dir=output_dir,
        prompt=prompt,
        camera_motion=camera_motion,
        num_frames=num_frames,
        width=width,
        height=height,
        fps=fps,
        seed=seed,
        **kwargs
    )
```

---

## File Structure

```
runpod/ltx2/
├── Dockerfile              # Docker image
├── handler_ltx2.py         # Serverless handler
├── start_ltx2.sh           # Startup script
└── README.md               # Deployment docs

generators/
└── ltx2.py                 # Generator module

runpod/runpod_client.py     # Updated with:
├── LTX2ServerlessClient    # Client class
└── LTX2Result              # Result dataclass

docs/
└── LTX2_INTEGRATION_PLAN.md  # This document
```

---

## UI Integration

### Location in Sidebar
**INPUT section** → "LTX-2 Video" (alongside Gen3C)

### UI Components
- Image input (drag & drop)
- Prompt text input
- Camera motion dropdown:
  - Dolly Out (default - best for 3D)
  - Dolly In
  - Dolly Left
  - Dolly Right
  - Jib Up
  - Static
- Resolution presets (768x512, 1024x768, etc.)
- Frame count slider (divisible by 8 + 1)
- Generate button
- Video preview

---

## Workflow Integration

### Standalone LTX-2
```
Image → LTX-2 → Video (high quality)
```

### LTX-2 + 2DGS Pipeline
```
Image → LTX-2 (camera motion) → 2DGS Pipeline → 3D Mesh (GLB)
```

### Comparison with Gen3C
```
Image → LTX-2 (4K, 50fps, dolly motion) → 2DGS → Mesh
  vs
Image → Gen3C (720p, 3D-aware depth) → 2DGS → Mesh
```

**Recommendation**: Use LTX-2 for higher quality video, Gen3C for better 3D consistency.

---

## Tasks Checklist

### Phase 1: Docker Image
- [ ] Create `runpod/ltx2/Dockerfile`
- [ ] Create `runpod/ltx2/start_ltx2.sh`
- [ ] Test locally with Docker
- [ ] Push to Docker Hub (`88dreams/ltx2-runpod:v1`)

### Phase 2: Serverless Handler
- [ ] Create `runpod/ltx2/handler_ltx2.py`
- [ ] Implement image loading (base64, S3 URL)
- [ ] Implement camera LoRA loading
- [ ] Implement video generation
- [ ] Implement S3 upload
- [ ] Create `runpod/ltx2/README.md`

### Phase 3: RunPod Deployment
- [ ] Create serverless endpoint
- [ ] Configure network volume for model weights
- [ ] Test endpoint health check
- [ ] Test video generation

### Phase 4: Client Integration
- [ ] Add `LTX2ServerlessClient` to `runpod_client.py`
- [ ] Add `LTX2Result` dataclass
- [ ] Implement `generate_sync()` method
- [ ] Create `generators/ltx2.py` module

### Phase 5: UI Integration
- [ ] Add LTX-2 to sidebar navigation
- [ ] Create LTX-2 generation page
- [ ] Add camera motion dropdown
- [ ] Add resolution presets
- [ ] Connect to 2DGS pipeline

---

## Hardware Requirements

### RunPod Serverless
- **GPU**: A100 40GB (recommended) or RTX A6000 (48GB)
- **Alternative**: RTX 4090 (24GB) with FP8
- **Network Volume**: 20GB for model weights + LoRAs

### Cost Estimate
- A100 40GB: ~$0.76/min active
- A6000: ~$0.38/min active
- Cold start: ~60-90 seconds (model loading)
- Inference (distilled): ~10-30 seconds per video

---

## Advantages Over Current Options

1. **No HuggingFace Access Needed** - Unlike SEVA, LTX-2 is fully open
2. **Higher Quality Output** - Native 4K vs Gen3C's lower resolution
3. **Faster Inference** - Distilled model: 8 steps vs 50+ for others
4. **Commercial License** - Can be used in production
5. **Flexible Camera Control** - Multiple LoRAs for different motions
6. **Audio Generation** - Bonus: can generate synchronized audio

---

## References

- [LTX-2 GitHub](https://github.com/Lightricks/LTX-2)
- [LTX-2 HuggingFace](https://huggingface.co/Lightricks/LTX-2)
- [Camera Control LoRAs](https://huggingface.co/collections/Lightricks/ltx-2-67af8dc4f217ccbb2f6dac81)
- [LTX-2 Paper (arXiv:2601.03233)](https://arxiv.org/abs/2601.03233)
- [Diffusers Integration](https://huggingface.co/docs/diffusers/api/pipelines/ltx2)

---

## Next Steps

1. **Immediate**: Create Docker image and handler
2. **Today**: Test locally, push to Docker Hub
3. **Next**: Deploy to RunPod, test endpoint
4. **Follow-up**: UI integration

---

*Created: January 13, 2026*
