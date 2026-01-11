# SV3D Setup Guide for RunPod

**Last Updated:** January 9, 2026  
**Status:** Ready for implementation  
**Estimated Time:** 1-2 hours

---

## Overview

SV3D (Stable Video 3D) generates orbital multi-view images from a single input image with **known camera poses**. This is ideal for 2DGS training because we don't need COLMAP for pose estimation.

### SV3D Variants

| Model | File | Purpose |
|-------|------|---------|
| SV3D_u | `sv3d_u.safetensors` | Unconditional orbital generation |
| SV3D_p | `sv3d_p.safetensors` | **Pose-conditioned** (we want this) |

SV3D_p allows specifying elevation and azimuth angles for controlled orbital views.

---

## Prerequisites

Before starting, ensure you have:

- [x] RunPod pod with GPU (A6000/A100 recommended)
- [x] Network volume mounted at `/workspace`
- [x] `sv3d-2dgs` conda environment created
- [x] HuggingFace account with access to `stabilityai/sv3d` (accept license)
- [x] Logged in via `huggingface-cli login`

---

## Installation Steps

### Step 1: Clone Stability's Generative Models Repository

```bash
# Activate environment
source /workspace/miniconda3/bin/activate sv3d-2dgs

# Clone the repo
cd /workspace/sv3d_2dgs
git clone https://github.com/Stability-AI/generative-models.git
cd generative-models
```

**Time:** ~2 minutes

---

### Step 2: Install Generative Models Dependencies

The repo has its own requirements. We'll install carefully to avoid conflicts:

```bash
# Install core dependencies (some may already be installed)
pip install kornia
pip install open-clip-torch
pip install einops
pip install omegaconf
pip install pytorch-lightning
pip install torchmetrics
pip install webdataset

# Install the package itself
pip install -e .
```

**Potential conflicts to watch for:**
- `pytorch-lightning` version
- `transformers` version
- `numpy` version (keep at 1.26.4!)

If you see numpy upgraded, run:
```bash
pip install numpy==1.26.4
```

**Time:** ~10-20 minutes

---

### Step 3: Download SV3D Weights

```bash
# Create models directory
mkdir -p /workspace/sv3d_2dgs/models/sv3d

# Download using huggingface-cli
huggingface-cli download stabilityai/sv3d sv3d_p.safetensors \
    --local-dir /workspace/sv3d_2dgs/models/sv3d

# Verify download
ls -lh /workspace/sv3d_2dgs/models/sv3d/sv3d_p.safetensors
```

Expected size: ~5GB

**Time:** ~5 minutes (depending on connection)

---

### Step 4: Download Required Sub-Models

SV3D requires additional models for the image encoder:

```bash
# These should auto-download on first run, but we can pre-fetch:
python -c "
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
print('Downloading CLIP...')
CLIPImageProcessor.from_pretrained('laion/CLIP-ViT-H-14-laion2B-s32B-b79K')
CLIPVisionModelWithProjection.from_pretrained('laion/CLIP-ViT-H-14-laion2B-s32B-b79K')
print('Done!')
"
```

**Time:** ~5 minutes

---

### Step 5: Create SV3D Inference Script

Create a Python script to run SV3D inference:

```bash
cat > /workspace/sv3d_2dgs/scripts/sv3d_inference.py << 'EOF'
#!/usr/bin/env python3
"""
SV3D Inference Script for Multi-View Generation

Generates orbital views from a single image using SV3D_p.
"""

import os
import sys
import math
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Tuple
from omegaconf import OmegaConf

# Add generative-models to path
sys.path.insert(0, '/workspace/sv3d_2dgs/generative-models')

from sgm.inference.api import (
    model_specs,
    SamplingParams,
    SamplingPipeline,
    Sampler,
)


class SV3DGenerator:
    """Generate multi-view images using SV3D_p."""
    
    def __init__(
        self,
        checkpoint_path: str = "/workspace/sv3d_2dgs/models/sv3d/sv3d_p.safetensors",
        device: str = "cuda",
    ):
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.pipeline = None
        
    def load_model(self):
        """Load SV3D model."""
        if self.pipeline is not None:
            return
            
        print("[SV3D] Loading model...")
        
        # Load SV3D_p configuration
        # This uses Stability's inference API
        self.pipeline = SamplingPipeline(
            model_spec=model_specs["sv3d_p"],
            checkpoint_path=self.checkpoint_path,
            device=self.device,
        )
        
        print("[SV3D] Model loaded!")
    
    def preprocess_image(self, image: Image.Image, size: int = 576) -> Image.Image:
        """Preprocess image for SV3D input."""
        # Resize maintaining aspect ratio
        image.thumbnail((size, size), Image.Resampling.LANCZOS)
        
        # Create square canvas (white background)
        canvas = Image.new("RGB", (size, size), (255, 255, 255))
        offset = (
            (size - image.width) // 2,
            (size - image.height) // 2,
        )
        
        if image.mode == "RGBA":
            canvas.paste(image, offset, image)
        else:
            canvas.paste(image, offset)
        
        return canvas
    
    def generate(
        self,
        image_path: str,
        output_dir: str,
        num_frames: int = 21,
        elevation_deg: float = 10.0,
        fps: int = 7,
    ) -> Tuple[List[Image.Image], List[np.ndarray]]:
        """
        Generate multi-view images.
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save outputs
            num_frames: Number of views to generate
            elevation_deg: Camera elevation in degrees
            fps: Output video FPS
            
        Returns:
            Tuple of (list of images, list of 4x4 camera poses)
        """
        self.load_model()
        
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        image = self.preprocess_image(image)
        
        # Save preprocessed image
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        image.save(output_path / "preprocessed_input.png")
        
        # Generate azimuths for orbital path
        azimuths_deg = [i * (360.0 / num_frames) for i in range(num_frames)]
        
        # SV3D_p expects polars (90 - elevation) and azimuths in radians
        polars_rad = [math.radians(90 - elevation_deg)] * num_frames
        azimuths_rad = [math.radians(a) for a in azimuths_deg]
        
        # Set up sampling parameters
        params = SamplingParams(
            num_frames=num_frames,
            cfg_scale=2.5,
            cond_aug=0.02,
        )
        
        # Generate
        print(f"[SV3D] Generating {num_frames} views at elevation {elevation_deg}°...")
        
        with torch.no_grad():
            frames = self.pipeline.sample(
                image=image,
                polars=polars_rad,
                azimuths=azimuths_rad,
                params=params,
            )
        
        # Save frames
        images_dir = output_path / "images"
        images_dir.mkdir(exist_ok=True)
        
        output_images = []
        for i, frame in enumerate(frames):
            frame_pil = Image.fromarray(frame)
            frame_pil.save(images_dir / f"frame_{i:04d}.png")
            output_images.append(frame_pil)
        
        # Compute camera poses
        poses = self._compute_poses(num_frames, elevation_deg)
        
        # Save poses
        poses_array = np.stack(poses)
        np.save(output_path / "poses.npy", poses_array)
        
        # Save intrinsics
        intrinsics = self._compute_intrinsics(image.size)
        np.save(output_path / "intrinsics.npy", intrinsics)
        
        print(f"[SV3D] Saved {len(output_images)} frames to {images_dir}")
        
        return output_images, poses
    
    def _compute_poses(self, num_frames: int, elevation_deg: float, radius: float = 1.5) -> List[np.ndarray]:
        """Compute camera-to-world poses for orbital path."""
        poses = []
        
        for i in range(num_frames):
            azimuth_deg = i * (360.0 / num_frames)
            
            az_rad = math.radians(azimuth_deg)
            el_rad = math.radians(elevation_deg)
            
            # Camera position on sphere
            x = radius * math.cos(el_rad) * math.sin(az_rad)
            y = radius * math.sin(el_rad)
            z = radius * math.cos(el_rad) * math.cos(az_rad)
            position = np.array([x, y, z])
            
            # Look-at matrix
            forward = -position / np.linalg.norm(position)
            up = np.array([0.0, 1.0, 0.0])
            right = np.cross(forward, up)
            right = right / np.linalg.norm(right)
            up = np.cross(right, forward)
            
            # Build 4x4 pose matrix
            pose = np.eye(4)
            pose[:3, 0] = right
            pose[:3, 1] = up
            pose[:3, 2] = -forward
            pose[:3, 3] = position
            
            poses.append(pose)
        
        return poses
    
    def _compute_intrinsics(self, image_size: Tuple[int, int]) -> np.ndarray:
        """Compute camera intrinsics (assuming 50mm equivalent FOV)."""
        W, H = image_size
        fov_deg = 46.8
        fov_rad = math.radians(fov_deg)
        focal_length = (W / 2) / math.tan(fov_rad / 2)
        
        return np.array([
            [focal_length, 0, W / 2],
            [0, focal_length, H / 2],
            [0, 0, 1],
        ])


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate multi-view images with SV3D")
    parser.add_argument("--image", "-i", required=True, help="Input image path")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--num-frames", "-n", type=int, default=21, help="Number of views")
    parser.add_argument("--elevation", "-e", type=float, default=10.0, help="Camera elevation (degrees)")
    
    args = parser.parse_args()
    
    generator = SV3DGenerator()
    generator.generate(
        image_path=args.image,
        output_dir=args.output,
        num_frames=args.num_frames,
        elevation_deg=args.elevation,
    )


if __name__ == "__main__":
    main()
EOF

chmod +x /workspace/sv3d_2dgs/scripts/sv3d_inference.py
```

**Note:** This script is a template. The exact API may need adjustment based on Stability's current code.

**Time:** ~5 minutes

---

### Step 6: Test SV3D Loading

```bash
cd /workspace/sv3d_2dgs

python -c "
import sys
sys.path.insert(0, '/workspace/sv3d_2dgs/generative-models')

# Test imports
try:
    from sgm.inference.api import model_specs, SamplingPipeline
    print('✓ SGM inference API loaded')
except ImportError as e:
    print(f'✗ Import failed: {e}')
    print('  You may need to install generative-models differently')

# Test checkpoint exists
import os
ckpt = '/workspace/sv3d_2dgs/models/sv3d/sv3d_p.safetensors'
if os.path.exists(ckpt):
    size_gb = os.path.getsize(ckpt) / 1e9
    print(f'✓ Checkpoint found ({size_gb:.1f} GB)')
else:
    print(f'✗ Checkpoint not found at {ckpt}')
"
```

**Time:** ~2 minutes

---

### Step 7: Test Full Inference (Optional)

If you have a test image:

```bash
# Copy test image
cp /path/to/your/test_image.jpg /workspace/sv3d_2dgs/inputs/

# Run SV3D
python /workspace/sv3d_2dgs/scripts/sv3d_inference.py \
    --image /workspace/sv3d_2dgs/inputs/test_image.jpg \
    --output /workspace/sv3d_2dgs/outputs/sv3d_test \
    --num-frames 21 \
    --elevation 10
```

**Time:** ~3-5 minutes for inference

---

## Troubleshooting

### "No module named 'sgm'"

The generative-models package wasn't installed correctly:

```bash
cd /workspace/sv3d_2dgs/generative-models
pip install -e .
```

### "model_specs doesn't have sv3d_p"

The model specs might have different names. Check available specs:

```bash
python -c "
import sys
sys.path.insert(0, '/workspace/sv3d_2dgs/generative-models')
from sgm.inference.api import model_specs
print('Available model specs:')
for k in model_specs.keys():
    print(f'  - {k}')
"
```

### NumPy version conflicts

After any pip install, verify numpy:

```bash
python -c "import numpy; print(numpy.__version__)"
# Should be 1.26.4

# If not:
pip install numpy==1.26.4
```

### CUDA out of memory

SV3D needs ~20GB VRAM. If you hit OOM:
- Use `--num-frames 14` instead of 21
- Enable CPU offloading (requires code modification)

---

## Alternative: Use SVD for Now

If SV3D setup is too complex, use SVD (Stable Video Diffusion) which IS diffusers-compatible:

```bash
python -c "
from diffusers import StableVideoDiffusionPipeline
import torch
pipe = StableVideoDiffusionPipeline.from_pretrained(
    'stabilityai/stable-video-diffusion-img2vid-xt',
    torch_dtype=torch.float16,
    variant='fp16'
)
print('SVD ready!')
"
```

SVD doesn't give orbital views, but the frames can still be used with 2DGS for a working pipeline.

---

## Summary Checklist

- [ ] Clone generative-models repo
- [ ] Install dependencies
- [ ] Download sv3d_p.safetensors
- [ ] Download CLIP models
- [ ] Test imports work
- [ ] Run test inference
- [ ] Integrate with 2DGS pipeline

---

## Next Steps After SV3D Works

1. Run SV3D on a test stage image
2. Feed multi-views + poses to 2DGS training
3. Extract mesh from trained 2DGS
4. Compare quality to Hunyuan results

---

*For the full 2DGS pipeline documentation, see `SV3D_IMPLEMENTATION_PLAN.md`*

