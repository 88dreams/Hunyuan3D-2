# SV3D Implementation Plan for ArkRunr Mesh Generation

**Last Updated:** January 8, 2026  
**Purpose:** Generate high-quality meshes from single interior images using SV3D multi-view generation → 2DGS training → mesh extraction.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Phase 1: SV3D Environment Setup](#phase-1-sv3d-environment-setup)
4. [Phase 2: Multi-View Generation](#phase-2-multi-view-generation)
5. [Phase 3: 2DGS Training Integration](#phase-3-2dgs-training-integration)
6. [Phase 4: Mesh Extraction](#phase-4-mesh-extraction)
7. [Phase 5: RunPod Deployment](#phase-5-runpod-deployment)
8. [Phase 6: Gradio UI Integration](#phase-6-gradio-ui-integration)
9. [Timeline](#timeline)
10. [Risk Mitigation](#risk-mitigation)

---

## Overview

### Goal

```
Single Interior Image → SV3D → Multi-View Images → 2DGS → High-Quality Mesh → Unity/ArkRunr
```

### Why This Pipeline

| Stage | Technology | Why |
|-------|------------|-----|
| Multi-view generation | **SV3D** | Best quality, known camera poses |
| 3D representation | **2DGS** | Surface-aligned, clean mesh extraction |
| Mesh extraction | **TSDF Fusion** | Works well with 2DGS depth/normals |

### Key Advantage

SV3D provides **known camera poses** - we don't need COLMAP. This eliminates a major failure point.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        SV3D → 2DGS → MESH PIPELINE                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  LOCAL (Gradio UI)                        RUNPOD SERVERLESS                     │
│  ─────────────────                        ──────────────────                    │
│                                                                                  │
│  ┌──────────────┐                         ┌─────────────────────────────────┐   │
│  │ app_sidebar  │                         │ handler_sv3d_2dgs.py            │   │
│  │    .py       │ ──── API Call ────────→ │                                 │   │
│  └──────────────┘                         │  ┌─────────────────────────┐    │   │
│         │                                 │  │ 1. SV3D Multi-View Gen  │    │   │
│         │                                 │  │    (21 orbital views)   │    │   │
│         │                                 │  └───────────┬─────────────┘    │   │
│         │                                 │              │                  │   │
│         │                                 │  ┌───────────▼─────────────┐    │   │
│         │                                 │  │ 2. Depth/Normal Est     │    │   │
│         │                                 │  │    (Depth Anything V2)  │    │   │
│         │                                 │  └───────────┬─────────────┘    │   │
│         │                                 │              │                  │   │
│         │                                 │  ┌───────────▼─────────────┐    │   │
│         │                                 │  │ 3. 2DGS Training        │    │   │
│         │                                 │  │    (15-30k iterations)  │    │   │
│         │                                 │  └───────────┬─────────────┘    │   │
│         │                                 │              │                  │   │
│         │                                 │  ┌───────────▼─────────────┐    │   │
│         │                                 │  │ 4. Mesh Extraction      │    │   │
│         │                                 │  │    (TSDF Fusion)        │    │   │
│         │                                 │  └───────────┬─────────────┘    │   │
│         │                                 │              │                  │   │
│         │                                 └──────────────┼──────────────────┘   │
│         │                                                │                      │
│         │                                 ┌──────────────▼──────────────────┐   │
│         │                                 │          S3 Upload              │   │
│         │                                 │  • mesh.glb                     │   │
│         │                                 │  • gaussians.ply                │   │
│         │                                 │  • preview_video.mp4            │   │
│         │                                 └──────────────┬──────────────────┘   │
│         │                                                │                      │
│  ┌──────▼──────┐                                         │                      │
│  │  Download   │ ◄───────────────────────────────────────┘                      │
│  │  & Display  │                                                                │
│  └─────────────┘                                                                │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: SV3D Environment Setup

**Duration:** 1 week  
**Goal:** Get SV3D running locally and on RunPod

### 1.1 Local Development Environment

```bash
# Create new conda environment
conda create -n sv3d-2dgs python=3.10
conda activate sv3d-2dgs

# Install PyTorch with CUDA
pip install torch==2.1.0 torchvision --index-url https://download.pytorch.org/whl/cu118

# Install SV3D dependencies (via diffusers)
pip install diffusers==0.27.0 transformers accelerate

# Install 2DGS dependencies
git clone https://github.com/hbb1/2d-gaussian-splatting.git
cd 2d-gaussian-splatting
pip install -r requirements.txt
pip install submodules/diff-gaussian-rasterization-2d
pip install submodules/simple-knn

# Install depth estimation
pip install transformers  # Depth Anything V2

# Install mesh processing
pip install trimesh open3d numpy-stl
```

### 1.2 SV3D Model Download

```python
# generators/sv3d_setup.py
"""
SV3D Model Setup and Verification
"""

from diffusers import StableVideo3DPipeline
import torch

def download_sv3d_models():
    """
    Download SV3D models to local cache.
    
    Models:
    - stabilityai/sv3d (main model, ~10GB)
    - stabilityai/sv3d_u (unconditional variant)
    - stabilityai/sv3d_p (pose-conditioned variant)
    """
    print("Downloading SV3D models...")
    
    # Download main SV3D model
    pipe = StableVideo3DPipeline.from_pretrained(
        "stabilityai/sv3d",
        torch_dtype=torch.float16,
        variant="fp16",
    )
    
    print(f"Model downloaded to: {pipe.config._name_or_path}")
    print("SV3D setup complete!")
    
    return pipe


def verify_sv3d_installation():
    """Verify SV3D can run a test generation."""
    from PIL import Image
    import numpy as np
    
    # Create test image
    test_image = Image.fromarray(np.random.randint(0, 255, (576, 576, 3), dtype=np.uint8))
    
    pipe = StableVideo3DPipeline.from_pretrained(
        "stabilityai/sv3d",
        torch_dtype=torch.float16,
    )
    pipe.to("cuda")
    pipe.enable_model_cpu_offload()
    
    # Quick test generation (2 frames only)
    result = pipe(
        test_image,
        num_frames=2,
        decode_chunk_size=1,
    )
    
    print(f"Test generation successful: {len(result.frames[0])} frames")
    return True


if __name__ == "__main__":
    download_sv3d_models()
    verify_sv3d_installation()
```

### 1.3 Requirements File

```txt
# requirements_sv3d_2dgs.txt

# Core
torch>=2.1.0
torchvision>=0.16.0
numpy>=1.24.0

# SV3D
diffusers>=0.27.0
transformers>=4.36.0
accelerate>=0.25.0
safetensors>=0.4.0

# Depth Estimation
einops>=0.7.0
timm>=0.9.0

# 2DGS (build from source)
# See: https://github.com/hbb1/2d-gaussian-splatting

# Mesh Processing
trimesh>=4.0.0
open3d>=0.17.0
pymeshlab>=2023.12

# Image Processing
pillow>=10.0.0
opencv-python>=4.8.0
imageio>=2.31.0
imageio-ffmpeg>=0.4.9

# Utilities
tqdm>=4.66.0
scipy>=1.11.0
```

---

## Phase 2: Multi-View Generation

**Duration:** 1-2 weeks  
**Goal:** Generate high-quality multi-view images from single interior photo

### 2.1 SV3D Multi-View Generator

```python
# generators/sv3d_multiview.py
"""
SV3D Multi-View Generator for ArkRunr

Generates orbital views from a single image with known camera poses.
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from diffusers import StableVideo3DPipeline


@dataclass
class SV3DConfig:
    """Configuration for SV3D generation."""
    num_frames: int = 21                    # Number of orbital views
    decode_chunk_size: int = 8              # Memory optimization
    motion_bucket_id: int = 127             # Motion amount (127 = standard)
    noise_aug_strength: float = 0.02        # Noise augmentation
    fps: int = 7                            # Output FPS
    
    # Camera parameters
    elevation: float = 10.0                 # Camera elevation (degrees)
    radius: float = 1.5                     # Distance from object center
    
    # Image preprocessing
    target_size: int = 576                  # SV3D input size
    remove_background: bool = True          # Remove background before generation
    
    # ArkRunr-specific
    front_arc_only: bool = True             # Generate only front-facing views
    arc_degrees: float = 180.0              # Arc width for front_arc_only mode


class SV3DMultiViewGenerator:
    """
    Generate multi-view images using Stable Video 3D.
    
    Key features:
    - Known camera poses (no COLMAP needed)
    - High multi-view consistency
    - Configurable elevation and arc coverage
    """
    
    def __init__(
        self,
        config: Optional[SV3DConfig] = None,
        device: str = "cuda",
        use_fp16: bool = True,
    ):
        self.config = config or SV3DConfig()
        self.device = device
        self.dtype = torch.float16 if use_fp16 else torch.float32
        
        self._pipeline = None
        self._rembg_session = None
    
    def _load_pipeline(self):
        """Lazy-load SV3D pipeline."""
        if self._pipeline is None:
            print("[SV3D] Loading pipeline...")
            self._pipeline = StableVideo3DPipeline.from_pretrained(
                "stabilityai/sv3d",
                torch_dtype=self.dtype,
                variant="fp16" if self.dtype == torch.float16 else None,
            )
            self._pipeline.to(self.device)
            self._pipeline.enable_model_cpu_offload()
            print("[SV3D] Pipeline loaded")
        return self._pipeline
    
    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image for SV3D.
        
        - Resize to target size
        - Remove background (optional)
        - Center and pad
        """
        # Remove background if requested
        if self.config.remove_background:
            image = self._remove_background(image)
        
        # Resize maintaining aspect ratio
        image.thumbnail(
            (self.config.target_size, self.config.target_size),
            Image.Resampling.LANCZOS,
        )
        
        # Create square canvas and center image
        canvas = Image.new("RGB", (self.config.target_size, self.config.target_size), (255, 255, 255))
        offset = (
            (self.config.target_size - image.width) // 2,
            (self.config.target_size - image.height) // 2,
        )
        
        # Handle RGBA images
        if image.mode == "RGBA":
            canvas.paste(image, offset, image)
        else:
            canvas.paste(image, offset)
        
        return canvas
    
    def _remove_background(self, image: Image.Image) -> Image.Image:
        """Remove background using rembg."""
        try:
            from rembg import remove, new_session
            
            if self._rembg_session is None:
                self._rembg_session = new_session("u2net")
            
            return remove(image, session=self._rembg_session)
        except ImportError:
            print("[SV3D] Warning: rembg not installed, skipping background removal")
            return image
    
    def generate(
        self,
        image_path: str,
        output_dir: Optional[str] = None,
    ) -> Tuple[List[Image.Image], List[np.ndarray]]:
        """
        Generate multi-view images from single input.
        
        Args:
            image_path: Path to input image
            output_dir: Optional directory to save frames
        
        Returns:
            Tuple of (list of images, list of 4x4 camera poses)
        """
        print(f"[SV3D] Generating multi-view from: {image_path}")
        
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        image = self._preprocess_image(image)
        
        # Load pipeline
        pipe = self._load_pipeline()
        
        # Generate orbital video
        print(f"[SV3D] Generating {self.config.num_frames} frames...")
        with torch.no_grad():
            result = pipe(
                image,
                num_frames=self.config.num_frames,
                decode_chunk_size=self.config.decode_chunk_size,
                motion_bucket_id=self.config.motion_bucket_id,
                noise_aug_strength=self.config.noise_aug_strength,
            )
        
        frames = result.frames[0]  # List of PIL images
        print(f"[SV3D] Generated {len(frames)} frames")
        
        # Compute camera poses
        poses = self._compute_camera_poses(len(frames))
        
        # Filter to front arc if requested
        if self.config.front_arc_only:
            frames, poses = self._filter_to_front_arc(frames, poses)
            print(f"[SV3D] Filtered to {len(frames)} front-arc frames")
        
        # Save frames if output directory specified
        if output_dir:
            self._save_frames(frames, poses, output_dir)
        
        return frames, poses
    
    def _compute_camera_poses(self, num_frames: int) -> List[np.ndarray]:
        """
        Compute camera poses for SV3D orbital trajectory.
        
        SV3D generates frames in a 360° orbit with fixed elevation.
        """
        poses = []
        
        for i in range(num_frames):
            # Azimuth evenly distributed over 360°
            azimuth = (360.0 / num_frames) * i
            
            # Convert to camera pose matrix
            pose = self._orbital_to_camera_matrix(
                azimuth=azimuth,
                elevation=self.config.elevation,
                radius=self.config.radius,
            )
            poses.append(pose)
        
        return poses
    
    def _orbital_to_camera_matrix(
        self,
        azimuth: float,
        elevation: float,
        radius: float,
    ) -> np.ndarray:
        """
        Convert orbital parameters to 4x4 camera-to-world matrix.
        
        Convention:
        - Y is up
        - Camera looks at origin
        - Azimuth 0 = front view (looking at -Z)
        """
        # Convert to radians
        az_rad = np.radians(azimuth)
        el_rad = np.radians(elevation)
        
        # Camera position on sphere
        x = radius * np.cos(el_rad) * np.sin(az_rad)
        y = radius * np.sin(el_rad)
        z = radius * np.cos(el_rad) * np.cos(az_rad)
        position = np.array([x, y, z])
        
        # Look-at matrix (camera looks at origin)
        forward = -position / np.linalg.norm(position)
        up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        
        # Build 4x4 pose matrix (camera-to-world)
        pose = np.eye(4)
        pose[:3, 0] = right
        pose[:3, 1] = up
        pose[:3, 2] = -forward
        pose[:3, 3] = position
        
        return pose
    
    def _filter_to_front_arc(
        self,
        frames: List[Image.Image],
        poses: List[np.ndarray],
    ) -> Tuple[List[Image.Image], List[np.ndarray]]:
        """
        Filter frames to front-facing arc only.
        
        For interiors, we only need views from the "audience" side.
        """
        arc_half = self.config.arc_degrees / 2
        
        filtered_frames = []
        filtered_poses = []
        
        num_frames = len(frames)
        for i, (frame, pose) in enumerate(zip(frames, poses)):
            # Compute azimuth for this frame
            azimuth = (360.0 / num_frames) * i
            
            # Normalize to -180 to 180
            if azimuth > 180:
                azimuth -= 360
            
            # Check if within arc
            if -arc_half <= azimuth <= arc_half:
                filtered_frames.append(frame)
                filtered_poses.append(pose)
        
        return filtered_frames, filtered_poses
    
    def _save_frames(
        self,
        frames: List[Image.Image],
        poses: List[np.ndarray],
        output_dir: str,
    ):
        """Save frames and poses to directory."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save images
        images_dir = output_path / "images"
        images_dir.mkdir(exist_ok=True)
        
        for i, frame in enumerate(frames):
            frame.save(images_dir / f"frame_{i:04d}.png")
        
        # Save poses as numpy
        poses_array = np.stack(poses)
        np.save(output_path / "poses.npy", poses_array)
        
        # Save camera intrinsics (assuming standard FOV)
        intrinsics = self._get_intrinsics(frames[0].size)
        np.save(output_path / "intrinsics.npy", intrinsics)
        
        print(f"[SV3D] Saved {len(frames)} frames to {output_dir}")
    
    def _get_intrinsics(self, image_size: Tuple[int, int]) -> np.ndarray:
        """
        Get camera intrinsic matrix.
        
        Assumes standard 50mm equivalent FOV.
        """
        W, H = image_size
        
        # Assume 50mm equivalent FOV (~46.8 degrees)
        fov_deg = 46.8
        fov_rad = np.radians(fov_deg)
        
        focal_length = (W / 2) / np.tan(fov_rad / 2)
        
        intrinsics = np.array([
            [focal_length, 0, W / 2],
            [0, focal_length, H / 2],
            [0, 0, 1],
        ])
        
        return intrinsics


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def generate_multiview_sv3d(
    image_path: str,
    output_dir: str,
    num_frames: int = 21,
    front_arc_only: bool = True,
    arc_degrees: float = 180.0,
    remove_background: bool = True,
) -> Dict[str, Any]:
    """
    Convenience function for SV3D multi-view generation.
    
    Args:
        image_path: Input image path
        output_dir: Output directory for frames
        num_frames: Number of frames to generate
        front_arc_only: Only keep front-facing views
        arc_degrees: Arc width for front-arc mode
        remove_background: Remove background before generation
    
    Returns:
        Dict with paths and metadata
    """
    config = SV3DConfig(
        num_frames=num_frames,
        front_arc_only=front_arc_only,
        arc_degrees=arc_degrees,
        remove_background=remove_background,
    )
    
    generator = SV3DMultiViewGenerator(config=config)
    frames, poses = generator.generate(image_path, output_dir)
    
    return {
        "num_frames": len(frames),
        "output_dir": output_dir,
        "images_dir": str(Path(output_dir) / "images"),
        "poses_path": str(Path(output_dir) / "poses.npy"),
        "intrinsics_path": str(Path(output_dir) / "intrinsics.npy"),
    }
```

### 2.2 Depth and Normal Estimation

```python
# generators/depth_normal_estimator.py
"""
Depth and Normal Estimation for Multi-View Images

Uses Depth Anything V2 for depth and derives normals from depth.
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Optional
from transformers import pipeline


class DepthNormalEstimator:
    """
    Estimate depth and surface normals for multi-view images.
    
    Uses:
    - Depth Anything V2 for depth estimation
    - Gradient-based normal computation from depth
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self._depth_pipeline = None
    
    def _load_depth_pipeline(self):
        """Lazy-load depth estimation pipeline."""
        if self._depth_pipeline is None:
            print("[Depth] Loading Depth Anything V2...")
            self._depth_pipeline = pipeline(
                "depth-estimation",
                model="depth-anything/Depth-Anything-V2-Large-hf",
                device=0 if self.device == "cuda" else -1,
            )
            print("[Depth] Pipeline loaded")
        return self._depth_pipeline
    
    def estimate_depth(self, image: Image.Image) -> np.ndarray:
        """
        Estimate depth from single image.
        
        Returns:
            Depth map as numpy array (H, W), values in relative units
        """
        pipe = self._load_depth_pipeline()
        result = pipe(image)
        
        # Convert to numpy
        depth = np.array(result["depth"])
        
        # Normalize to 0-1 range
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        
        return depth
    
    def depth_to_normals(
        self,
        depth: np.ndarray,
        intrinsics: np.ndarray,
    ) -> np.ndarray:
        """
        Compute surface normals from depth map.
        
        Args:
            depth: Depth map (H, W)
            intrinsics: 3x3 camera intrinsic matrix
        
        Returns:
            Normal map (H, W, 3), normalized vectors
        """
        H, W = depth.shape
        
        # Get focal lengths and principal point
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        
        # Create pixel coordinate grid
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        
        # Back-project to 3D
        z = depth
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        
        points = np.stack([x, y, z], axis=-1)
        
        # Compute gradients
        dz_dx = np.gradient(z, axis=1)
        dz_dy = np.gradient(z, axis=0)
        
        # Normal from cross product of tangent vectors
        normals = np.zeros((H, W, 3))
        normals[:, :, 0] = -dz_dx
        normals[:, :, 1] = -dz_dy
        normals[:, :, 2] = 1.0
        
        # Normalize
        norm = np.linalg.norm(normals, axis=-1, keepdims=True)
        normals = normals / (norm + 1e-8)
        
        return normals
    
    def process_multiview(
        self,
        images_dir: str,
        intrinsics_path: str,
        output_dir: str,
    ) -> dict:
        """
        Process all images in a directory.
        
        Args:
            images_dir: Directory with input images
            intrinsics_path: Path to camera intrinsics
            output_dir: Output directory for depth/normal maps
        
        Returns:
            Dict with paths to outputs
        """
        images_path = Path(images_dir)
        output_path = Path(output_dir)
        
        depth_dir = output_path / "depths"
        normal_dir = output_path / "normals"
        depth_dir.mkdir(parents=True, exist_ok=True)
        normal_dir.mkdir(parents=True, exist_ok=True)
        
        # Load intrinsics
        intrinsics = np.load(intrinsics_path)
        
        # Process each image
        image_files = sorted(images_path.glob("*.png"))
        print(f"[Depth] Processing {len(image_files)} images...")
        
        for img_file in image_files:
            image = Image.open(img_file).convert("RGB")
            
            # Estimate depth
            depth = self.estimate_depth(image)
            
            # Compute normals
            normals = self.depth_to_normals(depth, intrinsics)
            
            # Save
            stem = img_file.stem
            np.save(depth_dir / f"{stem}_depth.npy", depth)
            np.save(normal_dir / f"{stem}_normal.npy", normals)
            
            # Also save as images for visualization
            depth_vis = (depth * 255).astype(np.uint8)
            Image.fromarray(depth_vis).save(depth_dir / f"{stem}_depth.png")
            
            normal_vis = ((normals + 1) / 2 * 255).astype(np.uint8)
            Image.fromarray(normal_vis).save(normal_dir / f"{stem}_normal.png")
        
        print(f"[Depth] Saved depth/normals to {output_dir}")
        
        return {
            "depth_dir": str(depth_dir),
            "normal_dir": str(normal_dir),
            "num_processed": len(image_files),
        }


def estimate_depth_normals(
    images_dir: str,
    intrinsics_path: str,
    output_dir: str,
) -> dict:
    """Convenience function for depth/normal estimation."""
    estimator = DepthNormalEstimator()
    return estimator.process_multiview(images_dir, intrinsics_path, output_dir)
```

---

## Phase 3: 2DGS Training Integration

**Duration:** 2-3 weeks  
**Goal:** Train 2DGS on SV3D-generated multi-views

### 3.1 2DGS Training Wrapper

```python
# generators/train_2dgs.py
"""
2DGS Training for ArkRunr

Trains 2D Gaussian Splatting on multi-view images with depth/normal supervision.
"""

import os
import subprocess
import json
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field


@dataclass
class TwoDGSTrainingConfig:
    """Configuration for 2DGS training."""
    # Training iterations
    iterations: int = 30000
    
    # Learning rates
    position_lr_init: float = 0.00016
    position_lr_final: float = 0.0000016
    feature_lr: float = 0.0025
    opacity_lr: float = 0.05
    scaling_lr: float = 0.005
    rotation_lr: float = 0.001
    
    # 2DGS-specific
    depth_ratio: float = 1.0               # Depth distortion weight
    lambda_normal: float = 0.05            # Normal consistency weight
    
    # Densification
    densify_from_iter: int = 500
    densify_until_iter: int = 15000
    densify_grad_threshold: float = 0.0002
    
    # ArkRunr-specific
    depth_supervision: bool = True         # Use estimated depth
    depth_weight: float = 0.5              # Weight for depth loss
    normal_supervision: bool = True        # Use estimated normals
    normal_weight: float = 0.1             # Weight for normal loss
    
    # Output
    save_iterations: list = field(default_factory=lambda: [7000, 15000, 30000])


class TwoDGSTrainer:
    """
    Train 2DGS model on multi-view images.
    
    Wraps the 2DGS training process with custom supervision.
    """
    
    def __init__(
        self,
        two_dgs_path: str = "/workspace/2d-gaussian-splatting",
        config: Optional[TwoDGSTrainingConfig] = None,
    ):
        self.two_dgs_path = Path(two_dgs_path)
        self.config = config or TwoDGSTrainingConfig()
        
        # Verify 2DGS installation
        if not (self.two_dgs_path / "train.py").exists():
            raise RuntimeError(f"2DGS not found at {two_dgs_path}")
    
    def prepare_dataset(
        self,
        images_dir: str,
        poses_path: str,
        intrinsics_path: str,
        depths_dir: Optional[str] = None,
        normals_dir: Optional[str] = None,
        output_dir: str = None,
    ) -> str:
        """
        Prepare dataset in 2DGS-compatible format.
        
        2DGS expects COLMAP-style format:
        output_dir/
        ├── images/
        ├── sparse/0/
        │   ├── cameras.bin
        │   ├── images.bin
        │   └── points3D.bin
        └── depths/ (optional)
        """
        # Implementation details in full code...
        pass
    
    def train(
        self,
        dataset_path: str,
        output_path: str,
    ) -> Dict[str, Any]:
        """
        Run 2DGS training.
        
        Args:
            dataset_path: Path to prepared dataset
            output_path: Path for model output
        
        Returns:
            Dict with training results
        """
        print(f"[2DGS] Starting training...")
        print(f"[2DGS] Dataset: {dataset_path}")
        print(f"[2DGS] Output: {output_path}")
        print(f"[2DGS] Iterations: {self.config.iterations}")
        
        # Build command
        cmd = [
            "python", str(self.two_dgs_path / "train.py"),
            "-s", dataset_path,
            "-m", output_path,
            "--iterations", str(self.config.iterations),
            "--depth_ratio", str(self.config.depth_ratio),
            "--lambda_normal", str(self.config.lambda_normal),
            "--densify_grad_threshold", str(self.config.densify_grad_threshold),
        ]
        
        # Run training
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"
        
        subprocess.run(cmd, cwd=str(self.two_dgs_path), env=env, check=True)
        
        # Find output PLY
        ply_path = Path(output_path) / "point_cloud" / f"iteration_{self.config.iterations}" / "point_cloud.ply"
        
        return {
            "ply_path": str(ply_path),
            "model_path": output_path,
            "iterations": self.config.iterations,
        }
```

---

## Phase 4: Mesh Extraction

**Duration:** 1 week  
**Goal:** Extract high-quality mesh from trained 2DGS

### 4.1 TSDF Mesh Extraction

```python
# generators/mesh_extraction.py
"""
Mesh Extraction from 2DGS

Uses TSDF fusion on rendered depth maps for high-quality mesh extraction.
"""

import numpy as np
import open3d as o3d
import torch
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass


@dataclass
class MeshExtractionConfig:
    """Configuration for mesh extraction."""
    # TSDF parameters
    voxel_size: float = 0.01               # Voxel size in meters
    sdf_trunc: float = 0.04                # Truncation distance (4x voxel size)
    depth_scale: float = 1.0               # Depth scaling factor
    depth_max: float = 10.0                # Maximum depth to integrate
    
    # Rendering for TSDF
    num_render_views: int = 100            # Number of views to render
    render_resolution: Tuple[int, int] = (1024, 1024)
    
    # Mesh processing
    target_triangles: int = 150000         # Target triangle count
    smooth_iterations: int = 2             # Laplacian smoothing
    remove_small_components: bool = True   # Remove floating artifacts
    min_component_ratio: float = 0.01      # Minimum component size (fraction)
    
    # Unity export
    convert_to_y_up: bool = True           # Convert Z-up to Y-up
    scale_factor: float = 1.0              # Scale for export


class MeshExtractor:
    """
    Extract mesh from trained 2DGS model.
    
    Pipeline:
    1. Load trained 2DGS model
    2. Render depth maps from multiple viewpoints
    3. Fuse depth maps using TSDF
    4. Extract mesh with marching cubes
    5. Post-process (clean, decimate, smooth)
    """
    
    def __init__(self, config: Optional[MeshExtractionConfig] = None):
        self.config = config or MeshExtractionConfig()
    
    def extract_mesh(
        self,
        model_path: str,
        output_path: str,
        poses: Optional[np.ndarray] = None,
        intrinsics: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Extract mesh from 2DGS model.
        
        Args:
            model_path: Path to trained 2DGS model directory
            output_path: Path for output mesh (GLB)
            poses: Optional camera poses for rendering
            intrinsics: Optional camera intrinsics
        
        Returns:
            Dict with extraction results
        """
        # Implementation details...
        pass
    
    def _postprocess_mesh(self, mesh) -> Any:
        """Post-process extracted mesh."""
        # Remove small components
        if self.config.remove_small_components:
            # Cluster and filter
            pass
        
        # Smooth
        if self.config.smooth_iterations > 0:
            mesh = mesh.filter_smooth_laplacian(
                number_of_iterations=self.config.smooth_iterations
            )
        
        # Decimate
        if len(mesh.triangles) > self.config.target_triangles:
            mesh = mesh.simplify_quadric_decimation(
                target_number_of_triangles=self.config.target_triangles
            )
        
        return mesh
```

---

## Phase 5: RunPod Deployment

**Duration:** 1-2 weeks  
**Goal:** Deploy complete pipeline as serverless endpoint

### 5.1 Unified Handler

```python
# runpod/sv3d_2dgs/handler_sv3d_2dgs.py
"""
RunPod Serverless Handler for SV3D → 2DGS → Mesh Pipeline
"""

import runpod

def handler(job):
    """
    Main handler for SV3D → 2DGS → Mesh pipeline.
    
    Input:
        image_base64: Base64-encoded input image
        OR
        image_url: URL to download image from
        
        config: Optional configuration dict
            - num_frames: Number of SV3D frames (default: 21)
            - front_arc_only: Only use front-facing views (default: True)
            - iterations: 2DGS training iterations (default: 30000)
            - target_triangles: Mesh triangle count (default: 150000)
    
    Output:
        mesh_url: S3 URL to GLB mesh
        ply_url: S3 URL to 2DGS PLY
        stats: Processing statistics
    """
    # Stage 1: SV3D Multi-View Generation
    # Stage 2: Depth/Normal Estimation
    # Stage 3: 2DGS Training
    # Stage 4: Mesh Extraction
    # Stage 5: Upload to S3
    
    return {
        "mesh_url": mesh_url,
        "ply_url": ply_url,
        "stats": stats,
    }

runpod.serverless.start({"handler": handler})
```

### 5.2 Dockerfile

```dockerfile
# runpod/sv3d_2dgs/Dockerfile
FROM nvcr.io/nvidia/pytorch:24.01-py3

# Install system dependencies
RUN apt-get update && apt-get install -y git wget colmap

# Install Python dependencies
COPY requirements_sv3d_2dgs.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements_sv3d_2dgs.txt

# Clone and install 2DGS
RUN git clone https://github.com/hbb1/2d-gaussian-splatting.git && \
    cd 2d-gaussian-splatting && \
    pip install submodules/diff-gaussian-rasterization-2d && \
    pip install submodules/simple-knn

# Pre-download models
RUN python -c "from diffusers import StableVideo3DPipeline; StableVideo3DPipeline.from_pretrained('stabilityai/sv3d')"

CMD ["python", "handler_sv3d_2dgs.py"]
```

### 5.3 Docker Image Details

| Image | Tag | Contents |
|-------|-----|----------|
| `88dreams/sv3d-2dgs-runpod` | v1 | SV3D + 2DGS + Depth Anything V2 |

---

## Phase 6: Gradio UI Integration

**Duration:** 1 week  
**Goal:** Add SV3D → 2DGS tab to the Gradio UI

### 6.1 UI Tab

```python
# ui/tabs/sv3d_2dgs_tab.py
"""
SV3D → 2DGS → Mesh Tab for Gradio UI
"""

import gradio as gr

def create_sv3d_2dgs_tab():
    """Create the SV3D → 2DGS tab for Gradio UI."""
    
    with gr.Column():
        gr.Markdown("""
        ## 🎯 SV3D → 2DGS → High-Quality Mesh
        
        Generate production-quality 3D meshes from a single image.
        """)
        
        # Input
        input_image = gr.Image(label="Input Image", type="filepath")
        
        # Quality preset
        quality_preset = gr.Radio(
            choices=[
                "Fast (10 min)",
                "Balanced (20 min)",
                "High Quality (40 min)",
            ],
            value="Balanced (20 min)",
            label="Quality Preset",
        )
        
        # View coverage
        view_coverage = gr.Radio(
            choices=[
                "Front Arc (180°) - Interior/Stage",
                "Full Orbit (360°) - Object",
            ],
            value="Front Arc (180°) - Interior/Stage",
            label="View Coverage",
        )
        
        # Generate button
        generate_btn = gr.Button("Generate Mesh", variant="primary")
        
        # Output
        output_mesh = gr.File(label="GLB Mesh")
        model_viewer = gr.Model3D(label="Preview")
    
    return {
        "input_image": input_image,
        "quality_preset": quality_preset,
        "view_coverage": view_coverage,
        "generate_btn": generate_btn,
        "output_mesh": output_mesh,
        "model_viewer": model_viewer,
    }
```

---

## Timeline

```
SV3D → 2DGS IMPLEMENTATION TIMELINE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Week    1    2    3    4    5    6    7    8
        │    │    │    │    │    │    │    │
Phase 1 ████████                              Environment Setup
        SV3D + 2DGS local environment         
        Verify models work                    
                                              
Phase 2      ████████████                     Multi-View Generation
             SV3D generator                   
             Depth/Normal estimation          
             Front-arc filtering              
                                              
Phase 3           ████████████████            2DGS Training
                  Training wrapper            
                  COLMAP format               
                  Test on generated views     
                                              
Phase 4                     ████████          Mesh Extraction
                            TSDF fusion       
                            Post-processing   
                            GLB export        
                                              
Phase 5                          ████████     RunPod Deployment
                                 Dockerfile   
                                 Handler      
                                 Testing      
                                              
Phase 6                               ████    UI Integration
                                              Gradio tab
                                              End-to-end testing

MILESTONES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Week 2:  ◆ SV3D generating multi-views locally
Week 4:  ◆ 2DGS training on SV3D output
Week 5:  ◆ Mesh extraction working
Week 6:  ◆ RunPod endpoint deployed
Week 7:  ◆ Gradio tab integrated
Week 8:  ◆ Production ready
```

---

## Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| SV3D not ideal for interiors | Medium | High | Add depth supervision, test Zero123++ as fallback |
| 2DGS training instability | Low | Medium | Use proven hyperparameters from paper |
| VRAM constraints | Medium | Medium | Use model offloading, A6000+ GPUs |
| Long training times | High | Medium | Offer quality presets, optimize iterations |
| Mesh extraction artifacts | Medium | Medium | TSDF tuning, post-processing pipeline |

---

## Success Criteria

| Milestone | Success Metric |
|-----------|----------------|
| **SV3D Generation** | 21 consistent multi-view images with known poses |
| **2DGS Training** | PSNR > 25 on held-out views |
| **Mesh Quality** | Geometry score ≥ 28/35 on ArkRunr test images |
| **End-to-End** | Single image → Unity-ready GLB in < 30 minutes |
| **Production** | 95% success rate on diverse interior images |

---

## Resource Requirements

| Phase | GPU | VRAM | Storage | Time |
|-------|-----|------|---------|------|
| SV3D Generation | A100/A6000 | 24GB | 2GB | 2-3 min |
| Depth Estimation | Any GPU | 8GB | 500MB | 1 min |
| 2DGS Training | A100/A6000 | 24GB | 5GB | 15-40 min |
| Mesh Extraction | CPU | 16GB RAM | 1GB | 2-5 min |

**Total per job:** ~20-50 minutes, ~$0.50-1.50 on RunPod

---

## References

### Models

| Model | Source | Purpose |
|-------|--------|---------|
| SV3D | [Stability AI](https://huggingface.co/stabilityai/sv3d) | Multi-view generation |
| 2DGS | [hbb1/2d-gaussian-splatting](https://github.com/hbb1/2d-gaussian-splatting) | 3D representation |
| Depth Anything V2 | [Hugging Face](https://huggingface.co/depth-anything/Depth-Anything-V2-Large-hf) | Depth estimation |

### Papers

| Paper | Year | Contribution |
|-------|------|--------------|
| [SV3D](https://arxiv.org/abs/2403.12008) | 2024 | Multi-view from single image |
| [2D Gaussian Splatting](https://arxiv.org/abs/2403.17888) | 2024 | Surface-aligned Gaussians |
| [Depth Anything V2](https://arxiv.org/abs/2406.09414) | 2024 | Monocular depth estimation |

---

## Next Steps

1. **This Week**: Set up local dev environment (Phase 1)
2. **First Test**: Generate SV3D views from a stage photo
3. **Validate**: Check if views look reasonable for interiors
4. **Iterate**: Tune front-arc parameters based on results

---

*This document outlines the implementation plan for SV3D → 2DGS mesh generation pipeline. For general 2DGS information, see `2DGS_PLAN.md`. For ArkRunr-specific requirements, see `ArkRunr_2DGS.md`.*

