# Interior Reconstruction Approaches for ArkRunr

**Created:** January 9, 2026  
**Purpose:** Evaluate two promising approaches for converting interior photographs into 3D meshes suitable for ArkRunr's front-arc viewing requirements.

---

## Table of Contents

1. [Background: Why These Approaches](#background-why-these-approaches)
2. [Understanding Camera Poses](#understanding-camera-poses)
3. [Approach 1: Gen3C Video → Frame Extraction → 2DGS](#approach-1-gen3c-video--frame-extraction--2dgs)
4. [Approach 2: Depth Estimation → View Synthesis → 2DGS](#approach-2-depth-estimation--view-synthesis--2dgs)
5. [Comparison and Recommendations](#comparison-and-recommendations)
6. [Implementation Priority](#implementation-priority)

---

## Background: Why These Approaches

### The Problem with Object-Centric Models (SV3D, Zero123++)

Models like SV3D are trained on **object-centric data** — they expect to orbit around a central subject from the outside. When applied to interiors:

- They hallucinate "what's behind the stage" (impossible views)
- They treat the interior as an object to orbit around
- The result is blurry, unrealistic geometry in the center

### What ArkRunr Actually Needs

```
                    BACK WALL / STAGE
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        │                │                │
   LEFT │    PERFORMER   │    RIGHT       │
   WALL │      AREA      │    WALL        │
        │                │                │
        │                │                │
        └────────────────┼────────────────┘
                         │
             ◄───── 120° ARC ─────►
                         │
                    CAMERA POSITIONS
                    (audience area)

Requirements:
• Camera stays in FRONT of the stage (audience perspective)
• ~120° horizontal arc maximum
• Multiple elevations (floor level to balcony)
• NO views from behind the stage
```

### Why Gen3C and Depth Estimation Are Promising

| Approach | Why It Works for Interiors |
|----------|---------------------------|
| **Gen3C** | Trained on scene-level video; understands interior geometry; produces realistic camera motion within spaces |
| **Depth Estimation** | Doesn't hallucinate; uses actual image data to project to new viewpoints; stays grounded in reality |

---

## Understanding Camera Poses

### What Is a Camera Pose?

A **camera pose** describes where the camera is located and what direction it's looking. It's essential for 2DGS training because the model needs to know:

1. **Where was the camera?** (position in 3D space)
2. **What direction was it looking?** (orientation)

### Camera Pose Representation

Camera poses are typically represented as a **4×4 transformation matrix**:

```
┌                         ┐
│  R₀₀  R₀₁  R₀₂  Tx     │     R = 3×3 Rotation matrix
│  R₁₀  R₁₁  R₁₂  Ty     │     T = 3×1 Translation vector
│  R₂₀  R₂₁  R₂₂  Tz     │
│   0    0    0    1      │     (Camera-to-World transform)
└                         ┘
```

**Rotation (R)**: Defines camera orientation (which way it's pointing)
**Translation (T)**: Defines camera position (where it is in space)

### Intrinsics vs Extrinsics

| Parameter | What It Describes | Example Values |
|-----------|------------------|----------------|
| **Intrinsics (K)** | Camera's internal properties | Focal length, principal point, sensor size |
| **Extrinsics (Pose)** | Camera's position/orientation in world | The 4×4 matrix above |

For 2DGS, we need **both**:
- **Intrinsics**: Usually estimated or assumed (common values work for most photos)
- **Extrinsics**: Must be determined for each view

### Why Poses Matter for 2DGS

```
Without accurate poses:              With accurate poses:
┌─────────────────────────┐         ┌─────────────────────────┐
│                         │         │                         │
│  View 1 ────┐           │         │  View 1 ────┐           │
│             │ UNKNOWN   │         │             │ ALIGNED   │
│  View 2 ────┤ relative  │         │  View 2 ────┤ in 3D     │
│             │ positions │         │             │ space     │
│  View 3 ────┘           │         │  View 3 ────┘           │
│                         │         │                         │
│  ❌ 2DGS can't learn    │         │  ✅ 2DGS learns correct │
│     consistent geometry │         │     3D geometry         │
└─────────────────────────┘         └─────────────────────────┘
```

---

## Approach 1: Gen3C Video → Frame Extraction → 2DGS

### Overview

Gen3C generates high-quality videos with realistic camera motion. The idea:

1. Generate a Gen3C video with controlled camera motion
2. Extract frames from the video
3. Estimate camera poses for each frame
4. Train 2DGS on frames + poses
5. Extract mesh

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      GEN3C → 2DGS PIPELINE                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  INPUT IMAGE                                                                 │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────┐                    │
│  │  STEP 1: GEN3C VIDEO GENERATION                     │                    │
│  │                                                      │                    │
│  │  • Generate video with camera motion                │                    │
│  │  • Parameters: dolly, arc, elevation                │                    │
│  │  • Output: MP4 video (e.g., 25 frames)              │                    │
│  └───────────────────────┬─────────────────────────────┘                    │
│                          │                                                   │
│                          ▼                                                   │
│  ┌─────────────────────────────────────────────────────┐                    │
│  │  STEP 2: FRAME EXTRACTION                           │                    │
│  │                                                      │                    │
│  │  • Extract N frames (e.g., every 2nd frame)         │                    │
│  │  • Filter for quality/diversity                     │                    │
│  │  • Output: 10-15 PNG images                         │                    │
│  └───────────────────────┬─────────────────────────────┘                    │
│                          │                                                   │
│                          ▼                                                   │
│  ┌─────────────────────────────────────────────────────┐                    │
│  │  STEP 3: POSE ESTIMATION                            │                    │
│  │  ════════════════════════════════════════════════   │                    │
│  │                                                      │                    │
│  │  OPTION A: Structure-from-Motion (COLMAP)           │                    │
│  │  ─────────────────────────────────────────          │                    │
│  │  • Run COLMAP on extracted frames                   │                    │
│  │  • Automatically estimates relative poses           │                    │
│  │  • Works well when frames have sufficient overlap   │                    │
│  │  • Output: cameras.bin, images.bin, points3D.bin    │                    │
│  │                                                      │                    │
│  │  OPTION B: Visual Odometry / SLAM                   │                    │
│  │  ─────────────────────────────────────────          │                    │
│  │  • DROID-SLAM, ORB-SLAM3, or similar               │                    │
│  │  • Processes video sequentially                     │                    │
│  │  • Better for smooth camera motion                  │                    │
│  │  • Output: trajectory file with poses              │                    │
│  │                                                      │                    │
│  │  OPTION C: Relative Pose Estimation (DUSt3R)        │                    │
│  │  ─────────────────────────────────────────          │                    │
│  │  • Deep learning-based pose estimation             │                    │
│  │  • Works on image pairs                            │                    │
│  │  • More robust to textureless regions              │                    │
│  │  • Output: relative poses + dense correspondences  │                    │
│  │                                                      │                    │
│  └───────────────────────┬─────────────────────────────┘                    │
│                          │                                                   │
│                          ▼                                                   │
│  ┌─────────────────────────────────────────────────────┐                    │
│  │  STEP 4: 2DGS TRAINING                              │                    │
│  │                                                      │                    │
│  │  • Initialize from depth (or random)                │                    │
│  │  • Train on frames + estimated poses                │                    │
│  │  • ArkRunr-specific losses (geometry > appearance)  │                    │
│  │  • Output: trained 2DGS model                       │                    │
│  └───────────────────────┬─────────────────────────────┘                    │
│                          │                                                   │
│                          ▼                                                   │
│  ┌─────────────────────────────────────────────────────┐                    │
│  │  STEP 5: MESH EXTRACTION                            │                    │
│  │                                                      │                    │
│  │  • TSDF fusion from rendered depths                 │                    │
│  │  • Marching cubes for mesh                          │                    │
│  │  • Cleanup + decimation                             │                    │
│  │  • Output: GLB mesh for Unity                       │                    │
│  └───────────────────────┬─────────────────────────────┘                    │
│                          │                                                   │
│                          ▼                                                   │
│                      OUTPUT MESH                                             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Step 1: Gen3C Video Generation

**Current Status**: ✅ Already working on RunPod

**Key Parameters to Control**:
- Camera motion type (dolly, arc, zoom, etc.)
- Motion magnitude
- Number of frames

**Challenge**: Gen3C's camera motion is learned, not explicitly parameterized. We may need to:
- Experiment with prompts to control motion
- Generate multiple videos and select best motion
- Or accept the motion Gen3C provides and estimate poses

### Step 2: Frame Extraction

**Trivial Implementation**:

```python
import cv2
from pathlib import Path

def extract_frames(video_path: str, output_dir: str, every_n: int = 2) -> list:
    """
    Extract frames from Gen3C video.
    
    Args:
        video_path: Path to MP4 video
        output_dir: Directory to save frames
        every_n: Extract every Nth frame (2 = every other frame)
    
    Returns:
        List of paths to extracted frames
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % every_n == 0:
            frame_path = output_dir / f"frame_{frame_idx:04d}.png"
            cv2.imwrite(str(frame_path), frame)
            frames.append(str(frame_path))
        
        frame_idx += 1
    
    cap.release()
    return frames
```

### Step 3: Pose Estimation — THE CRITICAL STEP

This is where most of the work is needed. **Key discovery**: Gen3C's camera poses are not random — they're **deterministically generated** from the trajectory parameters.

#### Key Finding: Gen3C Camera Trajectories Are Parametric

From the [Gen3C GitHub repository](https://github.com/nv-tlabs/GEN3C):

```
Trajectory Types:        Camera Rotation Modes:         Movement:
──────────────────       ─────────────────────          ──────────
• left                   • center_facing (default)      • movement_distance
• right                  • no_rotation                    (default 0.3)
• up                     • trajectory_aligned
• down
• zoom_in
• zoom_out
• clockwise
• counterclockwise
```

This means: **Same parameters → Same camera path every time**

We have FOUR options for getting poses:

---

#### Option A: ViPE (NVIDIA's Recommended Approach) ⭐ RECOMMENDED

NVIDIA provides **ViPE** — a data annotation pipeline that jointly predicts depth and camera pose from video. This is what Gen3C uses internally for training and testing.

**What ViPE outputs**:
- Depth maps for each frame
- Camera intrinsics (focal length, principal point)
- Camera extrinsics (poses as 4×4 matrices)

**Installation and Usage**:
```bash
# ViPE requires separate environment (not compatible with Gen3C)
conda create -n vipe python=3.10
conda activate vipe
# Follow ViPE installation instructions from NVIDIA...

# Run on Gen3C video
vipe infer gen3c_output.mp4 --output vipe_results/
```

**ViPE Output Structure** (used by Gen3C's `--vipe_path` option):
```
vipe_results/
├── depth/           # Depth maps per frame
│   ├── 0000.npy
│   ├── 0001.npy
│   └── ...
├── intrinsics.npy   # Camera intrinsics
└── poses.npy        # Camera extrinsics (N × 4 × 4 matrices)
```

**Pros**:
- ✅ Official NVIDIA tool designed for Gen3C
- ✅ Outputs depth + poses together (both needed for 2DGS)
- ✅ Well-tested on Gen3C videos
- ✅ Single command to get everything

**Cons**:
- ❌ Separate conda environment required
- ❌ Must install another tool
- ❌ Still estimation (not ground truth from Gen3C)

**Status**: ✅ ViPE is publicly available at [github.com/nv-tlabs/vipe](https://github.com/nv-tlabs/vipe)

**Setup files created**:
- `runpod/vipe/setup_vipe.sh` - Installation script for RunPod
- `generators/vipe_integration.py` - Python wrapper
- `scripts/test_vipe.py` - Test script
- `docs/VIPE_SETUP_GUIDE.md` - Detailed setup guide

---

#### Option B: Reconstruct Poses from Trajectory Parameters

Since Gen3C trajectories are **deterministic**, we could mathematically reconstruct the poses:

```python
def gen3c_trajectory_to_poses(
    trajectory: str,           # "left", "clockwise", etc.
    movement_distance: float,  # 0.3 default
    camera_rotation: str,      # "center_facing", etc.
    num_frames: int,           # 121, 241, etc.
) -> List[np.ndarray]:
    """
    Reconstruct camera poses from Gen3C trajectory parameters.
    
    Gen3C uses specific trajectory math internally.
    This function would need to match that math exactly.
    """
    poses = []
    
    if trajectory == "left":
        # Camera moves left over num_frames
        for i in range(num_frames):
            t = i / (num_frames - 1)  # 0 to 1
            x_offset = -movement_distance * t
            
            pose = np.eye(4)
            pose[0, 3] = x_offset  # Translate left
            
            if camera_rotation == "center_facing":
                # Rotate to keep looking at center
                # Need to compute rotation based on position
                pass
            
            poses.append(pose)
    
    elif trajectory == "clockwise":
        # Camera orbits clockwise
        for i in range(num_frames):
            angle = 2 * np.pi * i / num_frames * movement_distance
            x = np.sin(angle) * movement_distance
            z = np.cos(angle) * movement_distance
            # ... build pose matrix
            pass
    
    # ... other trajectories
    
    return poses
```

**Pros**:
- ✅ No additional tools needed
- ✅ Potentially exact (if math matches Gen3C)
- ✅ Very fast
- ✅ Deterministic

**Cons**:
- ❌ Requires reverse-engineering Gen3C's trajectory code
- ❌ May not match exactly (coordinate systems, scale, etc.)
- ❌ Not documented publicly

**TO DO**: Examine Gen3C source code in `cosmos_predict1/` to find trajectory calculation logic.

**Where to look**:
```
GEN3C/
├── cosmos_predict1/
│   └── diffusion/
│       └── inference/
│           └── gen3c_single_image.py  # Look for trajectory logic here
└── scripts/
    └── ...  # May have camera utilities
```

---

#### Option C: COLMAP (Structure from Motion)

**What it does**: Matches features between images, finds correspondences, estimates relative camera poses

**Pros**:
- ✅ Industry standard, well-tested
- ✅ Accurate when it works
- ✅ Also produces sparse 3D points (can initialize 2DGS)

**Cons**:
- ❌ Can fail on textureless regions (common in interiors)
- ❌ Requires sufficient viewpoint change between frames
- ❌ May not converge if frames are too similar

**Implementation**:

```python
import subprocess
from pathlib import Path

def run_colmap_on_frames(
    image_dir: str,
    workspace_dir: str,
) -> dict:
    """
    Run COLMAP on extracted frames to estimate camera poses.
    
    Returns:
        Dict with paths to COLMAP outputs
    """
    workspace = Path(workspace_dir)
    database_path = workspace / "database.db"
    sparse_dir = workspace / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Feature extraction
    subprocess.run([
        "colmap", "feature_extractor",
        "--database_path", str(database_path),
        "--image_path", image_dir,
        "--ImageReader.single_camera", "1",  # Assume same camera for all frames
        "--SiftExtraction.use_gpu", "1",
    ], check=True)
    
    # Step 2: Feature matching
    subprocess.run([
        "colmap", "sequential_matcher",  # Good for video frames
        "--database_path", str(database_path),
    ], check=True)
    
    # Step 3: Sparse reconstruction
    subprocess.run([
        "colmap", "mapper",
        "--database_path", str(database_path),
        "--image_path", image_dir,
        "--output_path", str(sparse_dir),
    ], check=True)
    
    return {
        "database": str(database_path),
        "sparse": str(sparse_dir / "0"),  # COLMAP creates numbered subdirs
        "cameras": str(sparse_dir / "0" / "cameras.bin"),
        "images": str(sparse_dir / "0" / "images.bin"),
        "points3D": str(sparse_dir / "0" / "points3D.bin"),
    }


def read_colmap_poses(sparse_dir: str) -> dict:
    """
    Read camera poses from COLMAP output.
    
    Returns:
        Dict mapping image name to 4x4 pose matrix
    """
    import numpy as np
    from scipy.spatial.transform import Rotation
    
    # Parse images.bin (contains pose for each image)
    # COLMAP stores quaternion (qw, qx, qy, qz) + translation (tx, ty, tz)
    
    # This is pseudocode - actual parsing requires reading binary format
    # Use existing libraries: pycolmap or colmap_read_model
    
    poses = {}
    for image_name, (qvec, tvec) in colmap_images.items():
        # Convert quaternion to rotation matrix
        R = Rotation.from_quat([qvec[1], qvec[2], qvec[3], qvec[0]]).as_matrix()
        
        # Build 4x4 pose matrix (camera-to-world)
        pose = np.eye(4)
        pose[:3, :3] = R.T  # COLMAP stores world-to-camera, we want camera-to-world
        pose[:3, 3] = -R.T @ tvec
        
        poses[image_name] = pose
    
    return poses
```

---

#### Option D: Visual Odometry / SLAM

**What it does**: Tracks camera motion through a video sequence

**Pros**:
- ✅ Designed for video sequences
- ✅ Can handle smooth motion
- ✅ Real-time capable

**Cons**:
- ❌ May drift over time
- ❌ Requires tuning for indoor scenes

**Options**:
- **DROID-SLAM**: Deep learning-based, very accurate
- **ORB-SLAM3**: Classical approach, well-established
- **DPV-SLAM**: Hybrid approach

---

#### Option E: DUSt3R / MASt3R

**What it does**: Deep learning model that estimates relative poses and dense correspondences between image pairs

**Pros**:
- ✅ Works on textureless regions
- ✅ Produces dense depth as byproduct
- ✅ Very robust

**Cons**:
- ❌ Newer, less established
- ❌ Requires GPU
- ❌ Pairwise → need to chain for multi-view

**Could be useful for Gen3C frames** because interior scenes often have textureless walls.

---

### Pose Estimation Options Summary

| Option | Complexity | Accuracy | Best For |
|--------|------------|----------|----------|
| **A: ViPE** ⭐ | Medium | High | Gen3C videos specifically |
| **B: Trajectory Math** | Low | Exact (if correct) | When trajectory params are known |
| **C: COLMAP** | Medium | High | Well-textured scenes |
| **D: SLAM** | High | Medium-High | Long video sequences |
| **E: DUSt3R** | Medium | High | Textureless interiors |

**Recommendation for Gen3C → 2DGS**:
1. **First try**: ViPE (designed for this exact use case)
2. **Fallback**: Trajectory math reconstruction (if we can find Gen3C's source)
3. **Alternative**: DUSt3R (if ViPE unavailable)

### Step 4 & 5: 2DGS Training and Mesh Extraction

Once we have frames + poses, this follows the standard 2DGS pipeline documented in `ArkRunr_2DGS.md`.

### What Needs to Be Accomplished

| Task | Complexity | Dependencies | Status |
|------|------------|--------------|--------|
| Gen3C video generation | Low | ✅ Already working | ✅ Done |
| **ViPE installation** | Medium | Separate conda env | 🔨 1-2 hours |
| **ViPE pose extraction** | Low | ViPE | 🔨 Quick |
| Frame extraction script | Low | ffmpeg or OpenCV | 🔨 Easy |
| Trajectory math reconstruction | Medium | Gen3C source analysis | 🔨 Research needed |
| COLMAP installation | Medium | CUDA, build from source | 🔨 1-2 hours |
| COLMAP pose estimation | Medium | COLMAP | 🔨 1 day |
| DUSt3R installation | Medium | PyTorch, CUDA | 🔨 1 day |
| DUSt3R pose estimation | Medium | DUSt3R | 🔨 1-2 days |
| Pose format conversion | Low | NumPy | 🔨 Easy |
| 2DGS training integration | High | 2DGS codebase | 🔨 1 week |
| Mesh extraction | Medium | Open3D/trimesh | 🔨 2-3 days |

### Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| ViPE not publicly available | Medium | Fall back to COLMAP/DUSt3R |
| COLMAP fails on textureless interiors | Medium-High | Use DUSt3R instead |
| Gen3C motion is too smooth (frames too similar) | Medium | Extract fewer frames, use sequential matcher |
| Trajectory math doesn't match Gen3C | Medium | Use ViPE or COLMAP instead |
| Pose estimation drift | Medium | Use loop closure, bundle adjustment |
| Poses not accurate enough for 2DGS | Medium | Add depth supervision to reduce reliance on poses |

---

## Approach 2: Depth Estimation → View Synthesis → 2DGS

### Overview

Instead of generating new views with a generative model, we **warp the original image** using depth information to create synthetic views from nearby viewpoints.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                DEPTH-BASED VIEW SYNTHESIS → 2DGS PIPELINE                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  INPUT IMAGE                                                                 │
│       │                                                                      │
│       ├──────────────────────────────────────┐                              │
│       │                                      │                              │
│       ▼                                      ▼                              │
│  ┌─────────────────────────┐    ┌─────────────────────────────┐            │
│  │  DEPTH ESTIMATION       │    │  OPTIONAL: INPAINTING MODEL │            │
│  │                         │    │  (for disocclusions)        │            │
│  │  Models:                │    └─────────────────────────────┘            │
│  │  • Depth Anything V2    │                │                              │
│  │  • ZoeDepth             │                │                              │
│  │  • Marigold             │                │                              │
│  │                         │                │                              │
│  │  Output: Depth map      │                │                              │
│  └───────────┬─────────────┘                │                              │
│              │                              │                              │
│              ▼                              │                              │
│  ┌─────────────────────────────────────────────────────────────┐           │
│  │  VIEW SYNTHESIS (DEPTH-BASED WARPING)                       │           │
│  │                                                              │           │
│  │  For each target viewpoint in the ArkRunr arc:              │           │
│  │                                                              │           │
│  │  1. Define target camera pose                               │           │
│  │     • Azimuth: -60°, -45°, -30°, -15°, 0°, +15°, +30°...   │           │
│  │     • Elevation: 0°, 10°, 20° (multiple levels)            │           │
│  │     • Distance: fixed or varied                             │           │
│  │                                                              │           │
│  │  2. Project pixels using depth                              │           │
│  │     • pixel → 3D point → new pixel location                │           │
│  │                                                              │           │
│  │  3. Handle disocclusions                                    │           │
│  │     • Areas revealed that weren't visible in original      │           │
│  │     • Options: mask, inpaint, or leave black               │           │
│  │                                                              │           │
│  │  Output: N synthetic views + KNOWN poses                    │           │
│  └───────────────────────┬─────────────────────────────────────┘           │
│                          │                                                  │
│                          ▼                                                  │
│  ┌─────────────────────────────────────────────────────────────┐           │
│  │  2DGS TRAINING                                              │           │
│  │                                                              │           │
│  │  Advantages:                                                 │           │
│  │  • Poses are KNOWN (we defined them)                        │           │
│  │  • Depth is available for supervision                       │           │
│  │  • No pose estimation errors                                │           │
│  │                                                              │           │
│  │  Challenges:                                                 │           │
│  │  • Disocclusions create holes → mask in loss               │           │
│  │  • Depth estimation errors propagate                        │           │
│  │  • Limited viewpoint range (~±60° practical)               │           │
│  │                                                              │           │
│  └───────────────────────┬─────────────────────────────────────┘           │
│                          │                                                  │
│                          ▼                                                  │
│  ┌─────────────────────────────────────────────────────────────┐           │
│  │  MESH EXTRACTION (TSDF)                                     │           │
│  │                                                              │           │
│  │  • Render depth from multiple viewpoints                    │           │
│  │  • Fuse into TSDF volume                                    │           │
│  │  • Marching cubes for mesh                                  │           │
│  └───────────────────────┬─────────────────────────────────────┘           │
│                          │                                                  │
│                          ▼                                                  │
│                      OUTPUT MESH                                            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Step 1: Depth Estimation

**Goal**: Get a dense depth map from the input image

#### Depth Estimation Models

| Model | Quality | Speed | Best For |
|-------|---------|-------|----------|
| **Depth Anything V2** | ⭐⭐⭐⭐⭐ | Fast | General purpose, excellent indoor |
| **ZoeDepth** | ⭐⭐⭐⭐ | Medium | Indoor scenes specifically |
| **Marigold** | ⭐⭐⭐⭐⭐ | Slow | Highest detail, diffusion-based |
| **MiDaS** | ⭐⭐⭐ | Fast | Good baseline |

**Recommendation**: Start with **Depth Anything V2** — it's fast, accurate, and works well indoors.

#### Implementation

```python
import torch
import numpy as np
from PIL import Image

def estimate_depth_anything_v2(image_path: str) -> np.ndarray:
    """
    Estimate depth using Depth Anything V2.
    
    Returns:
        Depth map as numpy array (H, W), values in relative depth
    """
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    
    # Load model
    processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Large-hf")
    model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Large-hf")
    model.to("cuda")
    
    # Load and process image
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to("cuda")
    
    # Estimate depth
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth
    
    # Interpolate to original size
    depth = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],  # (H, W)
        mode="bicubic",
        align_corners=False,
    ).squeeze().cpu().numpy()
    
    return depth


def depth_to_metric(depth_relative: np.ndarray, scale: float = 10.0) -> np.ndarray:
    """
    Convert relative depth to approximate metric depth.
    
    Monocular depth is scale-ambiguous. We assume a reasonable scale
    based on expected room dimensions.
    
    Args:
        depth_relative: Relative depth (0-1 or arbitrary range)
        scale: Expected depth range in meters
    
    Returns:
        Metric depth in meters
    """
    # Normalize to 0-1
    depth_norm = (depth_relative - depth_relative.min()) / (depth_relative.max() - depth_relative.min())
    
    # Scale to metric (assuming far = scale meters, near = 0.5 meters)
    depth_metric = 0.5 + depth_norm * (scale - 0.5)
    
    return depth_metric
```

### Step 2: View Synthesis (Depth-Based Warping)

**Goal**: Generate new views of the scene by projecting the original image through depth

#### The Math

```
Given:
• Original image I at pose P₀
• Depth map D
• Target pose P₁
• Camera intrinsics K

For each pixel (u, v) in the original image:

1. UNPROJECT to 3D:
   z = D[v, u]                          # Depth at pixel
   x = (u - cx) * z / fx                # 3D X coordinate
   y = (v - cy) * z / fy                # 3D Y coordinate
   point_3d = [x, y, z, 1]              # Homogeneous 3D point

2. TRANSFORM to new camera frame:
   # P₀⁻¹ @ P₁ gives relative transform
   point_new = P₁⁻¹ @ P₀ @ point_3d

3. PROJECT to new image:
   u' = fx * point_new[0] / point_new[2] + cx
   v' = fy * point_new[1] / point_new[2] + cy

4. SAMPLE color from original image:
   warped_image[v', u'] = I[v, u]
```

#### Implementation

```python
import numpy as np
import cv2
from typing import Tuple

def warp_image_with_depth(
    image: np.ndarray,
    depth: np.ndarray,
    source_pose: np.ndarray,
    target_pose: np.ndarray,
    intrinsics: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Warp image to new viewpoint using depth.
    
    Args:
        image: Source image (H, W, 3)
        depth: Depth map (H, W) in meters
        source_pose: 4x4 camera-to-world matrix for source
        target_pose: 4x4 camera-to-world matrix for target
        intrinsics: 3x3 camera intrinsic matrix K
    
    Returns:
        Tuple of (warped_image, valid_mask)
        - warped_image: Image from new viewpoint (H, W, 3)
        - valid_mask: Binary mask of valid pixels (H, W)
    """
    H, W = depth.shape
    
    # Create pixel coordinate grid
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    ones = np.ones_like(u)
    
    # Pixel coordinates (homogeneous)
    pixels = np.stack([u, v, ones], axis=-1).reshape(-1, 3).T  # (3, H*W)
    
    # Unproject to 3D (in source camera frame)
    K_inv = np.linalg.inv(intrinsics)
    depths_flat = depth.flatten()
    points_cam = K_inv @ pixels * depths_flat  # (3, H*W)
    
    # Convert to homogeneous
    points_cam_h = np.vstack([points_cam, np.ones((1, H*W))])  # (4, H*W)
    
    # Transform: source camera → world → target camera
    source_to_world = source_pose
    world_to_target = np.linalg.inv(target_pose)
    transform = world_to_target @ source_to_world
    
    points_target = transform @ points_cam_h  # (4, H*W)
    
    # Project to target image
    points_target_3d = points_target[:3, :]  # (3, H*W)
    
    # Avoid division by zero
    z = points_target_3d[2, :]
    valid = z > 0.1  # Points must be in front of camera
    
    # Project
    pixels_target = intrinsics @ points_target_3d
    pixels_target = pixels_target[:2, :] / pixels_target[2:, :]  # (2, H*W)
    
    u_target = pixels_target[0, :].reshape(H, W)
    v_target = pixels_target[1, :].reshape(H, W)
    
    # Remap
    map_x = u_target.astype(np.float32)
    map_y = v_target.astype(np.float32)
    
    warped = cv2.remap(
        image,
        u.astype(np.float32),  # Source coordinates
        v.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    
    # Create output by forward mapping (handles occlusions better)
    warped_image = np.zeros_like(image)
    valid_mask = np.zeros((H, W), dtype=np.uint8)
    z_buffer = np.full((H, W), np.inf)
    
    for i in range(H):
        for j in range(W):
            if not valid[i * W + j]:
                continue
            
            u_t = int(round(u_target[i, j]))
            v_t = int(round(v_target[i, j]))
            
            if 0 <= u_t < W and 0 <= v_t < H:
                z_t = points_target_3d[2, i * W + j]
                if z_t < z_buffer[v_t, u_t]:
                    z_buffer[v_t, u_t] = z_t
                    warped_image[v_t, u_t] = image[i, j]
                    valid_mask[v_t, u_t] = 1
    
    return warped_image, valid_mask


def generate_arkrunr_views(
    image: np.ndarray,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    num_azimuth: int = 7,
    num_elevation: int = 3,
    arc_degrees: float = 120.0,
    elevation_range: Tuple[float, float] = (-10.0, 25.0),
    distance: float = 5.0,
) -> Tuple[list, list, list]:
    """
    Generate synthetic views across ArkRunr viewing arc.
    
    Args:
        image: Original image (H, W, 3)
        depth: Depth map (H, W)
        intrinsics: Camera intrinsic matrix
        num_azimuth: Number of horizontal positions
        num_elevation: Number of vertical positions
        arc_degrees: Total horizontal arc
        elevation_range: (min, max) elevation in degrees
        distance: Camera distance from center
    
    Returns:
        Tuple of (images, masks, poses)
        - images: List of warped images
        - masks: List of valid pixel masks
        - poses: List of 4x4 pose matrices
    """
    images = []
    masks = []
    poses = []
    
    # Source pose is identity (original camera at origin looking down -Z)
    source_pose = np.eye(4)
    
    # Generate view positions
    arc_half = arc_degrees / 2
    azimuths = np.linspace(-arc_half, arc_half, num_azimuth)
    elevations = np.linspace(elevation_range[0], elevation_range[1], num_elevation)
    
    for elevation in elevations:
        for azimuth in azimuths:
            # Skip center view (it's the original)
            if abs(azimuth) < 1 and abs(elevation) < 1:
                images.append(image.copy())
                masks.append(np.ones((image.shape[0], image.shape[1]), dtype=np.uint8))
                poses.append(source_pose.copy())
                continue
            
            # Create target pose
            target_pose = azimuth_elevation_to_pose(azimuth, elevation, distance)
            
            # Warp image
            warped, mask = warp_image_with_depth(
                image, depth, source_pose, target_pose, intrinsics
            )
            
            images.append(warped)
            masks.append(mask)
            poses.append(target_pose)
            
            print(f"Generated view: az={azimuth:.1f}°, el={elevation:.1f}°")
    
    return images, masks, poses


def azimuth_elevation_to_pose(
    azimuth: float,
    elevation: float,
    distance: float,
) -> np.ndarray:
    """
    Convert spherical coordinates to 4x4 camera pose.
    
    Camera looks at origin from specified position.
    """
    az_rad = np.radians(azimuth)
    el_rad = np.radians(elevation)
    
    # Position on sphere
    x = distance * np.cos(el_rad) * np.sin(az_rad)
    y = distance * np.sin(el_rad)
    z = distance * np.cos(el_rad) * np.cos(az_rad)
    
    position = np.array([x, y, z])
    
    # Look at origin
    forward = -position / np.linalg.norm(position)
    up = np.array([0, 1, 0])
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    
    # Rotation matrix
    R = np.stack([right, up, -forward], axis=1)
    
    # Pose matrix (camera-to-world)
    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = position
    
    return pose
```

### Step 3: Handling Disocclusions

**The Problem**: When we move the camera, parts of the scene that were hidden become visible. These areas have no color information in the original image.

```
Original View:                     Warped View (moved right):
┌─────────────────────────┐       ┌─────────────────────────┐
│                         │       │▓▓▓                      │
│    ┌───────┐            │       │▓▓▓┌───────┐             │
│    │PILLAR │  WALL      │  ──►  │▓▓▓│PILLAR │  WALL       │
│    │       │            │       │▓▓▓│       │             │
│    └───────┘            │       │▓▓▓└───────┘             │
│                         │       │▓▓▓                      │
└─────────────────────────┘       └─────────────────────────┘
                                   ▓▓▓ = Disoccluded region
                                        (no data in original)
```

#### Options for Handling Disocclusions

| Option | Pros | Cons |
|--------|------|------|
| **Mask out** | Simple, honest | Holes in training data |
| **Inpaint** | Complete images | May introduce artifacts |
| **Black fill** | Simple | May confuse 2DGS |
| **Masked loss** | Best for 2DGS | Need to propagate masks |

**Recommendation**: Use **masked loss** in 2DGS training — only compute loss on valid pixels.

```python
def compute_masked_loss(rendered, target, mask):
    """
    Compute loss only on valid pixels.
    """
    mask = mask.float()
    valid_pixels = mask.sum()
    
    if valid_pixels == 0:
        return torch.tensor(0.0)
    
    loss = ((rendered - target) ** 2 * mask).sum() / valid_pixels
    return loss
```

### What Needs to Be Accomplished

| Task | Complexity | Dependencies | Status |
|------|------------|--------------|--------|
| Depth Anything V2 integration | Low | transformers, torch | 🔨 1-2 hours |
| Depth-based warping function | Medium | numpy, OpenCV | 🔨 1 day |
| ArkRunr view generation | Low | Warping function | 🔨 2-3 hours |
| Disocclusion handling | Low | Masking in loss | 🔨 2-3 hours |
| 2DGS training with masks | Medium | 2DGS codebase | 🔨 2-3 days |
| End-to-end pipeline | Medium | All above | 🔨 1-2 days |
| Mesh extraction | Medium | Open3D/trimesh | 🔨 2-3 days |

### Advantages of This Approach

1. **Known poses** — We define the target poses, no estimation needed
2. **Depth supervision** — Depth map can supervise 2DGS training
3. **No hallucination** — Only uses actual image data
4. **Fast** — No generative model inference for views
5. **Controllable** — Exact viewpoints we specify

### Limitations

1. **Limited viewpoint range** — Practical limit ~±60° before too much disocclusion
2. **Depth errors propagate** — Inaccurate depth → incorrect warps
3. **Holes in warped views** — Disocclusions must be handled
4. **No new content** — Can't see what was originally hidden

---

## Comparison and Recommendations

### Side-by-Side Comparison

| Aspect | Gen3C + ViPE | Gen3C + COLMAP | Depth Warping |
|--------|-------------|----------------|---------------|
| **View Quality** | ⭐⭐⭐⭐⭐ High (AI-generated) | ⭐⭐⭐⭐⭐ High (AI-generated) | ⭐⭐⭐ Medium (warped) |
| **Pose Accuracy** | ⭐⭐⭐⭐ ViPE estimates | ⭐⭐⭐ May fail on interiors | ⭐⭐⭐⭐⭐ Exact (defined) |
| **Viewpoint Range** | ⭐⭐⭐⭐ Depends on trajectory | ⭐⭐⭐⭐ Depends on trajectory | ⭐⭐⭐ ~±60° practical |
| **Disocclusions** | ⭐⭐⭐⭐⭐ AI fills in | ⭐⭐⭐⭐⭐ AI fills in | ⭐⭐ Must mask/handle |
| **Depth Available** | ⭐⭐⭐⭐⭐ ViPE provides | ⭐⭐ Need separate | ⭐⭐⭐⭐⭐ Yes (direct) |
| **Implementation** | ⭐⭐⭐ Medium | ⭐⭐ Complex | ⭐⭐⭐⭐ Straightforward |
| **Dependencies** | ViPE (separate env) | COLMAP (may fail) | Depth model only |
| **Speed** | Slow (video + ViPE) | Slow (video + COLMAP) | Fast |

### Updated Recommendation

**NEW INSIGHT**: With ViPE available, the Gen3C approach becomes much more viable:

#### RECOMMENDED: Two-Track Parallel Approach

```
TRACK A: Gen3C + ViPE (Higher Quality Potential)
════════════════════════════════════════════════
• Gen3C generates high-quality interior views
• ViPE extracts both depth AND poses
• Best view quality for 2DGS training
• Handles disocclusions naturally

TRACK B: Depth Warping (Lower Risk, Faster)
════════════════════════════════════════════════
• Known poses (no estimation)
• Fast iteration
• Good for ~±60° arc
• May have disocclusion issues
```

### Suggested Order (REVISED)

```
PHASE 1: Quick Evaluation (1 week)
──────────────────────────────────────────────────
Day 1-2:
├── Install ViPE (check if publicly available)
├── Test on existing Gen3C video
└── Check ViPE output format

Day 3-4:
├── Install Depth Anything V2
├── Test depth-based warping on sample image
└── Compare view quality

Day 5-7:
├── Evaluate both approaches
└── Decide which to pursue first

PHASE 2A: Gen3C + ViPE Pipeline (if ViPE works)
──────────────────────────────────────────────────
Week 2:
├── Generate Gen3C video with clockwise/counterclockwise
├── Run ViPE to get poses + depth
└── Format data for 2DGS

Week 3:
├── 2DGS training on Gen3C frames
├── Mesh extraction
└── Quality evaluation

PHASE 2B: Depth Warping Pipeline (parallel or if ViPE fails)
──────────────────────────────────────────────────
Week 2:
├── Depth-based view generation
├── Handle disocclusions
└── Format data for 2DGS

Week 3:
├── 2DGS training
├── Mesh extraction
└── Quality evaluation

PHASE 3: Compare and Optimize (Week 4)
──────────────────────────────────────────────────
├── Compare mesh quality from both approaches
├── Test on diverse interior types
└── Select final pipeline
```

### Key Decision Points

1. **Is ViPE publicly available?**
   - If YES → Gen3C + ViPE is likely best
   - If NO → Fall back to depth warping or COLMAP

2. **Does Gen3C's trajectory work for interiors?**
   - `clockwise`/`counterclockwise` might work
   - `left`/`right` might give parallax views
   - Test needed to verify

3. **Is ~±60° viewpoint range sufficient?**
   - For ArkRunr's ~120° arc → YES
   - Depth warping should work

---

## Implementation Priority

### Immediate Next Steps

#### Step 1: Check ViPE Availability (TODAY)

ViPE is NVIDIA's official tool for extracting poses from video. Check if it's available:

```bash
# Search for ViPE repository
# Look for: github.com/nv-tlabs/ViPE or similar

# If found, check installation requirements
# Note: ViPE requires separate conda environment
```

**Action**: Search NVIDIA's GitHub for ViPE, or check if it's mentioned in Gen3C docs.

#### Step 2: Test Gen3C Trajectory Options

Generate videos with different trajectories to see which works best for interiors:

```bash
# On RunPod, test different trajectories:
python gen3c_single_image.py \
    --input_image_path your_interior.png \
    --trajectory clockwise \         # Try: clockwise, counterclockwise, left, right
    --movement_distance 0.3 \        # Try: 0.2, 0.3, 0.5
    --camera_rotation center_facing  # Try: center_facing, no_rotation
```

#### Step 3: Test Depth Anything V2 (Parallel)

```bash
# Install
pip install transformers torch

# Test
python -c "
from transformers import AutoModelForDepthEstimation
model = AutoModelForDepthEstimation.from_pretrained('depth-anything/Depth-Anything-V2-Large-hf')
print('Depth Anything V2 loaded successfully!')
"
```

#### Step 4: Implement Basic Warping

Test depth-based view synthesis on your BrightClub image.

### Files to Create

```
/home/arkrunr02/Hunyuan3D-2-Fork/
├── generators/
│   ├── depth_estimation.py      # Depth Anything V2 wrapper
│   ├── view_synthesis.py        # Depth-based warping
│   ├── vipe_wrapper.py          # ViPE integration (if available)
│   └── arkrunr_interior.py      # Combined pipeline
├── scripts/
│   ├── test_depth_estimation.py # Quick test
│   ├── test_gen3c_trajectories.py # Compare trajectories
│   └── generate_arkrunr_views.py # Generate views for a sample
└── runpod/
    └── arkrunr_interior/
        ├── handler.py           # RunPod serverless handler
        └── requirements.txt     # Dependencies
```

### Critical Questions to Answer

1. **Is ViPE publicly available?**
   - Check NVIDIA repos
   - If not, can we request access?

2. **Which Gen3C trajectory works best for interiors?**
   - `clockwise`/`counterclockwise` for orbital views
   - `left`/`right` for parallax motion

3. **Can we reconstruct poses from Gen3C parameters?**
   - Examine Gen3C source code for trajectory math
   - If deterministic, we can skip ViPE

---

## Summary: Two Viable Paths Forward

### Path 1: Gen3C → ViPE → 2DGS (Recommended if ViPE available)

```
Image → Gen3C (clockwise) → Video → ViPE → Frames + Poses + Depth → 2DGS → Mesh
```

**Pros**: High-quality views, handles disocclusions, official NVIDIA workflow
**Cons**: Requires ViPE installation, slower

### Path 2: Depth Warping → 2DGS (Recommended if ViPE unavailable)

```
Image → Depth Estimation → Warp to multiple views → 2DGS → Mesh
```

**Pros**: Fast, no pose estimation needed, controllable
**Cons**: Limited viewpoint range, disocclusion issues

Both paths end with 2DGS training → TSDF mesh extraction, which you need regardless.