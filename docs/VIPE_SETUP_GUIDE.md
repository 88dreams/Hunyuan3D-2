# ViPE Setup Guide for ArkRunr

**Created:** January 9, 2026  
**Purpose:** Set up NVIDIA ViPE for extracting camera poses and depth from Gen3C videos

---

## What is ViPE?

**ViPE (Video Pose Engine)** is NVIDIA's tool for extracting:
- Camera intrinsics (focal length, principal point)
- Camera poses (4×4 extrinsic matrices)
- Dense depth maps

From the [NVIDIA Research page](https://research.nvidia.com/labs/toronto-ai/vipe/):
> "ViPE efficiently estimates per-frame camera intrinsics, poses, and dense, near-metric depth maps by solving a dense bundle adjustment problem over keyframes."

**This is exactly what we need for the Gen3C → 2DGS pipeline!**

---

## Repository

**GitHub**: [https://github.com/nv-tlabs/vipe](https://github.com/nv-tlabs/vipe)

---

## Installation

### Prerequisites

- NVIDIA GPU with CUDA support
- Python 3.10+
- ~20GB disk space for models

### Important: Separate Environment Required

ViPE's dependencies may conflict with Gen3C. Install in a **separate conda environment**.

### Step-by-Step Installation

```bash
# 1. Create new conda environment
conda create -n vipe python=3.10
conda activate vipe

# 2. Clone ViPE repository
cd /workspace  # Or your preferred location
git clone https://github.com/nv-tlabs/vipe.git
cd vipe

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify installation
vipe --help
```

### Expected Output from `vipe --help`

```
Usage: vipe [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  infer      Process a video to extract poses and depth
  visualize  Visualize ViPE results
```

---

## Usage

### Basic Usage: Process Gen3C Video

```bash
# Activate ViPE environment
conda activate vipe

# Process a Gen3C video
vipe infer /path/to/gen3c_video.mp4 --output /path/to/vipe_results/

# With visualization (optional)
vipe infer /path/to/gen3c_video.mp4 --output /path/to/vipe_results/ --visualize
```

### Command Options

| Option | Default | Description |
|--------|---------|-------------|
| `--output` | `vipe_results/` | Output directory |
| `--visualize` | `false` | Enable visualization |
| `--pipeline` | `default` | Pipeline configuration |

---

## Output Format

ViPE outputs are stored in the specified output directory:

```
vipe_results/
├── depth/                    # Dense depth maps per frame
│   ├── 000000.npy           # Depth as numpy array (H, W)
│   ├── 000001.npy
│   └── ...
├── intrinsics.npy           # Camera intrinsics (N × 3 × 3)
├── poses.npy                # Camera poses (N × 4 × 4)
├── timestamps.npy           # Frame timestamps
├── config.yaml              # Processing configuration
└── visualization/           # Optional visualization outputs
    ├── depth_colored/
    └── trajectory.ply
```

### Reading ViPE Output

```python
import numpy as np

def load_vipe_results(vipe_dir: str) -> dict:
    """
    Load ViPE output for use with 2DGS.
    
    Returns:
        Dict with:
        - poses: (N, 4, 4) camera-to-world matrices
        - intrinsics: (N, 3, 3) camera intrinsic matrices
        - depths: List of (H, W) depth arrays
    """
    vipe_dir = Path(vipe_dir)
    
    # Load poses (N, 4, 4)
    poses = np.load(vipe_dir / "poses.npy")
    
    # Load intrinsics (N, 3, 3) or (3, 3) if shared
    intrinsics = np.load(vipe_dir / "intrinsics.npy")
    
    # Load depth maps
    depth_dir = vipe_dir / "depth"
    depth_files = sorted(depth_dir.glob("*.npy"))
    depths = [np.load(f) for f in depth_files]
    
    return {
        "poses": poses,
        "intrinsics": intrinsics,
        "depths": depths,
        "num_frames": len(depths),
    }
```

---

## Integration with 2DGS

### Pipeline: Gen3C → ViPE → 2DGS

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    GEN3C → VIPE → 2DGS PIPELINE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  STEP 1: Generate Video with Gen3C                                          │
│  ─────────────────────────────────                                          │
│  conda activate gen3c                                                        │
│  python gen3c_single_image.py \                                              │
│      --input_image_path interior.png \                                       │
│      --trajectory clockwise \                                                │
│      --movement_distance 0.3 \                                               │
│      --video_save_name interior_orbit                                        │
│                                                                              │
│  Output: interior_orbit.mp4                                                  │
│                                                                              │
│  STEP 2: Extract Poses + Depth with ViPE                                    │
│  ───────────────────────────────────────                                    │
│  conda activate vipe                                                         │
│  vipe infer interior_orbit.mp4 --output vipe_results/                       │
│                                                                              │
│  Output: vipe_results/                                                       │
│          ├── poses.npy (N × 4 × 4)                                          │
│          ├── intrinsics.npy                                                  │
│          └── depth/*.npy                                                     │
│                                                                              │
│  STEP 3: Extract Frames from Video                                          │
│  ────────────────────────────────                                           │
│  ffmpeg -i interior_orbit.mp4 -q:v 2 frames/frame_%04d.png                  │
│                                                                              │
│  Output: frames/frame_0001.png, frame_0002.png, ...                         │
│                                                                              │
│  STEP 4: Prepare Data for 2DGS                                              │
│  ─────────────────────────────                                              │
│  python prepare_2dgs_data.py \                                               │
│      --frames_dir frames/ \                                                  │
│      --vipe_dir vipe_results/ \                                              │
│      --output 2dgs_data/                                                     │
│                                                                              │
│  Output: 2dgs_data/ (COLMAP-compatible format)                              │
│                                                                              │
│  STEP 5: Train 2DGS                                                         │
│  ─────────────────                                                          │
│  python train_2dgs.py --data 2dgs_data/ --iterations 30000                  │
│                                                                              │
│  STEP 6: Extract Mesh                                                       │
│  ───────────────────                                                        │
│  python extract_mesh.py --model 2dgs_output/ --output stage.glb             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Converting ViPE to COLMAP Format

ViPE includes a script to convert its output to COLMAP format:

```bash
python scripts/vipe_to_colmap.py vipe_results/ --sequence interior
```

This creates a COLMAP-compatible sparse reconstruction that can be used directly with 2DGS.

---

## RunPod Deployment

### Option 1: Same Pod, Separate Environment

```bash
# On RunPod with Gen3C already installed

# Create ViPE environment
conda create -n vipe python=3.10
conda activate vipe

# Install ViPE
cd /workspace
git clone https://github.com/nv-tlabs/vipe.git
cd vipe
pip install -r requirements.txt
```

### Option 2: Separate RunPod Pod

If memory is a concern, run ViPE on a separate smaller pod.

---

## Troubleshooting

### Memory Issues

ViPE processes videos at 3-5 FPS. For a 121-frame Gen3C video:
- Processing time: ~25-40 seconds
- GPU memory: ~8-12 GB

If running out of memory:
```bash
# Process with lower resolution
vipe infer video.mp4 --output results/ --max_resolution 720
```

### Dependency Conflicts

If you see conflicts with Gen3C packages:
```bash
# Ensure you're in the ViPE environment
conda activate vipe
which python  # Should show vipe environment

# If still conflicts, create fresh environment
conda create -n vipe_fresh python=3.10 --yes
conda activate vipe_fresh
pip install --no-cache-dir -r requirements.txt
```

---

## Full Integration Script

```python
#!/usr/bin/env python3
"""
gen3c_vipe_2dgs.py

Complete pipeline: Gen3C → ViPE → 2DGS → Mesh
"""

import subprocess
import sys
from pathlib import Path
import numpy as np
import cv2


def run_gen3c(
    image_path: str,
    output_name: str,
    trajectory: str = "clockwise",
    movement_distance: float = 0.3,
) -> str:
    """Generate video with Gen3C."""
    cmd = [
        "python", "cosmos_predict1/diffusion/inference/gen3c_single_image.py",
        "--input_image_path", image_path,
        "--video_save_name", output_name,
        "--trajectory", trajectory,
        "--movement_distance", str(movement_distance),
        "--num_video_frames", "121",
    ]
    subprocess.run(cmd, check=True)
    return f"videos/{output_name}.mp4"


def run_vipe(video_path: str, output_dir: str) -> str:
    """Extract poses and depth with ViPE."""
    cmd = ["vipe", "infer", video_path, "--output", output_dir]
    subprocess.run(cmd, check=True)
    return output_dir


def extract_frames(video_path: str, output_dir: str) -> list:
    """Extract frames from video."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "ffmpeg", "-i", video_path,
        "-q:v", "2",
        str(output_dir / "frame_%04d.png")
    ]
    subprocess.run(cmd, check=True)
    
    return sorted(output_dir.glob("*.png"))


def prepare_2dgs_data(
    frames_dir: str,
    vipe_dir: str,
    output_dir: str,
) -> str:
    """
    Convert ViPE output to 2DGS-compatible format.
    """
    frames_dir = Path(frames_dir)
    vipe_dir = Path(vipe_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load ViPE results
    poses = np.load(vipe_dir / "poses.npy")
    intrinsics = np.load(vipe_dir / "intrinsics.npy")
    
    # Get frame files
    frame_files = sorted(frames_dir.glob("*.png"))
    
    # Create images directory
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    # Copy frames
    for i, frame_file in enumerate(frame_files):
        dest = images_dir / f"{i:06d}.png"
        shutil.copy(frame_file, dest)
    
    # Create cameras.txt (COLMAP format)
    with open(output_dir / "cameras.txt", "w") as f:
        # Assuming single camera model
        K = intrinsics[0] if intrinsics.ndim == 3 else intrinsics
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        
        img = cv2.imread(str(frame_files[0]))
        h, w = img.shape[:2]
        
        f.write(f"# Camera list with one line of data per camera:\n")
        f.write(f"# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"1 PINHOLE {w} {h} {fx} {fy} {cx} {cy}\n")
    
    # Create images.txt (COLMAP format)
    with open(output_dir / "images.txt", "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        
        for i, pose in enumerate(poses):
            # Convert pose to COLMAP format (world-to-camera)
            pose_inv = np.linalg.inv(pose)
            R = pose_inv[:3, :3]
            t = pose_inv[:3, 3]
            
            # Convert rotation matrix to quaternion
            from scipy.spatial.transform import Rotation
            quat = Rotation.from_matrix(R).as_quat()  # [x, y, z, w]
            qw, qx, qy, qz = quat[3], quat[0], quat[1], quat[2]
            
            f.write(f"{i+1} {qw} {qx} {qy} {qz} {t[0]} {t[1]} {t[2]} 1 {i:06d}.png\n")
            f.write("\n")  # Empty line for points
    
    # Create empty points3D.txt
    with open(output_dir / "points3D.txt", "w") as f:
        f.write("# 3D point list (empty for now)\n")
    
    print(f"2DGS data prepared at: {output_dir}")
    return str(output_dir)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Gen3C → ViPE → 2DGS Pipeline")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--output", default="pipeline_output", help="Output directory")
    parser.add_argument("--trajectory", default="clockwise", help="Gen3C trajectory")
    parser.add_argument("--movement", type=float, default=0.3, help="Movement distance")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Gen3C
    print("Step 1: Running Gen3C...")
    video_path = run_gen3c(
        args.image,
        "gen3c_output",
        args.trajectory,
        args.movement,
    )
    
    # Step 2: ViPE
    print("Step 2: Running ViPE...")
    vipe_dir = run_vipe(video_path, str(output_dir / "vipe"))
    
    # Step 3: Extract frames
    print("Step 3: Extracting frames...")
    frames = extract_frames(video_path, str(output_dir / "frames"))
    
    # Step 4: Prepare 2DGS data
    print("Step 4: Preparing 2DGS data...")
    data_dir = prepare_2dgs_data(
        str(output_dir / "frames"),
        vipe_dir,
        str(output_dir / "2dgs_data"),
    )
    
    print(f"\nPipeline complete!")
    print(f"2DGS data ready at: {data_dir}")
    print(f"\nNext: Train 2DGS with:")
    print(f"  python train.py -s {data_dir} -m {output_dir}/2dgs_model")


if __name__ == "__main__":
    main()
```

---

## Next Steps After ViPE Setup

1. **Test ViPE on existing Gen3C video**
   ```bash
   conda activate vipe
   vipe infer /workspace/outputs/gen3c/your_video.mp4 --output vipe_test/
   ```

2. **Inspect output**
   ```python
   import numpy as np
   poses = np.load("vipe_test/poses.npy")
   print(f"Poses shape: {poses.shape}")  # Should be (N, 4, 4)
   ```

3. **Prepare data for 2DGS** (see integration script above)

4. **Train 2DGS** on the prepared data

---

## Summary

| Component | What It Does | Location |
|-----------|-------------|----------|
| **Gen3C** | Generates high-quality orbital video | Already on RunPod |
| **ViPE** | Extracts poses + depth from video | github.com/nv-tlabs/vipe |
| **2DGS** | Trains 3D representation | To be set up |
| **Mesh Extraction** | Converts 2DGS to mesh | TSDF fusion |

**ViPE is the missing link** that gives us accurate camera poses for 2DGS training!
