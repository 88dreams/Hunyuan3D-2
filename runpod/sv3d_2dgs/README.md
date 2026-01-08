# SV3D → 2DGS Pipeline for RunPod

This directory contains everything needed to run the SV3D multi-view generation → 2DGS training → mesh extraction pipeline on RunPod.

## Files

| File | Purpose |
|------|---------|
| `setup_sv3d_2dgs.sh` | Setup script for interactive development on a RunPod pod |
| `Dockerfile` | Docker image definition for reproducible environment |
| `test_pipeline.py` | Complete pipeline test script |
| `README.md` | This file |

---

## Quick Start (Interactive Development)

### 1. Start a RunPod Pod

1. Go to [RunPod Console](https://www.runpod.io/console/pods)
2. Click "Deploy"
3. Choose:
   - **GPU**: RTX A6000 (48GB) or A100 (40GB) recommended
   - **Template**: `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel`
   - **Container Disk**: 50GB
   - **Volume**: Mount your existing network volume (optional)

### 2. Connect to the Pod

```bash
# SSH into the pod (use the SSH command from RunPod console)
ssh root@<pod-ip> -p <port> -i ~/.ssh/id_ed25519
```

### 3. Run Setup Script

```bash
# Download and run setup script
cd /workspace
wget https://raw.githubusercontent.com/88dreams/Hunyuan3D-2/2dgs/runpod/sv3d_2dgs/setup_sv3d_2dgs.sh
bash setup_sv3d_2dgs.sh
```

Or if you have the repo cloned:

```bash
cd /workspace
git clone https://github.com/88dreams/Hunyuan3D-2.git -b 2dgs
bash Hunyuan3D-2/runpod/sv3d_2dgs/setup_sv3d_2dgs.sh
```

### 4. Run Quick Test

```bash
# Verify installation
python /workspace/sv3d_2dgs/quick_test.py
```

### 5. Test the Pipeline

```bash
# Copy your test image
cp /path/to/your/image.jpg /workspace/sv3d_2dgs/inputs/

# Run quick test (5k iterations, ~10 minutes)
python /workspace/sv3d_2dgs/test_pipeline.py --image /workspace/sv3d_2dgs/inputs/your_image.jpg --quick

# Or full test (15k iterations, ~30 minutes)
python /workspace/sv3d_2dgs/test_pipeline.py --image /workspace/sv3d_2dgs/inputs/your_image.jpg
```

---

## Using the Dockerfile

### Build the Image

```bash
# On a machine with Docker
cd runpod/sv3d_2dgs
docker build -t 88dreams/sv3d-2dgs-runpod:v1 -f Dockerfile .
```

### Push to Docker Hub

```bash
docker push 88dreams/sv3d-2dgs-runpod:v1
```

### Use as RunPod Template

1. Go to RunPod Console → Templates
2. Create new template
3. Set Docker image: `88dreams/sv3d-2dgs-runpod:v1`
4. Deploy pods using this template

---

## Pipeline Stages

```
┌─────────────────────────────────────────────────────────────────┐
│                    PIPELINE OVERVIEW                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Stage 1: SV3D Multi-View Generation (~3 minutes)               │
│  ─────────────────────────────────────                          │
│  Input:  Single image                                           │
│  Output: 21 orbital views + camera poses                        │
│  Model:  stabilityai/sv3d                                       │
│                                                                  │
│  Stage 2: Depth Estimation (~1 minute)                          │
│  ─────────────────────────────────────                          │
│  Input:  Multi-view images                                      │
│  Output: Depth maps for each view                               │
│  Model:  depth-anything/Depth-Anything-V2-Large                 │
│                                                                  │
│  Stage 3: 2DGS Training (~15-30 minutes)                        │
│  ─────────────────────────────────────                          │
│  Input:  Images + poses + depths                                │
│  Output: Trained 2DGS model (PLY)                               │
│  Repo:   hbb1/2d-gaussian-splatting                             │
│                                                                  │
│  Stage 4: Mesh Extraction (~2 minutes)                          │
│  ─────────────────────────────────────                          │
│  Input:  2DGS point cloud                                       │
│  Output: Mesh (GLB)                                             │
│  Method: Poisson reconstruction                                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Configuration Options

### SV3D Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_frames` | 21 | Number of orbital views to generate |
| `front_arc_only` | True | Only keep front-facing views (180°) |
| `arc_degrees` | 180.0 | Arc width for filtering |
| `remove_background` | False | Remove background (False for interiors) |

### 2DGS Training Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `iterations` | 15000 | Training iterations (5000 for quick) |
| `depth_ratio` | 1.0 | Depth distortion loss weight |
| `lambda_normal` | 0.05 | Normal consistency loss weight |

### Mesh Extraction Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `target_triangles` | 100000 | Target mesh triangle count |
| `smooth_iterations` | 2 | Laplacian smoothing iterations |

---

## Output Structure

After running the pipeline:

```
/workspace/sv3d_2dgs/outputs/YYYYMMDD_HHMMSS/
├── preprocessed_input.png     # Input after preprocessing
├── images/                    # Generated multi-view images
│   ├── frame_0000.png
│   ├── frame_0001.png
│   └── ...
├── poses.npy                  # Camera poses (N, 4, 4)
├── intrinsics.npy             # Camera intrinsics (3, 3)
├── depths/                    # Estimated depth maps
│   ├── frame_0000_depth.npy
│   ├── frame_0000_depth.png
│   └── ...
├── dataset/                   # COLMAP-format dataset for 2DGS
├── model/                     # Trained 2DGS model
│   └── point_cloud/
│       └── iteration_15000/
│           └── point_cloud.ply
├── output_mesh.glb           # Final mesh (Unity-ready)
├── output_mesh.ply           # Final mesh (for viewing)
└── results.json              # Pipeline results and timing
```

---

## VRAM Requirements

| Stage | VRAM Usage |
|-------|------------|
| SV3D inference | ~20GB |
| Depth estimation | ~8GB |
| 2DGS training | ~15GB |
| Mesh extraction | ~4GB |

**Recommended GPU**: RTX A6000 (48GB) or A100 (40GB)

Stages run sequentially, so peak VRAM is ~20GB.

---

## Troubleshooting

### Out of Memory

```bash
# Clear CUDA cache between stages
python -c "import torch; torch.cuda.empty_cache()"

# Or reduce SV3D decode chunk size
# Edit test_pipeline.py: config.sv3d_decode_chunk_size = 4
```

### 2DGS CUDA Extension Errors

```bash
# Rebuild extensions
cd /workspace/2d-gaussian-splatting
pip uninstall diff-gaussian-rasterization-2d simple-knn -y
pip install submodules/diff-gaussian-rasterization-2d
pip install submodules/simple-knn
```

### SV3D Quality Issues

Try adjusting:
- `remove_background = True` (for objects)
- `motion_bucket_id` (127 = standard, lower = less motion)
- Input image size and centering

---

## Next Steps

After successful testing:

1. **Tune parameters** for your specific use case
2. **Build Dockerfile** for consistent environment
3. **Create serverless handler** for production deployment
4. **Integrate with Gradio UI**

See `docs/SV3D_IMPLEMENTATION_PLAN.md` for full implementation roadmap.

