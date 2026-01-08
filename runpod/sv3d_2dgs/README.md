# SV3D → 2DGS Pipeline for RunPod

This directory contains everything needed to run the SV3D multi-view generation → 2DGS training → mesh extraction pipeline on RunPod.

## Key Features

- **Installs to Network Volume** - Persists across pod restarts
- **Separate Environment** - Does NOT interfere with existing Gen3C/Lyra/SHARP
- **Isolated Directory** - All files in `/runpod-volume/sv3d_2dgs/`

## Files

| File | Purpose |
|------|---------|
| `setup_sv3d_2dgs.sh` | Setup script for network volume installation |
| `Dockerfile` | Docker image definition (for serverless later) |
| `test_pipeline.py` | Complete pipeline test script |
| `README.md` | This file |

---

## Quick Start

### First Time Setup (20-30 minutes)

1. **Start a RunPod Pod** with your network volume mounted
   - GPU: RTX A6000 (48GB) or A100 (40GB) recommended
   - Template: Any PyTorch template with CUDA
   - Make sure your network volume is attached

2. **Download and run setup script:**
   ```bash
   cd /workspace
   wget https://raw.githubusercontent.com/88dreams/Hunyuan3D-2/2dgs/runpod/sv3d_2dgs/setup_sv3d_2dgs.sh
   bash setup_sv3d_2dgs.sh
   ```

3. **Verify installation:**
   ```bash
   python /runpod-volume/sv3d_2dgs/scripts/quick_test.py
   ```

### Future Pod Starts (Instant)

When you start a new pod with your network volume:

```bash
# Activate the environment
source /runpod-volume/sv3d_2dgs/activate.sh

# Ready to use!
```

---

## Directory Structure (on Network Volume)

```
/runpod-volume/
├── sv3d_2dgs/                    # NEW - SV3D + 2DGS installation
│   ├── activate.sh               # Activation script
│   ├── 2d-gaussian-splatting/    # 2DGS repository
│   ├── inputs/                   # Your input images
│   ├── outputs/                  # Generated outputs
│   ├── models/                   # Downloaded models (~15GB)
│   │   └── huggingface/          # SV3D, Depth Anything V2
│   └── scripts/                  # Helper scripts
│       ├── quick_test.py
│       └── test_pipeline.py
│
├── (your existing files)         # UNTOUCHED
├── Gen3C/                        # UNTOUCHED
├── Lyra/                         # UNTOUCHED
└── ...
```

---

## Environments (Side by Side)

| Environment | Purpose | Activation |
|-------------|---------|------------|
| `cosmos-predict1` | Gen3C, Lyra, SHARP | `conda activate cosmos-predict1` |
| `sv3d-2dgs` | SV3D, 2DGS, mesh | `conda activate sv3d-2dgs` |

**They do NOT interfere with each other!**

---

## Running the Pipeline

### Quick Test (~10 minutes)

```bash
# Activate environment
source /runpod-volume/sv3d_2dgs/activate.sh

# Copy your test image
cp /path/to/your/stage.jpg /runpod-volume/sv3d_2dgs/inputs/

# Run quick test (5k iterations)
python /runpod-volume/sv3d_2dgs/scripts/test_pipeline.py \
    --image /runpod-volume/sv3d_2dgs/inputs/stage.jpg \
    --quick
```

### Full Test (~30 minutes)

```bash
# Full quality (15k iterations)
python /runpod-volume/sv3d_2dgs/scripts/test_pipeline.py \
    --image /runpod-volume/sv3d_2dgs/inputs/stage.jpg
```

### Output Location

After running, outputs are saved to:
```
/runpod-volume/sv3d_2dgs/outputs/YYYYMMDD_HHMMSS/
├── preprocessed_input.png
├── images/              # Multi-view images
├── depths/              # Depth maps
├── model/               # 2DGS model
├── output_mesh.glb      # Final mesh!
└── results.json         # Timing and stats
```

---

## Pipeline Stages

```
┌─────────────────────────────────────────────────────────────────┐
│                    PIPELINE OVERVIEW                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Stage 1: SV3D Multi-View Generation (~3 minutes)               │
│  Input:  Single image                                           │
│  Output: 21 orbital views + camera poses                        │
│                                                                  │
│  Stage 2: Depth Estimation (~1 minute)                          │
│  Input:  Multi-view images                                      │
│  Output: Depth maps for each view                               │
│                                                                  │
│  Stage 3: 2DGS Training (~15-30 minutes)                        │
│  Input:  Images + poses + depths                                │
│  Output: Trained 2DGS model (PLY)                               │
│                                                                  │
│  Stage 4: Mesh Extraction (~2 minutes)                          │
│  Input:  2DGS point cloud                                       │
│  Output: Mesh (GLB)                                             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Storage Requirements

| Component | Size |
|-----------|------|
| 2DGS repository | ~500MB |
| SV3D model | ~10GB |
| Depth Anything V2 | ~3GB |
| Other dependencies | ~2GB |
| **Total** | **~15GB** |

Plus space for outputs (~500MB per run)

---

## Troubleshooting

### "Network volume not found"

Make sure your network volume is attached to the pod:
- RunPod Console → Your Pod → Check volume mount
- Volume should be at `/runpod-volume/`

### "conda: command not found"

Use a RunPod template that includes conda:
- `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel`
- Or any official PyTorch template

### Re-installing

To completely reinstall:
```bash
rm /runpod-volume/sv3d_2dgs/.installed
bash setup_sv3d_2dgs.sh
```

### Switching Between Environments

```bash
# For Gen3C/Lyra/SHARP
conda activate cosmos-predict1

# For SV3D/2DGS
conda activate sv3d-2dgs
# OR
source /runpod-volume/sv3d_2dgs/activate.sh
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

---

## Next Steps After Testing

1. **Review outputs** - Check if multi-views and mesh look reasonable
2. **Tune parameters** - Adjust iterations, depth_ratio, etc.
3. **Build Docker image** - For consistent serverless deployment
4. **Integrate with Gradio UI** - Add tab to app_sidebar.py

See `docs/SV3D_IMPLEMENTATION_PLAN.md` for full roadmap.
