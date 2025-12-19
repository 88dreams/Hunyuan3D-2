# Dependency Matrix: Multi-Model 3D Generation

This document tracks dependencies across all models in the 3D Generation Studio to ensure compatibility.

## Model Overview

| Model | Source | Repository | Purpose | Status |
|-------|--------|------------|---------|--------|
| **Hunyuan3D** | Tencent | [Hunyuan3D-2](https://github.com/tencent/Hunyuan3D-2) | Image → GLB mesh | ✅ Working (Local ROCm) |
| **GEN3C** | NVIDIA | [nv-tlabs/GEN3C](https://github.com/nv-tlabs/GEN3C) | Image → Video | ✅ Working (RunPod) |
| **Lyra** | NVIDIA | [nv-tlabs/lyra](https://github.com/nv-tlabs/lyra) | Image/Video → 3DGS/4DGS | 📋 Planned |
| **SHARP** | Apple | [apple/ml-sharp](https://github.com/apple/ml-sharp) | Image → 3DGS PLY | 🔄 In Progress |
| **TRELLIS.2** | Microsoft | [microsoft/TRELLIS.2](https://github.com/microsoft/TRELLIS.2) | Image → GLB (O-Voxel) | 📋 Planned |

---

## 🎉 EXCELLENT NEWS: High Compatibility!

After analyzing all repositories, **GEN3C, Lyra, and TRELLIS.2 share nearly identical base requirements**:

| Package | GEN3C | Lyra | TRELLIS.2 | Compatible? |
|---------|-------|------|-----------|-------------|
| **Python** | 3.10 | 3.10 | 3.10 | ✅ YES |
| **PyTorch** | 2.6.0 | 2.6.0 | 2.6.0 | ✅ YES |
| **torchvision** | 0.21.0 | 0.21.0 | 0.21.0 | ✅ YES |
| **NumPy** | 1.26.4 | 1.26.4 | 1.x | ✅ YES |
| **CUDA** | 12.4 | 12.4 | 12.4 | ✅ YES |
| **transformers** | 4.49.0 | 4.49.0 | ✓ | ✅ YES |
| **huggingface-hub** | 0.29.2 | 0.29.2 | ✓ | ✅ YES |

**Lyra is built ON TOP of GEN3C** - they share the same base environment (`lyra.yaml` = `cosmos-predict1.yaml`)!

---

## Critical Dependencies by Model

### GEN3C (NVIDIA)
**Source**: [requirements.txt](https://github.com/nv-tlabs/GEN3C/blob/main/requirements.txt)

```yaml
# Conda (cosmos-predict1.yaml)
python: 3.10
cuda: 12.4
cuda-nvcc: 12.4
cuda-toolkit: 12.4

# Key pip packages
numpy==1.26.4
torch==2.6.0
torchvision==0.21.0
transformers==4.49.0
huggingface-hub==0.29.2
megatron-core==0.10.0
warp-lang==1.7.2
diffusers==0.32.2
```

### Lyra (NVIDIA) - Built on GEN3C!
**Source**: [INSTALL.md](https://github.com/nv-tlabs/lyra/blob/main/INSTALL.md), [lyra.yaml](https://github.com/nv-tlabs/lyra/blob/main/lyra.yaml)

```yaml
# Conda (lyra.yaml) - IDENTICAL to GEN3C!
python: 3.10
cuda: 12.4
cuda-nvcc: 12.4
cuda-toolkit: 12.4

# requirements_gen3c.txt - SAME as GEN3C
numpy==1.26.4
torch==2.6.0
torchvision==0.21.0
# ... all GEN3C deps

# requirements_lyra.txt - ADDITIONAL deps
flash_attn==2.7.4.post1
timm==1.0.19
kiui==0.2.17
gsplat (specific commit)
mamba (v2.2.4)
deepspeed==0.17.5
accelerate==1.10.0
plyfile==1.1.2
```

**Key Insight**: Lyra uses GEN3C for video generation, then adds a 3DGS decoder. The environments are designed to be compatible!

### TRELLIS.2 (Microsoft)
**Source**: [setup.sh](https://github.com/microsoft/TRELLIS.2/blob/main/setup.sh)

```yaml
# Conda (--new-env)
python: 3.10
torch==2.6.0 (cu124)
torchvision==0.21.0 (cu124)

# Basic deps (--basic)
imageio, imageio-ffmpeg, tqdm, easydict
opencv-python-headless, ninja, trimesh
transformers, gradio==6.0.1
tensorboard, pandas, lpips
pillow-simd, kornia, timm
utils3d (git)

# GPU-specific
flash-attn==2.7.3
nvdiffrast (v0.4.0)
nvdiffrec (renderutils branch)
CuMesh, FlexGEMM, O-Voxel (custom packages)
```

**Key Features**:
- Uses O-Voxel representation (not standard Gaussian splatting)
- Supports ROCm via `rocm6.2.4` PyTorch wheels
- 4B parameter model, requires H100/A100

### SHARP (Apple) - ⚠️ POTENTIAL CONFLICT
**Source**: [requirements.txt](https://github.com/apple/ml-sharp/blob/main/requirements.txt)

```yaml
# Key differences from NVIDIA stack
numpy==2.3.3        # ⚠️ NumPy 2.x vs 1.x
torch==2.8.0        # Newer PyTorch
torchvision==0.23.0
python: 3.13        # Newer Python (implied)

# Unique deps
gsplat==1.5.3
pillow-heif==1.1.1
scipy==1.16.2
```

---

## Compatibility Matrix

### Python 3.10 + NumPy 1.26.4 Stack (NVIDIA Models)

| Dependency | GEN3C | Lyra | TRELLIS.2 | Unified Version |
|------------|-------|------|-----------|-----------------|
| Python | 3.10 | 3.10 | 3.10 | **3.10** ✅ |
| NumPy | 1.26.4 | 1.26.4 | 1.x | **1.26.4** ✅ |
| PyTorch | 2.6.0 | 2.6.0 | 2.6.0 | **2.6.0** ✅ |
| torchvision | 0.21.0 | 0.21.0 | 0.21.0 | **0.21.0** ✅ |
| CUDA | 12.4 | 12.4 | 12.4 | **12.4** ✅ |
| transformers | 4.49.0 | 4.49.0 | ✓ | **4.49.0** ✅ |
| flash-attn | - | 2.7.4.post1 | 2.7.3 | **2.7.3** ✅ |
| timm | - | 1.0.19 | ✓ | **1.0.19** ✅ |
| gsplat | - | ✓ (git) | - | ✓ |

### SHARP Compatibility Issue

| Dependency | NVIDIA Stack | SHARP | Issue |
|------------|--------------|-------|-------|
| Python | 3.10 | 3.13 | ⚠️ May need testing |
| NumPy | 1.26.4 | 2.3.3 | ⚠️ Breaking changes in 2.x |
| PyTorch | 2.6.0 | 2.8.0 | Minor - can use 2.6.0 |
| gsplat | git commit | 1.5.3 | Should work |

---

## Deployment Architecture

### Recommended: Unified NVIDIA Stack + Separate SHARP

```
┌─────────────────────────────────────────────────────────────────┐
│                    Gradio UI (Local ROCm)                       │
├─────────────────────────────────────────────────────────────────┤
│  Hunyuan │  GEN3C  │  Lyra   │ TRELLIS.2 │  SHARP              │
│  (Local) │ (Cloud) │ (Cloud) │  (Cloud)  │ (Cloud)             │
└────┬─────┴────┬────┴────┬────┴─────┬─────┴────┬────────────────┘
     │          │         │          │          │
     ▼          └────┬────┴──────────┘          ▼
┌─────────┐         │                    ┌──────────────┐
│ Local   │         ▼                    │ RunPod       │
│ ROCm    │  ┌───────────────────────┐   │ Serverless   │
│         │  │ Unified NVIDIA Image  │   │ ┌──────────┐ │
│         │  │ Python 3.10           │   │ │ SHARP    │ │
│         │  │ NumPy 1.26.4          │   │ │ Py 3.10* │ │
│         │  │ PyTorch 2.6.0         │   │ │ NumPy1.x*│ │
│         │  │ ┌───────┬───────────┐ │   │ └──────────┘ │
│         │  │ │GEN3C  │  Lyra     │ │   └──────────────┘
│         │  │ │       │  TRELLIS.2│ │   * Test compatibility
│         │  │ └───────┴───────────┘ │
│         │  └───────────────────────┘
└─────────┘
```

### Option A: Single Unified Image (Recommended if SHARP works with 3.10)

```dockerfile
FROM nvcr.io/nvidia/pytorch:24.10-py3

# Python 3.10 + CUDA 12.4 environment
RUN conda create -n unified python=3.10 && \
    conda activate unified && \
    pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# GEN3C + Lyra deps
RUN pip install -r requirements_gen3c.txt
RUN pip install -r requirements_lyra.txt

# TRELLIS.2 deps
RUN pip install imageio imageio-ffmpeg tqdm trimesh transformers kornia timm

# SHARP deps (test if works with NumPy 1.x)
RUN pip install gsplat plyfile pillow-heif

# Model-specific extensions
RUN pip install flash-attn==2.7.3
RUN pip install transformer-engine[pytorch]==1.12.0
# ... apex, moge, mamba, etc.
```

### Option B: Two Separate Images

1. **`88dreams/nvidia-3d:v1`** - GEN3C + Lyra + TRELLIS.2
2. **`88dreams/sharp-runpod:v1`** - SHARP only (if incompatible)

---

## Testing Checklist

### Phase 1: Test SHARP Compatibility (BEFORE building Docker)

```bash
# On local system with gen3c-rocm environment
source ~/opt/miniconda3/bin/activate gen3c-rocm

# Check current versions
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"  # Should be 1.26.4
python -c "import torch; print(f'PyTorch: {torch.__version__}')"  # Should be 2.x

# Install SHARP deps without version constraints
pip install gsplat plyfile pillow-heif scipy click rich --no-deps

# Try installing SHARP
pip install git+https://github.com/apple/ml-sharp.git --no-deps

# Test import
python -c "import sharp; print('SHARP works with NumPy 1.x!')"
```

### Phase 2: Build Unified Docker

If SHARP works:
```bash
cd ~/Hunyuan3D-2-Fork/runpod/gen3c
docker build -f Dockerfile.unified -t 88dreams/gen3c-runpod:v9 .
docker push 88dreams/gen3c-runpod:v9
```

### Phase 3: Test on RunPod

1. Update endpoint to use new image
2. Test GEN3C job
3. Test SHARP job
4. Test Lyra job (when implemented)
5. Test TRELLIS.2 job (when implemented)

---

## Model-Specific Notes

### Lyra
- **GPU Memory**: ~43GB with full offloading, tested on H100/A100
- **Requires**: ViPE for dynamic scenes (separate conda environment!)
- **Outputs**: 3D Gaussians (PLY) for static, 4D Gaussians for dynamic
- **Relationship**: Built on GEN3C - uses same video diffusion, adds 3DGS decoder

### TRELLIS.2
- **GPU Memory**: H100 recommended (4B parameters)
- **Resolution**: 512³ (~3s), 1024³ (~17s), 1536³ (~60s)
- **Outputs**: GLB with PBR materials (Base Color, Roughness, Metallic, Opacity)
- **Unique**: Uses O-Voxel representation, not Gaussian splatting
- **ROCm Support**: Yes! `torch==2.6.0 --index-url https://download.pytorch.org/whl/rocm6.2.4`

### SHARP
- **Outputs**: PLY (Gaussian splatting) + optional video rendering
- **Unique**: Fast inference, uses gsplat renderer
- **Concern**: May require NumPy 2.x - needs testing

---

## Recommended Action Plan

### Immediate (Before RunPod Update)

1. ✅ **Test SHARP with Python 3.10 + NumPy 1.26.4**
   - Run: `./scripts/test_sharp_compat.sh`
   - If passes: Proceed with unified image
   - If fails: Create separate SHARP image

2. **Update Dockerfile.unified** to include:
   - GEN3C deps (already there)
   - Lyra deps (add `requirements_lyra.txt`)
   - TRELLIS.2 deps (add basic + custom packages)
   - SHARP deps (if compatible)

3. **Update handler_unified.py** to support:
   - `model_name: "gen3c"` (done)
   - `model_name: "sharp"` (done)
   - `model_name: "lyra"` (add)
   - `model_name: "trellis"` (add)

### Short-term

4. **Implement Lyra in Gradio UI**
   - Update `placeholder_tabs.py` → `lyra_tab.py`
   - Add `generators/lyra.py`
   - Lyra uses GEN3C for video generation + 3DGS decoder

5. **Implement TRELLIS.2 in Gradio UI**
   - Update `placeholder_tabs.py` → `trellis_tab.py`
   - Add `generators/trellis.py`

### Long-term

6. **Optimize Docker image size**
   - Use multi-stage builds
   - Share common layers between models

7. **Add model selection in unified handler**
   - Dynamic checkpoint loading
   - Memory management between models

---

## Version Lock File (Unified Environment)

```txt
# Core - NVIDIA Stack
python==3.10
numpy==1.26.4
torch==2.6.0+cu124
torchvision==0.21.0+cu124

# Transformers
transformers==4.49.0
huggingface-hub>=0.26.0,<1.0
safetensors>=0.5.0
diffusers==0.32.2

# GEN3C specific
megatron-core==0.10.0
warp-lang==1.7.2

# Lyra specific
flash-attn==2.7.3
timm==1.0.19
deepspeed==0.17.5
accelerate==1.10.0
mamba==2.2.4

# TRELLIS.2 specific
gradio==6.0.1
kornia
nvdiffrast==0.4.0
# CuMesh, FlexGEMM, O-Voxel (build from source)

# SHARP specific (if compatible)
gsplat>=1.5.0
plyfile>=1.1.0
pillow-heif>=1.1.0

# Common
imageio==2.37.0
imageio-ffmpeg>=0.6.0
opencv-python>=4.10.0
pillow>=11.0.0
trimesh>=4.0.0
scipy>=1.x
```

---

## Summary

| Model | Compatibility | Action |
|-------|--------------|--------|
| **GEN3C** | ✅ Baseline | Already working |
| **Lyra** | ✅ Same as GEN3C | Easy integration |
| **TRELLIS.2** | ✅ Same Python/PyTorch | Add custom packages |
| **SHARP** | ⚠️ NumPy 2.x issue | Test first |

**Bottom Line**: GEN3C, Lyra, and TRELLIS.2 can share a unified Docker image. SHARP needs testing before inclusion.
