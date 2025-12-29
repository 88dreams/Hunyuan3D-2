# State-of-the-Art: Converting Gaussian Splats to 3D Meshes (GLB)

**Last Updated:** December 27, 2025  
**Purpose:** Comprehensive guide on extracting high-quality meshes from 3D Gaussian Splatting representations

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [The Fundamental Challenge](#the-fundamental-challenge)
3. [Method Categories](#method-categories)
4. [State-of-the-Art Methods](#state-of-the-art-methods)
5. [Practical Workflow](#practical-workflow)
6. [Tools & Software](#tools--software)
7. [Quality Considerations](#quality-considerations)
8. [Recommended Pipeline](#recommended-pipeline)
9. [Research Papers](#research-papers)

---

## Executive Summary

Converting 3D Gaussian Splatting (3DGS) to mesh formats like GLB is **non-trivial** because Gaussians are volumetric primitives without explicit surfaces. The field has evolved rapidly, with several approaches now available:

| Approach | Quality | Speed | Best For |
|----------|---------|-------|----------|
| **SuGaR** | ⭐⭐⭐⭐ | Fast | General scenes, editing |
| **2DGS** | ⭐⭐⭐⭐⭐ | Medium | Accurate geometry |
| **GS2Mesh** | ⭐⭐⭐⭐⭐ | Fast | High-detail surfaces |
| **TSDF Fusion** | ⭐⭐⭐ | Fast | Simple scenes |
| **DreamGaussian** | ⭐⭐⭐⭐ | Very Fast | Single-image 3D |

**Bottom Line:** For most use cases, **SuGaR** or **GS2Mesh** provide the best balance of quality and practicality. For the highest geometric accuracy, use **2D Gaussian Splatting** from the start.

---

## The Fundamental Challenge

### Why This Is Hard

3D Gaussian Splatting represents scenes as **millions of anisotropic ellipsoids** (Gaussians), each with:
- Position (x, y, z)
- Covariance matrix (orientation, scale)
- Color (spherical harmonics or RGB)
- Opacity

**Meshes** represent surfaces as:
- Vertices (points)
- Faces (triangles connecting vertices)
- Normals (surface orientation)
- UV coordinates (texture mapping)

The core problem: **Gaussians don't define surfaces explicitly.** They're volumetric "blobs" that collectively approximate appearance, not geometry.

### What Makes It Worse

1. **Gaussians are unorganized** - After optimization, they scatter chaotically
2. **No explicit surface** - Gaussians can overlap, float, or be inside objects
3. **View-dependent appearance** - Gaussians encode view-dependent effects that don't map to surfaces
4. **Scale ambiguity** - Large flat Gaussians vs. many small ones can look identical

---

## Method Categories

### Category 1: Surface-Aligned Regularization

**Idea:** Modify the Gaussian optimization to encourage alignment with surfaces, then extract.

**Methods:**
- SuGaR (Surface-Aligned Gaussian Splatting)
- MeshGS (Mesh-Aligned Gaussian Splatting)
- GaMeS (Gaussian Mesh Splatting)

**Pros:** Clean meshes, fast extraction  
**Cons:** Requires retraining or regularization during optimization

---

### Category 2: Depth-Based Reconstruction

**Idea:** Render depth maps from the Gaussians, then fuse into a mesh.

**Methods:**
- GS2Mesh (Stereo depth estimation)
- TSDF Fusion (Truncated Signed Distance Field)
- Multi-view depth fusion

**Pros:** Works on any 3DGS model, no retraining  
**Cons:** Quality depends on depth estimation accuracy

---

### Category 3: 2D Gaussian Splatting

**Idea:** Use flat 2D Gaussians (disks) instead of 3D ellipsoids from the start.

**Methods:**
- 2DGS (2D Gaussian Splatting)
- SolidGS (Solid kernel functions)

**Pros:** Geometrically accurate by design  
**Cons:** Requires training with 2DGS, not a conversion method

---

### Category 4: Hybrid Neural-Explicit

**Idea:** Combine Gaussians with neural implicit surfaces (SDF/NeRF).

**Methods:**
- Neural Surface Priors
- NeuSG
- Gaussian Opacity Fields

**Pros:** High quality, editable  
**Cons:** Complex, slower

---

### Category 5: Direct Mesh Binding

**Idea:** Bind Gaussians directly to mesh faces during optimization.

**Methods:**
- MILo (Mesh-In-the-Loop)
- Mesh-based Gaussian Splatting

**Pros:** Mesh and Gaussians stay synchronized  
**Cons:** Requires mesh initialization

---

## State-of-the-Art Methods

### 1. SuGaR (Surface-Aligned Gaussian Splatting)

**Paper:** [arXiv:2311.12775](https://arxiv.org/abs/2311.12775) (Nov 2023)  
**Code:** [GitHub](https://github.com/Anttwo/SuGaR)

**How It Works:**
1. **Regularization:** Adds a loss term encouraging Gaussians to align with surfaces
2. **Point Sampling:** Samples points from the surface of aligned Gaussians
3. **Poisson Reconstruction:** Uses screened Poisson surface reconstruction to create mesh
4. **Optional Refinement:** Binds Gaussians to mesh faces for joint optimization

**Key Innovation:** The regularization term measures how "flat" Gaussians are and how well they align with local surface normals.

**Quality:** ⭐⭐⭐⭐ (Very Good)  
**Speed:** ~15 minutes for mesh extraction  
**Best For:** General scenes, editing workflows

```python
# SuGaR typical usage
from sugar import SuGaR

# Load trained 3DGS model
sugar = SuGaR.from_gaussian_splatting(gs_model)

# Extract mesh with regularization
mesh = sugar.extract_mesh(
    resolution=1024,
    poisson_depth=10,
    density_threshold=0.5
)

# Export
mesh.export("output.glb")
```

---

### 2. GS2Mesh (Stereo-Based Reconstruction)

**Paper:** [arXiv:2404.01810](https://arxiv.org/abs/2404.01810) (Apr 2024)  
**Project:** [gs2mesh.github.io](https://gs2mesh.github.io/)

**How It Works:**
1. **Render Stereo Pairs:** Generate calibrated stereo image pairs from 3DGS
2. **Stereo Matching:** Use pre-trained stereo models (RAFT-Stereo, etc.) for depth
3. **Depth Fusion:** Fuse multi-view depth maps using TSDF
4. **Mesh Extraction:** Marching cubes on the fused TSDF

**Key Innovation:** Leverages 3DGS's excellent novel view synthesis to generate synthetic stereo pairs, avoiding the need to modify the Gaussians.

**Quality:** ⭐⭐⭐⭐⭐ (Excellent)  
**Speed:** ~5-10 minutes  
**Best For:** High-detail surfaces, in-the-wild captures

**Advantages over SuGaR:**
- Works on any pre-trained 3DGS (no retraining)
- Better fine detail preservation
- State-of-the-art on Tanks and Temples benchmark

---

### 3. 2D Gaussian Splatting (2DGS)

**Paper:** [arXiv:2403.17888](https://arxiv.org/abs/2403.17888) (Mar 2024)  
**Code:** [GitHub](https://github.com/hbb1/2d-gaussian-splatting)

**How It Works:**
- Replaces 3D ellipsoid Gaussians with **2D oriented planar disks**
- Each Gaussian lies flat on the surface by design
- Ray-splat intersection provides accurate depth
- Mesh extraction is straightforward (the disks define the surface)

**Key Innovation:** By using 2D Gaussians, the representation is **intrinsically surface-aligned**, eliminating the fundamental ambiguity of 3DGS.

**Quality:** ⭐⭐⭐⭐⭐ (Best Geometry)  
**Speed:** Similar to 3DGS training  
**Best For:** Applications requiring accurate geometry

**Important:** This is not a conversion method—you must train with 2DGS from the start.

---

### 4. DreamGaussian (Mesh Extraction for Generated 3D)

**Paper:** [arXiv:2309.16653](https://arxiv.org/abs/2309.16653) (Sep 2023)  
**Code:** [GitHub](https://github.com/dreamgaussian/dreamgaussian)

**How It Works:**
1. **Generate 3DGS:** From single image using score distillation
2. **Densify to Point Cloud:** Sample points from Gaussian centers
3. **Mesh Extraction:** Use marching cubes on density field
4. **UV Unwrapping:** Automatic UV generation
5. **Texture Refinement:** Fine-tune texture in UV space

**Key Innovation:** End-to-end pipeline from image to textured mesh in ~2 minutes.

**Quality:** ⭐⭐⭐⭐ (Good for generated content)  
**Speed:** ~2 minutes total  
**Best For:** Single-image 3D generation

---

### 5. MeshGS (Mesh-Aligned Splatting)

**Paper:** [arXiv:2410.08941](https://arxiv.org/abs/2410.08941) (Oct 2024)

**How It Works:**
1. **Distance-Based Alignment:** Classifies Gaussians as "tightly-bound" or "loosely-bound" based on distance to mesh
2. **Tightly-Bound:** Flattened and aligned with mesh geometry
3. **Loosely-Bound:** Handle artifacts and fuzzy regions
4. **Regularization:** Forces tight Gaussians to stay on surface

**Quality:** ⭐⭐⭐⭐ (2dB better PSNR than baselines)  
**Speed:** Similar to 3DGS  
**Best For:** Large outdoor scenes

---

### 6. TSDF Fusion (Classic Approach)

**No specific paper—classical computer vision technique**

**How It Works:**
1. **Render Depth Maps:** From multiple viewpoints using 3DGS
2. **TSDF Volume:** Create a truncated signed distance field
3. **Integrate Depths:** Fuse all depth maps into TSDF
4. **Marching Cubes:** Extract mesh from TSDF

**Quality:** ⭐⭐⭐ (Decent)  
**Speed:** Fast  
**Best For:** Simple scenes, quick previews

```python
import open3d as o3d
import numpy as np

# Create TSDF volume
volume = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=0.01,
    sdf_trunc=0.04,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
)

# For each view
for i, (color, depth, intrinsic, extrinsic) in enumerate(views):
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth, depth_trunc=4.0, convert_rgb_to_intensity=False
    )
    volume.integrate(rgbd, intrinsic, extrinsic)

# Extract mesh
mesh = volume.extract_triangle_mesh()
mesh.compute_vertex_normals()
```

---

## Practical Workflow

### Workflow A: From Existing 3DGS (No Retraining)

```
┌─────────────────┐
│  3DGS PLY File  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  GS2Mesh        │  ← Render stereo pairs, depth fusion
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Raw Mesh       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Blender        │  ← Clean up, decimate, UV unwrap
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Texture Bake   │  ← Project colors from 3DGS renders
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  GLB Export     │
└─────────────────┘
```

### Workflow B: From Scratch (Best Quality)

```
┌─────────────────┐
│  Input Images   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Train 2DGS     │  ← Use 2D Gaussian Splatting
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Direct Mesh    │  ← 2DGS provides clean geometry
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  UV + Texture   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  GLB Export     │
└─────────────────┘
```

### Workflow C: With SuGaR (Editable Result)

```
┌─────────────────┐
│  3DGS PLY File  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  SuGaR          │  ← Regularize + Poisson reconstruction
│  Regularization │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Coarse Mesh    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  SuGaR          │  ← Bind Gaussians to mesh faces
│  Refinement     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Editable Mesh  │  ← Can sculpt/rig/animate
│  + Gaussians    │
└─────────────────┘
```

---

## Tools & Software

### Open Source Tools

| Tool | Purpose | Link |
|------|---------|------|
| **SuGaR** | Surface-aligned mesh extraction | [GitHub](https://github.com/Anttwo/SuGaR) |
| **2DGS** | Geometry-accurate Gaussian splatting | [GitHub](https://github.com/hbb1/2d-gaussian-splatting) |
| **GS2Mesh** | Stereo-based mesh extraction | [Project](https://gs2mesh.github.io/) |
| **gsplat** | Fast Gaussian splatting library | [GitHub](https://github.com/nerfstudio-project/gsplat) |
| **Open3D** | Point cloud/mesh processing | [GitHub](https://github.com/isl-org/Open3D) |
| **PyMeshLab** | Mesh processing (Poisson, etc.) | [GitHub](https://github.com/cnr-isti-vclab/PyMeshLab) |
| **Blender** | 3D editing, UV, export | [blender.org](https://www.blender.org/) |

### Commercial Tools

| Tool | Purpose | Link |
|------|---------|------|
| **KIRI Engine** | Mobile 3DGS capture + mesh export | [kiriengine.app](https://www.kiriengine.app/) |
| **Luma AI** | 3DGS capture + processing | [lumalabs.ai](https://lumalabs.ai/) |
| **Polycam** | 3DGS capture + export | [poly.cam](https://poly.cam/) |

### CLI Tools

| Tool | Purpose | Link |
|------|---------|------|
| **gsbox** | Format conversion (.ply, .splat, .spz) | [GitHub](https://github.com/gotoeasy/gsbox) |
| **splat-transform** | Edit/transform Gaussian splats | [GitHub](https://github.com/playcanvas/splat-transform) |

### Blender Addons

| Addon | Purpose | Link |
|-------|---------|------|
| **3DGS Render** | Import/render 3DGS in Blender | KIRI Engine |
| **SuperSplat Importer** | Import .splat files | Community |

---

## Quality Considerations

### Geometry Quality Factors

1. **Input 3DGS Quality**
   - More Gaussians ≠ better mesh (can cause noise)
   - Well-trained 3DGS with good coverage is essential
   - Sparse regions will have poor geometry

2. **Method Choice**
   - 2DGS > GS2Mesh > SuGaR > TSDF for pure geometry
   - Trade-off: 2DGS requires training from scratch

3. **Post-Processing**
   - Decimation: Reduce poly count without losing detail
   - Smoothing: Remove noise (but can lose detail)
   - Hole filling: Close gaps in reconstruction

### Texture Quality Factors

1. **UV Unwrapping**
   - Automatic UV often has seams
   - Manual UV for best quality
   - Smart UV Project in Blender is a good compromise

2. **Texture Baking**
   - Bake from 3DGS renders, not Gaussian colors directly
   - Higher resolution = better quality, larger files
   - Consider PBR materials (albedo, roughness, metallic)

3. **Texture Resolution**
   - 2K (2048×2048) for most use cases
   - 4K for hero assets
   - 1K for real-time/web applications

### File Size Considerations

| Format | Typical Size | Use Case |
|--------|--------------|----------|
| GLB (compressed) | 5-50 MB | Web, mobile |
| GLB (uncompressed) | 20-200 MB | Desktop apps |
| FBX | 50-500 MB | Game engines |
| USD | 100+ MB | Film/VFX |

---

## Recommended Pipeline

### For Lyra/3DGS Output → GLB

Given your project context (Lyra outputs 3DGS), here's the recommended pipeline:

```
┌─────────────────────────────────────────────────────────────┐
│                    RECOMMENDED PIPELINE                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. LYRA OUTPUT                                              │
│     └── PyTorch tensor .ply (2M+ Gaussians)                 │
│                                                              │
│  2. CONVERT FORMAT                                           │
│     └── convert_lyra_ply.py → Standard 3DGS .ply            │
│     └── Downsample if needed (500k-1M for mesh extraction)  │
│                                                              │
│  3. MESH EXTRACTION (choose one)                            │
│     ├── Option A: GS2Mesh (best quality, no retraining)     │
│     ├── Option B: SuGaR (good quality, editable)            │
│     └── Option C: TSDF Fusion (fast, lower quality)         │
│                                                              │
│  4. MESH CLEANUP (Blender)                                  │
│     ├── Remove floating geometry                            │
│     ├── Fill holes                                          │
│     ├── Decimate to target poly count                       │
│     └── Smooth if needed                                    │
│                                                              │
│  5. UV + TEXTURE                                            │
│     ├── Smart UV Project                                    │
│     ├── Bake texture from 3DGS renders                      │
│     └── Optional: Generate PBR maps                         │
│                                                              │
│  6. EXPORT                                                  │
│     └── GLB with Draco compression                          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Quick Start Commands

```bash
# 1. Convert Lyra output
python scripts/convert_lyra_ply.py input.ply output_3dgs.ply --max-points 1000000

# 2. Run GS2Mesh (if installed)
python -m gs2mesh --input output_3dgs.ply --output mesh.ply

# 3. Or use Open3D for TSDF fusion
python scripts/tsdf_mesh_extraction.py output_3dgs.ply mesh.ply

# 4. Clean up in Blender (manual or scripted)
blender --python scripts/mesh_cleanup.py -- mesh.ply cleaned.glb

# 5. Final export
# Done in Blender: File → Export → glTF 2.0 (.glb)
```

---

## Research Papers

### Core Methods

| Paper | Year | Key Contribution | Link |
|-------|------|------------------|------|
| **SuGaR** | 2023 | Surface-aligned regularization + Poisson | [arXiv](https://arxiv.org/abs/2311.12775) |
| **2DGS** | 2024 | 2D Gaussian disks for geometry | [arXiv](https://arxiv.org/abs/2403.17888) |
| **GS2Mesh** | 2024 | Stereo depth fusion from 3DGS | [arXiv](https://arxiv.org/abs/2404.01810) |
| **MeshGS** | 2024 | Distance-based mesh alignment | [arXiv](https://arxiv.org/abs/2410.08941) |
| **DreamGaussian** | 2023 | End-to-end image→mesh pipeline | [arXiv](https://arxiv.org/abs/2309.16653) |
| **MILo** | 2024 | Mesh-in-the-loop Gaussian optimization | [arXiv](https://arxiv.org/abs/2506.24096) |

### Texture & Materials

| Paper | Year | Key Contribution | Link |
|-------|------|------------------|------|
| **TexGaussian** | 2024 | PBR material generation from 3DGS | [arXiv](https://arxiv.org/abs/2411.19654) |
| **Gaussian Frosting** | 2024 | Mesh + Gaussian layer for fuzzy surfaces | [arXiv](https://arxiv.org/abs/2403.14554) |
| **Boosting 3D via PBR** | 2024 | PBR extraction from generated 3D | [arXiv](https://arxiv.org/abs/2411.16080) |

### Related Techniques

| Paper | Year | Key Contribution | Link |
|-------|------|------------------|------|
| **GaMeS** | 2024 | Gaussian components bound to mesh faces | [arXiv](https://arxiv.org/abs/2402.01459) |
| **SolidGS** | 2024 | Solid kernels for sparse-view reconstruction | [arXiv](https://arxiv.org/abs/2412.15400) |
| **Neural Surface Priors** | 2024 | SDF + Gaussians for editing | [arXiv](https://arxiv.org/abs/2411.18311) |
| **MeshSplat** | 2025 | Generalizable sparse-view mesh via 2DGS | [arXiv](https://arxiv.org/abs/2508.17811) |

---

## Conclusion

### Key Takeaways

1. **No perfect solution exists** - All methods involve trade-offs between quality, speed, and complexity

2. **Best geometry:** Train with **2DGS** from the start if mesh quality is critical

3. **Best for existing 3DGS:** Use **GS2Mesh** (stereo depth fusion) - works on any model, excellent quality

4. **Best for editing:** Use **SuGaR** - produces editable mesh + Gaussian hybrid

5. **Fastest:** **TSDF fusion** with Open3D - quick but lower quality

6. **Texture matters:** Even a perfect mesh looks bad without proper UV mapping and texture baking

### For Your Lyra Pipeline

Since Lyra outputs 3DGS (not 2DGS), your best options are:

1. **GS2Mesh** for highest quality mesh extraction
2. **SuGaR** if you need to edit the result
3. **TSDF Fusion** for quick previews

Consider adding a "Mesh Export" option to your Gradio UI that:
1. Converts Lyra's PyTorch PLY to standard format
2. Runs mesh extraction (GS2Mesh or TSDF)
3. Opens result in Blender for final cleanup and GLB export

---

*This document represents the state-of-the-art as of December 2025. The field is rapidly evolving—check the linked papers and repositories for the latest developments.*

