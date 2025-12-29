# Deep Dive: NeRF vs. Gaussian Splatting for Single-Image 3D Reconstruction

**Research Date:** December 26, 2025  
**Context:** Evaluating whether Neural Radiance Fields (NeRF) would be a good alternative to Gaussian Splatting for the Hunyuan3D-2-Fork project's goal of converting 2D architectural interior images to 3D models.

---

## Executive Summary

Based on comprehensive research across academic papers, current implementations, and the project's specific needs (architectural interior 2D→3D), **Gaussian Splatting (3DGS) is the better choice**. The current pipeline architecture (GEN3C video generation + Lyra 3DGS reconstruction) is well-designed and should be maintained.

---

## Table of Contents

1. [Understanding the Fundamental Difference](#understanding-the-fundamental-difference)
2. [Key Comparison for Architectural Interiors](#key-comparison-for-architectural-interiors)
3. [The Single-Image Problem](#the-single-image-problem)
4. [Why 3DGS Wins for This Use Case](#why-3dgs-wins-for-this-use-case)
5. [When NeRF Would Be Better](#when-nerf-would-be-better)
6. [Relevant Research Papers](#relevant-research-papers)
7. [Practical Recommendations](#practical-recommendations)
8. [The Hybrid Future](#the-hybrid-future)
9. [Conclusion](#conclusion)

---

## Understanding the Fundamental Difference

### Neural Radiance Fields (NeRF)

| Aspect | Description |
|--------|-------------|
| **Representation** | Implicit — the scene is encoded as a neural network that maps 3D coordinates + viewing direction → color + density |
| **How it works** | For each pixel, "ray march" through the scene, querying the network thousands of times |
| **Output** | A trained neural network (weights) that can render novel views |
| **Introduced** | 2020 (Mildenhall et al.) |

**Strengths:**
- Excellent at capturing fine details and complex lighting
- Continuous representation (theoretically unlimited resolution)
- Handles view-dependent effects (reflections, specular highlights)

**Weaknesses:**
- Computationally intensive (slow training and rendering)
- Requires many input views for quality results
- Difficult to extract explicit geometry (meshes)

### 3D Gaussian Splatting (3DGS)

| Aspect | Description |
|--------|-------------|
| **Representation** | Explicit — the scene is a collection of 3D Gaussian "blobs" with position, scale, rotation, color, and opacity |
| **How it works** | Project Gaussians onto the screen like particles (rasterization) |
| **Output** | A point cloud of Gaussians (typically PLY format) |
| **Introduced** | 2023 (Kerbl et al.) |

**Strengths:**
- Real-time rendering (100+ FPS)
- Fast training (minutes vs. hours)
- Explicit representation enables easier editing
- Lower memory during inference

**Weaknesses:**
- May struggle with extremely fine details
- Can produce "splotchy" artifacts in sparse regions
- Memory-intensive for very complex scenes

---

## Key Comparison for Architectural Interiors

| Factor | NeRF | 3D Gaussian Splatting | Winner |
|--------|------|----------------------|--------|
| **Single-image capability** | Poor without priors | Good with modern methods | 🏆 **3DGS** |
| **Rendering speed** | Slow (seconds/frame) | Real-time (100+ FPS) | 🏆 **3DGS** |
| **Training time** | Hours | Minutes | 🏆 **3DGS** |
| **Detail quality** | Excellent | Good-to-Excellent | Tie |
| **Indoor/architectural scenes** | Good with multi-view | Better with sparse views | 🏆 **3DGS** |
| **Mesh export** | Requires post-processing | Requires post-processing | Tie |
| **Memory during inference** | Higher | Lower | 🏆 **3DGS** |
| **Tooling maturity** | More mature | Rapidly evolving | Tie |
| **Large flat surfaces (walls)** | Can have floaters | Handles well | 🏆 **3DGS** |
| **Complex lighting** | Excellent | Good | NeRF |

---

## The Single-Image Problem

This is **the critical factor** for this project. Both NeRF and 3DGS were originally designed for **multi-view reconstruction** (dozens to hundreds of images). Single-image 3D is fundamentally under-constrained — you're asking "what does the back of this room look like?" from seeing only the front.

### How Modern Methods Solve This

#### NeRF-based Approaches

| Method | Description | Speed | Quality |
|--------|-------------|-------|---------|
| **LRM (Large Reconstruction Model)** | Transformer predicts NeRF-like triplane from single image | ~5 seconds | Good |
| **ReconFusion** | Diffusion priors hallucinate unseen views, then train NeRF | Minutes | High |
| **SSDNeRF** | Single-stage diffusion + NeRF joint optimization | Minutes | High |
| **PERF** | Panoramic NeRF from single panorama image | Minutes | Good for 360° |

#### 3DGS-based Approaches

| Method | Description | Speed | Quality |
|--------|-------------|-------|---------|
| **Splatter Image** | Direct image→Gaussians | 38 FPS | Good |
| **TriplaneGaussian** | Hybrid triplane + Gaussians | ~1 second | High |
| **Gamba** | Mamba architecture for efficient 3DGS | ~0.6 seconds | Good |
| **FDGaussian** | Geometric-aware diffusion + 3DGS | Seconds | High |
| **InstantMesh** | Multi-view diffusion + sparse-view LRM | ~10 seconds | High |
| **TripoSR** | Transformer-based fast mesh generation | ~0.5 seconds | Good |
| **CRM** | Convolutional reconstruction model | ~10 seconds | High |
| **Lyra** | Video diffusion → 3DGS reconstruction | ~30-60 seconds | Very High |

### The Video-to-3D Approach (Current Pipeline)

The Lyra pipeline used in this project represents the state-of-the-art approach:

```
Input Image → GEN3C Video Generation → Multi-view Frames → Lyra 3DGS Reconstruction → PLY Output
```

This is fundamentally superior because:
1. **Creates synthetic multi-view data** rather than hallucinating geometry
2. **Leverages video diffusion models** trained on massive datasets
3. **Provides temporal consistency** across generated views
4. **Enables high-quality 3DGS reconstruction** from the generated views

---

## Why 3DGS Wins for This Use Case

### 1. The Current Pipeline Already Uses 3DGS

The project's existing architecture:
- **Lyra** → outputs 3DGS (PLY)
- **SHARP** → outputs 3DGS (PLY)
- **GEN3C** → generates video (feeds into Lyra for 3DGS)
- **TRELLIS.2** → outputs GLB with PBR materials

Switching to NeRF would require:
- Different inference pipeline
- Different output handling
- Additional mesh extraction step (marching cubes, etc.)
- Re-training or finding NeRF models for architectural scenes

### 2. Architectural Interiors Have Large Flat Surfaces

3DGS handles planar surfaces well because Gaussians can be "flattened" (anisotropic scaling) to efficiently represent walls, floors, and ceilings. NeRF tends to create "floater" artifacts in open spaces where density should be zero.

### 3. Real-time Preview Matters

When iterating on 3D models:
- **3DGS:** Real-time preview at 100+ FPS
- **NeRF:** Seconds per frame, impractical for interactive work

### 4. The Video→3D Pipeline Is Superior for Single-Image Input

The Lyra approach of generating video first solves the fundamental single-image problem more elegantly than trying to hallucinate geometry directly.

### 5. Output Format Compatibility

3DGS outputs PLY files that can be:
- Viewed directly in 3D Gaussian Splatting viewers
- Converted to meshes using existing tools
- Integrated into standard 3D workflows

NeRF outputs require:
- Custom viewers or rendering code
- Mesh extraction (lossy process)
- More complex integration

---

## When NeRF Would Be Better

NeRF could be preferable if the project needed:

1. **Extremely fine detail** (e.g., text on signs, intricate patterns, fine textures)
2. **Complex view-dependent effects** (reflections, refractions, subsurface scattering)
3. **Unlimited resolution rendering** (NeRF is continuous, can theoretically render at any resolution)
4. **Specific pre-trained models** that only exist in NeRF format

However, for architectural interiors where the goals are:
- Clean geometry
- Fast iteration
- Standard 3D format export (GLB/PLY)
- Real-time preview

**3DGS is the clear winner.**

---

## Relevant Research Papers

### Foundational Papers

| Paper | Year | Key Contribution |
|-------|------|------------------|
| [NeRF: Representing Scenes as Neural Radiance Fields](https://arxiv.org/abs/2003.08934) | 2020 | Introduced neural radiance fields |
| [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://arxiv.org/abs/2308.04079) | 2023 | Introduced 3D Gaussian Splatting |

### Single-Image 3D Reconstruction

| Paper | Approach | Link |
|-------|----------|------|
| **LRM: Large Reconstruction Model** | NeRF-based, transformer | [HF Paper](https://hf.co/papers/2311.04400) |
| **Splatter Image** | 3DGS, ultra-fast | [HF Paper](https://hf.co/papers/2312.13150) |
| **TriplaneGaussian** | Hybrid triplane + 3DGS | [HF Paper](https://hf.co/papers/2312.09147) |
| **Gamba** | 3DGS + Mamba architecture | [HF Paper](https://hf.co/papers/2403.18795) |
| **InstantMesh** | Multi-view diffusion + LRM | [HF Paper](https://hf.co/papers/2404.07191) |
| **TripoSR** | Fast transformer-based | [HF Paper](https://hf.co/papers/2403.02151) |
| **CRM** | Convolutional reconstruction | [HF Paper](https://hf.co/papers/2403.05034) |
| **FDGaussian** | Geometric-aware diffusion | [HF Paper](https://hf.co/papers/2403.10242) |
| **ExScene** | Panoramic 3DGS from single image | [HF Paper](https://hf.co/papers/2503.23881) |

### Indoor/Architectural Scene Reconstruction

| Paper | Approach | Link |
|-------|----------|------|
| **NeRFVS** | NeRF with geometry scaffolds for indoor | [HF Paper](https://hf.co/papers/2304.06287) |
| **SurfelNeRF** | Neural surfels for indoor scenes | [HF Paper](https://hf.co/papers/2304.08971) |
| **SceneCraft** | Layout-guided indoor generation | [HF Paper](https://hf.co/papers/2410.09049) |
| **NeRF-Det** | NeRF for indoor 3D detection | [HF Paper](https://hf.co/papers/2307.14620) |

### Dynamic Scene / Video-to-3D

| Paper | Approach | Link |
|-------|----------|------|
| **StreamSplat** | Online dynamic 3DGS from video | [HF Paper](https://hf.co/papers/2506.08862) |
| **SplineGS** | Motion-adaptive 3DGS from monocular video | [HF Paper](https://hf.co/papers/2412.09982) |
| **GauFRe** | Gaussian deformation fields | [HF Paper](https://hf.co/papers/2312.11458) |
| **Hybrid 3D-4D Gaussian Splatting** | Adaptive static/dynamic representation | [HF Paper](https://hf.co/papers/2505.13215) |

### Comparative Studies

| Study | Finding | Source |
|-------|---------|--------|
| UAV Point Cloud Generation | 3DGS faster processing, comparable quality | [MDPI Sensors](https://www.mdpi.com/1424-8220/25/10/2995) |
| General Comparison | 3DGS 10-100x faster rendering | [nerfacc.com](https://nerfacc.com/comparing-nerf-and-gaussian-splatting-whats-the-difference/) |

---

## Practical Recommendations

### 1. Keep the Current Architecture

The pipeline (GEN3C → Lyra → 3DGS) is well-designed. Continue debugging the gsplat CUDA compilation issue rather than switching approaches.

### 2. Consider Adding Fast 3DGS Models for Quick Iteration

These could serve as faster alternatives for quick previews:

| Model | Speed | Quality | Notes |
|-------|-------|---------|-------|
| **TripoSR** | 0.5 sec | Good | MIT licensed, direct mesh output |
| **InstantMesh** | 10 sec | High | Open-source, sparse-view LRM |
| **CRM** | 10 sec | High | Textured mesh output |

### 3. Don't Add NeRF

Adding a NeRF-based model would:
- Complicate the pipeline (different output format)
- Not provide meaningful quality improvement for this use case
- Slow down iteration speed
- Require additional mesh extraction tooling

### 4. For Highest Quality: Fix Lyra

The Lyra pipeline (video diffusion → 3DGS) represents the best approach for single-image architectural reconstruction. The current gsplat compilation issues are tooling problems, not fundamental limitations.

---

## The Hybrid Future

Research is moving toward **hybrid approaches** that combine NeRF's detail with 3DGS's speed:

| Approach | Description | Status |
|----------|-------------|--------|
| **TriplaneGaussian** | NeRF-like triplane features predict Gaussian attributes | Available now |
| **SurfelNeRF** | Neural surfels combining explicit geometry with neural features | Research |
| **2D Gaussian Splatting** | View-consistent geometry with 2D Gaussian disks | Research |
| **GTR** | Geometry and texture refinement for LRM | Research |

These may become relevant in 6-12 months, but for now, pure 3DGS methods are more mature and practical.

---

## Conclusion

### Bottom Line

**Stick with Gaussian Splatting.** The current architecture (GEN3C video generation + Lyra 3DGS reconstruction) is the right approach for converting 2D architectural interior images to 3D models.

### Key Reasons

1. **3DGS is faster** — real-time rendering vs. seconds per frame
2. **The video-to-3D pipeline is superior** — creates multi-view data rather than hallucinating geometry
3. **Better for architectural scenes** — handles large flat surfaces well
4. **Simpler output format** — PLY files integrate easily into workflows
5. **The tooling exists** — the current pipeline is well-designed

### What NeRF Would NOT Solve

- The single-image problem (both approaches need priors or generated views)
- The gsplat compilation issue (unrelated to the representation)
- Output format complexity (would actually make it worse)

### Recommendation

Continue debugging the Lyra gsplat CUDA compilation issue. The underlying approach is sound, and once working, will produce high-quality results for architectural interior reconstruction.

---

## References

- Wikipedia: [Neural Radiance Field](https://en.wikipedia.org/wiki/Neural_radiance_field)
- Wikipedia: [Gaussian Splatting](https://en.wikipedia.org/wiki/Gaussian_splatting)
- [Comparing NeRF and Gaussian Splatting](https://nerfacc.com/comparing-nerf-and-gaussian-splatting-whats-the-difference/)
- [MDPI Sensors: NeRF vs 3DGS for Point Cloud Generation](https://www.mdpi.com/1424-8220/25/10/2995)
- Hugging Face Papers: Various (linked above)

---

*This research was conducted to inform architectural decisions for the Hunyuan3D-2-Fork project's 2D-to-3D reconstruction pipeline.*

