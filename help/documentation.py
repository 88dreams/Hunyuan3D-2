"""
Help documentation strings for the 3D Studio UI.

These are displayed in the Help tab accordions.
Extracted from app_sidebar.py for better maintainability.
"""

SHARP_HELP = """
## SHARP (Apple)

**What it does:** SHARP generates 3D Gaussian Splatting (3DGS) representations from a single image in under 1 second. The output is a PLY file containing Gaussian splats that can be rendered in real-time using 3DGS viewers. SHARP also supports optional video rendering to visualize the 3D reconstruction with camera movement.

**Key features:**
- Fastest single-image to 3DGS (sub-second inference)
- Metric scale output (real-world units)
- Optional video trajectory rendering (CUDA GPU required)

### Output Format

SHARP outputs standard 3DGS PLY files compatible with various Gaussian Splatting viewers:
- Contains: positions, spherical harmonics (colors), scales, rotations, opacities
- Coordinate system: OpenCV (x right, y down, z forward)
- Color space: sRGB (converted from internal linearRGB for compatibility)

### Video Rendering Options

| Setting | Description | Range | Default |
|---------|-------------|-------|---------|
| **Trajectory Type** | Camera movement pattern | rotate_forward, rotate, swipe, shake | rotate_forward |
| **Frames** | Number of video frames | 30-180 | 60 |
| **Repeats** | Trajectory loop count | 1-4 | 1 |
| **Lateral Offset** | Max horizontal/vertical movement | 0.02-0.20 | 0.08 |
| **Zoom/Forward** | Max forward camera movement | 0.05-0.40 | 0.15 |
| **Look-At Mode** | Camera focus behavior | point, ahead | point |

### Trajectory Types Explained

| Type | Description | Best For |
|------|-------------|----------|
| **rotate_forward** | Circular rotation + forward zoom | Most scenes (default) |
| **rotate** | Pure circular rotation | Objects, centered subjects |
| **swipe** | Left-to-right horizontal pan | Wide scenes, panoramas |
| **shake** | Horizontal then vertical shake | Dynamic preview |

### Architectural Interior Settings

```
Trajectory: rotate_forward (shows depth well)
Frames: 90-120 (smooth, longer preview)
Lateral Offset: 0.06-0.10 (moderate movement)
Zoom/Forward: 0.10-0.20 (subtle zoom effect)
Look-At: point (keeps focus on room center)
```

**Tips for Architectural Interiors:**
- SHARP excels at capturing room geometry and furniture
- Use high-resolution input images for best detail
- Video rendering requires CUDA GPU (use RunPod)
- The 3DGS output can be converted to mesh using the MESH tab
"""


GEN3C_HELP = """
## GEN3C (NVIDIA)

**What it does:** Gen3C generates **videos from single images** with precise camera control and 3D consistency. It uses a 3D cache (point clouds from depth prediction) to maintain spatial coherence as the camera moves through the scene. The model excels at creating smooth, realistic camera movements while keeping the scene consistent.

**Key capability:** Unlike other video generators, Gen3C maintains 3D consistency by using depth-based point clouds to guide generation. This means objects stay in place as the camera moves, rather than morphing or popping in/out.

### Key Settings

| Setting | Description | Range | Default |
|---------|-------------|-------|---------|
| **Seed** | Random seed for reproducibility | 0-999999 | 42 |
| **Num Frames** | Output video length (121*N - 1 pattern) | 121-361+ | 121 |
| **Guidance Scale** | Controls generation fidelity | 1.0-15.0 | 1.0 |
| **Trajectory** | Camera movement pattern | left/right/up/down/zoom_in/zoom_out/clockwise/counterclockwise | left |
| **Camera Rotation** | Rotation angle in degrees | 0-360 | varies |
| **Movement Distance** | How far camera moves | 0.1-2.0 | varies |

### Architectural Interior Settings

For architectural interiors with Gen3C:

```
Seed: Fixed for consistency
Num Frames: 121 (or 241 for longer tours)
Guidance Scale: 1.0 (default works well)
Trajectory: clockwise or counterclockwise (for room tours)
           zoom_out (to reveal full space)
           left/right (for corridor walkthroughs)
```

**Tips for Architectural Interiors:**
- Use high-quality input images (1024x1024+) with good depth cues
- Clockwise/counterclockwise trajectories create room tour effect
- Zoom_out reveals the full space from a detail shot
- Works best with images that have clear foreground/background separation
- Enable foreground_masking for better depth handling
- Ideal for: virtual tours, real estate walkthroughs, design visualization
- Output is VIDEO (mp4), not 3D model - use Lyra for 3DGS output
"""


LYRA_HELP = """
## LYRA (NVIDIA)

**What it does:** Lyra generates high-quality 3D Gaussian Splats (3DGS) from single images or 4D Gaussian Splats (4DGS) from videos. It uses a sophisticated diffusion-based approach to create detailed, renderable 3D scenes with realistic lighting and materials.

### Key Settings

| Setting | Description | Range | Default |
|---------|-------------|-------|---------|
| **Seed** | Random seed for reproducibility | 0-999999 | 42 |
| **Guidance Scale** | Controls adherence to input | 1.0-20.0 | 7.5 |
| **Inference Steps** | Diffusion steps | 20-100 | 50 |
| **SDG Steps** | Scene Diffusion Generation steps | 100-500 | 250 |
| **Resolution** | Output resolution | 256-1024 | 512 |
| **Mode** | 3DGS (image) or 4DGS (video) | 3dgs/4dgs | 3dgs |

### Architectural Interior Settings

For architectural interiors with Lyra:

```
Seed: Fixed for reproducibility
Guidance Scale: 9.0-12.0 (higher for detailed interiors)
Inference Steps: 75-100
SDG Steps: 350-500 (maximize for complex scenes)
Resolution: 512-1024 (higher for large spaces)
Mode: 3DGS for still images
```

**Tips for Architectural Interiors:**
- High-quality input images are critical
- Works exceptionally well for detailed furniture and fixtures
- SDG step is compute-intensive but crucial for quality
- Output PLY can be converted to mesh via MESH tab
- Ideal for: bedrooms, offices, detailed room corners
- May require longer processing for very detailed scenes
"""


TRELLIS_HELP = """
## TRELLIS.2 (Microsoft)

**What it does:** TRELLIS.2 generates structured 3D assets using a two-stage latent diffusion approach with O-Voxel representation. It produces clean, well-organized meshes with consistent topology and PBR materials, making outputs ideal for further editing in 3D software and game engines.

### Key Settings

| Setting | Description | Range | Default |
|---------|-------------|-------|---------|
| **Resolution** | Voxel resolution for generation | 512, 1024 | 1024 |
| **Guidance Scale** | Controls generation fidelity | 1.0-15.0 | 7.5 |
| **Seed** | Random seed for reproducibility | 0-999999 | Random |
| **Output Format** | GLB (with PBR), OBJ, or PLY | GLB/OBJ/PLY | GLB |

### Architectural Interior Settings

For architectural interiors with TRELLIS.2:

```
Resolution: 1024 (maximize detail)
Guidance Scale: 8.0-10.0
Output Format: GLB (preserves PBR materials)
```

**Tips for Architectural Interiors:**
- Produces cleaner meshes than diffusion-only methods
- Excellent for furniture and architectural elements
- Good topology makes outputs suitable for game engines
- Works well with: chairs, tables, cabinets, fixtures
- PBR materials include Base Color, Roughness, Metallic, Opacity
- Less suited for entire room reconstructions
- Best for individual objects within interiors
"""


HUNYUAN_HELP = """
## HUNYUAN3D 2.1 (Tencent)

**What it does:** Hunyuan3D 2.1 generates high-fidelity 3D models from images using a scalable diffusion-based pipeline. It features production-ready **Physically-Based Rendering (PBR)** materials with realistic light interactions (metallic reflections, subsurface scattering). Based on [Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1).

**Models Available:**
- **Mini Model (Faster):** 3.3B parameters, ~10 GB VRAM, 2-5 minutes
- **Full Model (Higher Quality):** Requires ~29 GB VRAM, 5-15 minutes

### Key Settings

| Setting | Description | Range | Default |
|---------|-------------|-------|---------|
| **Seed** | Random seed for reproducibility | 0-999999 | 42 |
| **Guidance Scale** | Controls adherence to input image | 1.0-15.0 | 9.0 |
| **Inference Steps** | Diffusion sampling steps | 10-100 | 40 |
| **Octree Resolution** | Mesh detail level (higher = more detail, slower) | 128-512 | 380 |
| **Remove Background** | Auto background removal before processing | true/false | true |

### Memory Optimization Options (Local Mode Only)

These options help run Hunyuan3D on GPUs with limited VRAM. They are only relevant for **Local** execution mode - RunPod Serverless handles memory management automatically.

| Setting | Description | VRAM Savings | Trade-off |
|---------|-------------|--------------|-----------|
| **FP16** | Half-precision floating point | ~50% | Minimal quality loss |
| **Attention Slicing** | Process attention in chunks | ~25-40% | Slower generation |
| **CPU Offload** | Move unused layers to RAM | ~60-70% | Much slower |

---

#### FP16 (Half Precision)

**What it does:** Converts model weights and computations from 32-bit (FP32) to 16-bit (FP16) floating point numbers.

**Technical details:**
- Uses `torch.float16` dtype instead of `torch.float32`
- Each number uses 2 bytes instead of 4 bytes
- Supported natively by modern NVIDIA GPUs (Tensor Cores)

**Benefits:**
- Reduces VRAM usage by approximately **50%**
- Often **faster** on GPUs with Tensor Cores (RTX 20/30/40 series)
- Minimal impact on output quality for most use cases

**When to use:**
- Always enable unless you have 24GB+ VRAM and notice quality issues
- Safe for all architectural interior work
- Very rare edge cases may show minor artifacts in fine details

---

#### Attention Slicing

**What it does:** Splits the attention computation into smaller sequential chunks instead of computing it all at once.

**Technical details:**
- The attention mechanism in diffusion models requires storing large intermediate matrices
- Attention slicing processes these in smaller "slices" sequentially
- Calls `pipeline.enable_attention_slicing()` on the diffusers pipeline

**Benefits:**
- Reduces **peak** VRAM usage by ~25-40%
- Allows running on GPUs that would otherwise run out of memory

**Trade-offs:**
- Generation takes **longer** (10-30% slower)
- Sequential processing can't be parallelized

**When to use:**
- Enable if you get OOM (Out of Memory) errors with just FP16
- Good for 8-12 GB GPUs running Mini Model
- Disable if you have plenty of VRAM and want faster generation

---

#### CPU Offload

**What it does:** Moves model layers to system RAM when not actively being used, then moves them back to GPU when needed.

**Technical details:**
- Uses `pipeline.enable_sequential_cpu_offload()` from diffusers
- Only the currently active layer stays on GPU
- Other layers wait in system RAM

**Benefits:**
- **Dramatically** reduces VRAM requirements (can run on 6-8 GB GPUs)
- Makes it possible to run large models on consumer GPUs

**Trade-offs:**
- **Significantly slower** (2-5x longer generation time)
- Requires sufficient system RAM (16GB+ recommended)
- Heavy CPU-GPU data transfer overhead

**When to use:**
- Only as a **last resort** for very limited VRAM (6-8 GB GPUs)
- When you need to run Full Model but only have 12 GB VRAM
- Avoid if possible - use RunPod Serverless instead for faster results

---

### VRAM Requirements

| Configuration | Approximate VRAM | Generation Time |
|--------------|------------------|-----------------|
| Full Model (no optimization) | 29 GB | 5-10 min |
| Full Model + FP16 | ~15 GB | 5-10 min |
| Full Model + FP16 + Attention Slicing | ~10-12 GB | 7-15 min |
| Mini Model (no optimization) | 21 GB | 2-5 min |
| Mini Model + FP16 | ~10 GB | 2-5 min |
| Mini Model + FP16 + Attention Slicing | ~6-8 GB | 3-7 min |
| Any + CPU Offload | ~6-8 GB | 15-30 min |

### Recommended Settings by GPU

| GPU VRAM | Recommended Configuration |
|----------|--------------------------|
| 24GB+ (RTX 4090, A100) | FP16 ON, others OFF |
| 16GB (RTX 4080, A4000) | FP16 ON, Attention Slicing ON |
| 12GB (RTX 3080, 4070) | FP16 ON, Attention Slicing ON, Mini Model |
| 8GB (RTX 3070, 4060) | All ON, Mini Model only |
| <8GB | Use RunPod Serverless instead |

### Architectural Interior Settings

For architectural interiors with Hunyuan3D:

```
Model: Mini Model (Faster) for iteration, Full Model for final
Seed: Fixed for reproducibility
Guidance Scale: 10.0-12.0 (higher for detailed objects)
Inference Steps: 40-60 (higher for complex shapes)
Octree Resolution: 380-450 (balance detail vs speed)
Remove Background: true (for object isolation)

Memory (if running locally):
FP16: ON (always recommended)
Attention Slicing: ON if needed
CPU Offload: OFF unless necessary
```

**Tips for Architectural Interiors:**
- Excellent for generating furniture from text descriptions
- "Modern minimalist sofa, white leather, chrome legs"
- "Art deco floor lamp, brass finish, geometric shade"
- Works well for: furniture, decor, lighting fixtures
- Can generate from reference images of real furniture
- Combine with other models for complete room scenes
- Use RunPod Serverless to avoid local VRAM limitations
"""


TWODGS_HELP = """
## 2DGS Pipeline (Video → Mesh)

**What it does:** The 2DGS Pipeline converts Gen3C videos into high-quality 3D meshes using NVIDIA ViPE for pose extraction and 2D Gaussian Splatting for reconstruction. This is a one-shot pipeline that handles the entire conversion process automatically.

**Pipeline Steps:**
1. **ViPE** extracts camera poses, intrinsics, and depth maps from video
2. **Converter** transforms ViPE output to COLMAP format for 2DGS
3. **Point Cloud Generator** creates initial 3D points from depth maps
4. **2DGS Training** produces 2D Gaussian splats (flat disks on surfaces)
5. **Mesh Extraction** via 2DGS native extraction with depth fusion

### Key Settings

| Setting | Description | Options | Default |
|---------|-------------|---------|---------|
| **Training Iterations** | 2DGS training steps | 1000-10000 | 5000 |
| **Mesh Quality** | Extraction quality preset | fast/balanced/high/ultra | high |
| **Output Format** | Mesh file format | GLB/OBJ/PLY | GLB |

### Mesh Quality Presets

| Preset | Resolution | Voxel Size | Clusters | Best For |
|--------|------------|------------|----------|----------|
| **fast** | 512 | 0.01 | 1 | Quick previews |
| **balanced** | 512 | 0.006 | 50 | Good balance |
| **high** | 1024 | 0.004 | 100 | Most use cases |
| **ultra** | 2048 | 0.002 | 200 | Maximum detail |

**Why 2DGS gives better meshes:** Unlike 3D Gaussian Splatting, 2DGS uses flat 2D disks that naturally align with surfaces. This means the Gaussians themselves define the geometry, resulting in ⭐⭐⭐⭐⭐ quality compared to ⭐⭐⭐ for standard TSDF.

### Processing Times (241-frame video)

| Stage | Time |
|-------|------|
| ViPE pose extraction | ~5 min |
| 2DGS training (5000 iter) | ~1 min |
| Mesh extraction (high) | ~1-2 min |
| **Total** | **~8-12 min** |

### Recommended Settings

For best results with Gen3C architectural videos:

```
Training Iterations: 5000-7000 (more for complex scenes)
Mesh Quality: high (use ultra for final renders)
Output Format: GLB (best compatibility)
```

### Input Requirements

- **Video source:** Gen3C output video (MP4)
- **Frame count:** 121-361 frames works best
- **Camera motion:** Smooth trajectories (clockwise, counterclockwise)
- **Content:** Architectural interiors work especially well

### Output Quality Tips

**Problem: Mesh has holes or gaps**
→ Increase training iterations to 7000-10000

**Problem: Mesh lacks fine detail**
→ Use "ultra" mesh quality (takes longer)

**Problem: Noisy geometry**
→ Use post-processing in Mesh Cleanup tab

**Problem: Slow processing**
→ Use "fast" quality for previews, then "high" for final

### Typical Workflow

1. Generate video using **Gen3C** tab (clockwise trajectory, 241 frames)
2. Use **2DGS** tab to convert video to mesh
3. Clean up mesh in **Mesh** tab if needed (remove floaters, decimate)
4. Export final mesh for use in Unity, Blender, or web viewers

### Technical Details

- Uses NVIDIA ViPE for monocular video pose estimation
- 2D Gaussian Splatting with depth supervision
- 2DGS native mesh extraction (flat disks define surface geometry)
- Outputs include vertex colors from video frames
"""


MESH_HELP = """
## MESH Extraction

**What it does:** Converts 3D Gaussian Splat (3DGS) files to traditional mesh formats (GLB/OBJ) using Poisson surface reconstruction. This allows 3DGS outputs from Lyra, SHARP, or other sources to be used in standard 3D software.

### Key Settings

| Setting | Description | Range | Default |
|---------|-------------|-------|---------|
| **Input PLY** | Source 3DGS PLY file | file path | - |
| **Poisson Depth** | Reconstruction detail level | 6-12 | 9 |
| **Point Weight** | Influence of input points | 0.0-10.0 | 4.0 |
| **Scale** | Output mesh scale | 0.1-10.0 | 1.0 |
| **Output Format** | GLB or OBJ | glb/obj | glb |

### Architectural Interior Settings

For architectural interiors:

```
Poisson Depth: 10-11 (higher for detailed interiors)
Point Weight: 3.0-5.0 (balance detail vs smoothness)
Scale: 1.0 (maintain original scale)
Output Format: GLB (preserves vertex colors as texture)
```

**Tips for Architectural Interiors:**
- Higher Poisson depth = more detail but longer processing
- Lower point weight = smoother surfaces (good for walls)
- Higher point weight = more detail preservation (good for furniture)
- Process individual objects separately for best results
- Large room scans may need to be segmented first
"""


MESH_CLEANUP_HELP = """
## Mesh Cleanup

**What it does:** Cleans up meshes generated by AI models (Hunyuan3D, SHARP, TRELLIS.2, etc.) for use in Unity, Blender, or other production environments. Addresses common issues like floating artifacts, excessive polygon counts, and inconsistent normals.

### Key Settings

| Setting | Description | Range | Default |
|---------|-------------|-------|---------|
| **Target Triangles** | Reduce mesh to this triangle count (0 = no decimation) | 0-500,000 | 150,000 |
| **Pre-Decimation Smooth** | Smoothing BEFORE decimation | 0-5 | 0 |
| **Preserve Detail** | Use higher quality decimation algorithm | on/off | on |
| **Post-Decimation Smooth** | Smoothing AFTER decimation (softens hard edges) | 0-5 | 2 |
| **Remove Small Components** | Delete disconnected floating artifacts | on/off | on |
| **Fix Normals** | Recompute and fix inconsistent normals | on/off | on |
| **Fill Holes** | Attempt to close holes in mesh | on/off | off |
| **Aggressive Mode** | Enable all cleanup options with stronger settings | on/off | off |
| **Min Component Ratio** | Keep components with at least this % of total faces | 0.1%-10% | 1% |

### Understanding Each Option

**Target Triangles:**
- 500,000: High detail, large file size
- 300,000: Recommended for detailed architectural interiors
- 150,000: Good balance for most uses
- 50,000: Low poly, fast rendering, mobile-friendly
- 0: No decimation, keep original triangle count

**Preserve Detail (NEW - IMPORTANT):**
When ON, uses quadric decimation which better preserves important edges and detail. Slower but produces much better results. **Always keep ON for architectural interiors.**

When OFF, uses fast_simplification which is faster but creates more faceted/sharp edges.

**Pre-Decimation Smooth:**
Smoothing applied BEFORE decimation. Generally keep at 0 - smoothing before decimation can blur important details.

**Post-Decimation Smooth (NEW - IMPORTANT):**
Smoothing applied AFTER decimation. This is the key to softening the hard/faceted edges that decimation creates.
- 0: No post-smoothing (keep sharp decimated edges)
- 1-2: Light smoothing (recommended for architectural)
- 3-4: Moderate smoothing (good for organic shapes)
- 5: Heavy smoothing (may over-smooth)

**Remove Small Components:**
AI-generated meshes often have floating artifacts - small disconnected pieces that appear as noise. This option removes components smaller than the Min Component Ratio threshold.

**Fix Normals:**
Ensures all face normals point outward consistently. Essential for proper lighting in game engines.

**Fill Holes:**
Attempts to close gaps in the mesh. Use with caution - can create unwanted geometry on intentionally open surfaces.

**Aggressive Mode:**
Enables: 100k triangle target, 2+ smoothing passes, hole filling, 2% component threshold. Use for heavily problematic meshes.

### Architectural Interior Settings

For architectural interior meshes (RECOMMENDED):

```
Target Triangles: 250,000-350,000 (preserve detail)
Pre-Decimation Smooth: 0 (don't blur before decimation)
Preserve Detail: ON (critical for quality)
Post-Decimation Smooth: 2 (soften decimation artifacts)
Remove Small Components: ON (clean up artifacts)
Fix Normals: ON (essential for lighting)
Fill Holes: OFF (preserve intentional openings like windows)
Aggressive Mode: OFF
Min Component Ratio: 1% (default)
```

**Why these settings work:**
- Higher triangle target (300k vs 150k) preserves more detail
- Preserve Detail ON uses better decimation algorithm
- Post-decimation smoothing softens the hard edges created by decimation
- No pre-decimation smoothing keeps original detail intact

**Tips for Architectural Interiors:**
- **Problem: Faceted/sharp edges after cleanup** → Increase Post-Decimation Smooth to 2-3
- **Problem: Lost too much detail** → Increase Target Triangles to 300k-400k
- **Problem: Edges still too sharp** → Enable Preserve Detail
- Higher triangle counts for detailed furniture, lower for walls
- Always fix normals for proper lighting in Unity/Unreal
- Use Analyze first to check mesh quality before cleanup
- For very large meshes, consider splitting into separate objects first

### Analyze vs Clean Up

**Analyze Button:**
- Inspects mesh without modifying
- Shows: vertices, triangles, components, size, watertight status
- Identifies issues: floating components, holes, high poly count
- Use this first to understand what cleanup is needed

**Clean Up Button:**
- Applies selected cleanup operations
- Saves cleaned mesh to output location
- Shows reduction statistics and operations performed
"""


GENERAL_TIPS_HELP = """
## Workflow Recommendations

### Best Model for Each Task

| Task | Recommended Model | Why |
|------|-------------------|-----|
| **Single room photo → 3D mesh** | SHARP | Fast, high detail, direct GLB output |
| **Single room photo → 3DGS** | Lyra | Best 3D Gaussian Splat quality |
| **Virtual tour video from photo** | Gen3C | Camera-controlled video with 3D consistency |
| **Furniture generation** | Hunyuan3D | Text-to-3D for custom pieces |
| **Clean mesh output** | TRELLIS.2 | Best topology for editing |
| **3DGS to mesh** | MESH tab | Poisson reconstruction |

### Input Image Guidelines

1. **Resolution:** Minimum 1024x1024, ideally 2048x2048
2. **Lighting:** Even, diffuse lighting without harsh shadows
3. **Angle:** 3/4 view captures more depth information
4. **Focus:** Sharp focus throughout, avoid depth-of-field blur
5. **Content:** Single room or object, avoid mirrors/glass

### Recommended Workflow for Complete Rooms

1. **Capture:** Take high-quality photos from key angles
2. **Generate 3D:** Use SHARP or Lyra for 3D reconstruction
3. **Create Video:** Use Gen3C to create walkthrough videos from photos
4. **Convert:** Use MESH tab to convert 3DGS to editable mesh
5. **Enhance:** Add furniture with Hunyuan3D or TRELLIS.2
6. **Composite:** Combine in Blender or Unity

### Output Format Guide

| Format | Best For | Software Compatibility |
|--------|----------|----------------------|
| **GLB** | Textured meshes | Blender, Unity, Unreal, Web |
| **OBJ** | Mesh editing | All 3D software |
| **PLY** | Point clouds, 3DGS | Blender, specialized viewers |
| **3DGS** | Real-time rendering | Gaussian splat viewers |

### Performance Tips

- Start with lower settings to test, then increase for final output
- Use fixed seeds for reproducibility when iterating
- Process during off-peak hours for faster RunPod response
- Save intermediate outputs (PLY) before mesh conversion
"""

