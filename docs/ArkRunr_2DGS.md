# ArkRunr Stage Reconstruction Pipeline

**Last Updated:** December 29, 2025  
**Purpose:** Accurate 3D architectural reconstruction for performer integration in Unity

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Use Case Analysis](#use-case-analysis)
3. [Stage Volume Reconstruction](#stage-volume-reconstruction)
4. [ArkRunr Pipeline Implementation](#arkrunr-pipeline-implementation)
5. [Front-Arc View Generation](#front-arc-view-generation)
6. [Unity Integration](#unity-integration)
7. [Quality Tiers](#quality-tiers)
8. [Implementation Code](#implementation-code)
9. [Production Workflow](#production-workflow)
10. [**Implementation Roadmap**](#implementation-roadmap) ← START HERE

---

## Executive Summary

### The ArkRunr Use Case

ArkRunr requires **accurate 3D architectural reconstructions** derived from interior photographs. The goal is to recreate real spaces as faithful 3D environments where:

- **Performers exist INSIDE the 3D space** - moving throughout the volume
- **Objects can be anywhere** - in front of, alongside, or behind the performer
- **Camera angles are user-defined** - arbitrary positions, always pointed at stage/performer
- **Lighting is handled by ArkRunr** - model should have neutral/no baked lighting
- **Textures can be replaced** - accurate geometry is the priority; surfaces can be retextured in ArkRunr/Unity
- Must be Unity-compatible (GLB/FBX mesh format)

### Key Priorities (In Order)

| Priority | What | Why |
|----------|------|-----|
| **1. Geometry Accuracy** | Walls, floors, ceilings, architectural features | Foundation of the 3D space |
| **2. Spatial Relationships** | Correct proportions, depths, angles | Performer must fit naturally |
| **3. Structural Elements** | Columns, stairs, railings, platforms | Define the usable space |
| **4. Neutral Appearance** | No baked lighting, clean surfaces | ArkRunr adds lighting/textures |
| **5. Base Textures** | Optional - can be replaced | Nice-to-have, not critical |

### Diverse Interior Types

ArkRunr stages can be **any architectural interior**. Examples include:

```
┌────────────────────────────────────────────────────────────────────────────┐
│                        EXAMPLE INTERIOR TYPES                               │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INDUSTRIAL/WAREHOUSE         CLASSICAL/THEATER         MODERN/MINIMAL     │
│  ───────────────────         ─────────────────         ────────────────    │
│  • Open floor plans          • Ornate columns          • Clean lines       │
│  • Exposed structure         • Balconies/boxes         • Large windows     │
│  • High ceilings             • Proscenium arch         • Open concept      │
│  • Metal/concrete            • Decorative molding      • Glass/steel       │
│                                                                             │
│  INTIMATE/CLUB               OUTDOOR/COVERED           HISTORIC/ORNATE     │
│  ─────────────               ───────────────           ───────────────     │
│  • Low ceilings              • Pavilions               • Detailed trim     │
│  • Tight spaces              • Amphitheaters           • Period features   │
│  • Bar/seating areas         • Covered stages          • Rich materials    │
│  • Mood lighting (ignored)   • Natural elements        • Complex geometry  │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

### What All Interiors Have in Common

Regardless of style, every ArkRunr stage needs:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│                         [CEILING / OVERHEAD]                                │
│                                                                              │
│        ┌─────────────────────────────────────────────────────────┐          │
│        │                                                         │          │
│  [WALL]│                    PERFORMANCE                          │[WALL]    │
│        │                       VOLUME                            │          │
│        │                                                         │          │
│        │           ┌─────────────────────────┐                   │          │
│        │           │                         │                   │          │
│        │  [OBJECT] │      PERFORMER(S)       │ [OBJECT]          │          │
│        │           │    (move throughout)    │                   │          │
│        │           │                         │                   │          │
│        │           └─────────────────────────┘                   │          │
│        │                                                         │          │
│        └─────────────────────────────────────────────────────────┘          │
│                                                                              │
│                              [FLOOR]                                        │
│                                                                              │
│                    ┌─────────────────────────┐                              │
│                    │   CAMERA (any angle,    │                              │
│                    │   always looking at     │                              │
│                    │   stage/performer)      │                              │
│                    └─────────────────────────┘                              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Critical Requirements

| Requirement | Why It Matters |
|-------------|----------------|
| **Accurate Geometry** | Faithful recreation of the real space |
| **True 3D Volume** | Performer moves throughout, not just in front |
| **Correct Proportions** | Scale must match reality for performer integration |
| **Structural Elements** | Columns, stairs, platforms define usable space |
| **Floor Surface** | Performer stands on it, must be accurate |
| **Neutral Lighting** | ArkRunr handles all lighting; no baked shadows |
| **Clean Surfaces** | Textures optional; can be replaced in post |

### Recommended Approach

| Tier | Method | Geometry | Speed | Use Case |
|------|--------|----------|-------|----------|
| **Quick** | Lyra → SuGaR | ⭐⭐⭐ | 2-3 min | Previews, prototyping |
| **Quality** | Multi-View 2DGS | ⭐⭐⭐⭐ | 15-30 min | Production stages |
| **Premium** | Multi-Photo Capture | ⭐⭐⭐⭐⭐ | 45+ min | Highest accuracy |

### What We're NOT Trying to Do

- ❌ Bake realistic lighting (ArkRunr handles this)
- ❌ Capture specific textures/graffiti (can be added later)
- ❌ Recreate decorative details (edge case, not typical)
- ❌ Limit camera angles (user decides)
- ❌ Restrict to one interior style (any space works)

---

## Use Case Analysis

### Core Requirements

ArkRunr needs to convert photographs of real interiors into accurate 3D reconstructions. The key requirements:

| Requirement | Description |
|-------------|-------------|
| **Geometry First** | Accurate architectural structure is the #1 priority |
| **Any Interior Type** | Industrial, classical, modern, intimate, outdoor covered, etc. |
| **True 3D Volume** | Performer moves throughout the space, not just in one area |
| **Arbitrary Camera** | User places camera anywhere, always pointed at stage/performer |
| **Neutral Lighting** | No baked lighting; ArkRunr adds all lighting dynamically |
| **Replaceable Textures** | Surfaces can be retextured in ArkRunr or Unity |

### What Gets Reconstructed

```yaml
# Architectural Elements to Reconstruct
critical:
  - floors: "Performer stands on these"
  - walls: "Define the space boundaries"
  - ceilings: "If visible in source photo"
  - structural_elements: "Columns, beams, stairs, platforms"
  - major_features: "Doors, windows, arches, alcoves"

important:
  - level_changes: "Steps, raised areas, balconies"
  - railings_barriers: "Safety features, visual boundaries"
  - large_objects: "Furniture, fixtures that define the space"

optional:
  - decorative_details: "Moldings, trim (edge case)"
  - wall_textures: "Can be replaced; geometry matters more"
  - small_objects: "Usually removed or added separately"
```

### Performer Integration

```
                         ┌─────────────────────────────────────┐
                         │                                     │
                         │     RECONSTRUCTED 3D VOLUME         │
                         │                                     │
                         │   ┌─────────────────────────────┐   │
                         │   │                             │   │
                         │   │    Performer can be:        │   │
                         │   │    • In front of objects    │   │
                         │   │    • Behind objects         │   │
                         │   │    • Between objects        │   │
                         │   │    • On platforms           │   │
                         │   │    • On stairs              │   │
                         │   │    • Anywhere in volume     │   │
                         │   │                             │   │
                         │   └─────────────────────────────┘   │
                         │                                     │
                         └─────────────────────────────────────┘
                                          │
                                          ▼
                         ┌─────────────────────────────────────┐
                         │   CAMERA (user-defined position)    │
                         │   • Any angle within the space      │
                         │   • Always looking at performer     │
                         │   • Distance varies per shot        │
                         └─────────────────────────────────────┘
```

### Typical Space Dimensions

Dimensions vary widely by interior type, but common ranges:

```yaml
# Space dimensions vary by type
small_intimate:
  width: 5-10 meters
  depth: 5-10 meters
  height: 3-4 meters
  examples: ["club", "small theater", "studio"]

medium_standard:
  width: 10-20 meters
  depth: 10-20 meters
  height: 4-8 meters
  examples: ["warehouse", "ballroom", "gallery"]

large_venue:
  width: 20-50 meters
  depth: 20-40 meters
  height: 8-15 meters
  examples: ["concert hall", "arena", "cathedral"]

# Camera behavior (user-controlled)
camera:
  position: "arbitrary"
  orientation: "always looking at stage/performer"
  distance: "varies by shot type"
```

### What We MUST Reconstruct

- ✅ **Accurate geometry** - faithful to the source photograph
- ✅ **Full 3D volume** - not a flat backdrop
- ✅ **Floor geometry** - performer stands on it
- ✅ **Walls** - back and sides as visible in source
- ✅ **Structural elements** - columns, beams, stairs, platforms
- ✅ **Level changes** - steps, raised areas, balconies
- ✅ **Correct proportions** - scale must match reality
- ✅ **Correct depth relationships** - parallax must work

### What We Can Simplify

- ⚠️ **Areas not in source photo** - can't reconstruct what we can't see
- ⚠️ **Extreme ceiling detail** - often dark/out of frame
- ⚠️ **Hidden areas** - behind objects, under platforms
- ⚠️ **Surface textures** - can be replaced in ArkRunr/Unity

### What We DON'T Need

- ❌ **Baked lighting** - ArkRunr handles all lighting
- ❌ **Specific textures/graffiti** - can be added later
- ❌ **Decorative details** - edge case, not typical
- ❌ **Physics-accurate materials** - not needed for visualization
- ❌ **Separate mesh for every object** - unified mesh is fine

---

## Stage Volume Reconstruction

### The Core Challenge

Given a single image (or few images) of an architectural interior, reconstruct an **accurate 3D volume** suitable for:

1. **Performer integration** - human exists INSIDE the 3D space, can move throughout
2. **Arbitrary camera angles** - user places camera anywhere, looking at performer
3. **Real-time rendering** - Unity at 60+ FPS
4. **Dynamic lighting** - ArkRunr adds lighting; model should be neutral
5. **Optional retexturing** - surfaces can be changed in post

### Why This Is Harder Than a Flat Backdrop

```
FLAT BACKDROP (Wrong):                    FULL 3D VOLUME (Correct):
                                          
┌─────────────────────┐                   ┌─────────────────────────────┐
│                     │                   │         BACK WALL           │
│   FLAT IMAGE        │                   │    ┌─────────────────┐      │
│                     │                   │    │                 │      │
│   ● Performer       │                   │    │  ● Performer    │      │
│     (in front)      │                   │    │    (INSIDE)     │      │
│                     │                   │    │                 │      │
└─────────────────────┘                   │    └─────────────────┘      │
        │                                 │          STAGE              │
        │                                 │            │                │
   No parallax                            │    ┌───────┴───────┐        │
   No depth                               │    │   SPEAKERS    │        │
   Performer floats                       └────┴───────────────┴────────┘
                                                      │
                                               FLOOR + DEPTH
                                               Real parallax!
```

### Depth Layers in a Typical ArkRunr Stage

```
CAMERA ──────────────────────────────────────────────────────────► DEPTH

Layer 0      Layer 1         Layer 2         Layer 3         Layer 4
(Nearest)                                                    (Farthest)
   │            │               │               │               │
   ▼            ▼               ▼               ▼               ▼
┌──────┐   ┌────────┐      ┌────────┐      ┌────────┐      ┌────────┐
│Floor │   │Speakers│      │ Stage  │      │Performer│     │ Back   │
│Edge  │   │Monitors│      │Platform│      │  Area   │     │ Wall   │
│      │   │        │      │        │      │         │     │        │
│ ~2m  │   │  ~4m   │      │  ~6m   │      │  ~8m    │     │ ~12m   │
└──────┘   └────────┘      └────────┘      └────────┘      └────────┘

PARALLAX EFFECT:
- When camera moves LEFT, Layer 0 shifts RIGHT more than Layer 4
- This depth relationship MUST be preserved in reconstruction
- Flat backdrop = no parallax = fake-looking result
```

### Why 2DGS Is Ideal for Stage Volumes

| Property | Benefit for Stage Reconstruction |
|----------|----------------------------------|
| **Flat Gaussians (2D disks)** | Perfect for walls, floors, ceilings - dominant surfaces |
| **Surface-aligned** | Clean planar geometry extraction |
| **View-consistent depth** | Accurate depth for all camera angles |
| **Normal consistency** | Smooth surfaces, sharp edges at intersections |
| **Direct mesh extraction** | TSDF fusion works well on 2DGS output |

### ArkRunr-Specific Reconstruction Priorities

| Priority | Challenge | Solution |
|----------|-----------|----------|
| **1. Geometry accuracy** | Must match source photo | Strong depth supervision |
| **2. Structural elements** | Columns, stairs, platforms | Don't over-smooth thin structures |
| **3. Correct proportions** | Scale must be accurate | Floor plane constraint |
| **4. Neutral appearance** | No baked lighting | Minimize lighting in texture |
| **5. Diverse interiors** | Any architectural style | Generic, not style-specific |

### Specialized Losses for ArkRunr

```python
def compute_arkrunr_loss(rendered, target, gaussians, config):
    """
    Loss function optimized for ArkRunr architectural reconstruction.
    
    PRIORITIES:
    1. Geometry accuracy (depth)
    2. Structural preservation
    3. Neutral appearance (minimize lighting artifacts)
    """
    # GEOMETRY ACCURACY (highest priority)
    # Depth is more important than photometric for ArkRunr
    loss_depth = depth_distortion_loss(rendered.depth_weights)
    
    # Depth edge preservation - maintain structural boundaries
    loss_depth_edge = depth_edge_aware_loss(
        rendered.depth, 
        target.depth,
        edge_threshold=0.3  # meters - detect structural changes
    )
    
    # Normal consistency for planar architectural surfaces
    loss_normal = normal_consistency_loss(gaussians)
    
    # STRUCTURAL ELEMENT PRESERVATION
    # Preserve thin structures (columns, railings, beams)
    loss_structure = thin_structure_loss(
        gaussians, 
        min_thickness=0.05  # 5cm minimum
    )
    
    # FLOOR PLANE CONSTRAINT
    # Critical for performer placement
    loss_floor = floor_plane_loss(
        gaussians,
        expected_height=config.floor_height,
        tolerance=0.1  # 10cm
    )
    
    # PHOTOMETRIC (lower priority - textures can be replaced)
    # Reduced weight compared to standard 2DGS
    loss_rgb = l1_loss(rendered.rgb, target.rgb) + \
               0.1 * (1 - ssim(rendered.rgb, target.rgb))  # Reduced SSIM weight
    
    # NEUTRAL LIGHTING ENCOURAGEMENT
    # Penalize high contrast (indicates baked lighting)
    loss_neutral = lighting_neutrality_loss(rendered.rgb)
    
    return (
        2.0 * loss_depth +        # Highest: geometry accuracy
        0.5 * loss_depth_edge +   # High: structural boundaries
        0.2 * loss_normal +       # Medium: surface consistency
        0.2 * loss_structure +    # Medium: thin structures
        0.1 * loss_floor +        # Medium: floor constraint
        0.5 * loss_rgb +          # Lower: appearance (can be replaced)
        0.1 * loss_neutral        # Low: encourage neutral lighting
    )


def lighting_neutrality_loss(rgb):
    """
    Encourage neutral lighting by penalizing high local contrast.
    This helps avoid baking shadows/highlights into the reconstruction.
    """
    # Compute local contrast
    kernel_size = 5
    local_mean = torch.nn.functional.avg_pool2d(rgb, kernel_size, stride=1, padding=kernel_size//2)
    local_variance = torch.nn.functional.avg_pool2d((rgb - local_mean)**2, kernel_size, stride=1, padding=kernel_size//2)
    
    # Penalize high variance (indicates lighting variation)
    return local_variance.mean()
```

### Depth Layer Handling

```python
def detect_depth_layers(depth_map):
    """
    Detect major depth layers in the interior image.
    Returns layer boundaries for loss computation.
    
    Works for any interior type - not specific to industrial stages.
    """
    # Histogram-based layer detection
    hist, bins = np.histogram(depth_map.flatten(), bins=50)
    
    # Find peaks (major depth layers)
    from scipy.signal import find_peaks
    peaks, properties = find_peaks(hist, height=hist.max() * 0.1, distance=3)
    
    layer_depths = bins[peaks]
    
    # Generic architectural layers (adapt to actual content)
    # These are starting points - actual layers detected from image
    typical_layers = {
        'near_floor': 'Nearest visible floor area',
        'foreground_objects': 'Furniture, fixtures in front',
        'mid_ground': 'Main usable space',
        'back_surfaces': 'Walls, back of room',
    }
    
    return {
        'detected': layer_depths,
        'num_layers': len(layer_depths),
        'boundaries': compute_layer_boundaries(layer_depths)
    }
```

---

## ArkRunr Pipeline Implementation

### Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    ARKRUNR STAGE VOLUME RECONSTRUCTION PIPELINE                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌──────────────┐                                                               │
│  │ Input Image  │  (Stage photo: industrial venue, graffiti walls,             │
│  │ (Stage)      │   scaffolding, speaker stacks, lighting rigs)                │
│  └──────┬───────┘                                                               │
│         │                                                                        │
│         ▼                                                                        │
│  ┌─────────────────────────────────────────────────────────────┐                │
│  │ PREPROCESSING                                                │                │
│  │ ─────────────                                                │                │
│  │ • Depth estimation (ZoeDepth for indoor scenes)             │                │
│  │ • Depth layer detection (speakers, stage, walls)            │                │
│  │ • Structural element detection (scaffolding, railings)      │                │
│  │ • Scale estimation (from known objects or user input)       │                │
│  └──────────────────────────┬──────────────────────────────────┘                │
│                             │                                                    │
│             ┌───────────────┼───────────────┐                                   │
│             │               │               │                                   │
│             ▼               ▼               ▼                                   │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐                   │
│  │ QUICK PATH     │ │ QUALITY PATH    │ │ PREMIUM PATH    │                   │
│  │ (Lyra)         │ │ (Multi-View     │ │ (Multi-Photo    │                   │
│  │                │ │  2DGS)          │ │  Capture)       │                   │
│  └────────┬────────┘ └────────┬────────┘ └────────┬────────┘                   │
│           │                   │                   │                             │
│           ▼                   ▼                   ▼                             │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐                   │
│  │ GEN3C Video    │ │ SV3D Front-Arc  │ │ COLMAP SfM      │                   │
│  │ (Orbital)      │ │ View Generation │ │ (Real Photos)   │                   │
│  │                │ │ (120° arc only) │ │                 │                   │
│  └────────┬────────┘ └────────┬────────┘ └────────┬────────┘                   │
│           │                   │                   │                             │
│           ▼                   ▼                   ▼                             │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐                   │
│  │ 3DGS Output    │ │ 2DGS Training   │ │ 2DGS Training   │                   │
│  │ (Lyra)        │ │ (Depth-aware)   │ │ (Full quality)  │                   │
│  └────────┬────────┘ └────────┬────────┘ └────────┬────────┘                   │
│           │                   │                   │                             │
│           ▼                   ▼                   ▼                             │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐                   │
│  │ SuGaR Mesh     │ │ TSDF Mesh       │ │ TSDF Mesh       │                   │
│  │ Extraction     │ │ Extraction      │ │ Extraction      │                   │
│  │ (Fast)         │ │ (Depth-guided)  │ │ (High quality)  │                   │
│  └────────┬────────┘ └────────┬────────┘ └────────┬────────┘                   │
│           │                   │                   │                             │
│           └───────────────────┼───────────────────┘                             │
│                               │                                                  │
│                               ▼                                                  │
│  ┌─────────────────────────────────────────────────────────────┐                │
│  │ UNITY EXPORT                                                 │                │
│  │ ─────────────                                                │                │
│  │ • Mesh decimation (100k-200k triangles)                     │                │
│  │ • UV unwrapping (preserve texture detail)                   │                │
│  │ • Texture baking (2K/4K with lighting baked in)             │                │
│  │ • Performer placement marker (stage center)                 │                │
│  │ • GLB export (Y-up, metric scale)                           │                │
│  └──────────────────────────┬──────────────────────────────────┘                │
│                             │                                                    │
│                             ▼                                                    │
│  ┌─────────────────────────────────────────────────────────────┐                │
│  │ OUTPUT: Unity-Ready Stage Volume                            │                │
│  │ ─────────────────────────────────                           │                │
│  │ • stage_volume.glb (full 3D geometry)                       │                │
│  │ • stage_diffuse.png (2K/4K baked texture)                   │                │
│  │ • metadata.json (scale, bounds, performer position)         │                │
│  │ • depth_layers.json (for parallax verification)             │                │
│  └─────────────────────────────────────────────────────────────┘                │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Key Differences from Flat Backdrop Pipeline

| Aspect | Flat Backdrop (Wrong) | Full Volume (Correct) |
|--------|----------------------|----------------------|
| **Geometry** | Single plane or shallow depth | Multi-layer 3D volume |
| **Depth** | ~0.5m variation | 10-15m depth range |
| **Parallax** | Minimal | Full parallax on camera movement |
| **Performer** | In front of backdrop | Inside the 3D space |
| **Floor** | Not needed | Critical for performer placement |
| **Foreground** | None | Speakers, monitors for depth |

### Core Implementation

```python
# generators/arkrunr_stage.py

#!/usr/bin/env python3
"""
ArkRunr Stage Reconstruction Pipeline

Specialized pipeline for converting architectural interior images
into Unity-ready 3D stage backdrops.
"""

import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum


class QualityTier(Enum):
    QUICK = "quick"      # Lyra → SuGaR (2-3 min)
    QUALITY = "quality"  # Front-Arc 2DGS (10-15 min)
    PREMIUM = "premium"  # Multi-photo capture (30+ min)


@dataclass
class StageConfig:
    """Configuration for stage volume reconstruction."""
    
    # Stage dimensions (meters)
    stage_width: float = 15.0           # Wall to wall
    stage_depth: float = 12.0           # Camera to back wall
    stage_height: float = 7.0           # Floor to ceiling
    
    # Performer area
    performer_position: Tuple[float, float, float] = (0.0, 0.0, 8.0)  # x, y, z from camera
    performer_area_radius: float = 3.0  # Movement space
    
    # Viewing arc parameters
    arc_degrees: float = 120.0          # Total horizontal arc
    arc_center: float = 0.0             # Center azimuth
    elevation_min: float = -10.0        # Lowest camera angle (floor level)
    elevation_max: float = 25.0         # Highest camera angle (mezzanine)
    
    # View generation
    num_azimuth_views: int = 9          # Views across arc (more for volume)
    num_elevation_views: int = 4        # Views at different heights
    camera_distance_min: float = 5.0    # Closest camera position
    camera_distance_max: float = 15.0   # Farthest camera position
    
    # Depth layer handling
    preserve_depth_layers: bool = True  # Maintain parallax structure
    depth_layer_count: int = 5          # Expected depth layers
    
    # 2DGS training (stronger depth emphasis)
    iterations: int = 20000             # More iterations for volume
    depth_weight: float = 1.5           # Stronger depth (was 1.0)
    depth_edge_weight: float = 0.3      # Preserve depth discontinuities
    normal_weight: float = 0.1          # Normal consistency
    structure_weight: float = 0.1       # Thin structure preservation
    
    # Output
    target_triangles: int = 150000      # Higher poly for volume
    texture_resolution: int = 2048      # Texture size
    output_format: str = "glb"          # Unity format
    
    # Unity-specific
    up_axis: str = "Y"                  # Unity uses Y-up
    scale_factor: float = 1.0           # Metric scale


@dataclass
class CameraPosition:
    """Represents a camera position in the viewing arc."""
    name: str
    azimuth: float      # Horizontal angle from center
    elevation: float    # Vertical angle from horizon
    distance: float     # Distance from stage center
    description: str = ""


# Standard ArkRunr camera positions for stage volumes
# Based on typical concert/performance camera placements
ARKRUNR_CAMERAS: List[CameraPosition] = [
    # Wide shots
    CameraPosition("Wide Master", 0, 5, 12, "Full stage view with floor visible"),
    CameraPosition("Wide Left", -45, 5, 12, "Wide from stage left"),
    CameraPosition("Wide Right", 45, 5, 12, "Wide from stage right"),
    
    # Medium shots (performer level)
    CameraPosition("Center Close", 0, 10, 6, "Medium shot at stage level"),
    CameraPosition("Left 30", -30, 10, 8, "Three-quarter from left"),
    CameraPosition("Right 30", 30, 10, 8, "Three-quarter from right"),
    CameraPosition("Left 60", -60, 8, 8, "Wide angle from left"),
    CameraPosition("Right 60", 60, 8, 8, "Wide angle from right"),
    
    # Low angles (dramatic)
    CameraPosition("Low Center", 0, -10, 5, "Dramatic low shot"),
    CameraPosition("Low Left", -30, -5, 6, "Low from left"),
    CameraPosition("Low Right", 30, -5, 6, "Low from right"),
    
    # High angles
    CameraPosition("High Wide", 0, 25, 15, "Overhead establishing"),
    CameraPosition("Balcony Left", -20, 20, 10, "Balcony view left"),
    CameraPosition("Balcony Right", 20, 20, 10, "Balcony view right"),
]


class ArkRunrStageGenerator:
    """
    Main generator class for ArkRunr stage reconstruction.
    
    Supports three quality tiers:
    - QUICK: Uses existing Lyra pipeline
    - QUALITY: Uses front-arc 2DGS
    - PREMIUM: Uses multi-photo capture
    """
    
    def __init__(
        self,
        config: Optional[StageConfig] = None,
        output_dir: str = "/srv/searidge_share/outputs/arkrunr_stages",
    ):
        self.config = config or StageConfig()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Lazy-load models
        self._sv3d_model = None
        self._depth_model = None
        self._normal_model = None
    
    def generate(
        self,
        image_path: str,
        quality: QualityTier = QualityTier.QUALITY,
        stage_name: str = "stage",
    ) -> Dict[str, Any]:
        """
        Generate Unity-ready stage from interior image.
        
        Args:
            image_path: Path to input interior image
            quality: Quality tier (QUICK, QUALITY, PREMIUM)
            stage_name: Name for output files
        
        Returns:
            Dict with paths to generated assets
        """
        print(f"[ArkRunr] Generating stage: {stage_name}")
        print(f"[ArkRunr] Quality tier: {quality.value}")
        print(f"[ArkRunr] Input: {image_path}")
        
        if quality == QualityTier.QUICK:
            return self._generate_quick(image_path, stage_name)
        elif quality == QualityTier.QUALITY:
            return self._generate_quality(image_path, stage_name)
        else:
            return self._generate_premium(image_path, stage_name)
    
    # =========================================================================
    # QUICK PATH: Lyra → SuGaR
    # =========================================================================
    
    def _generate_quick(self, image_path: str, stage_name: str) -> Dict[str, Any]:
        """
        Quick generation using existing Lyra pipeline.
        
        Pipeline: Image → Lyra (GEN3C) → 3DGS → SuGaR → GLB
        """
        from generators.lyra import run_lyra_runpod
        from generators.sugar import run_sugar_runpod
        
        # Step 1: Run Lyra
        print("[ArkRunr/Quick] Step 1: Running Lyra...")
        lyra_result = run_lyra_runpod(
            image_path=image_path,
            generation_mode="Static (Image → 3DGS)",
            num_views=8,
        )
        
        if not lyra_result.get("ply_path"):
            raise RuntimeError("Lyra failed to generate PLY")
        
        # Step 2: Run SuGaR mesh extraction
        print("[ArkRunr/Quick] Step 2: Running SuGaR mesh extraction...")
        sugar_result = run_sugar_runpod(
            ply_path=lyra_result["ply_path"],
            target_vertices=self.config.target_triangles,
        )
        
        if not sugar_result.get("mesh_path"):
            raise RuntimeError("SuGaR failed to extract mesh")
        
        # Step 3: Post-process for Unity
        print("[ArkRunr/Quick] Step 3: Unity post-processing...")
        unity_mesh = self._unity_postprocess(
            sugar_result["mesh_path"],
            stage_name,
        )
        
        return {
            "stage_name": stage_name,
            "quality": "quick",
            "glb_path": unity_mesh,
            "ply_path": lyra_result["ply_path"],
            "texture_path": None,  # Vertex colors only
        }
    
    # =========================================================================
    # QUALITY PATH: Front-Arc 2DGS
    # =========================================================================
    
    def _generate_quality(self, image_path: str, stage_name: str) -> Dict[str, Any]:
        """
        Quality generation using front-arc 2DGS.
        
        Pipeline: Image → SV3D (front-arc) → 2DGS → TSDF → GLB
        """
        # Step 1: Generate front-arc views
        print("[ArkRunr/Quality] Step 1: Generating front-arc views...")
        views, poses = self._generate_front_arc_views(image_path)
        
        # Step 2: Estimate depth and normals for each view
        print("[ArkRunr/Quality] Step 2: Estimating depth and normals...")
        depths, normals = self._estimate_geometry(views)
        
        # Step 3: Train 2DGS
        print("[ArkRunr/Quality] Step 3: Training 2DGS...")
        gaussians_2d = self._train_2dgs(views, poses, depths, normals)
        
        # Step 4: Extract mesh via TSDF
        print("[ArkRunr/Quality] Step 4: Extracting mesh...")
        mesh_path = self._extract_mesh_tsdf(gaussians_2d, poses)
        
        # Step 5: Bake textures
        print("[ArkRunr/Quality] Step 5: Baking textures...")
        texture_path = self._bake_texture(gaussians_2d, mesh_path, poses)
        
        # Step 6: Unity post-processing
        print("[ArkRunr/Quality] Step 6: Unity post-processing...")
        unity_mesh = self._unity_postprocess(mesh_path, stage_name, texture_path)
        
        return {
            "stage_name": stage_name,
            "quality": "quality",
            "glb_path": unity_mesh,
            "ply_path": self._save_2dgs_ply(gaussians_2d, stage_name),
            "texture_path": texture_path,
        }
    
    def _generate_front_arc_views(
        self,
        image_path: str,
    ) -> Tuple[List, List]:
        """
        Generate views in the front-facing arc using SV3D.
        
        Instead of full 360° orbital, generates views only in the
        audience-facing arc (-60° to +60°).
        """
        # Load SV3D model
        if self._sv3d_model is None:
            self._sv3d_model = self._load_sv3d()
        
        # Calculate view positions in the arc
        arc_half = self.config.arc_degrees / 2
        azimuths = np.linspace(
            self.config.arc_center - arc_half,
            self.config.arc_center + arc_half,
            self.config.num_azimuth_views,
        )
        
        elevations = np.linspace(
            self.config.elevation_min,
            self.config.elevation_max,
            self.config.num_elevation_views,
        )
        
        views = []
        poses = []
        
        # Generate views at each position
        for azimuth in azimuths:
            for elevation in elevations:
                view = self._render_sv3d_view(
                    image_path,
                    azimuth=azimuth,
                    elevation=elevation,
                )
                pose = self._azimuth_elevation_to_pose(azimuth, elevation)
                
                views.append(view)
                poses.append(pose)
        
        print(f"[ArkRunr] Generated {len(views)} views in {self.config.arc_degrees}° arc")
        
        return views, poses
    
    def _train_2dgs(
        self,
        views: List,
        poses: List,
        depths: List,
        normals: List,
    ):
        """
        Train 2DGS on the generated front-arc views.
        
        Uses specialized losses for architectural interiors.
        """
        # Initialize 2DGS from depth
        gaussians = self._initialize_2dgs_from_depth(views[0], depths[0], poses[0])
        
        # Create optimizer
        optimizer = self._create_2dgs_optimizer(gaussians)
        
        num_views = len(views)
        
        for iteration in range(self.config.iterations):
            # Sample random view
            idx = np.random.randint(0, num_views)
            
            target_rgb = views[idx]
            target_depth = depths[idx]
            target_normal = normals[idx]
            pose = poses[idx]
            
            # Render 2DGS
            rendered = self._render_2dgs(gaussians, pose)
            
            # Compute losses
            loss = self._compute_stage_loss(
                rendered,
                target_rgb,
                target_depth,
                target_normal,
                gaussians,
            )
            
            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Densification
            if iteration % 500 == 0 and iteration < self.config.iterations * 0.5:
                self._densify_and_prune(gaussians)
            
            # Progress
            if iteration % 1000 == 0:
                print(f"[ArkRunr/2DGS] Iteration {iteration}/{self.config.iterations}, Loss: {loss.item():.4f}")
        
        return gaussians
    
    def _compute_stage_loss(
        self,
        rendered,
        target_rgb,
        target_depth,
        target_normal,
        gaussians,
    ):
        """
        Compute loss optimized for architectural interior stages.
        """
        # RGB loss
        loss_rgb = l1_loss(rendered.rgb, target_rgb) + \
                   0.2 * (1 - ssim(rendered.rgb, target_rgb))
        
        # Depth loss
        loss_depth = self.config.depth_weight * l1_loss(
            rendered.depth,
            target_depth,
            mask=target_depth > 0,
        )
        
        # Normal loss
        loss_normal = self.config.normal_weight * cosine_loss(
            rendered.normal,
            target_normal,
        )
        
        # 2DGS regularization
        loss_reg = depth_distortion_loss(rendered.depth_weights)
        
        # Planar regularization (for architectural interiors)
        loss_planar = self.config.planar_weight * self._planar_loss(gaussians)
        
        return loss_rgb + loss_depth + loss_normal + 0.01 * loss_reg + loss_planar
    
    def _planar_loss(self, gaussians):
        """
        Encourage Gaussians to align with detected planar regions.
        
        Architectural interiors have dominant planes (walls, floor, ceiling).
        """
        # Compute local normal variance
        # Low variance = planar region = good
        normals = gaussians.get_normals()
        
        # K-nearest neighbors
        knn_indices = self._get_knn(gaussians.positions, k=8)
        
        # Compute normal variance in neighborhood
        neighbor_normals = normals[knn_indices]  # [N, K, 3]
        normal_variance = neighbor_normals.var(dim=1).mean()
        
        return normal_variance
    
    # =========================================================================
    # PREMIUM PATH: Multi-Photo Capture
    # =========================================================================
    
    def _generate_premium(self, image_path: str, stage_name: str) -> Dict[str, Any]:
        """
        Premium generation using multiple real photos.
        
        Expects a directory of images or a list of image paths.
        
        Pipeline: Multi-photos → COLMAP → 2DGS → TSDF → GLB
        """
        # Check if input is directory or single image
        input_path = Path(image_path)
        
        if input_path.is_dir():
            image_paths = list(input_path.glob("*.jpg")) + list(input_path.glob("*.png"))
        else:
            # Single image - fall back to quality path
            print("[ArkRunr/Premium] Single image provided, using quality path")
            return self._generate_quality(image_path, stage_name)
        
        if len(image_paths) < 3:
            print(f"[ArkRunr/Premium] Only {len(image_paths)} images, need 3+. Using quality path")
            return self._generate_quality(str(image_paths[0]), stage_name)
        
        print(f"[ArkRunr/Premium] Found {len(image_paths)} images")
        
        # Step 1: Run COLMAP
        print("[ArkRunr/Premium] Step 1: Running COLMAP...")
        colmap_output = self._run_colmap(image_paths)
        
        # Step 2: Train 2DGS on real images
        print("[ArkRunr/Premium] Step 2: Training 2DGS on real images...")
        gaussians_2d = self._train_2dgs_colmap(colmap_output)
        
        # Step 3: Extract mesh
        print("[ArkRunr/Premium] Step 3: Extracting mesh...")
        mesh_path = self._extract_mesh_tsdf(gaussians_2d, colmap_output.poses)
        
        # Step 4: Bake textures
        print("[ArkRunr/Premium] Step 4: Baking textures...")
        texture_path = self._bake_texture(gaussians_2d, mesh_path, colmap_output.poses)
        
        # Step 5: Unity post-processing
        print("[ArkRunr/Premium] Step 5: Unity post-processing...")
        unity_mesh = self._unity_postprocess(mesh_path, stage_name, texture_path)
        
        return {
            "stage_name": stage_name,
            "quality": "premium",
            "glb_path": unity_mesh,
            "ply_path": self._save_2dgs_ply(gaussians_2d, stage_name),
            "texture_path": texture_path,
            "num_input_images": len(image_paths),
        }
    
    # =========================================================================
    # UNITY POST-PROCESSING
    # =========================================================================
    
    def _unity_postprocess(
        self,
        mesh_path: str,
        stage_name: str,
        texture_path: Optional[str] = None,
    ) -> str:
        """
        Post-process mesh for Unity import.
        
        - Decimate to target triangle count
        - Convert to Y-up coordinate system
        - Apply metric scale
        - UV unwrap if needed
        - Export as GLB
        """
        import trimesh
        
        # Load mesh
        mesh = trimesh.load(mesh_path)
        
        # Decimate if needed
        if len(mesh.faces) > self.config.target_triangles:
            print(f"[ArkRunr/Unity] Decimating from {len(mesh.faces)} to {self.config.target_triangles} triangles")
            mesh = mesh.simplify_quadric_decimation(self.config.target_triangles)
        
        # Convert coordinate system (Z-up to Y-up for Unity)
        if self.config.up_axis == "Y":
            # Rotate -90° around X axis
            rotation = trimesh.transformations.rotation_matrix(
                -np.pi / 2, [1, 0, 0]
            )
            mesh.apply_transform(rotation)
        
        # Apply scale
        mesh.apply_scale(self.config.scale_factor)
        
        # Compute vertex normals
        mesh.fix_normals()
        
        # Export as GLB
        output_path = self.output_dir / f"{stage_name}.glb"
        mesh.export(str(output_path), file_type="glb")
        
        print(f"[ArkRunr/Unity] Exported: {output_path}")
        print(f"[ArkRunr/Unity] Triangles: {len(mesh.faces)}")
        print(f"[ArkRunr/Unity] Vertices: {len(mesh.vertices)}")
        
        # Create metadata
        metadata = {
            "stage_name": stage_name,
            "triangles": len(mesh.faces),
            "vertices": len(mesh.vertices),
            "bounds": mesh.bounds.tolist(),
            "center": mesh.centroid.tolist(),
            "scale": self.config.scale_factor,
            "up_axis": self.config.up_axis,
            "texture": texture_path,
        }
        
        metadata_path = self.output_dir / f"{stage_name}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        return str(output_path)
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _load_sv3d(self):
        """Load SV3D model for view generation."""
        from diffusers import StableVideo3DPipeline
        import torch
        
        pipe = StableVideo3DPipeline.from_pretrained(
            "stabilityai/sv3d",
            torch_dtype=torch.float16,
            variant="fp16",
        )
        pipe.to("cuda")
        pipe.enable_model_cpu_offload()
        
        return pipe
    
    def _azimuth_elevation_to_pose(
        self,
        azimuth: float,
        elevation: float,
        distance: float = 5.0,
    ):
        """
        Convert azimuth/elevation angles to camera pose matrix.
        
        Args:
            azimuth: Horizontal angle in degrees
            elevation: Vertical angle in degrees
            distance: Distance from origin
        
        Returns:
            4x4 camera pose matrix (camera-to-world)
        """
        import numpy as np
        
        # Convert to radians
        az_rad = np.radians(azimuth)
        el_rad = np.radians(elevation)
        
        # Camera position on sphere
        x = distance * np.cos(el_rad) * np.sin(az_rad)
        y = distance * np.sin(el_rad)
        z = distance * np.cos(el_rad) * np.cos(az_rad)
        
        position = np.array([x, y, z])
        
        # Look at origin
        forward = -position / np.linalg.norm(position)
        
        # Up vector
        up = np.array([0, 1, 0])
        
        # Right vector
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        
        # Recompute up
        up = np.cross(right, forward)
        
        # Build rotation matrix
        R = np.stack([right, up, -forward], axis=1)
        
        # Build 4x4 pose matrix
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = position
        
        return pose


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def generate_arkrunr_stage(
    image_path: str,
    quality: str = "quality",
    stage_name: str = "stage",
    output_dir: str = "/srv/searidge_share/outputs/arkrunr_stages",
) -> Dict[str, Any]:
    """
    Convenience function for generating ArkRunr stages.
    
    Args:
        image_path: Path to input interior image
        quality: "quick", "quality", or "premium"
        stage_name: Name for output files
        output_dir: Output directory
    
    Returns:
        Dict with paths to generated assets
    """
    generator = ArkRunrStageGenerator(output_dir=output_dir)
    
    quality_tier = QualityTier(quality.lower())
    
    return generator.generate(
        image_path=image_path,
        quality=quality_tier,
        stage_name=stage_name,
    )
```

---

## Front-Arc View Generation

### The Key Innovation

Instead of generating 360° orbital views (which don't make sense for stages viewed from the audience), we generate views only in the **audience-facing arc**. This is critical for ArkRunr because:

1. **Camera never goes behind the stage** - audience doesn't sit there
2. **Depth must be accurate within the arc** - parallax is visible
3. **Multi-elevation is essential** - floor-level to balcony shots
4. **Distance variation matters** - wide shots vs close-ups

### View Distribution for Stage Volumes

```
                        STAGE (Back Wall)
                              │
                              │
              ┌───────────────┴───────────────┐
              │                               │
              │        PERFORMER AREA         │
              │                               │
              └───────────────┬───────────────┘
                              │
    ════════════════════════════════════════════════
                              │
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        │  ● High Wide        │         ● High Wide │
        │    (el: 25°)        │           (el: 25°) │
        │                     │                     │
        │        ●────────────●────────────●        │
        │      Left 60°    Center     Right 60°    │
        │      (el: 8°)    (el: 10°)   (el: 8°)    │
        │                     │                     │
        │                     │                     │
        │    ●────────────────●────────────────●    │
        │  Wide Left       Wide Master      Wide Right
        │  (el: 5°)        (el: 5°)         (el: 5°)
        │                     │                     │
        │         ●───────────●───────────●         │
        │       Low Left    Low Center  Low Right   │
        │       (el: -5°)   (el: -10°)  (el: -5°)  │
        │                     │                     │
        └─────────────────────┴─────────────────────┘
        
                    AUDIENCE / CAMERA AREA
                    
        ◄──────────────── 120° ARC ────────────────►
```

### Why Multiple Distances Matter

For a full 3D volume, we need views at different distances:

| Distance | Purpose | What It Captures |
|----------|---------|-----------------|
| **5-6m** | Close-ups | Performer area detail, stage platform |
| **8-10m** | Medium shots | Full performer + immediate surroundings |
| **12-15m** | Wide shots | Full stage volume, floor to ceiling |

This ensures we have enough information to reconstruct the full depth range.

### SV3D Modification for Front-Arc

```python
# generators/sv3d_front_arc.py

"""
Modified SV3D for front-arc view generation.

Standard SV3D generates 21 frames in a 360° orbit.
We modify it to generate views in a configurable front-facing arc.
"""

import torch
import numpy as np
from PIL import Image
from diffusers import StableVideo3DPipeline
from typing import List, Tuple, Optional


class FrontArcSV3D:
    """
    SV3D wrapper for front-arc view generation.
    
    Generates views in a specified horizontal arc (e.g., -60° to +60°)
    with optional elevation variation.
    """
    
    def __init__(
        self,
        model_id: str = "stabilityai/sv3d",
        device: str = "cuda",
    ):
        self.device = device
        self.pipe = StableVideo3DPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            variant="fp16",
        )
        self.pipe.to(device)
        self.pipe.enable_model_cpu_offload()
    
    def generate_front_arc(
        self,
        image: Image.Image,
        arc_degrees: float = 120.0,
        arc_center: float = 0.0,
        num_views: int = 7,
        elevation: float = 10.0,
        elevation_range: Optional[Tuple[float, float]] = None,
    ) -> Tuple[List[Image.Image], List[np.ndarray]]:
        """
        Generate views in a front-facing arc.
        
        Args:
            image: Input image (PIL)
            arc_degrees: Total arc width in degrees
            arc_center: Center of arc (0 = directly facing)
            num_views: Number of views to generate
            elevation: Camera elevation angle
            elevation_range: Optional (min, max) for elevation variation
        
        Returns:
            Tuple of (list of images, list of camera poses)
        """
        # Calculate azimuth angles
        arc_half = arc_degrees / 2
        azimuths = np.linspace(
            arc_center - arc_half,
            arc_center + arc_half,
            num_views,
        )
        
        # Handle elevation
        if elevation_range:
            elevations = np.linspace(
                elevation_range[0],
                elevation_range[1],
                num_views,
            )
        else:
            elevations = np.full(num_views, elevation)
        
        # Generate views
        views = []
        poses = []
        
        for i, (az, el) in enumerate(zip(azimuths, elevations)):
            print(f"[FrontArcSV3D] Generating view {i+1}/{num_views}: az={az:.1f}°, el={el:.1f}°")
            
            # Generate single view with SV3D
            # Note: SV3D doesn't directly support per-frame camera control
            # We use a workaround with motion bucket and frame selection
            view = self._generate_view_at_angle(image, az, el)
            pose = self._angle_to_pose(az, el)
            
            views.append(view)
            poses.append(pose)
        
        return views, poses
    
    def _generate_view_at_angle(
        self,
        image: Image.Image,
        azimuth: float,
        elevation: float,
    ) -> Image.Image:
        """
        Generate a single view at specified angle.
        
        Uses SV3D's orbital generation and selects the appropriate frame.
        """
        # SV3D generates 21 frames in 360° orbit
        # Map our desired angle to frame index
        
        # Normalize azimuth to 0-360
        az_normalized = (azimuth + 180) % 360
        
        # Calculate frame index (21 frames over 360°)
        frame_idx = int((az_normalized / 360) * 21) % 21
        
        # Generate full orbit
        with torch.no_grad():
            frames = self.pipe(
                image,
                num_frames=21,
                decode_chunk_size=8,
                motion_bucket_id=127,
            ).frames[0]
        
        # Select the frame closest to our desired angle
        return frames[frame_idx]
    
    def _angle_to_pose(
        self,
        azimuth: float,
        elevation: float,
        distance: float = 5.0,
    ) -> np.ndarray:
        """Convert angles to 4x4 camera pose matrix."""
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
        
        # Build pose matrix
        R = np.stack([right, up, -forward], axis=1)
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = position
        
        return pose
    
    def generate_arkrunr_views(
        self,
        image: Image.Image,
    ) -> Tuple[List[Image.Image], List[np.ndarray]]:
        """
        Generate views specifically for ArkRunr stage use case.
        
        Generates 7 views across 120° arc at 3 elevation levels = 21 views total.
        """
        all_views = []
        all_poses = []
        
        # Three elevation levels: seated, standing, balcony
        elevations = [5, 15, 25]
        
        for elevation in elevations:
            views, poses = self.generate_front_arc(
                image,
                arc_degrees=120,
                arc_center=0,
                num_views=7,
                elevation=elevation,
            )
            all_views.extend(views)
            all_poses.extend(poses)
        
        print(f"[FrontArcSV3D] Generated {len(all_views)} total views for ArkRunr")
        
        return all_views, all_poses


# =============================================================================
# ALTERNATIVE: Zero123++ for faster generation
# =============================================================================

class FrontArcZero123:
    """
    Zero123++ wrapper for front-arc view generation.
    
    Faster than SV3D but generates fewer views (6 fixed positions).
    Good for quick previews.
    """
    
    def __init__(self):
        from diffusers import DiffusionPipeline
        
        self.pipe = DiffusionPipeline.from_pretrained(
            "sudo-ai/zero123plus-v1.2",
            torch_dtype=torch.float16,
        )
        self.pipe.to("cuda")
    
    def generate(self, image: Image.Image) -> Tuple[List[Image.Image], List[np.ndarray]]:
        """
        Generate 6 views using Zero123++.
        
        Zero123++ generates views at fixed positions:
        - 3 front-facing views
        - 3 side views
        
        We select the front-facing ones for stage use.
        """
        result = self.pipe(image, num_inference_steps=75)
        
        # Zero123++ outputs 6 images in a grid
        # Extract individual views
        views = self._extract_views(result.images[0])
        
        # Fixed poses for Zero123++
        poses = self._get_zero123_poses()
        
        # Filter to front-facing only
        front_indices = [0, 1, 2]  # Adjust based on Zero123++ layout
        views = [views[i] for i in front_indices]
        poses = [poses[i] for i in front_indices]
        
        return views, poses
```

### Camera Position Presets for ArkRunr

```python
# configs/arkrunr_cameras.py

"""
Standard camera positions for ArkRunr stage viewing.

These represent typical audience and camera positions for
viewing a performer on a stage backdrop.
"""

from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class CameraPreset:
    """A camera position preset."""
    name: str
    azimuth: float      # Horizontal angle from center
    elevation: float    # Vertical angle from horizon
    distance: float     # Distance from stage center
    description: str


# Standard ArkRunr camera positions
ARKRUNR_CAMERA_PRESETS: List[CameraPreset] = [
    # Center positions
    CameraPreset(
        name="center_low",
        azimuth=0, elevation=-5, distance=5,
        description="Low angle center (dramatic)"
    ),
    CameraPreset(
        name="center_eye",
        azimuth=0, elevation=10, distance=5,
        description="Eye level center (standard)"
    ),
    CameraPreset(
        name="center_high",
        azimuth=0, elevation=25, distance=6,
        description="High angle center (balcony)"
    ),
    
    # Left positions
    CameraPreset(
        name="left_30",
        azimuth=-30, elevation=10, distance=5,
        description="30° left of center"
    ),
    CameraPreset(
        name="left_60",
        azimuth=-60, elevation=10, distance=5.5,
        description="60° left (wide shot)"
    ),
    
    # Right positions
    CameraPreset(
        name="right_30",
        azimuth=30, elevation=10, distance=5,
        description="30° right of center"
    ),
    CameraPreset(
        name="right_60",
        azimuth=60, elevation=10, distance=5.5,
        description="60° right (wide shot)"
    ),
    
    # Special positions
    CameraPreset(
        name="closeup",
        azimuth=0, elevation=5, distance=2,
        description="Close-up shot"
    ),
    CameraPreset(
        name="wide_establishing",
        azimuth=0, elevation=15, distance=8,
        description="Wide establishing shot"
    ),
]


def get_arkrunr_poses() -> List[np.ndarray]:
    """Get all ArkRunr camera poses as 4x4 matrices."""
    poses = []
    for preset in ARKRUNR_CAMERA_PRESETS:
        pose = azimuth_elevation_to_pose(
            preset.azimuth,
            preset.elevation,
            preset.distance,
        )
        poses.append(pose)
    return poses


def get_viewing_arc_poses(
    arc_degrees: float = 120,
    num_positions: int = 7,
    elevation: float = 10,
    distance: float = 5,
) -> List[np.ndarray]:
    """
    Generate evenly-spaced poses across viewing arc.
    
    Args:
        arc_degrees: Total arc width
        num_positions: Number of camera positions
        elevation: Camera elevation
        distance: Distance from stage
    
    Returns:
        List of 4x4 camera pose matrices
    """
    arc_half = arc_degrees / 2
    azimuths = np.linspace(-arc_half, arc_half, num_positions)
    
    poses = []
    for az in azimuths:
        pose = azimuth_elevation_to_pose(az, elevation, distance)
        poses.append(pose)
    
    return poses


def azimuth_elevation_to_pose(
    azimuth: float,
    elevation: float,
    distance: float,
) -> np.ndarray:
    """Convert spherical coordinates to camera pose matrix."""
    az_rad = np.radians(azimuth)
    el_rad = np.radians(elevation)
    
    # Position on sphere looking at origin
    x = distance * np.cos(el_rad) * np.sin(az_rad)
    y = distance * np.sin(el_rad)
    z = distance * np.cos(el_rad) * np.cos(az_rad)
    
    position = np.array([x, y, z])
    
    # Look-at matrix
    forward = -position / np.linalg.norm(position)
    up = np.array([0, 1, 0])
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    
    # Build 4x4 matrix
    R = np.stack([right, up, -forward], axis=1)
    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = position
    
    return pose
```

---

## Unity Integration

### Why Full 3D Volume Matters for Unity

In Unity, the performer exists as a 3D object (video billboard, volumetric capture, or avatar) positioned within the stage volume. For this to look convincing:

```
UNITY SCENE HIERARCHY:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📁 ArkRunr_Stage_Scene
├── 🎭 StageVolume (imported GLB - NEUTRAL LIGHTING)
│   ├── Floor
│   ├── BackWall
│   ├── LeftWall
│   ├── RightWall
│   ├── StagePlatform
│   ├── Columns/Beams
│   └── Stairs/Railings
│
├── 👤 Performer (positioned INSIDE StageVolume)
│   └── VideoPlane / VolumetricCapture / Avatar
│
├── 🎥 CameraRig (user-controlled, arbitrary positions)
│   └── MainCamera (always looking at performer)
│
└── 💡 Lighting (ADDED BY ARKRUNR - not baked into model)
    ├── KeyLight
    ├── FillLight
    ├── RimLight
    └── EffectLights

CRITICAL NOTES:
• Performer.transform.position must be INSIDE the stage volume
• Stage model has NEUTRAL lighting (no baked shadows/highlights)
• ArkRunr adds all lighting dynamically
• Textures can be replaced/modified in ArkRunr or Unity
```

### Parallax Verification in Unity

```csharp
// Test that parallax is working correctly
public class ParallaxVerifier : MonoBehaviour
{
    [Header("Test Settings")]
    public Transform performer;
    public Transform stageVolume;
    public float arcRadius = 8f;
    public float arcDegrees = 120f;
    
    void Update()
    {
        // Move camera in arc
        float angle = Mathf.Sin(Time.time * 0.5f) * (arcDegrees / 2);
        float rad = angle * Mathf.Deg2Rad;
        
        transform.position = new Vector3(
            Mathf.Sin(rad) * arcRadius,
            2f,  // Eye height
            Mathf.Cos(rad) * arcRadius
        );
        
        transform.LookAt(performer);
    }
    
    // WHAT TO CHECK:
    // 1. Foreground objects (speakers) move MORE than background (walls)
    // 2. Performer appears to be IN the space, not floating
    // 3. No obvious depth discontinuities or "cardboard cutout" look
    // 4. Floor meets walls at correct angles
}
```

### Export Requirements

```yaml
# Unity import requirements for ArkRunr stage volumes

format:
  primary: "glb"           # Preferred format
  alternative: "fbx"       # Fallback
  
mesh:
  max_triangles: 200000    # Performance budget
  min_triangles: 50000     # Quality minimum for accurate geometry
  vertex_colors: false     # Use textures or plain materials
  
textures:
  resolution: 2048         # 2K standard (optional - can be replaced)
  format: "png"            # Lossless
  lighting: "NEUTRAL"      # NO baked lighting - critical!
  channels:
    diffuse: optional      # Can be replaced in ArkRunr/Unity
    normal: optional       # For surface detail if needed
    ao: false              # ArkRunr handles ambient occlusion
    
coordinate_system:
  up_axis: "Y"             # Unity convention
  forward_axis: "Z"
  handedness: "left"       # Unity is left-handed
  
scale:
  units: "meters"          # Real-world scale
  performer_height: 1.75   # Reference for scaling
  
volume_requirements:
  geometry_accuracy: "high"  # Accurate to source photograph
  floor_present: true        # Performer must stand on something
  walls_present: true        # As visible in source photo
  ceiling: "if_visible"      # Only if in source photo
  
lighting_requirements:
  baked_lighting: false      # NO baked shadows or highlights
  baked_ao: false            # NO ambient occlusion baked in
  neutral_appearance: true   # Even, neutral surface appearance
  # ArkRunr adds all lighting dynamically
```

### Unity Import Script

```csharp
// Unity/Scripts/ArkRunrStageImporter.cs

using UnityEngine;
using UnityEditor;
using System.IO;

/// <summary>
/// Imports ArkRunr stage backdrops generated from the 2DGS pipeline.
/// </summary>
public class ArkRunrStageImporter : MonoBehaviour
{
    [Header("Import Settings")]
    public string stageName = "stage";
    public float performerHeight = 1.75f;
    
    [Header("Stage Positioning")]
    public Vector3 stageOffset = Vector3.zero;
    public float stageRotation = 0f;
    
    /// <summary>
    /// Import a stage from GLB file.
    /// </summary>
    public void ImportStage(string glbPath)
    {
        // Load the GLB
        GameObject stageObject = LoadGLB(glbPath);
        
        if (stageObject == null)
        {
            Debug.LogError($"Failed to load stage from: {glbPath}");
            return;
        }
        
        // Apply positioning
        stageObject.transform.position = stageOffset;
        stageObject.transform.rotation = Quaternion.Euler(0, stageRotation, 0);
        
        // Verify scale
        VerifyScale(stageObject);
        
        // Add stage marker
        AddPerformerMarker(stageObject);
        
        Debug.Log($"Successfully imported stage: {stageName}");
    }
    
    private GameObject LoadGLB(string path)
    {
        // Use Unity's GLTFast or similar
        // This is a placeholder - actual implementation depends on your GLB loader
        return null;
    }
    
    private void VerifyScale(GameObject stage)
    {
        // Check that the stage is at reasonable scale
        Bounds bounds = GetBounds(stage);
        
        float height = bounds.size.y;
        float width = bounds.size.x;
        float depth = bounds.size.z;
        
        Debug.Log($"Stage bounds: {width:F2}m x {height:F2}m x {depth:F2}m");
        
        // Warn if scale seems off
        if (height < 2f || height > 10f)
        {
            Debug.LogWarning($"Stage height ({height:F2}m) may be incorrect");
        }
    }
    
    private void AddPerformerMarker(GameObject stage)
    {
        // Add a marker where the performer should stand
        GameObject marker = new GameObject("PerformerPosition");
        marker.transform.parent = stage.transform;
        marker.transform.localPosition = new Vector3(0, 0, 1); // 1m in front of backdrop
        
        // Add a visual indicator (editor only)
        #if UNITY_EDITOR
        marker.AddComponent<PerformerPositionGizmo>();
        #endif
    }
    
    private Bounds GetBounds(GameObject obj)
    {
        Bounds bounds = new Bounds(obj.transform.position, Vector3.zero);
        
        foreach (Renderer renderer in obj.GetComponentsInChildren<Renderer>())
        {
            bounds.Encapsulate(renderer.bounds);
        }
        
        return bounds;
    }
}

#if UNITY_EDITOR
/// <summary>
/// Draws a gizmo showing where the performer should stand.
/// </summary>
public class PerformerPositionGizmo : MonoBehaviour
{
    public float performerHeight = 1.75f;
    
    void OnDrawGizmos()
    {
        // Draw performer silhouette
        Gizmos.color = Color.cyan;
        
        Vector3 pos = transform.position;
        
        // Body (cylinder approximation)
        Gizmos.DrawWireCube(pos + Vector3.up * 0.9f, new Vector3(0.5f, 1.4f, 0.3f));
        
        // Head (sphere)
        Gizmos.DrawWireSphere(pos + Vector3.up * 1.6f, 0.15f);
        
        // Ground marker
        Gizmos.color = Color.yellow;
        Gizmos.DrawWireSphere(pos, 0.1f);
    }
}
#endif
```

### Texture Baking for Unity

```python
def bake_texture_for_unity(
    gaussians,
    mesh_path: str,
    poses: List[np.ndarray],
    resolution: int = 2048,
) -> str:
    """
    Bake Gaussian appearance into UV-mapped texture for Unity.
    
    Args:
        gaussians: Trained 2DGS representation
        mesh_path: Path to extracted mesh
        poses: Camera poses used for training
        resolution: Texture resolution
    
    Returns:
        Path to baked texture
    """
    import trimesh
    import xatlas  # For UV unwrapping
    
    # Load mesh
    mesh = trimesh.load(mesh_path)
    
    # UV unwrap if needed
    if not hasattr(mesh.visual, 'uv') or mesh.visual.uv is None:
        print("[TextureBake] Generating UV coordinates...")
        vmapping, indices, uvs = xatlas.parametrize(
            mesh.vertices,
            mesh.faces,
        )
        mesh = trimesh.Trimesh(
            vertices=mesh.vertices[vmapping],
            faces=indices,
            visual=trimesh.visual.TextureVisuals(uv=uvs),
        )
    
    # Create texture atlas
    texture = np.zeros((resolution, resolution, 3), dtype=np.uint8)
    
    # For each face, render from best view and project to UV
    for face_idx in range(len(mesh.faces)):
        # Find best view for this face
        face_center = mesh.vertices[mesh.faces[face_idx]].mean(axis=0)
        face_normal = mesh.face_normals[face_idx]
        
        best_view_idx = find_best_view(face_center, face_normal, poses)
        
        # Render from best view
        rendered = render_2dgs(gaussians, poses[best_view_idx])
        
        # Project face to UV and sample
        uv_coords = mesh.visual.uv[mesh.faces[face_idx]]
        sample_and_fill(texture, rendered, uv_coords)
    
    # Save texture
    texture_path = mesh_path.replace('.ply', '_diffuse.png')
    Image.fromarray(texture).save(texture_path)
    
    return texture_path
```

---

## Quality Tiers

### Tier Comparison for Stage Volumes

| Aspect | Quick | Quality | Premium |
|--------|-------|---------|---------|
| **Input** | Single image | Single image | 3-5 photos |
| **Method** | Lyra → SuGaR | Multi-View 2DGS | COLMAP → 2DGS |
| **Time** | 2-3 minutes | 15-30 minutes | 45+ minutes |
| **Depth Accuracy** | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Parallax Quality** | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Volume Fidelity** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Texture Detail** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Structural Elements** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Best For** | Previews, testing | Production stages | Hero stages |

### Critical Differences for Stage Volumes

| Quality | Depth Range | Parallax | Structural Detail |
|---------|-------------|----------|-------------------|
| **Quick** | ~5m effective | Limited | Scaffolding may be blobby |
| **Quality** | ~10m effective | Good within arc | Scaffolding recognizable |
| **Premium** | ~15m+ accurate | Excellent | Scaffolding sharp/detailed |

### When to Use Each Tier

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    QUALITY TIER DECISION TREE (Stage Volumes)               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Do you have multiple photos of the stage?                                  │
│  │                                                                          │
│  ├── YES (3+ photos from different angles)                                  │
│  │   │                                                                      │
│  │   └── Do photos cover the 120° viewing arc?                             │
│  │       │                                                                  │
│  │       ├── YES ────────────────────────────► PREMIUM                      │
│  │       │                                      Best depth + parallax       │
│  │       │                                                                  │
│  │       └── NO (clustered angles) ──────────► QUALITY                      │
│  │                                              Use best single image       │
│  │                                                                          │
│  └── NO (single image)                                                      │
│      │                                                                      │
│      └── How important is parallax quality?                                 │
│          │                                                                  │
│          ├── CRITICAL (hero stage, close-ups) ► QUALITY                     │
│          │                                       Worth the extra time       │
│          │                                                                  │
│          ├── IMPORTANT (production use) ────► QUALITY                       │
│          │                                     Standard recommendation      │
│          │                                                                  │
│          └── NOT CRITICAL (previews, testing)► QUICK                        │
│                                                 Fast iteration              │
│                                                                              │
│  SPECIAL CASES:                                                             │
│  ─────────────                                                              │
│  • Complex scaffolding/trusses ──────────────► PREMIUM (if possible)        │
│  • Simple walls + floor ─────────────────────► QUALITY                      │
│  • Just need rough preview ──────────────────► QUICK                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Depth Layer Handling by Tier

```
QUICK (Lyra → SuGaR):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Lyra generates 360° orbital video (not ideal for interiors)
• SuGaR extracts mesh with limited depth accuracy
• Depth layers may be compressed (speakers/stage/wall blend together)
• Parallax will be approximate but recognizable

QUALITY (Multi-View 2DGS):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• SV3D generates views within 120° arc
• 2DGS trained with depth supervision
• Depth layers preserved (clear speaker → stage → wall separation)
• Parallax accurate within viewing arc

PREMIUM (COLMAP → 2DGS):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Real photos provide ground truth depth
• COLMAP gives accurate camera poses
• 2DGS trained on real multi-view data
• Depth layers perfectly preserved
• Structural elements (scaffolding, trusses) accurately reconstructed
```

---

## Implementation Code

### Complete Gradio UI Tab

```python
# ui/tabs/arkrunr_stage_tab.py

"""
ArkRunr Stage Reconstruction Tab for Gradio UI

Provides interface for generating Unity-ready stage backdrops
from architectural interior images.
"""

import gradio as gr
from pathlib import Path
from typing import Dict, Any, Optional, Tuple


def create_arkrunr_stage_tab() -> Dict[str, Any]:
    """Create the ArkRunr Stage tab."""
    
    with gr.Column():
        gr.Markdown("""
        ## ArkRunr Stage Reconstruction
        
        Generate Unity-ready 3D stage backdrops from architectural interior images.
        
        **Use Case:** Create stages for placing real-world performers in Unity.
        
        **Viewing Arc:** ~120° (audience perspective)
        """)
        
        # Input section
        with gr.Row():
            with gr.Column(scale=2):
                input_image = gr.Image(
                    label="Interior Image",
                    type="filepath",
                    height=400,
                )
            
            with gr.Column(scale=1):
                gr.Markdown("""
                ### Input Guidelines
                
                ✅ **Good inputs:**
                - Clear, well-lit interior
                - Visible walls/floor
                - Minimal obstructions
                
                ❌ **Avoid:**
                - Extreme wide-angle
                - Heavy lens distortion
                - Dark/underexposed
                """)
        
        # Quality tier selection
        with gr.Row():
            quality_tier = gr.Radio(
                choices=[
                    "Quick (2-3 min) - Lyra + SuGaR",
                    "Quality (10-15 min) - Front-Arc 2DGS",
                    "Premium (30+ min) - Multi-Photo",
                ],
                value="Quality (10-15 min) - Front-Arc 2DGS",
                label="Quality Tier",
            )
        
        # Stage name
        with gr.Row():
            stage_name = gr.Textbox(
                value="stage_001",
                label="Stage Name",
                placeholder="Enter a name for this stage",
            )
        
        # Advanced options
        with gr.Accordion("Advanced Options", open=False):
            with gr.Row():
                arc_degrees = gr.Slider(
                    minimum=60,
                    maximum=180,
                    value=120,
                    step=10,
                    label="Viewing Arc (degrees)",
                    info="Horizontal arc for generated views",
                )
                num_views = gr.Slider(
                    minimum=5,
                    maximum=21,
                    value=7,
                    step=2,
                    label="Views per Elevation",
                    info="More views = better quality, slower",
                )
            
            with gr.Row():
                target_triangles = gr.Slider(
                    minimum=10000,
                    maximum=500000,
                    value=100000,
                    step=10000,
                    label="Target Triangles",
                    info="Mesh polygon budget for Unity",
                )
                texture_resolution = gr.Dropdown(
                    choices=["1024", "2048", "4096"],
                    value="2048",
                    label="Texture Resolution",
                )
            
            with gr.Row():
                iterations = gr.Slider(
                    minimum=5000,
                    maximum=50000,
                    value=15000,
                    step=5000,
                    label="2DGS Iterations",
                    info="More iterations = better quality",
                )
                depth_weight = gr.Slider(
                    minimum=0.0,
                    maximum=2.0,
                    value=1.0,
                    step=0.1,
                    label="Depth Supervision",
                )
        
        # Multi-photo upload (for Premium tier)
        with gr.Accordion("Multi-Photo Upload (Premium Tier)", open=False):
            gr.Markdown("""
            For Premium quality, upload 3-5 photos of the same interior
            from different angles within the viewing arc.
            """)
            multi_photos = gr.File(
                label="Additional Photos",
                file_count="multiple",
                file_types=["image"],
            )
        
        # Generate button
        with gr.Row():
            generate_btn = gr.Button(
                "Generate Stage",
                variant="primary",
                size="lg",
            )
            cancel_btn = gr.Button(
                "Cancel",
                variant="secondary",
            )
        
        # Progress
        progress_display = gr.Textbox(
            label="Progress",
            lines=8,
            interactive=False,
        )
        
        # Outputs
        gr.Markdown("### Generated Assets")
        
        with gr.Row():
            output_glb = gr.File(
                label="Unity Stage (GLB)",
            )
            output_texture = gr.File(
                label="Texture (PNG)",
            )
        
        with gr.Row():
            output_ply = gr.File(
                label="2DGS PLY (optional)",
            )
            output_metadata = gr.File(
                label="Metadata (JSON)",
            )
        
        # 3D Preview
        model_viewer = gr.Model3D(
            label="Stage Preview",
            clear_color=[0.2, 0.2, 0.2, 1.0],
            height=500,
        )
        
        # Camera position preview
        with gr.Accordion("Camera Positions", open=False):
            gr.Markdown("""
            ### ArkRunr Camera Presets
            
            These are the standard camera positions for viewing performers:
            
            | Position | Azimuth | Elevation | Use Case |
            |----------|---------|-----------|----------|
            | Center Low | 0° | -5° | Dramatic shots |
            | Center Eye | 0° | 10° | Standard view |
            | Center High | 0° | 25° | Balcony view |
            | Left 30 | -30° | 10° | Side angle |
            | Right 30 | 30° | 10° | Side angle |
            | Left 60 | -60° | 10° | Wide shot |
            | Right 60 | 60° | 10° | Wide shot |
            """)
    
    return {
        "input_image": input_image,
        "quality_tier": quality_tier,
        "stage_name": stage_name,
        "arc_degrees": arc_degrees,
        "num_views": num_views,
        "target_triangles": target_triangles,
        "texture_resolution": texture_resolution,
        "iterations": iterations,
        "depth_weight": depth_weight,
        "multi_photos": multi_photos,
        "generate_btn": generate_btn,
        "cancel_btn": cancel_btn,
        "progress_display": progress_display,
        "output_glb": output_glb,
        "output_texture": output_texture,
        "output_ply": output_ply,
        "output_metadata": output_metadata,
        "model_viewer": model_viewer,
    }


def handle_stage_generation(
    input_image: str,
    quality_tier: str,
    stage_name: str,
    arc_degrees: float,
    num_views: int,
    target_triangles: int,
    texture_resolution: str,
    iterations: int,
    depth_weight: float,
    multi_photos: Optional[list],
) -> Tuple[str, str, str, str, str, str]:
    """
    Handle stage generation request.
    
    Returns:
        Tuple of (glb_path, texture_path, ply_path, metadata_path, model_path, progress)
    """
    from generators.arkrunr_stage import ArkRunrStageGenerator, StageConfig, QualityTier
    
    # Parse quality tier
    if "Quick" in quality_tier:
        quality = QualityTier.QUICK
    elif "Premium" in quality_tier:
        quality = QualityTier.PREMIUM
    else:
        quality = QualityTier.QUALITY
    
    # Create config
    config = StageConfig(
        arc_degrees=arc_degrees,
        num_azimuth_views=num_views,
        iterations=iterations,
        depth_weight=depth_weight,
        target_triangles=int(target_triangles),
        texture_resolution=int(texture_resolution),
    )
    
    # Create generator
    generator = ArkRunrStageGenerator(config=config)
    
    # Determine input
    if quality == QualityTier.PREMIUM and multi_photos:
        # Use multi-photo input
        import tempfile
        import shutil
        
        temp_dir = tempfile.mkdtemp()
        for i, photo in enumerate(multi_photos):
            shutil.copy(photo.name, f"{temp_dir}/photo_{i:03d}.jpg")
        
        input_path = temp_dir
    else:
        input_path = input_image
    
    # Generate
    try:
        result = generator.generate(
            image_path=input_path,
            quality=quality,
            stage_name=stage_name,
        )
        
        progress = f"✅ Stage generated successfully!\n\n"
        progress += f"Quality: {result['quality']}\n"
        progress += f"GLB: {result['glb_path']}\n"
        
        return (
            result.get("glb_path", ""),
            result.get("texture_path", ""),
            result.get("ply_path", ""),
            result.get("glb_path", "").replace(".glb", "_metadata.json"),
            result.get("glb_path", ""),  # For model viewer
            progress,
        )
        
    except Exception as e:
        progress = f"❌ Error: {str(e)}"
        return "", "", "", "", "", progress
```

---

## Production Workflow

### End-to-End Workflow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ARKRUNR PRODUCTION WORKFLOW                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. CONTENT ACQUISITION                                                     │
│     ─────────────────────                                                   │
│     • Source interior images (stock, generated, or captured)                │
│     • Ensure good lighting and resolution                                   │
│     • Remove any existing people from images                                │
│                                                                              │
│  2. STAGE GENERATION                                                        │
│     ─────────────────────                                                   │
│     • Upload to Gradio UI or use CLI                                        │
│     • Select quality tier based on use case                                 │
│     • Generate Unity-ready GLB                                              │
│                                                                              │
│  3. QUALITY REVIEW                                                          │
│     ─────────────────────                                                   │
│     • Preview in 3D viewer                                                  │
│     • Check geometry at all camera angles                                   │
│     • Verify texture quality                                                │
│     • Approve or regenerate                                                 │
│                                                                              │
│  4. UNITY IMPORT                                                            │
│     ─────────────────────                                                   │
│     • Import GLB into Unity project                                         │
│     • Position stage in scene                                               │
│     • Set up performer placement marker                                     │
│     • Configure lighting                                                    │
│                                                                              │
│  5. PERFORMER INTEGRATION                                                   │
│     ─────────────────────                                                   │
│     • Place performer model/video on stage                                  │
│     • Adjust scale to match stage                                           │
│     • Set up camera system for viewing arc                                  │
│     • Test all camera angles                                                │
│                                                                              │
│  6. DEPLOYMENT                                                              │
│     ─────────────────────                                                   │
│     • Build for target platform                                             │
│     • Optimize for performance                                              │
│     • Deploy to ArkRunr platform                                            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### CLI Usage

```bash
# Quick generation
python -m generators.arkrunr_stage \
    --input interior.jpg \
    --quality quick \
    --name "concert_hall"

# Quality generation with custom arc
python -m generators.arkrunr_stage \
    --input interior.jpg \
    --quality quality \
    --name "theater_stage" \
    --arc-degrees 120 \
    --num-views 9 \
    --iterations 20000

# Premium generation with multiple photos
python -m generators.arkrunr_stage \
    --input ./photos/ \
    --quality premium \
    --name "hero_stage" \
    --target-triangles 200000 \
    --texture-resolution 4096
```

### Batch Processing

```python
# scripts/batch_stage_generation.py

"""
Batch generate ArkRunr stages from a folder of images.
"""

import os
from pathlib import Path
from generators.arkrunr_stage import generate_arkrunr_stage


def batch_generate(
    input_dir: str,
    output_dir: str,
    quality: str = "quality",
):
    """
    Generate stages from all images in a directory.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    image_files = list(input_path.glob("*.jpg")) + list(input_path.glob("*.png"))
    
    print(f"Found {len(image_files)} images to process")
    
    results = []
    
    for i, image_file in enumerate(image_files):
        print(f"\n[{i+1}/{len(image_files)}] Processing: {image_file.name}")
        
        stage_name = image_file.stem
        
        try:
            result = generate_arkrunr_stage(
                image_path=str(image_file),
                quality=quality,
                stage_name=stage_name,
                output_dir=str(output_path),
            )
            results.append({"file": image_file.name, "status": "success", **result})
            
        except Exception as e:
            print(f"Error: {e}")
            results.append({"file": image_file.name, "status": "error", "error": str(e)})
    
    # Summary
    success = sum(1 for r in results if r["status"] == "success")
    print(f"\n{'='*50}")
    print(f"Batch complete: {success}/{len(results)} successful")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--quality", default="quality", choices=["quick", "quality", "premium"])
    
    args = parser.parse_args()
    
    batch_generate(args.input, args.output, args.quality)
```

---

## Implementation Roadmap

> **IMPORTANT:** This section distinguishes between what EXISTS today vs. what needs to be BUILT.
> The code examples throughout this document are **reference/pseudocode**, not working implementations.

### Current State: What Exists Today

| Component | Status | Location | Notes |
|-----------|--------|----------|-------|
| **Lyra (Image → 3DGS)** | ✅ Working | `generators/lyra.py` | RunPod serverless |
| **SHARP (Image → 3DGS)** | ✅ Working | `generators/sharp.py` | RunPod serverless |
| **SuGaR (3DGS → Mesh)** | ✅ Working | `generators/sugar.py` | RunPod serverless |
| **Hunyuan3D (Image → Mesh)** | ✅ Working | `generators/hunyuan.py` | Direct mesh output |
| **PLY Conversion** | ✅ Working | `scripts/convert_lyra_ply.py` | With downsampling |
| **Gradio UI** | ✅ Working | `2d3d.py`, `ui/tabs/` | Multiple tabs |
| **RunPod Infrastructure** | ✅ Working | `runpod/` | Client + handlers |
| **TRELLIS** | ✅ Working | `generators/trellis.py` | Alternative pipeline |
| **ArkRunr-specific wrapper** | ❌ Not built | — | Needs implementation |
| **Multi-view generation** | ❌ Not built | — | Needs implementation |
| **2DGS training** | ❌ Not built | — | Needs implementation |
| **Neutral lighting export** | ❌ Not built | — | Needs implementation |

---

### ⚠️ REAL-WORLD TEST RESULTS (Updated Dec 2024)

Testing has been performed on a representative interior image with complex geometry:
- Multi-level platforms
- Illuminated 3D box structures
- Exposed ceiling infrastructure
- Multiple depth layers
- Consistent textured surfaces

```
ACTUAL TEST RESULTS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Pipeline                    │ Result      │ Quality │ Notes
────────────────────────────┼─────────────┼─────────┼─────────────────────────
GEN3C Video Generation      │ ✅ Good     │ ⭐⭐⭐⭐⭐ │ Clearly understands geometry
Lyra (3DGS) → SuGaR (Mesh)  │ ❌ Unusable │ ⭐      │ GLB barely resembles original
SHARP (3DGS) → SuGaR (Mesh) │ ⚠️ Better   │ ⭐⭐     │ Still not usable
Hunyuan3D (Direct → Mesh)   │ ⭐ Best     │ ⭐⭐⭐    │ Needs cleanup but recognizable

KEY FINDING: The bottleneck is NOT 3DGS generation—it's the 3DGS → Mesh conversion.
             GEN3C videos prove the geometry is understood.
             SuGaR/Poisson reconstruction loses the geometric detail.
```

#### Why 3DGS → Mesh Fails for Interiors

```
THE PROBLEM:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

3D Gaussian Splatting (3DGS):           SuGaR/Poisson Mesh Extraction:
┌─────────────────────────────┐         ┌─────────────────────────────┐
│                             │         │                             │
│   ○ ○ ○ ○ ○ ○ ○ ○ ○ ○       │         │   ╭─────────────────────╮   │
│   ○ ○ ○ ○ ○ ○ ○ ○ ○ ○       │   ──►   │   │                     │   │
│   ○ ○ ○ ○ ○ ○ ○ ○ ○ ○       │         │   │    Blobby mess      │   │
│   (3D ellipsoids in space)  │         │   │    Lost detail       │   │
│                             │         │   ╰─────────────────────╯   │
│   ✅ Captures geometry      │         │   ❌ Loses sharp edges      │
│   ✅ Good for rendering     │         │   ❌ Merges distinct objects│
│   ❌ Not a surface          │         │   ❌ Smooths over detail    │
│                             │         │                             │
└─────────────────────────────┘         └─────────────────────────────┘

The 3D Gaussians are volumetric blobs, not surfaces.
Poisson reconstruction tries to find a surface through them.
For complex interiors, this produces unusable results.
```

#### Why 2DGS Should Help

```
2D GAUSSIAN SPLATTING (2DGS):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Instead of 3D ellipsoids:              2D oriented disks:
┌─────────────────────────────┐        ┌─────────────────────────────┐
│                             │        │                             │
│   ○ ○ ○ ○ ○ ○ ○ ○ ○ ○       │        │   ═ ═ ═ ═ ═ ═ ═ ═ ═ ═       │
│   (volumetric blobs)        │        │   (flat disks ON surfaces)  │
│                             │        │                             │
│   No inherent surface       │        │   ✅ Aligned with surfaces  │
│                             │        │   ✅ Direct TSDF extraction │
│                             │        │   ✅ Sharp edges preserved  │
└─────────────────────────────┘        └─────────────────────────────┘

2DGS primitives ARE surfaces, making mesh extraction much cleaner.
This is why 2DGS is specifically designed for geometry reconstruction.
```

#### Why Hunyuan3D Works Better

Hunyuan3D bypasses the 3DGS → Mesh conversion entirely:

```
Hunyuan3D Pipeline:
┌─────────────┐     ┌─────────────────────────┐     ┌─────────────┐
│   Image     │ ──► │  Direct Mesh Generation │ ──► │    Mesh     │
└─────────────┘     │  (Learned 3D structure) │     └─────────────┘
                    └─────────────────────────┘
                    
                    No intermediate representation to lose detail
```

---

### Revised Strategy Based on Test Results

Given that **Lyra → SuGaR produces unusable results**, the roadmap is adjusted:

```
REVISED DECISION TREE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

                         Current State
                              │
              ┌───────────────┼───────────────┐
              │               │               │
              ▼               ▼               ▼
        Short-Term       Medium-Term      Long-Term
        (Now)            (1-2 months)     (2-4 months)
              │               │               │
              ▼               ▼               ▼
    ┌─────────────────┐ ┌─────────────┐ ┌─────────────┐
    │ Hunyuan3D       │ │ Improve     │ │ 2DGS        │
    │ + Cleanup       │ │ Hunyuan3D   │ │ Pipeline    │
    │                 │ │ + Try other │ │             │
    │ Best current    │ │ direct mesh │ │ Best long-  │
    │ option          │ │ methods     │ │ term option │
    └─────────────────┘ └─────────────┘ └─────────────┘
    
    Skip 3DGS → Mesh path (proven unusable for interiors)
```

---

### Phase 0: COMPLETED — Baseline Already Evaluated

> **STATUS: DONE** — Testing has shown that 3DGS → Mesh pipelines are NOT viable for ArkRunr.

#### Results Summary

| Pipeline | Score | Verdict |
|----------|-------|---------|
| Lyra → SuGaR | ~5/35 | ❌ Unusable |
| SHARP → SuGaR | ~10/35 | ❌ Not usable |
| Hunyuan3D (direct) | ~18/35 | ⚠️ Best current, needs work |

#### Key Finding

**The 3DGS → Mesh conversion is the bottleneck**, not the 3DGS generation itself.
- GEN3C videos prove the model understands the geometry
- SuGaR/Poisson reconstruction destroys the geometric detail
- Direct mesh generation (Hunyuan3D) preserves more structure

#### Decision: Skip to Revised Phases

```
BASELINE COMPLETE — REVISED PATH:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

                    3DGS → Mesh Path
                          │
                          ▼
                    ┌───────────┐
                    │  FAILED   │
                    │  (tested) │
                    └───────────┘
                          │
          ┌───────────────┴───────────────┐
          │                               │
          ▼                               ▼
    ┌───────────────┐              ┌───────────────┐
    │ Phase 1       │              │ Phase 2       │
    │ (Short-term)  │              │ (Long-term)   │
    │               │              │               │
    │ Hunyuan3D     │              │ 2DGS          │
    │ + Improvements│              │ Pipeline      │
    └───────────────┘              └───────────────┘
    
    START HERE                     PARALLEL TRACK
```

---

### Phase 1: Hunyuan3D Optimization (Weeks 1-4)

**Goal:** Maximize quality from the best-performing current pipeline

**Rationale:** Hunyuan3D produces the best results today. Optimize it while building 2DGS in parallel.

#### Tasks

| # | Task | Time | Dependencies | Deliverable |
|---|------|------|--------------|-------------|
| 1.1 | Document Hunyuan3D parameters that affect quality | 2 days | — | Parameter guide |
| 1.2 | Test different Hunyuan3D configurations | 3 days | 1.1 | Best config |
| 1.3 | Create mesh cleanup/repair pipeline | 3 days | — | Cleanup script |
| 1.4 | Add neutral lighting post-process | 2 days | 1.3 | Function |
| 1.5 | Create `generators/arkrunr.py` wrapper | 2 days | 1.2, 1.3 | New file |
| 1.6 | Create `ui/tabs/arkrunr_tab.py` | 3 days | 1.5 | New file |
| 1.7 | Test on diverse interior types | 3 days | All above | Test report |

#### Mesh Cleanup Pipeline

The Hunyuan3D output "needs cleanup" — let's automate that:

```python
# scripts/cleanup_mesh.py
"""
Automated mesh cleanup for Hunyuan3D output.
Addresses common issues found in direct mesh generation.
"""

import trimesh
import numpy as np
from typing import Optional


def cleanup_hunyuan_mesh(
    input_path: str,
    output_path: str,
    target_triangles: int = 150000,
    fix_normals: bool = True,
    remove_small_components: bool = True,
    fill_holes: bool = True,
    smooth_iterations: int = 1,
) -> dict:
    """
    Clean up Hunyuan3D mesh output for ArkRunr use.
    
    Common issues addressed:
    - Inverted/inconsistent normals
    - Small disconnected components (floaters)
    - Holes in surfaces
    - Excessive triangle count
    - Non-manifold geometry
    
    Args:
        input_path: Path to input mesh (GLB/OBJ)
        output_path: Path to save cleaned mesh
        target_triangles: Target triangle count for decimation
        fix_normals: Fix inconsistent normals
        remove_small_components: Remove small disconnected pieces
        fill_holes: Attempt to fill holes in surfaces
        smooth_iterations: Laplacian smoothing passes (0 = none)
    
    Returns:
        Dict with cleanup statistics
    """
    print(f"[Cleanup] Loading: {input_path}")
    mesh = trimesh.load(input_path)
    
    stats = {
        "input_triangles": len(mesh.faces),
        "input_vertices": len(mesh.vertices),
    }
    
    # 1. Remove small disconnected components
    if remove_small_components:
        print("[Cleanup] Removing small components...")
        components = mesh.split(only_watertight=False)
        if len(components) > 1:
            # Keep only components with > 1% of total faces
            min_faces = len(mesh.faces) * 0.01
            large_components = [c for c in components if len(c.faces) > min_faces]
            if large_components:
                mesh = trimesh.util.concatenate(large_components)
                stats["components_removed"] = len(components) - len(large_components)
            else:
                # Keep largest if all are small
                mesh = max(components, key=lambda c: len(c.faces))
                stats["components_removed"] = len(components) - 1
    
    # 2. Fix normals
    if fix_normals:
        print("[Cleanup] Fixing normals...")
        mesh.fix_normals()
    
    # 3. Fill holes (basic)
    if fill_holes:
        print("[Cleanup] Filling holes...")
        # trimesh doesn't have great hole filling, but we can try
        mesh.fill_holes()
    
    # 4. Smooth (gentle)
    if smooth_iterations > 0:
        print(f"[Cleanup] Smoothing ({smooth_iterations} iterations)...")
        trimesh.smoothing.filter_laplacian(mesh, iterations=smooth_iterations)
    
    # 5. Decimate if needed
    if len(mesh.faces) > target_triangles:
        print(f"[Cleanup] Decimating: {len(mesh.faces)} → {target_triangles}")
        mesh = mesh.simplify_quadric_decimation(target_triangles)
    
    # 6. Final stats
    stats["output_triangles"] = len(mesh.faces)
    stats["output_vertices"] = len(mesh.vertices)
    stats["is_watertight"] = mesh.is_watertight
    
    # 7. Export
    print(f"[Cleanup] Saving: {output_path}")
    mesh.export(output_path)
    
    print(f"[Cleanup] Done: {stats['input_triangles']} → {stats['output_triangles']} triangles")
    
    return stats


def neutralize_mesh_lighting(mesh):
    """
    Reduce baked lighting in vertex colors.
    """
    if not hasattr(mesh.visual, 'vertex_colors') or mesh.visual.vertex_colors is None:
        return mesh
    
    colors = mesh.visual.vertex_colors[:, :3].astype(float)
    
    # Flatten luminance variation
    luminance = 0.299 * colors[:, 0] + 0.587 * colors[:, 1] + 0.114 * colors[:, 2]
    mean_lum = luminance.mean()
    
    # Blend toward mean (reduces shadows/highlights)
    blend = 0.5
    adjusted = luminance * (1 - blend) + mean_lum * blend
    scale = np.where(luminance > 0, adjusted / luminance, 1.0)
    
    colors = np.clip(colors * scale[:, np.newaxis], 0, 255)
    mesh.visual.vertex_colors[:, :3] = colors.astype(np.uint8)
    
    return mesh


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean up Hunyuan3D mesh")
    parser.add_argument("input", help="Input mesh file")
    parser.add_argument("output", help="Output mesh file")
    parser.add_argument("--triangles", type=int, default=150000)
    parser.add_argument("--no-smooth", action="store_true")
    
    args = parser.parse_args()
    
    cleanup_hunyuan_mesh(
        args.input,
        args.output,
        target_triangles=args.triangles,
        smooth_iterations=0 if args.no_smooth else 1,
    )
```

#### Hunyuan3D Parameter Investigation

```python
# Investigation: What parameters affect Hunyuan3D quality?

# Current Hunyuan3D call (from generators/hunyuan.py):
# - What resolution is used?
# - What inference steps?
# - Any conditioning options?

# Questions to answer:
# 1. Does higher resolution input help?
# 2. Does image preprocessing (contrast, sharpening) help?
# 3. Are there quality/speed tradeoffs in the model?
# 4. Does the model have different modes (fast vs quality)?

# TODO: Review generators/hunyuan.py and document parameters
```

#### File: `generators/arkrunr.py` (REVISED — Uses Hunyuan3D)

```python
# generators/arkrunr.py
"""
ArkRunr Stage Reconstruction Pipeline

REVISED: Uses Hunyuan3D (direct mesh) instead of Lyra → SuGaR (which failed testing).
Includes mesh cleanup and neutral lighting post-processing.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

from .hunyuan import run_hunyuan_runpod  # Changed from lyra/sugar


@dataclass
class ArkRunrConfig:
    """Configuration for ArkRunr reconstruction."""
    
    # Output settings
    output_dir: str = "/srv/searidge_share/outputs/arkrunr"
    
    # Mesh settings
    target_triangles: int = 150000
    cleanup_mesh: bool = True
    remove_small_components: bool = True
    fill_holes: bool = True
    smooth_iterations: int = 1
    
    # Scale (meters)
    performer_height_reference: float = 1.75
    
    # Neutral lighting
    neutralize_lighting: bool = True
    
    # Quality tier
    quality: str = "hunyuan"  # "hunyuan" (current best), "2dgs" (future)


def run_arkrunr(
    image_path: str,
    config: Optional[ArkRunrConfig] = None,
    stage_name: str = "stage",
) -> Dict[str, Any]:
    """
    ArkRunr reconstruction using Hunyuan3D + cleanup.
    
    This is the CURRENT BEST pipeline based on testing.
    Lyra → SuGaR was tested and found unusable for interiors.
    
    Args:
        image_path: Path to interior photograph
        config: ArkRunr configuration
        stage_name: Name for output files
    
    Returns:
        Dict with paths to generated assets
    """
    config = config or ArkRunrConfig()
    
    output_dir = Path(config.output_dir) / stage_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[ArkRunr] Processing: {image_path}")
    print(f"[ArkRunr] Output: {output_dir}")
    
    # Step 1: Hunyuan3D (Image → Mesh directly)
    print("[ArkRunr] Step 1: Running Hunyuan3D...")
    hunyuan_result = run_hunyuan_runpod(image_path)
    
    raw_mesh_path = hunyuan_result.get("mesh_path") or hunyuan_result.get("glb_path")
    if not raw_mesh_path:
        raise RuntimeError("Hunyuan3D failed to generate mesh")
    
    print(f"[ArkRunr] Hunyuan3D output: {raw_mesh_path}")
    
    # Step 2: Mesh cleanup
    print("[ArkRunr] Step 2: Cleaning up mesh...")
    cleaned_mesh_path = output_dir / f"{stage_name}_cleaned.glb"
    cleanup_stats = cleanup_hunyuan_mesh(
        raw_mesh_path,
        str(cleaned_mesh_path),
        target_triangles=config.target_triangles,
        remove_small_components=config.remove_small_components,
        fill_holes=config.fill_holes,
        smooth_iterations=config.smooth_iterations,
    )
    
    print(f"[ArkRunr] Cleanup: {cleanup_stats}")
    
    # Step 3: Neutral lighting + Unity prep
    print("[ArkRunr] Step 3: Preparing for Unity...")
    final_mesh = prepare_for_unity(
        str(cleaned_mesh_path),
        output_dir / f"{stage_name}.glb",
        config,
    )
    
    print(f"[ArkRunr] Final output: {final_mesh}")
    
    return {
        "stage_name": stage_name,
        "quality": "hunyuan",
        "glb_path": str(final_mesh),
        "raw_mesh_path": raw_mesh_path,
        "cleanup_stats": cleanup_stats,
        "source_image": image_path,
    }


def cleanup_hunyuan_mesh(
    input_path: str,
    output_path: str,
    target_triangles: int = 150000,
    remove_small_components: bool = True,
    fill_holes: bool = True,
    smooth_iterations: int = 1,
) -> dict:
    """Clean up Hunyuan3D mesh output."""
    import trimesh
    import numpy as np
    
    mesh = trimesh.load(input_path)
    stats = {"input_triangles": len(mesh.faces)}
    
    # Remove small components
    if remove_small_components:
        components = mesh.split(only_watertight=False)
        if len(components) > 1:
            min_faces = len(mesh.faces) * 0.01
            large = [c for c in components if len(c.faces) > min_faces]
            mesh = trimesh.util.concatenate(large) if large else max(components, key=lambda c: len(c.faces))
            stats["components_removed"] = len(components) - len(large if large else [mesh])
    
    # Fix normals
    mesh.fix_normals()
    
    # Fill holes
    if fill_holes:
        mesh.fill_holes()
    
    # Smooth
    if smooth_iterations > 0:
        trimesh.smoothing.filter_laplacian(mesh, iterations=smooth_iterations)
    
    # Decimate
    if len(mesh.faces) > target_triangles:
        mesh = mesh.simplify_quadric_decimation(target_triangles)
    
    stats["output_triangles"] = len(mesh.faces)
    mesh.export(output_path)
    
    return stats


def prepare_for_unity(
    input_mesh: str,
    output_path: Path,
    config: ArkRunrConfig,
) -> Path:
    """Prepare mesh for Unity import with neutral lighting."""
    import trimesh
    import numpy as np
    
    mesh = trimesh.load(input_mesh)
    
    # Neutralize lighting
    if config.neutralize_lighting and hasattr(mesh.visual, 'vertex_colors'):
        mesh = neutralize_vertex_colors(mesh)
    
    # Convert to Y-up (Unity convention)
    rotation = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
    mesh.apply_transform(rotation)
    
    mesh.export(str(output_path), file_type="glb")
    return output_path


def neutralize_vertex_colors(mesh):
    """
    Reduce lighting variation in vertex colors.
    
    This helps remove baked shadows/highlights so ArkRunr
    can add its own dynamic lighting.
    """
    import numpy as np
    
    if mesh.visual.vertex_colors is None:
        return mesh
    
    colors = mesh.visual.vertex_colors[:, :3].astype(float)
    
    # Compute per-vertex luminance
    luminance = 0.299 * colors[:, 0] + 0.587 * colors[:, 1] + 0.114 * colors[:, 2]
    
    # Target: flatten luminance variation while preserving hue
    mean_luminance = luminance.mean()
    
    # Blend toward mean (50% blend = partial neutralization)
    blend_factor = 0.5
    adjusted_luminance = luminance * (1 - blend_factor) + mean_luminance * blend_factor
    
    # Scale colors to match adjusted luminance
    scale = np.where(luminance > 0, adjusted_luminance / luminance, 1.0)
    colors = colors * scale[:, np.newaxis]
    colors = np.clip(colors, 0, 255)
    
    mesh.visual.vertex_colors[:, :3] = colors.astype(np.uint8)
    
    return mesh
```

#### File: `ui/tabs/arkrunr_tab.py` (REVISED)

```python
# ui/tabs/arkrunr_tab.py
"""
ArkRunr Stage Reconstruction Tab for Gradio UI
REVISED: Uses Hunyuan3D instead of Lyra+SuGaR
"""

import gradio as gr
from typing import Dict, Any


def create_arkrunr_tab() -> Dict[str, Any]:
    """Create the ArkRunr tab for Gradio UI."""
    
    with gr.Column():
        gr.Markdown("""
        ## ArkRunr Stage Reconstruction
        
        Convert interior photographs into 3D stage volumes for Unity.
        
        **Current Pipeline:** Hunyuan3D → Cleanup → Unity Export
        
        **Key Features:**
        - Accurate architectural geometry (direct mesh generation)
        - Automatic mesh cleanup (removes artifacts)
        - Neutral lighting (ArkRunr adds lighting dynamically)
        - Unity-ready GLB output
        
        **Note:** 3DGS → Mesh pipelines (Lyra/SHARP → SuGaR) were tested 
        and found unsuitable for architectural interiors.
        """)
        
        # Input
        with gr.Row():
            input_image = gr.Image(
                label="Interior Photograph",
                type="filepath",
                height=400,
            )
        
        # Settings
        with gr.Row():
            stage_name = gr.Textbox(
                value="stage",
                label="Stage Name",
                info="Name for output files",
            )
            quality_tier = gr.Radio(
                choices=[
                    "Hunyuan3D (Current Best)",
                    "2DGS (Coming Soon)",
                ],
                value="Hunyuan3D (Current Best)",
                label="Method",
                info="Hunyuan3D is the current best option for interiors",
            )
        
        with gr.Accordion("Mesh Cleanup Settings", open=False):
            target_triangles = gr.Slider(
                minimum=50000,
                maximum=300000,
                value=150000,
                step=10000,
                label="Target Triangles",
            )
            remove_small_components = gr.Checkbox(
                value=True,
                label="Remove Small Components",
                info="Remove floating artifacts",
            )
            fill_holes = gr.Checkbox(
                value=True,
                label="Fill Holes",
                info="Attempt to fill holes in surfaces",
            )
            smooth_iterations = gr.Slider(
                minimum=0,
                maximum=5,
                value=1,
                step=1,
                label="Smoothing Iterations",
                info="0 = no smoothing, higher = smoother",
            )
            neutralize_lighting = gr.Checkbox(
                value=True,
                label="Neutralize Lighting",
                info="Reduce baked shadows/highlights",
            )
        
        # Generate
        generate_btn = gr.Button("Generate Stage", variant="primary", size="lg")
        
        # Progress
        progress = gr.Textbox(label="Progress", lines=5, interactive=False)
        
        # Output
        gr.Markdown("### Output")
        with gr.Row():
            output_glb = gr.File(label="GLB File")
            output_preview = gr.Model3D(label="Preview", height=400)
        
        # Cleanup stats
        cleanup_stats = gr.JSON(label="Cleanup Statistics", visible=True)
    
    return {
        "input_image": input_image,
        "stage_name": stage_name,
        "quality_tier": quality_tier,
        "target_triangles": target_triangles,
        "remove_small_components": remove_small_components,
        "fill_holes": fill_holes,
        "smooth_iterations": smooth_iterations,
        "neutralize_lighting": neutralize_lighting,
        "generate_btn": generate_btn,
        "progress": progress,
        "output_glb": output_glb,
        "output_preview": output_preview,
        "cleanup_stats": cleanup_stats,
    }
```

---

### Phase 2: 2DGS Implementation (Weeks 3-10) — PARALLEL TRACK

**Goal:** Build 2DGS pipeline for best geometry accuracy

**Rationale:** 2DGS is designed specifically for geometry reconstruction. The flat Gaussian disks align with surfaces, making mesh extraction much cleaner than 3DGS → Poisson.

**Run in parallel with Phase 1** — start immediately, don't wait.

#### Tasks

| # | Task | Time | Dependencies | Deliverable |
|---|------|------|--------------|-------------|
| 3.1 | Set up 2DGS training environment | 1 week | — | Docker/environment |
| 3.2 | Implement multi-view generation | 1 week | 3.1 | `generators/multiview.py` |
| 3.3 | Implement 2DGS training wrapper | 1 week | 3.1, 3.2 | `generators/train_2dgs.py` |
| 3.4 | Implement ArkRunr loss functions | 1 week | 3.3 | Loss module |
| 3.5 | Implement TSDF mesh extraction | 3 days | 3.3 | Extraction function |
| 3.6 | Integration testing | 1 week | All above | Test report |
| 3.7 | RunPod deployment | 1 week | 3.6 | Serverless handler |

#### Multi-View Generation Options

```
MULTI-VIEW GENERATION OPTIONS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Option A: SV3D (Stability AI)
├── Pros: High quality, good for objects
├── Cons: Designed for 360° orbital, not interiors
└── Adaptation: Select subset of views, modify camera paths

Option B: Zero123++ (Stability AI)
├── Pros: Fast, multiple views
├── Cons: Lower quality than SV3D
└── Adaptation: Good for quick previews

Option C: Depth-Warping
├── Pros: Fast, uses existing depth
├── Cons: Artifacts at occlusions
└── Adaptation: Good for small viewpoint changes

Option D: Video Diffusion (GEN3C-style)
├── Pros: Already integrated via Lyra
├── Cons: 360° orbital not ideal for interiors
└── Adaptation: Use as-is, accept limitations

RECOMMENDATION: Start with Option D (already working), 
               evaluate Option A for quality path
```

#### 2DGS Training Wrapper

```python
# generators/train_2dgs.py
"""
2DGS Training for ArkRunr

Wraps the 2DGS training process with ArkRunr-specific
configuration and loss functions.
"""

def train_2dgs_arkrunr(
    images: List[str],
    poses: List[np.ndarray],
    config: ArkRunrConfig,
) -> str:
    """
    Train 2DGS model on multi-view images.
    
    Args:
        images: List of image paths
        poses: List of 4x4 camera pose matrices
        config: ArkRunr configuration
    
    Returns:
        Path to trained 2DGS checkpoint
    """
    # This would wrap the actual 2DGS training code
    # See 2DGS_PLAN.md for full implementation details
    pass
```

---

### Phase 4: Premium Multi-Photo Path (Weeks 13-16)

**Goal:** Support real multi-photo input for highest accuracy

**Prerequisites:** Phase 3 complete

#### Tasks

| # | Task | Time | Dependencies | Deliverable |
|---|------|------|--------------|-------------|
| 4.1 | Integrate COLMAP for pose estimation | 1 week | Phase 3 | COLMAP wrapper |
| 4.2 | Multi-photo upload UI | 3 days | 4.1 | UI update |
| 4.3 | Automatic image selection | 3 days | 4.1 | Selection algorithm |
| 4.4 | End-to-end testing | 1 week | All above | Test report |

---

### Timeline Summary (REVISED)

```
REVISED IMPLEMENTATION TIMELINE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Week    1    2    3    4    5    6    7    8    9   10   11   12
        │    │    │    │    │    │    │    │    │    │    │    │
Phase 0 ████ (DONE - 3DGS→Mesh path failed testing)
        
Phase 1 ████████████████                                        
        Hunyuan3D + Cleanup + ArkRunr Tab                       
        (Short-term: use what works best today)                 
                                                                
Phase 2      ████████████████████████████████████████████       
             2DGS Pipeline (PARALLEL TRACK)                     
             (Long-term: best geometry quality)                 
                                                                
Phase 3                                          ████████████   
                                                 Premium Path   
                                                 (Multi-photo)  
                                                                
MILESTONES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Week 0:  ✅ Baseline evaluation complete (3DGS→Mesh = FAILED)
Week 2:  ◆ Hunyuan3D + cleanup script working
Week 4:  ◆ ArkRunr Gradio tab integrated
Week 6:  ◆ 2DGS environment set up
Week 10: ◆ 2DGS pipeline operational
Week 12: ◆ Multi-photo path available
```

**Key Change:** Phase 1 and Phase 2 run in PARALLEL:
- Phase 1 gives you a working solution NOW (Hunyuan3D)
- Phase 2 builds the better long-term solution (2DGS)

---

### Resource Requirements (REVISED)

| Phase | Compute | Storage | Dependencies |
|-------|---------|---------|--------------|
| Phase 1 | RunPod (existing) | 20 GB | trimesh, numpy |
| Phase 2 | RunPod (new GPU for training) | 100 GB | 2DGS, PyTorch, CUDA |
| Phase 3 | RunPod (existing) | 50 GB | COLMAP |

---

### Risk Assessment (REVISED)

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| ~~Lyra geometry insufficient~~ | ~~Medium~~ | ~~High~~ | **CONFIRMED** — use Hunyuan3D instead |
| Hunyuan3D cleanup insufficient | Medium | Medium | Manual cleanup, iterate on script |
| 2DGS training complexity | High | Medium | Use existing 2DGS implementations |
| Multi-view quality issues | Medium | Medium | Multiple generation options |
| RunPod costs for 2DGS training | Medium | Low | Optimize training, use spot instances |

---

### Success Criteria (REVISED)

| Phase | Success Criteria |
|-------|------------------|
| Phase 0 | ✅ **DONE** — 3DGS→Mesh path confirmed unsuitable |
| Phase 1 | Hunyuan3D + cleanup produces usable results (score ≥ 22/35) |
| Phase 2 | 2DGS produces score ≥ 28/35 on test images |
| Phase 3 | Multi-photo produces score ≥ 32/35 |

---

### Immediate Next Steps (REVISED)

Based on testing results:

1. **NOW:** Create `scripts/cleanup_mesh.py` for Hunyuan3D output
2. **This Week:** Test cleanup script on your test image
3. **Week 1-2:** Create `generators/arkrunr.py` wrapper
4. **Week 2-3:** Create `ui/tabs/arkrunr_tab.py` Gradio tab
5. **PARALLEL:** Begin 2DGS environment setup

---

## Summary

### Key Takeaways for ArkRunr

1. **Geometry accuracy is #1 priority** - Faithful recreation of the architectural space

2. **Full 3D volume, not flat backdrop** - Performer exists INSIDE the space, moves throughout

3. **Neutral lighting required** - NO baked lighting; ArkRunr adds all lighting dynamically

4. **Textures are optional** - Can be replaced/modified in ArkRunr or Unity

5. **Any interior type** - Industrial, classical, modern, intimate, outdoor covered, etc.

6. **Arbitrary camera angles** - User-defined positions, always looking at stage/performer

7. **2DGS is ideal** - Flat Gaussians align perfectly with architectural planar surfaces

8. **Three quality tiers** - Quick (Lyra), Quality (Multi-View 2DGS), Premium (Multi-Photo COLMAP)

### What Makes This Different from Generic 3D Reconstruction

| Aspect | Generic Room | ArkRunr Stage |
|--------|--------------|---------------|
| **Goal** | Full room model | Accurate architectural reconstruction |
| **Priority** | Visual fidelity | Geometry accuracy |
| **Lighting** | Often baked | Must be NEUTRAL (ArkRunr adds lighting) |
| **Textures** | Important | Optional (can be replaced) |
| **Interior type** | Usually one style | Any architectural interior |
| **Camera** | Fixed or limited | User-defined, arbitrary angles |
| **Performer** | N/A | Moves throughout the 3D volume |
| **Scale** | Relative | Absolute (performer height reference) |

### Recommended Next Steps

1. **Immediate:** Test the Quick path with existing Lyra infrastructure
   - Verify basic workflow works
   - Check geometry accuracy on simple interiors

2. **Week 1-2:** Implement multi-view generation
   - Focus on generating views that capture the full volume
   - Test on diverse interior types (not just industrial)

3. **Week 3-4:** Build full Quality path with 2DGS training
   - Add depth supervision for accurate geometry
   - Test with neutral lighting output
   - Verify in Unity with ArkRunr lighting

4. **Week 5-6:** Add Gradio UI tab and Unity export pipeline
   - Neutral lighting verification
   - Performer placement tools

5. **Ongoing:** 
   - Test on diverse interior types
   - Tune for different architectural styles
   - Optimize geometry accuracy over texture fidelity

### Quality Checklist for ArkRunr Stages

Before deploying a stage, verify:

- [ ] **Geometry is accurate** - Matches source photograph proportions?
- [ ] **Full 3D volume** - Not a flat backdrop?
- [ ] **Floor is present** - Performer can stand on it?
- [ ] **Scale is correct** - 1.75m performer looks right in the space?
- [ ] **Structural elements present** - Columns, stairs, platforms as in source?
- [ ] **Lighting is neutral** - No baked shadows or highlights?
- [ ] **Depth is correct** - Parallax works from different camera angles?
- [ ] **Works with ArkRunr lighting** - Dynamic lights look correct?

---

*This document is specifically tailored for the ArkRunr architectural reconstruction use case. The goal is accurate 3D geometry with neutral lighting, allowing ArkRunr to add dynamic lighting and optional retexturing. For general 2DGS implementation, see `2DGS_PLAN.md`.*

