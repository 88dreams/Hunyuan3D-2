#!/usr/bin/env python3
"""
Convert Lyra's PyTorch Gaussian format to standard PLY format.

Lyra outputs Gaussian splats as PyTorch tensors in a zip archive disguised as .ply.
This script converts them to actual PLY files that can be viewed in:
- MeshLab
- CloudCompare
- Blender
- SuperSplat (web viewer for 3DGS)

Usage:
    # Basic conversion
    python convert_lyra_ply.py input.ply output.ply
    
    # Downsample to 500k points (good for web viewers)
    python convert_lyra_ply.py input.ply output.ply --max-points 500000
    
    # Keep only high-opacity Gaussians (quality filter)
    python convert_lyra_ply.py input.ply output.ply --min-opacity 0.1
    
    # Combined: downsample + filter for web viewing
    python convert_lyra_ply.py input.ply output.ply --max-points 500000 --min-opacity 0.05
    
    # Simple point cloud for MeshLab/Blender
    python convert_lyra_ply.py input.ply output.ply --simple
    
    # Batch convert a directory
    python convert_lyra_ply.py /path/to/lyra/outputs/ --output-dir /path/to/converted/
"""

import argparse
import os
import sys
import struct
import zipfile
import tempfile
import shutil
from pathlib import Path

import numpy as np

# Try to import torch for loading the tensors
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Install with: pip install torch")


def load_lyra_gaussians(input_path: str) -> dict:
    """
    Load Gaussian parameters from Lyra's PyTorch format.
    
    Lyra saves Gaussians as a zip file containing:
    - gaussians_0/data.pkl (metadata)
    - gaussians_0/data/0 (tensor data)
    
    Returns dict with Gaussian parameters.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required to load Lyra Gaussians")
    
    # Check if it's actually a zip file
    if not zipfile.is_zipfile(input_path):
        raise ValueError(f"Input is not a valid Lyra Gaussian file (not a zip): {input_path}")
    
    # Load the PyTorch tensor
    # torch.load can handle the zip format directly
    try:
        gaussians = torch.load(input_path, map_location='cpu', weights_only=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load Lyra Gaussians: {e}")
    
    return gaussians


def gaussians_to_ply_data(gaussians) -> tuple:
    """
    Convert Gaussian tensor to PLY-compatible arrays.
    
    Standard 3DGS PLY format contains per-Gaussian:
    - x, y, z (position)
    - nx, ny, nz (normal, usually 0)
    - f_dc_0, f_dc_1, f_dc_2 (DC spherical harmonics = base color)
    - f_rest_0 ... f_rest_44 (higher-order SH coefficients)
    - opacity
    - scale_0, scale_1, scale_2
    - rot_0, rot_1, rot_2, rot_3 (quaternion)
    
    Returns (positions, colors, opacities, scales, rotations)
    """
    if isinstance(gaussians, torch.Tensor):
        # Convert to float32 numpy
        data = gaussians.float().cpu().numpy()
        
        # Handle batch dimension - Lyra outputs [1, N, features]
        if len(data.shape) == 3:
            print(f"Input shape: {data.shape} (batch, n_gaussians, features)")
            data = data.squeeze(0)  # Remove batch dimension -> [N, features]
        
        n_gaussians = data.shape[0]
        feat_dim = data.shape[1] if len(data.shape) > 1 else 1
        
        print(f"Processing {n_gaussians:,} Gaussians with {feat_dim} features each")
        
        # Try to infer the format based on feature dimension
        if len(data.shape) == 1:
            raise ValueError("Unexpected 1D tensor format")
        
        # Common 3DGS formats:
        # - 59 features: xyz(3) + SH(48) + opacity(1) + scale(3) + rot(4)
        # - 62 features: xyz(3) + SH(48) + opacity(1) + scale(3) + rot(4) + extra(3)
        # - 14 features: xyz(3) + rgb(3) + opacity(1) + scale(3) + rot(4) - LYRA FORMAT
        
        if feat_dim == 14:
            # Lyra format: xyz + rgb + opacity + scale + rot
            print("Detected Lyra 14-feature format")
            positions = data[:, 0:3]
            colors = data[:, 3:6]
            opacity = data[:, 6:7]
            scales = data[:, 7:10]
            rotations = data[:, 10:14]
            sh_rest = None
            
        elif feat_dim >= 59:
            # Full SH format
            print("Detected full SH format")
            positions = data[:, 0:3]
            sh_dc = data[:, 3:6]  # First 3 SH coefficients (DC term = base color)
            sh_rest = data[:, 6:51] if feat_dim >= 51 else None  # 45 more SH coefficients
            opacity = data[:, 51:52] if feat_dim >= 52 else data[:, -8:-7]
            scales = data[:, 52:55] if feat_dim >= 55 else data[:, -7:-4]
            rotations = data[:, 55:59] if feat_dim >= 59 else data[:, -4:]
            
            # Convert SH DC to RGB (simplified - just use DC term)
            # SH DC coefficients need to be converted: color = SH_C0 * dc + 0.5
            SH_C0 = 0.28209479177387814
            colors = sh_dc * SH_C0 + 0.5
            colors = np.clip(colors, 0, 1)
            
        elif feat_dim > 14:
            # Try to parse as xyz + rgb + opacity + scale + rot + extra
            print(f"Detected extended format with {feat_dim} features, using first 14")
            positions = data[:, 0:3]
            colors = data[:, 3:6]
            opacity = data[:, 6:7]
            scales = data[:, 7:10]
            rotations = data[:, 10:14]
            sh_rest = None
            
        else:
            raise ValueError(f"Unknown Gaussian format with {feat_dim} features")
        
        # Normalize colors to 0-1 range if needed
        if colors.max() > 1.0 or colors.min() < 0.0:
            print(f"Normalizing colors (range: {colors.min():.2f} to {colors.max():.2f})")
            colors = np.clip(colors, 0, 1)
        
        # Sigmoid for opacity if needed (raw values might be logits)
        if opacity.min() < -0.1 or opacity.max() > 1.1:
            print(f"Applying sigmoid to opacity (range: {opacity.min():.2f} to {opacity.max():.2f})")
            opacity = 1 / (1 + np.exp(-np.clip(opacity, -20, 20)))
        
        print(f"Position range: [{positions.min():.2f}, {positions.max():.2f}]")
        print(f"Color range: [{colors.min():.2f}, {colors.max():.2f}]")
        print(f"Opacity range: [{opacity.min():.2f}, {opacity.max():.2f}]")
        
        return positions, colors, opacity.squeeze(), scales, rotations, sh_rest
        
    elif isinstance(gaussians, dict):
        # Dictionary format - extract named fields
        positions = gaussians.get('xyz', gaussians.get('positions', gaussians.get('means3D')))
        colors = gaussians.get('rgb', gaussians.get('colors', gaussians.get('features_dc')))
        opacity = gaussians.get('opacity', gaussians.get('opacities'))
        scales = gaussians.get('scales', gaussians.get('scaling'))
        rotations = gaussians.get('rotations', gaussians.get('rotation'))
        
        if positions is None:
            raise ValueError("Could not find position data in Gaussian dict")
        
        # Convert to numpy
        if isinstance(positions, torch.Tensor):
            positions = positions.float().cpu().numpy()
        if colors is not None and isinstance(colors, torch.Tensor):
            colors = colors.float().cpu().numpy()
        if opacity is not None and isinstance(opacity, torch.Tensor):
            opacity = opacity.float().cpu().numpy()
        if scales is not None and isinstance(scales, torch.Tensor):
            scales = scales.float().cpu().numpy()
        if rotations is not None and isinstance(rotations, torch.Tensor):
            rotations = rotations.float().cpu().numpy()
        
        # Handle missing data
        n = positions.shape[0]
        if colors is None:
            colors = np.ones((n, 3)) * 0.5  # Gray
        if opacity is None:
            opacity = np.ones(n)
        if scales is None:
            scales = np.ones((n, 3)) * 0.01
        if rotations is None:
            rotations = np.tile([1, 0, 0, 0], (n, 1))  # Identity quaternion
        
        return positions, colors, opacity.squeeze(), scales, rotations, None
    
    else:
        raise ValueError(f"Unknown Gaussian format: {type(gaussians)}")


def write_ply(output_path: str, positions: np.ndarray, colors: np.ndarray, 
              opacities: np.ndarray, scales: np.ndarray, rotations: np.ndarray,
              sh_rest: np.ndarray = None):
    """
    Write Gaussians to standard PLY format.
    
    This format is compatible with:
    - SuperSplat viewer
    - gsplat library
    - Original 3DGS implementation
    """
    n = positions.shape[0]
    
    # Build header
    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {n}",
        "property float x",
        "property float y",
        "property float z",
        "property float nx",
        "property float ny", 
        "property float nz",
    ]
    
    # Add SH coefficients
    # DC term (base color)
    header_lines.extend([
        "property float f_dc_0",
        "property float f_dc_1",
        "property float f_dc_2",
    ])
    
    # Higher-order SH (if available)
    n_sh_rest = 45 if sh_rest is not None else 0
    for i in range(n_sh_rest):
        header_lines.append(f"property float f_rest_{i}")
    
    # Opacity, scale, rotation
    header_lines.extend([
        "property float opacity",
        "property float scale_0",
        "property float scale_1",
        "property float scale_2",
        "property float rot_0",
        "property float rot_1",
        "property float rot_2",
        "property float rot_3",
        "end_header",
    ])
    
    header = "\n".join(header_lines) + "\n"
    
    # Prepare data
    normals = np.zeros((n, 3), dtype=np.float32)
    
    # Convert colors to SH DC format
    # Inverse of: color = SH_C0 * dc + 0.5
    SH_C0 = 0.28209479177387814
    sh_dc = (colors - 0.5) / SH_C0
    
    # Ensure correct shapes
    positions = positions.astype(np.float32)
    normals = normals.astype(np.float32)
    sh_dc = sh_dc.astype(np.float32)
    opacities = opacities.reshape(-1, 1).astype(np.float32)
    scales = scales.astype(np.float32)
    rotations = rotations.astype(np.float32)
    
    # Write file
    with open(output_path, 'wb') as f:
        f.write(header.encode('ascii'))
        
        for i in range(n):
            # Position
            f.write(struct.pack('<3f', *positions[i]))
            # Normal
            f.write(struct.pack('<3f', *normals[i]))
            # SH DC
            f.write(struct.pack('<3f', *sh_dc[i]))
            # SH rest
            if sh_rest is not None:
                f.write(struct.pack(f'<{n_sh_rest}f', *sh_rest[i]))
            # Opacity
            f.write(struct.pack('<f', opacities[i, 0]))
            # Scale
            f.write(struct.pack('<3f', *scales[i]))
            # Rotation
            f.write(struct.pack('<4f', *rotations[i]))
    
    print(f"Wrote {n} Gaussians to {output_path}")


def write_simple_ply(output_path: str, positions: np.ndarray, colors: np.ndarray):
    """
    Write a simple point cloud PLY (for viewers that don't support 3DGS).
    
    This format works with:
    - MeshLab
    - CloudCompare
    - Blender
    """
    n = positions.shape[0]
    
    # Convert colors to 0-255 range
    colors_uint8 = (np.clip(colors, 0, 1) * 255).astype(np.uint8)
    
    header = f"""ply
format binary_little_endian 1.0
element vertex {n}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
    
    with open(output_path, 'wb') as f:
        f.write(header.encode('ascii'))
        
        for i in range(n):
            f.write(struct.pack('<3f', *positions[i].astype(np.float32)))
            f.write(struct.pack('<3B', *colors_uint8[i]))
    
    print(f"Wrote {n} points to {output_path} (simple point cloud format)")


def downsample_gaussians(positions, colors, opacities, scales, rotations, 
                         max_points: int = None, min_opacity: float = None,
                         sh_rest=None):
    """
    Downsample Gaussians by random sampling and/or opacity filtering.
    
    Args:
        max_points: Maximum number of points to keep (random sample)
        min_opacity: Minimum opacity threshold (filter low-opacity Gaussians)
    
    Returns:
        Filtered arrays
    """
    n_original = len(positions)
    mask = np.ones(n_original, dtype=bool)
    
    # Filter by opacity first (keeps the most visible Gaussians)
    if min_opacity is not None and min_opacity > 0:
        opacity_mask = opacities >= min_opacity
        mask = mask & opacity_mask
        n_after_opacity = mask.sum()
        print(f"Opacity filter (>={min_opacity}): {n_original:,} → {n_after_opacity:,} ({100*n_after_opacity/n_original:.1f}%)")
    
    # Then downsample if still too many
    n_remaining = mask.sum()
    if max_points is not None and n_remaining > max_points:
        # Get indices of remaining points
        remaining_indices = np.where(mask)[0]
        # Randomly sample from remaining
        np.random.seed(42)  # Reproducible
        sampled_indices = np.random.choice(remaining_indices, size=max_points, replace=False)
        # Create new mask
        new_mask = np.zeros(n_original, dtype=bool)
        new_mask[sampled_indices] = True
        mask = new_mask
        print(f"Random downsample: {n_remaining:,} → {max_points:,} ({100*max_points/n_remaining:.1f}%)")
    
    n_final = mask.sum()
    print(f"Final count: {n_final:,} Gaussians ({100*n_final/n_original:.1f}% of original)")
    
    # Apply mask
    positions = positions[mask]
    colors = colors[mask]
    opacities = opacities[mask]
    scales = scales[mask]
    rotations = rotations[mask]
    if sh_rest is not None:
        sh_rest = sh_rest[mask]
    
    return positions, colors, opacities, scales, rotations, sh_rest


def create_geometry_visualization(positions, colors, opacities, color_mode="depth"):
    """
    Create a visualization-friendly version of the point cloud.
    
    Color modes:
    - "depth": Color by Z-depth (blue=near, red=far) - shows depth layers
    - "height": Color by Y-height (blue=low, red=high) - shows floor/ceiling
    - "opacity": Color by opacity (dark=low, bright=high)
    - "original": Keep original colors
    
    Returns:
        Modified colors array for visualization
    """
    n = len(positions)
    
    if color_mode == "depth":
        # Color by Z-depth (assumes Z is depth axis)
        z = positions[:, 2]
        z_norm = (z - z.min()) / (z.max() - z.min() + 1e-8)
        
        # Blue (near) to Red (far) gradient
        vis_colors = np.zeros((n, 3))
        vis_colors[:, 0] = z_norm  # Red increases with depth
        vis_colors[:, 2] = 1 - z_norm  # Blue decreases with depth
        vis_colors[:, 1] = 0.2  # Small green component
        
        print(f"Depth visualization: Z range = [{z.min():.3f}, {z.max():.3f}]")
        
    elif color_mode == "height":
        # Color by Y-height (assumes Y is up)
        y = positions[:, 1]
        y_norm = (y - y.min()) / (y.max() - y.min() + 1e-8)
        
        # Blue (low/floor) to Red (high/ceiling) gradient
        vis_colors = np.zeros((n, 3))
        vis_colors[:, 0] = y_norm  # Red increases with height
        vis_colors[:, 2] = 1 - y_norm  # Blue decreases with height
        vis_colors[:, 1] = 0.2
        
        print(f"Height visualization: Y range = [{y.min():.3f}, {y.max():.3f}]")
        
    elif color_mode == "opacity":
        # Color by opacity (shows which Gaussians are most visible)
        op_norm = (opacities - opacities.min()) / (opacities.max() - opacities.min() + 1e-8)
        
        # Dark (low opacity) to Bright yellow (high opacity)
        vis_colors = np.zeros((n, 3))
        vis_colors[:, 0] = op_norm  # Red
        vis_colors[:, 1] = op_norm  # Green
        vis_colors[:, 2] = 0.1  # Low blue
        
        print(f"Opacity visualization: range = [{opacities.min():.3f}, {opacities.max():.3f}]")
        
    elif color_mode == "xyz":
        # Color by position (X=red, Y=green, Z=blue) - shows 3D structure
        pos_min = positions.min(axis=0)
        pos_max = positions.max(axis=0)
        pos_norm = (positions - pos_min) / (pos_max - pos_min + 1e-8)
        
        vis_colors = pos_norm
        
        print(f"XYZ visualization: ranges X=[{pos_min[0]:.3f}, {pos_max[0]:.3f}], "
              f"Y=[{pos_min[1]:.3f}, {pos_max[1]:.3f}], Z=[{pos_min[2]:.3f}, {pos_max[2]:.3f}]")
        
    else:  # "original"
        vis_colors = colors
        
    return vis_colors


def write_visualization_ply(output_path: str, positions: np.ndarray, colors: np.ndarray,
                            point_size_hint: float = None):
    """
    Write a visualization PLY with clear, sparse points.
    
    This is a simple XYZ+RGB format that works in all viewers:
    - Blender (import as point cloud)
    - MeshLab
    - CloudCompare
    - Open3D
    
    NOTE: Uses binary format for faster loading and better Blender compatibility.
    """
    n = positions.shape[0]
    
    # Convert colors to 0-255 range
    colors_uint8 = (np.clip(colors, 0, 1) * 255).astype(np.uint8)
    
    # Use binary format - better compatibility with Blender vertex colors
    header = f"""ply
format binary_little_endian 1.0
comment Geometry visualization - {n} points
comment Color mode shows depth/structure
comment Open in Blender: Material Preview mode, Base Color = Attribute "Col"
element vertex {n}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
    
    # Write binary format for better Blender compatibility
    with open(output_path, 'wb') as f:
        f.write(header.encode('ascii'))
        
        # Pack data: x, y, z (float32), r, g, b (uint8)
        for i in range(n):
            # Write position as 3 floats
            f.write(struct.pack('<fff', positions[i, 0], positions[i, 1], positions[i, 2]))
            # Write color as 3 bytes
            f.write(struct.pack('<BBB', colors_uint8[i, 0], colors_uint8[i, 1], colors_uint8[i, 2]))
    
    print(f"Wrote visualization PLY: {output_path}")
    print(f"  Points: {n:,}")
    print(f"  Format: Binary (fast loading, Blender compatible)")


def visualize_geometry(input_path: str, output_path: str, 
                       num_points: int = 50000,
                       color_mode: str = "depth",
                       min_opacity: float = 0.0):
    """
    Create a diagnostic visualization of 3DGS geometry.
    
    This aggressively downsamples and recolors the point cloud
    to reveal the underlying 3D structure.
    
    Args:
        input_path: Path to Lyra PLY file
        output_path: Path to write visualization PLY
        num_points: Number of points to keep (default 50k for clear viewing)
        color_mode: How to color points ("depth", "height", "opacity", "xyz", "original")
        min_opacity: Filter out low-opacity Gaussians first (default 0 = no filter)
    """
    print("=" * 60)
    print("GEOMETRY VISUALIZATION MODE")
    print("=" * 60)
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Target points: {num_points:,}")
    print(f"Color mode: {color_mode}")
    print(f"Min opacity filter: {min_opacity}")
    print("=" * 60)
    
    # Load Gaussians
    print("\nLoading Gaussians...")
    gaussians = load_lyra_gaussians(input_path)
    positions, colors, opacities, scales, rotations, sh_rest = gaussians_to_ply_data(gaussians)
    
    original_count = len(positions)
    print(f"\nOriginal: {original_count:,} Gaussians")
    
    # Check opacity range to understand the data
    print(f"\nOpacity range: [{opacities.min():.4f}, {opacities.max():.4f}], mean: {opacities.mean():.4f}")
    
    # Step 1: Filter by opacity (only if threshold is set)
    if min_opacity > 0:
        print(f"\nStep 1: Filtering by opacity >= {min_opacity}...")
        opacity_mask = opacities >= min_opacity
        positions = positions[opacity_mask]
        colors = colors[opacity_mask]
        opacities = opacities[opacity_mask]
        print(f"  After opacity filter: {len(positions):,} ({100*len(positions)/original_count:.1f}%)")
    else:
        print(f"\nStep 1: Skipping opacity filter (keeping all {len(positions):,} points)")
    
    # Step 2: Downsample to target
    if len(positions) > num_points:
        print(f"\nStep 2: Downsampling to {num_points:,} points...")
        np.random.seed(42)
        indices = np.random.choice(len(positions), size=num_points, replace=False)
        positions = positions[indices]
        colors = colors[indices]
        opacities = opacities[indices]
        print(f"  After downsample: {len(positions):,}")
    
    # Step 3: Create visualization colors
    print(f"\nStep 3: Creating {color_mode} visualization...")
    vis_colors = create_geometry_visualization(positions, colors, opacities, color_mode)
    
    # Step 4: Write output
    print(f"\nStep 4: Writing visualization PLY...")
    write_visualization_ply(output_path, positions, vis_colors)
    
    # Print viewing instructions
    print("\n" + "=" * 60)
    print("VIEWING INSTRUCTIONS")
    print("=" * 60)
    print("""
The output file can be viewed in:

1. BLENDER:
   - File → Import → Stanford (.ply)
   - In viewport, press Z → Material Preview to see colors
   - Use scroll wheel to zoom, middle-mouse to rotate

2. MESHLAB:
   - File → Import Mesh
   - Colors should show automatically

3. CLOUDCOMPARE:
   - File → Open
   - Colors should show automatically

4. SUPERSPLAT (web):
   - https://playcanvas.com/supersplat/editor
   - Drag and drop the file

COLOR INTERPRETATION:
""")
    
    if color_mode == "depth":
        print("  BLUE = Near (close to camera)")
        print("  RED  = Far (away from camera)")
        print("  → Look for distinct depth layers (floor, walls, objects)")
    elif color_mode == "height":
        print("  BLUE = Low (floor level)")
        print("  RED  = High (ceiling level)")
        print("  → Look for floor plane and ceiling")
    elif color_mode == "opacity":
        print("  DARK   = Low opacity (less visible)")
        print("  BRIGHT = High opacity (more visible)")
        print("  → Bright areas are the 'solid' parts of the scene")
    elif color_mode == "xyz":
        print("  RED   = X position")
        print("  GREEN = Y position")
        print("  BLUE  = Z position")
        print("  → Shows overall 3D structure")
    
    print("\n" + "=" * 60)
    print("WHAT TO LOOK FOR")
    print("=" * 60)
    print("""
GOOD GEOMETRY (3DGS is working):
  ✓ Clear floor plane (flat layer of one color)
  ✓ Distinct walls (vertical surfaces)
  ✓ Recognizable room shape
  ✓ Objects at correct depths
  ✓ Depth layers match your source image

BAD GEOMETRY (3DGS has problems):
  ✗ Scattered points with no structure
  ✗ Everything at similar depth (flat)
  ✗ No clear floor/walls
  ✗ Blobs instead of surfaces
  ✗ Points don't match source image layout
""")
    
    return output_path


def convert_lyra_ply(input_path: str, output_path: str, simple: bool = False,
                     max_points: int = None, min_opacity: float = None):
    """
    Convert Lyra's PyTorch Gaussian format to standard PLY.
    
    Args:
        input_path: Path to Lyra's .ply file (actually a PyTorch zip)
        output_path: Path to write the converted PLY
        simple: If True, write simple point cloud (for MeshLab/Blender)
                If False, write full 3DGS format (for SuperSplat)
        max_points: Maximum number of points (downsample if exceeded)
        min_opacity: Minimum opacity threshold (filter low-opacity Gaussians)
    """
    print(f"Loading: {input_path}")
    gaussians = load_lyra_gaussians(input_path)
    
    print("Converting to PLY format...")
    positions, colors, opacities, scales, rotations, sh_rest = gaussians_to_ply_data(gaussians)
    
    # Apply downsampling/filtering if requested
    if max_points is not None or min_opacity is not None:
        print("\n--- Applying filters ---")
        positions, colors, opacities, scales, rotations, sh_rest = downsample_gaussians(
            positions, colors, opacities, scales, rotations,
            max_points=max_points, min_opacity=min_opacity, sh_rest=sh_rest
        )
        print()
    
    if simple:
        write_simple_ply(output_path, positions, colors)
    else:
        write_ply(output_path, positions, colors, opacities, scales, rotations, sh_rest)
    
    print(f"Done! Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Lyra's PyTorch Gaussian format to standard PLY",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic conversion (full quality)
  python convert_lyra_ply.py input.ply output.ply

  # For web viewers (SuperSplat) - downsample to 500k
  python convert_lyra_ply.py input.ply output_web.ply --max-points 500000

  # High quality filter - keep only visible Gaussians
  python convert_lyra_ply.py input.ply output.ply --min-opacity 0.1

  # Lightweight for quick preview
  python convert_lyra_ply.py input.ply preview.ply --max-points 100000 --min-opacity 0.05

  # Simple point cloud for MeshLab/Blender
  python convert_lyra_ply.py input.ply output.ply --simple --max-points 500000

  # DIAGNOSTIC: Visualize geometry structure
  python convert_lyra_ply.py input.ply --visualize-geometry
  python convert_lyra_ply.py input.ply --visualize-geometry --color-mode height

Recommended settings by use case:
  - SuperSplat web viewer:  --max-points 500000
  - Local 3DGS viewer:      --max-points 1000000
  - Quick preview:          --max-points 100000 --min-opacity 0.05
  - Full quality:           (no flags)
  - Geometry diagnosis:     --visualize-geometry
"""
    )
    parser.add_argument("input", help="Input .ply file (Lyra format) or directory")
    parser.add_argument("output", nargs="?", help="Output .ply file or directory")
    parser.add_argument("--simple", action="store_true",
                        help="Write simple point cloud (for MeshLab/Blender)")
    parser.add_argument("--output-dir", help="Output directory for batch conversion")
    parser.add_argument("--max-points", type=int, default=None,
                        help="Maximum number of points (downsample if exceeded). "
                             "Recommended: 500000 for web, 1000000 for local")
    parser.add_argument("--min-opacity", type=float, default=None,
                        help="Minimum opacity threshold (0.0-1.0). "
                             "Filters out near-invisible Gaussians. Recommended: 0.05-0.1")
    
    # Visualization options
    parser.add_argument("--visualize-geometry", action="store_true",
                        help="Create a diagnostic visualization to inspect 3DGS geometry. "
                             "Outputs a sparse, color-coded point cloud for easy viewing.")
    parser.add_argument("--color-mode", type=str, default="depth",
                        choices=["depth", "height", "opacity", "xyz", "original"],
                        help="Color mode for --visualize-geometry: "
                             "depth (blue=near, red=far), "
                             "height (blue=floor, red=ceiling), "
                             "opacity (dark=invisible, bright=visible), "
                             "xyz (position-based RGB), "
                             "original (keep source colors). Default: depth")
    parser.add_argument("--vis-points", type=int, default=50000,
                        help="Number of points for visualization (default: 50000)")
    
    args = parser.parse_args()
    
    if not TORCH_AVAILABLE:
        print("Error: PyTorch is required. Install with: pip install torch")
        sys.exit(1)
    
    input_path = Path(args.input)
    
    # Handle visualization mode
    if args.visualize_geometry:
        if not input_path.is_file():
            print(f"Error: Input file not found: {input_path}")
            sys.exit(1)
        
        # Generate output path
        if args.output:
            output_path = args.output
        else:
            output_path = str(input_path.with_stem(input_path.stem + f"_vis_{args.color_mode}")) + ".ply"
        
        visualize_geometry(
            str(input_path),
            output_path,
            num_points=args.vis_points,
            color_mode=args.color_mode,
            min_opacity=args.min_opacity if args.min_opacity else 0.0,  # No opacity filter by default for viz
        )
        return
    
    # Normal conversion mode
    if input_path.is_file():
        # Single file conversion
        if args.output:
            output_path = args.output
        else:
            # Default: add _converted suffix
            suffix = "_converted"
            if args.max_points:
                suffix += f"_{args.max_points//1000}k"
            output_path = str(input_path.with_stem(input_path.stem + suffix))
        
        convert_lyra_ply(str(input_path), output_path, 
                        simple=args.simple,
                        max_points=args.max_points,
                        min_opacity=args.min_opacity)
        
    elif input_path.is_dir():
        # Batch conversion
        output_dir = Path(args.output_dir) if args.output_dir else input_path / "converted"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        ply_files = list(input_path.glob("*.ply"))
        print(f"Found {len(ply_files)} PLY files to convert")
        
        for ply_file in ply_files:
            suffix = "_converted"
            if args.max_points:
                suffix += f"_{args.max_points//1000}k"
            output_path = output_dir / (ply_file.stem + suffix + ".ply")
            try:
                convert_lyra_ply(str(ply_file), str(output_path), 
                               simple=args.simple,
                               max_points=args.max_points,
                               min_opacity=args.min_opacity)
            except Exception as e:
                print(f"Error converting {ply_file}: {e}")
    else:
        print(f"Error: Input not found: {input_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()

