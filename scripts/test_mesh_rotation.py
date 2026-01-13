#!/usr/bin/env python3
"""
Test mesh rotation - Apply 180° X-axis rotation to a GLB/mesh file.

Usage:
    python scripts/test_mesh_rotation.py /path/to/mesh.glb
    
This will create a rotated copy: /path/to/mesh_rotated.glb
You can then compare the original and rotated versions in Blender or the UI.
"""

import sys
import argparse
from pathlib import Path

import trimesh
import numpy as np


def rotate_mesh_180_x(input_path: str, output_path: str = None) -> str:
    """
    Rotate mesh 180 degrees around X axis.
    
    Args:
        input_path: Path to input mesh (GLB, OBJ, PLY, etc.)
        output_path: Path for output. If None, creates *_rotated.ext
        
    Returns:
        Path to the rotated mesh
    """
    input_path = Path(input_path)
    
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_rotated{input_path.suffix}"
    else:
        output_path = Path(output_path)
    
    print(f"Loading mesh: {input_path}")
    mesh = trimesh.load(input_path)
    
    # Get mesh info before rotation
    if hasattr(mesh, 'vertices'):
        print(f"  Vertices: {len(mesh.vertices):,}")
        print(f"  Faces: {len(mesh.faces):,}")
        bounds = mesh.bounds
        print(f"  Bounds: X[{bounds[0][0]:.2f}, {bounds[1][0]:.2f}] "
              f"Y[{bounds[0][1]:.2f}, {bounds[1][1]:.2f}] "
              f"Z[{bounds[0][2]:.2f}, {bounds[1][2]:.2f}]")
    
    # Create 180-degree rotation matrix around X axis
    # cos(180°) = -1, sin(180°) = 0
    # This flips Y and Z while keeping X the same
    rotation_matrix = np.array([
        [ 1,  0,  0,  0],
        [ 0, -1,  0,  0],
        [ 0,  0, -1,  0],
        [ 0,  0,  0,  1]
    ])
    
    print(f"\nApplying 180° X-axis rotation...")
    mesh.apply_transform(rotation_matrix)
    
    # Export
    print(f"Exporting to: {output_path}")
    mesh.export(output_path)
    
    file_size = output_path.stat().st_size / 1024 / 1024
    print(f"✅ Done! Output size: {file_size:.2f} MB")
    
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Rotate a mesh 180° around X axis (to fix 2DGS orientation)"
    )
    parser.add_argument("input", help="Input mesh file (GLB, OBJ, PLY, etc.)")
    parser.add_argument("-o", "--output", help="Output path (default: input_rotated.ext)")
    parser.add_argument("--inplace", action="store_true", 
                        help="Overwrite input file instead of creating new one")
    
    args = parser.parse_args()
    
    if not Path(args.input).exists():
        print(f"❌ File not found: {args.input}")
        sys.exit(1)
    
    output = args.input if args.inplace else args.output
    
    try:
        result = rotate_mesh_180_x(args.input, output)
        print(f"\nTo compare in Blender:")
        print(f"  1. Import original: {args.input}")
        print(f"  2. Import rotated:  {result}")
        print(f"  3. Compare orientation with your reference image/video")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
