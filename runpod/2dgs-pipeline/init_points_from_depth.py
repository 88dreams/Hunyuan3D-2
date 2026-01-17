#!/usr/bin/env python3
"""
Generate initial 3D points for 2DGS from depth maps.

This script:
1. Reads depth maps and camera intrinsics/poses
2. Backprojects pixels to 3D points
3. Writes a PLY file and/or updates points3D.txt

Usage:
    python init_points_from_depth.py --data_dir /path/to/2dgs_data --sample_rate 16
"""

import argparse
from pathlib import Path
import numpy as np

try:
    import cv2
except ImportError:
    import subprocess
    subprocess.run(["pip", "install", "opencv-python"], check=True)
    import cv2


def parse_cameras_txt(cameras_path: str) -> dict:
    """Parse COLMAP cameras.txt file."""
    cameras = {}
    with open(cameras_path, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split()
            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = [float(p) for p in parts[4:]]
            cameras[camera_id] = {
                'model': model,
                'width': width,
                'height': height,
                'params': params,  # fx, fy, cx, cy for PINHOLE
            }
    return cameras


def parse_images_txt(images_path: str) -> list:
    """Parse COLMAP images.txt file."""
    images = []
    with open(images_path, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('#') or not line:
            i += 1
            continue
        
        parts = line.split()
        if len(parts) >= 9:
            image_id = int(parts[0])
            qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
            camera_id = int(parts[8])
            name = parts[9] if len(parts) > 9 else ""
            
            images.append({
                'image_id': image_id,
                'qvec': [qw, qx, qy, qz],
                'tvec': [tx, ty, tz],
                'camera_id': camera_id,
                'name': name,
            })
        i += 1
    
    return images


def qvec_to_rotmat(qvec):
    """Convert quaternion to rotation matrix."""
    qw, qx, qy, qz = qvec
    R = np.array([
        [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy]
    ])
    return R


def backproject_depth(depth, K, R, t, sample_rate=16, max_depth=10.0):
    """
    Backproject depth map to 3D points.
    
    Args:
        depth: (H, W) depth map
        K: (3, 3) intrinsic matrix
        R: (3, 3) rotation matrix (world-to-camera)
        t: (3,) translation vector (world-to-camera)
        sample_rate: Sample every Nth pixel
        max_depth: Maximum valid depth
    
    Returns:
        points: (N, 3) 3D points in world coordinates
        colors: (N, 3) dummy colors (will be 128, 128, 128)
    """
    H, W = depth.shape
    
    # Create pixel grid
    u = np.arange(0, W, sample_rate)
    v = np.arange(0, H, sample_rate)
    u, v = np.meshgrid(u, v)
    u = u.flatten()
    v = v.flatten()
    
    # Get depths at sampled pixels
    d = depth[v.astype(int), u.astype(int)]
    
    # Filter invalid depths
    valid = (d > 0) & (d < max_depth) & np.isfinite(d)
    u = u[valid]
    v = v[valid]
    d = d[valid]
    
    if len(d) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8)
    
    # Backproject to camera coordinates
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    x_cam = (u - cx) * d / fx
    y_cam = (v - cy) * d / fy
    z_cam = d
    
    points_cam = np.stack([x_cam, y_cam, z_cam], axis=1)  # (N, 3)
    
    # Transform to world coordinates
    # camera-to-world: P_world = R^T @ (P_cam - t) = R^T @ P_cam - R^T @ t
    R_inv = R.T
    t_world = -R_inv @ t
    
    points_world = (R_inv @ points_cam.T).T + t_world
    
    # Dummy colors (gray)
    colors = np.full((len(points_world), 3), 128, dtype=np.uint8)
    
    return points_world, colors


def write_ply(filename, points, colors):
    """Write points to PLY file compatible with 2DGS/3DGS (binary format for Open3D compatibility)."""
    try:
        # Try using plyfile for better compatibility
        from plyfile import PlyData, PlyElement
        
        # Create structured array for vertices
        vertex_dtype = [
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
        ]
        
        vertices = np.zeros(len(points), dtype=vertex_dtype)
        vertices['x'] = points[:, 0]
        vertices['y'] = points[:, 1]
        vertices['z'] = points[:, 2]
        vertices['nx'] = 0.0  # Placeholder normals
        vertices['ny'] = 0.0
        vertices['nz'] = 0.0
        vertices['red'] = (colors[:, 0] * 255).astype(np.uint8) if colors.max() <= 1 else colors[:, 0].astype(np.uint8)
        vertices['green'] = (colors[:, 1] * 255).astype(np.uint8) if colors.max() <= 1 else colors[:, 1].astype(np.uint8)
        vertices['blue'] = (colors[:, 2] * 255).astype(np.uint8) if colors.max() <= 1 else colors[:, 2].astype(np.uint8)
        
        el = PlyElement.describe(vertices, 'vertex')
        PlyData([el], text=False).write(str(filename))
        return
    except ImportError:
        pass  # Fall back to ASCII format
    
    # ASCII fallback
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        # 2DGS requires normals (nx, ny, nz) - use zeros as placeholder
        f.write("property float nx\n")
        f.write("property float ny\n")
        f.write("property float nz\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        for i in range(len(points)):
            # x, y, z, nx, ny, nz, r, g, b
            f.write(f"{points[i, 0]} {points[i, 1]} {points[i, 2]} "
                   f"0 0 0 "  # normals (zeros)
                   f"{colors[i, 0]} {colors[i, 1]} {colors[i, 2]}\n")


def write_points3d_txt(filename, points, colors):
    """Write points to COLMAP points3D.txt format."""
    with open(filename, 'w') as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        
        for i in range(len(points)):
            # POINT3D_ID, X, Y, Z, R, G, B, ERROR (no tracks)
            f.write(f"{i + 1} {points[i, 0]} {points[i, 1]} {points[i, 2]} "
                   f"{colors[i, 0]} {colors[i, 1]} {colors[i, 2]} 0.0\n")


def generate_points_from_depth(
    data_dir: str,
    sample_rate: int = 16,
    max_depth: float = 100.0,
    max_points: int = 100000,
    num_frames: int = 10
) -> str:
    """
    Generate initial 3D points from depth maps.
    This is the API function called by the handler.
    
    Args:
        data_dir: Path to 2DGS data directory (with images/, depths/, sparse/0/)
        sample_rate: Sample every Nth pixel from depth maps
        max_depth: Maximum valid depth value
        max_points: Maximum total points to generate
        num_frames: Number of frames to use for point generation
    
    Returns:
        Path to generated points3D.ply
    """
    data_dir = Path(data_dir)
    depths_dir = data_dir / "depths"
    sparse_dir = data_dir / "sparse" / "0"
    
    # Check if depths directory exists
    if not depths_dir.exists():
        print(f"Warning: depths directory not found at {depths_dir}")
        print("Will create minimal point cloud from camera positions")
        
        # Parse camera info to at least get something
        cameras = parse_cameras_txt(sparse_dir / "cameras.txt")
        images = parse_images_txt(sparse_dir / "images.txt")
        
        # Create minimal point cloud from camera positions
        all_points = []
        all_colors = []
        
        for img_info in images:
            R = qvec_to_rotmat(img_info['qvec'])
            t = np.array(img_info['tvec'])
            # Camera position in world coordinates
            cam_pos = -R.T @ t
            all_points.append(cam_pos)
            all_colors.append([128, 128, 128])
        
        all_points = np.array(all_points)
        all_colors = np.array(all_colors, dtype=np.uint8)
        
        # Add some points in front of cameras
        for img_info in images[:10]:
            R = qvec_to_rotmat(img_info['qvec'])
            t = np.array(img_info['tvec'])
            cam_pos = -R.T @ t
            forward = R.T @ np.array([0, 0, 1])
            for dist in [1, 2, 5, 10]:
                pt = cam_pos + dist * forward
                all_points = np.vstack([all_points, pt])
                all_colors = np.vstack([all_colors, [128, 128, 128]])
        
        ply_path = sparse_dir / "points3D.ply"
        write_ply(ply_path, all_points, all_colors)
        print(f"Created minimal point cloud with {len(all_points)} points")
        return str(ply_path)
    
    # Parse camera info
    cameras = parse_cameras_txt(sparse_dir / "cameras.txt")
    images = parse_images_txt(sparse_dir / "images.txt")
    
    print(f"Generating points from {len(images)} images with {len(list(depths_dir.glob('*.npy')))} depth maps")
    
    # Get intrinsic matrix (assume single camera)
    cam = cameras[1]
    fx, fy, cx, cy = cam['params']
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    
    # Collect points from depth maps
    all_points = []
    all_colors = []
    
    # Use evenly spaced frames
    n_images = len(images)
    frame_indices = np.linspace(0, n_images - 1, min(num_frames, n_images), dtype=int)
    
    for idx in frame_indices:
        img_info = images[idx]
        
        # Load depth
        depth_name = Path(img_info['name']).stem + ".npy"
        depth_path = depths_dir / depth_name
        
        if not depth_path.exists():
            print(f"  Skipping {depth_name}: depth not found")
            continue
        
        depth = np.load(depth_path)
        
        # Get pose
        R = qvec_to_rotmat(img_info['qvec'])
        t = np.array(img_info['tvec'])
        
        # Backproject
        points, colors = backproject_depth(
            depth, K, R, t, 
            sample_rate=sample_rate,
            max_depth=max_depth
        )
        
        valid_depths = depth[depth > 0]
        if len(valid_depths) > 0:
            print(f"  Frame {idx}: {len(points)} points (depth range: {valid_depths.min():.2f} - {valid_depths.max():.2f})")
        else:
            print(f"  Frame {idx}: {len(points)} points")
        
        all_points.append(points)
        all_colors.append(colors)
    
    if not all_points:
        # Fallback: create random points in a unit sphere
        print("  [Warning] No depth maps found, using random point initialization")
        n_random = min(max_points, 10000)
        
        # Random points in unit sphere
        theta = np.random.uniform(0, 2*np.pi, n_random)
        phi = np.random.uniform(0, np.pi, n_random)
        r = np.random.uniform(0.3, 1.0, n_random)
        
        all_points = np.column_stack([
            r * np.sin(phi) * np.cos(theta),
            r * np.sin(phi) * np.sin(theta),
            r * np.cos(phi)
        ])
        all_colors = np.random.uniform(0.3, 0.7, (n_random, 3))
        print(f"  Generated {n_random} random points")
    else:
        # Combine all points
        all_points = np.vstack(all_points)
        all_colors = np.vstack(all_colors)
    
    print(f"Total points: {len(all_points)}")
    
    # Subsample if too many
    if len(all_points) > max_points:
        indices = np.random.choice(len(all_points), max_points, replace=False)
        all_points = all_points[indices]
        all_colors = all_colors[indices]
        print(f"Subsampled to: {len(all_points)}")
    
    # Write outputs
    ply_path = sparse_dir / "points3D.ply"
    txt_path = sparse_dir / "points3D.txt"
    
    write_ply(ply_path, all_points, all_colors)
    write_points3d_txt(txt_path, all_points, all_colors)
    
    print(f"Written {len(all_points)} points to:")
    print(f"  {ply_path}")
    print(f"  {txt_path}")
    
    return str(ply_path)


def main():
    parser = argparse.ArgumentParser(description="Generate initial 3D points from depth maps")
    parser.add_argument("--data_dir", "-d", required=True, help="Path to 2DGS data directory")
    parser.add_argument("--sample_rate", "-s", type=int, default=16, help="Sample every Nth pixel")
    parser.add_argument("--max_depth", type=float, default=100.0, help="Maximum valid depth")
    parser.add_argument("--max_points", type=int, default=100000, help="Maximum total points")
    parser.add_argument("--num_frames", "-n", type=int, default=10, help="Number of frames to use")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    depths_dir = data_dir / "depths"
    sparse_dir = data_dir / "sparse" / "0"
    
    # Parse camera info
    cameras = parse_cameras_txt(sparse_dir / "cameras.txt")
    images = parse_images_txt(sparse_dir / "images.txt")
    
    print(f"Found {len(cameras)} cameras and {len(images)} images")
    
    # Get intrinsic matrix (assume single camera)
    cam = cameras[1]
    fx, fy, cx, cy = cam['params']
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    
    # Collect points from depth maps
    all_points = []
    all_colors = []
    
    # Use evenly spaced frames
    frame_indices = np.linspace(0, len(images) - 1, args.num_frames, dtype=int)
    
    for idx in frame_indices:
        img_info = images[idx]
        
        # Load depth
        depth_name = Path(img_info['name']).stem + ".npy"
        depth_path = depths_dir / depth_name
        
        if not depth_path.exists():
            print(f"  Skipping {depth_name}: depth not found")
            continue
        
        depth = np.load(depth_path)
        
        # Get pose
        R = qvec_to_rotmat(img_info['qvec'])
        t = np.array(img_info['tvec'])
        
        # Backproject
        points, colors = backproject_depth(
            depth, K, R, t, 
            sample_rate=args.sample_rate,
            max_depth=args.max_depth
        )
        
        print(f"  Frame {idx}: {len(points)} points (depth range: {depth[depth > 0].min():.2f} - {depth[depth > 0].max():.2f})")
        
        all_points.append(points)
        all_colors.append(colors)
    
    # Combine all points
    all_points = np.vstack(all_points)
    all_colors = np.vstack(all_colors)
    
    print(f"\nTotal points: {len(all_points)}")
    
    # Subsample if too many
    if len(all_points) > args.max_points:
        indices = np.random.choice(len(all_points), args.max_points, replace=False)
        all_points = all_points[indices]
        all_colors = all_colors[indices]
        print(f"Subsampled to: {len(all_points)}")
    
    # Write outputs
    ply_path = sparse_dir / "points3D.ply"
    txt_path = sparse_dir / "points3D.txt"
    
    write_ply(ply_path, all_points, all_colors)
    write_points3d_txt(txt_path, all_points, all_colors)
    
    print(f"\nWritten {len(all_points)} points to:")
    print(f"  {ply_path}")
    print(f"  {txt_path}")
    
    # Print stats
    print(f"\nPoint cloud stats:")
    print(f"  X range: {all_points[:, 0].min():.3f} to {all_points[:, 0].max():.3f}")
    print(f"  Y range: {all_points[:, 1].min():.3f} to {all_points[:, 1].max():.3f}")
    print(f"  Z range: {all_points[:, 2].min():.3f} to {all_points[:, 2].max():.3f}")


if __name__ == "__main__":
    main()
