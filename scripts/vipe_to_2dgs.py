#!/usr/bin/env python3
"""
Convert ViPE output to 2DGS-compatible COLMAP format.

Usage:
    python vipe_to_2dgs.py --vipe_dir /path/to/vipe_results --output_dir /path/to/2dgs_data

ViPE output structure:
    vipe_results/
    ├── pose/video_name.npz          # Camera poses (N, 4, 4)
    ├── intrinsics/video_name.npz    # Intrinsics (N, 4) = [fx, fy, cx, cy]
    ├── depth/video_name.zip         # Depth maps as EXR files
    ├── rgb/video_name.zip           # RGB frames
    └── mask/video_name.zip          # Segmentation masks

Output (COLMAP-compatible):
    2dgs_data/
    ├── images/                      # RGB images
    │   ├── 000000.png
    │   └── ...
    ├── depths/                      # Depth maps (optional, for supervision)
    │   ├── 000000.npy
    │   └── ...
    ├── sparse/0/                    # COLMAP format
    │   ├── cameras.txt
    │   ├── images.txt
    │   └── points3D.txt
    └── masks/                       # Optional masks
"""

import argparse
import zipfile
import shutil
from pathlib import Path
import numpy as np

try:
    from scipy.spatial.transform import Rotation
except ImportError:
    print("Installing scipy...")
    import subprocess
    subprocess.run(["pip", "install", "scipy"], check=True)
    from scipy.spatial.transform import Rotation

try:
    import cv2
except ImportError:
    print("Installing opencv-python...")
    import subprocess
    subprocess.run(["pip", "install", "opencv-python"], check=True)
    import cv2

try:
    import OpenEXR
    import Imath
    HAS_OPENEXR = True
except ImportError:
    HAS_OPENEXR = False
    print("Warning: OpenEXR not installed. Depth maps will be skipped.")
    print("Install with: pip install OpenEXR")


def load_vipe_poses(npz_path: str) -> np.ndarray:
    """Load poses from ViPE npz file."""
    data = np.load(npz_path)
    poses = data['data']  # (N, 4, 4)
    return poses


def load_vipe_intrinsics(npz_path: str) -> np.ndarray:
    """Load intrinsics from ViPE npz file."""
    data = np.load(npz_path)
    intrinsics = data['data']  # (N, 4) = [fx, fy, cx, cy]
    return intrinsics


def pose_to_colmap(pose: np.ndarray) -> tuple:
    """
    Convert 4x4 pose matrix to COLMAP format.
    
    ViPE outputs camera-to-world pose.
    COLMAP expects world-to-camera (R, t).
    
    Returns:
        (qw, qx, qy, qz, tx, ty, tz)
    """
    # Invert pose: camera-to-world -> world-to-camera
    pose_inv = np.linalg.inv(pose)
    
    R = pose_inv[:3, :3]
    t = pose_inv[:3, 3]
    
    # Convert rotation matrix to quaternion
    quat = Rotation.from_matrix(R).as_quat()  # [x, y, z, w]
    qw, qx, qy, qz = quat[3], quat[0], quat[1], quat[2]
    
    return qw, qx, qy, qz, t[0], t[1], t[2]


def read_exr_depth(exr_path: str) -> np.ndarray:
    """Read depth from EXR file."""
    if not HAS_OPENEXR:
        return None
    
    exr_file = OpenEXR.InputFile(exr_path)
    header = exr_file.header()
    
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    
    # Read the depth channel (usually 'Y' or 'Z')
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    
    # Try different channel names
    channels = header['channels'].keys()
    depth_channel = None
    for ch in ['Y', 'Z', 'R', 'depth']:
        if ch in channels:
            depth_channel = ch
            break
    
    if depth_channel is None:
        depth_channel = list(channels)[0]
    
    depth_str = exr_file.channel(depth_channel, pt)
    depth = np.frombuffer(depth_str, dtype=np.float32).reshape(height, width)
    
    return depth


def convert_vipe_to_2dgs(
    vipe_dir: str,
    output_dir: str,
    video_name: str = None,
    skip_depth: bool = False,
    every_n: int = 1,
):
    """
    Convert ViPE output to 2DGS-compatible format.
    
    Args:
        vipe_dir: Path to ViPE output directory
        output_dir: Path to output directory for 2DGS
        video_name: Name of the video (auto-detected if None)
        skip_depth: Skip depth map extraction
        every_n: Use every Nth frame (for subsampling)
    """
    vipe_dir = Path(vipe_dir)
    output_dir = Path(output_dir)
    
    # Auto-detect video name from pose directory
    if video_name is None:
        pose_files = list((vipe_dir / "pose").glob("*.npz"))
        if not pose_files:
            raise FileNotFoundError("No pose files found in ViPE output")
        video_name = pose_files[0].stem
    
    print(f"Converting ViPE output for: {video_name}")
    print(f"Output directory: {output_dir}")
    
    # Create output directories
    images_dir = output_dir / "images"
    sparse_dir = output_dir / "sparse" / "0"
    depths_dir = output_dir / "depths"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)
    if not skip_depth:
        depths_dir.mkdir(parents=True, exist_ok=True)
    
    # Load poses and intrinsics
    poses = load_vipe_poses(vipe_dir / "pose" / f"{video_name}.npz")
    intrinsics = load_vipe_intrinsics(vipe_dir / "intrinsics" / f"{video_name}.npz")
    
    n_frames = len(poses)
    print(f"Total frames: {n_frames}")
    
    # Determine which frames to use
    frame_indices = list(range(0, n_frames, every_n))
    n_output = len(frame_indices)
    print(f"Using {n_output} frames (every {every_n})")
    
    # Extract RGB images (handle both zip and mp4 formats)
    rgb_zip = vipe_dir / "rgb" / f"{video_name}.zip"
    rgb_mp4 = vipe_dir / "rgb" / f"{video_name}.mp4"
    
    if rgb_zip.exists():
        print("Extracting RGB images from zip...")
        with zipfile.ZipFile(rgb_zip, 'r') as z:
            all_files = sorted(z.namelist())
            for new_idx, orig_idx in enumerate(frame_indices):
                if orig_idx < len(all_files):
                    src_name = all_files[orig_idx]
                    ext = Path(src_name).suffix
                    dst_name = f"{new_idx:06d}{ext}"
                    
                    # Extract and rename
                    data = z.read(src_name)
                    with open(images_dir / dst_name, 'wb') as f:
                        f.write(data)
        print(f"  Extracted {n_output} images")
    elif rgb_mp4.exists():
        print("Extracting RGB frames from video...")
        cap = cv2.VideoCapture(str(rgb_mp4))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"  Video has {frame_count} frames")
        
        for new_idx, orig_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, orig_idx)
            ret, frame = cap.read()
            if ret:
                dst_path = images_dir / f"{new_idx:06d}.png"
                cv2.imwrite(str(dst_path), frame)
        cap.release()
        print(f"  Extracted {n_output} frames")
    else:
        print(f"Warning: No RGB source found (tried {rgb_zip} and {rgb_mp4})")
    
    # Get image dimensions from first extracted image
    sample_images = list(images_dir.glob("*"))
    if sample_images:
        sample_img = cv2.imread(str(sample_images[0]))
        if sample_img is not None:
            height, width = sample_img.shape[:2]
        else:
            # Estimate from intrinsics (cx, cy are usually at center)
            width = int(intrinsics[0, 2] * 2)
            height = int(intrinsics[0, 3] * 2)
    else:
        width = int(intrinsics[0, 2] * 2)
        height = int(intrinsics[0, 3] * 2)
    
    print(f"Image dimensions: {width}x{height}")
    
    # Write cameras.txt
    # Use average intrinsics (they should be similar for all frames)
    avg_fx = np.mean(intrinsics[frame_indices, 0])
    avg_fy = np.mean(intrinsics[frame_indices, 1])
    avg_cx = np.mean(intrinsics[frame_indices, 2])
    avg_cy = np.mean(intrinsics[frame_indices, 3])
    
    cameras_path = sparse_dir / "cameras.txt"
    with open(cameras_path, 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"1 PINHOLE {width} {height} {avg_fx} {avg_fy} {avg_cx} {avg_cy}\n")
    
    print(f"  Intrinsics: fx={avg_fx:.1f}, fy={avg_fy:.1f}, cx={avg_cx:.1f}, cy={avg_cy:.1f}")
    
    # Write images.txt
    images_path = sparse_dir / "images.txt"
    with open(images_path, 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        
        for new_idx, orig_idx in enumerate(frame_indices):
            pose = poses[orig_idx]
            qw, qx, qy, qz, tx, ty, tz = pose_to_colmap(pose)
            
            # Get image filename
            img_files = list(images_dir.glob(f"{new_idx:06d}.*"))
            if img_files:
                img_name = img_files[0].name
            else:
                img_name = f"{new_idx:06d}.png"
            
            f.write(f"{new_idx + 1} {qw} {qx} {qy} {qz} {tx} {ty} {tz} 1 {img_name}\n")
            f.write("\n")  # Empty line for points (no 2D points)
    
    # Write empty points3D.txt
    points_path = sparse_dir / "points3D.txt"
    with open(points_path, 'w') as f:
        f.write("# 3D point list (empty - 2DGS will initialize from depth)\n")
    
    # Extract depth maps
    if not skip_depth:
        depth_zip = vipe_dir / "depth" / f"{video_name}.zip"
        if depth_zip.exists() and HAS_OPENEXR:
            print("Extracting depth maps...")
            import tempfile
            with tempfile.TemporaryDirectory() as tmp_dir:
                with zipfile.ZipFile(depth_zip, 'r') as z:
                    z.extractall(tmp_dir)
                    
                    all_files = sorted(Path(tmp_dir).glob("*.exr"))
                    for new_idx, orig_idx in enumerate(frame_indices):
                        if orig_idx < len(all_files):
                            exr_path = all_files[orig_idx]
                            depth = read_exr_depth(str(exr_path))
                            if depth is not None:
                                np.save(depths_dir / f"{new_idx:06d}.npy", depth)
            
            print(f"  Extracted {len(list(depths_dir.glob('*.npy')))} depth maps")
        else:
            if not depth_zip.exists():
                print(f"Warning: Depth zip not found at {depth_zip}")
            if not HAS_OPENEXR:
                print("Warning: OpenEXR not installed, skipping depth extraction")
    
    # Summary
    print("\n" + "="*60)
    print("CONVERSION COMPLETE")
    print("="*60)
    print(f"Output: {output_dir}")
    print(f"  Images:     {len(list(images_dir.glob('*')))} files")
    print(f"  Cameras:    {cameras_path}")
    print(f"  Poses:      {images_path}")
    if not skip_depth:
        print(f"  Depths:     {len(list(depths_dir.glob('*.npy')))} files")
    
    print(f"\nTo train 2DGS:")
    print(f"  python train.py -s {output_dir} -m {output_dir}/model")
    
    return str(output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Convert ViPE output to 2DGS-compatible COLMAP format"
    )
    parser.add_argument(
        "--vipe_dir", "-v",
        required=True,
        help="Path to ViPE output directory"
    )
    parser.add_argument(
        "--output_dir", "-o",
        required=True,
        help="Path to output directory for 2DGS"
    )
    parser.add_argument(
        "--video_name", "-n",
        default=None,
        help="Video name (auto-detected if not specified)"
    )
    parser.add_argument(
        "--skip_depth",
        action="store_true",
        help="Skip depth map extraction"
    )
    parser.add_argument(
        "--every_n", "-e",
        type=int,
        default=1,
        help="Use every Nth frame (default: 1 = all frames)"
    )
    
    args = parser.parse_args()
    
    convert_vipe_to_2dgs(
        vipe_dir=args.vipe_dir,
        output_dir=args.output_dir,
        video_name=args.video_name,
        skip_depth=args.skip_depth,
        every_n=args.every_n,
    )


if __name__ == "__main__":
    main()
