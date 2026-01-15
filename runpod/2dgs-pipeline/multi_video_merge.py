"""
Multi-Video Merge Utilities for 2DGS Pipeline

Functions for aligning and merging multiple video outputs from ViPE
into a single COLMAP-format dataset for 2DGS training.

Key concept: Frame 0 Alignment
- All videos are generated from the same input image
- Frame 0 of each video represents the same camera position
- We use this as an anchor point to align all pose coordinate systems
"""

import os
import shutil
import zipfile
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np


def align_poses_to_reference(poses_list: List[np.ndarray]) -> List[np.ndarray]:
    """
    Align all pose sequences using frame 0 as the anchor point.
    
    Since all videos start from the same input image, frame 0 poses should
    represent the same camera position. We align all subsequent videos to
    video 1's frame 0 pose.
    
    Args:
        poses_list: List of (N, 4, 4) pose arrays, one per video
        
    Returns:
        List of aligned pose arrays (with frame 0 removed from videos 2+)
    """
    if len(poses_list) <= 1:
        return poses_list
    
    print(f"\n[Pose Alignment] Aligning {len(poses_list)} video pose sequences...")
    
    # Reference pose is frame 0 of first video
    reference_pose = poses_list[0][0]  # (4, 4)
    
    aligned_poses = [poses_list[0]]  # Keep all frames from video 1
    
    for idx, poses in enumerate(poses_list[1:], start=2):
        # Compute transform to align this video's frame 0 to reference
        source_pose = poses[0]  # Frame 0 of this video
        
        # Transform = reference * inverse(source)
        # This gives us the transform that maps source coordinate system to reference
        transform = reference_pose @ np.linalg.inv(source_pose)
        
        # Apply transform to all poses in this video
        aligned = np.array([transform @ p for p in poses])
        
        # Skip frame 0 (it's a duplicate of the reference)
        aligned_poses.append(aligned[1:])
        
        print(f"  Video {idx}: Aligned {len(poses)-1} poses (skipped duplicate frame 0)")
    
    return aligned_poses


def filter_frames_by_depth_coverage(
    depth_dir: Path, 
    threshold: float = 0.5
) -> Tuple[List[int], float, int, int]:
    """
    Filter frames based on depth coverage.
    
    Args:
        depth_dir: Directory containing depth .npy files
        threshold: Minimum fraction of valid (non-zero) depth pixels
        
    Returns:
        Tuple of (valid_frame_indices, avg_coverage, total_frames, skipped_frames)
    """
    valid_frames = []
    total_coverage = 0.0
    skipped = 0
    
    # Find all depth files
    depth_files = sorted(depth_dir.glob("*.npy"))
    
    if not depth_files:
        # Try loading from zip
        zip_path = depth_dir.parent / "depth" / "input.zip"
        if zip_path.exists():
            with zipfile.ZipFile(zip_path, 'r') as z:
                depth_files = sorted([n for n in z.namelist() if n.endswith('.npy')])
                for i, name in enumerate(depth_files):
                    with z.open(name) as f:
                        depth = np.load(f)
                        valid_pixels = np.sum(depth > 0)
                        total_pixels = depth.size
                        coverage = valid_pixels / total_pixels
                        total_coverage += coverage
                        
                        if coverage >= threshold:
                            valid_frames.append(i)
                        else:
                            skipped += 1
                            print(f"    [Quality] Skipping frame {i}: depth coverage {coverage:.1%} < {threshold:.0%}")
        else:
            print(f"    [Quality] Warning: No depth files found in {depth_dir}")
            return list(range(100)), 0.0, 0, 0  # Fallback - keep all frames
    else:
        for i, depth_file in enumerate(depth_files):
            depth = np.load(depth_file)
            valid_pixels = np.sum(depth > 0)
            total_pixels = depth.size
            coverage = valid_pixels / total_pixels
            total_coverage += coverage
            
            if coverage >= threshold:
                valid_frames.append(i)
            else:
                skipped += 1
                print(f"    [Quality] Skipping frame {i}: depth coverage {coverage:.1%} < {threshold:.0%}")
    
    total_frames = len(depth_files) if depth_files else len(valid_frames) + skipped
    avg_coverage = total_coverage / total_frames if total_frames > 0 else 0.0
    
    print(f"    [Quality] Kept {len(valid_frames)}/{total_frames} frames (avg coverage: {avg_coverage:.1%})")
    return valid_frames, avg_coverage, total_frames, skipped


def deduplicate_frame0(
    frames_list: List[List[Path]], 
    keep_first: bool = True
) -> List[Path]:
    """
    Keep only one frame 0, combine rest.
    
    Args:
        frames_list: List of frame path lists, one per video
        keep_first: If True, keep frame 0 from first video only
        
    Returns:
        Flattened list of unique frame paths
    """
    combined = []
    
    for video_idx, frames in enumerate(frames_list):
        if video_idx == 0:
            # Keep all frames from first video
            combined.extend(frames)
        else:
            # Skip frame 0 from subsequent videos (it's a duplicate)
            combined.extend(frames[1:])
    
    return combined


def merge_vipe_outputs(
    vipe_dirs: List[Path], 
    output_dir: Path,
    valid_frames_per_video: Optional[List[List[int]]] = None
) -> Tuple[Path, int]:
    """
    Merge multiple ViPE outputs into single COLMAP-format dataset.
    
    Args:
        vipe_dirs: List of paths to ViPE output directories
        output_dir: Directory for merged output
        valid_frames_per_video: Optional list of valid frame indices per video
        
    Returns:
        Tuple of (path to merged ViPE directory, total frame count)
    """
    print(f"\n[Merge] Combining {len(vipe_dirs)} video outputs...")
    
    merged_dir = output_dir / "merged_vipe"
    merged_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all poses
    all_poses = []
    all_intrinsics = []
    
    for vipe_dir in vipe_dirs:
        poses_path = vipe_dir / "pose" / "input.npz"
        intrinsics_path = vipe_dir / "intrinsics" / "input.npz"
        
        poses_data = np.load(poses_path)
        poses = poses_data["data"]  # (N, 4, 4)
        all_poses.append(poses)
        
        intrinsics_data = np.load(intrinsics_path)
        intrinsics = intrinsics_data["data"]  # (N, 3, 3) or (3, 3)
        all_intrinsics.append(intrinsics)
    
    # Align poses using frame 0 anchor
    aligned_poses = align_poses_to_reference(all_poses)
    
    # If no valid_frames specified, use all frames
    if valid_frames_per_video is None:
        valid_frames_per_video = [list(range(len(p))) for p in all_poses]
    
    # Flatten aligned poses and corresponding intrinsics
    merged_poses = []
    merged_intrinsics = []
    frame_mapping = []  # Maps merged index to (video_idx, original_frame_idx)
    
    for video_idx, (poses, intrinsics) in enumerate(zip(aligned_poses, all_intrinsics)):
        valid_frames = valid_frames_per_video[video_idx]
        
        for frame_idx in valid_frames:
            # Skip frame 0 for videos after the first
            if video_idx > 0 and frame_idx == 0:
                continue
            
            # Adjust index for aligned poses (which skip frame 0)
            adjusted_idx = frame_idx if video_idx == 0 else frame_idx - 1
            
            if adjusted_idx >= 0 and adjusted_idx < len(poses):
                merged_poses.append(poses[adjusted_idx])
                
                # Handle intrinsics (may be per-frame or shared)
                if len(intrinsics.shape) == 3:
                    idx = frame_idx if frame_idx < len(intrinsics) else 0
                    merged_intrinsics.append(intrinsics[idx])
                else:
                    merged_intrinsics.append(intrinsics)
                
                frame_mapping.append((video_idx, frame_idx))
    
    # Save merged poses and intrinsics
    merged_poses = np.array(merged_poses)
    merged_intrinsics = np.array(merged_intrinsics)
    
    pose_dir = merged_dir / "pose"
    pose_dir.mkdir(parents=True, exist_ok=True)
    np.savez(pose_dir / "input.npz", data=merged_poses)
    
    intrinsics_dir = merged_dir / "intrinsics"
    intrinsics_dir.mkdir(parents=True, exist_ok=True)
    np.savez(intrinsics_dir / "input.npz", data=merged_intrinsics)
    
    # Create merged depth directory
    merged_depth_dir = merged_dir / "depth"
    merged_depth_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract and merge depth maps
    for merged_idx, (video_idx, frame_idx) in enumerate(frame_mapping):
        src_zip = vipe_dirs[video_idx] / "depth" / "input.zip"
        
        if src_zip.exists():
            with zipfile.ZipFile(src_zip, 'r') as z:
                # ViPE names depth files as 000000.npy, 000001.npy, etc.
                src_name = f"{frame_idx:06d}.npy"
                if src_name in z.namelist():
                    depth_data = z.read(src_name)
                    dst_path = merged_depth_dir / f"{merged_idx:06d}.npy"
                    with open(dst_path, 'wb') as f:
                        f.write(depth_data)
    
    # Create merged RGB directory
    merged_rgb_dir = merged_dir / "rgb"
    merged_rgb_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract frames from each video
    for video_idx, vipe_dir in enumerate(vipe_dirs):
        video_path = vipe_dir / "rgb" / "input.mp4"
        
        if video_path.exists():
            temp_frames = merged_dir / f"temp_frames_{video_idx}"
            temp_frames.mkdir(parents=True, exist_ok=True)
            
            cmd = [
                "ffmpeg", "-i", str(video_path),
                "-q:v", "2",
                str(temp_frames / "%06d.jpg")
            ]
            subprocess.run(cmd, capture_output=True)
    
    # Copy valid frames to merged rgb directory
    for merged_idx, (video_idx, frame_idx) in enumerate(frame_mapping):
        temp_frames = merged_dir / f"temp_frames_{video_idx}"
        src_frame = temp_frames / f"{frame_idx+1:06d}.jpg"  # ffmpeg is 1-indexed
        
        if src_frame.exists():
            dst_frame = merged_rgb_dir / f"{merged_idx:06d}.jpg"
            shutil.copy(src_frame, dst_frame)
    
    # Cleanup temp directories
    for video_idx in range(len(vipe_dirs)):
        temp_frames = merged_dir / f"temp_frames_{video_idx}"
        if temp_frames.exists():
            shutil.rmtree(temp_frames)
    
    print(f"[Merge] Created merged dataset with {len(merged_poses)} frames")
    print(f"  Poses: {pose_dir / 'input.npz'}")
    print(f"  Intrinsics: {intrinsics_dir / 'input.npz'}")
    print(f"  Depth: {merged_depth_dir} ({len(list(merged_depth_dir.glob('*.npy')))} files)")
    print(f"  RGB: {merged_rgb_dir} ({len(list(merged_rgb_dir.glob('*.jpg')))} files)")
    
    return merged_dir, len(merged_poses)


def compute_pose_consistency(poses: np.ndarray) -> float:
    """
    Compute a consistency score for a sequence of poses.
    
    Measures the smoothness of camera motion by checking for
    sudden jumps in position/rotation.
    
    Args:
        poses: (N, 4, 4) array of camera poses
        
    Returns:
        Consistency score between 0 (inconsistent) and 1 (smooth)
    """
    if len(poses) < 2:
        return 1.0
    
    # Compute position changes
    positions = poses[:, :3, 3]  # (N, 3)
    position_deltas = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    
    # Compute rotation changes (simplified - using rotation matrix difference)
    rotations = poses[:, :3, :3]  # (N, 3, 3)
    rotation_deltas = []
    for i in range(len(rotations) - 1):
        # Frobenius norm of rotation difference
        delta = np.linalg.norm(rotations[i+1] - rotations[i])
        rotation_deltas.append(delta)
    rotation_deltas = np.array(rotation_deltas)
    
    # Score based on standard deviation of changes
    # Smooth motion = low std dev
    pos_std = np.std(position_deltas)
    rot_std = np.std(rotation_deltas)
    
    # Normalize to 0-1 range (empirical thresholds)
    pos_score = max(0, 1 - pos_std / 0.1)
    rot_score = max(0, 1 - rot_std / 0.5)
    
    return (pos_score + rot_score) / 2


if __name__ == "__main__":
    # Test with dummy data
    print("Testing multi_video_merge.py...")
    
    # Create dummy poses
    poses1 = np.eye(4)[np.newaxis, :, :].repeat(10, axis=0)
    poses2 = np.eye(4)[np.newaxis, :, :].repeat(10, axis=0)
    
    # Add some motion to poses
    for i in range(10):
        poses1[i, 0, 3] = i * 0.1  # Move along X
        poses2[i, 1, 3] = i * 0.1  # Move along Y
    
    aligned = align_poses_to_reference([poses1, poses2])
    
    print(f"Input: 2 videos with {len(poses1)} frames each")
    print(f"Output: {len(aligned)} pose arrays")
    print(f"  Video 1: {len(aligned[0])} frames")
    print(f"  Video 2: {len(aligned[1])} frames (frame 0 skipped)")
    
    consistency = compute_pose_consistency(poses1)
    print(f"Pose consistency score: {consistency:.2f}")
