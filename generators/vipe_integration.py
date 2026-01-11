#!/usr/bin/env python3
"""
ViPE Integration for Gen3C → 2DGS Pipeline

Extracts camera poses and depth from Gen3C videos using NVIDIA ViPE.
Repository: https://github.com/nv-tlabs/vipe
"""

import os
import subprocess
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import json


class ViPEIntegration:
    """
    Wrapper for NVIDIA ViPE (Video Pose Engine).
    
    ViPE extracts:
    - Camera intrinsics (focal length, principal point)
    - Camera extrinsics (4×4 pose matrices)
    - Dense depth maps per frame
    
    This is the official NVIDIA tool for processing Gen3C videos.
    """
    
    def __init__(
        self,
        vipe_dir: str = "/workspace/vipe",
        output_base: str = "/workspace/vipe_results",
    ):
        """
        Initialize ViPE integration.
        
        Args:
            vipe_dir: Path to cloned ViPE repository
            output_base: Base directory for ViPE outputs
        """
        self.vipe_dir = Path(vipe_dir)
        self.output_base = Path(output_base)
        self.output_base.mkdir(parents=True, exist_ok=True)
        
        self._vipe_available = None
    
    def is_available(self) -> bool:
        """Check if ViPE is installed and available."""
        if self._vipe_available is not None:
            return self._vipe_available
        
        try:
            result = subprocess.run(
                ["vipe", "--help"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            self._vipe_available = result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            self._vipe_available = False
        
        return self._vipe_available
    
    def process_video(
        self,
        video_path: str,
        output_name: Optional[str] = None,
        visualize: bool = False,
        max_resolution: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Process a video with ViPE to extract poses and depth.
        
        Args:
            video_path: Path to input video (MP4)
            output_name: Name for output directory (default: video filename)
            visualize: Enable visualization output
            max_resolution: Maximum resolution for processing (saves memory)
        
        Returns:
            Dict with paths to outputs and loaded data
        """
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        if not self.is_available():
            raise RuntimeError("ViPE is not installed or not in PATH")
        
        # Determine output directory
        output_name = output_name or video_path.stem
        output_dir = self.output_base / output_name
        
        # Build command
        cmd = [
            "vipe", "infer",
            str(video_path),
            "--output", str(output_dir),
        ]
        
        if visualize:
            cmd.append("--visualize")
        
        if max_resolution:
            cmd.extend(["--max_resolution", str(max_resolution)])
        
        print(f"[ViPE] Processing: {video_path}")
        print(f"[ViPE] Output: {output_dir}")
        print(f"[ViPE] Command: {' '.join(cmd)}")
        
        # Run ViPE
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
        )
        
        if result.returncode != 0:
            print(f"[ViPE] STDERR: {result.stderr}")
            raise RuntimeError(f"ViPE failed: {result.stderr[-1000:]}")
        
        print(f"[ViPE] Processing complete")
        
        # Load results
        return self.load_results(str(output_dir))
    
    def load_results(self, vipe_dir: str) -> Dict[str, Any]:
        """
        Load ViPE output files.
        
        Args:
            vipe_dir: Path to ViPE output directory
        
        Returns:
            Dict with:
            - poses: (N, 4, 4) camera-to-world matrices
            - intrinsics: Camera intrinsic matrix/matrices
            - depths: List of depth arrays
            - depth_paths: List of depth file paths
            - num_frames: Number of frames
            - output_dir: Path to output directory
        """
        vipe_dir = Path(vipe_dir)
        
        if not vipe_dir.exists():
            raise FileNotFoundError(f"ViPE output not found: {vipe_dir}")
        
        result = {
            "output_dir": str(vipe_dir),
            "poses": None,
            "intrinsics": None,
            "depths": [],
            "depth_paths": [],
            "num_frames": 0,
        }
        
        # Load poses
        poses_path = vipe_dir / "poses.npy"
        if poses_path.exists():
            result["poses"] = np.load(poses_path)
            result["num_frames"] = len(result["poses"])
            print(f"[ViPE] Loaded poses: {result['poses'].shape}")
        
        # Load intrinsics
        intrinsics_path = vipe_dir / "intrinsics.npy"
        if intrinsics_path.exists():
            result["intrinsics"] = np.load(intrinsics_path)
            print(f"[ViPE] Loaded intrinsics: {result['intrinsics'].shape}")
        
        # Load depth maps
        depth_dir = vipe_dir / "depth"
        if depth_dir.exists():
            depth_files = sorted(depth_dir.glob("*.npy"))
            result["depth_paths"] = [str(f) for f in depth_files]
            
            # Load first few depths to verify
            if depth_files:
                sample_depth = np.load(depth_files[0])
                print(f"[ViPE] Loaded {len(depth_files)} depth maps, shape: {sample_depth.shape}")
        
        return result
    
    def extract_frames(
        self,
        video_path: str,
        output_dir: str,
        every_n: int = 1,
    ) -> List[str]:
        """
        Extract frames from video.
        
        Args:
            video_path: Path to video
            output_dir: Output directory for frames
            every_n: Extract every Nth frame
        
        Returns:
            List of frame file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Use ffmpeg for frame extraction
        cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-q:v", "2",  # High quality
        ]
        
        if every_n > 1:
            cmd.extend(["-vf", f"select='not(mod(n\\,{every_n}))'", "-vsync", "vfr"])
        
        cmd.append(str(output_dir / "frame_%06d.png"))
        
        print(f"[ViPE] Extracting frames from: {video_path}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            # ffmpeg often returns warnings, check if frames were created
            pass
        
        frame_files = sorted(output_dir.glob("frame_*.png"))
        print(f"[ViPE] Extracted {len(frame_files)} frames")
        
        return [str(f) for f in frame_files]
    
    def prepare_2dgs_data(
        self,
        frames_dir: str,
        vipe_results: Dict[str, Any],
        output_dir: str,
    ) -> str:
        """
        Convert ViPE output to 2DGS-compatible COLMAP format.
        
        Args:
            frames_dir: Directory containing extracted frames
            vipe_results: Output from load_results()
            output_dir: Output directory for 2DGS data
        
        Returns:
            Path to prepared 2DGS data directory
        """
        import cv2
        from scipy.spatial.transform import Rotation
        
        frames_dir = Path(frames_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        poses = vipe_results["poses"]
        intrinsics = vipe_results["intrinsics"]
        
        # Get frame files
        frame_files = sorted(frames_dir.glob("*.png"))
        
        if len(frame_files) != len(poses):
            print(f"[ViPE] Warning: {len(frame_files)} frames but {len(poses)} poses")
            # Use minimum
            n_frames = min(len(frame_files), len(poses))
            frame_files = frame_files[:n_frames]
            poses = poses[:n_frames]
        
        # Create sparse directory structure (COLMAP format)
        sparse_dir = output_dir / "sparse" / "0"
        sparse_dir.mkdir(parents=True, exist_ok=True)
        
        images_dir = output_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        # Copy frames to images directory
        for i, frame_file in enumerate(frame_files):
            dest = images_dir / f"{i:06d}.png"
            shutil.copy(frame_file, dest)
        
        # Read image dimensions
        sample_img = cv2.imread(str(frame_files[0]))
        h, w = sample_img.shape[:2]
        
        # Get intrinsics (handle both shared and per-frame)
        if intrinsics.ndim == 2:
            K = intrinsics
        else:
            K = intrinsics[0]  # Use first frame's intrinsics
        
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        
        # Write cameras.txt
        cameras_path = sparse_dir / "cameras.txt"
        with open(cameras_path, "w") as f:
            f.write("# Camera list with one line of data per camera:\n")
            f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
            f.write(f"1 PINHOLE {w} {h} {fx} {fy} {cx} {cy}\n")
        
        # Write images.txt
        images_path = sparse_dir / "images.txt"
        with open(images_path, "w") as f:
            f.write("# Image list with two lines of data per image:\n")
            f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
            f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
            
            for i, pose in enumerate(poses):
                # ViPE outputs camera-to-world, COLMAP expects world-to-camera
                pose_inv = np.linalg.inv(pose)
                R = pose_inv[:3, :3]
                t = pose_inv[:3, 3]
                
                # Convert rotation matrix to quaternion
                quat = Rotation.from_matrix(R).as_quat()  # [x, y, z, w]
                qw, qx, qy, qz = quat[3], quat[0], quat[1], quat[2]
                
                f.write(f"{i+1} {qw} {qx} {qy} {qz} {t[0]} {t[1]} {t[2]} 1 {i:06d}.png\n")
                f.write("\n")  # Empty line for points
        
        # Write empty points3D.txt (2DGS will create its own points)
        points_path = sparse_dir / "points3D.txt"
        with open(points_path, "w") as f:
            f.write("# 3D point list\n")
        
        # Save depth maps in a separate directory for 2DGS depth supervision
        if vipe_results.get("depth_paths"):
            depth_out_dir = output_dir / "depths"
            depth_out_dir.mkdir(exist_ok=True)
            
            for i, depth_path in enumerate(vipe_results["depth_paths"][:len(frame_files)]):
                depth = np.load(depth_path)
                np.save(depth_out_dir / f"{i:06d}.npy", depth)
        
        print(f"[ViPE] 2DGS data prepared at: {output_dir}")
        print(f"[ViPE]   Images: {len(frame_files)}")
        print(f"[ViPE]   Poses: {len(poses)}")
        print(f"[ViPE]   Image size: {w}x{h}")
        print(f"[ViPE]   Focal: fx={fx:.1f}, fy={fy:.1f}")
        
        return str(output_dir)


def run_gen3c_vipe_pipeline(
    image_path: str,
    output_dir: str,
    trajectory: str = "clockwise",
    movement_distance: float = 0.3,
    num_frames: int = 121,
) -> Dict[str, Any]:
    """
    Run the complete Gen3C → ViPE pipeline.
    
    Args:
        image_path: Path to input image
        output_dir: Output directory
        trajectory: Gen3C trajectory type
        movement_distance: Camera movement amount
        num_frames: Number of video frames
    
    Returns:
        Dict with paths to all outputs
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Import Gen3C generator
    from generators.gen3c import run_gen3c_generation
    
    result = {
        "input_image": str(image_path),
        "output_dir": str(output_dir),
    }
    
    # Step 1: Generate video with Gen3C
    print("\n" + "="*60)
    print("STEP 1: Generating video with Gen3C")
    print("="*60)
    
    video_name = f"gen3c_{trajectory}"
    gen3c_result = run_gen3c_generation(
        image_path=image_path,
        video_name=video_name,
        trajectory=trajectory,
        movement_distance=movement_distance,
        num_frames=num_frames,
    )
    
    video_path = gen3c_result.get("video_path")
    if not video_path:
        raise RuntimeError("Gen3C did not produce video")
    
    result["video_path"] = video_path
    print(f"Video generated: {video_path}")
    
    # Step 2: Process with ViPE
    print("\n" + "="*60)
    print("STEP 2: Extracting poses and depth with ViPE")
    print("="*60)
    
    vipe = ViPEIntegration(output_base=str(output_dir / "vipe"))
    
    if not vipe.is_available():
        print("WARNING: ViPE not available!")
        print("Install with: git clone https://github.com/nv-tlabs/vipe.git && pip install -r requirements.txt")
        result["vipe_available"] = False
        return result
    
    vipe_results = vipe.process_video(video_path, output_name="poses")
    result["vipe_results"] = vipe_results
    print(f"ViPE output: {vipe_results['output_dir']}")
    
    # Step 3: Extract frames
    print("\n" + "="*60)
    print("STEP 3: Extracting video frames")
    print("="*60)
    
    frames_dir = output_dir / "frames"
    frame_files = vipe.extract_frames(video_path, str(frames_dir))
    result["frames_dir"] = str(frames_dir)
    result["num_frames"] = len(frame_files)
    print(f"Extracted {len(frame_files)} frames")
    
    # Step 4: Prepare 2DGS data
    print("\n" + "="*60)
    print("STEP 4: Preparing 2DGS training data")
    print("="*60)
    
    data_dir = vipe.prepare_2dgs_data(
        str(frames_dir),
        vipe_results,
        str(output_dir / "2dgs_data"),
    )
    result["2dgs_data_dir"] = data_dir
    
    # Summary
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"Input image: {image_path}")
    print(f"Video: {video_path}")
    print(f"ViPE output: {vipe_results['output_dir']}")
    print(f"2DGS data: {data_dir}")
    print(f"\nNext step: Train 2DGS with:")
    print(f"  python train.py -s {data_dir} -m {output_dir}/2dgs_model")
    
    return result


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Gen3C → ViPE → 2DGS Pipeline"
    )
    parser.add_argument(
        "--image", "-i",
        required=True,
        help="Input image path"
    )
    parser.add_argument(
        "--output", "-o",
        default="pipeline_output",
        help="Output directory"
    )
    parser.add_argument(
        "--trajectory", "-t",
        default="clockwise",
        choices=["left", "right", "up", "down", "zoom_in", "zoom_out", 
                 "clockwise", "counterclockwise"],
        help="Gen3C camera trajectory"
    )
    parser.add_argument(
        "--movement", "-m",
        type=float,
        default=0.3,
        help="Movement distance (0.1-1.0)"
    )
    parser.add_argument(
        "--frames", "-f",
        type=int,
        default=121,
        help="Number of video frames"
    )
    
    args = parser.parse_args()
    
    run_gen3c_vipe_pipeline(
        image_path=args.image,
        output_dir=args.output,
        trajectory=args.trajectory,
        movement_distance=args.movement,
        num_frames=args.frames,
    )
