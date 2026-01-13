"""
SEVA (Stable Virtual Camera) Generator Module

Generates novel view videos from single images with precise camera control.
Uses Stability AI's Stable Virtual Camera (1.3B parameter diffusion model).

Supported trajectories:
    - orbit: 360° rotation around subject
    - pan: Horizontal camera movement  
    - tilt: Vertical camera angle change
    - spiral: Spiral path around subject
    - zoom-out: Camera moves backward
    - dolly-zoom-out: Vertigo/Hitchcock effect
    - arc: Curved path
    - crane: Vertical + horizontal movement
    - left, right, up, down: Simple directional movements
    - custom: User-defined camera poses (C2W matrices)

Usage:
    from generators.seva import run_seva_runpod
    
    result = run_seva_runpod(
        image_path="input.png",
        output_dir="./outputs/seva",
        trajectory="orbit",
        duration=5.0,
        api_key="your_key",
        endpoint_id="your_endpoint"
    )
    
    if result.success:
        print(f"Video saved to: {result.video_path}")
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable

# Import client from runpod module
from runpod.runpod_client import SEVAServerlessClient, SEVAResult, get_seva_client


# Valid trajectory options
VALID_TRAJECTORIES = [
    "orbit", "pan", "tilt", "spiral", 
    "zoom-out", "dolly-zoom-out", "arc", "crane",
    "left", "right", "up", "down",
    "custom"
]


def run_seva_runpod(
    image_path: str,
    output_dir: str = "./outputs/seva",
    output_name: str = "seva_output",
    trajectory: str = "orbit",
    duration: float = 5.0,
    fps: int = 24,
    num_frames: Optional[int] = None,
    custom_poses: Optional[List] = None,
    seed: Optional[int] = None,
    api_key: Optional[str] = None,
    endpoint_id: Optional[str] = None,
    s3_bucket: str = "arkrunr",
    s3_region: str = "us-west-1",
    poll_interval: int = 10,
    max_wait: int = 600,
    progress_callback: Optional[Callable] = None
) -> SEVAResult:
    """
    Generate novel view video using SEVA on RunPod serverless.
    
    Args:
        image_path: Path to input image
        output_dir: Local directory for output video
        output_name: Name for output file (without extension)
        trajectory: Camera trajectory type (orbit, pan, tilt, etc.)
        duration: Video duration in seconds (1-30)
        fps: Frames per second (12-60)
        num_frames: Override frame count (ignores duration if set)
        custom_poses: List of 4x4 C2W matrices for custom trajectory
        seed: Random seed for reproducibility
        api_key: RunPod API key
        endpoint_id: RunPod endpoint ID for SEVA
        s3_bucket: S3 bucket for file transfer
        s3_region: S3 region
        poll_interval: Seconds between status polls
        max_wait: Maximum wait time in seconds
        progress_callback: Optional callback(status_dict, elapsed_seconds)
        
    Returns:
        SEVAResult with video path and metadata
    """
    # Validate inputs
    if not os.path.exists(image_path):
        return SEVAResult(
            success=False,
            error=f"Input image not found: {image_path}"
        )
    
    if trajectory not in VALID_TRAJECTORIES:
        return SEVAResult(
            success=False,
            error=f"Invalid trajectory '{trajectory}'. Valid options: {VALID_TRAJECTORIES}"
        )
    
    if not api_key:
        return SEVAResult(
            success=False,
            error="RunPod API key is required"
        )
    
    if not endpoint_id:
        return SEVAResult(
            success=False,
            error="RunPod endpoint ID is required"
        )
    
    # Create client
    client = SEVAServerlessClient(
        endpoint_id=endpoint_id,
        api_key=api_key
    )
    
    # Generate video
    return client.generate_sync(
        image_path=image_path,
        output_dir=output_dir,
        output_name=output_name,
        trajectory=trajectory,
        duration=duration,
        fps=fps,
        num_frames=num_frames,
        custom_poses=custom_poses,
        seed=seed,
        s3_bucket=s3_bucket,
        s3_region=s3_region,
        poll_interval=poll_interval,
        max_wait=max_wait,
        progress_callback=progress_callback
    )


def create_custom_trajectory(
    start_position: tuple = (0, 0, 0),
    end_position: tuple = (0, 0.5, 0),
    start_rotation: tuple = (0, 0, 0),
    end_rotation: tuple = (-15, 0, 0),
    num_frames: int = 120
) -> List[List[List[float]]]:
    """
    Create a custom camera trajectory as a list of C2W matrices.
    
    This helper function creates a linear interpolation between
    start and end positions/rotations.
    
    Args:
        start_position: (x, y, z) starting position
        end_position: (x, y, z) ending position  
        start_rotation: (pitch, yaw, roll) starting rotation in degrees
        end_rotation: (pitch, yaw, roll) ending rotation in degrees
        num_frames: Number of frames in trajectory
        
    Returns:
        List of 4x4 C2W matrices as nested lists
        
    Example:
        # Camera moves up 0.5 units and tilts down 15 degrees
        poses = create_custom_trajectory(
            start_position=(0, 0, 0),
            end_position=(0, 0.5, 0),
            start_rotation=(0, 0, 0),
            end_rotation=(-15, 0, 0),
            num_frames=120
        )
        
        result = run_seva_runpod(
            image_path="input.png",
            trajectory="custom",
            custom_poses=poses,
            ...
        )
    """
    import numpy as np
    
    def euler_to_rotation_matrix(pitch, yaw, roll):
        """Convert Euler angles (degrees) to rotation matrix."""
        pitch = np.radians(pitch)
        yaw = np.radians(yaw)
        roll = np.radians(roll)
        
        # Rotation matrices for each axis
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(pitch), -np.sin(pitch)],
            [0, np.sin(pitch), np.cos(pitch)]
        ])
        
        Ry = np.array([
            [np.cos(yaw), 0, np.sin(yaw)],
            [0, 1, 0],
            [-np.sin(yaw), 0, np.cos(yaw)]
        ])
        
        Rz = np.array([
            [np.cos(roll), -np.sin(roll), 0],
            [np.sin(roll), np.cos(roll), 0],
            [0, 0, 1]
        ])
        
        return Rz @ Ry @ Rx
    
    poses = []
    
    for i in range(num_frames):
        t = i / (num_frames - 1) if num_frames > 1 else 0
        
        # Interpolate position
        pos = np.array([
            start_position[0] + t * (end_position[0] - start_position[0]),
            start_position[1] + t * (end_position[1] - start_position[1]),
            start_position[2] + t * (end_position[2] - start_position[2])
        ])
        
        # Interpolate rotation
        rot = (
            start_rotation[0] + t * (end_rotation[0] - start_rotation[0]),
            start_rotation[1] + t * (end_rotation[1] - start_rotation[1]),
            start_rotation[2] + t * (end_rotation[2] - start_rotation[2])
        )
        
        # Build C2W matrix
        R = euler_to_rotation_matrix(*rot)
        c2w = np.eye(4)
        c2w[:3, :3] = R
        c2w[:3, 3] = pos
        
        poses.append(c2w.tolist())
    
    return poses


def create_orbit_trajectory(
    radius: float = 1.0,
    elevation: float = 0.0,
    num_frames: int = 120,
    start_angle: float = 0.0,
    end_angle: float = 360.0
) -> List[List[List[float]]]:
    """
    Create an orbital camera trajectory.
    
    Args:
        radius: Distance from center
        elevation: Height above center (Y axis)
        num_frames: Number of frames
        start_angle: Starting azimuth angle in degrees
        end_angle: Ending azimuth angle in degrees
        
    Returns:
        List of 4x4 C2W matrices
    """
    import numpy as np
    
    poses = []
    
    for i in range(num_frames):
        t = i / (num_frames - 1) if num_frames > 1 else 0
        angle = np.radians(start_angle + t * (end_angle - start_angle))
        
        # Camera position
        x = radius * np.sin(angle)
        z = radius * np.cos(angle)
        y = elevation
        
        # Camera looks at center
        forward = np.array([-x, -y, -z])
        forward = forward / np.linalg.norm(forward)
        
        # Up vector
        up = np.array([0, 1, 0])
        
        # Right vector
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        
        # Recompute up
        up = np.cross(right, forward)
        
        # Build rotation matrix
        R = np.stack([right, up, -forward], axis=1)
        
        # Build C2W matrix
        c2w = np.eye(4)
        c2w[:3, :3] = R
        c2w[:3, 3] = [x, y, z]
        
        poses.append(c2w.tolist())
    
    return poses


# Convenience function for the most common use case
def generate_camera_video(
    image_path: str,
    output_dir: str,
    trajectory: str = "orbit",
    duration: float = 5.0,
    api_key: str = "",
    endpoint_id: str = "",
    **kwargs
) -> SEVAResult:
    """
    Simplified interface for generating camera movement videos.
    
    This is a convenience wrapper around run_seva_runpod() with
    sensible defaults.
    
    Args:
        image_path: Path to input image
        output_dir: Directory for output video
        trajectory: Camera movement type
        duration: Video length in seconds
        api_key: RunPod API key
        endpoint_id: SEVA endpoint ID
        **kwargs: Additional arguments passed to run_seva_runpod()
        
    Returns:
        SEVAResult with video path
    """
    output_name = kwargs.pop("output_name", Path(image_path).stem + "_camera")
    
    return run_seva_runpod(
        image_path=image_path,
        output_dir=output_dir,
        output_name=output_name,
        trajectory=trajectory,
        duration=duration,
        api_key=api_key,
        endpoint_id=endpoint_id,
        **kwargs
    )
