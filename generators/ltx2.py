"""
LTX-2 Video Generation Module

Generates high-quality videos from images using Lightricks' LTX-2 model.
Supports camera control via LoRA adapters.

Model: Lightricks LTX-2 (19B parameters)
- Open weights (no HuggingFace approval needed)
- Up to 4K @ 50fps output
- Camera control LoRAs for directional movement
- Fast inference with distilled model (8 steps)

Camera Motion Options:
    - dolly_left: Camera moves laterally left
    - dolly_right: Camera moves laterally right  
    - dolly_in: Camera pushes toward subject
    - dolly_out: Camera pulls away (best for 3D reconstruction)
    - jib_up: Camera rises vertically
    - static: No camera movement
    - none: No camera LoRA applied

Usage:
    from generators.ltx2 import run_ltx2_runpod
    
    result = run_ltx2_runpod(
        image_path="input.png",
        output_dir="./outputs/ltx2",
        prompt="A detailed 3D object rotating",
        camera_motion="dolly_out",
        api_key="your_key",
        endpoint_id="your_endpoint"
    )
    
    if result.success:
        print(f"Video saved to: {result.video_path}")
"""

import os
from pathlib import Path
from typing import Optional, Callable

# Import client from runpod module
from runpod.runpod_client import LTX2ServerlessClient, LTX2Result, get_ltx2_client


# Valid camera motion options
VALID_CAMERA_MOTIONS = [
    "dolly_left",
    "dolly_right", 
    "dolly_in",
    "dolly_out",
    "jib_up",
    "static",
    "none"
]

# Recommended motions for 3D reconstruction
RECOMMENDED_FOR_3D = ["dolly_out", "dolly_left", "dolly_right"]


def run_ltx2_runpod(
    image_path: str,
    output_dir: str = "./outputs/ltx2",
    output_name: str = "ltx2_output",
    prompt: str = "",
    negative_prompt: str = "",
    camera_motion: str = "dolly_out",
    model_variant: str = "19b-dev-fp8",
    num_frames: int = 97,
    width: int = 768,
    height: int = 512,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    fps: int = 24,
    seed: Optional[int] = None,
    api_key: Optional[str] = None,
    endpoint_id: Optional[str] = None,
    s3_bucket: str = "arkrunr",
    s3_region: str = "us-west-1",
    poll_interval: int = 10,
    max_wait: int = 600,
    progress_callback: Optional[Callable] = None
) -> LTX2Result:
    """
    Generate video using LTX-2 on RunPod serverless.
    
    Args:
        image_path: Path to input image
        output_dir: Local directory for output video
        output_name: Name for output file (without extension)
        prompt: Text prompt describing desired video content
        negative_prompt: What to avoid in generation
        camera_motion: Camera LoRA to use (dolly_out recommended for 3D)
        num_frames: Number of frames (must be 8n+1, e.g., 97, 121, 145)
        width: Output width (must be divisible by 32)
        height: Output height (must be divisible by 32)
        num_inference_steps: Diffusion steps (50 default, 8 for distilled)
        guidance_scale: CFG scale (7.5 default)
        fps: Frames per second (24 default)
        seed: Random seed for reproducibility
        api_key: RunPod API key
        endpoint_id: RunPod endpoint ID for LTX-2
        s3_bucket: S3 bucket for file transfer
        s3_region: S3 region
        poll_interval: Seconds between status polls
        max_wait: Maximum wait time in seconds
        progress_callback: Optional callback(status_dict, elapsed_seconds)
        
    Returns:
        LTX2Result with video path and metadata
    """
    # Validate inputs
    if not os.path.exists(image_path):
        return LTX2Result(
            success=False,
            error=f"Input image not found: {image_path}"
        )
    
    if camera_motion not in VALID_CAMERA_MOTIONS:
        return LTX2Result(
            success=False,
            error=f"Invalid camera_motion '{camera_motion}'. Valid options: {VALID_CAMERA_MOTIONS}"
        )
    
    if not api_key:
        return LTX2Result(
            success=False,
            error="RunPod API key is required"
        )
    
    if not endpoint_id:
        return LTX2Result(
            success=False,
            error="RunPod endpoint ID is required"
        )
    
    # Validate dimensions
    if width % 32 != 0:
        width = (width // 32) * 32
    if height % 32 != 0:
        height = (height // 32) * 32
    
    # Validate num_frames (must be 8n+1)
    if (num_frames - 1) % 8 != 0:
        num_frames = ((num_frames - 1) // 8) * 8 + 1
    
    # Create client
    client = LTX2ServerlessClient(
        endpoint_id=endpoint_id,
        api_key=api_key
    )
    
    # Generate video
    return client.generate_sync(
        image_path=image_path,
        output_dir=output_dir,
        output_name=output_name,
        prompt=prompt,
        negative_prompt=negative_prompt,
        camera_motion=camera_motion,
        model_variant=model_variant,
        num_frames=num_frames,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        fps=fps,
        seed=seed,
        s3_bucket=s3_bucket,
        s3_region=s3_region,
        poll_interval=poll_interval,
        max_wait=max_wait,
        progress_callback=progress_callback
    )


def generate_video_for_3d(
    image_path: str,
    output_dir: str,
    api_key: str,
    endpoint_id: str,
    camera_motion: str = "dolly_out",
    num_frames: int = 121,
    **kwargs
) -> LTX2Result:
    """
    Generate video optimized for 3D reconstruction pipeline.
    
    Uses settings tuned for feeding into 2DGS or similar mesh reconstruction.
    
    Args:
        image_path: Path to input image
        output_dir: Directory for output video
        api_key: RunPod API key
        endpoint_id: LTX-2 endpoint ID
        camera_motion: Camera movement (dolly_out recommended)
        num_frames: More frames = better reconstruction (121+ recommended)
        **kwargs: Additional arguments passed to run_ltx2_runpod()
        
    Returns:
        LTX2Result with video path
    """
    # Default prompt for 3D-friendly generation
    default_prompt = kwargs.pop("prompt", "Smooth camera movement revealing object details")
    
    return run_ltx2_runpod(
        image_path=image_path,
        output_dir=output_dir,
        prompt=default_prompt,
        camera_motion=camera_motion,
        num_frames=num_frames,
        # Lower resolution for faster processing in 2DGS
        width=kwargs.pop("width", 768),
        height=kwargs.pop("height", 512),
        api_key=api_key,
        endpoint_id=endpoint_id,
        **kwargs
    )


# Convenience function for quick generation
def quick_generate(
    image_path: str,
    output_dir: str,
    camera_motion: str = "dolly_out",
    api_key: str = "",
    endpoint_id: str = "",
    **kwargs
) -> LTX2Result:
    """
    Quick video generation with sensible defaults.
    
    Args:
        image_path: Path to input image
        output_dir: Directory for output video
        camera_motion: Camera movement type
        api_key: RunPod API key
        endpoint_id: LTX-2 endpoint ID
        **kwargs: Additional overrides
        
    Returns:
        LTX2Result with video path
    """
    output_name = kwargs.pop("output_name", Path(image_path).stem + f"_{camera_motion}")
    
    return run_ltx2_runpod(
        image_path=image_path,
        output_dir=output_dir,
        output_name=output_name,
        camera_motion=camera_motion,
        api_key=api_key,
        endpoint_id=endpoint_id,
        **kwargs
    )
