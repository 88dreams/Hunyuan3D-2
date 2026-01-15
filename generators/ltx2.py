"""
LTX-2 Video Generation Module

Generates high-quality videos from images using Lightricks' LTX-2 API.

API Documentation: https://docs.ltx.video/welcome

Models:
    - ltx-2-fast: Faster generation, good quality
    - ltx-2-pro: Best quality, slower

Resolutions:
    - 1080p (1920x1080)
    - 1440p (2560x1440)
    - 4K (3840x2160)

Durations: 6, 8, 10 seconds (more options for fast model)
FPS: 25 or 50

Camera Motion:
    Camera movement is controlled via the prompt. Include descriptions like:
    - "Camera slowly pulls back from the subject" (dolly out)
    - "Camera moves left revealing the scene" (dolly left)
    - "Camera rises up showing the object from above" (jib up)
    - "Static camera, object rotates in place" (static)

Usage:
    from generators.ltx2 import run_ltx2_api
    
    result = run_ltx2_api(
        image_path="input.png",
        output_dir="./outputs/ltx2",
        prompt="Camera slowly orbits around the object, revealing all angles",
        api_key="your_ltx_api_key"
    )
    
    if result.success:
        print(f"Video saved to: {result.video_path}")
"""

import os
from pathlib import Path
from typing import Optional, Callable

# Import clients from runpod module
from runpod.runpod_client import LTXAPIClient, LTX2Result, get_ltx_api_client

# Also keep old imports for backwards compatibility
try:
    from runpod.runpod_client import LTX2ServerlessClient, get_ltx2_client
except ImportError:
    LTX2ServerlessClient = None
    get_ltx2_client = None


# Available models
MODELS = ["ltx-2-fast", "ltx-2-pro"]

# Available resolutions
RESOLUTIONS = ["1080p", "1440p", "4K"]

# Available durations (seconds)
DURATIONS = [6, 8, 10]

# Available FPS options
FPS_OPTIONS = [25, 50]

# Camera motion prompt suggestions (for UI hints)
CAMERA_MOTION_PROMPTS = {
    "dolly_out": "Camera slowly pulls back from the subject, revealing the full scene",
    "dolly_in": "Camera pushes forward toward the subject, focusing on details",
    "dolly_left": "Camera moves laterally to the left, revealing the side of the subject",
    "dolly_right": "Camera moves laterally to the right, showing another angle",
    "jib_up": "Camera rises vertically, showing the subject from above",
    "jib_down": "Camera lowers, revealing the subject from a lower angle",
    "orbit": "Camera orbits around the subject in a circular motion",
    "static": "Camera remains stationary, subject may rotate or animate"
}


def run_ltx2_api(
    image_path: str,
    output_dir: str = "./outputs/ltx2",
    output_name: str = "ltx2_output",
    prompt: str = "Camera slowly reveals the subject from multiple angles",
    model: str = "ltx-2-pro",
    resolution: str = "1080p",
    duration: int = 6,
    fps: int = 25,
    generate_audio: bool = False,
    api_key: Optional[str] = None,
    use_s3: bool = True,
    progress_callback: Optional[Callable] = None
) -> LTX2Result:
    """
    Generate video using the official LTX API.
    
    This is the recommended method for LTX-2 video generation.
    
    Args:
        image_path: Path to input image (PNG, JPEG, WEBP)
        output_dir: Local directory for output video
        output_name: Name for output file (without extension)
        prompt: Text description guiding animation (include camera motion description)
        model: "ltx-2-fast" or "ltx-2-pro"
        resolution: "1080p", "1440p", or "4K"
        duration: Video duration in seconds (6, 8, 10)
        fps: Frame rate (25 or 50)
        generate_audio: Whether to generate AI audio (default False)
        api_key: LTX API key (from https://ltx.video)
        use_s3: Upload image to S3 (True) or use base64 encoding (False)
        progress_callback: Optional callback(status_msg, elapsed_seconds)
        
    Returns:
        LTX2Result with video path and metadata
    """
    # Validate inputs
    if not os.path.exists(image_path):
        return LTX2Result(
            success=False,
            error=f"Input image not found: {image_path}"
        )
    
    if not api_key:
        return LTX2Result(
            success=False,
            error="LTX API key is required (get one at https://ltx.video)"
        )
    
    if model not in MODELS:
        return LTX2Result(
            success=False,
            error=f"Invalid model '{model}'. Choose from: {MODELS}"
        )
    
    if resolution not in RESOLUTIONS:
        return LTX2Result(
            success=False,
            error=f"Invalid resolution '{resolution}'. Choose from: {RESOLUTIONS}"
        )
    
    if duration not in DURATIONS:
        # Find closest valid duration
        duration = min(DURATIONS, key=lambda x: abs(x - duration))
    
    # Create client and generate
    client = LTXAPIClient(api_key=api_key)
    
    return client.generate_video(
        image_path=image_path,
        prompt=prompt,
        output_dir=output_dir,
        output_name=output_name,
        model=model,
        resolution=resolution,
        duration=duration,
        fps=fps,
        generate_audio=generate_audio,
        use_s3=use_s3,
        progress_callback=progress_callback
    )


def generate_video_for_3d(
    image_path: str,
    output_dir: str,
    api_key: str,
    camera_motion: str = "dolly_out",
    model: str = "ltx-2-pro",
    resolution: str = "1080p",
    duration: int = 10,
    **kwargs
) -> LTX2Result:
    """
    Generate video optimized for 3D reconstruction pipeline.
    
    Uses settings tuned for feeding into 2DGS or similar mesh reconstruction.
    Automatically generates a camera-motion-aware prompt.
    
    Args:
        image_path: Path to input image
        output_dir: Directory for output video
        api_key: LTX API key
        camera_motion: Desired camera movement type (used to generate prompt)
        model: "ltx-2-fast" or "ltx-2-pro"
        resolution: Output resolution
        duration: Video duration in seconds (10 recommended for 3D)
        **kwargs: Additional arguments passed to run_ltx2_api()
        
    Returns:
        LTX2Result with video path
    """
    # Get camera motion prompt suggestion
    base_prompt = CAMERA_MOTION_PROMPTS.get(
        camera_motion, 
        "Camera slowly reveals the subject from multiple angles"
    )
    
    # Enhance prompt for 3D reconstruction
    prompt = kwargs.pop("prompt", None)
    if not prompt:
        prompt = f"{base_prompt}. Smooth continuous motion, sharp focus, clear lighting, detailed textures."
    
    return run_ltx2_api(
        image_path=image_path,
        output_dir=output_dir,
        prompt=prompt,
        model=model,
        resolution=resolution,
        duration=duration,
        api_key=api_key,
        generate_audio=False,  # No audio needed for 3D
        **kwargs
    )


def text_to_video(
    prompt: str,
    output_dir: str = "./outputs/ltx2",
    output_name: str = "ltx2_text",
    model: str = "ltx-2-pro",
    resolution: str = "1080p",
    duration: int = 6,
    fps: int = 25,
    generate_audio: bool = False,
    api_key: Optional[str] = None,
    progress_callback: Optional[Callable] = None
) -> LTX2Result:
    """
    Generate video from text prompt only (no input image).
    
    Args:
        prompt: Text description of desired video
        output_dir: Directory for output video
        output_name: Name for output file (without extension)
        model: "ltx-2-fast" or "ltx-2-pro"
        resolution: "1080p", "1440p", or "4K"
        duration: Video duration in seconds
        fps: Frame rate (25 or 50)
        generate_audio: Whether to generate AI audio
        api_key: LTX API key
        progress_callback: Optional callback
        
    Returns:
        LTX2Result with video path
    """
    if not api_key:
        return LTX2Result(
            success=False,
            error="LTX API key is required"
        )
    
    client = LTXAPIClient(api_key=api_key)
    
    return client.text_to_video(
        prompt=prompt,
        output_dir=output_dir,
        output_name=output_name,
        model=model,
        resolution=resolution,
        duration=duration,
        fps=fps,
        generate_audio=generate_audio,
        progress_callback=progress_callback
    )


# Convenience function for quick generation
def quick_generate(
    image_path: str,
    output_dir: str,
    camera_motion: str = "dolly_out",
    api_key: str = "",
    model: str = "ltx-2-pro",
    **kwargs
) -> LTX2Result:
    """
    Quick video generation with sensible defaults.
    
    Args:
        image_path: Path to input image
        output_dir: Directory for output video
        camera_motion: Camera movement type (used to generate prompt)
        api_key: LTX API key
        model: Model to use
        **kwargs: Additional overrides
        
    Returns:
        LTX2Result with video path
    """
    output_name = kwargs.pop("output_name", Path(image_path).stem + f"_{camera_motion}")
    
    return generate_video_for_3d(
        image_path=image_path,
        output_dir=output_dir,
        output_name=output_name,
        camera_motion=camera_motion,
        api_key=api_key,
        model=model,
        **kwargs
    )


# =============================================================================
# LEGACY SUPPORT (RunPod Serverless - kept for backwards compatibility)
# =============================================================================

# Valid camera motion options (legacy)
VALID_CAMERA_MOTIONS = [
    "dolly_left", "dolly_right", "dolly_in", "dolly_out",
    "jib_up", "static", "none"
]

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
    LEGACY: Generate video using LTX-2 on RunPod serverless.
    
    NOTE: This function is deprecated. Use run_ltx2_api() instead for better
    quality and simpler setup.
    
    Args:
        image_path: Path to input image
        output_dir: Local directory for output video
        output_name: Name for output file (without extension)
        prompt: Text prompt describing desired video content
        negative_prompt: What to avoid in generation
        camera_motion: Camera LoRA to use (dolly_out recommended for 3D)
        ... (other legacy parameters)
        
    Returns:
        LTX2Result with video path and metadata
    """
    if LTX2ServerlessClient is None:
        return LTX2Result(
            success=False,
            error="RunPod LTX-2 client not available. Use run_ltx2_api() instead."
        )
    
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
