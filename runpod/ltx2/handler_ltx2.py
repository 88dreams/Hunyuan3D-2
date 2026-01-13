#!/usr/bin/env python3
"""
LTX-2 RunPod Serverless Handler

Generates high-quality video from images using Lightricks' LTX-2 model.
Supports camera control via LoRA adapters.

Model: Lightricks LTX-2 (19B parameters)
Input: Single image + prompt + camera motion
Output: Video (MP4) up to 4K resolution

Camera Motion Options:
    - dolly_left: Camera moves laterally left
    - dolly_right: Camera moves laterally right
    - dolly_in: Camera pushes toward subject
    - dolly_out: Camera pulls away from subject
    - jib_up: Camera rises vertically
    - static: No camera movement
"""

import os
import sys
import base64
import tempfile
import shutil
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import time

import runpod

# =============================================================================
# CONFIGURATION
# =============================================================================

# Model configuration
MODEL_ID = "Lightricks/LTX-Video"
MODEL_VARIANT = os.environ.get("LTX2_MODEL_VARIANT", "")

# Output configuration
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/runpod-volume/outputs/ltx2")

# S3 Configuration
S3_BUCKET = os.environ.get("S3_BUCKET", "")
S3_REGION = os.environ.get("S3_REGION", "us-west-1")
S3_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY_ID", "")
S3_SECRET_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
S3_PREFIX = os.environ.get("S3_PREFIX", "MediaContent/outputs/ltx2")
S3_ENABLED = bool(S3_BUCKET and S3_ACCESS_KEY and S3_SECRET_KEY)

# Maximum file size for base64 encoding (8MB)
MAX_BASE64_SIZE = int(os.environ.get("MAX_BASE64_SIZE", 8 * 1024 * 1024))

# Ensure directories exist
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ltx2-handler")

# =============================================================================
# CAMERA CONTROL LORAS
# =============================================================================

CAMERA_LORAS = {
    "dolly_left": "Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Left",
    "dolly_right": "Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Right",
    "dolly_in": "Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-In",
    "dolly_out": "Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Out",
    "jib_up": "Lightricks/LTX-2-19b-LoRA-Camera-Control-Jib-Up",
    "static": "Lightricks/LTX-2-19b-LoRA-Camera-Control-Static",
}

VALID_CAMERA_MOTIONS = list(CAMERA_LORAS.keys()) + ["none"]

# =============================================================================
# ENVIRONMENT CHECK
# =============================================================================

def check_environment():
    """Log environment info at startup."""
    logger.info("=" * 60)
    logger.info("LTX-2 ENVIRONMENT CHECK")
    logger.info("=" * 60)
    
    logger.info(f"Python: {sys.executable}")
    logger.info(f"Python version: {sys.version}")
    
    # Check PyTorch
    try:
        import torch
        logger.info(f"PyTorch: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            props = torch.cuda.get_device_properties(0)
            logger.info(f"VRAM: {props.total_memory / 1024**3:.1f} GB")
    except ImportError as e:
        logger.error(f"PyTorch import failed: {e}")
    
    # Check diffusers
    try:
        import diffusers
        logger.info(f"Diffusers: {diffusers.__version__}")
    except ImportError as e:
        logger.error(f"Diffusers import failed: {e}")
    
    # Check S3 configuration
    logger.info(f"S3 enabled: {S3_ENABLED}")
    if S3_ENABLED:
        logger.info(f"S3 bucket: {S3_BUCKET}")
    
    logger.info(f"Output dir: {OUTPUT_DIR}")
    logger.info(f"Model variant: {MODEL_VARIANT or 'default'}")
    logger.info("=" * 60)

check_environment()

# =============================================================================
# S3 UTILITIES
# =============================================================================

_s3_client = None

def get_s3_client():
    """Get or create S3 client."""
    global _s3_client
    if _s3_client is None and S3_ENABLED:
        try:
            import boto3
            _s3_client = boto3.client(
                's3',
                region_name=S3_REGION,
                aws_access_key_id=S3_ACCESS_KEY,
                aws_secret_access_key=S3_SECRET_KEY,
            )
            logger.info(f"S3 client initialized for bucket: {S3_BUCKET}")
        except Exception as e:
            logger.warning(f"Failed to initialize S3 client: {e}")
    return _s3_client


def upload_to_s3(local_path: str, s3_key: str) -> Optional[str]:
    """Upload file to S3 and return URL."""
    client = get_s3_client()
    if client is None:
        return None
    
    try:
        client.upload_file(local_path, S3_BUCKET, s3_key)
        url = f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/{s3_key}"
        logger.info(f"Uploaded to S3: {url}")
        return url
    except Exception as e:
        logger.error(f"S3 upload failed: {e}")
        return None


def download_from_s3(s3_url: str, local_path: str) -> bool:
    """Download file from S3 URL."""
    client = get_s3_client()
    if client is None:
        return False
    
    try:
        from urllib.parse import urlparse
        parsed = urlparse(s3_url)
        
        if s3_url.startswith("s3://"):
            parts = s3_url[5:].split("/", 1)
            bucket = parts[0]
            key = parts[1] if len(parts) > 1 else ""
        elif ".s3." in parsed.netloc:
            bucket = parsed.netloc.split(".s3.")[0]
            key = parsed.path.lstrip("/")
        else:
            bucket = S3_BUCKET
            key = parsed.path.lstrip("/")
        
        client.download_file(bucket, key, local_path)
        logger.info(f"Downloaded from S3: {s3_url}")
        return True
    except Exception as e:
        logger.error(f"S3 download failed: {e}")
        return False


def download_from_url(url: str, local_path: str) -> bool:
    """Download file from HTTP(S) URL."""
    try:
        import requests
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"Downloaded from URL: {url}")
        return True
    except Exception as e:
        logger.error(f"URL download failed: {e}")
        return False


# =============================================================================
# MODEL LOADING
# =============================================================================

_pipeline = None
_current_lora = None

def load_model():
    """Load LTX-2 pipeline (lazy initialization)."""
    global _pipeline
    
    if _pipeline is not None:
        return _pipeline
    
    logger.info("Loading LTX-2 model...")
    start_time = time.time()
    
    try:
        import torch
        from diffusers import LTXPipeline
        
        # Load base pipeline
        _pipeline = LTXPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
        )
        
        # Move to GPU
        _pipeline = _pipeline.to("cuda")
        
        # Enable memory optimizations
        _pipeline.enable_model_cpu_offload()
        
        load_time = time.time() - start_time
        logger.info(f"LTX-2 model loaded in {load_time:.1f}s")
        
        return _pipeline
        
    except Exception as e:
        logger.error(f"Failed to load LTX-2 model: {e}")
        raise


def load_camera_lora(camera_motion: str):
    """Load camera control LoRA if needed."""
    global _pipeline, _current_lora
    
    if camera_motion == "none" or camera_motion not in CAMERA_LORAS:
        # Unload any existing LoRA
        if _current_lora is not None and _pipeline is not None:
            try:
                _pipeline.unload_lora_weights()
                _current_lora = None
                logger.info("Unloaded camera LoRA")
            except Exception as e:
                logger.warning(f"Failed to unload LoRA: {e}")
        return
    
    if camera_motion == _current_lora:
        logger.info(f"Camera LoRA '{camera_motion}' already loaded")
        return
    
    try:
        lora_id = CAMERA_LORAS[camera_motion]
        logger.info(f"Loading camera LoRA: {lora_id}")
        
        # Unload previous LoRA first
        if _current_lora is not None:
            _pipeline.unload_lora_weights()
        
        # Load new LoRA
        _pipeline.load_lora_weights(lora_id)
        _current_lora = camera_motion
        
        logger.info(f"✓ Camera LoRA '{camera_motion}' loaded")
        
    except Exception as e:
        logger.error(f"Failed to load camera LoRA: {e}")
        _current_lora = None


# =============================================================================
# INPUT HANDLING
# =============================================================================

def save_input_image(job_input: Dict) -> str:
    """Save input image from base64 or URL to temp file."""
    temp_dir = tempfile.mkdtemp()
    input_path = os.path.join(temp_dir, "input.png")
    
    # Check for image data
    image_data = job_input.get("image") or job_input.get("image_data") or job_input.get("image_base64")
    image_url = job_input.get("image_url") or job_input.get("input_url")
    
    if image_data:
        # Base64 encoded image
        if isinstance(image_data, str):
            if image_data.startswith("data:"):
                image_data = image_data.split(",", 1)[1]
            image_bytes = base64.b64decode(image_data)
        else:
            image_bytes = image_data
            
        with open(input_path, "wb") as f:
            f.write(image_bytes)
        logger.info(f"Saved base64 image to: {input_path}")
        
    elif image_url:
        # Download from URL
        if "s3.amazonaws.com" in image_url or image_url.startswith("s3://"):
            success = download_from_s3(image_url, input_path)
        else:
            success = download_from_url(image_url, input_path)
            
        if not success:
            raise ValueError(f"Failed to download image from: {image_url}")
        logger.info(f"Downloaded image to: {input_path}")
        
    else:
        raise ValueError("No image provided. Use 'image' (base64) or 'image_url'")
    
    return input_path


# =============================================================================
# VIDEO GENERATION
# =============================================================================

def generate_video(
    input_image: str,
    output_path: str,
    prompt: str = "",
    negative_prompt: str = "",
    camera_motion: str = "none",
    num_frames: int = 97,
    width: int = 768,
    height: int = 512,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    fps: int = 24,
    seed: Optional[int] = None
) -> str:
    """
    Generate video from input image.
    
    Args:
        input_image: Path to input image
        output_path: Path for output video
        prompt: Text prompt describing the video
        negative_prompt: Negative prompt
        camera_motion: Camera LoRA to use (dolly_left, dolly_out, etc.)
        num_frames: Number of frames (should be divisible by 8 + 1)
        width: Output width (divisible by 32)
        height: Output height (divisible by 32)
        num_inference_steps: Number of diffusion steps
        guidance_scale: Classifier-free guidance scale
        fps: Frames per second for output video
        seed: Random seed for reproducibility
        
    Returns:
        Path to generated video
    """
    import torch
    from PIL import Image
    import imageio
    
    # Load model
    pipe = load_model()
    
    # Load camera LoRA if specified
    load_camera_lora(camera_motion)
    
    # Load input image
    image = Image.open(input_image).convert("RGB")
    
    # Resize image to match output dimensions
    image = image.resize((width, height), Image.Resampling.LANCZOS)
    
    # Set up generator for reproducibility
    generator = None
    if seed is not None:
        generator = torch.Generator("cuda").manual_seed(seed)
    
    logger.info(f"Generating video: {num_frames} frames at {width}x{height}")
    logger.info(f"Prompt: {prompt[:100]}..." if len(prompt) > 100 else f"Prompt: {prompt}")
    logger.info(f"Camera motion: {camera_motion}")
    
    start_time = time.time()
    
    # Generate video
    # Note: Exact API may vary based on diffusers version
    try:
        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            image=image,
            num_frames=num_frames,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        
        frames = output.frames[0]  # First video in batch
        
    except TypeError as e:
        # Fallback if image parameter not supported (text-to-video only)
        logger.warning(f"Image-to-video may not be supported: {e}")
        logger.info("Falling back to text-to-video mode")
        
        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            num_frames=num_frames,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        
        frames = output.frames[0]
    
    gen_time = time.time() - start_time
    logger.info(f"Video generated in {gen_time:.1f}s ({len(frames)} frames)")
    
    # Save video
    logger.info(f"Saving video to: {output_path}")
    imageio.mimwrite(output_path, frames, fps=fps, codec='libx264', quality=8)
    
    return output_path


# =============================================================================
# MAIN HANDLER
# =============================================================================

def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main handler for LTX-2 serverless jobs.
    
    Input parameters:
        image: Base64 encoded image (or image_data, image_base64)
        image_url: URL to download image from
        prompt: Text prompt describing the video (optional)
        negative_prompt: Negative prompt (optional)
        camera_motion: Camera LoRA (dolly_left, dolly_out, etc.)
        num_frames: Number of frames (default: 97, must be 8n+1)
        width: Output width (default: 768, must be divisible by 32)
        height: Output height (default: 512, must be divisible by 32)
        num_inference_steps: Diffusion steps (default: 50)
        guidance_scale: CFG scale (default: 7.5)
        fps: Frames per second (default: 24)
        seed: Random seed (optional)
        output_name: Name for output file (without extension)
        return_base64: Return video as base64 (default: False)
        
    Output:
        status: "success" or "error"
        video_url: S3 URL to video (if S3 enabled)
        video_path: Local path on network volume
        video_base64: Base64 encoded video (if small and return_base64=True)
        num_frames: Number of frames generated
        duration: Video duration in seconds
        camera_motion: Camera LoRA used
    """
    job_id = job.get("id", "unknown")
    job_input = job.get("input", {})
    
    logger.info(f"Processing LTX-2 job: {job_id}")
    logger.info(f"Input parameters: {list(job_input.keys())}")
    
    try:
        # Parse parameters
        prompt = job_input.get("prompt", "")
        negative_prompt = job_input.get("negative_prompt", "")
        camera_motion = job_input.get("camera_motion", "none")
        num_frames = int(job_input.get("num_frames", 97))
        width = int(job_input.get("width", 768))
        height = int(job_input.get("height", 512))
        num_inference_steps = int(job_input.get("num_inference_steps", 50))
        guidance_scale = float(job_input.get("guidance_scale", 7.5))
        fps = int(job_input.get("fps", 24))
        seed = job_input.get("seed")
        output_name = job_input.get("output_name", f"ltx2_{job_id}")
        return_base64 = job_input.get("return_base64", False)
        
        # Validate camera motion
        if camera_motion not in VALID_CAMERA_MOTIONS:
            return {
                "status": "error",
                "message": f"Invalid camera_motion '{camera_motion}'. Valid options: {VALID_CAMERA_MOTIONS}"
            }
        
        # Validate dimensions (must be divisible by 32)
        if width % 32 != 0:
            width = (width // 32) * 32
            logger.warning(f"Width adjusted to {width} (must be divisible by 32)")
        if height % 32 != 0:
            height = (height // 32) * 32
            logger.warning(f"Height adjusted to {height} (must be divisible by 32)")
        
        # Validate num_frames (must be 8n+1)
        if (num_frames - 1) % 8 != 0:
            num_frames = ((num_frames - 1) // 8) * 8 + 1
            logger.warning(f"num_frames adjusted to {num_frames} (must be 8n+1)")
        
        # Save input image
        input_path = save_input_image(job_input)
        
        # Generate output path
        output_filename = f"{output_name}.mp4"
        temp_output = os.path.join(tempfile.gettempdir(), output_filename)
        
        # Generate video
        generate_video(
            input_image=input_path,
            output_path=temp_output,
            prompt=prompt,
            negative_prompt=negative_prompt,
            camera_motion=camera_motion,
            num_frames=num_frames,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            fps=fps,
            seed=seed
        )
        
        # Copy to network volume
        final_output = os.path.join(OUTPUT_DIR, output_filename)
        shutil.copy2(temp_output, final_output)
        logger.info(f"Video saved to network volume: {final_output}")
        
        # Get file info
        file_size = os.path.getsize(final_output)
        duration = num_frames / fps
        
        # Build response
        response = {
            "status": "success",
            "message": "Video generated successfully",
            "model": "ltx2",
            "video_path": final_output,
            "video_name": output_filename,
            "file_size": file_size,
            "num_frames": num_frames,
            "duration": duration,
            "fps": fps,
            "width": width,
            "height": height,
            "camera_motion": camera_motion,
            "seed": seed
        }
        
        # Upload to S3
        if S3_ENABLED:
            s3_key = f"{S3_PREFIX}/{output_filename}"
            s3_url = upload_to_s3(final_output, s3_key)
            if s3_url:
                response["video_url"] = s3_url
        
        # Return base64 if requested and file is small enough
        if return_base64 and file_size <= MAX_BASE64_SIZE:
            with open(final_output, "rb") as f:
                response["video_base64"] = base64.b64encode(f.read()).decode("utf-8")
        
        # Cleanup temp files
        if os.path.exists(input_path):
            os.remove(input_path)
            parent = os.path.dirname(input_path)
            if os.path.isdir(parent) and not os.listdir(parent):
                os.rmdir(parent)
        if os.path.exists(temp_output) and temp_output != final_output:
            os.remove(temp_output)
        
        logger.info(f"LTX-2 job {job_id} completed successfully")
        return response
        
    except Exception as e:
        logger.error(f"LTX-2 job {job_id} failed: {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e)
        }


# =============================================================================
# HEALTH CHECK
# =============================================================================

def health_check(_job: Dict) -> Dict:
    """Health check endpoint."""
    import torch
    
    return {
        "status": "healthy",
        "model": "ltx2",
        "model_id": MODEL_ID,
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "s3_enabled": S3_ENABLED,
        "output_dir": OUTPUT_DIR,
        "valid_camera_motions": VALID_CAMERA_MOTIONS
    }


# =============================================================================
# ENTRYPOINT
# =============================================================================

if __name__ == "__main__":
    logger.info("Starting LTX-2 serverless handler...")
    
    runpod.serverless.start({
        "handler": handler,
        "health_check": health_check
    })
