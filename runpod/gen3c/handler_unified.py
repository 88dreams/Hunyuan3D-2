#!/usr/bin/env python3
"""
Unified 3D Generation RunPod Serverless Handler

This handler supports multiple models:
- GEN3C: Image to video generation (NVIDIA Cosmos)
- SHARP: Image to 3D Gaussian Splatting (Apple)
- Lyra: Image/Video to 3D/4D Gaussian Splatting (NVIDIA)
- TRELLIS.2: Image to 3D with O-Voxel (Microsoft)

Environment:
    Python 3.10 + NumPy 1.26.4 + PyTorch 2.6.0 (NVIDIA stack)

Usage:
    RunPod automatically calls this handler when a job is submitted
    to the serverless endpoint.
"""

import os
import sys
import base64
import tempfile
import subprocess
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
import logging

import runpod

# =============================================================================
# CONFIGURATION
# =============================================================================

GEN3C_DIR = os.environ.get("GEN3C_DIR", "/workspace/GEN3C")
SHARP_DIR = os.environ.get("SHARP_DIR", "/workspace/ml-sharp")
LYRA_DIR = os.environ.get("LYRA_DIR", "/workspace/lyra")
TRELLIS_DIR = os.environ.get("TRELLIS_DIR", "/workspace/TRELLIS2")
SUGAR_DIR = os.environ.get("SUGAR_DIR", "/workspace/SuGaR")
CHECKPOINT_DIR = os.environ.get("GEN3C_CHECKPOINT_DIR", "/workspace/checkpoints")
SHARP_CHECKPOINT = os.environ.get("SHARP_CHECKPOINT", "/workspace/checkpoints/sharp/sharp_2572gikvuh.pt")
TRELLIS_CHECKPOINT = os.environ.get("TRELLIS_CHECKPOINT", "/workspace/checkpoints/trellis")
# Use network volume for persistent storage (survives worker restarts)
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/runpod-volume/outputs")

# Maximum file size for base64 encoding (8MB) - larger files saved to volume only
MAX_BASE64_SIZE = int(os.environ.get("MAX_BASE64_SIZE", 8 * 1024 * 1024))

# S3 Configuration (optional - for uploading large files)
S3_BUCKET = os.environ.get("S3_BUCKET", "")
S3_REGION = os.environ.get("S3_REGION", "us-east-1")
S3_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY_ID", "")
S3_SECRET_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
S3_PREFIX = os.environ.get("S3_PREFIX", "MediaContent/outputs")  # Path prefix in bucket
S3_ENABLED = bool(S3_BUCKET and S3_ACCESS_KEY and S3_SECRET_KEY)

# Ensure directories exist
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("3dgen-handler")

# =============================================================================
# STARTUP ENVIRONMENT CHECKS
# =============================================================================
def check_environment():
    """Log environment info and check critical dependencies at startup."""
    logger.info("=" * 60)
    logger.info("ENVIRONMENT CHECK")
    logger.info("=" * 60)
    
    # Python and conda info
    logger.info(f"Python: {sys.executable}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"CONDA_PREFIX: {os.environ.get('CONDA_PREFIX', 'NOT SET')}")
    logger.info(f"CONDA_DEFAULT_ENV: {os.environ.get('CONDA_DEFAULT_ENV', 'NOT SET')}")
    
    # PATH (first few entries)
    path_entries = os.environ.get('PATH', '').split(':')[:5]
    logger.info(f"PATH (first 5): {path_entries}")
    
    # Check critical packages
    try:
        import torch
        logger.info(f"PyTorch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        logger.error(f"PyTorch import failed: {e}")
    
    # Check gsplat (required for SHARP video rendering)
    try:
        import gsplat
        logger.info(f"gsplat: {gsplat.__version__} - SHARP video rendering AVAILABLE")
    except ImportError as e:
        logger.warning(f"gsplat NOT available: {e}")
        logger.warning("SHARP video rendering will NOT work!")
    except Exception as e:
        logger.warning(f"gsplat check failed: {e}")
    
    # Check sharp CLI
    sharp_path = shutil.which("sharp")
    if sharp_path:
        logger.info(f"sharp CLI: {sharp_path}")
    else:
        logger.warning("sharp CLI not found in PATH")
    
    logger.info("=" * 60)

# Run environment check at module load
check_environment()

# S3 client (lazy initialization)
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
        except ImportError:
            logger.warning("boto3 not installed - S3 upload disabled")
        except Exception as e:
            logger.warning(f"Failed to initialize S3 client: {e}")
    return _s3_client


def upload_to_s3(local_path: str, s3_key: str) -> Optional[str]:
    """
    Upload a file to S3.
    
    Args:
        local_path: Path to local file
        s3_key: S3 object key (path in bucket)
    
    Returns:
        S3 URL if successful, None otherwise
    """
    client = get_s3_client()
    if client is None:
        return None
    
    try:
        file_size = os.path.getsize(local_path)
        logger.info(f"Uploading to S3: {local_path} -> s3://{S3_BUCKET}/{s3_key} ({file_size / 1024 / 1024:.1f}MB)")
        
        client.upload_file(local_path, S3_BUCKET, s3_key)
        
        # Generate URL (public or presigned based on bucket settings)
        s3_url = f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/{s3_key}"
        logger.info(f"S3 upload complete: {s3_url}")
        return s3_url
    except Exception as e:
        logger.error(f"S3 upload failed: {e}")
        return None


def upload_file_to_s3_if_large(file_path: str, model: str, filename: str) -> Dict[str, Any]:
    """
    Upload file to S3 if it's too large for base64 encoding.
    
    Args:
        file_path: Path to local file
        model: Model name (sharp, gen3c, lyra, trellis)
        filename: Output filename
    
    Returns:
        Dict with upload info (s3_url, file_size, uploaded)
    """
    file_size = os.path.getsize(file_path)
    result = {
        "file_size": file_size,
        "uploaded": False,
        "s3_url": None,
        "s3_key": None,
    }
    
    logger.info(f"S3 upload check: file_size={file_size / 1024 / 1024:.1f}MB, MAX={MAX_BASE64_SIZE / 1024 / 1024:.1f}MB, S3_ENABLED={S3_ENABLED}")
    
    # Only upload to S3 if file is too large for base64
    if file_size > MAX_BASE64_SIZE and S3_ENABLED:
        # Use configured prefix (e.g., MediaContent/outputs/sharp/file.ply)
        s3_key = f"{S3_PREFIX}/{model}/{filename}"
        s3_url = upload_to_s3(file_path, s3_key)
        if s3_url:
            result["uploaded"] = True
            result["s3_url"] = s3_url
            result["s3_key"] = s3_key
    elif file_size > MAX_BASE64_SIZE and not S3_ENABLED:
        logger.warning(f"S3 NOT enabled - large file cannot be transferred. S3_BUCKET={S3_BUCKET}, S3_ACCESS_KEY={'set' if S3_ACCESS_KEY else 'NOT SET'}, S3_SECRET_KEY={'set' if S3_SECRET_KEY else 'NOT SET'}")
    
    return result


def upload_file_to_s3_always(file_path: str, model: str, filename: str) -> Dict[str, Any]:
    """
    Always upload file to S3 (regardless of size).
    This ensures files are always available for download even if the API response times out.
    
    Args:
        file_path: Path to local file
        model: Model name (sharp, gen3c, lyra, trellis)
        filename: Output filename
    
    Returns:
        Dict with upload info (s3_url, file_size, uploaded)
    """
    file_size = os.path.getsize(file_path)
    result = {
        "file_size": file_size,
        "uploaded": False,
        "s3_url": None,
        "s3_key": None,
    }
    
    if not S3_ENABLED:
        logger.warning(f"S3 NOT enabled - cannot upload {filename}. Configure S3_BUCKET, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY")
        return result
    
    logger.info(f"S3 upload (always): {filename} ({file_size / 1024 / 1024:.1f}MB)")
    
    # Use configured prefix (e.g., MediaContent/outputs/sharp/file.ply)
    s3_key = f"{S3_PREFIX}/{model}/{filename}"
    s3_url = upload_to_s3(file_path, s3_key)
    if s3_url:
        result["uploaded"] = True
        result["s3_url"] = s3_url
        result["s3_key"] = s3_key
        logger.info(f"S3 upload complete: {s3_url}")
    else:
        logger.error(f"S3 upload failed for {filename}")
    
    return result


def encode_file_if_small(file_path: str, max_size: int = MAX_BASE64_SIZE) -> Optional[str]:
    """
    Encode file as base64 if it's under the size limit.
    
    Args:
        file_path: Path to the file
        max_size: Maximum file size in bytes for base64 encoding
        
    Returns:
        Base64 encoded string if file is under limit, None otherwise
    """
    try:
        file_size = os.path.getsize(file_path)
        if file_size <= max_size:
            with open(file_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        else:
            logger.info(f"File too large for base64 ({file_size / 1024 / 1024:.1f}MB > {max_size / 1024 / 1024:.1f}MB): {file_path}")
            return None
    except Exception as e:
        logger.error(f"Error encoding file: {e}")
        return None

# =============================================================================
# VERSION QUERY
# =============================================================================

def get_git_version(repo_dir: str) -> Dict[str, str]:
    """
    Get git commit information for a repository.
    
    Args:
        repo_dir: Path to the git repository
        
    Returns:
        Dict with commit_sha, commit_date, branch
    """
    result = {
        "commit_sha": "unknown",
        "commit_date": "unknown",
        "branch": "unknown",
    }
    
    if not os.path.exists(repo_dir):
        result["error"] = f"Directory not found: {repo_dir}"
        return result
    
    try:
        # Get commit SHA (short)
        sha_result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=5
        )
        if sha_result.returncode == 0:
            result["commit_sha"] = sha_result.stdout.strip()
        
        # Get commit date
        date_result = subprocess.run(
            ["git", "log", "-1", "--format=%ci"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=5
        )
        if date_result.returncode == 0:
            # Parse date and format as YYYY-MM-DD
            full_date = date_result.stdout.strip()
            result["commit_date"] = full_date[:10] if full_date else "unknown"
        
        # Get current branch
        branch_result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=5
        )
        if branch_result.returncode == 0:
            result["branch"] = branch_result.stdout.strip()
            
    except subprocess.TimeoutExpired:
        result["error"] = "Git command timed out"
    except Exception as e:
        result["error"] = str(e)
    
    return result


def get_all_versions() -> Dict[str, Any]:
    """
    Get version information for all installed models.
    
    Returns:
        Dict with version info for each model
    """
    versions = {
        "sharp": get_git_version(SHARP_DIR),
        "gen3c": get_git_version(GEN3C_DIR),
        "lyra": get_git_version(LYRA_DIR),
        "trellis": get_git_version(TRELLIS_DIR),
    }
    
    # Format for display
    for model, info in versions.items():
        if info.get("commit_sha") != "unknown" and info.get("commit_date") != "unknown":
            info["display"] = f"{info['commit_sha']} ({info['commit_date']})"
        elif info.get("commit_sha") != "unknown":
            info["display"] = info["commit_sha"]
        else:
            info["display"] = info.get("error", "Not installed")
    
    return versions


# =============================================================================
# MODEL VALIDATION
# =============================================================================

_gen3c_validated = False
_sharp_validated = False


def validate_gen3c():
    """Validate GEN3C environment."""
    global _gen3c_validated
    if _gen3c_validated:
        return True
    
    script_path = Path(GEN3C_DIR) / "cosmos_predict1" / "diffusion" / "inference" / "gen3c_single_image.py"
    if not script_path.exists():
        logger.warning(f"GEN3C script not found: {script_path}")
        return False
    
    if not os.path.exists(CHECKPOINT_DIR):
        logger.warning(f"Checkpoint directory not found: {CHECKPOINT_DIR}")
        return False
    
    _gen3c_validated = True
    logger.info("GEN3C environment validated")
    return True


def validate_sharp():
    """Validate SHARP environment."""
    global _sharp_validated
    if _sharp_validated:
        return True
    
    # Check if 'sharp' CLI is available
    sharp_path = shutil.which("sharp")
    if not sharp_path:
        logger.warning("SHARP CLI not found in PATH")
        return False
    
    # Check if gsplat is available (needed for video rendering)
    try:
        import gsplat
        logger.info(f"gsplat version: {getattr(gsplat, '__version__', 'unknown')}")
    except ImportError as e:
        logger.warning(f"gsplat not available (video rendering will fail): {e}")
    except Exception as e:
        logger.warning(f"gsplat import error: {e}")
    
    _sharp_validated = True
    logger.info(f"SHARP environment validated (CLI: {sharp_path})")
    return True


# =============================================================================
# GEN3C INFERENCE
# =============================================================================

def run_gen3c(
    input_image_path: str,
    video_name: str,
    guidance: float = 1.0,
    num_frames: int = 121,
    trajectory: str = "left",
    foreground_masking: bool = True,
    seed: Optional[int] = None,
    movement_distance: float = 0.3,
    camera_rotation: str = "center_facing"
) -> str:
    """
    Run GEN3C inference.
    
    Args:
        input_image_path: Path to input image
        video_name: Name for output video
        guidance: Guidance scale (0.5-3.0)
        num_frames: Number of frames (121, 241, 361)
        trajectory: Camera trajectory (left, right, up, down, zoom_in, zoom_out, etc.)
        foreground_masking: Enable foreground masking
        seed: Random seed
        movement_distance: How far camera moves (0.1-1.0, default 0.3)
        camera_rotation: How camera rotates (center_facing, no_rotation, trajectory_aligned)
    
    Returns:
        Path to the generated video file
    """
    script_path = Path(GEN3C_DIR) / "cosmos_predict1" / "diffusion" / "inference" / "gen3c_single_image.py"
    
    cmd = [
        sys.executable,
        str(script_path),
        "--checkpoint_dir", CHECKPOINT_DIR,
        "--input_image_path", input_image_path,
        "--video_save_name", video_name,
        "--guidance", str(guidance),
        "--num_video_frames", str(num_frames),
        "--trajectory", trajectory,
        "--movement_distance", str(movement_distance),
        "--camera_rotation", camera_rotation,
        "--offload_diffusion_transformer",
        "--offload_tokenizer",
        "--offload_text_encoder_model",
        "--disable_prompt_upsampler",
        "--disable_guardrail",
    ]
    
    if foreground_masking:
        cmd.append("--foreground_masking")
    
    if seed is not None:
        cmd.extend(["--seed", str(seed)])
    
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{GEN3C_DIR}:{env.get('PYTHONPATH', '')}"
    env["CUDA_VISIBLE_DEVICES"] = "0"
    
    logger.info(f"Running GEN3C: {' '.join(cmd)}")
    
    result = subprocess.run(
        cmd,
        env=env,
        cwd=GEN3C_DIR,
        capture_output=True,
        text=True,
        timeout=7200  # 2 hour timeout
    )
    
    if result.returncode != 0:
        logger.error(f"GEN3C stderr: {result.stderr}")
        raise RuntimeError(f"GEN3C failed: {result.stderr[-2000:]}")
    
    # Find output
    output_video_1 = Path(GEN3C_DIR) / "videos" / f"{video_name}.mp4"
    output_video_2 = Path(GEN3C_DIR) / "outputs" / f"{video_name}.mp4"
    
    if output_video_1.exists():
        return str(output_video_1)
    elif output_video_2.exists():
        return str(output_video_2)
    else:
        raise RuntimeError(f"Output video not found at {output_video_1} or {output_video_2}")


# =============================================================================
# SHARP INFERENCE
# =============================================================================

def run_sharp(
    input_image_path: str,
    output_name: str,
    render_video: bool = False,
    trajectory_type: str = "rotate_forward",
    num_steps: int = 60,
    num_repeats: int = 1,
    max_disparity: float = 0.08,
    max_zoom: float = 0.15,
    lookat_mode: str = "point",
) -> Dict[str, str]:
    """
    Run SHARP inference with optional video rendering.
    
    Args:
        input_image_path: Path to input image
        output_name: Base name for output files
        render_video: Whether to render video trajectory
        trajectory_type: Camera trajectory type (rotate_forward, rotate, swipe, shake)
        num_steps: Number of frames in video
        num_repeats: Number of trajectory loops
        max_disparity: Maximum lateral camera offset
        max_zoom: Maximum forward camera movement
        lookat_mode: Camera focus mode (point, ahead)
    
    Returns:
        Dict with paths to generated files (ply_path, video_path if applicable)
    """
    # Create temp directories
    temp_input_dir = tempfile.mkdtemp(prefix="sharp_input_")
    temp_output_dir = tempfile.mkdtemp(prefix="sharp_output_")
    
    try:
        # Copy input image to temp directory (SHARP expects a directory)
        input_ext = os.path.splitext(input_image_path)[1]
        temp_input_path = os.path.join(temp_input_dir, f"input{input_ext}")
        shutil.copy2(input_image_path, temp_input_path)
        
        # Build command - use pre-downloaded checkpoint if available
        cmd = [
            "sharp", "predict",
            "-i", temp_input_dir,
            "-o", temp_output_dir,
        ]
        
        # Use pre-downloaded checkpoint to avoid runtime download
        if os.path.exists(SHARP_CHECKPOINT):
            cmd.extend(["-c", SHARP_CHECKPOINT])
            logger.info(f"Using pre-downloaded checkpoint: {SHARP_CHECKPOINT}")
        else:
            logger.warning(f"Checkpoint not found at {SHARP_CHECKPOINT}, SHARP will download on first run")
        
        if render_video:
            cmd.append("--render")
            logger.info(f"Video trajectory params: type={trajectory_type}, steps={num_steps}, "
                       f"repeats={num_repeats}, disparity={max_disparity}, zoom={max_zoom}, lookat={lookat_mode}")
            # Note: SHARP CLI doesn't expose trajectory params yet
            # Custom trajectory will be supported in future via Python API
            
            # Pre-check: verify gsplat can render
            try:
                import torch
                if not torch.cuda.is_available():
                    logger.warning("CUDA not available - video rendering will fail!")
                else:
                    logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
            except Exception as e:
                logger.warning(f"CUDA check failed: {e}")
        
        logger.info(f"Running SHARP: {' '.join(cmd)}")
        
        # Log environment for debugging subprocess issues
        logger.info(f"Subprocess CONDA_PREFIX: {os.environ.get('CONDA_PREFIX', 'NOT SET')}")
        logger.info(f"Subprocess PATH (first entry): {os.environ.get('PATH', '').split(':')[0]}")
        
        # Use longer timeout for first run (model download can take 5+ min)
        # SHARP downloads ~2.6GB model on first run
        # Explicitly pass current environment to ensure conda env is inherited
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 minute timeout for model download
            env=os.environ.copy(),  # Explicitly pass environment
        )
        
        # Log SHARP output for debugging
        if result.stdout:
            logger.info(f"SHARP stdout: {result.stdout[-2000:]}")
        if result.stderr:
            logger.warning(f"SHARP stderr: {result.stderr[-2000:]}")
        
        # Find output files FIRST (before checking return code)
        # This way we can recover PLY even if video rendering failed
        output_files = os.listdir(temp_output_dir)
        logger.info(f"SHARP output files: {output_files}")
        ply_files = [f for f in output_files if f.endswith('.ply')]
        mp4_files = [f for f in output_files if f.endswith('.mp4')]
        
        if render_video and not mp4_files:
            logger.warning(f"Video rendering was requested but no MP4 files found in output!")
            logger.warning(f"This usually means gsplat failed to compile or CUDA is not available.")
        
        results = {}
        
        # Use model-specific subdirectory
        sharp_output_dir = os.path.join(OUTPUT_DIR, "sharp")
        Path(sharp_output_dir).mkdir(parents=True, exist_ok=True)
        
        # Copy PLY to output directory
        if ply_files:
            src_ply = os.path.join(temp_output_dir, ply_files[0])
            dst_ply = os.path.join(sharp_output_dir, f"{output_name}.ply")
            shutil.copy2(src_ply, dst_ply)
            results["ply_path"] = dst_ply
            logger.info(f"SHARP PLY saved: {dst_ply}")
        
        # Copy video if rendered
        if mp4_files:
            src_video = os.path.join(temp_output_dir, mp4_files[0])
            dst_video = os.path.join(sharp_output_dir, f"{output_name}.mp4")
            shutil.copy2(src_video, dst_video)
            results["video_path"] = dst_video
            logger.info(f"SHARP video saved: {dst_video}")
        
        # Now check return code - but if we have a PLY, return it even if video failed
        if result.returncode != 0:
            stderr = result.stderr or ""
            stdout = result.stdout or ""
            combined = stdout + "\n" + stderr
            
            logger.warning(f"SHARP process returned code {result.returncode}")
            
            # If we have a PLY, return it with a warning about video
            if "ply_path" in results:
                logger.info(f"SHARP PLY was generated despite error (video rendering may have failed)")
                if "video_path" not in results and render_video:
                    results["warning"] = "Video rendering failed (gsplat compilation timeout). PLY was saved successfully."
                return results
            
            # No PLY means real failure
            logger.error(f"SHARP failed completely. stdout: {stdout[-1000:]}")
            logger.error(f"SHARP stderr: {stderr[-1000:]}")
            raise RuntimeError(f"SHARP failed (code {result.returncode}): {combined[-1500:]}")
        
        if not results:
            raise RuntimeError("No output files generated by SHARP")
        
        return results
        
    finally:
        # Cleanup
        shutil.rmtree(temp_input_dir, ignore_errors=True)
        shutil.rmtree(temp_output_dir, ignore_errors=True)


# =============================================================================
# LYRA INFERENCE
# =============================================================================

def validate_lyra() -> bool:
    """Check if Lyra environment is available."""
    # Check if Lyra repo exists
    lyra_exists = os.path.isdir(LYRA_DIR)
    
    # Check if lyra_inference.py script exists (our wrapper)
    inference_script = "/workspace/lyra_inference.py"
    script_exists = os.path.exists(inference_script)
    
    if lyra_exists:
        # Check for any Python files or common repo indicators
        has_readme = os.path.exists(os.path.join(LYRA_DIR, "README.md"))
        has_sample = os.path.exists(os.path.join(LYRA_DIR, "sample.py"))
        has_yaml = os.path.exists(os.path.join(LYRA_DIR, "lyra.yaml"))
        repo_valid = has_readme or has_sample or has_yaml
        
        if repo_valid:
            logger.info(f"Lyra repo validated: README={has_readme}, sample.py={has_sample}, lyra.yaml={has_yaml}")
            return True
        else:
            logger.warning(f"Lyra directory exists but missing expected files at {LYRA_DIR}")
            # List contents for debugging
            try:
                contents = os.listdir(LYRA_DIR)[:10]
                logger.warning(f"Lyra dir contents (first 10): {contents}")
            except Exception as e:
                logger.warning(f"Could not list Lyra dir: {e}")
    else:
        logger.warning(f"Lyra directory not found: {LYRA_DIR}")
    
    # Fallback: if inference script exists, we can still try to run
    if script_exists:
        logger.info("Lyra inference script found, marking as available")
        return True
    
    return False


def run_lyra(
    input_image_path: str,
    output_name: str,
    generation_mode: str = "static",
    num_views: int = 8,
    camera_motion_scale: float = 1.0,
    multi_trajectory: bool = True,
    foreground_masking: bool = True,
    max_gaussians: int = 100000,
    seed: Optional[int] = None,
    input_video_path: str = None,
) -> Dict[str, str]:
    """
    Run Lyra inference for 3D/4D Gaussian Splatting generation.
    
    Lyra is a 2-step pipeline:
    1. SDG (Synthetic Data Generation) - generates multi-view videos using GEN3C
    2. 3DGS decoder - reconstructs Gaussian splats
    
    This function calls the lyra_inference.py wrapper script which handles both steps.
    """
    results = {}
    
    # Create temp output directory
    temp_output_dir = tempfile.mkdtemp(prefix="lyra_output_")
    
    try:
        # Build command
        # Use conda environment's Python
        python_path = "/root/miniforge3/envs/cosmos-predict1/bin/python"
        lyra_script = "/workspace/lyra_inference.py"
        
        cmd = [
            python_path,
            lyra_script,
            "--output_dir", temp_output_dir,
            "--output_name", output_name,
            "--checkpoint_dir", CHECKPOINT_DIR,  # Lyra uses GEN3C checkpoints
            "--mode", generation_mode,
            "--movement_factor", str(camera_motion_scale),
        ]
        
        # Add input based on mode
        if generation_mode == "static":
            cmd.extend(["--input_image", input_image_path])
        else:
            if input_video_path:
                cmd.extend(["--input_video", input_video_path])
            else:
                raise RuntimeError("Dynamic mode requires input_video_path")
        
        # Add flags
        if foreground_masking:
            cmd.append("--foreground_masking")
        else:
            cmd.append("--no_foreground_masking")
        
        if multi_trajectory:
            cmd.append("--multi_trajectory")
        else:
            cmd.append("--no_multi_trajectory")
        
        logger.info(f"Running Lyra: {' '.join(cmd)}")
        
        # Run the inference
        # Lyra is very slow - SDG (diffusion) + 3DGS (reconstruction) can take 60-90 minutes
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hour timeout for full Lyra pipeline
        )
        
        if result.returncode != 0:
            logger.error(f"Lyra failed. stderr: {result.stderr[-2000:]}")
            raise RuntimeError(f"Lyra failed: {result.stderr[-2000:]}")
        
        # Find output PLY
        ply_path = None
        for root, dirs, files in os.walk(temp_output_dir):
            for f in files:
                if f.endswith('.ply'):
                    ply_path = os.path.join(root, f)
                    break
            if ply_path:
                break
        
        if not ply_path:
            raise RuntimeError("Lyra completed but no PLY file found")
        
        # Copy to persistent output location
        output_dir = os.path.join(OUTPUT_DIR, "lyra")
        os.makedirs(output_dir, exist_ok=True)
        
        final_ply_path = os.path.join(output_dir, f"{output_name}.ply")
        shutil.copy2(ply_path, final_ply_path)
        results["ply_path"] = final_ply_path
        logger.info(f"Lyra PLY saved: {final_ply_path}")
        
        return results
        
    finally:
        # Cleanup
        shutil.rmtree(temp_output_dir, ignore_errors=True)


# =============================================================================
# TRELLIS.2 INFERENCE
# =============================================================================

def validate_trellis() -> bool:
    """Check if TRELLIS.2 environment is available."""
    # Check if TRELLIS repo exists or checkpoint is available
    trellis_exists = os.path.isdir(TRELLIS_DIR)
    trellis_script = os.path.join(TRELLIS_DIR, "scripts", "inference.py")
    checkpoint_exists = os.path.isdir(TRELLIS_CHECKPOINT) or os.path.exists(TRELLIS_CHECKPOINT)
    return trellis_exists or checkpoint_exists


def run_trellis(
    input_image_path: str,
    output_name: str,
    resolution: int = 1024,
    guidance_scale: float = 7.5,
    output_glb: bool = True,
    seed: Optional[int] = None,
) -> Dict[str, str]:
    """
    Run TRELLIS.2 inference.
    
    NOTE: TRELLIS.2 uses a Python API approach, not a CLI script:
    
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    mesh = pipeline.run(image)[0]
    
    Full implementation requires:
    - TRELLIS.2-4B checkpoint (~10GB) from HuggingFace
    - Complex setup with O-Voxel, FlexGEMM, CuMesh dependencies
    - Separate conda environment (trellis2)
    
    For now, this returns an error indicating TRELLIS.2 is not yet fully implemented.
    """
    # TRELLIS.2 is not yet fully implemented for serverless
    # The repo uses a Python API approach, not a CLI script
    raise RuntimeError(
        "TRELLIS.2 is not yet fully implemented for serverless deployment. "
        "TRELLIS.2 uses a Python API (Trellis2ImageTo3DPipeline), not a CLI script. "
        "Full implementation requires: "
        "1) TRELLIS.2-4B checkpoint (~10GB) from HuggingFace, "
        "2) Separate conda environment with O-Voxel, FlexGEMM, CuMesh, "
        "3) Custom inference wrapper script. "
        "Please use SHARP for fast 3DGS generation, or GEN3C for video generation."
    )


# =============================================================================
# UNIFIED HANDLER
# =============================================================================

def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod serverless handler function.
    
    Input (job["input"]):
        Special Actions:
        - action: "version" - Returns installed versions of all models (no other params needed)
        
        Common (for generation):
        - model: "gen3c", "sharp", "lyra", or "trellis" (required)
        - image_base64: Base64 encoded input image (required)
        - output_name: Output file name (optional)
        - return_base64: Return output as base64 (optional, default True)
        
        GEN3C specific:
        - guidance: Guidance scale (default 1.0)
        - num_frames: Number of frames (default 121)
        - trajectory: Camera trajectory (default "left")
        - foreground_masking: Enable masking (default True)
        - seed: Random seed (optional)
        
        SHARP specific:
        - render_video: Render video trajectory (default False)
        
        Lyra specific:
        - generation_mode: "static" or "dynamic" (default "static")
        - num_views: Number of views (default 8)
        - camera_motion_scale: Motion scale (default 1.0)
        - multi_trajectory: Enable multi-trajectory (default True)
        - max_gaussians: Max splats (default 100000)
        
        TRELLIS.2 specific:
        - resolution: Voxel resolution 512/1024/1536 (default 1024)
        - guidance_scale: CFG scale (default 7.5)
        - output_glb: Output GLB format (default True)
    
    Output:
        - status: "success" or "error"
        - message: Status message
        - Model-specific outputs (video_base64, ply_base64, etc.)
    """
    try:
        job_input = job.get("input", {})
        
        # Check for special actions first
        action = job_input.get("action", "").lower()
        
        # Version query - returns installed versions of all models
        if action == "version":
            versions = get_all_versions()
            return {
                "status": "success",
                "action": "version",
                "versions": versions,
                "message": "Version information retrieved successfully"
            }
        
        # Determine model type
        model = job_input.get("model", "gen3c").lower()
        
        # Default to False - PLY/GLB files can be 50-100+ MB which exceeds RunPod's 10MB response limit
        return_base64 = job_input.get("return_base64", False)
        
        # SuGaR uses PLY input, not image - handle separately
        if model == "sugar":
            return handle_sugar(job, job_input, "", return_base64)
        
        # All other models require image input
        if "image_base64" not in job_input:
            return {"status": "error", "message": "Missing required field: image_base64"}
        
        # Decode image to temp file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(base64.b64decode(job_input["image_base64"]))
            input_path = f.name
        
        try:
            if model == "gen3c":
                return handle_gen3c(job, job_input, input_path, return_base64)
            elif model == "sharp":
                return handle_sharp(job, job_input, input_path, return_base64)
            elif model == "lyra":
                return handle_lyra(job, job_input, input_path, return_base64)
            elif model == "trellis":
                return handle_trellis(job, job_input, input_path, return_base64)
            else:
                return {"status": "error", "message": f"Unknown model: {model}. Supported: gen3c, sharp, lyra, trellis, sugar"}
        finally:
            # Cleanup temp input
            Path(input_path).unlink(missing_ok=True)
            
    except Exception as e:
        logger.exception("Handler error")
        return {"status": "error", "message": str(e)}


def handle_gen3c(job: Dict, job_input: Dict, input_path: str, return_base64: bool) -> Dict:
    """Handle GEN3C job."""
    if not validate_gen3c():
        return {"status": "error", "message": "GEN3C environment not available"}
    
    video_name = job_input.get("video_name", job_input.get("output_name", f"gen3c_{job.get('id', 'output')}"))
    guidance = float(job_input.get("guidance", 1.0))
    num_frames = int(job_input.get("num_frames", 121))
    trajectory = job_input.get("trajectory", "left")
    foreground_masking = job_input.get("foreground_masking", True)
    seed = job_input.get("seed")
    movement_distance = float(job_input.get("movement_distance", 0.3))
    camera_rotation = job_input.get("camera_rotation", "center_facing")
    
    # Validate frames
    valid_frames = [121, 241, 361, 481]
    if num_frames not in valid_frames:
        return {"status": "error", "message": f"num_frames must be one of {valid_frames}"}
    
    # Validate camera_rotation
    valid_rotations = ["center_facing", "no_rotation", "trajectory_aligned"]
    if camera_rotation not in valid_rotations:
        camera_rotation = "center_facing"
    
    # Clamp movement_distance
    movement_distance = max(0.1, min(1.0, movement_distance))
    
    logger.info(f"GEN3C job: video_name={video_name}, frames={num_frames}, trajectory={trajectory}, "
                f"movement={movement_distance}, rotation={camera_rotation}")
    
    temp_output_path = run_gen3c(
        input_image_path=input_path,
        video_name=video_name,
        guidance=guidance,
        num_frames=num_frames,
        trajectory=trajectory,
        foreground_masking=foreground_masking,
        seed=seed,
        movement_distance=movement_distance,
        camera_rotation=camera_rotation
    )
    
    # Copy to network volume with model-specific subdirectory
    gen3c_output_dir = os.path.join(OUTPUT_DIR, "gen3c")
    Path(gen3c_output_dir).mkdir(parents=True, exist_ok=True)
    output_path = os.path.join(gen3c_output_dir, f"{video_name}.mp4")
    shutil.copy2(temp_output_path, output_path)
    logger.info(f"GEN3C video copied to: {output_path}")
    
    file_size = os.path.getsize(output_path)
    response = {
        "status": "success",
        "message": "Video generated successfully",
        "model": "gen3c",
        "video_path": output_path,
        "video_name": f"{video_name}.mp4",
        "file_size": file_size,
    }
    
    if return_base64:
        encoded = encode_file_if_small(output_path)
        if encoded:
            response["video_base64"] = encoded
        else:
            # Try S3 upload for large files
            s3_result = upload_file_to_s3_if_large(output_path, "gen3c", f"{video_name}.mp4")
            if s3_result["uploaded"]:
                response["video_s3_url"] = s3_result["s3_url"]
                response["message"] = f"Video uploaded to S3 ({file_size / 1024 / 1024:.1f}MB)"
            else:
                response["download_required"] = True
                response["message"] = f"Video generated ({file_size / 1024 / 1024:.1f}MB). File saved to network volume - download required."
    
    return response


def handle_sharp(job: Dict, job_input: Dict, input_path: str, return_base64: bool) -> Dict:
    """Handle SHARP job."""
    if not validate_sharp():
        return {"status": "error", "message": "SHARP environment not available"}
    
    output_name = job_input.get("output_name", f"sharp_{job.get('id', 'output')}")
    render_video = job_input.get("render_video", False)
    
    # Extract trajectory parameters
    trajectory_type = job_input.get("trajectory_type", "rotate_forward")
    num_steps = int(job_input.get("num_steps", 60))
    num_repeats = int(job_input.get("num_repeats", 1))
    max_disparity = float(job_input.get("max_disparity", 0.08))
    max_zoom = float(job_input.get("max_zoom", 0.15))
    lookat_mode = job_input.get("lookat_mode", "point")
    
    # Validate trajectory type
    valid_trajectories = ["rotate_forward", "rotate", "swipe", "shake"]
    if trajectory_type not in valid_trajectories:
        trajectory_type = "rotate_forward"
    
    # Validate lookat mode
    if lookat_mode not in ["point", "ahead"]:
        lookat_mode = "point"
    
    logger.info(f"SHARP job: output_name={output_name}, render_video={render_video}, "
                f"trajectory={trajectory_type}, steps={num_steps}")
    
    results = run_sharp(
        input_image_path=input_path,
        output_name=output_name,
        render_video=render_video,
        trajectory_type=trajectory_type,
        num_steps=num_steps,
        num_repeats=num_repeats,
        max_disparity=max_disparity,
        max_zoom=max_zoom,
        lookat_mode=lookat_mode,
    )
    
    response = {
        "status": "success",
        "message": "PLY generated successfully",
        "model": "sharp",
        "ply_name": f"{output_name}.ply",
    }
    
    download_required = False
    
    if "ply_path" in results:
        ply_size = os.path.getsize(results["ply_path"])
        response["ply_path"] = results["ply_path"]
        response["ply_size"] = ply_size
        
        # ALWAYS upload to S3 first (ensures file is available even if API times out)
        s3_result = upload_file_to_s3_always(results["ply_path"], "sharp", f"{output_name}.ply")
        if s3_result["uploaded"]:
            response["ply_s3_url"] = s3_result["s3_url"]
            response["message"] = f"PLY uploaded to S3 ({ply_size / 1024 / 1024:.1f}MB)"
        else:
            download_required = True
        
        # Also try base64 for small files (faster if API response succeeds)
        if return_base64:
            encoded = encode_file_if_small(results["ply_path"])
            if encoded:
                response["ply_base64"] = encoded
    
    if "video_path" in results:
        video_size = os.path.getsize(results["video_path"])
        response["video_path"] = results["video_path"]
        response["video_name"] = f"{output_name}.mp4"
        response["video_size"] = video_size
        
        # ALWAYS upload video to S3 first (ensures file is available even if API times out)
        s3_result = upload_file_to_s3_always(results["video_path"], "sharp", f"{output_name}.mp4")
        if s3_result["uploaded"]:
            response["video_s3_url"] = s3_result["s3_url"]
            response["message"] = f"Video uploaded to S3 ({video_size / 1024 / 1024:.1f}MB)"
        else:
            download_required = True
        
        # Also try base64 for small files (faster if API response succeeds)
        if return_base64:
            encoded = encode_file_if_small(results["video_path"])
            if encoded:
                response["video_base64"] = encoded
    elif render_video:
        # Video was requested but not generated
        response["video_error"] = "Video rendering failed. This usually means gsplat is not installed or CUDA is unavailable."
        if "warning" in results:
            response["video_error"] = results["warning"]
        logger.warning(f"Video rendering was requested but no video was generated")
    
    if download_required:
        response["download_required"] = True
        response["message"] = "Files generated but too large for API response. Download from network volume or configure S3."
    
    return response


def handle_lyra(job: Dict, job_input: Dict, input_path: str, return_base64: bool) -> Dict:
    """Handle Lyra job."""
    if not validate_lyra():
        return {"status": "error", "message": "Lyra environment not available"}
    
    output_name = job_input.get("output_name", f"lyra_{job.get('id', 'output')}")
    generation_mode = job_input.get("generation_mode", "static")
    num_views = int(job_input.get("num_views", 8))
    camera_motion_scale = float(job_input.get("camera_motion_scale", 1.0))
    multi_trajectory = job_input.get("multi_trajectory", True)
    foreground_masking = job_input.get("foreground_masking", True)
    max_gaussians = int(job_input.get("max_gaussians", 100000))
    seed = job_input.get("seed")
    
    logger.info(f"Lyra job: output_name={output_name}, mode={generation_mode}, views={num_views}")
    
    results = run_lyra(
        input_image_path=input_path,
        output_name=output_name,
        generation_mode=generation_mode,
        num_views=num_views,
        camera_motion_scale=camera_motion_scale,
        multi_trajectory=multi_trajectory,
        foreground_masking=foreground_masking,
        max_gaussians=max_gaussians,
        seed=seed,
    )
    
    response = {
        "status": "success",
        "message": "3DGS generated successfully",
        "model": "lyra",
        "output_name": output_name,
    }
    
    download_required = False
    
    if "ply_path" in results:
        ply_size = os.path.getsize(results["ply_path"])
        response["ply_path"] = results["ply_path"]
        response["ply_name"] = f"{output_name}.ply"
        response["ply_size"] = ply_size
        
        # Try base64 for small files
        if return_base64:
            encoded = encode_file_if_small(results["ply_path"])
            if encoded:
                response["ply_base64"] = encoded
        
        # Always try S3 upload for large files
        if "ply_base64" not in response:
            s3_result = upload_file_to_s3_if_large(results["ply_path"], "lyra", f"{output_name}.ply")
            if s3_result["uploaded"]:
                response["ply_s3_url"] = s3_result["s3_url"]
            else:
                download_required = True
    
    if "video_path" in results:
        video_size = os.path.getsize(results["video_path"])
        response["video_path"] = results["video_path"]
        response["video_name"] = f"{output_name}.mp4"
        response["video_size"] = video_size
        
        # Try base64 for small files
        if return_base64:
            encoded = encode_file_if_small(results["video_path"])
            if encoded:
                response["video_base64"] = encoded
        
        # Always try S3 upload for large files
        if "video_base64" not in response:
            s3_result = upload_file_to_s3_if_large(results["video_path"], "lyra", f"{output_name}.mp4")
            if s3_result["uploaded"]:
                response["video_s3_url"] = s3_result["s3_url"]
            else:
                download_required = True
    
    if download_required:
        response["download_required"] = True
        response["message"] = "Files generated but too large for API response. Download from network volume or configure S3."
    
    return response


def handle_trellis(job: Dict, job_input: Dict, input_path: str, return_base64: bool) -> Dict:
    """Handle TRELLIS.2 job."""
    if not validate_trellis():
        return {"status": "error", "message": "TRELLIS.2 environment not available"}
    
    output_name = job_input.get("output_name", f"trellis_{job.get('id', 'output')}")
    resolution = int(job_input.get("resolution", 1024))
    guidance_scale = float(job_input.get("guidance_scale", 7.5))
    output_glb = job_input.get("output_glb", True)
    seed = job_input.get("seed")
    
    # Validate resolution
    valid_resolutions = [512, 1024, 1536]
    if resolution not in valid_resolutions:
        return {"status": "error", "message": f"resolution must be one of {valid_resolutions}"}
    
    logger.info(f"TRELLIS.2 job: output_name={output_name}, resolution={resolution}, guidance={guidance_scale}")
    
    results = run_trellis(
        input_image_path=input_path,
        output_name=output_name,
        resolution=resolution,
        guidance_scale=guidance_scale,
        output_glb=output_glb,
        seed=seed,
    )
    
    response = {
        "status": "success",
        "message": "3D model generated successfully",
        "model": "trellis",
        "output_name": output_name,
    }
    
    download_required = False
    
    if "glb_path" in results:
        glb_size = os.path.getsize(results["glb_path"])
        response["glb_path"] = results["glb_path"]
        response["glb_name"] = f"{output_name}.glb"
        response["glb_size"] = glb_size
        
        # Try base64 for small files
        if return_base64:
            encoded = encode_file_if_small(results["glb_path"])
            if encoded:
                response["glb_base64"] = encoded
        
        # Always try S3 upload for large files
        if "glb_base64" not in response:
            s3_result = upload_file_to_s3_if_large(results["glb_path"], "trellis", f"{output_name}.glb")
            if s3_result["uploaded"]:
                response["glb_s3_url"] = s3_result["s3_url"]
            else:
                download_required = True
    
    if "ply_path" in results:
        ply_size = os.path.getsize(results["ply_path"])
        response["ply_path"] = results["ply_path"]
        response["ply_name"] = f"{output_name}.ply"
        response["ply_size"] = ply_size
        
        # Try base64 for small files
        if return_base64:
            encoded = encode_file_if_small(results["ply_path"])
            if encoded:
                response["ply_base64"] = encoded
        
        # Always try S3 upload for large files
        if "ply_base64" not in response:
            s3_result = upload_file_to_s3_if_large(results["ply_path"], "trellis", f"{output_name}.ply")
            if s3_result["uploaded"]:
                response["ply_s3_url"] = s3_result["s3_url"]
            else:
                download_required = True
    
    if download_required:
        response["download_required"] = True
        response["message"] = "Files generated but too large for API response. Download from network volume or configure S3."
    
    return response


# =============================================================================
# SUGAR MESH EXTRACTION
# =============================================================================

def validate_sugar() -> bool:
    """Check if SuGaR environment is available."""
    sugar_exists = os.path.isdir(SUGAR_DIR)
    
    if sugar_exists:
        # Check for key SuGaR files
        has_pipeline = os.path.exists(os.path.join(SUGAR_DIR, "train_full_pipeline.py"))
        has_extract = os.path.exists(os.path.join(SUGAR_DIR, "extract_mesh.py"))
        has_extractors = os.path.isdir(os.path.join(SUGAR_DIR, "sugar_extractors"))
        
        if has_pipeline and has_extract and has_extractors:
            logger.info("SuGaR validated: pipeline, extract_mesh, extractors found")
            return True
        else:
            logger.warning(f"SuGaR directory exists but missing files: pipeline={has_pipeline}, extract={has_extract}, extractors={has_extractors}")
    else:
        logger.warning(f"SuGaR directory not found: {SUGAR_DIR}")
    
    # Check for sugar_inference.py wrapper
    if os.path.exists("/workspace/sugar_inference.py"):
        logger.info("SuGaR inference wrapper found")
        return True
    
    return False


def run_sugar(
    input_ply_path: str,
    output_name: str,
    input_format: str = "auto",
    regularization: str = "dn_consistency",
    poisson_depth: int = 10,
    refinement_time: str = "short",
    target_vertices: int = 1_000_000,
    export_texture: bool = True,
    texture_resolution: int = 2048,
    export_glb: bool = True,
    export_obj: bool = False,
) -> Dict[str, str]:
    """
    Run SuGaR mesh extraction.
    
    Returns:
        Dictionary with output paths: {"mesh_path": ..., "texture_path": ...}
    """
    results = {}
    
    # Create temp output directory
    temp_output_dir = tempfile.mkdtemp(prefix="sugar_output_")
    
    try:
        # Use conda environment's Python
        python_path = "/root/miniforge3/envs/cosmos-predict1/bin/python"
        sugar_script = "/workspace/sugar_inference.py"
        
        cmd = [
            python_path,
            sugar_script,
            "--input_ply", input_ply_path,
            "--output_dir", temp_output_dir,
            "--output_name", output_name,
            "--regularization", regularization,
            "--poisson_depth", str(poisson_depth),
            "--refinement_time", refinement_time,
            "--target_vertices", str(target_vertices),
            "--texture_resolution", str(texture_resolution),
        ]
        
        if export_texture:
            cmd.append("--export_texture")
        if export_glb:
            cmd.append("--export_glb")
        if export_obj:
            cmd.append("--export_obj")
        
        logger.info(f"Running SuGaR: {' '.join(cmd)}")
        
        # SuGaR can take 15-60 minutes
        timeout = 7200 if refinement_time == "long" else (3600 if refinement_time == "medium" else 1800)
        
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        
        if proc.returncode != 0:
            logger.error(f"SuGaR failed: {proc.stderr}")
            raise RuntimeError(f"SuGaR failed: {proc.stderr[-500:]}")
        
        logger.info("SuGaR completed successfully")
        
        # Find output files
        sugar_output_dir = os.path.join(OUTPUT_DIR, "sugar")
        Path(sugar_output_dir).mkdir(parents=True, exist_ok=True)
        
        # Look for mesh files
        for ext in [".glb", ".obj", ".ply"]:
            mesh_file = os.path.join(temp_output_dir, f"{output_name}{ext}")
            if os.path.exists(mesh_file):
                dest_path = os.path.join(sugar_output_dir, f"{output_name}{ext}")
                shutil.copy2(mesh_file, dest_path)
                results["mesh_path"] = dest_path
                logger.info(f"SuGaR mesh saved: {dest_path}")
                break
        
        # Look for texture files
        for ext in [".png", ".jpg"]:
            tex_file = os.path.join(temp_output_dir, f"{output_name}_texture{ext}")
            if os.path.exists(tex_file):
                dest_path = os.path.join(sugar_output_dir, f"{output_name}_texture{ext}")
                shutil.copy2(tex_file, dest_path)
                results["texture_path"] = dest_path
                break
        
        return results
    
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"SuGaR timeout after {timeout}s")
    finally:
        shutil.rmtree(temp_output_dir, ignore_errors=True)


def handle_sugar(job: Dict, job_input: Dict, input_path: str, return_base64: bool) -> Dict:
    """
    Handle SuGaR mesh extraction job.
    
    Note: SuGaR expects a PLY file as input, not an image.
    Supports three input methods:
    1. input_ply_path: Direct path on network volume (preferred for large files)
    2. ply_base64: Base64-encoded PLY data
    3. Fallback to input_path if neither is provided
    """
    if not validate_sugar():
        return {"status": "error", "message": "SuGaR environment not available"}
    
    output_name = job_input.get("output_name", f"sugar_{job.get('id', 'output')}")
    input_format = job_input.get("input_format", "auto")
    regularization = job_input.get("regularization", "dn_consistency")
    poisson_depth = int(job_input.get("poisson_depth", 10))
    refinement_time = job_input.get("refinement_time", "short")
    target_vertices = int(job_input.get("target_vertices", 1_000_000))
    export_texture = job_input.get("export_texture", True)
    texture_resolution = int(job_input.get("texture_resolution", 2048))
    export_glb = job_input.get("export_glb", True)
    export_obj = job_input.get("export_obj", False)
    
    logger.info(f"SuGaR job: output_name={output_name}, reg={regularization}, depth={poisson_depth}")
    
    # Determine input PLY path - support multiple input methods
    cleanup_ply = False  # Track if we need to delete the PLY after
    
    if "input_s3_url" in job_input:
        # Download PLY from S3 (for large files uploaded by client)
        s3_url = job_input["input_s3_url"]
        logger.info(f"SuGaR downloading PLY from S3...")
        
        try:
            import requests
            response = requests.get(s3_url, timeout=300)  # 5 min timeout for large files
            response.raise_for_status()
            
            with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
                f.write(response.content)
                ply_input_path = f.name
            
            file_size_mb = len(response.content) / (1024 * 1024)
            logger.info(f"SuGaR downloaded {file_size_mb:.1f} MB from S3")
            cleanup_ply = True
            
        except Exception as e:
            logger.error(f"Failed to download from S3: {e}")
            return {"status": "error", "message": f"Failed to download PLY from S3: {e}"}
    
    elif "input_ply_path" in job_input:
        # Direct path on network volume (for files already on RunPod)
        ply_input_path = job_input["input_ply_path"]
        logger.info(f"SuGaR using network volume path: {ply_input_path}")
        
        if not os.path.exists(ply_input_path):
            # Log what we can see to help debug
            parent_dir = os.path.dirname(ply_input_path)
            logger.error(f"PLY file not found: {ply_input_path}")
            logger.error(f"Parent dir exists: {os.path.exists(parent_dir)}")
            if os.path.exists(parent_dir):
                try:
                    files = os.listdir(parent_dir)
                    logger.error(f"Files in {parent_dir}: {files[:20]}")  # First 20 files
                except Exception as e:
                    logger.error(f"Cannot list {parent_dir}: {e}")
            
            return {"status": "error", "message": f"PLY file not found: {ply_input_path}"}
        
        cleanup_ply = False  # Don't delete the original file!
        
    elif "ply_base64" in job_input:
        # Decode PLY from base64 to temp file
        logger.info("SuGaR decoding PLY from base64...")
        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
            f.write(base64.b64decode(job_input["ply_base64"]))
            ply_input_path = f.name
        cleanup_ply = True
        
    else:
        # Assume input_path is already a PLY (renamed from .png)
        ply_input_path = input_path.replace(".png", ".ply")
        if os.path.exists(input_path):
            shutil.move(input_path, ply_input_path)
        cleanup_ply = True
    
    try:
        results = run_sugar(
            input_ply_path=ply_input_path,
            output_name=output_name,
            input_format=input_format,
            regularization=regularization,
            poisson_depth=poisson_depth,
            refinement_time=refinement_time,
            target_vertices=target_vertices,
            export_texture=export_texture,
            texture_resolution=texture_resolution,
            export_glb=export_glb,
            export_obj=export_obj,
        )
    finally:
        # Only cleanup temp PLY files, not files from network volume
        if cleanup_ply:
            Path(ply_input_path).unlink(missing_ok=True)
    
    response = {
        "status": "success",
        "message": "Mesh extracted successfully",
        "model": "sugar",
        "output_name": output_name,
    }
    
    download_required = False
    
    if "mesh_path" in results:
        mesh_size = os.path.getsize(results["mesh_path"])
        response["mesh_path"] = results["mesh_path"]
        response["mesh_name"] = os.path.basename(results["mesh_path"])
        response["mesh_size"] = mesh_size
        
        # Always try base64 for small files (< 8MB)
        encoded = encode_file_if_small(results["mesh_path"])
        if encoded:
            response["mesh_base64"] = encoded
            logger.info(f"Mesh encoded as base64 ({mesh_size / 1024 / 1024:.1f}MB)")
        
        # Try S3 upload for large files that couldn't be base64 encoded
        if "mesh_base64" not in response:
            s3_result = upload_file_to_s3_if_large(
                results["mesh_path"], "sugar", os.path.basename(results["mesh_path"])
            )
            if s3_result["uploaded"]:
                response["mesh_s3_url"] = s3_result["s3_url"]
                response["message"] = f"Mesh uploaded to S3 ({mesh_size / 1024 / 1024:.1f}MB)"
            else:
                download_required = True
    
    if "texture_path" in results:
        tex_size = os.path.getsize(results["texture_path"])
        response["texture_path"] = results["texture_path"]
        response["texture_name"] = os.path.basename(results["texture_path"])
        response["texture_size"] = tex_size
        
        # Try S3 for texture
        s3_result = upload_file_to_s3_if_large(
            results["texture_path"], "sugar", os.path.basename(results["texture_path"])
        )
        if s3_result["uploaded"]:
            response["texture_s3_url"] = s3_result["s3_url"]
    
    if download_required:
        response["download_required"] = True
        response["message"] = "Mesh generated but too large for API response. Download from network volume or configure S3."
    
    return response


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    logger.info("Starting Unified 3D Generation RunPod Serverless Handler")
    logger.info(f"GEN3C Directory: {GEN3C_DIR}")
    logger.info(f"SHARP Directory: {SHARP_DIR}")
    logger.info(f"Lyra Directory: {LYRA_DIR}")
    logger.info(f"TRELLIS Directory: {TRELLIS_DIR}")
    logger.info(f"SuGaR Directory: {SUGAR_DIR}")
    logger.info(f"Checkpoint Directory: {CHECKPOINT_DIR}")
    
    # Validate environments at startup
    gen3c_ok = validate_gen3c()
    sharp_ok = validate_sharp()
    lyra_ok = validate_lyra()
    trellis_ok = validate_trellis()
    sugar_ok = validate_sugar()
    
    logger.info(f"GEN3C available: {gen3c_ok}")
    logger.info(f"SHARP available: {sharp_ok}")
    logger.info(f"Lyra available: {lyra_ok}")
    logger.info(f"TRELLIS.2 available: {trellis_ok}")
    logger.info(f"SuGaR available: {sugar_ok}")
    logger.info(f"S3 enabled: {S3_ENABLED} (bucket={S3_BUCKET}, region={S3_REGION}, access_key={'set' if S3_ACCESS_KEY else 'NOT SET'})")
    
    # Start the serverless worker
    runpod.serverless.start({"handler": handler})

