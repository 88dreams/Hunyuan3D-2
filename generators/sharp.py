"""
SHARP Generator for 3D Generation Studio

Apple's SHARP: Sharp Monocular View Synthesis in Less Than a Second
- Single image → 3D Gaussian Splatting (PLY) in <1 second
- Optional video rendering (CUDA GPU required)

Reference: https://github.com/apple/ml-sharp
Paper: https://arxiv.org/abs/2512.10685
"""

import os
import subprocess
import shutil
from typing import Optional, Tuple, Union

import gradio as gr  # type: ignore


# =============================================================================
# CONFIGURATION
# =============================================================================

# Try to load from config module, fall back to defaults
try:
    from config import Config
    _cfg = Config()
    SHARP_DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(_cfg.hunyuan_outputs), "sharp")
except ImportError:
    SHARP_DEFAULT_OUTPUT_DIR = "/srv/searidge_share/outputs/sharp"

# Ensure output directory exists
os.makedirs(SHARP_DEFAULT_OUTPUT_DIR, exist_ok=True)

# Check if SHARP is installed
_sharp_available = shutil.which("sharp") is not None


def is_sharp_available() -> bool:
    """Check if SHARP CLI is installed and available."""
    return _sharp_available


def check_sharp_installation() -> str:
    """Check SHARP installation status and return formatted string."""
    if _sharp_available:
        try:
            result = subprocess.run(
                ["sharp", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            version_info = result.stdout.strip() or "installed"
            return f"✅ SHARP {version_info}"
        except Exception:
            return "✅ SHARP installed"
    else:
        return "❌ SHARP not installed. Run: pip install -r requirements.txt (from ml-sharp repo)"


# =============================================================================
# LOCAL PLY GENERATION
# =============================================================================

def run_sharp_local(
    image_path: Union[str, None],
    output_name: str,
    output_dir: str,
    render_video: bool = False,
) -> Tuple[Optional[str], str, str]:
    """
    Run SHARP locally to generate 3D Gaussian Splatting PLY from a single image.
    
    Args:
        image_path: Path to input image
        output_name: Base name for output files
        output_dir: Directory to save outputs
        render_video: Whether to render a video trajectory (CUDA only)
    
    Returns:
        Tuple of (output_path, logs, progress_status)
        - output_path: Path to generated PLY file (or MP4 if render_video)
        - logs: Generation logs
        - progress_status: Status message for UI
    """
    if not image_path:
        raise gr.Error("Please provide an image.")
    if not os.path.exists(image_path):
        raise gr.Error("Provided image path does not exist.")
    if not _sharp_available:
        raise gr.Error("SHARP is not installed. Please install it first: pip install -r requirements.txt (from ml-sharp repo)")
    
    # Resolve output directory
    resolved_output_dir = output_dir.strip() if output_dir else SHARP_DEFAULT_OUTPUT_DIR
    resolved_output_dir = os.path.abspath(os.path.expanduser(resolved_output_dir))
    os.makedirs(resolved_output_dir, exist_ok=True)
    
    # Clean output name
    output_name = output_name.strip() or "sharp_output"
    base_name = os.path.splitext(output_name)[0]
    
    # Create temp directory for this run
    import tempfile
    import time
    
    temp_input_dir = tempfile.mkdtemp(prefix="sharp_input_")
    temp_output_dir = tempfile.mkdtemp(prefix="sharp_output_")
    
    logs = []
    logs.append("🚀 Starting SHARP generation")
    logs.append(f"   Input: {os.path.basename(image_path)}")
    logs.append(f"   Output: {base_name}")
    logs.append(f"   Render video: {render_video}")
    
    try:
        # Copy input image to temp directory (SHARP expects a directory)
        input_ext = os.path.splitext(image_path)[1]
        temp_input_path = os.path.join(temp_input_dir, f"input{input_ext}")
        shutil.copy2(image_path, temp_input_path)
        
        # Build command
        cmd = [
            "sharp", "predict",
            "-i", temp_input_dir,
            "-o", temp_output_dir,
        ]
        
        if render_video:
            cmd.append("--render")
        
        logs.append(f"   Command: {' '.join(cmd)}")
        
        start_time = time.time()
        
        # Run SHARP
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout (should be <1s for PLY, longer for video)
        )
        
        elapsed = time.time() - start_time
        
        if result.stdout:
            logs.append(f"STDOUT:\n{result.stdout.strip()}")
        if result.stderr:
            logs.append(f"STDERR:\n{result.stderr.strip()}")
        
        if result.returncode != 0:
            logs.append(f"❌ SHARP exited with code {result.returncode}")
            return None, "\n".join(logs), "❌ SHARP generation failed"
        
        # Find output files
        output_files = os.listdir(temp_output_dir)
        logs.append(f"   Generated files: {output_files}")
        
        # Look for PLY file
        ply_files = [f for f in output_files if f.endswith('.ply')]
        mp4_files = [f for f in output_files if f.endswith('.mp4')]
        
        final_output_path = None
        
        if render_video and mp4_files:
            # Copy video to output directory
            src_video = os.path.join(temp_output_dir, mp4_files[0])
            final_output_path = os.path.join(resolved_output_dir, f"{base_name}.mp4")
            shutil.copy2(src_video, final_output_path)
            logs.append(f"✅ Video saved: {final_output_path}")
        
        if ply_files:
            # Copy PLY to output directory
            src_ply = os.path.join(temp_output_dir, ply_files[0])
            ply_output_path = os.path.join(resolved_output_dir, f"{base_name}.ply")
            shutil.copy2(src_ply, ply_output_path)
            logs.append(f"✅ PLY saved: {ply_output_path}")
            
            # If no video was requested, PLY is the main output
            if not render_video or not mp4_files:
                final_output_path = ply_output_path
        
        if final_output_path is None:
            logs.append("❌ No output files generated")
            return None, "\n".join(logs), "❌ No output generated"
        
        logs.append(f"✅ Generation complete in {elapsed:.1f}s")
        
        return final_output_path, "\n".join(logs), f"✅ SHARP complete ({elapsed:.1f}s)"
        
    except subprocess.TimeoutExpired:
        logs.append("❌ SHARP timed out after 5 minutes")
        return None, "\n".join(logs), "❌ SHARP timed out"
    except Exception as e:
        logs.append(f"❌ Error: {e}")
        return None, "\n".join(logs), f"❌ Error: {e}"
    finally:
        # Cleanup temp directories
        try:
            shutil.rmtree(temp_input_dir, ignore_errors=True)
            shutil.rmtree(temp_output_dir, ignore_errors=True)
        except Exception:
            pass


# =============================================================================
# RUNPOD EXECUTION (for video rendering on cloud GPU)
# =============================================================================

# RunPod client imports (optional)
_runpod_available = False
UnifiedServerlessClient = None

try:
    from runpod.runpod_client import UnifiedServerlessClient
    _runpod_available = True
except ImportError:
    pass


def is_runpod_available() -> bool:
    """Check if RunPod client is available."""
    return _runpod_available


def run_sharp_runpod(
    image_path: Union[str, None],
    endpoint_id: str,
    api_key: str,
    output_name: str,
    output_dir: str,
    render_video: bool = True,
) -> Tuple[Optional[str], str, str]:
    """
    Run SHARP on RunPod Serverless for video rendering.
    
    Args:
        image_path: Path to input image
        endpoint_id: RunPod serverless endpoint ID
        api_key: RunPod API key
        output_name: Base name for output files
        output_dir: Directory to save outputs
        render_video: Whether to render video trajectory
    
    Returns:
        Tuple of (output_path, logs, progress_status)
    """
    if not image_path:
        raise gr.Error("Please provide an image.")
    if not os.path.exists(image_path):
        raise gr.Error("Provided image path does not exist.")
    if not _runpod_available:
        raise gr.Error("RunPod client not available. Install runpod package.")
    if not endpoint_id or not api_key:
        raise gr.Error("RunPod endpoint ID and API key are required.")
    
    # Resolve output directory
    resolved_output_dir = output_dir.strip() if output_dir else SHARP_DEFAULT_OUTPUT_DIR
    resolved_output_dir = os.path.abspath(os.path.expanduser(resolved_output_dir))
    os.makedirs(resolved_output_dir, exist_ok=True)
    
    # Clean output name
    output_name = output_name.strip() or "sharp_output"
    base_name = os.path.splitext(output_name)[0]
    
    logs = []
    logs.append("🚀 Starting SHARP on RunPod")
    logs.append(f"   Input: {os.path.basename(image_path)}")
    logs.append(f"   Output: {base_name}")
    logs.append(f"   Render video: {render_video}")
    logs.append(f"   Endpoint: {endpoint_id}")
    
    try:
        client = UnifiedServerlessClient(endpoint_id, api_key)
        
        def progress_callback(status: str, elapsed: float):
            logs.append(f"   Status: {status} ({elapsed:.0f}s)")
        
        result = client.generate_sharp_sync(
            image_path=image_path,
            output_dir=resolved_output_dir,
            output_name=base_name,
            render_video=render_video,
            poll_interval=10,
            max_wait=600,  # 10 minute timeout for SHARP
            progress_callback=progress_callback,
        )
        
        if result.success:
            logs.append(f"✅ SHARP complete in {result.duration_seconds:.1f}s")
            logs.append(f"   Output: {result.output_path}")
            return result.output_path, "\n".join(logs), f"✅ SHARP complete ({result.duration_seconds:.1f}s)"
        else:
            logs.append(f"❌ SHARP failed: {result.error}")
            return None, "\n".join(logs), f"❌ SHARP failed: {result.error}"
    
    except Exception as e:
        logs.append(f"❌ Error: {e}")
        return None, "\n".join(logs), f"❌ Error: {e}"

