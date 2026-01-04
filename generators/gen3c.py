"""
GEN3C Generator for 3D Generation Studio

This module provides GEN3C video generation functionality:
- RunPod Pod execution
- RunPod Serverless execution
- Local cluster execution
"""

import os
import subprocess
from typing import Optional, Tuple, Union

import gradio as gr  # type: ignore


# =============================================================================
# CONFIGURATION
# =============================================================================

# Try to load from config module, fall back to defaults
try:
    from config import Config
    _cfg = Config()
    GEN3C_DEFAULT_CHECKPOINT = _cfg.gen3c_checkpoints
    GEN3C_DEFAULT_OUTPUT_DIR = _cfg.gen3c_outputs
except ImportError:
    GEN3C_DEFAULT_CHECKPOINT = "/srv/searidge_share/checkpoints/gen3c"
    GEN3C_DEFAULT_OUTPUT_DIR = "/srv/searidge_share/outputs/gen3c"

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEN3C_SCRIPT = os.path.join(PROJECT_ROOT, "scripts", "run_gen3c.sh")

# Ensure output directory exists
os.makedirs(GEN3C_DEFAULT_OUTPUT_DIR, exist_ok=True)


# =============================================================================
# RUNPOD CLIENT IMPORTS
# =============================================================================

_runpod_available = False
RunPodGEN3CClient = None
RunPodServerlessClient = None
RunPodJobResult = None

try:
    from runpod.runpod_client import (
        RunPodGEN3CClient,
        RunPodServerlessClient,
        RunPodJobResult,
    )
    _runpod_available = True
except ImportError:
    pass


def is_runpod_available() -> bool:
    """Check if RunPod client is available."""
    return _runpod_available


# =============================================================================
# RUNPOD POD EXECUTION
# =============================================================================

def run_gen3c_runpod(
    image_path: Union[str, None],
    runpod_url: str,
    guidance: float,
    frames: Union[str, int],
    trajectory: str,
    foreground_masking: bool,
    video_name: str,
    seed: Optional[int],
    output_dir: str,
) -> Tuple[Optional[str], str, str]:
    """Run GEN3C on RunPod cloud GPU and return the generated video."""
    if not image_path:
        raise gr.Error("Please provide an image.")
    if not os.path.exists(image_path):
        raise gr.Error("Provided image path does not exist.")
    if not _runpod_available:
        raise gr.Error("RunPod client not available. Check installation.")
    
    # Parse frames
    try:
        frames_int = int(frames)
    except (ValueError, TypeError):
        frames_int = 121
    
    video_name = video_name.strip() or "gen3c_video"
    resolved_output_dir = output_dir.strip() or GEN3C_DEFAULT_OUTPUT_DIR
    resolved_output_dir = os.path.abspath(os.path.expanduser(resolved_output_dir))
    os.makedirs(resolved_output_dir, exist_ok=True)
    
    logs = []
    logs.append(f"🚀 Starting RunPod GEN3C generation")
    logs.append(f"   API URL: {runpod_url}")
    logs.append(f"   Frames: {frames_int}, Trajectory: {trajectory}")
    logs.append(f"   Guidance: {guidance}, Foreground Mask: {foreground_masking}")
    
    # Create client
    client = RunPodGEN3CClient(runpod_url)
    
    # Check health
    health = client.health_check()
    if health.get("status") != "healthy":
        error_msg = health.get("error", "Unknown error")
        logs.append(f"❌ RunPod not ready: {error_msg}")
        return None, "\n".join(logs), "❌ RunPod not ready"
    
    logs.append(f"✓ RunPod ready: GPU={health.get('gpu', 'unknown')}")
    
    # Progress callback
    def progress_callback(status: str, elapsed: float):
        elapsed_str = f"{int(elapsed // 60)}m {int(elapsed % 60)}s"
        print(f"[RunPod] {elapsed_str}: {status}")
    
    # Run generation
    try:
        result = client.generate_sync(
            image_path=image_path,
            output_dir=resolved_output_dir,
            video_name=video_name,
            num_frames=frames_int,
            trajectory=trajectory,
            guidance=guidance,
            foreground_masking=foreground_masking,
            seed=seed if seed and seed > 0 else None,
            poll_interval=30,
            max_wait=3600,
            progress_callback=progress_callback
        )
    except Exception as e:
        logs.append(f"❌ Error: {e}")
        return None, "\n".join(logs), f"❌ RunPod error: {e}"
    
    logs.append(result.logs)
    
    if result.success:
        logs.append(f"✅ Video generated in {result.duration_seconds:.1f}s")
        logs.append(f"   Saved to: {result.output_path}")
        return result.output_path, "\n".join(logs), f"✅ RunPod GEN3C complete ({result.duration_seconds:.0f}s)"
    else:
        logs.append(f"❌ Generation failed: {result.error}")
        return None, "\n".join(logs), "❌ RunPod GEN3C failed"


# =============================================================================
# RUNPOD SERVERLESS EXECUTION
# =============================================================================

def run_gen3c_serverless(
    image_path: Union[str, None],
    endpoint_id: str,
    api_key: str,
    guidance: float,
    frames: Union[str, int],
    trajectory: str,
    foreground_masking: bool,
    video_name: str,
    seed: Optional[int],
    output_dir: str,
    movement_distance: float = 0.3,
    camera_rotation: str = "center_facing",
) -> Tuple[Optional[str], str, str]:
    """Run GEN3C on RunPod Serverless and return the generated video."""
    if not image_path:
        raise gr.Error("Please provide an image.")
    if not os.path.exists(image_path):
        raise gr.Error("Provided image path does not exist.")
    if not _runpod_available:
        raise gr.Error("RunPod client not available. Check installation.")
    if not endpoint_id or not endpoint_id.strip():
        raise gr.Error("Please enter your Serverless Endpoint ID.")
    if not api_key or not api_key.strip():
        raise gr.Error("Please enter your RunPod API Key.")
    
    # Parse frames
    try:
        frames_int = int(frames)
    except (ValueError, TypeError):
        frames_int = 121
    
    video_name = video_name.strip() or "gen3c_video"
    resolved_output_dir = output_dir.strip() or GEN3C_DEFAULT_OUTPUT_DIR
    resolved_output_dir = os.path.abspath(os.path.expanduser(resolved_output_dir))
    os.makedirs(resolved_output_dir, exist_ok=True)
    
    logs = []
    logs.append(f"🚀 Starting RunPod Serverless GEN3C generation")
    logs.append(f"   Endpoint: {endpoint_id}")
    logs.append(f"   Frames: {frames_int}, Trajectory: {trajectory}")
    logs.append(f"   Movement: {movement_distance}, Rotation: {camera_rotation}")
    logs.append(f"   Guidance: {guidance}, Foreground Mask: {foreground_masking}")
    
    # Create client
    client = RunPodServerlessClient(endpoint_id.strip(), api_key.strip())
    
    # Check health
    health = client.health_check()
    if health.get("status") != "healthy":
        error_msg = health.get("error", "Unknown error")
        logs.append(f"⚠️ Serverless status: {error_msg}")
    else:
        workers = health.get("workers", {})
        logs.append(f"✓ Endpoint ready: {workers.get('ready', 0)} workers ready")
    
    # Progress callback
    def progress_callback(status: str, elapsed: float):
        elapsed_str = f"{int(elapsed // 60)}m {int(elapsed % 60)}s"
        print(f"[Serverless] {elapsed_str}: {status}")
    
    # Run generation
    try:
        result = client.generate_sync(
            image_path=image_path,
            output_dir=resolved_output_dir,
            video_name=video_name,
            num_frames=frames_int,
            trajectory=trajectory,
            guidance=guidance,
            foreground_masking=foreground_masking,
            seed=seed if seed and seed > 0 else None,
            movement_distance=movement_distance if movement_distance else 0.3,
            camera_rotation=camera_rotation if camera_rotation else "center_facing",
            poll_interval=30,
            max_wait=3600,
            progress_callback=progress_callback
        )
    except Exception as e:
        logs.append(f"❌ Error: {e}")
        return None, "\n".join(logs), f"❌ Serverless error: {e}"
    
    logs.append(result.logs)
    
    if result.success:
        logs.append(f"✅ Video generated in {result.duration_seconds:.1f}s")
        logs.append(f"   Saved to: {result.output_path}")
        return result.output_path, "\n".join(logs), f"✅ Serverless GEN3C complete ({result.duration_seconds:.0f}s)"
    else:
        logs.append(f"❌ Generation failed: {result.error}")
        return None, "\n".join(logs), "❌ Serverless GEN3C failed"


# =============================================================================
# LOCAL EXECUTION
# =============================================================================

def run_gen3c_local(
    image_path: Union[str, None],
    guidance: float,
    frames: Union[str, int, None],
    video_name: str,
    checkpoint_dir: str,
    output_dir: str,
    extra_args: str,
) -> Tuple[Optional[str], str, str]:
    """Launch GEN3C via the wrapper script and return the generated video (local execution)."""
    if not image_path:
        raise gr.Error("Please provide an image.")
    if not os.path.exists(image_path):
        raise gr.Error("Provided image path does not exist.")
    if not os.path.isfile(GEN3C_SCRIPT):
        raise gr.Error(f"GEN3C launcher script not found at {GEN3C_SCRIPT}")

    video_name = video_name.strip() or "gen3c_video"
    resolved_checkpoint = checkpoint_dir.strip() or GEN3C_DEFAULT_CHECKPOINT
    resolved_output_dir = output_dir.strip() or GEN3C_DEFAULT_OUTPUT_DIR
    resolved_checkpoint = os.path.abspath(os.path.expanduser(resolved_checkpoint))
    resolved_output_dir = os.path.abspath(os.path.expanduser(resolved_output_dir))
    os.makedirs(resolved_output_dir, exist_ok=True)

    cmd = [
        "bash",
        GEN3C_SCRIPT,
        "--input",
        image_path,
        "--video-name",
        video_name,
        "--guidance",
        str(guidance),
        "--checkpoint-dir",
        resolved_checkpoint,
        "--output-dir",
        resolved_output_dir,
    ]

    if frames not in (None, "", "None"):
        try:
            frames_int = int(frames)
            cmd.extend(["--frames", str(frames_int)])
        except Exception:
            raise gr.Error("GEN3C frames must be an integer (e.g., 121, 241, 361).")

    if extra_args and extra_args.strip():
        cmd.extend(["--extra", extra_args.strip()])

    launch_msg = f"Launching GEN3C CLI:\n{' '.join(cmd)}"
    print(launch_msg, flush=True)
    logs = [launch_msg]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise gr.Error(f"Failed to run GEN3C launcher: {exc}")

    if result.stdout:
        logs.append("STDOUT:\n" + result.stdout.strip())
    if result.stderr:
        logs.append("STDERR:\n" + result.stderr.strip())

    if result.returncode != 0:
        logs.append(f"❌ GEN3C exited with status {result.returncode}")
        failure_message = "\n\n".join(logs)
        return None, failure_message, "❌ GEN3C generation failed"

    video_path = os.path.join(resolved_output_dir, f"{video_name}.mp4")
    if not os.path.exists(video_path):
        raise gr.Error(f"GEN3C reported success but video not found at {video_path}")

    return video_path, "\n\n".join(logs), "✅ GEN3C video generated"


# =============================================================================
# STATUS CHECKS
# =============================================================================

def check_runpod_status(runpod_url: str) -> str:
    """Check RunPod Pod API status and return formatted string."""
    if not _runpod_available:
        return "❌ RunPod client not installed"
    
    if not runpod_url or not runpod_url.strip():
        return "⚠️ Please enter a RunPod URL"
    
    try:
        client = RunPodGEN3CClient(runpod_url.strip())
        health = client.health_check()
        
        if health.get("status") == "healthy":
            gpu = health.get("gpu", "unknown")
            model_ok = "✓" if health.get("model_exists") else "✗"
            tokenizer_ok = "✓" if health.get("tokenizer_exists") else "✗"
            return f"✅ Connected | GPU: {gpu} | Model: {model_ok} | Tokenizer: {tokenizer_ok}"
        else:
            error = health.get("error", "Unknown error")
            return f"❌ Not ready: {error}"
    except Exception as e:
        return f"❌ Connection failed: {e}"


def check_serverless_status(endpoint_id: str, api_key: str) -> str:
    """Check RunPod Serverless endpoint status and return formatted string."""
    if not _runpod_available:
        return "❌ RunPod client not installed"
    
    if not endpoint_id or not endpoint_id.strip():
        return "⚠️ Please enter an Endpoint ID"
    
    if not api_key or not api_key.strip():
        return "⚠️ Please enter your RunPod API Key"
    
    try:
        client = RunPodServerlessClient(endpoint_id.strip(), api_key.strip())
        health = client.health_check()
        
        if health.get("status") == "healthy":
            workers = health.get("workers", {})
            ready = workers.get("ready", 0)
            running = workers.get("running", 0)
            return f"✅ Endpoint OK | Workers: {ready} ready, {running} running"
        else:
            error = health.get("error", "Unknown error")
            return f"❌ Error: {error}"
    except Exception as e:
        return f"❌ Connection failed: {e}"


def cancel_serverless_job(endpoint_id: str, api_key: str, job_id: Optional[str]) -> Tuple[str, Optional[str]]:
    """Cancel a running serverless job."""
    if not job_id:
        return "⚠️ No active job to cancel", None
    
    if not _runpod_available:
        return "❌ RunPod client not installed", job_id
    
    if not endpoint_id or not api_key:
        return "⚠️ Missing endpoint ID or API key", job_id
    
    try:
        client = RunPodServerlessClient(endpoint_id.strip(), api_key.strip())
        result = client.cancel_job(job_id)
        
        if result.get("success"):
            return f"✅ Job {job_id} cancelled", None
        else:
            return f"❌ Cancel failed: {result.get('error', 'Unknown error')}", job_id
    except Exception as e:
        return f"❌ Cancel failed: {e}", job_id

