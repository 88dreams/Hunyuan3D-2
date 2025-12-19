"""
Lyra Generator for 3D Generation Studio

NVIDIA's Lyra: Image/Video → 3D/4D Gaussian Splatting
Built on GEN3C video diffusion + 3DGS reconstruction decoder.

This module provides:
- run_lyra_runpod: Execute Lyra on RunPod serverless
- check_lyra_status: Check Lyra availability

Lyra Pipeline:
1. Input image/video
2. GEN3C generates multi-view video (diffusion phase)
3. 3DGS decoder reconstructs Gaussian splats (reconstruction phase)
4. Output: PLY file + optional rendered video
"""

import os
import base64
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

# Default output directory
LYRA_DEFAULT_OUTPUT_DIR = os.environ.get(
    "LYRA_OUTPUT_DIR",
    "/srv/searidge_share/outputs/lyra"
)


def check_lyra_status(endpoint_id: str = "", api_key: str = "") -> str:
    """
    Check Lyra availability on RunPod.
    
    Args:
        endpoint_id: RunPod serverless endpoint ID
        api_key: RunPod API key
    
    Returns:
        Status message string
    """
    if not endpoint_id or not api_key:
        return "⚠️ Enter RunPod credentials to use Lyra"
    
    try:
        # Import the unified client
        from runpod.runpod_client import UnifiedServerlessClient
        
        client = UnifiedServerlessClient(
            endpoint_id=endpoint_id,
            api_key=api_key,
        )
        
        health = client.health_check()
        
        if health.get("status") == "healthy":
            models = health.get("available_models", [])
            if "lyra" in models:
                return "✅ Lyra available on RunPod"
            else:
                return f"⚠️ Endpoint healthy but Lyra not listed. Available: {models}"
        else:
            return f"❌ Endpoint unhealthy: {health.get('message', 'Unknown error')}"
            
    except ImportError:
        return "❌ RunPod client not available"
    except Exception as e:
        return f"❌ Connection error: {str(e)}"


def run_lyra_runpod(
    image_path: Optional[str] = None,
    video_path: Optional[str] = None,
    endpoint_id: str = "",
    api_key: str = "",
    generation_mode: str = "Static (Image → 3DGS)",
    num_views: int = 8,
    camera_motion: float = 1.0,
    multi_trajectory: bool = True,
    foreground_masking: bool = True,
    num_gaussians: int = 100000,
    seed: Optional[int] = None,
    output_name: str = "lyra_output",
    output_dir: str = LYRA_DEFAULT_OUTPUT_DIR,
    output_ply: bool = True,
    output_video: bool = True,
) -> Tuple[Optional[str], str, str]:
    """
    Run Lyra inference on RunPod serverless.
    
    Args:
        image_path: Path to input image (for static mode)
        video_path: Path to input video (for dynamic mode)
        endpoint_id: RunPod serverless endpoint ID
        api_key: RunPod API key
        generation_mode: "Static (Image → 3DGS)" or "Dynamic (Video → 4DGS)"
        num_views: Number of multi-view camera positions
        camera_motion: Camera motion scale (1.0 = normal)
        multi_trajectory: Generate multiple camera trajectories
        foreground_masking: Apply foreground masking
        num_gaussians: Maximum number of Gaussian splats
        seed: Random seed (None for random)
        output_name: Base name for output files
        output_dir: Directory to save outputs
        output_ply: Export PLY file
        output_video: Export rendered video
    
    Returns:
        Tuple of (output_path, logs, progress_message)
    """
    logs = []
    
    # Validate inputs
    is_static = "Static" in generation_mode
    
    if is_static:
        if not image_path or not os.path.exists(image_path):
            return None, "Error: No input image provided", "❌ No input image"
        input_path = image_path
        input_type = "image"
    else:
        if not video_path or not os.path.exists(video_path):
            return None, "Error: No input video provided", "❌ No input video"
        input_path = video_path
        input_type = "video"
    
    if not endpoint_id or not api_key:
        return None, "Error: RunPod credentials required", "❌ Missing credentials"
    
    logs.append(f"[Lyra] Mode: {generation_mode}")
    logs.append(f"[Lyra] Input: {input_path}")
    logs.append(f"[Lyra] Views: {num_views}, Motion: {camera_motion}")
    
    try:
        # Import the unified client
        from runpod.runpod_client import UnifiedServerlessClient
        
        client = UnifiedServerlessClient(
            endpoint_id=endpoint_id,
            api_key=api_key,
        )
        
        # Read and encode input
        with open(input_path, "rb") as f:
            input_base64 = base64.b64encode(f.read()).decode("utf-8")
        
        logs.append(f"[Lyra] Encoded input ({len(input_base64)} bytes)")
        logs.append(f"[Lyra] Submitting to RunPod...")
        
        # Build job parameters
        job_params = {
            "model": "lyra",
            f"{input_type}_base64": input_base64,
            "output_name": output_name,
            "generation_mode": "static" if is_static else "dynamic",
            "num_views": num_views,
            "camera_motion_scale": camera_motion,
            "multi_trajectory": multi_trajectory,
            "foreground_masking": foreground_masking,
            "max_gaussians": num_gaussians,
            "output_ply": output_ply,
            "output_video": output_video,
            "return_base64": True,
        }
        
        if seed is not None:
            job_params["seed"] = int(seed)
        
        # Submit job
        result = client.generate_sync(job_params, timeout=7200)  # 2 hour timeout
        
        if result.status == "error":
            error_msg = result.error or "Unknown error"
            logs.append(f"[Lyra] Error: {error_msg}")
            return None, "\n".join(logs), f"❌ {error_msg}"
        
        logs.append(f"[Lyra] Job completed successfully")
        
        # Ensure output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        output_path = None
        
        # Save PLY output
        if result.ply_base64:
            ply_path = os.path.join(output_dir, f"{output_name}.ply")
            with open(ply_path, "wb") as f:
                f.write(base64.b64decode(result.ply_base64))
            logs.append(f"[Lyra] Saved PLY: {ply_path}")
            output_path = ply_path
        
        # Save video output
        if result.video_base64:
            video_out_path = os.path.join(output_dir, f"{output_name}.mp4")
            with open(video_out_path, "wb") as f:
                f.write(base64.b64decode(result.video_base64))
            logs.append(f"[Lyra] Saved video: {video_out_path}")
            if not output_path:
                output_path = video_out_path
        
        if output_path:
            return output_path, "\n".join(logs), "✅ Generation complete!"
        else:
            logs.append("[Lyra] Warning: No output files received")
            return None, "\n".join(logs), "⚠️ No output files"
        
    except ImportError as e:
        logs.append(f"[Lyra] Import error: {e}")
        return None, "\n".join(logs), "❌ RunPod client not available"
    except Exception as e:
        logs.append(f"[Lyra] Error: {str(e)}")
        return None, "\n".join(logs), f"❌ {str(e)}"


def check_lyra_installation() -> str:
    """
    Check if Lyra is available locally.
    
    Note: Lyra requires H100/A100 GPUs and is not typically run locally.
    This is mainly for status display purposes.
    
    Returns:
        Status message string
    """
    # Lyra is too heavy for local execution on consumer GPUs
    return "⚠️ Lyra requires H100/A100 GPUs. Use RunPod Serverless."

