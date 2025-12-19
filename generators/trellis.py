"""
TRELLIS.2 Generator for 3D Generation Studio

Microsoft's TRELLIS.2: Image → High-Quality 3D with O-Voxel representation
4B parameter model for high-fidelity 3D generation with PBR materials.

This module provides:
- run_trellis_runpod: Execute TRELLIS.2 on RunPod serverless
- check_trellis_status: Check TRELLIS.2 availability

TRELLIS.2 Features:
- O-Voxel representation (Native & Compact Structured Latents)
- GLB output with PBR materials (Base Color, Roughness, Metallic, Opacity)
- Multiple resolution options (512³, 1024³, 1536³)
"""

import os
import base64
from pathlib import Path
from typing import Optional, Tuple

# Default output directory
TRELLIS_DEFAULT_OUTPUT_DIR = os.environ.get(
    "TRELLIS_OUTPUT_DIR",
    "/srv/searidge_share/outputs/trellis"
)


def check_trellis_status(endpoint_id: str = "", api_key: str = "") -> str:
    """
    Check TRELLIS.2 availability on RunPod.
    
    Args:
        endpoint_id: RunPod serverless endpoint ID
        api_key: RunPod API key
    
    Returns:
        Status message string
    """
    if not endpoint_id or not api_key:
        return "⚠️ Enter RunPod credentials to use TRELLIS.2"
    
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
            if "trellis" in models:
                return "✅ TRELLIS.2 available on RunPod"
            else:
                return f"⚠️ Endpoint healthy but TRELLIS.2 not listed. Available: {models}"
        else:
            return f"❌ Endpoint unhealthy: {health.get('message', 'Unknown error')}"
            
    except ImportError:
        return "❌ RunPod client not available"
    except Exception as e:
        return f"❌ Connection error: {str(e)}"


def _parse_resolution(resolution_str: str) -> int:
    """Parse resolution string like '1024³ (~17s)' to integer 1024."""
    if "512" in resolution_str:
        return 512
    elif "1024" in resolution_str:
        return 1024
    elif "1536" in resolution_str:
        return 1536
    else:
        return 1024  # default


def run_trellis_runpod(
    image_path: Optional[str] = None,
    endpoint_id: str = "",
    api_key: str = "",
    resolution: str = "1024³ (~17s)",
    guidance_scale: float = 7.5,
    seed: Optional[int] = None,
    output_name: str = "trellis_output",
    output_dir: str = TRELLIS_DEFAULT_OUTPUT_DIR,
    output_format: str = "GLB (with PBR)",
) -> Tuple[Optional[str], str, str]:
    """
    Run TRELLIS.2 inference on RunPod serverless.
    
    Args:
        image_path: Path to input image
        endpoint_id: RunPod serverless endpoint ID
        api_key: RunPod API key
        resolution: Resolution string ("512³", "1024³", or "1536³")
        guidance_scale: Classifier-free guidance scale
        seed: Random seed (None for random)
        output_name: Base name for output files
        output_dir: Directory to save outputs
        output_format: "GLB (with PBR)" or "PLY (geometry only)"
    
    Returns:
        Tuple of (output_path, logs, progress_message)
    """
    logs = []
    
    # Validate inputs
    if not image_path or not os.path.exists(image_path):
        return None, "Error: No input image provided", "❌ No input image"
    
    if not endpoint_id or not api_key:
        return None, "Error: RunPod credentials required", "❌ Missing credentials"
    
    res_value = _parse_resolution(resolution)
    export_glb = "GLB" in output_format
    
    logs.append(f"[TRELLIS.2] Resolution: {res_value}³")
    logs.append(f"[TRELLIS.2] Guidance: {guidance_scale}")
    logs.append(f"[TRELLIS.2] Format: {'GLB' if export_glb else 'PLY'}")
    logs.append(f"[TRELLIS.2] Input: {image_path}")
    
    try:
        # Import the unified client
        from runpod.runpod_client import UnifiedServerlessClient
        
        client = UnifiedServerlessClient(
            endpoint_id=endpoint_id,
            api_key=api_key,
        )
        
        # Read and encode input image
        with open(image_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode("utf-8")
        
        logs.append(f"[TRELLIS.2] Encoded image ({len(image_base64)} bytes)")
        logs.append(f"[TRELLIS.2] Submitting to RunPod...")
        
        # Build job parameters
        job_params = {
            "model": "trellis",
            "image_base64": image_base64,
            "output_name": output_name,
            "resolution": res_value,
            "guidance_scale": guidance_scale,
            "output_glb": export_glb,
            "output_ply": not export_glb,
            "return_base64": True,
        }
        
        if seed is not None:
            job_params["seed"] = int(seed)
        
        # Estimate timeout based on resolution
        timeout_map = {512: 300, 1024: 600, 1536: 1800}  # seconds
        timeout = timeout_map.get(res_value, 600)
        
        # Submit job
        result = client.generate_sync(job_params, timeout=timeout)
        
        if result.status == "error":
            error_msg = result.error or "Unknown error"
            logs.append(f"[TRELLIS.2] Error: {error_msg}")
            return None, "\n".join(logs), f"❌ {error_msg}"
        
        logs.append(f"[TRELLIS.2] Job completed successfully")
        
        # Ensure output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        output_path = None
        
        # Save GLB output
        if hasattr(result, 'glb_base64') and result.glb_base64:
            glb_path = os.path.join(output_dir, f"{output_name}.glb")
            with open(glb_path, "wb") as f:
                f.write(base64.b64decode(result.glb_base64))
            logs.append(f"[TRELLIS.2] Saved GLB: {glb_path}")
            output_path = glb_path
        
        # Save PLY output
        if hasattr(result, 'ply_base64') and result.ply_base64:
            ply_path = os.path.join(output_dir, f"{output_name}.ply")
            with open(ply_path, "wb") as f:
                f.write(base64.b64decode(result.ply_base64))
            logs.append(f"[TRELLIS.2] Saved PLY: {ply_path}")
            if not output_path:
                output_path = ply_path
        
        if output_path:
            return output_path, "\n".join(logs), "✅ Generation complete!"
        else:
            logs.append("[TRELLIS.2] Warning: No output files received")
            return None, "\n".join(logs), "⚠️ No output files"
        
    except ImportError as e:
        logs.append(f"[TRELLIS.2] Import error: {e}")
        return None, "\n".join(logs), "❌ RunPod client not available"
    except Exception as e:
        logs.append(f"[TRELLIS.2] Error: {str(e)}")
        return None, "\n".join(logs), f"❌ {str(e)}"


def check_trellis_installation() -> str:
    """
    Check if TRELLIS.2 is available locally.
    
    Note: TRELLIS.2 requires H100 GPUs (4B parameters) and is not typically run locally.
    This is mainly for status display purposes.
    
    Returns:
        Status message string
    """
    # TRELLIS.2 is too heavy for local execution on consumer GPUs
    return "⚠️ TRELLIS.2 requires H100 GPUs (4B params). Use RunPod Serverless."

