"""
Hunyuan3D Generator for 3D Generation Studio

This module provides Hunyuan3D mesh generation functionality:
- Pipeline loading and caching (Local)
- Memory optimization (Local)
- GLB mesh generation from images
- RunPod Serverless execution

Supports both Local CUDA and RunPod Serverless modes.
"""

import os
import gc
import time
import base64
from typing import Optional, Tuple, Union

import torch  # type: ignore
import gradio as gr  # type: ignore
from PIL import Image

from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

from utils.image_utils import load_image


# =============================================================================
# CONFIGURATION
# =============================================================================

# Try to load from config module, fall back to defaults
try:
    from config import Config
    _cfg = Config()
    CACHE_DIR = _cfg.hf_cache_dir
    HUNYUAN_DEFAULT_OUTPUT_DIR = _cfg.hunyuan_outputs
except ImportError:
    CACHE_DIR = "/srv/searidge_share/checkpoints/huggingface"
    HUNYUAN_DEFAULT_OUTPUT_DIR = "/srv/searidge_share/outputs/hunyuan"

# Ensure output directory exists
os.makedirs(HUNYUAN_DEFAULT_OUTPUT_DIR, exist_ok=True)


# =============================================================================
# PIPELINE MANAGEMENT
# =============================================================================

_shape_pipeline: Optional[Hunyuan3DDiTFlowMatchingPipeline] = None
_current_dtype: Optional[torch.dtype] = None
_current_model: Optional[str] = None


def _apply_memory_savers(
    pipeline: object, 
    attention_slicing: bool, 
    cpu_offload: bool, 
    dtype: torch.dtype
) -> None:
    """Apply memory optimizations when available on the pipeline instance."""
    if hasattr(pipeline, "to"):
        pipeline.to(dtype=dtype)
    if attention_slicing and hasattr(pipeline, "enable_attention_slicing"):
        pipeline.enable_attention_slicing()
    if cpu_offload and hasattr(pipeline, "enable_sequential_cpu_offload"):
        pipeline.enable_sequential_cpu_offload()


def ensure_pipeline(
    model_choice: str,
    use_fp16: bool,
    attention_slicing: bool,
    cpu_offload: bool,
) -> torch.dtype:
    """Load and cache the shape pipeline if needed, configure dtype and memory options."""
    global _shape_pipeline, _current_dtype, _current_model

    requested_dtype = torch.float16 if use_fp16 else torch.float32

    # Determine which model to use
    selected_model = "tencent/Hunyuan3D-2" if "Full Model" in model_choice else "tencent/Hunyuan3D-2mini"

    # Load or reload shape pipeline when model or dtype changes
    if _shape_pipeline is None or _current_dtype != requested_dtype or _current_model != selected_model:
        print(f"Loading shape pipeline (model={selected_model}, dtype={'fp16' if use_fp16 else 'fp32'})...")

        # Determine subfolder based on model
        if "mini" in selected_model:
            subfolder = "hunyuan3d-dit-v2-mini-turbo"
        else:
            subfolder = "hunyuan3d-dit-v2-0"

        _shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            selected_model,
            subfolder=subfolder,
            cache_dir=CACHE_DIR,
            torch_dtype=requested_dtype,
        )
        _current_dtype = requested_dtype
        _current_model = selected_model

    _apply_memory_savers(_shape_pipeline, attention_slicing, cpu_offload, requested_dtype)

    return requested_dtype


def get_pipeline() -> Optional[Hunyuan3DDiTFlowMatchingPipeline]:
    """Get the current pipeline instance."""
    return _shape_pipeline


# =============================================================================
# GENERATION
# =============================================================================

def run_hunyuan(
    image_path: Union[str, None],
    guidance_scale: float,
    steps: int,
    seed: Optional[int],
    model_choice: str,
    use_fp16: bool,
    attention_slicing: bool,
    cpu_offload: bool,
    remove_background: bool,
    output_name: str,
    save_location: str,
) -> Tuple[Optional[str], str, str]:
    """
    Shape generation entrypoint.
    
    Returns:
        Tuple of (output_file_path, logs, progress_status)
    """
    if not image_path:
        raise gr.Error("Please provide an image.")
    if not os.path.exists(image_path):
        raise gr.Error("Provided image path does not exist.")

    if not output_name:
        output_name = "output_model"

    # Handle save location
    if save_location and save_location.strip():
        save_dir = os.path.abspath(os.path.expanduser(save_location.strip()))
    else:
        save_dir = HUNYUAN_DEFAULT_OUTPUT_DIR
    os.makedirs(save_dir, exist_ok=True)

    # Normalize names
    base_out = os.path.splitext(output_name)[0]
    output_path = os.path.join(save_dir, f"{base_out}_shape.glb")

    # Optional determinism
    if seed is not None and str(seed).strip() != "":
        try:
            torch.manual_seed(int(seed))
        except Exception:
            pass

    # Aggressive GPU memory cleanup before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()

    ensure_pipeline(model_choice, use_fp16, attention_slicing, cpu_offload)

    img = load_image(image_path)

    logs = []
    logs.append("GPU memory cleared and synchronized")

    # Apply background removal if requested
    if remove_background:
        logs.append("Removing background...")
        from hy3dgen.rembg import BackgroundRemover
        rembg = BackgroundRemover()
        img = rembg(img.convert("RGB")).convert("RGBA")
        logs.append("Background removed")

        # Clear memory after background removal
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    mesh = None

    # Set timeout (45 minutes for shape generation)
    timeout_duration = 2700
    start_time = time.time()

    try:
        def check_timeout():
            if time.time() - start_time > timeout_duration:
                raise gr.Error(f"Generation timed out after {timeout_duration} seconds.")

        logs.append("Generating 3D shape...")
        check_timeout()

        progress_status = "🔄 Stage 1/3: Diffusion sampling (~40s)..."
        logs.append(progress_status)
        
        logs.append("Running diffusion and volume decoding (this will take ~20-25 minutes)...")
        
        # Adaptive resolution based on model for maximum quality
        if "Mini Model" in model_choice:
            octree_res = 380
            chunks = 6000
            logs.append("Using maximum quality settings for Mini Model (resolution=380)")
        else:
            octree_res = 360
            chunks = 5000
            logs.append("Using high quality settings for Full Model (resolution=360, chunks=5000)")
        
        logs.append(f"Using octree_resolution={octree_res}, num_chunks={chunks} for {model_choice}")
        
        result = _shape_pipeline(
            image=img,
            guidance_scale=float(guidance_scale),
            num_inference_steps=int(steps),
            octree_resolution=octree_res,
            num_chunks=chunks,
        )
        
        progress_status = "🔄 Stage 3/3: Extracting surface mesh (5-7 minutes, CPU-intensive)..."
        logs.append("Volume decoding complete!")
        logs.append(progress_status)
        logs.append("Note: Surface extraction runs on CPU and has no progress bar")
        logs.append("Please wait patiently - this is normal and will complete...")
        
        # Clear GPU memory before surface extraction
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
            logs.append("GPU memory cleared before surface extraction")
        
        mesh = result[0] if isinstance(result, (list, tuple)) else result
        
        progress_status = "💾 Exporting GLB file..."
        logs.append("Surface extraction complete!")
        logs.append(progress_status)
        
        mesh.export(output_path)
        logs.append(f"✅ Saved shape: {output_path}")
        
        progress_status = "✅ Generation complete!"
        check_timeout()

        return output_path, "\n".join(logs), progress_status

    except RuntimeError as e:
        error_msg = str(e)
        if "out of memory" in error_msg.lower():
            return None, f"❌ Error: Out of GPU memory. Enable CPU offload / FP16 / attention slicing.\n\n{error_msg}", "❌ Failed: Out of memory"
        return None, f"❌ Error: {error_msg}", "❌ Generation failed"
    except gr.Error:
        raise
    except Exception as e:
        return None, f"❌ Unexpected error: {str(e)}", "❌ Generation failed"
    finally:
        try:
            del mesh
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass


# =============================================================================
# RUNPOD SERVERLESS EXECUTION
# =============================================================================

def check_hunyuan_runpod_status(endpoint_id: str = "", api_key: str = "") -> str:
    """
    Check Hunyuan3D availability on RunPod.
    
    Args:
        endpoint_id: RunPod serverless endpoint ID
        api_key: RunPod API key
    
    Returns:
        Status message string
    """
    if not endpoint_id or not api_key:
        return "⚠️ Enter RunPod credentials to use Hunyuan3D on cloud"
    
    try:
        from runpod.runpod_client import UnifiedServerlessClient
        
        client = UnifiedServerlessClient(
            endpoint_id=endpoint_id,
            api_key=api_key,
        )
        
        health = client.health_check()
        
        if health.get("status") == "healthy":
            return "✅ Hunyuan3D endpoint available"
        else:
            return f"❌ Endpoint unhealthy: {health.get('message', 'Unknown error')}"
            
    except ImportError:
        return "❌ RunPod client not available"
    except Exception as e:
        return f"❌ Connection error: {str(e)}"


def run_hunyuan_runpod(
    image_path: Optional[str] = None,
    endpoint_id: str = "",
    api_key: str = "",
    model_choice: str = "Mini Model (Faster)",
    guidance_scale: float = 9.0,
    steps: int = 40,
    octree_resolution: int = 380,
    seed: Optional[int] = None,
    remove_background: bool = True,
    output_name: str = "hunyuan_output",
    output_dir: str = HUNYUAN_DEFAULT_OUTPUT_DIR,
) -> Tuple[Optional[str], str, str]:
    """
    Run Hunyuan3D inference on RunPod serverless.
    
    Args:
        image_path: Path to input image
        endpoint_id: RunPod serverless endpoint ID
        api_key: RunPod API key
        model_choice: "Mini Model (Faster)" or "Full Model (Higher Quality)"
        guidance_scale: Guidance scale for generation
        steps: Number of inference steps
        octree_resolution: Octree resolution for mesh extraction
        seed: Random seed (None for random)
        remove_background: Whether to remove background
        output_name: Base name for output files
        output_dir: Directory to save outputs
    
    Returns:
        Tuple of (output_path, logs, progress_message)
    """
    logs = []
    
    # Validate inputs
    if not image_path or not os.path.exists(image_path):
        return None, "Error: No input image provided", "❌ No input image"
    
    if not endpoint_id or not api_key:
        return None, "Error: RunPod credentials required", "❌ Missing credentials"
    
    # Determine model type
    model_type = "full" if "Full" in model_choice else "mini"
    
    logs.append(f"[Hunyuan3D] Model: {model_type}")
    logs.append(f"[Hunyuan3D] Steps: {steps}, Guidance: {guidance_scale}")
    logs.append(f"[Hunyuan3D] Resolution: {octree_resolution}")
    logs.append(f"[Hunyuan3D] Input: {image_path}")
    
    try:
        from runpod.runpod_client import UnifiedServerlessClient
        
        client = UnifiedServerlessClient(
            endpoint_id=endpoint_id,
            api_key=api_key,
        )
        
        logs.append(f"[Hunyuan3D] Submitting to RunPod...")
        
        # Encode image
        with open(image_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode("utf-8")
        
        # Build job payload
        job_input = {
            "model": "hunyuan",
            "image_base64": image_base64,
            "model_variant": model_type,
            "guidance_scale": guidance_scale,
            "steps": steps,
            "octree_resolution": octree_resolution,
            "remove_background": remove_background,
            "output_name": output_name,
            "return_base64": False,  # Use S3 for large GLB files
        }
        
        if seed is not None:
            job_input["seed"] = int(seed)
        
        # Submit job
        submit_result = client.submit_generic_job(job_input)
        job_id = submit_result.get("job_id", "")
        logs.append(f"[Hunyuan3D] Job submitted: {job_id}")
        
        # Wait for completion (Hunyuan can take 5-20 minutes)
        timeout = 2700 if model_type == "full" else 1200  # 45 min for full, 20 min for mini
        
        final_status = client.wait_for_completion(
            job_id=job_id,
            poll_interval=30,
            max_wait=timeout,
        )
        
        if final_status.get("status") == "completed":
            logs.append(f"[Hunyuan3D] Job completed successfully")
            
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            output_path = None
            
            # Handle S3 download
            if final_status.get("glb_s3_url"):
                from runpod.runpod_client import download_from_s3
                local_path, s3_msg = download_from_s3(final_status["glb_s3_url"], output_dir)
                logs.append(s3_msg)
                if local_path:
                    output_path = local_path
            
            # Handle base64 (for small files)
            elif final_status.get("glb_base64"):
                output_path = os.path.join(output_dir, f"{output_name}.glb")
                with open(output_path, "wb") as f:
                    f.write(base64.b64decode(final_status["glb_base64"]))
                logs.append(f"[Hunyuan3D] GLB saved: {output_path}")
            
            if output_path:
                return output_path, "\n".join(logs), "✅ Generation complete!"
            else:
                logs.append("[Hunyuan3D] Warning: No output files received")
                return None, "\n".join(logs), "⚠️ No output files"
        else:
            error = final_status.get("error", "Unknown error")
            logs.append(f"[Hunyuan3D] Error: {error}")
            return None, "\n".join(logs), f"❌ {error}"
        
    except ImportError as e:
        logs.append(f"[Hunyuan3D] Import error: {e}")
        return None, "\n".join(logs), "❌ RunPod client not available"
    except Exception as e:
        logs.append(f"[Hunyuan3D] Error: {str(e)}")
        return None, "\n".join(logs), f"❌ {str(e)}"

