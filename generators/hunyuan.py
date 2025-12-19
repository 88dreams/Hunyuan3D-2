"""
Hunyuan3D Generator for 3D Generation Studio

This module provides Hunyuan3D mesh generation functionality:
- Pipeline loading and caching
- Memory optimization
- GLB mesh generation from images
"""

import os
import gc
import time
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

