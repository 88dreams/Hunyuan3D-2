#!/usr/bin/env python3

import os
import gc
from typing import Optional, Tuple, Union

import gradio as gr
import torch
from PIL import Image
import psutil
import time
import threading
import subprocess

from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline


# Repositories and cache configuration
SHAPE_REPO = "tencent/Hunyuan3D-2mini"
CACHE_DIR = "/home/arkrunr/.cache/huggingface/hub"


_shape_pipeline: Optional[Hunyuan3DDiTFlowMatchingPipeline] = None
_current_dtype: Optional[torch.dtype] = None
_current_model: Optional[str] = None

# System monitoring
_system_metrics = {
    "cpu_percent": 0.0,
    "memory_percent": 0.0,
    "gpu_percent": 0.0,
    "gpu_memory_percent": 0.0,
    "last_update": 0.0
}

def _update_system_metrics():
    """Update system metrics every 2 seconds"""
    while True:
        try:
            _system_metrics["cpu_percent"] = psutil.cpu_percent(interval=1)
            _system_metrics["memory_percent"] = psutil.virtual_memory().percent
            _system_metrics["gpu_percent"] = 0.0
            _system_metrics["gpu_memory_percent"] = 0.0

            # Try ROCm/AMD GPU monitoring
            try:
                import subprocess
                result = subprocess.run(['rocm-smi', '--showuse'], capture_output=True, text=True, timeout=2)
                if result.returncode == 0:
                    lines = result.stdout.split('\n')
                    for line in lines:
                        if 'GPU use' in line:
                            parts = line.split()
                            if len(parts) >= 3:
                                try:
                                    _system_metrics["gpu_percent"] = float(parts[-1].rstrip('%'))
                                except:
                                    pass
            except:
                pass

            _system_metrics["last_update"] = time.time()
        except Exception:
            pass

        time.sleep(2.0)

def get_system_metrics():
    """Get current system metrics for display"""
    return _system_metrics

def _format_system_metrics(metrics):
    """Format system metrics for display"""
    cpu = metrics.get("cpu_percent", 0)
    mem = metrics.get("memory_percent", 0)
    gpu = metrics.get("gpu_percent", 0)
    gpu_mem = metrics.get("gpu_memory_percent", 0)

    def color_code(value, thresholds=[50, 80]):
        if value >= thresholds[1]:
            return f"🔴 {value:.1f}%"
        elif value >= thresholds[0]:
            return f"🟡 {value:.1f}%"
        else:
            return f"🟢 {value:.1f}%"

    lines = [
        f"CPU: {color_code(cpu)} | Memory: {color_code(mem)}",
        f"GPU: {color_code(gpu)} | GPU Memory: {color_code(gpu_mem)}"
    ]
    return "\n".join(lines)

# Background monitoring thread (initialized later)
_monitor_thread = None


def _apply_memory_savers(pipeline: object, attention_slicing: bool, cpu_offload: bool, dtype: torch.dtype) -> None:
    """Apply memory optimizations when available on the pipeline instance."""
    if hasattr(pipeline, "to"):
        pipeline.to(dtype=dtype)
    if attention_slicing and hasattr(pipeline, "enable_attention_slicing"):
        pipeline.enable_attention_slicing()
    if cpu_offload and hasattr(pipeline, "enable_sequential_cpu_offload"):
        pipeline.enable_sequential_cpu_offload()


def _ensure_pipelines(
    model_choice: str,
    use_fp16: bool,
    attention_slicing: bool,
    cpu_offload: bool,
) -> torch.dtype:
    """Load and cache the shape pipeline if needed, configure dtype and memory options."""
    global _shape_pipeline, _current_dtype

    requested_dtype = torch.float16 if use_fp16 else torch.float32

    # Determine which model to use
    global _current_model
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


def _load_image(image_path: str) -> Image.Image:
    img = Image.open(image_path)
    return img.convert("RGB")


def run(
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
) -> Tuple[str, str, str]:
    """Shape generation entrypoint. Returns (output_file_path, logs)."""
    import time

    if not image_path:
        raise gr.Error("Please provide an image.")
    if not os.path.exists(image_path):
        raise gr.Error("Provided image path does not exist.")

    if not output_name:
        output_name = "output_model"

    # Handle save location
    if save_location and save_location.strip():
        # Ensure save location exists
        os.makedirs(save_location, exist_ok=True)
        # Use absolute path for save location
        save_dir = os.path.abspath(save_location)
    else:
        # Default to current directory
        save_dir = os.getcwd()

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

    _ensure_pipelines(model_choice, use_fp16, attention_slicing, cpu_offload)

    img = _load_image(image_path)

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

    # Set timeout (45 minutes for shape generation to handle complex architectural models at high resolution)
    timeout_duration = 2700
    start_time = time.time()

    try:
        # Check timeout periodically during generation
        def check_timeout():
            if time.time() - start_time > timeout_duration:
                raise gr.Error(f"Generation timed out after {timeout_duration} seconds. Try reducing steps or enabling memory optimizations.")

        logs.append("Generating 3D shape...")
        check_timeout()

        import sys
        from io import StringIO
        
        # Capture progress output
        progress_status = "🔄 Stage 1/3: Diffusion sampling (~40s)..."
        logs.append(progress_status)
        # This will be updated below with actual values
        
        # Run the pipeline (this includes diffusion + volume decoding)
        logs.append("Running diffusion and volume decoding (this will take ~20-25 minutes)...")
        
        # Adaptive resolution based on model for maximum quality
        if "Mini Model" in model_choice:
            octree_res = 380  # Maximum quality for mini model
            chunks = 6000     # Optimized for resolution 380
            logs.append("Using maximum quality settings for Mini Model (resolution=380)")
        else:  # Full Model
            octree_res = 360  # Higher quality with smaller chunks
            chunks = 5000     # Smaller chunks allow higher resolution
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
            return None, f"❌ Error: Out of GPU memory. Enable CPU offload / FP16 / attention slicing, or reduce steps.\n\n{error_msg}", "❌ Failed: Out of memory"
        return None, f"❌ Error: {error_msg}", "❌ Generation failed"
    except gr.Error:
        raise  # Re-raise Gradio errors (like timeout)
    except Exception as e:
        return None, f"❌ Unexpected error: {str(e)}", "❌ Generation failed"
    finally:
        # Proactively release GPU memory
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


with gr.Blocks(title="Hunyuan3D 2D→3D") as demo:
    gr.Markdown("## Hunyuan3D 2D→3D\nUpload an image, set options, and generate a GLB.")

    with gr.Row():
        # Left column - Image and system metrics
        with gr.Column(scale=1):
            image = gr.Image(type="filepath", label="Input image")

            # System monitoring display (under image)
            system_metrics = gr.Textbox(
                label="System Resources (updates every 2s)",
                value="Loading system metrics...",
                interactive=False,
                lines=2
            )

            # Progress display (shows generation stages)
            progress_display = gr.Textbox(
                label="Generation Progress",
                value="Ready to generate...",
                interactive=False,
                lines=1
            )

        # Right column - Controls
        with gr.Column(scale=1):
            guidance_scale = gr.Slider(1.0, 15.0, value=9.0, step=0.5, label="Guidance scale")
            steps = gr.Slider(10, 100, value=40, step=1, label="Inference steps")
            seed = gr.Number(value=42, precision=0, label="Seed (optional)")
            model_choice = gr.Radio(
                choices=["Mini Model (Faster)", "Full Model (Higher Quality)"],
                value="Mini Model (Faster)",
                label="Model Selection",
                info="Note: Full model may occasionally have GPU memory issues. If it hangs, use Mini model."
            )
            use_fp16 = gr.Checkbox(value=True, label="Use FP16 (half precision)")
            attention_slicing = gr.Checkbox(value=True, label="Enable attention slicing")
            cpu_offload = gr.Checkbox(value=True, label="Enable CPU offload")
            remove_background = gr.Checkbox(value=False, label="Remove background automatically")
            output_name = gr.Textbox(value="output_model", label="Output name (no extension)")
            save_location = gr.Textbox(value="", label="Save location (optional - leave empty for current directory)", placeholder="/path/to/save/directory")
            run_btn = gr.Button("Generate 3D Shape", variant="primary")

    with gr.Row():
        output_file = gr.File(label="Output GLB")
        logs_box = gr.Textbox(label="Logs", lines=8)

    # Add progress indicator
    progress_bar = gr.Progress()

    run_btn.click(
        fn=run,
        inputs=[image, guidance_scale, steps, seed, model_choice, use_fp16, attention_slicing, cpu_offload, remove_background, output_name, save_location],
        outputs=[output_file, logs_box, progress_display],
    )

    # Timer to update system metrics every 2 seconds
    def update_metrics():
        return _format_system_metrics(get_system_metrics())

    timer = gr.Timer(2.0)
    timer.tick(fn=update_metrics, outputs=[system_metrics])

if __name__ == "__main__":
    # Start monitoring thread when Gradio launches
    _monitor_thread = threading.Thread(target=_update_system_metrics, daemon=True)
    _monitor_thread.start()
    
    # Give monitoring thread time to start
    time.sleep(0.5)

    # Enable queuing to avoid overlapping runs consuming VRAM
    # (no args for compatibility with your Gradio version)
    demo.queue()
    demo.launch(server_port=7860, share=False)
    