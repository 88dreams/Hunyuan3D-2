#!/usr/bin/env python3
# pyright: reportMissingImports=false
print("DEBUG: STARTING NEW VERSION OF 2D3D.PY (Port 5683)")
import os
import gc
import tempfile
from typing import Optional, Tuple, Union

import gradio as gr  # type: ignore
import torch  # type: ignore
from PIL import Image
import psutil  # type: ignore
import time
import threading
import subprocess

from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

# Repositories and cache configuration
SHAPE_REPO = "tencent/Hunyuan3D-2mini"
CACHE_DIR = "/home/arkrunr/.cache/huggingface/hub"

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
GEN3C_SCRIPT = os.path.join(PROJECT_ROOT, "scripts", "run_gen3c.sh")
GEN3C_DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "assets", "gen3c_outputs")
GEN3C_DEFAULT_CHECKPOINT = "/home/arkrunr/GEN3C/checkpoints"
HUNYUAN_DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "assets", "hunyuan_outputs")


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
def _clamp_scale_value(value: Union[float, int, None]) -> float:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return 1.0
    return min(1.0, max(0.25, val))


def _get_image_dimensions(image_path: Optional[str]) -> Tuple[Optional[int], Optional[int]]:
    if not image_path or not os.path.exists(image_path):
        return None, None
    try:
        with Image.open(image_path) as img:
            return img.width, img.height
    except Exception:
        return None, None


def _format_resize_text(
    width: Optional[int],
    height: Optional[int],
    scale: Union[float, int, None],
) -> str:
    if width is None or height is None:
        return "No image loaded."
    scale = _clamp_scale_value(scale)
    target_w = max(64, int(round(width * scale)))
    target_h = max(64, int(round(height * scale)))
    percent = int(round(scale * 100))
    return f"Original: {width}×{height}\nTarget ({percent}%): {target_w}×{target_h}"


def update_image_info_display(image_path: Optional[str], scale: Union[float, int, None]):
    width, height = _get_image_dimensions(image_path)
    return _format_resize_text(width, height, scale or 1.0)


def _safe_preview_update(image_path: Optional[str] = None, scale: Union[float, int, None] = None):
    return update_image_info_display(image_path, scale or 1.0)


def poll_image_preview(*args, **kwargs):
    image_path = None
    scale = 1.0
    if args:
        image_path = args[0]
    if len(args) > 1:
        scale = args[1]
    scale = kwargs.get("scale", scale)
    return update_image_info_display(image_path, scale), (None, None), None


def _maybe_downscale_image(image_path: Optional[str], scale: Union[float, int, None]) -> Tuple[Optional[str], Optional[str]]:
    scale = _clamp_scale_value(scale)
    if not image_path or scale >= 0.999:
        return image_path, None
    try:
        with Image.open(image_path) as img:
            target_w = max(64, int(round(img.width * scale)))
            target_h = max(64, int(round(img.height * scale)))
            if target_w == img.width and target_h == img.height:
                return image_path, None
            resample_attr = getattr(Image, "Resampling", None)
            resample_filter = resample_attr.LANCZOS if resample_attr else Image.LANCZOS
            resized = img.resize((target_w, target_h), resample_filter)
            suffix = os.path.splitext(image_path)[1] or ".png"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            resized.save(tmp.name)
            tmp_path = tmp.name
            tmp.close()
            return tmp_path, tmp_path
    except Exception:
        return image_path, None


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


def _resolve_directory_preference(text_value: str, explorer_value: Union[str, list, None], default_path: str) -> str:
    """Return a normalized directory path based on textbox fallback and file explorer selection."""
    candidate: Optional[str] = None
    if explorer_value:
        if isinstance(explorer_value, list):
            explorer_value = explorer_value[0] if explorer_value else None
        candidate = explorer_value
    elif text_value and text_value.strip():
        candidate = text_value.strip()
    else:
        candidate = default_path
    candidate = os.path.abspath(os.path.expanduser(candidate))
    return candidate


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


def run_gen3c_backend(
    image_path: Union[str, None],
    guidance: float,
    frames: Union[str, int, None],
    video_name: str,
    checkpoint_dir: str,
    output_dir: str,
    extra_args: str,
) -> Tuple[Optional[str], str, str]:
    """Launch GEN3C via the wrapper script and return the generated video."""
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


def handle_generation(
    backend_choice: str,
    image_path: Union[str, None],
    image_scale: Union[float, int],
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
    save_location_selection: Union[str, list, None],
    gen3c_guidance: float,
    gen3c_frames: Union[str, int, None],
    gen3c_video_name: str,
    gen3c_checkpoint_dir: str,
    gen3c_output_dir: str,
    gen3c_dir_selection: Union[str, list, None],
    gen3c_extra_args: str,
) -> Tuple[Optional[str], str, str]:
    resolved_hunyuan_dir = _resolve_directory_preference(
        save_location, save_location_selection, HUNYUAN_DEFAULT_OUTPUT_DIR
    )
    resolved_gen3c_dir = _resolve_directory_preference(
        gen3c_output_dir, gen3c_dir_selection, GEN3C_DEFAULT_OUTPUT_DIR
    )
    scale_value = _clamp_scale_value(image_scale)
    scaled_path, temp_scaled = _maybe_downscale_image(image_path, scale_value)
    effective_image_path = scaled_path or image_path

    try:
        if backend_choice == "GEN3C Video":
            return run_gen3c_backend(
                image_path=effective_image_path,
                guidance=gen3c_guidance,
                frames=gen3c_frames,
                video_name=gen3c_video_name,
                checkpoint_dir=gen3c_checkpoint_dir,
                output_dir=resolved_gen3c_dir,
                extra_args=gen3c_extra_args,
            )
        return run_hunyuan(
            image_path=effective_image_path,
            guidance_scale=guidance_scale,
            steps=steps,
            seed=seed,
            model_choice=model_choice,
            use_fp16=use_fp16,
            attention_slicing=attention_slicing,
            cpu_offload=cpu_offload,
            remove_background=remove_background,
            output_name=output_name,
            save_location=resolved_hunyuan_dir,
        )
    finally:
        if temp_scaled and os.path.exists(temp_scaled):
            try:
                os.remove(temp_scaled)
            except Exception:
                pass


def _backend_ui_state(choice: str):
    """Toggle control groups depending on the selected backend."""
    is_hunyuan = choice != "GEN3C Video"
    run_label = "Generate Hunyuan GLB" if is_hunyuan else "Generate GEN3C Video"
    progress_hint = (
        "Ready to generate a GLB mesh..."
        if is_hunyuan
        else "Ready to generate a GEN3C video..."
    )
    return (
        gr.update(visible=is_hunyuan),
        gr.update(visible=not is_hunyuan),
        gr.update(value=run_label),
        gr.update(value=progress_hint),
    )


with gr.Blocks(title="Hunyuan3D 2D→3D") as demo:
    gr.Markdown("## Hunyuan3D 2D→3D\nUpload an image, set options, and generate a GLB.")

    with gr.Row():
        # Left column - Image and system metrics
        with gr.Column(scale=1):
            image = gr.Image(type="filepath", label="Input image")
            with gr.Group():
                image_resolution_display = gr.Textbox(
                    label="Input Resolution",
                    value="No image loaded.",
                    interactive=False,
                    lines=2,
                )
                image_scale_slider = gr.Slider(
                    minimum=0.25,
                    maximum=1.0,
                    value=1.0,
                    step=0.05,
                    label="Scale Before Generation",
                    info="Downscale large images before running Hunyuan or GEN3C.",
                )
                preview_button = gr.Button("Refresh Resolution Preview", variant="secondary")

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
                value="Ready to generate a GLB mesh...",
                interactive=False,
                lines=1
            )

        # Right column - Controls
        with gr.Column(scale=1):
            backend_choice = gr.Radio(
                choices=["Hunyuan GLB", "GEN3C Video"],
                value="Hunyuan GLB",
                label="Backend Selection",
                info="Choose Hunyuan for GLB meshes or GEN3C for video outputs."
            )
            with gr.Group(visible=True) as hunyuan_controls:
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
                save_location = gr.Textbox(
                    value=HUNYUAN_DEFAULT_OUTPUT_DIR,
                    label="Save Location",
                    info="Defaults to assets/hunyuan_outputs unless changed below."
                )
                save_location_picker = gr.FileExplorer(
                    value=None,
                    label="Browse Save Location",
                    file_count="single",
                    root_dir=PROJECT_ROOT,
                    height=150,
                )

            with gr.Accordion("GEN3C Options", open=False, visible=False) as gen3c_options:
                gen3c_guidance = gr.Slider(0.5, 5.0, value=1.0, step=0.1, label="GEN3C Guidance")
                gen3c_frames = gr.Dropdown(
                    choices=["121", "242", "363", "484"],
                    value="121",
                    label="GEN3C Frames",
                    info="GEN3C currently supports frame counts that are multiples of 121."
                )
                gen3c_video_name = gr.Textbox(value="gen3c_video", label="GEN3C Video Name")
                gen3c_checkpoint_dir = gr.Textbox(
                    value=GEN3C_DEFAULT_CHECKPOINT,
                    label="GEN3C Checkpoint Directory"
                )
                gen3c_output_dir = gr.Textbox(
                    value=GEN3C_DEFAULT_OUTPUT_DIR,
                    label="Save Location",
                    info="Defaults to assets/gen3c_outputs unless changed below."
                )
                gen3c_dir_picker = gr.FileExplorer(
                    value=None,
                    label="Browse Save Location",
                    file_count="single",
                    root_dir=PROJECT_ROOT,
                    height=150,
                )
                gen3c_extra_args = gr.Textbox(
                    value="--trajectory left --foreground_masking",
                    label="Additional GEN3C Arguments"
                )

            run_btn = gr.Button("Generate Hunyuan GLB", variant="primary")

    with gr.Row():
        output_file = gr.File(label="Output (GLB or MP4)")
        logs_box = gr.Textbox(label="Logs", lines=8)

    # Add progress indicator
    progress_bar = gr.Progress()

    run_btn.click(
        fn=handle_generation,
        inputs=[
            backend_choice,
            image,
            image_scale_slider,
            guidance_scale,
            steps,
            seed,
            model_choice,
            use_fp16,
            attention_slicing,
            cpu_offload,
            remove_background,
            output_name,
            save_location,
            save_location_picker,
            gen3c_guidance,
            gen3c_frames,
            gen3c_video_name,
            gen3c_checkpoint_dir,
            gen3c_output_dir,
            gen3c_dir_picker,
            gen3c_extra_args,
        ],
        outputs=[output_file, logs_box, progress_display],
    )

    backend_choice.change(
        fn=_backend_ui_state,
        inputs=[backend_choice],
        outputs=[hunyuan_controls, gen3c_options, run_btn, progress_display],
    )

    def _dummy_handler(*args):
        pass

    preview_button.click(
        fn=_safe_preview_update,
        inputs=[image, image_scale_slider],
        outputs=[image_resolution_display],
        queue=False,
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

    demo.launch(server_port=5683, share=False)
    