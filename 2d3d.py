#!/usr/bin/env python3

import os
import gc
from typing import Optional, Tuple, Union

import gradio as gr
import torch
from PIL import Image

from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline


# Repositories and cache configuration
SHAPE_REPO = "tencent/Hunyuan3D-2mini"
CACHE_DIR = "/home/arkrunr/.cache/huggingface/hub"


_shape_pipeline: Optional[Hunyuan3DDiTFlowMatchingPipeline] = None
_current_dtype: Optional[torch.dtype] = None
_current_model: Optional[str] = None


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
        subfolder = None
        if "mini" in selected_model:
            subfolder = "hunyuan3d-dit-v2-mini-turbo"

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
    output_name: str,
    save_location: str,
) -> Tuple[str, str]:
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

    _ensure_pipelines(model_choice, use_fp16, attention_slicing, cpu_offload)

    img = _load_image(image_path)

    logs = []
    mesh = None

    # Set timeout (10 minutes for shape generation to handle complex models)
    timeout_duration = 600
    start_time = time.time()

    try:
        # Check timeout periodically during generation
        def check_timeout():
            if time.time() - start_time > timeout_duration:
                raise gr.Error(f"Generation timed out after {timeout_duration} seconds. Try reducing steps or enabling memory optimizations.")

        logs.append("Generating 3D shape...")
        check_timeout()

        result = _shape_pipeline(
            image=img,
            guidance_scale=float(guidance_scale),
            num_inference_steps=int(steps),
        )
        mesh = result[0] if isinstance(result, (list, tuple)) else result
        mesh.export(output_path)
        logs.append(f"Saved shape: {output_path}")
        check_timeout()

        return output_path, "\n".join(logs)

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            raise gr.Error("Out of GPU memory. Enable CPU offload / FP16 / attention slicing, or reduce steps.")
        raise
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
        image = gr.Image(type="filepath", label="Input image")
        with gr.Column():
            guidance_scale = gr.Slider(1.0, 15.0, value=7.5, step=0.5, label="Guidance scale")
            steps = gr.Slider(10, 100, value=30, step=1, label="Inference steps")
            seed = gr.Number(value=42, precision=0, label="Seed (optional)")
            model_choice = gr.Radio(
                choices=["Mini Model (Faster)", "Full Model (Higher Quality)"],
                value="Mini Model (Faster)",
                label="Model Selection"
            )
            use_fp16 = gr.Checkbox(value=True, label="Use FP16 (half precision)")
            attention_slicing = gr.Checkbox(value=True, label="Enable attention slicing")
            cpu_offload = gr.Checkbox(value=True, label="Enable CPU offload")
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
        inputs=[image, guidance_scale, steps, seed, model_choice, use_fp16, attention_slicing, cpu_offload, output_name, save_location],
        outputs=[output_file, logs_box],
    )


if __name__ == "__main__":
    # Enable queuing to avoid overlapping runs consuming VRAM
    # (no args for compatibility with your Gradio version)
    demo.queue()
    demo.launch(server_port=7860, share=False)
    