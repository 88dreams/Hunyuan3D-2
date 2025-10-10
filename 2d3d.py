#!/usr/bin/env python3

import os
import gc
from typing import Optional, Tuple, Union

import gradio as gr
import torch
from PIL import Image

from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.texgen import Hunyuan3DPaintPipeline


# Repositories and cache configuration
SHAPE_REPO = "tencent/Hunyuan3D-2"
PAINT_REPO = "tencent/Hunyuan3D-2"
CACHE_DIR = "./weights"


_shape_pipeline: Optional[Hunyuan3DDiTFlowMatchingPipeline] = None
_paint_pipeline: Optional[Hunyuan3DPaintPipeline] = None
_current_dtype: Optional[torch.dtype] = None


def _apply_memory_savers(pipeline: object, attention_slicing: bool, cpu_offload: bool, dtype: torch.dtype) -> None:
    """Apply memory optimizations when available on the pipeline instance."""
    if hasattr(pipeline, "to"):
        pipeline.to(dtype=dtype)
    if attention_slicing and hasattr(pipeline, "enable_attention_slicing"):
        pipeline.enable_attention_slicing()
    if cpu_offload and hasattr(pipeline, "enable_sequential_cpu_offload"):
        pipeline.enable_sequential_cpu_offload()


def _ensure_pipelines(
    use_texture: bool,
    use_fp16: bool,
    attention_slicing: bool,
    cpu_offload: bool,
) -> torch.dtype:
    """Load and cache the pipelines if needed, configure dtype and memory options."""
    global _shape_pipeline, _paint_pipeline, _current_dtype

    requested_dtype = torch.float16 if use_fp16 else torch.float32

    # Load or reload shape pipeline when dtype changes or not initialized
    if _shape_pipeline is None or _current_dtype != requested_dtype:
        print(f"Loading shape pipeline (dtype={'fp16' if use_fp16 else 'fp32'})...")
        _shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            SHAPE_REPO,
            cache_dir=CACHE_DIR,
            torch_dtype=requested_dtype,
        )
        _current_dtype = requested_dtype

    _apply_memory_savers(_shape_pipeline, attention_slicing, cpu_offload, requested_dtype)

    # Texture pipeline only if requested
    if use_texture:
        if _paint_pipeline is None or _current_dtype != requested_dtype:
            print(f"Loading texture pipeline (dtype={'fp16' if use_fp16 else 'fp32'})...")
            # Load with only supported parameters
            _paint_pipeline = Hunyuan3DPaintPipeline.from_pretrained(
                'tencent/Hunyuan3D-2',
                subfolder='hunyuan3d-paint-v2-0-turbo'
            )

            # Note: Hunyuan3DPaintPipeline doesn't support .to() method like shape pipeline
        _apply_memory_savers(_paint_pipeline, attention_slicing, cpu_offload, requested_dtype)

    return requested_dtype


def _load_image(image_path: str) -> Image.Image:
    img = Image.open(image_path)
    return img.convert("RGB")


def run(
    image_path: Union[str, None],
    mode: str,
    guidance_scale: float,
    steps: int,
    seed: Optional[int],
    use_fp16: bool,
    attention_slicing: bool,
    cpu_offload: bool,
    output_name: str,
) -> Tuple[str, str]:
    """Main generation entrypoint. Returns (output_file_path, logs)."""
    import time
    # Explicitly declare global variables
    global _shape_pipeline, _paint_pipeline
    if not image_path:
        raise gr.Error("Please provide an image.")
    if not os.path.exists(image_path):
        raise gr.Error("Provided image path does not exist.")

    if not output_name:
        output_name = "output_model"

    # Normalize names
    base_out = os.path.splitext(output_name)[0]
    shape_out_path = f"{base_out}_shape.glb"
    final_out_path = f"{base_out}.glb"

    # Optional determinism
    if seed is not None and str(seed).strip() != "":
        try:
            torch.manual_seed(int(seed))
        except Exception:
            pass

    use_texture = mode == "Shape + Texture"

    _ensure_pipelines(use_texture, use_fp16, attention_slicing, cpu_offload)

    img = _load_image(image_path)

    logs = []
    result = None
    mesh = None
    painted = None
    textured_mesh = None

    # Set timeout (8 minutes for texture generation, 4 for shape only)
    timeout_duration = 480 if use_texture else 240
    start_time = time.time()

    try:
        # Check timeout periodically during generation
        def check_timeout():
            if time.time() - start_time > timeout_duration:
                raise gr.Error(f"Generation timed out after {timeout_duration} seconds. Try reducing steps or enabling memory optimizations.")

        logs.append("Generating shape...")
        check_timeout()

        # Ensure shape pipeline is available
        if _shape_pipeline is None or _current_dtype != (torch.float16 if use_fp16 else torch.float32):
            _ensure_pipelines(use_texture=False, use_fp16=use_fp16, attention_slicing=attention_slicing, cpu_offload=cpu_offload)

        result = _shape_pipeline(
            image=img,
            guidance_scale=float(guidance_scale),
            num_inference_steps=int(steps),
        )
        mesh = result[0] if isinstance(result, (list, tuple)) else result
        mesh.export(shape_out_path)
        logs.append(f"Saved shape: {shape_out_path}")
        check_timeout()

        if use_texture:
            logs.append("Applying texture...")
            check_timeout()

            # Ensure texture pipeline is available
            if _paint_pipeline is None or _current_dtype != (torch.float16 if use_fp16 else torch.float32):
                _ensure_pipelines(use_texture=True, use_fp16=use_fp16, attention_slicing=attention_slicing, cpu_offload=cpu_offload)

            # Note: We keep the shape pipeline loaded for texture generation
            # The memory clearing happens after successful completion

            painted = _paint_pipeline(
                mesh=mesh,
                image=img,
            )
            textured_mesh = painted[0] if isinstance(painted, (list, tuple)) else painted
            textured_mesh.export(final_out_path)
            logs.append(f"Saved textured model: {final_out_path}")
            check_timeout()

            # Clear shape pipeline from memory after successful texture generation
            if _shape_pipeline is not None:
                try:
                    del _shape_pipeline
                    _shape_pipeline = None
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

            return final_out_path, "\n".join(logs)

        return shape_out_path, "\n".join(logs)

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            raise gr.Error("Out of GPU memory. Enable CPU offload / FP16 / attention slicing, or reduce steps.")
        raise
    finally:
        # Proactively release GPU memory between runs
        try:
            del textured_mesh
        except Exception:
            pass
        try:
            del painted
        except Exception:
            pass
        try:
            del mesh
        except Exception:
            pass
        try:
            del result
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
            mode = gr.Radio(choices=["Shape only", "Shape + Texture"], value="Shape only", label="Pipeline")
            guidance_scale = gr.Slider(1.0, 15.0, value=7.5, step=0.5, label="Guidance scale")
            steps = gr.Slider(10, 100, value=50, step=1, label="Inference steps")
            seed = gr.Number(value=42, precision=0, label="Seed (optional)")
            use_fp16 = gr.Checkbox(value=True, label="Use FP16 (half precision)")
            attention_slicing = gr.Checkbox(value=True, label="Enable attention slicing")
            cpu_offload = gr.Checkbox(value=False, label="Enable CPU offload")
            output_name = gr.Textbox(value="output_model", label="Output name (no extension)")
            run_btn = gr.Button("Generate", variant="primary")

    with gr.Row():
        output_file = gr.File(label="Output GLB")
        logs_box = gr.Textbox(label="Logs", lines=8)

    # Add progress indicator
    progress_bar = gr.Progress()

    run_btn.click(
        fn=run,
        inputs=[image, mode, guidance_scale, steps, seed, use_fp16, attention_slicing, cpu_offload, output_name],
        outputs=[output_file, logs_box],
    )


if __name__ == "__main__":
    # Default to port 8080 to match your note; change if needed
    # Enable queuing to avoid overlapping runs consuming VRAM
    # (no args for compatibility with your Gradio version)
    demo.queue()
    demo.launch(server_port=8080, share=False)


