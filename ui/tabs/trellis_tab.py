"""
TRELLIS.2 Tab for 3D Generation Studio

Microsoft's TRELLIS.2: Image → High-Quality 3D with O-Voxel representation
4B parameter model for high-fidelity 3D generation with PBR materials.

Features:
- O-Voxel representation (Native & Compact Structured Latents)
- GLB output with PBR materials (Base Color, Roughness, Metallic, Opacity)
- Multiple resolution options (512³, 1024³, 1536³)
"""

import gradio as gr  # type: ignore


# Default output directory (will be set from config)
TRELLIS_DEFAULT_OUTPUT_DIR = "/srv/searidge_share/outputs/trellis"


def create_trellis_tab(
    default_endpoint_id: str = "",
    default_api_key: str = "",
):
    """
    Create the TRELLIS.2 tab UI components.
    
    Args:
        default_endpoint_id: Default RunPod endpoint ID (from saved config)
        default_api_key: Default RunPod API key (from saved config)
    
    Returns:
        Dictionary of all UI components for event handler binding
    """
    gr.Markdown("""
    **Image → High-Quality 3D with PBR Materials**
    
    Microsoft's TRELLIS.2 generates production-ready 3D models with O-Voxel representation.
    Outputs GLB files with full PBR materials (Base Color, Roughness, Metallic, Opacity).
    
    *4B parameter model - Requires H100 GPU for best performance*
    """)
    
    # Execution Settings
    with gr.Group():
        gr.Markdown("#### Execution")
        exec_mode = gr.Radio(
            choices=["RunPod Serverless"],
            value="RunPod Serverless",
            label="Mode",
            info="TRELLIS.2 requires H100 GPUs. RunPod Serverless recommended.",
        )
        status = gr.Textbox(
            value="Configure RunPod credentials below",
            label="Status",
            interactive=False,
        )
        check_btn = gr.Button("Check Connection", size="sm")
    
    # RunPod Settings
    with gr.Group() as runpod_settings:
        gr.Markdown("#### RunPod Settings")
        endpoint_id = gr.Textbox(
            value=default_endpoint_id,
            label="Endpoint ID",
            placeholder="Your serverless endpoint ID",
        )
        api_key = gr.Textbox(
            value=default_api_key,
            label="API Key",
            type="password",
            placeholder="rp_...",
        )
        with gr.Row():
            check_runpod_btn = gr.Button("Check Connection", size="sm")
            save_creds_btn = gr.Button("Save Credentials", size="sm")
    
    # Model Settings
    with gr.Group():
        gr.Markdown("#### Model Settings")
        resolution = gr.Radio(
            choices=["512³ (~3s)", "1024³ (~17s)", "1536³ (~60s)"],
            value="1024³ (~17s)",
            label="Resolution",
            info="Higher resolution = better quality but slower",
        )
        with gr.Row():
            seed = gr.Number(
                value=None,
                label="Seed",
                precision=0,
                info="Random seed (leave empty for random)",
            )
            guidance_scale = gr.Slider(
                minimum=1.0,
                maximum=10.0,
                value=7.5,
                step=0.5,
                label="Guidance Scale",
                info="Higher = more faithful to input",
            )
    
    # Material Settings
    with gr.Group():
        gr.Markdown("#### Material Settings")
        gr.Markdown("*TRELLIS.2 automatically generates PBR materials*")
        with gr.Row():
            export_base_color = gr.Checkbox(
                value=True,
                label="Base Color",
                interactive=False,
            )
            export_roughness = gr.Checkbox(
                value=True,
                label="Roughness",
                interactive=False,
            )
            export_metallic = gr.Checkbox(
                value=True,
                label="Metallic",
                interactive=False,
            )
            export_opacity = gr.Checkbox(
                value=True,
                label="Opacity",
                interactive=False,
            )
    
    # Output Settings
    with gr.Group():
        gr.Markdown("#### Output")
        output_name = gr.Textbox(
            value="trellis_output",
            label="Output Name",
        )
        output_dir = gr.Textbox(
            value=TRELLIS_DEFAULT_OUTPUT_DIR,
            label="Output Directory",
        )
        output_format = gr.Radio(
            choices=["GLB (with PBR)", "PLY (geometry only)"],
            value="GLB (with PBR)",
            label="Output Format",
            info="GLB includes textures and materials",
        )
    
    # Generate Button
    with gr.Group():
        generate_btn = gr.Button(
            "Generate 3D Model",
            variant="primary",
            size="lg",
        )
        queue_btn = gr.Button(
            "+ Add to Queue",
            variant="secondary",
        )
    
    # Progress and Logs
    with gr.Group():
        gr.Markdown("#### Progress")
        progress_display = gr.Textbox(
            value="Ready to generate...",
            interactive=False,
            lines=1,
            show_label=False,
        )
    
    with gr.Accordion("Show Logs", open=False):
        logs_box = gr.Textbox(
            label="Generation Logs",
            lines=8,
            interactive=False,
        )
    
    return {
        "exec_mode": exec_mode,
        "status": status,
        "check_btn": check_btn,
        "runpod_settings": runpod_settings,
        "endpoint_id": endpoint_id,
        "api_key": api_key,
        "check_runpod_btn": check_runpod_btn,
        "save_creds_btn": save_creds_btn,
        "resolution": resolution,
        "seed": seed,
        "guidance_scale": guidance_scale,
        "output_name": output_name,
        "output_dir": output_dir,
        "output_format": output_format,
        "generate_btn": generate_btn,
        "queue_btn": queue_btn,
        "progress_display": progress_display,
        "logs_box": logs_box,
    }

