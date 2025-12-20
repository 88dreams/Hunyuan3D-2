"""
Lyra Tab for 3D Generation Studio

NVIDIA's Lyra: Image/Video → 3D/4D Gaussian Splatting
Built on GEN3C video diffusion + 3DGS reconstruction decoder.

Supports:
- Static 3DGS from single image (Image → multi-view video → 3DGS)
- Dynamic 4DGS from video input (Video → multi-view video → 4DGS)
"""

import gradio as gr  # type: ignore


# Default output directory (will be set from config)
LYRA_DEFAULT_OUTPUT_DIR = "/srv/searidge_share/outputs/lyra"


def create_lyra_tab(
    default_endpoint_id: str = "",
    default_api_key: str = "",
):
    """
    Create the Lyra 3DGS/4DGS tab UI components.
    
    Args:
        default_endpoint_id: Default RunPod endpoint ID (from saved config)
        default_api_key: Default RunPod API key (from saved config)
    
    Returns:
        Dictionary of all UI components for event handler binding
    """
    gr.Markdown("""
    **Image/Video → 3D/4D Gaussian Splatting**
    
    ⚠️ **Lyra is not yet fully implemented for serverless deployment.**
    
    Lyra requires a complex multi-step pipeline:
    1. SDG (multi-view video generation via GEN3C diffusion)
    2. 3DGS decoder (accelerate launch with config files)
    
    **For fast 3DGS generation, please use SHARP instead.**
    
    *Full Lyra support coming in a future update.*
    """)
    
    # Execution Settings
    with gr.Group():
        gr.Markdown("#### Execution")
        exec_mode = gr.Radio(
            choices=["RunPod Serverless"],
            value="RunPod Serverless",
            label="Mode",
            info="Lyra requires H100/A100 GPUs. RunPod Serverless recommended.",
        )
        status = gr.Textbox(
            value="Select execution mode",
            label="Status",
            interactive=False,
        )
        
        # RunPod Serverless Settings
        with gr.Group(visible=True) as serverless_settings:
            _has_creds = bool(default_endpoint_id and default_api_key)
            with gr.Row():
                check_serverless_btn = gr.Button("Check Status", size="sm")
                cancel_btn = gr.Button("Cancel Job", variant="stop", size="sm")
                edit_creds_btn = gr.Button("Edit Credentials", size="sm")
            
            with gr.Group(visible=not _has_creds) as creds_group:
                endpoint_id = gr.Textbox(
                    value=default_endpoint_id,
                    label="Endpoint ID",
                )
                api_key = gr.Textbox(
                    value=default_api_key,
                    label="API Key",
                    type="password",
                )
                save_creds_btn = gr.Button("Save Credentials", variant="primary")
            
            current_job_id = gr.State(value=None)
    
    # Generation Mode
    with gr.Group():
        gr.Markdown("#### Generation Mode")
        generation_mode = gr.Radio(
            choices=["Static (Image → 3DGS)", "Dynamic (Video → 4DGS)"],
            value="Static (Image → 3DGS)",
            label="Mode",
            info="Static: Single image to 3D. Dynamic: Video to animated 4D.",
        )
    
    # Diffusion Settings (for video generation phase)
    with gr.Group():
        gr.Markdown("#### Diffusion Settings")
        with gr.Row():
            num_views = gr.Slider(
                minimum=4,
                maximum=16,
                value=8,
                step=1,
                label="Multi-view Count",
                info="Number of camera views to generate",
            )
            camera_motion = gr.Slider(
                minimum=0.5,
                maximum=2.0,
                value=1.0,
                step=0.1,
                label="Camera Motion Scale",
                info="1.0=normal, 2.0=more motion (may cause artifacts)",
            )
        multi_trajectory = gr.Checkbox(
            value=True,
            label="Multi-Trajectory",
            info="Generate multiple camera trajectories for better coverage",
        )
        foreground_masking = gr.Checkbox(
            value=True,
            label="Foreground Masking",
            info="Mask background for object-centric scenes",
        )
    
    # Reconstruction Settings
    with gr.Group():
        gr.Markdown("#### Reconstruction Settings")
        with gr.Row():
            num_gaussians = gr.Slider(
                minimum=10000,
                maximum=500000,
                value=100000,
                step=10000,
                label="Max Gaussians",
                info="Maximum number of Gaussian splats",
            )
            seed = gr.Number(
                value=None,
                label="Seed",
                precision=0,
                info="Random seed (leave empty for random)",
            )
    
    # Output Settings
    with gr.Group():
        gr.Markdown("#### Output")
        output_name = gr.Textbox(
            value="lyra_output",
            label="Output Name",
        )
        output_dir = gr.Textbox(
            value=LYRA_DEFAULT_OUTPUT_DIR,
            label="Output Directory",
        )
        with gr.Row():
            output_ply = gr.Checkbox(
                value=True,
                label="Export PLY",
                info="3D Gaussian Splat format",
            )
            output_video = gr.Checkbox(
                value=True,
                label="Export Render Video",
                info="Rendered video of the 3DGS",
            )
    
    # Generate Button
    with gr.Group():
        generate_btn = gr.Button(
            "Generate 3DGS",
            variant="primary",
            size="lg",
        )
        queue_btn = gr.Button(
            "+ Add to Queue",
            variant="secondary",
        )
    
    # Progress and Logs (with scrolling)
    with gr.Group():
        gr.Markdown("#### Progress")
        progress_display = gr.Textbox(
            value="Ready to generate...",
            interactive=False,
            lines=2,
            max_lines=4,
            show_label=False,
            autoscroll=True,
        )
    
    with gr.Accordion("Show Logs", open=False):
        logs_box = gr.Textbox(
            label="Generation Logs",
            lines=10,
            max_lines=20,
            interactive=False,
            autoscroll=True,
        )
    
    return {
        "exec_mode": exec_mode,
        "status": status,
        "serverless_settings": serverless_settings,
        "check_serverless_btn": check_serverless_btn,
        "cancel_btn": cancel_btn,
        "edit_creds_btn": edit_creds_btn,
        "creds_group": creds_group,
        "endpoint_id": endpoint_id,
        "api_key": api_key,
        "save_creds_btn": save_creds_btn,
        "current_job_id": current_job_id,
        "generation_mode": generation_mode,
        "num_views": num_views,
        "camera_motion": camera_motion,
        "multi_trajectory": multi_trajectory,
        "foreground_masking": foreground_masking,
        "num_gaussians": num_gaussians,
        "seed": seed,
        "output_name": output_name,
        "output_dir": output_dir,
        "output_ply": output_ply,
        "output_video": output_video,
        "generate_btn": generate_btn,
        "queue_btn": queue_btn,
        "progress_display": progress_display,
        "logs_box": logs_box,
    }

