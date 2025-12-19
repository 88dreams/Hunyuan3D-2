"""
SHARP Tab for 3D Generation Studio

Apple's SHARP: Single image → 3D Gaussian Splatting in <1 second
Supports local execution and RunPod serverless.
"""

import gradio as gr  # type: ignore

from generators.sharp import SHARP_DEFAULT_OUTPUT_DIR, check_sharp_installation


def create_sharp_tab(
    default_endpoint_id: str = "",
    default_api_key: str = "",
):
    """
    Create the SHARP tab UI components.
    
    Args:
        default_endpoint_id: Default RunPod endpoint ID (from saved config)
        default_api_key: Default RunPod API key (from saved config)
    
    Returns:
        Dictionary of all UI components for event handler binding
    """
    gr.Markdown("""
    **Image → 3D Gaussian Splatting (Fast)**
    
    Apple's SHARP generates photorealistic 3D Gaussian Splats from a single image in less than a second.
    """)
    
    # Execution Settings
    with gr.Group():
        gr.Markdown("#### Execution")
        exec_mode = gr.Radio(
            choices=["Local (PLY only)", "RunPod Serverless (PLY + Video)"],
            value="Local (PLY only)",
            label="Mode",
            info="Local: CPU/GPU PLY generation. RunPod: Full CUDA with video rendering.",
        )
        status = gr.Textbox(
            value=check_sharp_installation(),
            label="Status",
            interactive=False,
        )
        check_btn = gr.Button("Check Status", size="sm")
    
    # RunPod Settings (hidden by default)
    with gr.Group(visible=False) as runpod_settings:
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
    
    # Generation Settings
    with gr.Group():
        gr.Markdown("#### Generation Settings")
        render_video = gr.Checkbox(
            value=False,
            label="Render Video Trajectory",
            info="Generate a video of camera movement around the object (RunPod only, CUDA required)",
        )
    
    # Output Settings
    with gr.Group():
        gr.Markdown("#### Output")
        output_name = gr.Textbox(
            value="sharp_output",
            label="Output Name",
        )
        output_dir = gr.Textbox(
            value=SHARP_DEFAULT_OUTPUT_DIR,
            label="Output Directory",
        )
    
    # Generate Button
    with gr.Group():
        generate_btn = gr.Button(
            "Generate PLY",
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
        "render_video": render_video,
        "output_name": output_name,
        "output_dir": output_dir,
        "generate_btn": generate_btn,
        "queue_btn": queue_btn,
        "progress_display": progress_display,
        "logs_box": logs_box,
    }
