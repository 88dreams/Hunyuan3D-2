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
            choices=["Local", "RunPod Serverless"],
            value="RunPod Serverless",
            label="Mode",
            info="Local: CPU/GPU PLY generation. RunPod: Full CUDA.",
        )
        status = gr.Textbox(
            value="Select execution mode",
            label="Status",
            interactive=False,
        )
        
        # Local Settings
        with gr.Group(visible=False) as local_settings:
            local_status = gr.Textbox(
                value=check_sharp_installation(),
                label="Local Installation",
                interactive=False,
            )
            check_local_btn = gr.Button("Check Local", size="sm")
        
        # RunPod Serverless Settings (visible by default)
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
    
    # Generation Settings (hidden for now - video rendering not working)
    # with gr.Group():
    #     gr.Markdown("#### Generation Settings")
    render_video = gr.Checkbox(
        value=False,
        label="Render Video Trajectory",
        visible=False,  # Hidden until video rendering is fixed
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
        "local_settings": local_settings,
        "local_status": local_status,
        "check_local_btn": check_local_btn,
        "serverless_settings": serverless_settings,
        "check_serverless_btn": check_serverless_btn,
        "cancel_btn": cancel_btn,
        "edit_creds_btn": edit_creds_btn,
        "creds_group": creds_group,
        "endpoint_id": endpoint_id,
        "api_key": api_key,
        "save_creds_btn": save_creds_btn,
        "current_job_id": current_job_id,
        "render_video": render_video,
        "output_name": output_name,
        "output_dir": output_dir,
        "generate_btn": generate_btn,
        "queue_btn": queue_btn,
        "progress_display": progress_display,
        "logs_box": logs_box,
    }
