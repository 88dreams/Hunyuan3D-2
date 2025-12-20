"""
GEN3C Video Tab for 3D Generation Studio

This module defines the GEN3C tab UI components and layout.
"""

import gradio as gr  # type: ignore

from generators.gen3c import GEN3C_DEFAULT_CHECKPOINT, GEN3C_DEFAULT_OUTPUT_DIR


def create_gen3c_tab(
    default_runpod_url: str,
    default_endpoint_id: str,
    default_api_key: str,
    format_cluster_status_fn,
):
    """
    Create the GEN3C Video tab UI components.
    
    Args:
        default_runpod_url: Default RunPod pod URL
        default_endpoint_id: Default serverless endpoint ID
        default_api_key: Default serverless API key
        format_cluster_status_fn: Function to format cluster status
    
    Returns:
        Dictionary of all UI components for event handler binding
    """
    gr.Markdown("**Image → Video** | Camera trajectory video generation")
    
    # Execution Settings
    with gr.Group():
        gr.Markdown("#### Execution")
        exec_mode = gr.Radio(
            choices=["RunPod Serverless", "RunPod Pod", "Local Cluster"],
            value="RunPod Serverless",
            label="Mode",
        )
        status = gr.Textbox(
            value="Select execution mode",
            label="Status",
            interactive=False,
        )
        
        # RunPod Pod Settings
        with gr.Group(visible=False) as pod_settings:
            runpod_url = gr.Textbox(
                value=default_runpod_url,
                label="Pod API URL",
            )
            with gr.Row():
                check_pod_btn = gr.Button("Check Status", size="sm")
                save_pod_btn = gr.Button("Save", size="sm")
        
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
        
        # Local Cluster Settings
        with gr.Group(visible=False) as local_settings:
            check_cluster_btn = gr.Button("Check Cluster", size="sm")
            checkpoint_dir = gr.Textbox(
                value=GEN3C_DEFAULT_CHECKPOINT,
                label="Checkpoint Directory",
            )
            extra_args = gr.Textbox(
                value="",
                label="Extra Arguments",
            )
    
    # Video Settings
    with gr.Group():
        gr.Markdown("#### Video Settings")
        with gr.Row():
            frames = gr.Dropdown(
                choices=["121", "241", "361", "481"],
                value="121",
                label="Frames",
                info="121=5s, 241=10s, 361=15s, 481=20s",
            )
            trajectory = gr.Dropdown(
                choices=["left", "right", "up", "down", "zoom_in", "zoom_out", "clockwise", "counterclockwise", "none"],
                value="left",
                label="Trajectory",
            )
        with gr.Row():
            guidance = gr.Slider(0.5, 3.0, value=1.0, step=0.1, label="Guidance")
            foreground_mask = gr.Checkbox(value=True, label="Foreground Mask")
        with gr.Row():
            video_name = gr.Textbox(value="gen3c_video", label="Output Name")
            seed = gr.Number(value=None, precision=0, label="Seed")
    
    # Output Settings
    with gr.Group():
        gr.Markdown("#### Output")
        output_dir = gr.Textbox(
            value=GEN3C_DEFAULT_OUTPUT_DIR,
            label="Output Directory",
        )
    
    # Generate Button (Fixed Position)
    with gr.Group():
        generate_btn = gr.Button(
            "Generate Video",
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
        "pod_settings": pod_settings,
        "runpod_url": runpod_url,
        "check_pod_btn": check_pod_btn,
        "save_pod_btn": save_pod_btn,
        "serverless_settings": serverless_settings,
        "check_serverless_btn": check_serverless_btn,
        "cancel_btn": cancel_btn,
        "edit_creds_btn": edit_creds_btn,
        "creds_group": creds_group,
        "endpoint_id": endpoint_id,
        "api_key": api_key,
        "save_creds_btn": save_creds_btn,
        "current_job_id": current_job_id,
        "local_settings": local_settings,
        "check_cluster_btn": check_cluster_btn,
        "checkpoint_dir": checkpoint_dir,
        "extra_args": extra_args,
        "frames": frames,
        "trajectory": trajectory,
        "guidance": guidance,
        "foreground_mask": foreground_mask,
        "video_name": video_name,
        "seed": seed,
        "output_dir": output_dir,
        "generate_btn": generate_btn,
        "queue_btn": queue_btn,
        "progress_display": progress_display,
        "logs_box": logs_box,
    }

