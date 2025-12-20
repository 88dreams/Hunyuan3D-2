"""
Hunyuan3D Tab for 3D Generation Studio

This module defines the Hunyuan3D tab UI components and layout.
Supports both Local CUDA and RunPod Serverless execution modes.
"""

import torch  # type: ignore
import gradio as gr  # type: ignore

from generators.hunyuan import HUNYUAN_DEFAULT_OUTPUT_DIR

# Hunyuan3D dedicated endpoint
HUNYUAN_DEFAULT_ENDPOINT_ID = "4wiztgeyjcy1y9"


def create_hunyuan_tab(
    default_endpoint_id: str = "",
    default_api_key: str = "",
):
    """
    Create the Hunyuan3D tab UI components.
    
    Args:
        default_endpoint_id: Default RunPod endpoint ID (for Hunyuan-specific endpoint)
        default_api_key: Default RunPod API key (shared across endpoints)
    
    Returns:
        Dictionary of all UI components for event handler binding
    """
    # Use Hunyuan-specific endpoint if available, otherwise use provided default
    hunyuan_endpoint = HUNYUAN_DEFAULT_ENDPOINT_ID or default_endpoint_id
    
    gr.Markdown("**Image → GLB Mesh** | High-quality 3D mesh generation")
    
    # Execution Settings
    with gr.Group():
        gr.Markdown("#### Execution")
        exec_mode = gr.Radio(
            choices=["Local CUDA", "RunPod Serverless"],
            value="Local CUDA" if torch.cuda.is_available() else "RunPod Serverless",
            label="Mode",
            info="Local: Uses your GPU. RunPod: Cloud GPU (faster for Full model)",
        )
        status = gr.Textbox(
            value="Local GPU Ready" if torch.cuda.is_available() else "Select RunPod mode",
            label="Status",
            interactive=False,
        )
        
        # Local Settings (visible by default if CUDA available)
        with gr.Group(visible=torch.cuda.is_available()) as local_settings:
            local_status = gr.Textbox(
                value="✅ CUDA Available" if torch.cuda.is_available() else "⚠️ No CUDA - CPU mode",
                label="Local GPU",
                interactive=False,
            )
        
        # RunPod Serverless Settings
        with gr.Group(visible=not torch.cuda.is_available()) as serverless_settings:
            _has_api_key = bool(default_api_key)
            
            with gr.Row():
                check_serverless_btn = gr.Button("Check Status", size="sm")
                cancel_btn = gr.Button("Cancel Job", variant="stop", size="sm")
                edit_creds_btn = gr.Button("Edit Credentials", size="sm")
            
            with gr.Group(visible=not _has_api_key) as creds_group:
                endpoint_id = gr.Textbox(
                    value=hunyuan_endpoint,
                    label="Endpoint ID",
                    info="Create a Hunyuan3D endpoint on RunPod",
                )
                api_key = gr.Textbox(
                    value=default_api_key,
                    label="API Key",
                    type="password",
                )
                save_creds_btn = gr.Button("Save Credentials", variant="primary")
            
            current_job_id = gr.State(value=None)
    
    # Model Settings
    with gr.Group():
        gr.Markdown("#### Model Settings")
        model_choice = gr.Radio(
            choices=["Mini Model (Faster)", "Full Model (Higher Quality)"],
            value="Mini Model (Faster)",
            label="Model",
            info="Mini: ~9GB, 2-5 min. Full: ~23GB, 5-15 min",
        )
        with gr.Row():
            guidance = gr.Slider(1.0, 15.0, value=9.0, step=0.5, label="Guidance")
            steps = gr.Slider(10, 100, value=40, step=1, label="Steps")
        with gr.Row():
            seed = gr.Number(value=42, precision=0, label="Seed (optional)")
            octree_resolution = gr.Slider(
                128, 512, value=380, step=10,
                label="Octree Resolution",
                info="Higher = more detail, slower",
            )
    
    # Memory Settings (Local only)
    with gr.Group() as memory_settings:
        gr.Markdown("#### Memory Optimization (Local only)")
        with gr.Row():
            fp16 = gr.Checkbox(value=True, label="FP16")
            attention_slicing = gr.Checkbox(value=True, label="Attention Slicing")
            cpu_offload = gr.Checkbox(value=True, label="CPU Offload")
        remove_bg = gr.Checkbox(value=True, label="Remove Background")
    
    # Output Settings
    with gr.Group():
        gr.Markdown("#### Output")
        output_name = gr.Textbox(value="hunyuan_output", label="Output Name")
        save_location = gr.Textbox(
            value=HUNYUAN_DEFAULT_OUTPUT_DIR,
            label="Save Location",
        )
    
    # Generate Button
    with gr.Group():
        generate_btn = gr.Button(
            "Generate GLB",
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
            max_lines=4,
            show_label=False,
            autoscroll=True,
        )
    
    with gr.Accordion("Show Logs", open=False):
        logs_box = gr.Textbox(
            label="Generation Logs",
            lines=8,
            max_lines=20,
            interactive=False,
            autoscroll=True,
        )
    
    # Mode change handlers
    def on_mode_change(mode):
        is_local = mode == "Local CUDA"
        if is_local:
            status_msg = "✅ Local GPU Ready" if torch.cuda.is_available() else "⚠️ No CUDA available"
        else:
            status_msg = "RunPod Serverless mode selected"
        return (
            gr.update(visible=is_local),  # local_settings
            gr.update(visible=not is_local),  # serverless_settings
            status_msg,  # status
        )
    
    exec_mode.change(
        fn=on_mode_change,
        inputs=[exec_mode],
        outputs=[local_settings, serverless_settings, status],
    )
    
    # Edit credentials toggle
    def toggle_creds(visible):
        return gr.update(visible=not visible)
    
    # Note: The actual toggle needs the current visibility state
    # This will be wired up in 2d3d.py
    
    return {
        "exec_mode": exec_mode,
        "status": status,
        "local_settings": local_settings,
        "local_status": local_status,
        "serverless_settings": serverless_settings,
        "check_serverless_btn": check_serverless_btn,
        "cancel_btn": cancel_btn,
        "edit_creds_btn": edit_creds_btn,
        "creds_group": creds_group,
        "endpoint_id": endpoint_id,
        "api_key": api_key,
        "save_creds_btn": save_creds_btn,
        "current_job_id": current_job_id,
        "model_choice": model_choice,
        "guidance": guidance,
        "steps": steps,
        "seed": seed,
        "octree_resolution": octree_resolution,
        "memory_settings": memory_settings,
        "fp16": fp16,
        "attention_slicing": attention_slicing,
        "cpu_offload": cpu_offload,
        "remove_bg": remove_bg,
        "output_name": output_name,
        "save_location": save_location,
        "generate_btn": generate_btn,
        "queue_btn": queue_btn,
        "progress_display": progress_display,
        "logs_box": logs_box,
    }
