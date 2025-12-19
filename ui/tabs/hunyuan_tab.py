"""
Hunyuan3D Tab for 3D Generation Studio

This module defines the Hunyuan3D tab UI components and layout.
"""

import torch  # type: ignore
import gradio as gr  # type: ignore

from generators.hunyuan import HUNYUAN_DEFAULT_OUTPUT_DIR


def create_hunyuan_tab():
    """
    Create the Hunyuan3D tab UI components.
    
    Returns:
        Dictionary of all UI components for event handler binding
    """
    gr.Markdown("**Image → GLB Mesh** | High-quality 3D mesh generation")
    
    # Execution Settings
    with gr.Group():
        gr.Markdown("#### Execution")
        exec_mode = gr.Radio(
            choices=["Local CUDA"],
            value="Local CUDA",
            label="Mode",
            info="Hunyuan runs locally on your GPU",
        )
        status = gr.Textbox(
            value="Local GPU Ready" if torch.cuda.is_available() else "CPU Mode (slower)",
            label="Status",
            interactive=False,
        )
    
    # Model Settings
    with gr.Group():
        gr.Markdown("#### Model Settings")
        model_choice = gr.Radio(
            choices=["Mini Model (Faster)", "Full Model (Higher Quality)"],
            value="Mini Model (Faster)",
            label="Model",
        )
        with gr.Row():
            guidance = gr.Slider(1.0, 15.0, value=9.0, step=0.5, label="Guidance")
            steps = gr.Slider(10, 100, value=40, step=1, label="Steps")
        seed = gr.Number(value=42, precision=0, label="Seed (optional)")
    
    # Memory Settings
    with gr.Group():
        gr.Markdown("#### Memory Optimization")
        with gr.Row():
            fp16 = gr.Checkbox(value=True, label="FP16")
            attention_slicing = gr.Checkbox(value=True, label="Attention Slicing")
            cpu_offload = gr.Checkbox(value=True, label="CPU Offload")
        remove_bg = gr.Checkbox(value=False, label="Remove Background")
    
    # Output Settings
    with gr.Group():
        gr.Markdown("#### Output")
        output_name = gr.Textbox(value="hunyuan_output", label="Output Name")
        save_location = gr.Textbox(
            value=HUNYUAN_DEFAULT_OUTPUT_DIR,
            label="Save Location",
        )
    
    # Generate Button (Fixed Position)
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
        "model_choice": model_choice,
        "guidance": guidance,
        "steps": steps,
        "seed": seed,
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

