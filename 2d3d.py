#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""
3D Generation Studio - Multi-Model Gradio Interface

This module provides a unified web interface for multiple 3D generation models:
- Hunyuan3D: Image to GLB mesh generation
- GEN3C: Image to video generation
- Lyra: Image/Video to 3D Gaussian Splatting (coming soon)
- SHARP: Fast Gaussian splatting (coming soon)
- TRELLIS.2: High-quality 3D generation (coming soon)

Multi-System Configuration:
    All paths are loaded from config/multi_system.yaml to support distributed
    deployment across the searidge cluster (searidge01, searidge02, searidge03).

Execution Modes:
    - RunPod Serverless: Jobs run on RunPod cloud GPUs (recommended for GEN3C)
    - RunPod Pod: Jobs run on dedicated RunPod pod
    - Local CUDA: Jobs run on local GPU
    - Local Cluster (Ray): Jobs distributed across local cluster
"""
print("DEBUG: STARTING 3D GENERATION STUDIO v2.1 (Port 5683) - Refactored")

import os
import json
import time
from typing import Optional, Tuple, Union

import torch  # type: ignore
import gradio as gr  # type: ignore

# =============================================================================
# LOCAL IMPORTS
# =============================================================================

from ui.styles import CUSTOM_CSS
from ui.tabs import (
    create_hunyuan_tab,
    create_gen3c_tab,
    create_lyra_tab,
    create_sharp_tab,
    create_trellis_tab,
)

from generators import (
    run_hunyuan,
    run_gen3c_runpod,
    run_gen3c_serverless,
    run_gen3c_local,
    check_runpod_status,
    check_serverless_status,
    cancel_serverless_job,
    GEN3C_DEFAULT_OUTPUT_DIR,
)

from job_queue import (
    get_queue_display,
    clear_completed_jobs,
)

from utils import (
    clamp_scale_value,
    update_image_info_display,
    maybe_downscale_image,
    start_monitoring,
    get_system_metrics,
    format_system_metrics,
)


# =============================================================================
# CLUSTER / JOB MANAGER
# =============================================================================

_job_manager = None
_cluster_available = False

try:
    from cluster import JobManager, get_cluster_status
    _cluster_available = True
    print("[CLUSTER] Cluster module loaded successfully")
except ImportError as e:
    print(f"[CLUSTER] Warning: Cluster module not available ({e})")
    get_cluster_status = lambda: {"available": False, "mode": "local", "message": "Cluster module not installed"}


def _init_job_manager(use_ray: Optional[bool] = None) -> Optional['JobManager']:
    """Initialize the job manager with optional Ray support."""
    global _job_manager
    
    if not _cluster_available:
        print("[CLUSTER] Job manager not available (cluster module not installed)")
        return None
    
    try:
        if use_ray is None:
            try:
                from config import is_ray_enabled, should_auto_connect_ray
                use_ray = is_ray_enabled() and should_auto_connect_ray()
            except ImportError:
                use_ray = True
        
        _job_manager = JobManager(use_ray=use_ray)
        print(f"[CLUSTER] Job manager initialized in {_job_manager.mode} mode")
        return _job_manager
    except Exception as e:
        print(f"[CLUSTER] Failed to initialize job manager: {e}")
        return None


def format_cluster_status() -> str:
    """Format cluster status for display in UI."""
    if not _cluster_available:
        return "🔴 Cluster module not installed - Local mode only"
    
    if _job_manager is None:
        return "🟡 Job manager not initialized"
    
    status = _job_manager.cluster_status
    
    if status.get("available"):
        mode = status.get("mode", "unknown")
        gpus = status.get("gpus", 0)
        nodes = len(status.get("nodes", []))
        return f"🟢 {mode.title()} Mode | {nodes} nodes | {gpus} GPUs available"
    else:
        message = status.get("message", "Not connected")
        return f"🟡 Local Mode | {message}"


# =============================================================================
# PERSISTENT RUNPOD CONFIG
# =============================================================================

RUNPOD_CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".runpod_config.json")


def _load_runpod_config() -> dict:
    """Load RunPod config from local file."""
    if os.path.exists(RUNPOD_CONFIG_FILE):
        try:
            with open(RUNPOD_CONFIG_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_runpod_config(config: dict) -> None:
    """Save RunPod config to local file."""
    try:
        with open(RUNPOD_CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        print(f"[CONFIG] Warning: Could not save RunPod config: {e}")


def save_serverless_credentials(endpoint_id: str, api_key: str) -> Tuple[str, dict]:
    """Save serverless credentials to config file."""
    config = _load_runpod_config()
    config["serverless_endpoint_id"] = endpoint_id.strip() if endpoint_id else ""
    config["serverless_api_key"] = api_key.strip() if api_key else ""
    _save_runpod_config(config)
    return "✅ Credentials saved", gr.update(visible=False)


def save_pod_url(pod_url: str) -> str:
    """Save pod URL to config file."""
    config = _load_runpod_config()
    config["pod_url"] = pod_url.strip() if pod_url else ""
    _save_runpod_config(config)
    return "✅ Pod URL saved"


# Load saved config
_runpod_config = _load_runpod_config()
DEFAULT_RUNPOD_URL = _runpod_config.get("pod_url", "https://iv94zuokozefxc-8000.proxy.runpod.net")
DEFAULT_SERVERLESS_ENDPOINT = _runpod_config.get("serverless_endpoint_id", "")
DEFAULT_SERVERLESS_API_KEY = _runpod_config.get("serverless_api_key", "")

print(f"[CONFIG] Loaded RunPod config: endpoint={'set' if DEFAULT_SERVERLESS_ENDPOINT else 'not set'}")


# =============================================================================
# UNIFIED GENERATION HANDLERS
# =============================================================================

def handle_hunyuan_generation(
    image_path: Union[str, None],
    image_scale: Union[float, int],
    exec_mode: str,
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
) -> Tuple[Optional[str], str, str]:
    """Handle Hunyuan3D generation."""
    scale_value = clamp_scale_value(image_scale)
    scaled_path, temp_scaled = maybe_downscale_image(image_path, scale_value)
    effective_image_path = scaled_path or image_path

    try:
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
            save_location=save_location,
        )
    finally:
        if temp_scaled and os.path.exists(temp_scaled):
            try:
                os.remove(temp_scaled)
            except Exception:
                pass


def handle_gen3c_generation(
    image_path: Union[str, None],
    image_scale: Union[float, int],
    exec_mode: str,
    runpod_url: str,
    serverless_endpoint_id: str,
    serverless_api_key: str,
    guidance: float,
    frames: Union[str, int],
    trajectory: str,
    foreground_mask: bool,
    video_name: str,
    seed: Optional[int],
    checkpoint_dir: str,
    output_dir: str,
    extra_args: str,
) -> Tuple[Optional[str], str, str]:
    """Handle GEN3C generation based on execution mode."""
    scale_value = clamp_scale_value(image_scale)
    scaled_path, temp_scaled = maybe_downscale_image(image_path, scale_value)
    effective_image_path = scaled_path or image_path
    
    resolved_output_dir = output_dir.strip() if output_dir else GEN3C_DEFAULT_OUTPUT_DIR
    
    try:
        if exec_mode == "RunPod Pod":
            return run_gen3c_runpod(
                image_path=effective_image_path,
                runpod_url=runpod_url,
                guidance=guidance,
                frames=frames,
                trajectory=trajectory,
                foreground_masking=foreground_mask,
                video_name=video_name,
                seed=seed,
                output_dir=resolved_output_dir,
            )
        elif exec_mode == "RunPod Serverless":
            return run_gen3c_serverless(
                image_path=effective_image_path,
                endpoint_id=serverless_endpoint_id,
                api_key=serverless_api_key,
                guidance=guidance,
                frames=frames,
                trajectory=trajectory,
                foreground_masking=foreground_mask,
                video_name=video_name,
                seed=seed,
                output_dir=resolved_output_dir,
            )
        else:
            # Local execution
            extra_args_parts = []
            if trajectory:
                extra_args_parts.append(f"--trajectory {trajectory}")
            if foreground_mask:
                extra_args_parts.append("--foreground_masking")
            if extra_args and extra_args.strip():
                extra_args_parts.append(extra_args.strip())
            
            combined_extra_args = " ".join(extra_args_parts)
            
            return run_gen3c_local(
                image_path=effective_image_path,
                guidance=guidance,
                frames=frames,
                video_name=video_name,
                checkpoint_dir=checkpoint_dir,
                output_dir=resolved_output_dir,
                extra_args=combined_extra_args,
            )
    finally:
        if temp_scaled and os.path.exists(temp_scaled):
            try:
                os.remove(temp_scaled)
            except Exception:
                pass


# =============================================================================
# GRADIO INTERFACE
# =============================================================================

with gr.Blocks(title="3D Generation Studio") as demo:
    
    # Header
    gr.Markdown("""
    # 3D Generation Studio
    **Multi-model 3D generation from images and videos**
    """)

    with gr.Row():
        # =================================================================
        # LEFT COLUMN - Input, Output, Queue, System
        # =================================================================
        with gr.Column(scale=1):
            
            # --- Input Upload Area ---
            with gr.Group():
                gr.Markdown("### Input")
                input_image = gr.Image(
                    type="filepath",
                    label="Input Image",
                    height=250,
                )
                input_video = gr.Video(
                    label="Input Video",
                    visible=False,
                    height=250,
                )
            
            # --- Input Parameters ---
            with gr.Group():
                gr.Markdown("### Input Parameters")
                with gr.Row():
                    input_type_selector = gr.Dropdown(
                        choices=["Image"],
                        value="Image",
                        label="Input Type",
                        scale=1,
                        interactive=False,
                    )
                    image_scale_slider = gr.Slider(
                        minimum=0.25,
                        maximum=1.0,
                        value=1.0,
                        step=0.05,
                        label="Scale",
                        scale=2,
                    )
                image_resolution_display = gr.Textbox(
                    label="Resolution Info",
                    value="No image loaded.",
                    interactive=False,
                    lines=1,
                )
            
            # --- Output Viewer ---
            with gr.Group():
                gr.Markdown("### Output")
                output_model_viewer = gr.Model3D(
                    label="3D Model Viewer",
                    height=300,
                    visible=True,
                )
                output_video_player = gr.Video(
                    label="Video Output",
                    height=300,
                    visible=False,
                )
            
            # --- Job Queue ---
            with gr.Group():
                with gr.Row():
                    gr.Markdown("### Job Queue")
                    clear_queue_btn = gr.Button("Clear", size="sm", scale=0)
                
                job_queue_display = gr.Dataframe(
                    headers=["ID", "Model", "Status", "Input"],
                    datatype=["str", "str", "str", "str"],
                    value=[],
                    row_count=(3, "dynamic"),
                    interactive=False,
                )
            
            # --- System Resources ---
            with gr.Group():
                gr.Markdown("### System Resources")
                system_metrics_display = gr.Textbox(
                    value="Loading system metrics...",
                    interactive=False,
                    lines=2,
                    show_label=False,
                )

        # =================================================================
        # RIGHT COLUMN - Model Tabs and Settings
        # =================================================================
        with gr.Column(scale=1):
            
            with gr.Tabs() as model_tabs:
                
                # TAB 1: HUNYUAN 3D
                with gr.TabItem("Hunyuan 3D", id="hunyuan"):
                    hunyuan = create_hunyuan_tab()
                
                # TAB 2: GEN3C VIDEO
                with gr.TabItem("GEN3C Video", id="gen3c"):
                    gen3c = create_gen3c_tab(
                        default_runpod_url=DEFAULT_RUNPOD_URL,
                        default_endpoint_id=DEFAULT_SERVERLESS_ENDPOINT,
                        default_api_key=DEFAULT_SERVERLESS_API_KEY,
                        format_cluster_status_fn=format_cluster_status,
                    )
                
                # TAB 3: LYRA 3DGS (Coming Soon)
                with gr.TabItem("Lyra 3DGS", id="lyra"):
                    lyra = create_lyra_tab()
                
                # TAB 4: SHARP (Coming Soon)
                with gr.TabItem("SHARP", id="sharp"):
                    sharp = create_sharp_tab()
                
                # TAB 5: TRELLIS.2 (Coming Soon)
                with gr.TabItem("TRELLIS.2", id="trellis"):
                    trellis = create_trellis_tab()
    
    # =========================================================================
    # EVENT HANDLERS
    # =========================================================================
    
    # --- Image Preview Updates ---
    def update_resolution_on_image_change(image_path, scale):
        return update_image_info_display(image_path, scale)
    
    input_image.change(
        fn=update_resolution_on_image_change,
        inputs=[input_image, image_scale_slider],
        outputs=[image_resolution_display],
    )
    
    image_scale_slider.change(
        fn=update_resolution_on_image_change,
        inputs=[input_image, image_scale_slider],
        outputs=[image_resolution_display],
    )
    
    # --- GEN3C Mode Toggle ---
    def gen3c_mode_change(mode: str):
        """Toggle visibility of GEN3C execution mode settings."""
        show_pod = mode == "RunPod Pod"
        show_serverless = mode == "RunPod Serverless"
        show_local = mode == "Local Cluster"
        
        if show_pod:
            status = "Enter Pod URL and click Check Status"
        elif show_serverless:
            if DEFAULT_SERVERLESS_ENDPOINT and DEFAULT_SERVERLESS_API_KEY:
                status = "✅ Credentials loaded - Click Check Status"
            else:
                status = "⚠️ Enter credentials below"
        else:
            status = format_cluster_status()
        
        return (
            gr.update(visible=show_pod),
            gr.update(visible=show_serverless),
            gr.update(visible=show_local),
            status,
        )
    
    gen3c["exec_mode"].change(
        fn=gen3c_mode_change,
        inputs=[gen3c["exec_mode"]],
        outputs=[gen3c["pod_settings"], gen3c["serverless_settings"], gen3c["local_settings"], gen3c["status"]],
    )
    
    # --- GEN3C Status Checks ---
    gen3c["check_pod_btn"].click(
        fn=check_runpod_status,
        inputs=[gen3c["runpod_url"]],
        outputs=[gen3c["status"]],
    )
    
    gen3c["save_pod_btn"].click(
        fn=save_pod_url,
        inputs=[gen3c["runpod_url"]],
        outputs=[gen3c["status"]],
    )
    
    gen3c["check_serverless_btn"].click(
        fn=check_serverless_status,
        inputs=[gen3c["endpoint_id"], gen3c["api_key"]],
        outputs=[gen3c["status"]],
    )
    
    gen3c["save_creds_btn"].click(
        fn=save_serverless_credentials,
        inputs=[gen3c["endpoint_id"], gen3c["api_key"]],
        outputs=[gen3c["status"], gen3c["creds_group"]],
    )
    
    gen3c["cancel_btn"].click(
        fn=cancel_serverless_job,
        inputs=[gen3c["endpoint_id"], gen3c["api_key"], gen3c["current_job_id"]],
        outputs=[gen3c["status"], gen3c["current_job_id"]],
    )
    
    def toggle_creds_visibility():
        return gr.update(visible=True)
    
    gen3c["edit_creds_btn"].click(
        fn=toggle_creds_visibility,
        outputs=[gen3c["creds_group"]],
    )
    
    gen3c["check_cluster_btn"].click(
        fn=format_cluster_status,
        outputs=[gen3c["status"]],
    )
    
    # --- Hunyuan Generation ---
    hunyuan["generate_btn"].click(
        fn=handle_hunyuan_generation,
        inputs=[
            input_image, image_scale_slider, hunyuan["exec_mode"],
            hunyuan["guidance"], hunyuan["steps"], hunyuan["seed"], hunyuan["model_choice"],
            hunyuan["fp16"], hunyuan["attention_slicing"], hunyuan["cpu_offload"], hunyuan["remove_bg"],
            hunyuan["output_name"], hunyuan["save_location"],
        ],
        outputs=[output_model_viewer, hunyuan["logs_box"], hunyuan["progress_display"]],
    )
    
    # --- GEN3C Generation ---
    gen3c["generate_btn"].click(
        fn=handle_gen3c_generation,
        inputs=[
            input_image, image_scale_slider, gen3c["exec_mode"],
            gen3c["runpod_url"], gen3c["endpoint_id"], gen3c["api_key"],
            gen3c["guidance"], gen3c["frames"], gen3c["trajectory"], gen3c["foreground_mask"],
            gen3c["video_name"], gen3c["seed"], gen3c["checkpoint_dir"], gen3c["output_dir"], gen3c["extra_args"],
        ],
        outputs=[output_video_player, gen3c["logs_box"], gen3c["progress_display"]],
    )
    
    # --- Clear Queue ---
    def clear_queue_and_status():
        data, msg = clear_completed_jobs()
        return data
    
    clear_queue_btn.click(
        fn=clear_queue_and_status,
        outputs=[job_queue_display],
    )
    
    # --- System Metrics Timer ---
    def update_metrics():
        return format_system_metrics(get_system_metrics())

    timer = gr.Timer(2.0)
    timer.tick(fn=update_metrics, outputs=[system_metrics_display])


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Initialize job manager for distributed execution
    print("[STARTUP] Initializing job manager...")
    _init_job_manager()
    
    # Start monitoring thread
    start_monitoring()
    
    # Give monitoring thread time to start
    time.sleep(0.5)
    
    # Print startup status
    print(f"[STARTUP] Cluster status: {format_cluster_status()}")
    print("[STARTUP] 3D Generation Studio v2.1 ready!")

    demo.launch(server_port=5683, share=False, css=CUSTOM_CSS)
