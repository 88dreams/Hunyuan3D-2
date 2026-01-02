"""
Unified Credentials Manager Component

Provides a single place to manage all RunPod endpoint credentials.
"""

import gradio as gr
import json
import os
from typing import Dict, Any, Tuple
from pathlib import Path


# Config file path
RUNPOD_CONFIG_FILE = Path(__file__).parent.parent.parent / ".runpod_config.json"


def load_all_credentials() -> Dict[str, Dict[str, str]]:
    """Load all saved credentials from config file."""
    if RUNPOD_CONFIG_FILE.exists():
        try:
            with open(RUNPOD_CONFIG_FILE, "r") as f:
                config = json.load(f)
            
            # Organize by model
            credentials = {}
            models = ["gen3c", "sharp", "lyra", "trellis", "hunyuan", "mesh_extraction"]
            
            for model in models:
                endpoint_key = f"{model}_endpoint_id"
                api_key = f"{model}_api_key"
                credentials[model] = {
                    "endpoint_id": config.get(endpoint_key, ""),
                    "api_key": config.get(api_key, ""),
                }
            
            return credentials
        except Exception:
            pass
    
    return {}


def save_model_credentials(model: str, endpoint_id: str, api_key: str) -> str:
    """Save credentials for a specific model."""
    try:
        # Load existing config
        config = {}
        if RUNPOD_CONFIG_FILE.exists():
            with open(RUNPOD_CONFIG_FILE, "r") as f:
                config = json.load(f)
        
        # Update credentials for this model
        config[f"{model}_endpoint_id"] = endpoint_id.strip()
        config[f"{model}_api_key"] = api_key.strip()
        
        # Also update legacy keys for gen3c (backwards compatibility)
        if model == "gen3c":
            config["serverless_endpoint_id"] = endpoint_id.strip()
            config["serverless_api_key"] = api_key.strip()
        
        # Save
        with open(RUNPOD_CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=2)
        
        return f"✅ {model.upper()} credentials saved"
    except Exception as e:
        return f"❌ Failed to save: {e}"


def test_endpoint_connection(endpoint_id: str, api_key: str) -> str:
    """Test connection to a RunPod endpoint."""
    if not endpoint_id or not api_key:
        return "⚠️ Enter endpoint ID and API key first"
    
    try:
        import requests
        
        url = f"https://api.runpod.ai/v2/{endpoint_id}/health"
        headers = {"Authorization": f"Bearer {api_key}"}
        
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            workers = data.get("workers", {})
            ready = workers.get("ready", 0)
            running = workers.get("running", 0)
            return f"✅ Connected | Workers: {ready} ready, {running} running"
        elif response.status_code == 401:
            return "❌ Invalid API key"
        elif response.status_code == 404:
            return "❌ Endpoint not found"
        else:
            return f"⚠️ Status: {response.status_code}"
    except Exception as e:
        return f"❌ Connection failed: {str(e)[:50]}"


def create_credentials_manager() -> Dict[str, Any]:
    """
    Create the unified credentials manager UI.
    
    Returns:
        Dictionary of Gradio components.
    """
    components = {}
    
    # Load existing credentials
    all_creds = load_all_credentials()
    
    gr.Markdown("""
    ## 🔑 RunPod Credentials
    
    Manage API credentials for all RunPod endpoints. Credentials are saved locally
    and persist between sessions.
    """)
    
    # Model-specific credential sections
    models_info = [
        ("gen3c", "GEN3C", "Video generation from single image", "Also used for SHARP, Lyra, and Mesh Extraction"),
        ("trellis", "TRELLIS.2", "High-quality 3D generation", "Separate endpoint required"),
        ("hunyuan", "Hunyuan3D", "Image to GLB mesh", "Separate endpoint required"),
    ]
    
    for model_id, model_name, description, note in models_info:
        creds = all_creds.get(model_id, {})
        
        with gr.Group():
            with gr.Row():
                gr.Markdown(f"### {model_name}")
                components[f"{model_id}_status"] = gr.Textbox(
                    value="",
                    show_label=False,
                    interactive=False,
                    scale=1,
                    max_lines=1,
                )
            
            gr.Markdown(f"*{description}*" + (f" — {note}" if note else ""))
            
            with gr.Row():
                components[f"{model_id}_endpoint"] = gr.Textbox(
                    label="Endpoint ID",
                    value=creds.get("endpoint_id", ""),
                    placeholder="abc123xyz",
                    scale=2,
                )
                components[f"{model_id}_api_key"] = gr.Textbox(
                    label="API Key",
                    value=creds.get("api_key", ""),
                    placeholder="rp_xxxxxxxx",
                    type="password",
                    scale=2,
                )
            
            with gr.Row():
                components[f"{model_id}_save_btn"] = gr.Button(
                    "💾 Save",
                    size="sm",
                )
                components[f"{model_id}_test_btn"] = gr.Button(
                    "🔍 Test",
                    size="sm",
                )
    
    # AWS S3 Credentials (for large file transfers)
    with gr.Accordion("☁️ AWS S3 Credentials (Optional)", open=False):
        gr.Markdown("""
        S3 credentials are used for transferring large files (>30MB) to/from RunPod.
        These are loaded from `~/.config/3d_studio/aws_credentials.env`.
        """)
        
        aws_status = "✅ Configured" if os.environ.get("AWS_ACCESS_KEY_ID") else "⚠️ Not configured"
        components["aws_status"] = gr.Textbox(
            value=aws_status,
            label="AWS Status",
            interactive=False,
        )
        
        gr.Markdown("""
        To configure, create `~/.config/3d_studio/aws_credentials.env` with:
        ```
        AWS_ACCESS_KEY_ID=your_key_id
        AWS_SECRET_ACCESS_KEY=your_secret_key
        ```
        """)
    
    return components

