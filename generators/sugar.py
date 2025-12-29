#!/usr/bin/env python3
"""
SuGaR Mesh Extraction Generator

This module provides client-side functions for extracting meshes from
3D Gaussian Splatting representations using SuGaR (Surface-Aligned
Gaussian Splatting for Efficient 3D Mesh Reconstruction).

SuGaR Reference:
    Paper: https://arxiv.org/abs/2311.12775
    Code: https://github.com/Anttwo/SuGaR
    
The extraction runs on RunPod serverless using the unified GEN3C endpoint.
"""

import os
import base64
from pathlib import Path
from typing import Optional, Tuple

# =============================================================================
# CONSTANTS
# =============================================================================

MESH_DEFAULT_OUTPUT_DIR = "/srv/searidge_share/outputs/mesh"

# Regularization method mapping
REGULARIZATION_MAP = {
    "dn_consistency (Best Quality)": "dn_consistency",
    "density (Object-Centered)": "density",
    "sdf (Background Scenes)": "sdf",
}

# Quality preset mapping
QUALITY_PRESET_MAP = {
    "High Poly (1M vertices, detailed)": {"vertices": 1_000_000, "gaussians_per_triangle": 1},
    "Low Poly (200k vertices, fast)": {"vertices": 200_000, "gaussians_per_triangle": 6},
    "Custom": None,
}

# Refinement time mapping
REFINEMENT_MAP = {
    "short (2k iter)": "short",
    "medium (7k iter)": "medium",
    "long (15k iter)": "long",
}

# S3 configuration (same as used by RunPod handler)
S3_BUCKET = os.environ.get("S3_BUCKET", "arkrunr")
S3_PREFIX = os.environ.get("S3_PREFIX", "MediaContent")
S3_REGION = os.environ.get("S3_REGION", "us-west-1")


def _upload_ply_to_s3(local_path: str, output_name: str) -> Optional[str]:
    """
    Upload a PLY file to S3 for RunPod to download.
    
    Args:
        local_path: Path to local PLY file
        output_name: Base name for the S3 key
    
    Returns:
        S3 URL if successful, None otherwise
    """
    try:
        import boto3
        from botocore.exceptions import ClientError
        
        # Get AWS credentials from environment
        aws_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
        
        if not aws_access_key or not aws_secret_key:
            print("[S3] AWS credentials not configured")
            return None
        
        s3_client = boto3.client(
            "s3",
            region_name=S3_REGION,
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
        )
        
        # Generate S3 key
        filename = os.path.basename(local_path)
        s3_key = f"{S3_PREFIX}/inputs/sugar/{output_name}_{filename}"
        
        # Upload file
        print(f"[S3] Uploading {local_path} to s3://{S3_BUCKET}/{s3_key}")
        s3_client.upload_file(local_path, S3_BUCKET, s3_key)
        
        # Generate presigned URL (valid for 1 hour)
        url = s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": S3_BUCKET, "Key": s3_key},
            ExpiresIn=3600,
        )
        
        return url
        
    except ImportError:
        print("[S3] boto3 not installed")
        return None
    except ClientError as e:
        print(f"[S3] Upload failed: {e}")
        return None
    except Exception as e:
        print(f"[S3] Error: {e}")
        return None


def check_sugar_status(endpoint_id: str = "", api_key: str = "") -> str:
    """
    Check if SuGaR is available on the RunPod endpoint.
    
    Args:
        endpoint_id: RunPod endpoint ID
        api_key: RunPod API key
    
    Returns:
        Status message string
    """
    if not endpoint_id or not api_key:
        return "⚠️ Credentials required"
    
    try:
        from runpod.runpod_client import UnifiedServerlessClient
        
        client = UnifiedServerlessClient(
            endpoint_id=endpoint_id,
            api_key=api_key,
        )
        
        health = client.health_check()
        
        if health.get("status") == "healthy":
            # Check if SuGaR is specifically available
            models = health.get("available_models", [])
            if "sugar" in models or not models:  # Empty means all available
                return "✅ Connected - SuGaR available"
            else:
                return f"⚠️ Connected but SuGaR not in available models: {models}"
        else:
            return f"❌ Endpoint unhealthy: {health.get('message', 'Unknown error')}"
    
    except ImportError:
        return "❌ RunPod client not installed"
    except Exception as e:
        return f"❌ Connection error: {str(e)}"


def run_sugar_extraction(
    input_ply: str,
    input_format: str = "Auto-detect",
    regularization: str = "dn_consistency (Best Quality)",
    quality_preset: str = "High Poly (1M vertices, detailed)",
    poisson_depth: int = 10,
    decimate_faces: int = 0,
    export_texture: bool = True,
    texture_resolution: int = 2048,
    refinement_time: str = "short (2k iter)",
    output_name: str = "mesh_output",
    output_format: str = "GLB (Universal)",
    output_dir: str = MESH_DEFAULT_OUTPUT_DIR,
    endpoint_id: str = "",
    api_key: str = "",
) -> Tuple[Optional[str], str, str]:
    """
    Extract mesh from 3DGS PLY using SuGaR on RunPod.
    
    Args:
        input_ply: Path to input 3DGS PLY file
        input_format: Format hint for the input file
        regularization: SuGaR regularization method
        quality_preset: Mesh quality preset
        poisson_depth: Depth for Poisson reconstruction (6-12)
        decimate_faces: Target face count for decimation (0 = no decimation)
        export_texture: Whether to export textured mesh
        texture_resolution: Texture resolution (1024, 2048, 4096)
        refinement_time: Time to spend on refinement
        output_name: Base name for output files
        output_format: Export format (GLB, OBJ, PLY)
        output_dir: Directory to save outputs
        endpoint_id: RunPod endpoint ID
        api_key: RunPod API key
    
    Returns:
        Tuple of (output_path, logs, status_message)
    """
    logs = []
    
    # Validate inputs
    if not input_ply or not os.path.exists(input_ply):
        return None, "Error: Input PLY file not found", "❌ File not found"
    
    if not endpoint_id or not api_key:
        return None, "Error: RunPod credentials required", "❌ Missing credentials"
    
    # Parse settings
    reg_method = REGULARIZATION_MAP.get(regularization, "dn_consistency")
    quality = QUALITY_PRESET_MAP.get(quality_preset)
    refine = REFINEMENT_MAP.get(refinement_time, "short")
    
    # Determine export format
    export_glb = "GLB" in output_format
    export_obj = "OBJ" in output_format
    
    logs.append(f"[SuGaR] Input: {input_ply}")
    logs.append(f"[SuGaR] Format: {input_format}")
    logs.append(f"[SuGaR] Regularization: {reg_method}")
    logs.append(f"[SuGaR] Quality: {quality_preset}")
    logs.append(f"[SuGaR] Poisson Depth: {poisson_depth}")
    logs.append(f"[SuGaR] Refinement: {refine}")
    
    try:
        from runpod.runpod_client import UnifiedServerlessClient
        
        client = UnifiedServerlessClient(
            endpoint_id=endpoint_id,
            api_key=api_key,
        )
        
        logs.append("[SuGaR] Submitting to RunPod...")
        
        # Check file size
        file_size_mb = os.path.getsize(input_ply) / (1024 * 1024)
        logs.append(f"[SuGaR] Input file size: {file_size_mb:.1f} MB")
        
        # Build job payload
        job_input = {
            "model": "sugar",
            "input_format": input_format.lower().replace(" ", "_"),
            "regularization": reg_method,
            "poisson_depth": poisson_depth,
            "refinement_time": refine,
            "export_texture": export_texture,
            "texture_resolution": texture_resolution,
            "export_glb": export_glb,
            "export_obj": export_obj,
            "output_name": output_name,
            "return_base64": False,  # Use S3 for large mesh files
        }
        
        # For large files (>30MB), upload to S3 first
        # For small files, use base64 encoding
        if file_size_mb > 30:
            logs.append("[SuGaR] Large file detected - uploading to S3...")
            
            # Upload to S3
            s3_url = _upload_ply_to_s3(input_ply, output_name)
            if s3_url:
                job_input["input_s3_url"] = s3_url
                logs.append(f"[SuGaR] Uploaded to S3: {s3_url}")
            else:
                # Fallback to base64 if S3 upload fails
                logs.append("[SuGaR] S3 upload failed, falling back to base64...")
                logs.append("[SuGaR] Warning: Large file may cause API timeout")
                with open(input_ply, "rb") as f:
                    ply_data = f.read()
                ply_base64 = base64.b64encode(ply_data).decode("utf-8")
                job_input["ply_base64"] = ply_base64
        else:
            # Small file - use base64 encoding
            logs.append("[SuGaR] Encoding file as base64...")
            with open(input_ply, "rb") as f:
                ply_data = f.read()
            ply_base64 = base64.b64encode(ply_data).decode("utf-8")
            job_input["ply_base64"] = ply_base64
        
        # Add quality settings
        if quality:
            job_input["target_vertices"] = quality["vertices"]
            job_input["gaussians_per_triangle"] = quality["gaussians_per_triangle"]
        
        # Add decimation if specified
        if decimate_faces > 0:
            job_input["decimate_faces"] = decimate_faces
        
        # Submit job
        submit_result = client.submit_generic_job(job_input)
        job_id = submit_result.get("job_id", "")
        logs.append(f"[SuGaR] Job submitted: {job_id}")
        
        # Wait for completion
        # SuGaR can take 15-60 minutes depending on settings
        timeout = 3600 if refine == "long" else (1800 if refine == "medium" else 900)
        logs.append(f"[SuGaR] Timeout set to {timeout}s ({timeout//60} minutes)")
        
        final_status = client.wait_for_completion(
            job_id=job_id,
            poll_interval=30,
            max_wait=timeout,
        )
        
        if final_status.get("status") == "completed":
            # Note: get_status() normalizes the response and puts mesh_* fields
            # directly in final_status, not in a nested "output" dict
            logs.append(f"[SuGaR] Job completed. Checking for output...")
            logs.append(f"[SuGaR] Response keys: {list(final_status.keys())}")
            
            # Handle S3 download
            mesh_s3_url = final_status.get("mesh_s3_url")
            if mesh_s3_url:
                logs.append(f"[SuGaR] Downloading from S3...")
                
                # Ensure output directory exists
                os.makedirs(output_dir, exist_ok=True)
                
                # Use module-level download function
                from runpod.runpod_client import download_from_s3
                local_path, s3_msg = download_from_s3(mesh_s3_url, output_dir)
                logs.append(s3_msg)
                
                if local_path and os.path.exists(local_path):
                    logs.append(f"[SuGaR] Downloaded: {local_path}")
                    return local_path, "\n".join(logs), "✅ Mesh extraction complete!"
                else:
                    logs.append("[SuGaR] S3 download failed")
            
            # Check for base64 data (small files)
            mesh_base64 = final_status.get("mesh_base64")
            if mesh_base64:
                logs.append(f"[SuGaR] Received base64 data, saving...")
                os.makedirs(output_dir, exist_ok=True)
                ext = ".glb" if export_glb else (".obj" if export_obj else ".ply")
                local_path = os.path.join(output_dir, f"{output_name}{ext}")
                
                with open(local_path, "wb") as f:
                    f.write(base64.b64decode(mesh_base64))
                
                logs.append(f"[SuGaR] Saved: {local_path}")
                return local_path, "\n".join(logs), "✅ Mesh extraction complete!"
            
            # Check for remote path (file too large, saved on RunPod)
            remote_path = final_status.get("mesh_path")
            if remote_path:
                logs.append(f"[SuGaR] Mesh saved on RunPod: {remote_path}")
                return None, "\n".join(logs), f"⚠️ Mesh on RunPod: {remote_path}"
            
            logs.append("[SuGaR] No output received in response")
            logs.append(f"[SuGaR] Response keys: {list(final_status.keys())}")
            return None, "\n".join(logs), "❌ No mesh output"
        
        elif final_status.get("status") == "failed":
            error = final_status.get("error", "Unknown error")
            logs.append(f"[SuGaR] Failed: {error}")
            return None, "\n".join(logs), f"❌ {error}"
        
        else:
            logs.append(f"[SuGaR] Timeout after {timeout}s")
            return None, "\n".join(logs), "❌ Timeout"
    
    except ImportError as e:
        logs.append(f"[SuGaR] Import error: {e}")
        return None, "\n".join(logs), "❌ RunPod client not available"
    
    except Exception as e:
        logs.append(f"[SuGaR] Error: {str(e)}")
        return None, "\n".join(logs), f"❌ {str(e)}"


def run_tsdf_extraction(
    input_ply: str,
    input_format: str = "Auto-detect",
    voxel_size: float = 0.01,
    num_views: int = 32,
    output_name: str = "mesh_output",
    output_dir: str = MESH_DEFAULT_OUTPUT_DIR,
    endpoint_id: str = "",
    api_key: str = "",
) -> Tuple[Optional[str], str, str]:
    """
    Extract mesh from 3DGS PLY using TSDF fusion (fast, lower quality).
    
    This is a simpler alternative to SuGaR that renders depth maps from
    the Gaussians and fuses them using TSDF + marching cubes.
    
    Args:
        input_ply: Path to input 3DGS PLY file
        input_format: Format hint for the input file
        voxel_size: TSDF voxel size (smaller = more detail)
        num_views: Number of views to render for depth fusion
        output_name: Base name for output files
        output_dir: Directory to save outputs
        endpoint_id: RunPod endpoint ID
        api_key: RunPod API key
    
    Returns:
        Tuple of (output_path, logs, status_message)
    """
    logs = []
    
    # Validate inputs
    if not input_ply or not os.path.exists(input_ply):
        return None, "Error: Input PLY file not found", "❌ File not found"
    
    if not endpoint_id or not api_key:
        return None, "Error: RunPod credentials required", "❌ Missing credentials"
    
    logs.append(f"[TSDF] Input: {input_ply}")
    logs.append(f"[TSDF] Voxel size: {voxel_size}")
    logs.append(f"[TSDF] Views: {num_views}")
    
    try:
        from runpod.runpod_client import UnifiedServerlessClient
        
        client = UnifiedServerlessClient(
            endpoint_id=endpoint_id,
            api_key=api_key,
        )
        
        logs.append("[TSDF] Submitting to RunPod...")
        
        # Check file size
        file_size_mb = os.path.getsize(input_ply) / (1024 * 1024)
        logs.append(f"[TSDF] Input file size: {file_size_mb:.1f} MB")
        
        # Determine if we should use path or base64
        use_path = input_ply.startswith("/srv/searidge_share/") or input_ply.startswith("/runpod-volume/")
        
        # Build job payload
        job_input = {
            "model": "tsdf",
            "input_format": input_format.lower().replace(" ", "_"),
            "voxel_size": voxel_size,
            "num_views": num_views,
            "output_name": output_name,
            "return_base64": False,
        }
        
        if use_path:
            # Convert local path to RunPod network volume path
            runpod_path = input_ply.replace("/srv/searidge_share/", "/runpod-volume/")
            job_input["input_ply_path"] = runpod_path
            logs.append(f"[TSDF] Using network volume path: {runpod_path}")
        else:
            # Small file or not on shared volume - use base64
            with open(input_ply, "rb") as f:
                ply_data = f.read()
            ply_base64 = base64.b64encode(ply_data).decode("utf-8")
            job_input["ply_base64"] = ply_base64
        
        # Submit and wait (TSDF is much faster)
        submit_result = client.submit_generic_job(job_input)
        job_id = submit_result.get("job_id", "")
        logs.append(f"[TSDF] Job submitted: {job_id}")
        
        final_status = client.wait_for_completion(
            job_id=job_id,
            poll_interval=10,
            max_wait=300,  # 5 minutes should be plenty for TSDF
        )
        
        if final_status.get("status") == "completed":
            output = final_status.get("output", {})
            
            # Handle S3 download
            mesh_s3_url = output.get("mesh_s3_url")
            if mesh_s3_url:
                os.makedirs(output_dir, exist_ok=True)
                local_path = os.path.join(output_dir, f"{output_name}.ply")
                
                downloaded = client.download_from_s3(mesh_s3_url, local_path)
                if downloaded and os.path.exists(local_path):
                    logs.append(f"[TSDF] Downloaded: {local_path}")
                    return local_path, "\n".join(logs), "✅ TSDF extraction complete!"
            
            logs.append("[TSDF] No output received")
            return None, "\n".join(logs), "❌ No mesh output"
        
        elif final_status.get("status") == "failed":
            error = final_status.get("error", "Unknown error")
            logs.append(f"[TSDF] Failed: {error}")
            return None, "\n".join(logs), f"❌ {error}"
        
        else:
            logs.append("[TSDF] Timeout")
            return None, "\n".join(logs), "❌ Timeout"
    
    except Exception as e:
        logs.append(f"[TSDF] Error: {str(e)}")
        return None, "\n".join(logs), f"❌ {str(e)}"


def detect_ply_format(ply_path: str) -> str:
    """
    Detect the format of a PLY file.
    
    Args:
        ply_path: Path to PLY file
    
    Returns:
        Format string: "lyra", "standard_3dgs", "sharp", or "unknown"
    """
    try:
        # Check if it's a PyTorch file (Lyra format)
        import torch
        try:
            data = torch.load(ply_path, map_location="cpu", weights_only=False)
            if isinstance(data, torch.Tensor):
                return "lyra"
        except:
            pass
        
        # Check PLY header
        with open(ply_path, "rb") as f:
            header = f.read(1024).decode("utf-8", errors="ignore")
        
        # Look for characteristic properties
        if "f_dc_0" in header or "f_rest_" in header:
            return "standard_3dgs"
        elif "red" in header and "green" in header and "blue" in header:
            if "scale_0" in header or "rot_0" in header:
                return "standard_3dgs"
            else:
                return "point_cloud"
        
        return "unknown"
    
    except Exception:
        return "unknown"

