#!/usr/bin/env python3
"""
Centralized Experiment Logger for ArkRunr Pipeline

Provides consistent parameter logging across all models:
- SHARP, Gen3C, Lyra, Trellis, Hunyuan, Mesh Cleanup

Creates:
1. JSON sidecar files per output (output_name.model.json)
2. CSV experiment logs per model (model_name.csv)

All CSV logs are stored in: /srv/searidge_share/outputs/logs/
"""

import json
import csv
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

# Central log directory for all experiment CSVs
LOG_DIR = Path("/srv/searidge_share/outputs/logs")


def ensure_log_dir():
    """Ensure the central log directory exists."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    return LOG_DIR


def get_csv_path(model_name: str) -> Path:
    """Get the CSV path for a specific model."""
    ensure_log_dir()
    return LOG_DIR / f"{model_name}.csv"


def save_experiment_log(
    model_name: str,
    output_path: str,
    params: Dict[str, Any],
    results: Dict[str, Any],
    save_json_sidecar: bool = True,
) -> Dict[str, str]:
    """
    Save experiment parameters and results.
    
    Args:
        model_name: Name of the model (sharp, gen3c, lyra, trellis, hunyuan, mesh_cleanup)
        output_path: Path to the output file
        params: Dictionary of input parameters used
        results: Dictionary of results/statistics
        save_json_sidecar: Whether to save JSON file alongside output
    
    Returns:
        Dictionary with paths to created log files
    """
    timestamp = datetime.now().isoformat()
    output_path = Path(output_path)
    
    # Build the full record
    record = {
        "timestamp": timestamp,
        "model": model_name,
        "output_file": str(output_path),
        "parameters": params,
        "results": results,
    }
    
    created_files = {}
    
    # 1. Save JSON sidecar file
    if save_json_sidecar:
        json_path = output_path.with_suffix(output_path.suffix + f".{model_name}.json")
        try:
            with open(json_path, "w") as f:
                json.dump(record, f, indent=2, default=str)
            created_files["json"] = str(json_path)
            print(f"[Logger] Parameter log saved: {json_path}")
        except Exception as e:
            print(f"[Logger] Warning: Could not save JSON sidecar: {e}")
    
    # 2. Append to CSV log
    csv_path = get_csv_path(model_name)
    csv_exists = csv_path.exists()
    
    # Flatten params and results for CSV
    csv_row = {"timestamp": timestamp, "output_file": output_path.name}
    
    # Add all params with prefix
    for key, value in params.items():
        csv_row[f"p_{key}"] = value
    
    # Add all results with prefix
    for key, value in results.items():
        # Skip nested dicts/lists for CSV
        if not isinstance(value, (dict, list)):
            csv_row[f"r_{key}"] = value
    
    try:
        # Read existing headers if file exists
        if csv_exists:
            with open(csv_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                existing_headers = reader.fieldnames or []
        else:
            existing_headers = []
        
        # Merge headers (keep existing + add new)
        all_headers = list(existing_headers)
        for key in csv_row.keys():
            if key not in all_headers:
                all_headers.append(key)
        
        # If we have new headers, we need to rewrite the file
        if set(all_headers) != set(existing_headers) and csv_exists:
            # Read existing data
            with open(csv_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                existing_rows = list(reader)
            
            # Rewrite with new headers
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=all_headers)
                writer.writeheader()
                for row in existing_rows:
                    writer.writerow(row)
                writer.writerow(csv_row)
        else:
            # Simple append
            with open(csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=all_headers if all_headers else csv_row.keys())
                if not csv_exists:
                    writer.writeheader()
                writer.writerow(csv_row)
        
        created_files["csv"] = str(csv_path)
        print(f"[Logger] Experiment logged to: {csv_path}")
        
    except Exception as e:
        print(f"[Logger] Warning: Could not append to CSV: {e}")
    
    return created_files


def generate_param_filename(
    base_name: str,
    model_name: str,
    key_params: Dict[str, Any],
    ext: str = ".glb",
) -> str:
    """
    Generate a filename that encodes key parameters.
    
    Args:
        base_name: Original file name (without extension)
        model_name: Model name for prefix
        key_params: Dictionary of key parameters to encode
        ext: File extension
    
    Returns:
        Filename with encoded parameters
    
    Examples:
        sharp_kitchen_g10_s50.glb
        lyra_room_g9_sdg350.glb
        cleanup_kitchen_t300k_pd1_ps2.glb
    """
    # Build param string from key params
    param_parts = []
    for key, value in key_params.items():
        if isinstance(value, bool):
            param_parts.append(f"{key}{1 if value else 0}")
        elif isinstance(value, float):
            param_parts.append(f"{key}{value:.1f}".replace(".", ""))
        elif isinstance(value, int):
            # Shorten large numbers
            if value >= 1000:
                param_parts.append(f"{key}{value // 1000}k")
            else:
                param_parts.append(f"{key}{value}")
        else:
            param_parts.append(f"{key}{value}")
    
    param_str = "_".join(param_parts)
    
    # Clean base name
    base_name = base_name.replace("-CLEAN", "").replace("_cleaned", "")
    base_name = base_name.replace("-OUTPUT", "").replace("_output", "")
    
    return f"{model_name}_{base_name}_{param_str}{ext}"


# =============================================================================
# Model-specific logging helpers
# =============================================================================

def log_sharp_experiment(
    output_path: str,
    input_image: str,
    seed: int,
    guidance_scale: float,
    inference_steps: int,
    output_format: str,
    results: Dict[str, Any],
    save_json: bool = True,
) -> Dict[str, str]:
    """Log SHARP experiment."""
    params = {
        "input_image": Path(input_image).name if input_image else "unknown",
        "seed": seed,
        "guidance_scale": guidance_scale,
        "inference_steps": inference_steps,
        "output_format": output_format,
    }
    return save_experiment_log("sharp", output_path, params, results, save_json)


def log_gen3c_experiment(
    output_path: str,
    input_image: str,
    seed: int,
    num_frames: int,
    guidance_scale: float,
    trajectory: str,
    camera_rotation: float,
    movement_distance: float,
    results: Dict[str, Any],
    save_json: bool = True,
) -> Dict[str, str]:
    """Log Gen3C experiment."""
    params = {
        "input_image": Path(input_image).name if input_image else "unknown",
        "seed": seed,
        "num_frames": num_frames,
        "guidance_scale": guidance_scale,
        "trajectory": trajectory,
        "camera_rotation": camera_rotation,
        "movement_distance": movement_distance,
    }
    return save_experiment_log("gen3c", output_path, params, results, save_json)


def log_lyra_experiment(
    output_path: str,
    input_image: str,
    seed: int,
    guidance_scale: float,
    inference_steps: int,
    sdg_steps: int,
    resolution: int,
    mode: str,
    results: Dict[str, Any],
    save_json: bool = True,
) -> Dict[str, str]:
    """Log Lyra experiment."""
    params = {
        "input_image": Path(input_image).name if input_image else "unknown",
        "seed": seed,
        "guidance_scale": guidance_scale,
        "inference_steps": inference_steps,
        "sdg_steps": sdg_steps,
        "resolution": resolution,
        "mode": mode,
    }
    return save_experiment_log("lyra", output_path, params, results, save_json)


def log_trellis_experiment(
    output_path: str,
    input_image: str,
    seed: int,
    resolution: int,
    guidance_scale: float,
    output_format: str,
    results: Dict[str, Any],
    save_json: bool = True,
) -> Dict[str, str]:
    """Log Trellis experiment."""
    params = {
        "input_image": Path(input_image).name if input_image else "unknown",
        "seed": seed,
        "resolution": resolution,
        "guidance_scale": guidance_scale,
        "output_format": output_format,
    }
    return save_experiment_log("trellis", output_path, params, results, save_json)


def log_hunyuan_experiment(
    output_path: str,
    input_image: str,
    seed: int,
    guidance_scale: float,
    inference_steps: int,
    octree_depth: int,
    model_type: str,
    remove_bg: bool,
    results: Dict[str, Any],
    save_json: bool = True,
) -> Dict[str, str]:
    """Log Hunyuan experiment."""
    params = {
        "input_image": Path(input_image).name if input_image else "unknown",
        "seed": seed,
        "guidance_scale": guidance_scale,
        "inference_steps": inference_steps,
        "octree_depth": octree_depth,
        "model_type": model_type,
        "remove_bg": remove_bg,
    }
    return save_experiment_log("hunyuan", output_path, params, results, save_json)


def log_mesh_cleanup_experiment(
    output_path: str,
    input_file: str,
    target_triangles: int,
    pre_smooth: int,
    preserve_detail: bool,
    post_smooth: int,
    remove_components: bool,
    fix_normals: bool,
    fill_holes: bool,
    aggressive: bool,
    min_ratio: float,
    results: Dict[str, Any],
    save_json: bool = True,
) -> Dict[str, str]:
    """Log Mesh Cleanup experiment."""
    params = {
        "input_file": Path(input_file).name if input_file else "unknown",
        "target_triangles": target_triangles,
        "pre_smooth": pre_smooth,
        "preserve_detail": preserve_detail,
        "post_smooth": post_smooth,
        "remove_components": remove_components,
        "fix_normals": fix_normals,
        "fill_holes": fill_holes,
        "aggressive": aggressive,
        "min_ratio": min_ratio,
    }
    return save_experiment_log("mesh_cleanup", output_path, params, results, save_json)


# =============================================================================
# Filename generation helpers for each model
# =============================================================================

def sharp_param_filename(base_name: str, guidance: float, steps: int, ext: str = ".glb") -> str:
    """Generate SHARP param-encoded filename for PLY generation."""
    return generate_param_filename(base_name, "sharp", {"g": guidance, "s": steps}, ext)


def sharp_video_param_filename(
    base_name: str, 
    guidance: float, 
    steps: int, 
    lateral_offset: float,
    zoom_forward: float,
    ext: str = ".mp4"
) -> str:
    """Generate SHARP video param-encoded filename.
    
    Args:
        base_name: Base filename
        guidance: Guidance scale
        steps: Inference steps
        lateral_offset: Lateral camera movement (max_disparity)
        zoom_forward: Forward zoom movement (max_zoom)
        ext: File extension
    
    Returns:
        Encoded filename like: sharp_kitchen_g75_s50_lo008_zf015.mp4
    """
    # Format floats without decimal: 0.08 -> 008, 0.15 -> 015
    lo_str = f"{int(lateral_offset * 100):03d}"
    zf_str = f"{int(zoom_forward * 100):03d}"
    return generate_param_filename(
        base_name, "sharp", 
        {"g": guidance, "s": steps, "lo": lo_str, "zf": zf_str}, 
        ext
    )


def gen3c_param_filename(
    base_name: str, 
    frames: int, 
    trajectory: str, 
    movement_distance: float = 1.0,
    guidance: float = 1.0,
    seed: int = 42,
    ext: str = ".mp4"
) -> str:
    """Generate Gen3C param-encoded filename.
    
    Args:
        base_name: Base filename
        frames: Number of frames
        trajectory: Camera trajectory name
        movement_distance: Camera movement distance
        guidance: Guidance scale
        seed: Random seed
        ext: File extension
    
    Returns:
        Encoded filename like: gen3c_room_f121_tclo_md10_g10_s42.mp4
    """
    traj_short = trajectory[:3] if trajectory else "unk"
    # Format movement distance: 1.0 -> 10, 0.5 -> 05
    md_str = f"{int(movement_distance * 10):02d}"
    # Format guidance: 1.0 -> 10
    g_str = f"{int(guidance * 10):02d}"
    return generate_param_filename(
        base_name, "gen3c", 
        {"f": frames, "t": traj_short, "md": md_str, "g": g_str, "s": seed}, 
        ext
    )


def lyra_param_filename(base_name: str, guidance: float, sdg_steps: int, ext: str = ".ply") -> str:
    """Generate Lyra param-encoded filename."""
    return generate_param_filename(base_name, "lyra", {"g": guidance, "sdg": sdg_steps}, ext)


def trellis_param_filename(base_name: str, resolution: int, guidance: float, ext: str = ".glb") -> str:
    """Generate Trellis param-encoded filename."""
    return generate_param_filename(base_name, "trellis", {"r": resolution, "g": guidance}, ext)


def hunyuan_param_filename(base_name: str, guidance: float, octree: int, steps: int = 40, ext: str = ".glb") -> str:
    """Generate Hunyuan param-encoded filename."""
    return generate_param_filename(base_name, "hunyuan", {"g": guidance, "o": octree, "s": steps}, ext)


def cleanup_param_filename(base_name: str, target_tris: int, preserve_detail: bool, post_smooth: int, ext: str = ".glb") -> str:
    """Generate Cleanup param-encoded filename."""
    return generate_param_filename(base_name, "cleanup", {"t": target_tris, "pd": preserve_detail, "ps": post_smooth}, ext)


def ltx2_param_filename(
    base_name: str,
    model: str,
    duration: float,
    resolution: str,
    camera_motion: str,
    ext: str = ".mp4"
) -> str:
    """Generate LTX-2 param-encoded filename.
    
    Args:
        base_name: Base filename
        model: Model name (ltx-video-2-1 or ltx-video-2-1-fast)
        duration: Video duration in seconds
        resolution: Resolution string (e.g., "1080p", "720p")
        camera_motion: Camera motion type (underscores removed)
        ext: File extension
    
    Returns:
        Encoded filename like: ltx2_CBGB1_mpro_d6_r1080p_dollyout.mp4
    """
    # Shorten model name: ltx-video-2-1 -> pro, ltx-video-2-1-fast -> fast
    model_short = "fast" if "fast" in model.lower() else "pro"
    # Remove underscores from motion
    motion_clean = camera_motion.replace("_", "") if camera_motion else "static"
    # Format duration as integer
    dur_int = int(duration) if duration else 5
    # Clean resolution (keep just the number+p)
    res_clean = resolution.lower().replace(" ", "") if resolution else "720p"
    
    return f"ltx2_{base_name}_m{model_short}_d{dur_int}_{res_clean}_{motion_clean}{ext}"


def twodgs_param_filename(input_basenames: list, ext: str = ".glb") -> str:
    """Generate 2DGS param-encoded filename from input video basenames.
    
    Args:
        input_basenames: List of input video base names (without extension/motion suffix)
        ext: File extension
    
    Returns:
        Encoded filename like: 2dgs_CBGB1.glb or 2dgs_CBGB1_CBGB2.glb
    """
    if not input_basenames:
        return f"2dgs_output{ext}"
    
    # Deduplicate and join basenames
    unique_names = []
    for name in input_basenames:
        if name and name not in unique_names:
            unique_names.append(name)
    
    names_str = "_".join(unique_names[:3])  # Limit to 3 to avoid overly long names
    if len(unique_names) > 3:
        names_str += f"_plus{len(unique_names) - 3}"
    
    return f"2dgs_{names_str}{ext}"


def mesh_extract_param_filename(input_filename: str, ext: str = ".glb") -> str:
    """Generate Mesh Extraction param-encoded filename.
    
    Args:
        input_filename: Full input filename (with or without extension)
        ext: Output file extension
    
    Returns:
        Encoded filename like: lyra_bedroom_g75_sdg250_processed.glb
    """
    from pathlib import Path
    # Get the full stem (name without extension)
    stem = Path(input_filename).stem if input_filename else "mesh"
    return f"{stem}_processed{ext}"


if __name__ == "__main__":
    # Test the logger
    print(f"Log directory: {LOG_DIR}")
    ensure_log_dir()
    print(f"Log directory exists: {LOG_DIR.exists()}")
    
    # Test filename generation
    print("\nFilename examples:")
    print(f"  SHARP (PLY):   {sharp_param_filename('kitchen', 10.0, 50)}")
    print(f"  SHARP (video): {sharp_video_param_filename('kitchen', 7.5, 50, 0.08, 0.15)}")
    print(f"  Gen3C:         {gen3c_param_filename('room', 121, 'clockwise', 1.0, 1.0, 42)}")
    print(f"  Lyra:          {lyra_param_filename('bedroom', 9.0, 350)}")
    print(f"  Trellis:       {trellis_param_filename('chair', 1024, 7.5)}")
    print(f"  Hunyuan:       {hunyuan_param_filename('sofa', 10.0, 9)}")
    print(f"  Cleanup:       {cleanup_param_filename('mesh', 300000, True, 2)}")
    print(f"  LTX-2:         {ltx2_param_filename('CBGB1', 'ltx-video-2-1', 6.0, '1080p', 'dolly_out')}")
    print(f"  2DGS:          {twodgs_param_filename(['CBGB1', 'CBGB2'])}")
    print(f"  Mesh Extract:  {mesh_extract_param_filename('lyra_bedroom_g75.ply')}")

