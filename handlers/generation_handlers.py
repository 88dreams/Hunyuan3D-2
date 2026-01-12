"""
Generation handlers for the 3D Studio UI.

These handlers bridge UI inputs to generator functions, handling:
- Image preprocessing (scaling, temp files)
- Parameter encoding for filenames
- Experiment logging
- Cleanup of temporary files
"""

import os
from typing import Optional, Tuple, Union

from utils import clamp_scale_value, maybe_downscale_image

from generators import (
    run_hunyuan,
    run_hunyuan_runpod,
    run_gen3c_serverless,
    run_sharp_local,
    run_sharp_runpod,
    run_lyra_runpod,
    run_trellis_runpod,
    run_sugar_extraction,
    run_tsdf_extraction,
    GEN3C_DEFAULT_OUTPUT_DIR,
)


def handle_sharp_generation(
    image_path: Union[str, None],
    image_scale: Union[float, int],
    exec_mode: str,
    endpoint_id: str,
    api_key: str,
    render_video: bool,
    output_name: str,
    output_dir: str,
    trajectory_type: str = "rotate_forward",
    num_steps: int = 60,
    num_repeats: int = 1,
    max_disparity: float = 0.08,
    max_zoom: float = 0.15,
    lookat_mode: str = "point",
    log_params: bool = True,
    encode_params: bool = False,
) -> Tuple[Optional[str], str, str]:
    """Handle SHARP generation with optional video rendering.
    
    Args:
        image_path: Path to input image
        image_scale: Scale factor (0.25-1.0)
        exec_mode: "Local" or "RunPod Serverless"
        endpoint_id: RunPod endpoint ID
        api_key: RunPod API key
        render_video: Whether to render video
        output_name: Base name for output file
        output_dir: Directory for output
        trajectory_type: Video trajectory type
        num_steps: Number of video frames
        num_repeats: Number of trajectory repeats
        max_disparity: Maximum lateral camera offset
        max_zoom: Maximum forward camera movement
        lookat_mode: Camera focus mode ("point" or "ahead")
        log_params: Whether to log experiment parameters
        encode_params: Whether to encode params in filename
        
    Returns:
        Tuple of (output_path, logs, progress_status)
    """
    scale_value = clamp_scale_value(image_scale)
    scaled_path, temp_scaled = maybe_downscale_image(image_path, scale_value)
    effective_image_path = scaled_path or image_path
    
    # Generate parameter-encoded filename if requested
    effective_output_name = output_name
    if encode_params:
        try:
            from scripts.experiment_logger import sharp_param_filename
            encoded_name = sharp_param_filename(
                base_name=output_name,
                guidance=7.5,  # Default - not exposed in UI
                steps=50,  # Default - not exposed in UI
                ext=""  # No extension
            )
            effective_output_name = encoded_name.rstrip(".")
        except Exception as e:
            print(f"[SHARP] Warning: Could not encode params in filename: {e}")
    
    try:
        if "RunPod" in exec_mode:
            result = run_sharp_runpod(
                image_path=effective_image_path,
                endpoint_id=endpoint_id,
                api_key=api_key,
                output_name=effective_output_name,
                output_dir=output_dir,
                render_video=render_video,
                trajectory_type=trajectory_type,
                num_steps=int(num_steps),
                num_repeats=int(num_repeats),
                max_disparity=float(max_disparity),
                max_zoom=float(max_zoom),
                lookat_mode=lookat_mode,
            )
        else:
            result = run_sharp_local(
                image_path=effective_image_path,
                output_name=effective_output_name,
                output_dir=output_dir,
                render_video=render_video,
                trajectory_type=trajectory_type,
                num_steps=int(num_steps),
                num_repeats=int(num_repeats),
                max_disparity=float(max_disparity),
                max_zoom=float(max_zoom),
                lookat_mode=lookat_mode,
            )
        
        # Log experiment if enabled and successful
        output_path, logs, progress = result
        if log_params and output_path and os.path.exists(output_path):
            try:
                from scripts.experiment_logger import log_sharp_experiment
                log_files = log_sharp_experiment(
                    output_path=output_path,
                    input_image=effective_image_path or "",
                    seed=0,  # SHARP doesn't expose seed in current UI
                    guidance_scale=7.5,  # Default
                    inference_steps=50,  # Default
                    output_format=os.path.splitext(output_path)[1],
                    results={"success": True, "exec_mode": exec_mode},
                    save_json=True,
                )
                logs += f"\n📋 Logged to: {log_files.get('csv', 'N/A')}"
            except Exception as e:
                logs += f"\n⚠️ Logging failed: {e}"
        
        return result
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
    endpoint_id: str,
    api_key: str,
    guidance: float,
    frames: Union[str, int],
    trajectory: str,
    movement_distance: float,
    camera_rotation: str,
    foreground_mask: bool,
    video_name: str,
    seed: Optional[int],
    output_dir: str,
    log_params: bool = True,
    encode_params: bool = False,
) -> Tuple[Optional[str], str, str]:
    """Handle GEN3C generation.
    
    Args:
        image_path: Path to input image
        image_scale: Scale factor (0.25-1.0)
        exec_mode: Execution mode (currently only serverless)
        endpoint_id: RunPod endpoint ID
        api_key: RunPod API key
        guidance: Guidance scale
        frames: Number of output frames
        trajectory: Camera trajectory type
        movement_distance: Camera movement distance
        camera_rotation: Camera rotation angle
        foreground_mask: Whether to use foreground masking
        video_name: Output video filename
        seed: Random seed
        output_dir: Output directory
        log_params: Whether to log experiment parameters
        encode_params: Whether to encode params in filename
        
    Returns:
        Tuple of (output_path, logs, progress_status)
    """
    scale_value = clamp_scale_value(image_scale)
    scaled_path, temp_scaled = maybe_downscale_image(image_path, scale_value)
    effective_image_path = scaled_path or image_path
    resolved_output_dir = output_dir.strip() if output_dir else GEN3C_DEFAULT_OUTPUT_DIR
    
    # Generate parameter-encoded filename if requested
    effective_video_name = video_name
    if encode_params:
        try:
            from scripts.experiment_logger import gen3c_param_filename
            encoded_name = gen3c_param_filename(
                base_name=video_name,
                frames=int(frames) if frames else 121,
                trajectory=trajectory,
                ext=""  # No extension
            )
            effective_video_name = encoded_name.rstrip(".")
        except Exception as e:
            print(f"[Gen3C] Warning: Could not encode params in filename: {e}")
    
    try:
        result = run_gen3c_serverless(
            image_path=effective_image_path,
            endpoint_id=endpoint_id,
            api_key=api_key,
            guidance=guidance,
            frames=frames,
            trajectory=trajectory,
            movement_distance=movement_distance,
            camera_rotation=camera_rotation,
            foreground_masking=foreground_mask,
            video_name=effective_video_name,
            seed=seed,
            output_dir=resolved_output_dir,
        )
        
        # Log experiment if enabled and successful
        output_path, logs, progress = result
        if log_params and output_path and os.path.exists(output_path):
            try:
                from scripts.experiment_logger import log_gen3c_experiment
                # Parse camera rotation
                try:
                    cam_rot = float(camera_rotation) if camera_rotation else 0.0
                except (ValueError, TypeError):
                    cam_rot = 0.0
                
                log_files = log_gen3c_experiment(
                    output_path=output_path,
                    input_image=effective_image_path or "",
                    seed=int(seed) if seed else 0,
                    num_frames=int(frames) if frames else 121,
                    guidance_scale=guidance,
                    trajectory=trajectory,
                    camera_rotation=cam_rot,
                    movement_distance=movement_distance,
                    results={"success": True, "foreground_mask": foreground_mask},
                    save_json=True,
                )
                logs += f"\n📋 Logged to: {log_files.get('csv', 'N/A')}"
            except Exception as e:
                logs += f"\n⚠️ Logging failed: {e}"
        
        return output_path, logs, progress
    finally:
        if temp_scaled and os.path.exists(temp_scaled):
            try:
                os.remove(temp_scaled)
            except Exception:
                pass


def handle_lyra_generation(
    image_path: Union[str, None],
    video_path: Union[str, None],
    image_scale: Union[float, int],
    endpoint_id: str,
    api_key: str,
    generation_mode: str,
    num_views: int,
    camera_motion: float,
    multi_trajectory: bool,
    foreground_masking: bool,
    num_gaussians: int,
    seed: Optional[int],
    output_name: str,
    output_dir: str,
    output_ply: bool,
    output_video: bool,
    log_params: bool = True,
    encode_params: bool = False,
) -> Tuple[Optional[str], str, str, str]:
    """Handle Lyra generation.
    
    Args:
        image_path: Path to input image (for 3DGS mode)
        video_path: Path to input video (for 4DGS mode)
        image_scale: Scale factor (0.25-1.0)
        endpoint_id: RunPod endpoint ID
        api_key: RunPod API key
        generation_mode: "Static (3DGS)" or "Dynamic (4DGS)"
        num_views: Number of views to generate
        camera_motion: Camera motion intensity
        multi_trajectory: Whether to use multiple trajectories
        foreground_masking: Whether to use foreground masking
        num_gaussians: Number of Gaussians
        seed: Random seed
        output_name: Output filename
        output_dir: Output directory
        output_ply: Whether to output PLY
        output_video: Whether to output video
        log_params: Whether to log experiment parameters
        encode_params: Whether to encode params in filename
        
    Returns:
        Tuple of (output_path, logs, progress_status, ply_path)
    """
    scale_value = clamp_scale_value(image_scale)
    is_static = "Static" in generation_mode
    
    if is_static:
        scaled_path, temp_scaled = maybe_downscale_image(image_path, scale_value)
        effective_image_path = scaled_path or image_path
        effective_video_path = None
    else:
        effective_image_path = None
        effective_video_path = video_path
        temp_scaled = None
    
    # Generate parameter-encoded filename if requested
    effective_output_name = output_name
    if encode_params:
        try:
            from scripts.experiment_logger import lyra_param_filename
            encoded_name = lyra_param_filename(
                base_name=output_name,
                guidance=7.5,  # Default - not exposed in UI
                sdg_steps=250,  # Default - not exposed in UI
                ext=""  # No extension
            )
            effective_output_name = encoded_name.rstrip(".")
        except Exception as e:
            print(f"[Lyra] Warning: Could not encode params in filename: {e}")
    
    try:
        result = run_lyra_runpod(
            image_path=effective_image_path,
            video_path=effective_video_path,
            endpoint_id=endpoint_id,
            api_key=api_key,
            generation_mode=generation_mode,
            num_views=num_views,
            camera_motion=camera_motion,
            multi_trajectory=multi_trajectory,
            foreground_masking=foreground_masking,
            num_gaussians=num_gaussians,
            seed=int(seed) if seed else None,
            output_name=effective_output_name,
            output_dir=output_dir,
            output_ply=output_ply,
            output_video=output_video,
        )
        
        output_path, logs, progress = result
        ply_path = output_path if output_path and output_path.endswith(".ply") else ""
        
        # Log experiment if enabled and successful
        if log_params and output_path and os.path.exists(output_path):
            try:
                from scripts.experiment_logger import log_lyra_experiment
                log_files = log_lyra_experiment(
                    output_path=output_path,
                    input_image=effective_image_path or effective_video_path or "",
                    seed=int(seed) if seed else 0,
                    guidance_scale=7.5,  # Default - Lyra doesn't expose this
                    inference_steps=50,  # Default
                    sdg_steps=250,  # Default
                    resolution=512,  # Default
                    mode="3dgs" if is_static else "4dgs",
                    results={
                        "success": True,
                        "generation_mode": generation_mode,
                        "num_views": num_views,
                        "num_gaussians": num_gaussians,
                    },
                    save_json=True,
                )
                logs += f"\n📋 Logged to: {log_files.get('csv', 'N/A')}"
            except Exception as e:
                logs += f"\n⚠️ Logging failed: {e}"
        
        return output_path, logs, progress, ply_path
    finally:
        if temp_scaled and os.path.exists(temp_scaled):
            try:
                os.remove(temp_scaled)
            except Exception:
                pass


def handle_trellis_generation(
    image_path: Union[str, None],
    image_scale: Union[float, int],
    endpoint_id: str,
    api_key: str,
    resolution: str,
    guidance_scale: float,
    seed: Optional[int],
    output_name: str,
    output_dir: str,
    output_format: str,
    log_params: bool = True,
    encode_params: bool = False,
) -> Tuple[Optional[str], str, str]:
    """Handle TRELLIS.2 generation.
    
    Args:
        image_path: Path to input image
        image_scale: Scale factor (0.25-1.0)
        endpoint_id: RunPod endpoint ID
        api_key: RunPod API key
        resolution: Output resolution
        guidance_scale: Guidance scale
        seed: Random seed
        output_name: Output filename
        output_dir: Output directory
        output_format: Output format (glb, obj, ply)
        log_params: Whether to log experiment parameters
        encode_params: Whether to encode params in filename
        
    Returns:
        Tuple of (output_path, logs, progress_status)
    """
    scale_value = clamp_scale_value(image_scale)
    scaled_path, temp_scaled = maybe_downscale_image(image_path, scale_value)
    effective_image_path = scaled_path or image_path
    
    # Generate parameter-encoded filename if requested
    effective_output_name = output_name
    if encode_params:
        try:
            from scripts.experiment_logger import trellis_param_filename
            # Parse resolution
            try:
                res_int = int(resolution) if resolution else 1024
            except (ValueError, TypeError):
                res_int = 1024
            
            encoded_name = trellis_param_filename(
                base_name=output_name,
                resolution=res_int,
                guidance=guidance_scale,
                ext=""  # No extension
            )
            effective_output_name = encoded_name.rstrip(".")
        except Exception as e:
            print(f"[Trellis] Warning: Could not encode params in filename: {e}")
    
    try:
        result = run_trellis_runpod(
            image_path=effective_image_path,
            endpoint_id=endpoint_id,
            api_key=api_key,
            resolution=resolution,
            guidance_scale=guidance_scale,
            seed=int(seed) if seed else None,
            output_name=effective_output_name,
            output_dir=output_dir,
            output_format=output_format,
        )
        
        # Log experiment if enabled and successful
        output_path, logs, progress = result
        if log_params and output_path and os.path.exists(output_path):
            try:
                from scripts.experiment_logger import log_trellis_experiment
                # Parse resolution
                try:
                    res_int = int(resolution) if resolution else 1024
                except (ValueError, TypeError):
                    res_int = 1024
                
                log_files = log_trellis_experiment(
                    output_path=output_path,
                    input_image=effective_image_path or "",
                    seed=int(seed) if seed else 0,
                    resolution=res_int,
                    guidance_scale=guidance_scale,
                    output_format=output_format,
                    results={"success": True},
                    save_json=True,
                )
                logs += f"\n📋 Logged to: {log_files.get('csv', 'N/A')}"
            except Exception as e:
                logs += f"\n⚠️ Logging failed: {e}"
        
        return output_path, logs, progress
    finally:
        if temp_scaled and os.path.exists(temp_scaled):
            try:
                os.remove(temp_scaled)
            except Exception:
                pass


def handle_hunyuan_generation(
    image_path: Union[str, None],
    image_scale: Union[float, int],
    exec_mode: str,
    endpoint_id: str,
    api_key: str,
    guidance_scale: float,
    steps: int,
    seed: Optional[int],
    octree_resolution: int,
    model_choice: str,
    use_fp16: bool,
    attention_slicing: bool,
    cpu_offload: bool,
    remove_background: bool,
    output_name: str,
    save_location: str,
    log_params: bool = True,
    encode_params: bool = False,
) -> Tuple[Optional[str], str, str]:
    """Handle Hunyuan3D generation.
    
    Args:
        image_path: Path to input image
        image_scale: Scale factor (0.25-1.0)
        exec_mode: "Local" or "RunPod Serverless"
        endpoint_id: RunPod endpoint ID
        api_key: RunPod API key
        guidance_scale: Guidance scale
        steps: Number of inference steps
        seed: Random seed
        octree_resolution: Octree resolution for mesh detail
        model_choice: Model variant ("mini" or "full")
        use_fp16: Whether to use FP16 precision
        attention_slicing: Whether to use attention slicing
        cpu_offload: Whether to use CPU offload
        remove_background: Whether to remove background
        output_name: Output filename
        save_location: Output directory
        log_params: Whether to log experiment parameters
        encode_params: Whether to encode params in filename
        
    Returns:
        Tuple of (output_path, logs, progress_status)
    """
    scale_value = clamp_scale_value(image_scale)
    scaled_path, temp_scaled = maybe_downscale_image(image_path, scale_value)
    effective_image_path = scaled_path or image_path

    # Generate parameter-encoded filename if requested
    effective_output_name = output_name
    if encode_params:
        try:
            from scripts.experiment_logger import hunyuan_param_filename
            # Generate encoded filename (without extension - the generator adds .glb)
            encoded_name = hunyuan_param_filename(
                base_name=output_name,
                guidance=guidance_scale,
                octree=int(octree_resolution) if octree_resolution else 380,
                steps=int(steps) if steps else 40,
                ext=""  # No extension, generator adds it
            )
            # Remove any trailing dots from the name
            effective_output_name = encoded_name.rstrip(".")
        except Exception as e:
            print(f"[Hunyuan] Warning: Could not encode params in filename: {e}")
            effective_output_name = output_name

    try:
        if exec_mode == "RunPod Serverless":
            result = run_hunyuan_runpod(
                image_path=effective_image_path,
                endpoint_id=endpoint_id,
                api_key=api_key,
                model_choice=model_choice,
                guidance_scale=guidance_scale,
                steps=steps,
                octree_resolution=int(octree_resolution) if octree_resolution else 380,
                seed=int(seed) if seed else None,
                remove_background=remove_background,
                output_name=effective_output_name,
                output_dir=save_location,
            )
        else:
            result = run_hunyuan(
                image_path=effective_image_path,
                guidance_scale=guidance_scale,
                steps=steps,
                seed=seed,
                model_choice=model_choice,
                use_fp16=use_fp16,
                attention_slicing=attention_slicing,
                cpu_offload=cpu_offload,
                remove_background=remove_background,
                output_name=effective_output_name,
                save_location=save_location,
            )
        
        # Log experiment if enabled and successful
        output_path, logs, progress = result
        if log_params and output_path and os.path.exists(output_path):
            try:
                from scripts.experiment_logger import log_hunyuan_experiment
                log_files = log_hunyuan_experiment(
                    output_path=output_path,
                    input_image=effective_image_path or "",
                    seed=int(seed) if seed else 0,
                    guidance_scale=guidance_scale,
                    inference_steps=steps,
                    octree_depth=int(octree_resolution) if octree_resolution else 380,
                    model_type=model_choice,
                    remove_bg=remove_background,
                    results={"success": True, "exec_mode": exec_mode},
                    save_json=True,
                )
                logs += f"\n📋 Logged to: {log_files.get('csv', 'N/A')}"
            except Exception as e:
                logs += f"\n⚠️ Logging failed: {e}"
        
        return output_path, logs, progress
    finally:
        if temp_scaled and os.path.exists(temp_scaled):
            try:
                os.remove(temp_scaled)
            except Exception:
                pass


def handle_mesh_extraction(
    input_ply: str,
    input_format: str,
    method: str,
    regularization: str,
    quality_preset: str,
    poisson_depth: int,
    decimate_faces: int,
    export_texture: bool,
    texture_resolution: str,
    refinement_time: str,
    tsdf_voxel_size: float,
    tsdf_num_views: int,
    output_name: str,
    output_format: str,
    output_dir: str,
    endpoint_id: str,
    api_key: str,
):
    """Handle mesh extraction request.
    
    Args:
        input_ply: Path to input PLY file
        input_format: Input format type
        method: Extraction method (SuGaR/Poisson or TSDF)
        regularization: Regularization type
        quality_preset: Quality preset
        poisson_depth: Poisson reconstruction depth
        decimate_faces: Target face count for decimation
        export_texture: Whether to export texture
        texture_resolution: Texture resolution
        refinement_time: Refinement time setting
        tsdf_voxel_size: TSDF voxel size
        tsdf_num_views: TSDF number of views
        output_name: Output filename
        output_format: Output format (glb, obj)
        output_dir: Output directory
        endpoint_id: RunPod endpoint ID
        api_key: RunPod API key
        
    Returns:
        Tuple of (output_path, logs, status)
    """
    if not input_ply or not input_ply.strip():
        return "", "Error: No input PLY file specified", "❌ Missing input"
    
    # Clean up dropdown value - remove format info in parentheses if present
    if " (" in input_ply:
        input_ply = input_ply.rsplit(" (", 1)[0]
    
    if "SuGaR" in method or "Poisson" in method:
        result = run_sugar_extraction(
            input_ply=input_ply,
            input_format=input_format,
            regularization=regularization,
            quality_preset=quality_preset,
            poisson_depth=poisson_depth,
            decimate_faces=decimate_faces,
            export_texture=export_texture,
            texture_resolution=int(texture_resolution),
            refinement_time=refinement_time,
            output_name=output_name,
            output_format=output_format,
            output_dir=output_dir,
            endpoint_id=endpoint_id,
            api_key=api_key,
        )
    else:
        result = run_tsdf_extraction(
            input_ply=input_ply,
            input_format=input_format,
            voxel_size=tsdf_voxel_size,
            num_views=tsdf_num_views,
            output_name=output_name,
            output_dir=output_dir,
            endpoint_id=endpoint_id,
            api_key=api_key,
        )
    
    output_path, logs, status = result
    return output_path or "", logs, status


def handle_mesh_analyze(
    file_input,
    path_input: str,
) -> Tuple[str, str]:
    """Analyze a mesh file and return statistics.
    
    Args:
        file_input: Uploaded file object
        path_input: Path string input
        
    Returns:
        Tuple of (analysis_text, status)
    """
    # Determine input path
    if file_input is not None:
        input_path = file_input.name if hasattr(file_input, 'name') else str(file_input)
    elif path_input and path_input.strip():
        input_path = path_input.strip()
    else:
        return "No input file specified", "❌ Missing input"
    
    if not os.path.exists(input_path):
        return f"File not found: {input_path}", "❌ File not found"
    
    try:
        # Import the cleanup script functions
        from scripts.cleanup_mesh import analyze_mesh, TRIMESH_AVAILABLE
        
        if not TRIMESH_AVAILABLE:
            return "trimesh not installed. Install with: pip install trimesh", "❌ Missing dependency"
        
        stats = analyze_mesh(input_path)
        
        # Format analysis results
        result_lines = [
            f"File: {os.path.basename(input_path)}",
            f"",
            f"Geometry:",
            f"  Vertices: {stats['vertices']:,}",
            f"  Triangles: {stats['triangles']:,}",
            f"  Components: {stats['components']}",
            f"",
            f"Size:",
            f"  X: {stats['size'][0]:.3f}",
            f"  Y: {stats['size'][1]:.3f}",
            f"  Z: {stats['size'][2]:.3f}",
            f"",
            f"Quality:",
            f"  Watertight: {'✅' if stats['is_watertight'] else '❌'}",
            f"  Consistent Winding: {'✅' if stats['is_winding_consistent'] else '❌'}",
        ]
        
        if stats.get('issues'):
            result_lines.append("")
            result_lines.append("Issues Found:")
            for issue in stats['issues']:
                result_lines.append(f"  ⚠️ {issue}")
        else:
            result_lines.append("")
            result_lines.append("✅ No issues detected")
        
        return "\n".join(result_lines), "✅ Analysis complete"
        
    except Exception as e:
        return f"Error analyzing mesh: {str(e)}", f"❌ {str(e)}"


def handle_mesh_cleanup(
    file_input,
    path_input: str,
    target_triangles: int,
    smooth_iterations: int,
    preserve_detail: bool,
    post_decimate_smooth: int,
    remove_components: bool,
    fix_normals: bool,
    fill_holes: bool,
    aggressive: bool,
    min_component_ratio: float,
    output_name: str,
    output_dir: str,
    output_format: str,
    log_params: bool,
    encode_params: bool,
) -> Tuple[str, str, str]:
    """Clean up a mesh file.
    
    Args:
        file_input: Uploaded file object
        path_input: Path string input
        target_triangles: Target triangle count
        smooth_iterations: Pre-decimation smoothing iterations
        preserve_detail: Whether to use quality decimation
        post_decimate_smooth: Post-decimation smoothing iterations
        remove_components: Whether to remove small components
        fix_normals: Whether to fix normals
        fill_holes: Whether to fill holes
        aggressive: Whether to use aggressive cleanup
        min_component_ratio: Minimum component ratio to keep
        output_name: Output filename
        output_dir: Output directory
        output_format: Output format
        log_params: Whether to log parameters
        encode_params: Whether to encode params in filename
        
    Returns:
        Tuple of (output_path, log_text, status)
    """
    # Determine input path
    if file_input is not None:
        input_path = file_input.name if hasattr(file_input, 'name') else str(file_input)
    elif path_input and path_input.strip():
        input_path = path_input.strip()
    else:
        return "", "No input file specified", "❌ Missing input"
    
    if not os.path.exists(input_path):
        return "", f"File not found: {input_path}", "❌ File not found"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Build output path
    ext = output_format.lower()
    output_path = os.path.join(output_dir, f"{output_name}.{ext}")
    
    try:
        # Import the cleanup script functions
        from scripts.cleanup_mesh import cleanup_mesh, TRIMESH_AVAILABLE
        
        if not TRIMESH_AVAILABLE:
            return "", "trimesh not installed. Install with: pip install trimesh", "❌ Missing dependency"
        
        stats = cleanup_mesh(
            input_path=input_path,
            output_path=output_path,
            target_triangles=int(target_triangles),
            remove_small_components=remove_components,
            min_component_ratio=min_component_ratio,
            fix_normals=fix_normals,
            fill_holes=fill_holes,
            smooth_iterations=int(smooth_iterations),
            aggressive=aggressive,
            preserve_detail=preserve_detail,
            post_decimate_smooth=int(post_decimate_smooth),
            log_params=log_params,
            encode_params_in_filename=encode_params,
        )
        
        # Get actual output path (may have been modified if encode_params)
        actual_output = stats.get('output_file', output_path)
        
        # Format log output
        log_lines = [
            f"Input: {stats['input_triangles']:,} triangles",
            f"Output: {stats['output_triangles']:,} triangles",
            f"Reduction: {100 * (1 - stats['output_triangles'] / stats['input_triangles']):.1f}%",
            f"",
            f"Operations performed:",
        ]
        for op in stats.get('operations', []):
            log_lines.append(f"  - {op}")
        
        log_lines.append("")
        log_lines.append(f"Components removed: {stats.get('components_removed', 0)}")
        log_lines.append(f"Watertight: {'✅' if stats.get('is_watertight') else '❌'}")
        log_lines.append(f"Consistent winding: {'✅' if stats.get('is_winding_consistent') else '❌'}")
        
        # Add parameter log info
        if log_params and 'param_log' in stats:
            log_lines.append("")
            log_lines.append(f"📋 Parameter log: {stats['param_log']}")
            log_lines.append(f"📊 Experiment CSV: {stats.get('csv_log', '/srv/searidge_share/outputs/logs/mesh_cleanup.csv')}")
        log_lines.append("")
        log_lines.append(f"Saved to: {actual_output}")
        
        return actual_output, "\n".join(log_lines), "✅ Cleanup complete!"
        
    except Exception as e:
        import traceback
        return "", f"Error: {str(e)}\n\n{traceback.format_exc()}", f"❌ {str(e)}"


def handle_2dgs_pipeline(
    video_source: str,
    video_path: str,
    video_upload,
    video_dropdown: str,
    gen3c_output_dir: str,
    iterations: int,
    mesh_quality: str,
    output_format: str,
    output_dir: str,
    endpoint_id: str,
    api_key: str,
    s3_bucket: str = "arkrunr",
    s3_region: str = "us-west-1",
    progress_callback=None,
) -> Tuple[Optional[str], dict, str]:
    """
    Handle 2DGS pipeline: Video → 3D Mesh.
    
    Args:
        video_source: "Upload Video", "Video Path", or "Use Gen3C Output"
        video_path: Path or URL to video (if video_source == "Video Path")
        video_upload: Uploaded video file (if video_source == "Upload Video")
        video_dropdown: Selected video from dropdown (if video_source == "Use Gen3C Output")
        gen3c_output_dir: Gen3C output directory
        iterations: 2DGS training iterations (1000-10000)
        mesh_quality: Mesh extraction quality (fast, balanced, high, ultra)
        output_format: Output format (glb, obj, ply)
        output_dir: Local directory to save output
        endpoint_id: RunPod endpoint ID
        api_key: RunPod API key
        s3_bucket: S3 bucket for file transfer
        s3_region: S3 region
        progress_callback: Optional callback for progress updates
        
    Returns:
        Tuple of (output_file_path, stats_dict, status_message)
    """
    import uuid
    from pathlib import Path
    
    logs = []
    
    # Validate API key
    if not api_key:
        return None, {}, "❌ API key required. Set in Endpoint Settings."
    
    # Determine video input
    video_url = None
    local_video_path = None
    
    if video_source == "Upload Video":
        if video_upload is None:
            return None, {}, "❌ No video uploaded"
        local_video_path = video_upload
        logs.append(f"Using uploaded video: {local_video_path}")
        
    elif video_source == "Video Path":
        if not video_path:
            return None, {}, "❌ No video path provided"
        
        # Check if it's a URL
        if video_path.startswith(("http://", "https://", "s3://")):
            video_url = video_path
            logs.append(f"Using video URL: {video_url[:60]}...")
        else:
            local_video_path = video_path
            if not Path(local_video_path).exists():
                return None, {}, f"❌ Video file not found: {local_video_path}"
            logs.append(f"Using local video: {local_video_path}")
            
    elif video_source == "Use Gen3C Output":
        if not video_dropdown:
            return None, {}, "❌ No video selected from Gen3C outputs"
        local_video_path = str(Path(gen3c_output_dir) / video_dropdown)
        if not Path(local_video_path).exists():
            return None, {}, f"❌ Gen3C video not found: {local_video_path}"
        logs.append(f"Using Gen3C video: {local_video_path}")
    
    # Upload to S3 if local file
    if local_video_path and not video_url:
        try:
            import boto3
            
            logs.append("Uploading video to S3...")
            
            # Generate unique S3 key
            unique_id = str(uuid.uuid4())[:8]
            filename = Path(local_video_path).name
            s3_key = f"MediaContent/2dgs-pipeline/inputs/{unique_id}_{filename}"
            
            s3 = boto3.client("s3", region_name=s3_region)
            s3.upload_file(str(local_video_path), s3_bucket, s3_key)
            
            # Generate presigned URL
            video_url = s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": s3_bucket, "Key": s3_key},
                ExpiresIn=3600
            )
            logs.append(f"✅ Uploaded to S3: s3://{s3_bucket}/{s3_key}")
            
        except Exception as e:
            return None, {}, f"❌ S3 upload failed: {e}"
    
    if not video_url:
        return None, {}, "❌ Could not resolve video URL"
    
    # Create 2DGS client and run pipeline
    try:
        from runpod.runpod_client import TwoDGSPipelineClient
        
        client = TwoDGSPipelineClient(
            endpoint_id=endpoint_id,
            api_key=api_key
        )
        
        logs.append(f"Starting 2DGS pipeline...")
        logs.append(f"Iterations: {iterations}, Quality: {mesh_quality}")
        logs.append(f"Output format: {output_format}")
        
        # Custom progress callback that updates both logs and UI
        def _progress_callback(status, elapsed):
            msg = f"[{elapsed:.0f}s] Status: {status}"
            logs.append(msg)
            if progress_callback:
                progress_callback(status, elapsed)
        
        result = client.generate_sync(
            video_url=video_url,
            output_dir=output_dir,
            iterations=iterations,
            mesh_quality=mesh_quality,
            output_format=output_format,
            s3_bucket=s3_bucket,
            s3_region=s3_region,
            poll_interval=15,
            max_wait=1200,  # 20 minutes
            progress_callback=_progress_callback,
        )
        
        if result.success:
            stats = {
                "job_id": result.job_id,
                "duration_seconds": round(result.duration_seconds, 1),
                "output_path": result.output_path,
            }
            
            logs.append("")
            logs.append(f"✅ Pipeline completed in {result.duration_seconds:.1f}s")
            logs.append(f"Output: {result.output_path}")
            
            return result.output_path, stats, "\n".join(logs)
        else:
            logs.append(f"❌ Pipeline failed: {result.error}")
            return None, {"error": result.error}, "\n".join(logs)
            
    except Exception as e:
        import traceback
        logs.append(f"❌ Error: {str(e)}")
        logs.append(traceback.format_exc())
        return None, {"error": str(e)}, "\n".join(logs)


def list_gen3c_videos(output_dir: str) -> list:
    """List MP4 videos in the Gen3C output directory."""
    from pathlib import Path
    
    output_path = Path(output_dir)
    if not output_path.exists():
        return []
    
    videos = sorted(output_path.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [v.name for v in videos[:20]]  # Return 20 most recent


def get_video_info(video_path: str) -> str:
    """Get basic info about a video file."""
    from pathlib import Path
    import subprocess
    
    if not video_path:
        return "No video selected"
    
    path = Path(video_path)
    if not path.exists():
        return f"File not found: {video_path}"
    
    size_mb = path.stat().st_size / 1024 / 1024
    info = f"File: {path.name}\nSize: {size_mb:.1f} MB"
    
    # Try to get duration with ffprobe
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            duration = float(result.stdout.strip())
            info += f"\nDuration: {duration:.1f}s"
    except:
        pass
    
    return info
