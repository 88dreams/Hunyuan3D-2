#!/usr/bin/env python3
"""
Lyra Inference Script for RunPod Serverless

Lyra is a 2-step pipeline:
1. SDG (Synthetic Data Generation) - Uses GEN3C to generate multi-view video latents
2. 3DGS Reconstruction - Uses Lyra decoder to reconstruct 3D Gaussian Splats

This script wraps both steps for serverless execution.

Usage:
    python lyra_inference.py \
        --input_image /path/to/image.png \
        --output_dir /path/to/output \
        --output_name my_output \
        --checkpoint_dir /path/to/checkpoints \
        --mode static
"""

import os
import sys
import argparse
import subprocess
import tempfile
import shutil
import logging
import yaml
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Environment setup
os.environ.setdefault("CUDA_HOME", os.environ.get("CONDA_PREFIX", "/root/miniforge3/envs/cosmos-predict1"))


def run_command(cmd: list, cwd: str = None, env: dict = None) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    
    # Enable PyTorch error file for better debugging of distributed failures
    error_dir = tempfile.mkdtemp(prefix="torch_errors_")
    full_env["TORCHELASTIC_ERROR_FILE"] = f"{error_dir}/error.json"
    full_env["TORCH_SHOW_CPP_STACKTRACES"] = "1"
    
    logger.info(f"Running: {' '.join(cmd)}")
    logger.info(f"Error file location: {error_dir}/error.json")
    
    result = subprocess.run(
        cmd,
        cwd=cwd,
        env=full_env,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        logger.error(f"Command failed with code {result.returncode}")
        logger.error(f"stdout: {result.stdout[-2000:] if result.stdout else 'None'}")
        logger.error(f"stderr: {result.stderr[-2000:] if result.stderr else 'None'}")
        
        # Check for PyTorch error file
        error_file = f"{error_dir}/error.json"
        if os.path.exists(error_file):
            try:
                with open(error_file, 'r') as f:
                    logger.error(f"PyTorch error file contents: {f.read()}")
            except Exception as e:
                logger.error(f"Could not read error file: {e}")
        
        # Also check for any .txt error files in the error dir
        import glob
        for err_file in glob.glob(f"{error_dir}/*.txt"):
            try:
                with open(err_file, 'r') as f:
                    logger.error(f"Error file {err_file}: {f.read()}")
            except Exception as e:
                logger.error(f"Could not read {err_file}: {e}")
    
    return result


def step1_sdg_static(
    input_image: str,
    output_latents_dir: str,
    checkpoint_dir: str,
    movement_factor: float = 1.0,
    foreground_masking: bool = True,
    multi_trajectory: bool = True,
) -> bool:
    """
    Step 1: Generate multi-view video latents from input image using GEN3C.
    
    This runs the SDG (Synthetic Data Generation) phase.
    """
    logger.info("Step 1: Running SDG (multi-view video generation)...")
    
    lyra_dir = "/workspace/lyra"
    pythonpath = f"{lyra_dir}:{os.environ.get('PYTHONPATH', '')}"
    
    # Verify input image exists and log its properties
    if not os.path.exists(input_image):
        logger.error(f"Input image does not exist: {input_image}")
        return False
    
    logger.info(f"Input image: {input_image}")
    logger.info(f"Input image size: {os.path.getsize(input_image)} bytes")
    
    # Verify checkpoint directory
    logger.info(f"Checkpoint dir: {checkpoint_dir}")
    gen3c_ckpt = f"{checkpoint_dir}/Gen3C-Cosmos-7B"
    logger.info(f"GEN3C checkpoint exists: {os.path.exists(gen3c_ckpt)}")
    if os.path.exists(gen3c_ckpt):
        try:
            logger.info(f"GEN3C checkpoint contents: {os.listdir(gen3c_ckpt)[:10]}")
        except Exception as e:
            logger.warning(f"Could not list checkpoint contents: {e}")
    
    cmd = [
        "torchrun", "--nproc_per_node=1",
        f"{lyra_dir}/cosmos_predict1/diffusion/inference/gen3c_single_image_sdg.py",
        "--checkpoint_dir", checkpoint_dir,
        "--num_gpus", "1",
        "--input_image_path", input_image,
        "--video_save_folder", output_latents_dir,
        "--total_movement_distance_factor", str(movement_factor),
    ]
    
    if foreground_masking:
        cmd.append("--foreground_masking")
    if multi_trajectory:
        cmd.append("--multi_trajectory")
    
    env = {
        "CUDA_HOME": os.environ.get("CUDA_HOME", os.environ.get("CONDA_PREFIX", "")),
        "PYTHONPATH": pythonpath,
    }
    
    logger.info(f"SDG command: {' '.join(cmd)}")
    logger.info(f"SDG working directory: {lyra_dir}")
    logger.info(f"Output latents dir: {output_latents_dir}")
    
    result = run_command(cmd, cwd=lyra_dir, env=env)
    
    if result.returncode != 0:
        logger.error("SDG step failed")
        # Log full stdout/stderr for debugging
        if result.stdout:
            logger.error(f"Full stdout:\n{result.stdout}")
        if result.stderr:
            logger.error(f"Full stderr:\n{result.stderr}")
        return False
    
    # Log success and check output
    logger.info("SDG step completed successfully")
    if os.path.exists(output_latents_dir):
        try:
            logger.info(f"Output latents dir contents: {os.listdir(output_latents_dir)}")
        except Exception as e:
            logger.warning(f"Could not list output dir: {e}")
    
    return True


def step1_sdg_dynamic(
    input_video: str,
    output_latents_dir: str,
    checkpoint_dir: str,
    foreground_masking: bool = True,
    multi_trajectory: bool = True,
) -> bool:
    """
    Step 1: Generate multi-view video latents from input video using GEN3C.
    
    This runs the SDG (Synthetic Data Generation) phase for dynamic scenes.
    """
    logger.info("Step 1: Running SDG for dynamic scene...")
    
    lyra_dir = "/workspace/lyra"
    pythonpath = f"{lyra_dir}:{os.environ.get('PYTHONPATH', '')}"
    
    cmd = [
        "torchrun", "--nproc_per_node=1",
        f"{lyra_dir}/cosmos_predict1/diffusion/inference/gen3c_dynamic_sdg.py",
        "--checkpoint_dir", checkpoint_dir,
        "--vipe_path", input_video,
        "--video_save_folder", output_latents_dir,
        "--disable_prompt_upsampler",
        "--num_gpus", "1",
    ]
    
    if foreground_masking:
        cmd.append("--foreground_masking")
    if multi_trajectory:
        cmd.append("--multi_trajectory")
    
    env = {
        "CUDA_HOME": os.environ.get("CUDA_HOME", os.environ.get("CONDA_PREFIX", "")),
        "PYTHONPATH": pythonpath,
    }
    
    result = run_command(cmd, cwd=lyra_dir, env=env)
    
    if result.returncode != 0:
        logger.error("SDG dynamic step failed")
        return False
    
    logger.info("SDG dynamic step completed successfully")
    return True


def create_lyra_config(
    latents_dir: str,
    output_dir: str,
    output_name: str,
    mode: str = "static",
    checkpoint_dir: str = "/workspace/checkpoints",
) -> str:
    """
    Create a temporary Lyra config file for the 3DGS reconstruction step.
    
    Returns path to the created config file.
    """
    lyra_dir = "/workspace/lyra"
    
    # Base config template
    if mode == "static":
        base_config = f"{lyra_dir}/configs/demo/lyra_static.yaml"
    else:
        base_config = f"{lyra_dir}/configs/demo/lyra_dynamic.yaml"
    
    # Load base config
    if os.path.exists(base_config):
        with open(base_config, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded base config from: {base_config}")
    else:
        # Create minimal config if base doesn't exist
        logger.warning(f"Base config not found at {base_config}, creating minimal config")
        config = {
            "ckpt_path": f"{checkpoint_dir}/lyra/lyra_{mode}.pt",
            "out_dir_inference": output_dir,
        }
    
    # Override paths - ensure we use our checkpoint location
    # The Lyra repo's config uses 'ckpt_path' key with relative path 'checkpoints/Lyra/lyra_static.pt'
    # We need to override with absolute path to our network volume checkpoints
    ckpt_filename = "lyra_static.pt" if mode == "static" else "lyra_dynamic.pt"
    config["ckpt_path"] = f"{checkpoint_dir}/lyra/{ckpt_filename}"
    
    # Override output directory
    config["out_dir_inference"] = output_dir
    
    # CRITICAL: Enable PLY output - disabled by default in Lyra's inference config!
    config["save_gaussians"] = True
    config["save_gaussians_orig"] = True  # Also save in original format for compatibility
    
    # Log the checkpoint path for debugging
    logger.info(f"Lyra checkpoint path set to: {config['ckpt_path']}")
    
    # Write temporary config
    temp_config = tempfile.NamedTemporaryFile(
        mode='w',
        suffix='.yaml',
        delete=False
    )
    yaml.dump(config, temp_config)
    temp_config.close()
    
    logger.info(f"Created temporary config: {temp_config.name}")
    return temp_config.name


def setup_lyra_symlinks(checkpoint_dir: str = "/workspace/checkpoints"):
    """
    Create all symlinks needed for Lyra's hardcoded relative paths.
    
    Lyra has several hardcoded relative paths in its configs:
    1. vae_path: ./checkpoints/cosmos_predict1/Cosmos-Tokenize1-CV8x8x8-720p
    2. ckpt_path: checkpoints/Lyra/lyra_static.pt (handled via CLI override)
    3. dataset root_path: assets/demo/static/diffusion_output (handled separately)
    
    This function creates symlinks from the expected relative paths to our actual locations.
    """
    lyra_dir = "/workspace/lyra"
    
    # 1. Create ./checkpoints/cosmos_predict1/Cosmos-Tokenize1-CV8x8x8-720p symlink
    # The config uses vae_path: ./checkpoints/cosmos_predict1/Cosmos-Tokenize1-CV8x8x8-720p
    # which resolves to /workspace/lyra/checkpoints/cosmos_predict1/Cosmos-Tokenize1-CV8x8x8-720p
    cosmos_tokenizer_src = f"{checkpoint_dir}/Gen3C-Cosmos-7B/Cosmos-Tokenize1-CV8x8x8-720p"
    cosmos_tokenizer_dst = f"{lyra_dir}/checkpoints/cosmos_predict1/Cosmos-Tokenize1-CV8x8x8-720p"
    
    if os.path.exists(cosmos_tokenizer_src):
        os.makedirs(os.path.dirname(cosmos_tokenizer_dst), exist_ok=True)
        if os.path.exists(cosmos_tokenizer_dst) or os.path.islink(cosmos_tokenizer_dst):
            if os.path.islink(cosmos_tokenizer_dst):
                os.unlink(cosmos_tokenizer_dst)
            elif os.path.isdir(cosmos_tokenizer_dst):
                shutil.rmtree(cosmos_tokenizer_dst)
            else:
                os.remove(cosmos_tokenizer_dst)
        os.symlink(cosmos_tokenizer_src, cosmos_tokenizer_dst)
        logger.info(f"Created symlink: {cosmos_tokenizer_dst} -> {cosmos_tokenizer_src}")
    else:
        logger.warning(f"Cosmos tokenizer not found at {cosmos_tokenizer_src}")
    
    # 2. Create checkpoints/Lyra symlink (capital L) for any remaining hardcoded refs
    lyra_ckpt_src = f"{checkpoint_dir}/lyra"
    lyra_ckpt_dst = f"{lyra_dir}/checkpoints/Lyra"
    
    if os.path.exists(lyra_ckpt_src):
        os.makedirs(os.path.dirname(lyra_ckpt_dst), exist_ok=True)
        if os.path.exists(lyra_ckpt_dst) or os.path.islink(lyra_ckpt_dst):
            if os.path.islink(lyra_ckpt_dst):
                os.unlink(lyra_ckpt_dst)
            elif os.path.isdir(lyra_ckpt_dst):
                shutil.rmtree(lyra_ckpt_dst)
            else:
                os.remove(lyra_ckpt_dst)
        os.symlink(lyra_ckpt_src, lyra_ckpt_dst)
        logger.info(f"Created symlink: {lyra_ckpt_dst} -> {lyra_ckpt_src}")
    else:
        logger.warning(f"Lyra checkpoints not found at {lyra_ckpt_src}")


def step2_3dgs_reconstruction(
    config_path: str,
    checkpoint_dir: str = "/workspace/checkpoints",
    mode: str = "static",
) -> bool:
    """
    Step 2: Reconstruct 3D Gaussian Splats from multi-view video latents.
    
    Uses accelerate to launch the Lyra sample.py script.
    """
    logger.info("Step 2: Running 3DGS reconstruction...")
    
    lyra_dir = "/workspace/lyra"
    pythonpath = f"{lyra_dir}:{os.environ.get('PYTHONPATH', '')}"
    
    # Setup all required symlinks for Lyra's hardcoded paths
    setup_lyra_symlinks(checkpoint_dir)
    
    # Build absolute checkpoint paths
    ckpt_filename = "lyra_static.pt" if mode == "static" else "lyra_dynamic.pt"
    ckpt_path = f"{checkpoint_dir}/lyra/{ckpt_filename}"
    vae_path = f"{checkpoint_dir}/Gen3C-Cosmos-7B/Cosmos-Tokenize1-CV8x8x8-720p"
    
    logger.info(f"Using Lyra checkpoint: {ckpt_path}")
    logger.info(f"Lyra checkpoint exists: {os.path.exists(ckpt_path)}")
    logger.info(f"Using VAE path: {vae_path}")
    logger.info(f"VAE path exists: {os.path.exists(vae_path)}")
    
    # Check for mean_std.pt specifically
    mean_std_path = os.path.join(vae_path, "mean_std.pt")
    logger.info(f"mean_std.pt exists: {os.path.exists(mean_std_path)}")
    
    # Pass paths as CLI arguments to override any config settings
    # OmegaConf CLI overrides take precedence over all config files
    cmd = [
        "accelerate", "launch",
        f"{lyra_dir}/sample.py",
        "--config", config_path,
        f"ckpt_path={ckpt_path}",        # CLI override for model checkpoint
        f"vae_path={vae_path}",          # CLI override for VAE/tokenizer path
        "save_gaussians=True",           # CRITICAL: Enable PLY output (disabled by default!)
        "save_gaussians_orig=True",      # Also save in original format
    ]
    
    env = {
        "CUDA_HOME": os.environ.get("CUDA_HOME", os.environ.get("CONDA_PREFIX", "")),
        "PYTHONPATH": pythonpath,
    }
    
    result = run_command(cmd, cwd=lyra_dir, env=env)
    
    if result.returncode != 0:
        logger.error("3DGS reconstruction step failed")
        return False
    
    logger.info("3DGS reconstruction completed successfully")
    return True


def find_output_ply(output_dir: str, output_name: str, mode: str = "static") -> str:
    """Find the generated PLY file in the output directory.
    
    Lyra saves PLY files to:
    - {outdir}/gaussians/gaussians_{idx}.ply (when save_gaussians=True)
    - {outdir}/gaussians_orig/gaussians_{idx}.ply (when save_gaussians_orig=True)
    
    The outdir can include subdirectories based on view indices, e.g.:
    - outputs/demo/lyra_static/static_view_indices_fixed_5_0_1_2_3_4/gaussians/
    """
    lyra_dir = "/workspace/lyra"
    
    # Lyra's default output locations based on config
    lyra_output_dirs = [
        output_dir,  # Check our specified output dir first
        os.path.join(lyra_dir, f"outputs/demo/lyra_{mode}"),
        os.path.join(lyra_dir, "outputs/demo/lyra_static"),
        os.path.join(lyra_dir, "outputs/demo/lyra_dynamic"),
        os.path.join(lyra_dir, "outputs"),
    ]
    
    # Lyra-specific PLY patterns (in order of preference)
    # Lyra saves to gaussians/ or gaussians_orig/ subdirectories with gaussians_*.ply names
    lyra_patterns = [
        "gaussians/gaussians_0.ply",
        "gaussians_orig/gaussians_0.ply",
        "gaussians/gaussians_*.ply",
        "gaussians_orig/gaussians_*.ply",
    ]
    
    # Generic patterns as fallback
    generic_patterns = [
        f"{output_name}.ply",
        f"{output_name}_gaussians.ply",
        "point_cloud.ply",
        "gaussians.ply",
        "output.ply",
    ]
    
    # First, do a deep recursive search for any .ply file
    # This handles the case where Lyra creates nested subdirs like static_view_indices_fixed_*/
    for search_dir in lyra_output_dirs:
        if os.path.exists(search_dir):
            logger.info(f"Searching for PLY in: {search_dir}")
            for root, dirs, files in os.walk(search_dir):
                for f in files:
                    if f.endswith('.ply'):
                        ply_path = os.path.join(root, f)
                        logger.info(f"Found PLY at: {ply_path}")
                        return ply_path
    
    # If recursive search didn't find anything, log detailed info for debugging
    logger.warning(f"No PLY found. Searched directories: {lyra_output_dirs}")
    for search_dir in lyra_output_dirs:
        if os.path.exists(search_dir):
            # Show full directory tree for debugging
            for root, dirs, files in os.walk(search_dir):
                if files:
                    logger.warning(f"  {root}: {files[:10]}...")
    
    return None


def run_lyra_inference(
    input_image: str = None,
    input_video: str = None,
    output_dir: str = "/tmp/lyra_output",
    output_name: str = "lyra_output",
    checkpoint_dir: str = "/workspace/checkpoints",
    mode: str = "static",
    movement_factor: float = 1.0,
    foreground_masking: bool = True,
    multi_trajectory: bool = True,
) -> dict:
    """
    Run the full Lyra inference pipeline.
    
    Returns:
        dict with keys: success, ply_path, error, logs
    """
    result = {
        "success": False,
        "ply_path": None,
        "error": None,
        "logs": [],
    }
    
    # Validate inputs
    if mode == "static":
        if not input_image or not os.path.exists(input_image):
            result["error"] = f"Input image not found: {input_image}"
            return result
    else:
        if not input_video or not os.path.exists(input_video):
            result["error"] = f"Input video not found: {input_video}"
            return result
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create temp directory for intermediate latents
    latents_dir = tempfile.mkdtemp(prefix="lyra_latents_")
    result["logs"].append(f"Latents directory: {latents_dir}")
    
    try:
        # Step 1: SDG
        if mode == "static":
            sdg_success = step1_sdg_static(
                input_image=input_image,
                output_latents_dir=latents_dir,
                checkpoint_dir=checkpoint_dir,
                movement_factor=movement_factor,
                foreground_masking=foreground_masking,
                multi_trajectory=multi_trajectory,
            )
        else:
            sdg_success = step1_sdg_dynamic(
                input_video=input_video,
                output_latents_dir=latents_dir,
                checkpoint_dir=checkpoint_dir,
                foreground_masking=foreground_masking,
                multi_trajectory=multi_trajectory,
            )
        
        if not sdg_success:
            result["error"] = "SDG step failed"
            return result
        
        result["logs"].append("SDG step completed")
        
        # Create symlink from expected path to our latents directory
        # Lyra's dataset registry has hardcoded paths like 'assets/demo/static/diffusion_output'
        lyra_dir = "/workspace/lyra"
        if mode == "static":
            expected_path = os.path.join(lyra_dir, "assets/demo/static/diffusion_output")
        else:
            expected_path = os.path.join(lyra_dir, "assets/demo/dynamic/diffusion_output")
        
        # Create parent directories and symlink
        os.makedirs(os.path.dirname(expected_path), exist_ok=True)
        if os.path.exists(expected_path) or os.path.islink(expected_path):
            os.remove(expected_path) if os.path.isfile(expected_path) else shutil.rmtree(expected_path) if os.path.isdir(expected_path) and not os.path.islink(expected_path) else os.unlink(expected_path)
        os.symlink(latents_dir, expected_path)
        result["logs"].append(f"Created symlink: {expected_path} -> {latents_dir}")
        logger.info(f"Created symlink: {expected_path} -> {latents_dir}")
        
        # Step 2: 3DGS Reconstruction
        config_path = create_lyra_config(
            latents_dir=latents_dir,
            output_dir=output_dir,
            output_name=output_name,
            mode=mode,
            checkpoint_dir=checkpoint_dir,
        )
        
        recon_success = step2_3dgs_reconstruction(
            config_path=config_path,
            checkpoint_dir=checkpoint_dir,
            mode=mode,
        )
        
        # Cleanup temp config
        if os.path.exists(config_path):
            os.unlink(config_path)
        
        if not recon_success:
            result["error"] = "3DGS reconstruction step failed"
            return result
        
        result["logs"].append("3DGS reconstruction completed")
        
        # Find output PLY
        ply_path = find_output_ply(output_dir, output_name, mode)
        if ply_path:
            result["success"] = True
            result["ply_path"] = ply_path
            result["logs"].append(f"Output PLY: {ply_path}")
        else:
            result["error"] = "No PLY file found in output"
        
    finally:
        # Cleanup latents directory
        if os.path.exists(latents_dir):
            shutil.rmtree(latents_dir, ignore_errors=True)
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Lyra Inference for RunPod")
    parser.add_argument("--input_image", type=str, help="Input image path (for static mode)")
    parser.add_argument("--input_video", type=str, help="Input video path (for dynamic mode)")
    parser.add_argument("--output_dir", type=str, default="/tmp/lyra_output", help="Output directory")
    parser.add_argument("--output_name", type=str, default="lyra_output", help="Output name prefix")
    parser.add_argument("--checkpoint_dir", type=str, default="/workspace/checkpoints", help="Checkpoint directory")
    parser.add_argument("--mode", type=str, choices=["static", "dynamic"], default="static", help="Generation mode")
    parser.add_argument("--movement_factor", type=float, default=1.0, help="Camera movement factor (1.0=normal, 2.0=more)")
    parser.add_argument("--foreground_masking", action="store_true", default=True, help="Apply foreground masking")
    parser.add_argument("--no_foreground_masking", action="store_false", dest="foreground_masking")
    parser.add_argument("--multi_trajectory", action="store_true", default=True, help="Generate multiple trajectories")
    parser.add_argument("--no_multi_trajectory", action="store_false", dest="multi_trajectory")
    parser.add_argument("--output_glb", action="store_true", help="Also export GLB (not yet supported)")
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Lyra Inference Script")
    logger.info("=" * 60)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Input image: {args.input_image}")
    logger.info(f"Input video: {args.input_video}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"Output name: {args.output_name}")
    logger.info(f"Checkpoint dir: {args.checkpoint_dir}")
    
    result = run_lyra_inference(
        input_image=args.input_image,
        input_video=args.input_video,
        output_dir=args.output_dir,
        output_name=args.output_name,
        checkpoint_dir=args.checkpoint_dir,
        mode=args.mode,
        movement_factor=args.movement_factor,
        foreground_masking=args.foreground_masking,
        multi_trajectory=args.multi_trajectory,
    )
    
    if result["success"]:
        logger.info("=" * 60)
        logger.info("SUCCESS!")
        logger.info(f"Output PLY: {result['ply_path']}")
        logger.info("=" * 60)
        sys.exit(0)
    else:
        logger.error("=" * 60)
        logger.error("FAILED!")
        logger.error(f"Error: {result['error']}")
        logger.error("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()

