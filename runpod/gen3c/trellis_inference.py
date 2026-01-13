#!/usr/bin/env python3
"""
TRELLIS.2 Inference Script for RunPod Serverless

This script wraps TRELLIS.2's Python API for use in the RunPod handler.
TRELLIS.2 uses a Python API approach rather than CLI, so we need this wrapper.

Usage:
    python trellis_inference.py \
        --input_image /path/to/image.png \
        --output_dir /path/to/output \
        --output_name my_model \
        --resolution 1024 \
        --guidance_scale 7.5 \
        --checkpoint_dir /path/to/checkpoints \
        --output_glb \
        --seed 42
"""

import argparse
import os
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="TRELLIS.2 Inference")
    parser.add_argument("--input_image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--output_name", type=str, default="trellis_output", help="Output filename (without extension)")
    parser.add_argument("--resolution", type=int, default=1024, choices=[512, 768, 1024, 1536], help="Generation resolution")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale for generation")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Path to TRELLIS.2 checkpoints")
    parser.add_argument("--output_glb", action="store_true", help="Export GLB format")
    parser.add_argument("--output_ply", action="store_true", help="Export PLY format")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Validate input
    if not os.path.exists(args.input_image):
        logger.error(f"Input image not found: {args.input_image}")
        sys.exit(1)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set seed if provided
    if args.seed is not None:
        import random
        import numpy as np
        import torch
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        logger.info(f"Set random seed to {args.seed}")

    # Import TRELLIS.2 modules
    try:
        logger.info("Loading TRELLIS.2 modules...")
        
        # Add TRELLIS.2 to path if needed
        trellis_path = "/workspace/TRELLIS2"
        if trellis_path not in sys.path:
            sys.path.insert(0, trellis_path)
        
        # Set required env vars before importing
        os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        
        import torch
        from PIL import Image
        
        # TRELLIS.2 uses trellis2 module (not trellis)
        from trellis2.pipelines import Trellis2ImageTo3DPipeline
        import o_voxel
        
        logger.info("TRELLIS.2 modules loaded successfully")
        
    except ImportError as e:
        logger.error(f"Failed to import TRELLIS.2 modules: {e}")
        logger.error("Make sure TRELLIS.2 is properly installed")
        sys.exit(1)

    # Initialize pipeline
    try:
        logger.info("Initializing TRELLIS.2 pipeline...")
        
        # Determine checkpoint path
        if args.checkpoint_dir:
            checkpoint_path = args.checkpoint_dir
        else:
            checkpoint_path = os.environ.get("TRELLIS_CHECKPOINT_DIR", "/runpod-volume/checkpoints/trellis")
        
        # Check for local checkpoints - they're directly in the checkpoint_path (not in a subdirectory)
        # The structure is: checkpoint_path/pipeline.json, checkpoint_path/ckpts/
        pipeline_json = os.path.join(checkpoint_path, "pipeline.json")
        ckpts_dir = os.path.join(checkpoint_path, "ckpts")
        
        if os.path.exists(pipeline_json) and os.path.isdir(ckpts_dir):
            logger.info(f"Loading from local checkpoint: {checkpoint_path}")
            logger.info(f"  - pipeline.json: {pipeline_json}")
            logger.info(f"  - ckpts dir: {ckpts_dir}")
            
            # List available checkpoints
            ckpt_files = os.listdir(ckpts_dir)
            logger.info(f"  - Found {len(ckpt_files)} checkpoint files")
            
            pipeline = Trellis2ImageTo3DPipeline.from_pretrained(checkpoint_path)
        else:
            logger.info(f"Local checkpoints not found at {checkpoint_path}")
            logger.info("Loading from HuggingFace: microsoft/TRELLIS.2-4B")
            # This will download to HF cache
            pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
        
        # Move to GPU
        pipeline.cuda()
        logger.info("Pipeline initialized and moved to GPU")
        
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Load input image
    try:
        logger.info(f"Loading input image: {args.input_image}")
        image = Image.open(args.input_image).convert("RGBA")
        logger.info(f"Image loaded: {image.size}")
    except Exception as e:
        logger.error(f"Failed to load image: {e}")
        sys.exit(1)

    # Run inference
    try:
        logger.info(f"Running TRELLIS.2 inference (resolution={args.resolution}, guidance={args.guidance_scale})...")
        
        # TRELLIS.2 pipeline.run() returns a list of mesh objects
        mesh = pipeline.run(image, seed=args.seed if args.seed is not None else 42)[0]
        
        # Simplify mesh to stay within nvdiffrast limits
        mesh.simplify(16777216)
        
        logger.info("Inference complete")
        
        output_paths = []
        
        # Export GLB using o_voxel.postprocess.to_glb
        if args.output_glb or (not args.output_glb and not args.output_ply):
            glb_path = os.path.join(args.output_dir, f"{args.output_name}.glb")
            logger.info(f"Exporting GLB: {glb_path}")
            
            # Use o_voxel postprocessing to export GLB
            glb = o_voxel.postprocess.to_glb(
                vertices=mesh.vertices,
                faces=mesh.faces,
                attr_volume=mesh.attrs,
                coords=mesh.coords,
                attr_layout=mesh.layout,
                voxel_size=mesh.voxel_size,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                decimation_target=1000000,
                texture_size=4096,
                remesh=True,
                remesh_band=1,
                remesh_project=0,
                verbose=True
            )
            glb.export(glb_path, extension_webp=True)
            
            if os.path.exists(glb_path):
                file_size = os.path.getsize(glb_path) / 1024 / 1024
                logger.info(f"GLB exported: {glb_path} ({file_size:.1f}MB)")
                output_paths.append(glb_path)
            else:
                logger.warning("GLB export failed - file not created")
        
        # Export PLY if requested
        if args.output_ply:
            ply_path = os.path.join(args.output_dir, f"{args.output_name}.ply")
            logger.info(f"Exporting PLY: {ply_path}")
            
            # Use o_voxel postprocessing to export PLY
            try:
                ply = o_voxel.postprocess.to_ply(
                    vertices=mesh.vertices,
                    faces=mesh.faces,
                    attr_volume=mesh.attrs,
                    coords=mesh.coords,
                    attr_layout=mesh.layout,
                    voxel_size=mesh.voxel_size,
                    aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                )
                ply.export(ply_path)
                
                if os.path.exists(ply_path):
                    file_size = os.path.getsize(ply_path) / 1024 / 1024
                    logger.info(f"PLY exported: {ply_path} ({file_size:.1f}MB)")
                    output_paths.append(ply_path)
                else:
                    logger.warning("PLY export failed - file not created")
            except Exception as e:
                logger.warning(f"PLY export not available: {e}")
        
        if output_paths:
            logger.info(f"Successfully exported: {output_paths}")
            print(f"OUTPUT_PATHS: {','.join(output_paths)}")
        else:
            logger.error("No output files were created")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    logger.info("TRELLIS.2 inference completed successfully")

if __name__ == "__main__":
    main()

