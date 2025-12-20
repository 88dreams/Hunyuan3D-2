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
        
        import torch
        from PIL import Image
        
        # TRELLIS.2 uses a pipeline approach
        from trellis.pipelines import TrellisImageTo3DPipeline
        
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
        
        # Check if we have local checkpoints or need to download
        local_model_path = os.path.join(checkpoint_path, "TRELLIS.2-4B")
        
        if os.path.exists(local_model_path):
            logger.info(f"Loading from local checkpoint: {local_model_path}")
            pipeline = TrellisImageTo3DPipeline.from_pretrained(local_model_path)
        else:
            logger.info("Loading from HuggingFace: microsoft/TRELLIS-image-large")
            # This will download to HF cache
            pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
        
        # Move to GPU
        pipeline = pipeline.to("cuda")
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
        
        # TRELLIS.2 pipeline call
        outputs = pipeline(
            image,
            seed=args.seed if args.seed is not None else 42,
            guidance_scale=args.guidance_scale,
        )
        
        logger.info("Inference complete")
        
        # Extract the 3D representation
        # TRELLIS.2 outputs O-Voxel representation that can be exported to various formats
        
        output_paths = []
        
        # Export GLB
        if args.output_glb:
            glb_path = os.path.join(args.output_dir, f"{args.output_name}.glb")
            logger.info(f"Exporting GLB: {glb_path}")
            
            # TRELLIS.2 has built-in export functionality
            outputs.save_glb(glb_path)
            
            if os.path.exists(glb_path):
                file_size = os.path.getsize(glb_path) / 1024 / 1024
                logger.info(f"GLB exported: {glb_path} ({file_size:.1f}MB)")
                output_paths.append(glb_path)
            else:
                logger.warning("GLB export failed - file not created")
        
        # Export PLY
        if args.output_ply:
            ply_path = os.path.join(args.output_dir, f"{args.output_name}.ply")
            logger.info(f"Exporting PLY: {ply_path}")
            
            # TRELLIS.2 has built-in export functionality
            outputs.save_ply(ply_path)
            
            if os.path.exists(ply_path):
                file_size = os.path.getsize(ply_path) / 1024 / 1024
                logger.info(f"PLY exported: {ply_path} ({file_size:.1f}MB)")
                output_paths.append(ply_path)
            else:
                logger.warning("PLY export failed - file not created")
        
        # Default to GLB if neither specified
        if not args.output_glb and not args.output_ply:
            glb_path = os.path.join(args.output_dir, f"{args.output_name}.glb")
            logger.info(f"Exporting default GLB: {glb_path}")
            outputs.save_glb(glb_path)
            output_paths.append(glb_path)
        
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

