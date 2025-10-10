#!/usr/bin/env python3
"""
CLI tool for Hunyuan3D generation with customizable parameters
"""
import argparse
import time
import torch
from PIL import Image
import os

from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
try:
    from hy3dgen.texgen import Hunyuan3DPaintPipeline
    TEXGEN_AVAILABLE = True
except ImportError:
    TEXGEN_AVAILABLE = False

def main():
    parser = argparse.ArgumentParser(description='Generate 3D models from images using Hunyuan3D')
    parser.add_argument('image_path', help='Path to input image')
    parser.add_argument('output_path', help='Path for output GLB file')
    parser.add_argument('--model',
                       choices=['mini', 'full'],
                       default='mini',
                       help='Shape generation model (mini=faster, full=higher quality)')
    parser.add_argument('--texture-model',
                       choices=['mini', 'full'],
                       help='Texture generation model (optional, uses shape model if not specified)')
    parser.add_argument('--steps', type=int, default=30,
                       help='Number of inference steps (default: 30)')
    parser.add_argument('--guidance-scale', type=float, default=7.5,
                       help='Guidance scale (default: 7.5)')
    parser.add_argument('--octree-resolution', type=int, default=256,
                       help='Octree resolution (default: 256)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--fp16', action='store_true',
                       help='Use FP16 precision')
    parser.add_argument('--no-rembg', action='store_true',
                       help='Skip background removal')
    parser.add_argument('--texture', action='store_true',
                       help='Generate texture (requires texture model)')

    args = parser.parse_args()

    # Convert simplified model names to full model names
    model_map = {
        'mini': 'tencent/Hunyuan3D-2mini',
        'full': 'tencent/Hunyuan3D-2'
    }
    shape_model = model_map[args.model]
    texture_model = model_map[args.texture_model] if args.texture_model else shape_model

    print(f"Loading image: {args.image_path}")
    print(f"Output will be saved to: {args.output_path}")
    print(f"Using shape model: {shape_model}")
    if args.texture:
        print(f"Using texture model: {texture_model}")

    # Load and preprocess image
    image = Image.open(args.image_path).convert("RGBA")

    if not args.no_rembg and image.mode == 'RGBA':
        print("Removing background...")
        rembg = BackgroundRemover()
        image = rembg(image.convert("RGB")).convert("RGBA")

    # Load shape pipeline
    print(f"Loading shape model: {shape_model}")

    # Determine subfolder based on model
    subfolder = None
    if "mini" in shape_model:
        subfolder = "hunyuan3d-dit-v2-mini-turbo"

    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        shape_model,
        subfolder=subfolder,
        torch_dtype=torch.float16 if args.fp16 else torch.float32,
    )

    # Load texture pipeline if requested
    texture_pipeline = None
    if args.texture:
        tex_model = texture_model  # Use the mapped texture model
        if TEXGEN_AVAILABLE:
            print(f"Loading texture model: {tex_model}")

            # Determine subfolder for texture model
            tex_subfolder = None
            if "mini" in tex_model:
                tex_subfolder = "hunyuan3d-dit-v2-mini-turbo"

            texture_pipeline = Hunyuan3DPaintPipeline.from_pretrained(
                tex_model,
                subfolder=tex_subfolder,
                torch_dtype=torch.float16 if args.fp16 else torch.float32,
            )
        else:
            print("Warning: Texture generation requested but texgen module not available")
            args.texture = False

    # Set seed for reproducibility
    if args.seed is not None:
        torch.manual_seed(args.seed)

    # Generate 3D model
    print(f"Generating 3D model with {args.steps} steps...")
    start_time = time.time()

    mesh = pipeline(
        image=image,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.steps,
        octree_resolution=args.octree_resolution,
        generator=torch.Generator().manual_seed(args.seed) if args.seed else None,
    )[0]

    # Apply texture if requested
    if args.texture and texture_pipeline:
        print("Applying texture...")
        texture_start = time.time()
        mesh = texture_pipeline(mesh, image=image)
        texture_time = time.time() - texture_start
        print(f"Texture applied in {texture_time:.2f} seconds")

    total_time = time.time() - start_time
    print(f"Total generation completed in {total_time:.2f} seconds")

    # Export to GLB
    print(f"Exporting to GLB: {args.output_path}")
    mesh.export(args.output_path)

    print("✅ 3D model generation completed!")
    print(f"📁 GLB file saved: {args.output_path}")
    print(f"⏱️  Total time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()
