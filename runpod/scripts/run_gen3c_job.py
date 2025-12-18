#!/usr/bin/env python3
"""
GEN3C RunPod Job Submission Script

Usage:
    python run_gen3c_job.py <image_path> [options]

Example:
    python run_gen3c_job.py /srv/searidge_share/inputs/BrightClub_v23.png
"""

import base64
import requests
import time
import argparse
import sys
from pathlib import Path

# Default RunPod API URL - update this with your pod ID
DEFAULT_API_URL = "https://iv94zuokozefxc-8000.proxy.runpod.net"

def main():
    parser = argparse.ArgumentParser(description="Submit GEN3C job to RunPod")
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="RunPod API URL")
    parser.add_argument("--output", "-o", help="Output video path (default: same as input with .mp4)")
    parser.add_argument("--name", default="gen3c_output", help="Video name")
    parser.add_argument("--frames", type=int, default=121, choices=[121, 241, 361, 481], help="Number of frames (N*120+1)")
    parser.add_argument("--trajectory", default="left", choices=["left", "right", "up", "down", "zoom_in", "zoom_out", "clockwise", "counterclockwise", "none"])
    parser.add_argument("--guidance", type=float, default=1.0, help="Guidance scale (0.5-3.0)")
    parser.add_argument("--no-foreground-mask", action="store_true", help="Disable foreground masking")
    parser.add_argument("--poll-interval", type=int, default=30, help="Status check interval in seconds")
    
    args = parser.parse_args()
    
    # Validate image exists
    image_path = Path(args.image_path)
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)
    
    # Set output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path("/srv/searidge_share/outputs/gen3c") / f"{image_path.stem}.mp4"
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check API health
    print(f"Checking API at {args.api_url}...")
    try:
        health = requests.get(f"{args.api_url}/health", timeout=10).json()
        print(f"  GPU: {health.get('gpu', 'unknown')}")
        print(f"  Model exists: {health.get('model_exists', 'unknown')}")
        print(f"  Tokenizer exists: {health.get('tokenizer_exists', 'unknown')}")
        if not health.get('model_exists') or not health.get('tokenizer_exists'):
            print("Warning: Checkpoints may not be properly loaded!")
    except Exception as e:
        print(f"Error connecting to API: {e}")
        sys.exit(1)
    
    # Read and encode image
    print(f"\nEncoding image: {image_path}")
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()
    print(f"  Image size: {len(img_b64):,} bytes (base64)")
    
    # Submit job
    print(f"\nSubmitting job...")
    print(f"  Frames: {args.frames}")
    print(f"  Trajectory: {args.trajectory}")
    print(f"  Guidance: {args.guidance}")
    print(f"  Foreground masking: {not args.no_foreground_mask}")
    
    response = requests.post(f"{args.api_url}/generate", json={
        "image_base64": img_b64,
        "video_name": args.name,
        "num_frames": args.frames,
        "trajectory": args.trajectory,
        "guidance": args.guidance,
        "foreground_masking": not args.no_foreground_mask
    }, timeout=30)
    
    result = response.json()
    if "job_id" not in result:
        print(f"Error submitting job: {result}")
        sys.exit(1)
    
    job_id = result["job_id"]
    print(f"\nJob submitted: {job_id}")
    print(f"Waiting for completion (estimated 15-20 minutes for 121 frames)...")
    print(f"Polling every {args.poll_interval} seconds...\n")
    
    # Poll for completion
    start_time = time.time()
    while True:
        time.sleep(args.poll_interval)
        
        try:
            status = requests.get(f"{args.api_url}/status/{job_id}", timeout=10).json()
        except Exception as e:
            print(f"  Warning: Status check failed: {e}")
            continue
        
        elapsed = int(time.time() - start_time)
        elapsed_str = f"{elapsed // 60}m {elapsed % 60}s"
        print(f"  [{elapsed_str}] Status: {status['status']}")
        
        if status["status"] == "completed":
            print(f"\n✓ Job completed successfully!")
            
            # Save video
            if status.get("video_base64"):
                video_data = base64.b64decode(status["video_base64"])
                with open(output_path, "wb") as f:
                    f.write(video_data)
                print(f"✓ Video saved to: {output_path}")
                print(f"  Size: {len(video_data):,} bytes")
            else:
                print("Warning: No video data in response")
            break
            
        elif status["status"] == "failed":
            print(f"\n✗ Job failed!")
            print(f"Error: {status.get('error', 'Unknown error')}")
            sys.exit(1)

if __name__ == "__main__":
    main()

