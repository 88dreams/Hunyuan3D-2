#!/usr/bin/env python3
"""
Test script for the 2DGS Stage Pipeline serverless endpoint.
Converts a Gen3C video to a 3D mesh.

Supports:
- Local video files (uploaded to S3 automatically)
- S3 URLs
- HTTP URLs
"""

import os
import sys
import time
import uuid
from pathlib import Path

import runpod
import requests

# Try to import boto3 for S3 uploads
try:
    import boto3
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False

# Local output directory
LOCAL_OUTPUT_DIR = Path("/srv/searidge_share/outputs/mesh_vipe")

# Configuration
ENDPOINT_ID = "s9txp6edtf2vg4"

# S3 Configuration (from aws_credentials.env)
S3_BUCKET = os.environ.get("S3_BUCKET", "arkrunr")
S3_REGION = os.environ.get("S3_REGION", "us-west-1")
S3_PREFIX = os.environ.get("S3_PREFIX", "MediaContent")

# Try to load API key
API_KEY = os.environ.get("RUNPOD_API_KEY")
if not API_KEY:
    try:
        import json
        config_paths = [
            Path(__file__).parent.parent / ".runpod_config.json",
            Path.home() / ".runpod_config.json",
        ]
        for config_path in config_paths:
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)
                    API_KEY = config.get("api_key") or config.get("gen3c_api_key")
                    if API_KEY:
                        break
    except:
        pass

# Load AWS credentials
def load_aws_credentials():
    """Load AWS credentials from config file."""
    aws_creds_file = Path.home() / ".config" / "3d_studio" / "aws_credentials.env"
    if aws_creds_file.exists():
        with open(aws_creds_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and value:
                        os.environ[key] = value

load_aws_credentials()


def upload_to_s3(local_path: str) -> str:
    """
    Upload a local file to S3 and return the URL.
    
    Args:
        local_path: Path to local file
        
    Returns:
        S3 URL
    """
    if not HAS_BOTO3:
        raise RuntimeError("boto3 not installed. Run: pip install boto3")
    
    local_path = Path(local_path)
    if not local_path.exists():
        raise FileNotFoundError(f"File not found: {local_path}")
    
    # Generate unique S3 key
    unique_id = str(uuid.uuid4())[:8]
    s3_key = f"{S3_PREFIX}/stage-pipeline/inputs/{unique_id}_{local_path.name}"
    
    print(f"Uploading to S3...")
    print(f"  Local: {local_path}")
    print(f"  S3: s3://{S3_BUCKET}/{s3_key}")
    
    s3 = boto3.client("s3", region_name=S3_REGION)
    s3.upload_file(str(local_path), S3_BUCKET, s3_key)
    
    # Generate presigned URL (valid for 1 hour)
    url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": S3_BUCKET, "Key": s3_key},
        ExpiresIn=3600
    )
    
    print(f"  URL: {url[:80]}...")
    return url


def resolve_video_input(video_input: str) -> str:
    """
    Resolve video input to a URL.
    
    - If it's a URL (http/https/s3), return as-is
    - If it's a local file, upload to S3 and return URL
    """
    if video_input.startswith(("http://", "https://", "s3://")):
        return video_input
    
    # Local file - upload to S3
    return upload_to_s3(video_input)


def test_health_check():
    """Check if the endpoint is ready."""
    print("Checking endpoint health...")
    endpoint = runpod.Endpoint(ENDPOINT_ID)
    health = endpoint.health()
    print(f"Health: {health}")
    
    workers = health.get("workers", {})
    ready = workers.get("ready", 0)
    idle = workers.get("idle", 0)
    
    if ready > 0:
        print(f"✅ Endpoint ready ({ready} workers)")
    else:
        print(f"⚠️ No workers ready (idle: {idle})")
    
    return health


def run_pipeline(video_input: str, iterations: int = 5000, output_format: str = "glb"):
    """
    Run the full stage pipeline.
    
    Args:
        video_input: Path to local video OR URL
        iterations: 2DGS training iterations (default 5000)
        output_format: Output mesh format ("glb", "obj", or "ply")
    """
    print(f"\n{'='*60}")
    print("STAGE PIPELINE TEST")
    print(f"{'='*60}")
    print(f"Video input: {video_input}")
    print(f"Iterations: {iterations}")
    print(f"Output format: {output_format}")
    print(f"{'='*60}\n")
    
    # Resolve video input (upload to S3 if local)
    video_url = resolve_video_input(video_input)
    print(f"Video URL: {video_url[:80]}...")
    
    endpoint = runpod.Endpoint(ENDPOINT_ID)
    
    job_input = {
        "video_url": video_url,
        "iterations": iterations,
        "output_format": output_format,
        "mesh_resolution": 512,
        "output_s3": {
            "bucket": S3_BUCKET,
            "region": S3_REGION,
            "prefix": f"{S3_PREFIX}/stage-pipeline/outputs/"
        }
    }
    
    print("\nStarting job...")
    start_time = time.time()
    
    # Run async and poll for status
    run_request = endpoint.run(job_input)
    job_id = run_request.job_id
    print(f"Job ID: {job_id}")
    
    # Poll for completion
    while True:
        status = run_request.status()
        elapsed = time.time() - start_time
        print(f"  [{elapsed:.0f}s] Status: {status}")
        
        if status == "COMPLETED":
            result = run_request.output()
            print(f"\n{'='*60}")
            print("RESULT:")
            print(f"{'='*60}")
            for key, value in result.items():
                if key == "mesh_url" and len(str(value)) > 100:
                    print(f"  {key}: {str(value)[:100]}...")
                else:
                    print(f"  {key}: {value}")
            
            # Download mesh if successful
            if result.get("status") == "success" and result.get("mesh_url"):
                mesh_url = result["mesh_url"]
                
                # Create output directory if needed
                LOCAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                
                # Generate output filename from video input
                video_name = Path(video_input).stem if not video_input.startswith("http") else "stage_mesh"
                output_name = f"{video_name}_{job_id[:8]}.{output_format}"
                local_path = LOCAL_OUTPUT_DIR / output_name
                
                print(f"\n📥 Downloading mesh to local storage...")
                print(f"  URL: {mesh_url[:80]}...")
                print(f"  Local: {local_path}")
                
                try:
                    response = requests.get(mesh_url, timeout=300)
                    response.raise_for_status()
                    with open(local_path, "wb") as f:
                        f.write(response.content)
                    file_size = local_path.stat().st_size
                    print(f"  ✅ Downloaded: {file_size / 1024 / 1024:.2f} MB")
                except Exception as e:
                    print(f"  ❌ Download failed: {e}")
                    print(f"\n📥 Manual download:")
                    print(f"  curl -o {output_name} '{mesh_url}'")
            
            return result
            
        elif status == "FAILED":
            result = run_request.output()
            print(f"\n❌ JOB FAILED: {result}")
            return result
            
        time.sleep(15)


if __name__ == "__main__":
    import argparse
    
    if not API_KEY:
        print("ERROR: Set RUNPOD_API_KEY environment variable")
        sys.exit(1)
    
    runpod.api_key = API_KEY
    
    parser = argparse.ArgumentParser(description="Test the 2DGS Stage Pipeline")
    parser.add_argument("--video", "-v", help="Path to local video OR URL")
    parser.add_argument("--iterations", "-i", type=int, default=5000, help="Training iterations")
    parser.add_argument("--format", "-f", default="glb", choices=["glb", "obj", "ply"])
    parser.add_argument("--health", action="store_true", help="Just check health")
    
    args = parser.parse_args()
    
    if args.health:
        test_health_check()
    elif args.video:
        run_pipeline(args.video, args.iterations, args.format)
    else:
        print("Usage:")
        print("  Check health: python test_stage_pipeline.py --health")
        print("  Run pipeline: python test_stage_pipeline.py --video <path_or_url>")
        print("\nExamples:")
        print("  python test_stage_pipeline.py --video /path/to/gen3c_video.mp4")
        print("  python test_stage_pipeline.py --video https://example.com/video.mp4")
        print("  python test_stage_pipeline.py --video /path/to/video.mp4 --iterations 3000 --format glb")
