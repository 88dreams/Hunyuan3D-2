#!/usr/bin/env python3
"""
Test ViPE Installation and Process a Gen3C Video

Usage:
    # Test installation only
    python scripts/test_vipe.py --check
    
    # Process a video
    python scripts/test_vipe.py --video /path/to/gen3c_video.mp4
    
    # Full test with existing Gen3C video
    python scripts/test_vipe.py --video /workspace/outputs/gen3c/latest.mp4 --output vipe_test/
"""

import argparse
import subprocess
import sys
from pathlib import Path


def check_vipe_installation():
    """Check if ViPE is installed and working."""
    print("="*60)
    print("Checking ViPE Installation")
    print("="*60)
    
    checks = []
    
    # Check 1: vipe command
    print("\n[1] Checking vipe command...")
    try:
        result = subprocess.run(
            ["vipe", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            print("    ✓ vipe command available")
            checks.append(True)
        else:
            print(f"    ❌ vipe command failed: {result.stderr[:200]}")
            checks.append(False)
    except FileNotFoundError:
        print("    ❌ vipe command not found in PATH")
        print("    → Run: source /workspace/activate_vipe.sh")
        checks.append(False)
    except Exception as e:
        print(f"    ❌ Error: {e}")
        checks.append(False)
    
    # Check 2: Python imports
    print("\n[2] Checking Python imports...")
    try:
        import numpy as np
        print("    ✓ numpy available")
        
        import cv2
        print("    ✓ opencv available")
        
        import torch
        print(f"    ✓ torch available: {torch.__version__}")
        print(f"    ✓ CUDA available: {torch.cuda.is_available()}")
        
        checks.append(True)
    except ImportError as e:
        print(f"    ❌ Import error: {e}")
        checks.append(False)
    
    # Check 3: ViPE directory
    print("\n[3] Checking ViPE directory...")
    vipe_dir = Path("/workspace/vipe")
    if vipe_dir.exists():
        print(f"    ✓ ViPE directory exists: {vipe_dir}")
        
        # Check for key files
        key_files = ["requirements.txt", "README.md"]
        for f in key_files:
            if (vipe_dir / f).exists():
                print(f"    ✓ Found: {f}")
        
        checks.append(True)
    else:
        print(f"    ❌ ViPE directory not found at {vipe_dir}")
        print("    → Run: bash runpod/vipe/setup_vipe.sh")
        checks.append(False)
    
    # Summary
    print("\n" + "="*60)
    if all(checks):
        print("✓ ViPE installation looks good!")
        return True
    else:
        print("❌ ViPE installation incomplete")
        print("\nTo install ViPE:")
        print("  1. Copy setup_vipe.sh to RunPod")
        print("  2. Run: bash setup_vipe.sh")
        print("  3. Activate: source /workspace/activate_vipe.sh")
        return False


def process_video(video_path: str, output_dir: str):
    """Process a video with ViPE."""
    print("="*60)
    print("Processing Video with ViPE")
    print("="*60)
    
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    
    if not video_path.exists():
        print(f"❌ Video not found: {video_path}")
        return False
    
    print(f"\nInput video: {video_path}")
    print(f"Output directory: {output_dir}")
    
    # Run ViPE
    cmd = [
        "vipe", "infer",
        str(video_path),
        "--output", str(output_dir),
    ]
    
    print(f"\nRunning: {' '.join(cmd)}")
    print("-"*60)
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print(f"\n❌ ViPE failed with return code: {result.returncode}")
        return False
    
    # Check outputs
    print("\n" + "-"*60)
    print("Checking outputs...")
    
    expected_files = ["poses.npy", "intrinsics.npy"]
    for f in expected_files:
        path = output_dir / f
        if path.exists():
            import numpy as np
            data = np.load(path)
            print(f"  ✓ {f}: shape={data.shape}")
        else:
            print(f"  ❌ {f}: not found")
    
    depth_dir = output_dir / "depth"
    if depth_dir.exists():
        depth_files = list(depth_dir.glob("*.npy"))
        print(f"  ✓ depth/: {len(depth_files)} files")
    else:
        print("  ❌ depth/: not found")
    
    print("\n" + "="*60)
    print("✓ ViPE processing complete!")
    print(f"Output: {output_dir}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Test ViPE Installation")
    parser.add_argument("--check", action="store_true", help="Only check installation")
    parser.add_argument("--video", help="Path to video to process")
    parser.add_argument("--output", default="vipe_test", help="Output directory")
    
    args = parser.parse_args()
    
    if args.check:
        success = check_vipe_installation()
        sys.exit(0 if success else 1)
    
    if args.video:
        # First check installation
        if not check_vipe_installation():
            sys.exit(1)
        
        # Then process video
        success = process_video(args.video, args.output)
        sys.exit(0 if success else 1)
    
    # Default: just check installation
    success = check_vipe_installation()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
