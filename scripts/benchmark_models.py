#!/usr/bin/env python3
"""
Model Benchmark Script

Systematically benchmarks different 3D generation models:
- SHARP (feed-forward)
- GEN3C (diffusion video)
- Lyra (GEN3C + 3DGS)
- Hunyuan3D (diffusion mesh)

Measures:
- Wall-clock time
- GPU memory usage
- Output file sizes

Usage:
    # Benchmark all models on a single image
    python benchmark_models.py --image /path/to/image.png --output-dir /path/to/results
    
    # Benchmark specific models
    python benchmark_models.py --image /path/to/image.png --models sharp lyra
    
    # Benchmark with multiple runs for averaging
    python benchmark_models.py --image /path/to/image.png --runs 3
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import shutil


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    model: str
    image: str
    run_number: int
    success: bool
    wall_time_seconds: float
    output_path: Optional[str]
    output_size_bytes: int
    error: Optional[str]
    gpu_memory_mb: Optional[float]
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ModelBenchmark:
    """Benchmark runner for 3D generation models."""
    
    def __init__(
        self,
        output_dir: str,
        endpoint_id: str = "",
        api_key: str = "",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.endpoint_id = endpoint_id
        self.api_key = api_key
        self.results: List[BenchmarkResult] = []
    
    def run_sharp(self, image_path: str, run_number: int) -> BenchmarkResult:
        """Benchmark SHARP model."""
        output_name = f"benchmark_sharp_run{run_number}"
        output_path = self.output_dir / f"{output_name}.ply"
        
        start_time = time.time()
        error = None
        success = False
        
        try:
            # Check if SHARP is available
            if not shutil.which("sharp"):
                raise RuntimeError("SHARP CLI not found. Install from ml-sharp repo.")
            
            # Create temp directories
            import tempfile
            temp_input = tempfile.mkdtemp(prefix="sharp_bench_in_")
            temp_output = tempfile.mkdtemp(prefix="sharp_bench_out_")
            
            # Copy input
            ext = Path(image_path).suffix
            shutil.copy2(image_path, os.path.join(temp_input, f"input{ext}"))
            
            # Run SHARP (PLY only, no video for fair comparison)
            cmd = ["sharp", "predict", "-i", temp_input, "-o", temp_output]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                # Find output PLY
                ply_files = list(Path(temp_output).glob("*.ply"))
                if ply_files:
                    shutil.copy2(ply_files[0], output_path)
                    success = True
                else:
                    error = "No PLY output generated"
            else:
                error = result.stderr[:500]
            
            # Cleanup
            shutil.rmtree(temp_input, ignore_errors=True)
            shutil.rmtree(temp_output, ignore_errors=True)
            
        except subprocess.TimeoutExpired:
            error = "Timeout after 5 minutes"
        except Exception as e:
            error = str(e)
        
        wall_time = time.time() - start_time
        output_size = output_path.stat().st_size if output_path.exists() else 0
        
        return BenchmarkResult(
            model="sharp",
            image=image_path,
            run_number=run_number,
            success=success,
            wall_time_seconds=wall_time,
            output_path=str(output_path) if success else None,
            output_size_bytes=output_size,
            error=error,
            gpu_memory_mb=None,  # SHARP doesn't report this easily
            timestamp=datetime.now().isoformat(),
        )
    
    def run_gen3c(self, image_path: str, run_number: int, trajectory: str = "orbit") -> BenchmarkResult:
        """Benchmark GEN3C model (via RunPod)."""
        output_name = f"benchmark_gen3c_run{run_number}"
        
        if not self.endpoint_id or not self.api_key:
            return BenchmarkResult(
                model="gen3c",
                image=image_path,
                run_number=run_number,
                success=False,
                wall_time_seconds=0,
                output_path=None,
                output_size_bytes=0,
                error="RunPod credentials not provided",
                gpu_memory_mb=None,
                timestamp=datetime.now().isoformat(),
            )
        
        start_time = time.time()
        error = None
        success = False
        output_path = None
        output_size = 0
        
        try:
            from runpod.runpod_client import UnifiedServerlessClient
            
            client = UnifiedServerlessClient(self.endpoint_id, self.api_key)
            
            result = client.generate_gen3c_sync(
                image_path=image_path,
                output_dir=str(self.output_dir),
                output_name=output_name,
                trajectory=trajectory,
                num_frames=61,  # Shorter for benchmark
                poll_interval=30,
                max_wait=1800,  # 30 min timeout
            )
            
            if result.success:
                success = True
                output_path = result.output_path
                if output_path and os.path.exists(output_path):
                    output_size = os.path.getsize(output_path)
            else:
                error = result.error
                
        except Exception as e:
            error = str(e)
        
        wall_time = time.time() - start_time
        
        return BenchmarkResult(
            model="gen3c",
            image=image_path,
            run_number=run_number,
            success=success,
            wall_time_seconds=wall_time,
            output_path=output_path,
            output_size_bytes=output_size,
            error=error,
            gpu_memory_mb=None,
            timestamp=datetime.now().isoformat(),
        )
    
    def run_lyra(self, image_path: str, run_number: int) -> BenchmarkResult:
        """Benchmark Lyra model (via RunPod)."""
        output_name = f"benchmark_lyra_run{run_number}"
        
        if not self.endpoint_id or not self.api_key:
            return BenchmarkResult(
                model="lyra",
                image=image_path,
                run_number=run_number,
                success=False,
                wall_time_seconds=0,
                output_path=None,
                output_size_bytes=0,
                error="RunPod credentials not provided",
                gpu_memory_mb=None,
                timestamp=datetime.now().isoformat(),
            )
        
        start_time = time.time()
        error = None
        success = False
        output_path = None
        output_size = 0
        
        try:
            from runpod.runpod_client import UnifiedServerlessClient
            
            client = UnifiedServerlessClient(self.endpoint_id, self.api_key)
            
            result = client.generate_lyra_sync(
                image_path=image_path,
                output_dir=str(self.output_dir),
                output_name=output_name,
                num_views=8,
                poll_interval=30,
                max_wait=2400,  # 40 min timeout
            )
            
            if result.success:
                success = True
                output_path = result.output_path
                if output_path and os.path.exists(output_path):
                    output_size = os.path.getsize(output_path)
            else:
                error = result.error
                
        except Exception as e:
            error = str(e)
        
        wall_time = time.time() - start_time
        
        return BenchmarkResult(
            model="lyra",
            image=image_path,
            run_number=run_number,
            success=success,
            wall_time_seconds=wall_time,
            output_path=output_path,
            output_size_bytes=output_size,
            error=error,
            gpu_memory_mb=None,
            timestamp=datetime.now().isoformat(),
        )
    
    def run_hunyuan(self, image_path: str, run_number: int) -> BenchmarkResult:
        """Benchmark Hunyuan3D model (via RunPod)."""
        output_name = f"benchmark_hunyuan_run{run_number}"
        
        # Note: Hunyuan uses a different endpoint
        # For now, skip if not configured
        return BenchmarkResult(
            model="hunyuan",
            image=image_path,
            run_number=run_number,
            success=False,
            wall_time_seconds=0,
            output_path=None,
            output_size_bytes=0,
            error="Hunyuan benchmark not yet implemented (different endpoint)",
            gpu_memory_mb=None,
            timestamp=datetime.now().isoformat(),
        )
    
    def run_benchmark(
        self,
        image_path: str,
        models: List[str],
        num_runs: int = 1,
    ) -> List[BenchmarkResult]:
        """Run benchmarks for specified models."""
        
        print(f"\n{'='*60}")
        print(f"BENCHMARK: {Path(image_path).name}")
        print(f"Models: {', '.join(models)}")
        print(f"Runs per model: {num_runs}")
        print(f"Output: {self.output_dir}")
        print(f"{'='*60}\n")
        
        results = []
        
        for model in models:
            print(f"\n--- {model.upper()} ---")
            
            for run in range(1, num_runs + 1):
                print(f"  Run {run}/{num_runs}...", end=" ", flush=True)
                
                if model == "sharp":
                    result = self.run_sharp(image_path, run)
                elif model == "gen3c":
                    result = self.run_gen3c(image_path, run)
                elif model == "lyra":
                    result = self.run_lyra(image_path, run)
                elif model == "hunyuan":
                    result = self.run_hunyuan(image_path, run)
                else:
                    print(f"Unknown model: {model}")
                    continue
                
                results.append(result)
                
                if result.success:
                    print(f"✅ {result.wall_time_seconds:.1f}s")
                else:
                    print(f"❌ {result.error[:50] if result.error else 'Unknown error'}")
        
        self.results.extend(results)
        return results
    
    def generate_report(self) -> str:
        """Generate a summary report of all benchmark results."""
        
        if not self.results:
            return "No benchmark results available."
        
        # Group by model
        by_model: Dict[str, List[BenchmarkResult]] = {}
        for r in self.results:
            if r.model not in by_model:
                by_model[r.model] = []
            by_model[r.model].append(r)
        
        lines = [
            "=" * 60,
            "BENCHMARK REPORT",
            "=" * 60,
            f"Generated: {datetime.now().isoformat()}",
            f"Total runs: {len(self.results)}",
            "",
        ]
        
        # Summary table
        lines.append("SUMMARY")
        lines.append("-" * 60)
        lines.append(f"{'Model':<15} {'Runs':>5} {'Success':>8} {'Avg Time':>12} {'Speedup':>10}")
        lines.append("-" * 60)
        
        # Calculate baseline (slowest successful model)
        avg_times = {}
        for model, runs in by_model.items():
            successful = [r for r in runs if r.success]
            if successful:
                avg_times[model] = sum(r.wall_time_seconds for r in successful) / len(successful)
        
        baseline = max(avg_times.values()) if avg_times else 1
        
        for model, runs in sorted(by_model.items()):
            successful = [r for r in runs if r.success]
            success_rate = len(successful) / len(runs) * 100 if runs else 0
            
            if successful:
                avg_time = sum(r.wall_time_seconds for r in successful) / len(successful)
                speedup = baseline / avg_time if avg_time > 0 else 0
                lines.append(f"{model:<15} {len(runs):>5} {success_rate:>7.0f}% {avg_time:>10.1f}s {speedup:>9.1f}x")
            else:
                lines.append(f"{model:<15} {len(runs):>5} {success_rate:>7.0f}% {'N/A':>12} {'N/A':>10}")
        
        lines.append("-" * 60)
        lines.append("")
        
        # Detailed results
        lines.append("DETAILED RESULTS")
        lines.append("-" * 60)
        
        for model, runs in sorted(by_model.items()):
            lines.append(f"\n{model.upper()}:")
            for r in runs:
                status = "✅" if r.success else "❌"
                size_mb = r.output_size_bytes / (1024 * 1024) if r.output_size_bytes else 0
                lines.append(f"  Run {r.run_number}: {status} {r.wall_time_seconds:.1f}s, {size_mb:.1f}MB")
                if r.error:
                    lines.append(f"           Error: {r.error[:60]}")
        
        lines.append("")
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def save_results(self, filename: str = "benchmark_results.json"):
        """Save results to JSON file."""
        output_path = self.output_dir / filename
        with open(output_path, "w") as f:
            json.dump([r.to_dict() for r in self.results], f, indent=2)
        print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark 3D generation models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quick benchmark of SHARP only
    python benchmark_models.py --image test.png --models sharp
    
    # Full benchmark with 3 runs each
    python benchmark_models.py --image test.png --runs 3
    
    # Benchmark with RunPod credentials
    python benchmark_models.py --image test.png --endpoint-id xxx --api-key yyy
"""
    )
    
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--output-dir", default="/tmp/benchmark_results",
                        help="Output directory for results")
    parser.add_argument("--models", nargs="+",
                        default=["sharp", "gen3c", "lyra"],
                        choices=["sharp", "gen3c", "lyra", "hunyuan"],
                        help="Models to benchmark")
    parser.add_argument("--runs", type=int, default=1,
                        help="Number of runs per model")
    parser.add_argument("--endpoint-id", default=os.environ.get("RUNPOD_ENDPOINT_ID", ""),
                        help="RunPod endpoint ID")
    parser.add_argument("--api-key", default=os.environ.get("RUNPOD_API_KEY", ""),
                        help="RunPod API key")
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)
    
    # Run benchmark
    benchmark = ModelBenchmark(
        output_dir=args.output_dir,
        endpoint_id=args.endpoint_id,
        api_key=args.api_key,
    )
    
    benchmark.run_benchmark(
        image_path=args.image,
        models=args.models,
        num_runs=args.runs,
    )
    
    # Generate and print report
    report = benchmark.generate_report()
    print(report)
    
    # Save results
    benchmark.save_results()
    
    # Save report
    report_path = Path(args.output_dir) / "benchmark_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()

