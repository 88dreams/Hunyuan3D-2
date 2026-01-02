# Why SHARP is 10-20x Faster Than Lyra/GEN3C

## Executive Summary

Your observation is correct and architecturally expected:

| Model | Time | Architecture | Why |
|-------|------|--------------|-----|
| **SHARP** | ~60-120s | **Feed-Forward** | Single neural network pass |
| **GEN3C** | ~10-15 min | **Diffusion** | 50-100+ iterative denoising steps |
| **Lyra** | ~15-20 min | **Diffusion + 3DGS** | GEN3C video + 3DGS reconstruction |

**SHARP is fundamentally a different type of model** - it's not doing the same thing faster, it's doing something architecturally simpler.

---

## Architectural Comparison

### SHARP: Feed-Forward Network (Single Pass)

```
Input Image → Encoder → Decoder → 3D Gaussians
                   ↓
              ONE forward pass
              ~1 second for PLY
              ~60s with video rendering
```

**Key characteristics:**
- **One-shot prediction**: The network directly predicts all Gaussian parameters in a single forward pass
- **No iterative refinement**: No denoising loops
- **Trained end-to-end**: Learns to map images → 3DGS directly
- **Inference complexity**: O(1) - constant time regardless of quality

**Reference**: [Splatter Image](https://arxiv.org/abs/2312.13150) - "Ultra-fast approach for monocular 3D object reconstruction which operates at 38 FPS"

### GEN3C/Lyra: Diffusion Model (Iterative)

```
Input Image → Depth Estimation → 3D Cache → 
    ↓
Video Diffusion Model (50-100 steps)
    ↓
Each step: Full U-Net forward pass
    ↓
Multi-view Video (121 frames)
    ↓
[Lyra only] 3DGS Reconstruction Decoder
    ↓
Output
```

**Key characteristics:**
- **Iterative denoising**: Each step requires a full neural network forward pass
- **50-100 steps typical**: 50-100x more compute than feed-forward
- **Higher quality ceiling**: Can generate novel content, not just reconstruct
- **Inference complexity**: O(n) where n = number of diffusion steps

**Reference**: [GEN3C Paper](https://arxiv.org/abs/2503.03751) - "guided by a 3D cache: point clouds obtained by predicting the pixel-wise depth"

---

## Why The Quality/Speed Tradeoff Exists

### Feed-Forward (SHARP)
- ✅ **Fast**: Single forward pass
- ✅ **Deterministic**: Same input → same output
- ❌ **Limited generalization**: Can only reconstruct what it learned
- ❌ **No novel content**: Cannot hallucinate unseen regions well
- ❌ **Training-data bound**: Quality limited by training distribution

### Diffusion (GEN3C/Lyra)
- ❌ **Slow**: 50-100 iterative steps
- ✅ **High quality**: Can generate photorealistic novel views
- ✅ **Novel content**: Can hallucinate plausible unseen regions
- ✅ **Flexible**: Works on diverse scenes
- ✅ **State-of-the-art**: Best visual quality currently achievable

---

## Compute Breakdown

### SHARP (~60-120 seconds total)
| Step | Time | Notes |
|------|------|-------|
| Image encoding | ~0.1s | Single encoder pass |
| Gaussian prediction | ~0.5s | Feed-forward decoder |
| PLY export | ~0.1s | File I/O |
| **Video rendering** | ~60-90s | Optional, CUDA rasterization |
| **Total (PLY only)** | **<1s** | |
| **Total (with video)** | **~60-120s** | |

### GEN3C (~10-15 minutes)
| Step | Time | Notes |
|------|------|-------|
| Depth estimation (MoGe) | ~30s | Single forward pass |
| 3D cache creation | ~10s | Point cloud from depth |
| **Video diffusion** | **~8-12 min** | 50-100 U-Net passes × 121 frames |
| Video encoding | ~30s | MP4 export |
| **Total** | **~10-15 min** | |

### Lyra (~15-20 minutes)
| Step | Time | Notes |
|------|------|-------|
| GEN3C video generation | ~10-15 min | Same as above |
| **3DGS reconstruction** | **~5-8 min** | Decoder + optimization |
| PLY export | ~10s | File I/O |
| **Total** | **~15-20 min** | |

---

## Mathematical Comparison

### Feed-Forward Complexity
```
Time = T_encoder + T_decoder
     ≈ O(1) constant
     ≈ 0.5-1 second
```

### Diffusion Complexity
```
Time = T_depth + T_cache + N_steps × T_unet × N_frames
     = 30s + 10s + 100 × 0.5s × 121
     ≈ 6050 seconds (theoretical)
     ≈ 10-15 minutes (optimized with batching)
```

The **100x difference** comes from the iterative nature of diffusion.

---

## Quality vs Speed: What You're Trading

| Aspect | SHARP | GEN3C/Lyra |
|--------|-------|------------|
| **Speed** | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **Novel view synthesis** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Geometric accuracy** | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Hallucination quality** | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Multi-view consistency** | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Handles occlusions** | ⭐⭐ | ⭐⭐⭐⭐ |

---

## When to Use Each

### Use SHARP when:
- ✅ Speed is critical
- ✅ Object-centric scenes (not interiors)
- ✅ Quick previews/prototyping
- ✅ Batch processing many images
- ✅ Real-time applications

### Use GEN3C/Lyra when:
- ✅ Quality is paramount
- ✅ Complex scenes with occlusions
- ✅ Need to hallucinate unseen regions
- ✅ Interior/architectural scenes
- ✅ Final production output

---

## Benchmark Protocol

To properly compare these models, you should measure:

### 1. Speed Metrics
- **Wall-clock time**: Total time from input to output
- **GPU utilization**: % GPU used during inference
- **Memory usage**: Peak VRAM consumption
- **Cold start vs warm start**: First run vs cached run

### 2. Quality Metrics
- **PSNR**: Peak Signal-to-Noise Ratio (pixel accuracy)
- **SSIM**: Structural Similarity Index (perceptual structure)
- **LPIPS**: Learned Perceptual Image Patch Similarity (perceptual quality)
- **Chamfer Distance**: 3D geometric accuracy (if ground truth available)

### 3. Consistency Metrics
- **Multi-view consistency**: How stable across viewpoints
- **Temporal consistency**: For video outputs

---

## Conclusion

**SHARP is not "better" - it's fundamentally different.**

The 10-20x speed advantage comes from:
1. **Single forward pass** vs **50-100 iterative steps**
2. **Direct prediction** vs **iterative refinement**
3. **Simpler task** (reconstruction) vs **harder task** (generation)

For your ArkRunr use case (architectural interiors), the quality difference may matter more than speed. But for rapid prototyping or batch processing, SHARP's speed is invaluable.

---

## References

1. [Splatter Image: Ultra-Fast Single-View 3D Reconstruction](https://arxiv.org/abs/2312.13150)
2. [GEN3C: 3D-Informed World-Consistent Video Generation](https://arxiv.org/abs/2503.03751)
3. [SHARP: Sharp Monocular View Synthesis](https://arxiv.org/abs/2512.10685)

