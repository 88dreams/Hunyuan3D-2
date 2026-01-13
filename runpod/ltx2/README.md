# LTX-2 RunPod Serverless Deployment

This directory contains the RunPod serverless deployment for Lightricks' LTX-2 video generation model.

## Overview

LTX-2 is a 19B parameter DiT-based video generation model that produces high-quality videos from images with camera control.

### Features
- **Image-to-Video**: Generate videos from single images
- **Camera Control LoRAs**: Dolly left/right/in/out, Jib up, Static
- **High Resolution**: Up to 4K @ 50fps
- **Fast Inference**: Distilled model runs in 8 steps

## Prerequisites

1. **RunPod Account** with serverless access
2. **Docker Hub Account** (or your preferred registry)
3. **AWS S3 Bucket** for large file storage (optional but recommended)

## Deployment Steps

### 1. Build Docker Image

```bash
cd /home/arkrunr02/Hunyuan3D-2-Fork/runpod/ltx2

# Build the image
docker build -t 88dreams/ltx2-runpod:v1 .

# Push to Docker Hub
docker push 88dreams/ltx2-runpod:v1
```

### 2. Create RunPod Serverless Endpoint

1. Go to [RunPod Console](https://www.runpod.io/console/serverless)
2. Click "New Endpoint"
3. Configure:
   - **Name**: `ltx2-serverless`
   - **Docker Image**: `88dreams/ltx2-runpod:v1`
   - **GPU Type**: A100 40GB (recommended) or RTX A6000
   - **Max Workers**: 1-2
   - **Idle Timeout**: 60 seconds
   - **Execution Timeout**: 600 seconds

4. Add Environment Variables:
   ```
   AWS_ACCESS_KEY_ID=your_key
   AWS_SECRET_ACCESS_KEY=your_secret
   S3_BUCKET=arkrunr
   S3_REGION=us-west-1
   HF_TOKEN=your_huggingface_token (optional)
   ```

5. Attach Network Volume (for model caching):
   - Create or use existing volume
   - Mount at `/runpod-volume`

### 3. Test the Endpoint

```bash
# Health check
curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/health \
  -H "Authorization: Bearer YOUR_API_KEY"

# Generate video
curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "image_url": "https://example.com/image.jpg",
      "prompt": "A cat walking forward",
      "camera_motion": "dolly_out",
      "num_frames": 97,
      "width": 768,
      "height": 512
    }
  }'
```

## API Reference

### Input Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | string | - | Base64 encoded input image |
| `image_url` | string | - | URL to download input image |
| `prompt` | string | "" | Text prompt for video generation |
| `negative_prompt` | string | "" | Negative prompt |
| `camera_motion` | string | "none" | Camera LoRA to use |
| `num_frames` | int | 97 | Number of frames (8n+1) |
| `width` | int | 768 | Output width (divisible by 32) |
| `height` | int | 512 | Output height (divisible by 32) |
| `num_inference_steps` | int | 50 | Diffusion steps |
| `guidance_scale` | float | 7.5 | CFG scale |
| `fps` | int | 24 | Frames per second |
| `seed` | int | - | Random seed |
| `output_name` | string | auto | Output filename |
| `return_base64` | bool | false | Return video as base64 |

### Camera Motion Options

| Value | Description |
|-------|-------------|
| `none` | No camera LoRA |
| `dolly_left` | Camera moves laterally left |
| `dolly_right` | Camera moves laterally right |
| `dolly_in` | Camera pushes toward subject |
| `dolly_out` | Camera pulls away from subject |
| `jib_up` | Camera rises vertically |
| `static` | Locked camera position |

### Output

```json
{
  "status": "success",
  "video_url": "https://arkrunr.s3.us-west-1.amazonaws.com/MediaContent/outputs/ltx2/video.mp4",
  "video_path": "/runpod-volume/outputs/ltx2/video.mp4",
  "num_frames": 97,
  "duration": 4.04,
  "fps": 24,
  "width": 768,
  "height": 512,
  "camera_motion": "dolly_out"
}
```

## Hardware Requirements

| GPU | VRAM | Model Variant | Speed |
|-----|------|---------------|-------|
| A100 40GB | 40GB | Full/FP8 | Fast |
| A100 80GB | 80GB | Full | Fastest |
| RTX A6000 | 48GB | Full | Good |
| RTX 4090 | 24GB | FP8 | Medium |

## Cost Estimate

- **Cold Start**: ~60-90 seconds (model loading)
- **Inference**: ~20-60 seconds per video
- **A100 40GB**: ~$0.76/min active

## Troubleshooting

### Out of Memory
- Reduce `width` and `height`
- Reduce `num_frames`
- Use FP8 model variant

### Slow Generation
- Use distilled model (`LTX2_MODEL_VARIANT=ltx-2-19b-distilled`)
- Reduce `num_inference_steps` to 8 (for distilled model)

### Camera LoRA Not Working
- Check that the LoRA ID is valid
- Try using `camera_motion: "none"` first

## Files

| File | Description |
|------|-------------|
| `Dockerfile` | Docker image definition |
| `handler_ltx2.py` | Serverless handler |
| `start_ltx2.sh` | Container startup script |
| `README.md` | This file |

## Links

- [LTX-2 GitHub](https://github.com/Lightricks/LTX-2)
- [LTX-2 HuggingFace](https://huggingface.co/Lightricks/LTX-2)
- [Camera Control LoRAs](https://huggingface.co/collections/Lightricks/ltx-2-67af8dc2f217ccbb2f6dac81)
- [RunPod Documentation](https://docs.runpod.io/serverless)

---

*Created: January 13, 2026*
