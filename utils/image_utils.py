"""
Image Utilities for 3D Generation Studio

This module contains image processing utilities including:
- Image scaling/downscaling
- Dimension extraction
- Resolution display formatting
"""

import os
import tempfile
from typing import Optional, Tuple, Union

from PIL import Image


def clamp_scale_value(value: Union[float, int, None]) -> float:
    """Clamp scale value to valid range [0.25, 1.0]."""
    try:
        val = float(value)
    except (TypeError, ValueError):
        return 1.0
    return min(1.0, max(0.25, val))


def get_image_dimensions(image_path: Optional[str]) -> Tuple[Optional[int], Optional[int]]:
    """Get width and height of an image file."""
    if not image_path or not os.path.exists(image_path):
        return None, None
    try:
        with Image.open(image_path) as img:
            return img.width, img.height
    except Exception:
        return None, None


def format_resize_text(
    width: Optional[int],
    height: Optional[int],
    scale: Union[float, int, None],
) -> str:
    """Format resolution information for display."""
    if width is None or height is None:
        return "No image loaded."
    scale = clamp_scale_value(scale)
    target_w = max(64, int(round(width * scale)))
    target_h = max(64, int(round(height * scale)))
    percent = int(round(scale * 100))
    return f"Original: {width}×{height}\nTarget ({percent}%): {target_w}×{target_h}"


def update_image_info_display(image_path: Optional[str], scale: Union[float, int, None]) -> str:
    """Update image info display with current dimensions and scale."""
    width, height = get_image_dimensions(image_path)
    return format_resize_text(width, height, scale or 1.0)


def maybe_downscale_image(
    image_path: Optional[str], 
    scale: Union[float, int, None]
) -> Tuple[Optional[str], Optional[str]]:
    """
    Downscale image if scale < 1.0.
    
    Returns:
        Tuple of (effective_path, temp_path_to_cleanup)
        - effective_path: Path to use for processing (original or scaled)
        - temp_path_to_cleanup: Path to delete after processing (or None)
    """
    scale = clamp_scale_value(scale)
    if not image_path or scale >= 0.999:
        return image_path, None
    
    try:
        with Image.open(image_path) as img:
            target_w = max(64, int(round(img.width * scale)))
            target_h = max(64, int(round(img.height * scale)))
            
            if target_w == img.width and target_h == img.height:
                return image_path, None
            
            # Get best resampling filter
            resample_attr = getattr(Image, "Resampling", None)
            resample_filter = resample_attr.LANCZOS if resample_attr else Image.LANCZOS
            
            resized = img.resize((target_w, target_h), resample_filter)
            
            suffix = os.path.splitext(image_path)[1] or ".png"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            resized.save(tmp.name)
            tmp_path = tmp.name
            tmp.close()
            
            return tmp_path, tmp_path
    except Exception:
        return image_path, None


def load_image(image_path: str) -> Image.Image:
    """Load an image and convert to RGB."""
    img = Image.open(image_path)
    return img.convert("RGB")

