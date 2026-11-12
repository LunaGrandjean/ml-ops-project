"""Même transformation en entraînement et en inférence."""

import numpy as np
from PIL import Image, ImageOps


def image_array(image: Image.Image, size: int) -> np.ndarray:
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = image.resize((size, size), Image.Resampling.BILINEAR)
    return np.ascontiguousarray(np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255)
