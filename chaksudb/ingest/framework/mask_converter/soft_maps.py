"""Soft map (probability map) loading utilities."""

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image as PILImage


def load_soft_map(soft_map_path: Path, image_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Load soft map (probability map) as-is. Returns probability values, not binary.

    Soft maps should be stored with unified_format="probability_map".
    This function does NOT convert to binary - it returns the probability values.

    Supports:
    - Image files (PNG, JPG, TIFF) with grayscale probability values (0-255, normalized to 0-1)
    - NumPy arrays (.npy, .npz files) with probability values (0-1 range)

    Args:
        soft_map_path: Path to soft map file
        image_size: Optional (width, height) to resize. If None, uses original size.

    Returns:
        Probability map as numpy array (float32, 0.0-1.0 range)

    Raises:
        FileNotFoundError: If soft map file does not exist
        ValueError: If file cannot be read
    """
    if not soft_map_path.exists():
        raise FileNotFoundError(f"Soft map file not found: {soft_map_path}")

    # Load soft map based on file extension
    ext = soft_map_path.suffix.lower()

    if ext in [".npy", ".npz"]:
        # NumPy array file
        if ext == ".npy":
            soft_map = np.load(soft_map_path)
        else:  # .npz
            data = np.load(soft_map_path)
            # Try common keys, or use first array
            if len(data.files) == 1:
                soft_map = data[data.files[0]]
            else:
                # Try common keys
                for key in ["mask", "prob", "probability", "soft_map"]:
                    if key in data:
                        soft_map = data[key]
                        break
                else:
                    soft_map = data[data.files[0]]
    else:
        # Image file
        img = cv2.imread(str(soft_map_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Try with PIL
            with PILImage.open(soft_map_path) as pil_img:
                img = np.array(pil_img.convert("L"))
        soft_map = img.astype(np.float32) / 255.0  # Normalize to 0-1

    # Ensure values are in 0-1 range
    if soft_map.max() > 1.0:
        soft_map = soft_map.astype(np.float32) / 255.0

    # Resize if needed
    if image_size is not None:
        width, height = image_size
        if soft_map.shape[1] != width or soft_map.shape[0] != height:
            soft_map = cv2.resize(soft_map, (width, height), interpolation=cv2.INTER_LINEAR)

    return soft_map.astype(np.float32)
