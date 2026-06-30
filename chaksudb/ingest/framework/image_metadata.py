"""
Image metadata extraction utilities.

Uses OpenCV and Pillow to extract image metadata including resolution,
format, color channels, file size, and EXIF data.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
from PIL import Image as PILImage
from PIL.ExifTags import TAGS

from chaksudb.config.config import constants

logger = logging.getLogger(__name__)


@dataclass
class ImageMetadata:
    """Image metadata extracted from image file."""

    file_path: Path
    file_format: Optional[str] = None
    resolution_width: Optional[int] = None
    resolution_height: Optional[int] = None
    color_channels: Optional[int] = None
    file_size: Optional[int] = None
    exif_data: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "file_path": str(self.file_path),
            "file_format": self.file_format,
            "resolution_width": self.resolution_width,
            "resolution_height": self.resolution_height,
            "color_channels": self.color_channels,
            "file_size": self.file_size,
            "exif_data": self.exif_data,
        }


def _normalize_file_format(extension: str) -> Optional[str]:
    """
    Normalize file extension to format string matching schema constraints.
    
    Handles all extensions from constants.IMAGE_EXTENSIONS including both
    lowercase and uppercase variants (.jpg, .JPG, etc.).
    
    Args:
        extension: File extension (e.g., ".jpg", ".JPG", ".jpeg", ".png", ".bmp")
        
    Returns:
        Normalized format string (e.g., "jpg", "jpeg", "png") or None if unsupported.
        Returns None for extensions that are in IMAGE_EXTENSIONS but not in FILE_FORMATS
        (e.g., .bmp is supported for reading but not in schema).
    """
    # Normalize to lowercase to handle both .jpg and .JPG, .BMP, etc.
    ext_lower = extension.lower()
    
    # First check if extension is in IMAGE_EXTENSIONS (handles all case variants)
    if ext_lower not in {ext.lower() for ext in constants.IMAGE_EXTENSIONS}:
        return None
    
    # Map extensions to format strings matching schema constraints
    # Only include formats that are in FILE_FORMATS
    format_map = {
        ".jpg": "jpg",
        ".jpeg": "jpeg",
        ".png": "png",
        ".tif": "tif",
        ".tiff": "tiff",
        ".ppm": "ppm",
        ".dcm": "dicom",
        ".dicom": "dicom",
    }
    
    if ext_lower in format_map:
        return format_map[ext_lower]
    
    # Check if the format (without dot) is in FILE_FORMATS
    format_str = ext_lower.lstrip(".")
    if format_str in constants.FILE_FORMATS:
        return format_str
    
    # Extension is in IMAGE_EXTENSIONS but format not in FILE_FORMATS
    # (e.g., .bmp, .BMP) - return None as it's not schema-compliant
    return None


def _extract_exif_data(image_path: Path) -> Optional[Dict[str, Any]]:
    """
    Extract EXIF data from image using Pillow.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Dictionary of EXIF tags and values, or None if no EXIF data
    """
    try:
        with PILImage.open(image_path) as img:
            exif = img.getexif()
            
            if exif is None:
                return None
            
            exif_dict = {}
            for tag_id, value in exif.items():
                tag = TAGS.get(tag_id, tag_id)
                exif_dict[tag] = value
            
            return exif_dict if exif_dict else None
    except Exception as e:
        logger.debug(f"Failed to extract EXIF data from {image_path}: {e}")
        return None


def extract_image_metadata(file_path: Path) -> ImageMetadata:
    """
    Extract image metadata using OpenCV and Pillow.
    
    Extracts:
    - File format (from extension, normalized to schema format)
    - Resolution (width, height) from image data
    - Color channels (from image shape)
    - File size (from filesystem)
    - Basic EXIF data (if available, using Pillow)
    
    Note: The original file_path is preserved with its exact case for all file
    operations (important for case-sensitive filesystems like Linux). Only the
    extension is normalized to lowercase for format detection.
    
    Args:
        file_path: Path to image file (case-sensitive, preserved as-is)
        
    Returns:
        ImageMetadata object with extracted metadata
        
    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If file cannot be read as an image
    """
    # All file operations use the original file_path (preserves case)
    # This is critical for case-sensitive filesystems (Linux)
    if not file_path.exists():
        raise FileNotFoundError(f"Image file not found: {file_path}")
    
    # Get file size using original path
    file_size = file_path.stat().st_size
    
    # Get file format from extension (normalize extension to lowercase for format detection)
    # The file_path itself remains unchanged with original case
    file_format = _normalize_file_format(file_path.suffix)
    
    # Extract resolution and color channels using OpenCV
    resolution_width = None
    resolution_height = None
    color_channels = None
    
    try:
        # Read image with OpenCV using original file_path (preserves case)
        img = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
        
        if img is None:
            # Try with Pillow as fallback for formats OpenCV might not support
            # Still using original file_path (preserves case)
            try:
                with PILImage.open(file_path) as pil_img:
                    width, height = pil_img.size
                    resolution_width = width
                    resolution_height = height
                    
                    # Get color channels from mode
                    mode = pil_img.mode
                    if mode in ("L", "LA"):
                        color_channels = 1
                    elif mode in ("RGB", "RGBA"):
                        color_channels = 3
                    elif mode == "CMYK":
                        color_channels = 4
                    else:
                        # Try to convert to RGB to get channels
                        rgb_img = pil_img.convert("RGB")
                        color_channels = 3
            except Exception as e:
                raise ValueError(f"Failed to read image {file_path}: {e}")
        else:
            # OpenCV successfully read the image
            height, width = img.shape[:2]
            resolution_width = width
            resolution_height = height
            
            # Determine color channels from image shape
            if len(img.shape) == 2:
                # Grayscale
                color_channels = 1
            elif len(img.shape) == 3:
                # Color image
                color_channels = img.shape[2]
            else:
                color_channels = None
                
    except Exception as e:
        logger.warning(f"Failed to extract image dimensions from {file_path}: {e}")
        # Continue with None values for resolution/channels
    
    # Extract EXIF data using Pillow (using original file_path, preserves case)
    exif_data = _extract_exif_data(file_path)
    
    return ImageMetadata(
        file_path=file_path,
        file_format=file_format,
        resolution_width=resolution_width,
        resolution_height=resolution_height,
        color_channels=color_channels,
        file_size=file_size,
        exif_data=exif_data,
    )
