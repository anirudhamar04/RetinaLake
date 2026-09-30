"""
Simple file type detection utilities.
Extension-based only - no content inspection.
"""

from pathlib import Path
from typing import List, Optional

# Supported file extensions
IMAGE_EXTENSIONS = [
    ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".ppm", ".bmp", ".gif",
    ".dcm", ".dicom"  # DICOM support
]

ANNOTATION_EXTENSIONS = [
    ".csv", ".json", ".jsonl", ".xlsx", ".xls", ".xml", ".txt"
]

MASK_EXTENSIONS = [
    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".ppm", ".bmp", ".npy"
]


def get_image_extensions() -> List[str]:
    """Get list of supported image file extensions."""
    return IMAGE_EXTENSIONS.copy()


def get_annotation_extensions() -> List[str]:
    """Get list of supported annotation file extensions."""
    return ANNOTATION_EXTENSIONS.copy()


def get_mask_extensions() -> List[str]:
    """Get list of supported mask file extensions."""
    return MASK_EXTENSIONS.copy()


def detect_file_type(file_path: Path) -> Optional[str]:
    """
    Detect file type from extension.
    
    Returns: "image", "annotation", "mask", or None if unknown.
    """
    if not file_path.suffix:
        return None
    
    ext = file_path.suffix.lower()
    
    if ext in IMAGE_EXTENSIONS:
        return "image"
    elif ext in ANNOTATION_EXTENSIONS:
        return "annotation"
    elif ext in MASK_EXTENSIONS:
        return "mask"
    
    return None


def is_image_file(file_path: Path) -> bool:
    """Check if file is an image based on extension."""
    if not file_path.suffix:
        return False
    return file_path.suffix.lower() in IMAGE_EXTENSIONS


def is_annotation_file(file_path: Path) -> bool:
    """Check if file is an annotation file based on extension."""
    if not file_path.suffix:
        return False
    return file_path.suffix.lower() in ANNOTATION_EXTENSIONS


def is_mask_file(file_path: Path) -> bool:
    """Check if file is a mask file based on extension."""
    if not file_path.suffix:
        return False
    return file_path.suffix.lower() in MASK_EXTENSIONS


def is_csv_file(file_path: Path) -> bool:
    """Check if file is a CSV file."""
    return file_path.suffix.lower() == ".csv"


def is_json_file(file_path: Path) -> bool:
    """Check if file is a JSON file."""
    return file_path.suffix.lower() in [".json", ".jsonl"]


def is_excel_file(file_path: Path) -> bool:
    """Check if file is an Excel file."""
    return file_path.suffix.lower() in [".xlsx", ".xls"]


def is_xml_file(file_path: Path) -> bool:
    """Check if file is an XML file."""
    return file_path.suffix.lower() == ".xml"
