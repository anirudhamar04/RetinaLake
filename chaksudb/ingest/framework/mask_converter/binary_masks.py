"""Binary mask operations for validation, extraction, and multi-class handling."""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from PIL import Image as PILImage

logger = logging.getLogger(__name__)


def is_multiclass_mask(mask_path: Path) -> bool:
    """
    Check if a mask file contains multiple classes (more than 2 unique values).
    
    Args:
        mask_path: Path to mask file
        
    Returns:
        True if mask has more than 2 unique values (multi-class), False otherwise
        
    Raises:
        FileNotFoundError: If mask file does not exist
        ValueError: If mask cannot be read
    """
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask file not found: {mask_path}")
    
    # Read mask
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        try:
            with PILImage.open(mask_path) as img:
                mask = np.array(img.convert("L"))
        except Exception as e:
            raise ValueError(f"Failed to read mask file {mask_path}: {e}")
    
    unique_values = np.unique(mask)
    return len(unique_values) > 2


def get_mask_classes(mask_path: Path) -> np.ndarray:
    """
    Get all unique class values in a mask, excluding background (0).
    
    Args:
        mask_path: Path to mask file
        
    Returns:
        Array of unique non-zero class values
        
    Raises:
        FileNotFoundError: If mask file does not exist
        ValueError: If mask cannot be read
    """
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask file not found: {mask_path}")
    
    # Read mask
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        try:
            with PILImage.open(mask_path) as img:
                mask = np.array(img.convert("L"))
        except Exception as e:
            raise ValueError(f"Failed to read mask file {mask_path}: {e}")
    
    unique_values = np.unique(mask)
    # Return all non-zero values
    return unique_values[unique_values > 0]


def extract_class_from_mask(
    mask_path: Path,
    class_id: int,
    output_value: int = 255,
    fill_holes: bool = False
) -> np.ndarray:
    """
    Extract a specific class from a multi-class mask as a binary mask.
    
    Args:
        mask_path: Path to mask file (can be binary or multi-class)
        class_id: Class ID to extract (e.g., 1 for optic disc, 2 for cup)
        output_value: Value to use for extracted class pixels (default: 255)
        fill_holes: If True, fills any holes in the extracted mask using 
                    morphological operations. Useful for disc/cup masks that
                    may only have boundary pixels.
        
    Returns:
        Binary mask where pixels of class_id are set to output_value, others to 0
        
    Raises:
        FileNotFoundError: If mask file does not exist
        ValueError: If mask cannot be read or class_id not found
    """
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask file not found: {mask_path}")
    
    # Read mask
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        try:
            with PILImage.open(mask_path) as img:
                mask = np.array(img.convert("L"))
        except Exception as e:
            raise ValueError(f"Failed to read mask file {mask_path}: {e}")
    
    # Check if class exists
    if class_id not in mask:
        raise ValueError(
            f"Class ID {class_id} not found in mask. "
            f"Available classes: {np.unique(mask)}"
        )
    
    # Extract class as binary mask
    binary_mask = np.where(mask == class_id, output_value, 0).astype(np.uint8)
    
    # Fill holes if requested
    if fill_holes:
        binary_mask = _fill_mask_holes(binary_mask)
    
    return binary_mask


def _fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    """
    Fill holes in a binary mask using contour-based filling.
    
    Args:
        mask: Binary mask (uint8, 0 and 255)
        
    Returns:
        Binary mask with holes filled
    """
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create filled mask
    filled_mask = np.zeros_like(mask)
    
    # Fill each contour
    for contour in contours:
        cv2.drawContours(filled_mask, [contour], -1, 255, -1)
    
    return filled_mask


def extract_classes_from_multiclass_mask(
    mask_path: Path,
    class_names: Optional[Dict[int, str]] = None,
    classes_to_extract: Optional[List[int]] = None,
    merge_classes: Optional[List[int]] = None,
    fill_holes: bool = False
) -> Dict[str, np.ndarray]:
    """
    Extract multiple classes from a multi-class mask into separate binary masks.
    
    Args:
        mask_path: Path to multi-class mask file
        class_names: Optional mapping of class IDs to names 
                     (e.g., {1: "optic_disc", 2: "cup"})
        classes_to_extract: Optional list of class IDs to extract. 
                           If None, extracts all non-zero classes.
        merge_classes: Optional list of class IDs to merge into single foreground.
                      If provided, returns single "merged" mask instead of individual.
        fill_holes: If True, fills any holes in the extracted masks using 
                    morphological operations.
        
    Returns:
        Dictionary mapping class names/IDs to binary masks (uint8, 0 and 255)
        If merge_classes is provided, returns {"merged": binary_mask}
        
    Raises:
        FileNotFoundError: If mask file does not exist
        ValueError: If mask cannot be read
        
    Example:
        # Extract specific classes with names
        masks = extract_classes_from_multiclass_mask(
            path,
            class_names={1: "optic_disc", 2: "cup"},
            classes_to_extract=[1, 2]
        )
        # Returns: {"optic_disc": mask1, "cup": mask2}
        
        # Merge multiple classes
        masks = extract_classes_from_multiclass_mask(
            path,
            merge_classes=[1, 2]
        )
        # Returns: {"merged": combined_mask}
        
        # Extract with hole filling
        masks = extract_classes_from_multiclass_mask(
            path,
            class_names={1: "optic_disc"},
            fill_holes=True
        )
        # Returns: {"optic_disc": filled_mask}
    """
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask file not found: {mask_path}")
    
    # Read mask
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        try:
            with PILImage.open(mask_path) as img:
                mask = np.array(img.convert("L"))
        except Exception as e:
            raise ValueError(f"Failed to read mask file {mask_path}: {e}")
    
    # If merge_classes is specified, create merged binary mask
    if merge_classes is not None:
        merged_mask = np.zeros_like(mask)
        for class_id in merge_classes:
            if class_id in mask:
                merged_mask = np.where(mask == class_id, 255, merged_mask)
        if fill_holes:
            merged_mask = _fill_mask_holes(merged_mask.astype(np.uint8))
        return {"merged": merged_mask.astype(np.uint8)}
    
    # Determine which classes to extract
    available_classes = get_mask_classes(mask_path)
    if classes_to_extract is None:
        classes_to_extract = available_classes.tolist()
    
    # Extract each class
    result = {}
    for class_id in classes_to_extract:
        if class_id not in available_classes:
            logger.warning(
                f"Class ID {class_id} not found in mask {mask_path}. "
                f"Available: {available_classes}"
            )
            continue
        
        # Extract binary mask for this class
        binary_mask = np.where(mask == class_id, 255, 0).astype(np.uint8)
        
        # Fill holes if requested
        if fill_holes:
            binary_mask = _fill_mask_holes(binary_mask)
        
        # Use provided name or default to class_id
        if class_names and class_id in class_names:
            key = class_names[class_id]
        else:
            key = f"class_{class_id}"
        
        result[key] = binary_mask
    
    return result


def validate_binary_mask(
    mask_path: Path,
    extract_class: Optional[int] = None,
    merge_nonzero: bool = False,
    fill_holes: bool = False
) -> np.ndarray:
    """
    Validate and load binary mask. Returns the mask array directly.

    Binary masks are stored as-is (no conversion needed).
    Validates that the mask is binary (0 and 255, or 0 and 1).
    
    For multi-class masks, can extract a specific class or merge all non-zero classes.

    Args:
        mask_path: Path to binary mask file (PNG, TIF, JPG, etc.)
        extract_class: Optional class ID to extract from multi-class mask.
                       If provided and mask is multi-class, extracts only this class.
        merge_nonzero: If True and mask is multi-class, merges all non-zero 
                       values into foreground (255). Useful for ORIGA-style masks
                       with disc=1 and cup=2.
        fill_holes: If True, fills any holes in the mask using morphological
                    operations. Useful for disc/cup masks that may only have
                    boundary pixels.

    Returns:
        Binary mask as numpy array (uint8, 0 and 255)

    Raises:
        FileNotFoundError: If mask file does not exist
        ValueError: If mask cannot be read or is invalid
        
    Example:
        # Load binary mask (original behavior)
        mask = validate_binary_mask(path)
        
        # Extract optic disc (class 1) from ORIGA multi-class mask
        disc_mask = validate_binary_mask(path, extract_class=1)
        
        # Merge optic disc and cup into single foreground
        combined_mask = validate_binary_mask(path, merge_nonzero=True)
        
        # Extract disc with hole filling
        disc_filled = validate_binary_mask(path, extract_class=1, fill_holes=True)
    """
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask file not found: {mask_path}")

    # Read mask using OpenCV
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        # Try with PIL as fallback
        try:
            with PILImage.open(mask_path) as img:
                mask = np.array(img.convert("L"))
        except Exception as e:
            raise ValueError(f"Failed to read mask file {mask_path}: {e}")

    # Validate and normalize binary mask
    unique_values = np.unique(mask)
    
    if len(unique_values) > 2:
        # Multi-class mask detected
        
        # Option 1: Extract specific class
        if extract_class is not None:
            if extract_class not in unique_values:
                raise ValueError(
                    f"Class ID {extract_class} not found in mask. "
                    f"Available classes: {unique_values}"
                )
            mask = np.where(mask == extract_class, 255, 0).astype(np.uint8)
            logger.debug(
                f"Extracted class {extract_class} from multi-class mask {mask_path}"
            )
        
        # Option 2: Merge all non-zero classes
        elif merge_nonzero:
            mask = np.where(mask > 0, 255, 0).astype(np.uint8)
            logger.debug(
                f"Merged {len(unique_values)-1} non-zero classes from {mask_path}"
            )
        
        # Option 3: Threshold (default behavior for backwards compatibility)
        else:
            # Normalize if values are 0-1 range
            if set(unique_values).issubset({0, 1}):
                mask = (mask * 255).astype(np.uint8)
            else:
                logger.warning(
                    f"Mask {mask_path} contains non-binary values: {unique_values}. "
                    "Thresholding to binary. Use extract_class= or merge_nonzero=True "
                    "for multi-class masks."
                )
                # Threshold to binary: values > 127 become 255, else 0
                mask = np.where(mask > 127, 255, 0).astype(np.uint8)
    else:
        # Binary mask - ensure uint8 type and normalize to 0/255
        if set(unique_values).issubset({0, 1}):
            mask = (mask * 255).astype(np.uint8)
        else:
            mask = mask.astype(np.uint8)
    
    # Fill holes if requested
    if fill_holes:
        mask = _fill_mask_holes(mask)

    return mask
