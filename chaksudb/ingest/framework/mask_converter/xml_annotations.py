"""Unified XML annotation parser supporting multiple formats."""


import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def parse_xml_annotations(
    xml_path: Path,
    image_size: Tuple[int, int],
    output_format: str = "masks_by_class",
    class_filter: Optional[Union[str, List[str]]] = None,
    **kwargs
) -> Union[np.ndarray, Dict[str, np.ndarray], List[Dict[str, Any]]]:
    """
    Universal XML annotation parser with auto-detection.
    
    Automatically detects and parses:
    - PASCAL VOC bounding boxes (OIA-DDR, SUSTech-SYSU)
    - ImageRet circle regions (DiaRetDB1)
    - Polygon coordinates (custom formats)
    
    Args:
        xml_path: Path to XML annotation file
        image_size: (width, height) of target image
        output_format: Output format - "masks_by_class", "mask", or "annotations"
            - "masks_by_class": Dict of {class_name: binary_mask} (DEFAULT)
            - "mask": Single binary mask with all annotations combined
            - "annotations": List of annotation dictionaries (bboxes, circles, etc.)
        class_filter: Optional class name(s) to filter. If None, includes all.
                     Can be string or list of strings.
        **kwargs: Additional format-specific options
            - confidence_threshold: For DiaRetDB1 (High/Medium/Low)
            - fill_circles: For DiaRetDB1, whether to fill circles (default: True)
            - fill_bboxes: For bounding boxes, whether to fill (default: True)
    
    Returns:
        Depending on output_format:
        - "masks_by_class": Dict mapping class names to binary masks (DEFAULT)
        - "mask": Single combined binary mask (uint8, 0 and 255)
        - "annotations": List of dicts with annotation data
    
    Raises:
        FileNotFoundError: If XML file does not exist
        ValueError: If XML cannot be parsed or format is unknown
        
    Example:
        # Default: Get separate masks per class
        masks = parse_xml_annotations(xml_path, (1152, 1500))
        # Returns: {"Haemorrhages": mask1, "Hard_exudates": mask2, ...}
        
        # Access specific class
        hemorrhages_mask = masks.get("Haemorrhages")
        
        # Get single combined mask
        mask = parse_xml_annotations(
            xml_path, (1152, 1500),
            output_format="mask"
        )
        
        # Get only specific lesion types
        masks = parse_xml_annotations(
            xml_path, (1152, 1500),
            class_filter=["Haemorrhages", "Hard_exudates"]
        )
        
        # Get raw annotation data
        annotations = parse_xml_annotations(
            xml_path, (1152, 1500),
            output_format="annotations"
        )
    """
    if not xml_path.exists():
        raise FileNotFoundError(f"XML file not found: {xml_path}")
    
    # Parse XML
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        raise ValueError(f"Failed to parse XML file {xml_path}: {e}")
    
    # Auto-detect format
    xml_format = _detect_xml_format(root)
    
    # Parse based on detected format
    if xml_format == "imageret_circles":
        return _parse_imageret_circles(
            root, xml_path, image_size, output_format, class_filter, **kwargs
        )
    elif xml_format == "pascal_voc":
        return _parse_pascal_voc_bboxes(
            root, xml_path, image_size, output_format, class_filter, **kwargs
        )
    elif xml_format == "polygon":
        return _parse_polygon_coordinates(
            root, xml_path, image_size, output_format, class_filter, **kwargs
        )
    else:
        raise ValueError(
            f"Unknown XML annotation format in {xml_path}. "
            f"Supported: PASCAL VOC bboxes, ImageRet circles, polygon coordinates."
        )


def _detect_xml_format(root: ET.Element) -> str:
    """
    Auto-detect XML annotation format.
    
    Returns:
        "imageret_circles" - DiaRetDB1 style circle annotations
        "pascal_voc" - PASCAL VOC style bounding boxes
        "polygon" - Polygon coordinate annotations
    """
    # Check for ImageRet circle format (DiaRetDB1)
    if root.find(".//circleregion") is not None:
        return "imageret_circles"
    
    # Check for PASCAL VOC format (bndbox)
    if root.find(".//bndbox") is not None:
        return "pascal_voc"
    
    # Check for polygon format
    if root.find(".//polygon") is not None:
        return "polygon"
    
    # If annotation tag with object/bndbox
    if root.tag == "annotation" and root.find(".//object") is not None:
        return "pascal_voc"
    
    # If imgannotooldata with markinglist
    if root.tag == "imgannotooldata" and root.find(".//markinglist") is not None:
        return "imageret_circles"
    
    return "unknown"


def _parse_imageret_circles(
    root: ET.Element,
    xml_path: Path,
    image_size: Tuple[int, int],
    output_format: str,
    class_filter: Optional[Union[str, List[str]]],
    **kwargs
) -> Union[np.ndarray, Dict[str, np.ndarray], List[Dict[str, Any]]]:
    """Parse ImageRet circle region annotations (DiaRetDB1 format)."""
    width, height = image_size
    confidence_threshold = kwargs.get("confidence_threshold", None)
    fill_circles = kwargs.get("fill_circles", True)
    
    # Normalize class filter
    if isinstance(class_filter, str):
        class_filter = [class_filter]
    
    annotations = []
    
    # Extract all markings
    for marking in root.findall(".//marking"):
        circle_region = marking.find(".//circleregion")
        if circle_region is None:
            continue
        
        # Extract centroid
        centroid = circle_region.find(".//centroid/coords2d")
        if centroid is None:
            continue
        coords = centroid.text.strip().split(",")
        if len(coords) != 2:
            continue
        cx, cy = int(coords[0]), int(coords[1])
        
        # Extract radius
        radius_elem = circle_region.find(".//radius")
        if radius_elem is None:
            continue
        radius = int(radius_elem.text.strip())
        
        # Extract marking type (class name)
        marking_type_elem = marking.find(".//markingtype")
        marking_type = marking_type_elem.text.strip() if marking_type_elem is not None else "unknown"
        
        # Extract confidence level
        confidence_elem = marking.find(".//confidencelevel")
        confidence = confidence_elem.text.strip() if confidence_elem is not None else "Unknown"
        
        # Filter by confidence if specified
        if confidence_threshold:
            levels = {"High": 3, "Medium": 2, "Low": 1, "Unknown": 0}
            threshold_val = levels.get(confidence_threshold, 0)
            current_val = levels.get(confidence, 0)
            if current_val < threshold_val:
                continue
        
        # Filter by class
        if class_filter and marking_type not in class_filter:
            continue
        
        annotations.append({
            "type": "circle",
            "class": marking_type,
            "centroid": (cx, cy),
            "radius": radius,
            "confidence": confidence,
        })
    
    if output_format == "annotations":
        return annotations
    
    # Create masks
    if output_format == "masks_by_class":
        masks = {}
        for ann in annotations:
            class_name = ann["class"]
            if class_name not in masks:
                masks[class_name] = np.zeros((height, width), dtype=np.uint8)
            
            thickness = -1 if fill_circles else 2  # -1 fills, positive draws outline
            cv2.circle(masks[class_name], ann["centroid"], ann["radius"], 255, thickness)
        
        return masks
    
    else:  # output_format == "mask"
        mask = np.zeros((height, width), dtype=np.uint8)
        for ann in annotations:
            thickness = -1 if fill_circles else 2
            cv2.circle(mask, ann["centroid"], ann["radius"], 255, thickness)
        
        return mask


def _parse_pascal_voc_bboxes(
    root: ET.Element,
    xml_path: Path,
    image_size: Tuple[int, int],
    output_format: str,
    class_filter: Optional[Union[str, List[str]]],
    **kwargs
) -> Union[np.ndarray, Dict[str, np.ndarray], List[Dict[str, Any]]]:
    """Parse PASCAL VOC bounding box annotations."""
    width, height = image_size
    fill_bboxes = kwargs.get("fill_bboxes", True)
    
    # Normalize class filter
    if isinstance(class_filter, str):
        class_filter = [class_filter]
    
    annotations = []
    
    # Extract all objects
    for obj in root.findall(".//object"):
        name_elem = obj.find("name")
        if name_elem is None:
            continue
        class_name = name_elem.text.strip()
        
        # Filter by class
        if class_filter and class_name not in class_filter:
            continue
        
        # Extract bounding box
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue
        
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)
        
        # Clip to image bounds
        xmin = max(0, min(xmin, width - 1))
        ymin = max(0, min(ymin, height - 1))
        xmax = max(0, min(xmax, width - 1))
        ymax = max(0, min(ymax, height - 1))
        
        # Extract optional fields
        truncated_elem = obj.find("truncated")
        truncated = int(truncated_elem.text) if truncated_elem is not None else 0
        
        difficult_elem = obj.find("difficult")
        difficult = int(difficult_elem.text) if difficult_elem is not None else 0
        
        annotations.append({
            "type": "bbox",
            "class": class_name,
            "bbox": (xmin, ymin, xmax, ymax),
            "truncated": bool(truncated),
            "difficult": bool(difficult),
        })
    
    if output_format == "annotations":
        return annotations
    
    # Create masks
    if output_format == "masks_by_class":
        masks = {}
        for ann in annotations:
            class_name = ann["class"]
            if class_name not in masks:
                masks[class_name] = np.zeros((height, width), dtype=np.uint8)
            
            xmin, ymin, xmax, ymax = ann["bbox"]
            if fill_bboxes:
                cv2.rectangle(masks[class_name], (xmin, ymin), (xmax, ymax), 255, -1)
            else:
                cv2.rectangle(masks[class_name], (xmin, ymin), (xmax, ymax), 255, 2)
        
        return masks
    
    else:  # output_format == "mask"
        mask = np.zeros((height, width), dtype=np.uint8)
        for ann in annotations:
            xmin, ymin, xmax, ymax = ann["bbox"]
            if fill_bboxes:
                cv2.rectangle(mask, (xmin, ymin), (xmax, ymax), 255, -1)
            else:
                cv2.rectangle(mask, (xmin, ymin), (xmax, ymax), 255, 2)
        
        return mask


def _parse_polygon_coordinates(
    root: ET.Element,
    xml_path: Path,
    image_size: Tuple[int, int],
    output_format: str,
    class_filter: Optional[Union[str, List[str]]],
    **kwargs
) -> Union[np.ndarray, Dict[str, np.ndarray], List[Dict[str, Any]]]:
    """Parse polygon coordinate annotations (existing format)."""
    # Use existing parse_xml_polygon_to_binary_mask logic
    width, height = image_size
    
    # This is simpler - just use existing function
    if output_format == "annotations":
        # Extract polygons and return as annotations
        logger.warning(
            f"Annotation output format not fully supported for polygon XML. "
            f"Returning mask format instead."
        )
    
    # For now, just return the mask using existing function
    mask = parse_xml_polygon_to_binary_mask(xml_path, image_size)
    
    if output_format == "masks_by_class":
        return {"polygon": mask}
    else:
        return mask


def parse_xml_polygon_to_binary_mask(xml_path: Path, image_size: Tuple[int, int]) -> np.ndarray:
    """
    Parse XML polygon coordinates and rasterize to binary mask.

    Supports common XML annotation formats with polygon coordinates.
    Looks for:
    - <polygon> elements with <point> or <pt> children
    - Coordinates in attributes (x, y) or text content
    - Multiple polygons (combines them into single mask)

    Args:
        xml_path: Path to XML file
        image_size: (width, height) of target image

    Returns:
        Binary mask as numpy array (uint8, 0 and 255)

    Raises:
        FileNotFoundError: If XML file does not exist
        ValueError: If XML cannot be parsed or coordinates are invalid
    """
    if not xml_path.exists():
        raise FileNotFoundError(f"XML file not found: {xml_path}")

    width, height = image_size

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        raise ValueError(f"Failed to parse XML file {xml_path}: {e}")

    # Create binary mask
    mask = np.zeros((height, width), dtype=np.uint8)

    # Extract all polygons from XML
    all_polygons = []

    # Method 1: Look for <polygon> elements with <point> or <pt> children
    for polygon in root.findall(".//polygon"):
        polygon_points = []
        for point in polygon.findall(".//point") + polygon.findall(".//pt"):
            x = point.get("x") or point.get("X")
            y = point.get("y") or point.get("Y")
            if x is not None and y is not None:
                polygon_points.append([float(x), float(y)])
        if len(polygon_points) >= 3:
            all_polygons.append(polygon_points)

    # Method 2: Look for coordinates in text content (comma or space separated)
    if not all_polygons:
        for polygon in root.findall(".//polygon"):
            text = polygon.text
            if text and text.strip():
                # Try parsing as "x,y x,y ..." or "x y x y ..."
                coords = text.strip().replace(",", " ").split()
                if len(coords) % 2 == 0 and len(coords) >= 6:
                    polygon_points = []
                    for i in range(0, len(coords), 2):
                        polygon_points.append([float(coords[i]), float(coords[i + 1])])
                    all_polygons.append(polygon_points)

    if not all_polygons:
        raise ValueError(
            f"Could not find polygon coordinates in XML file {xml_path}. "
            "Expected <polygon> with <point> elements or coordinate attributes."
        )

    # Rasterize each polygon
    for polygon_points in all_polygons:
        points = np.array(polygon_points, dtype=np.int32)

        # Validate and clip coordinates
        if np.any(points < 0) or np.any(points[:, 0] >= width) or np.any(points[:, 1] >= height):
            logger.warning(
                f"Some XML polygon coordinates are outside image bounds ({width}x{height}). "
                "Clipping to bounds."
            )
            points[:, 0] = np.clip(points[:, 0], 0, width - 1)
            points[:, 1] = np.clip(points[:, 1], 0, height - 1)

        # Reshape points for OpenCV fillPoly
        points_reshaped = points.reshape(1, -1, 2)

        # Fill polygon
        cv2.fillPoly(mask, points_reshaped, 255)

    return mask
