"""
XML parsers for localization annotations.

Supports:
- Pascal VOC bounding boxes (SUSTech-SYSU, OIA-DDR)
- ImageRet circle regions (DiaRetDB1)
"""

import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


def extract_localization_from_xml(
    xml_path: Path,
    class_filter: Optional[Union[str, List[str]]] = None,
    include_metadata: bool = True,
) -> Dict[str, Any]:
    """
    Extract localization data from XML annotations in JSONB-compatible format.

    Automatically detects and parses:
    - PASCAL VOC bounding boxes (SUSTech-SYSU OD/Fovea, OIA-DDR lesions)
    - ImageRet circle regions (DiaRetDB1 lesions)

    Args:
        xml_path: Path to XML annotation file
        class_filter: Optional class name(s) to extract. If None, extracts all.
                     Can be string or list of strings.
        include_metadata: If True, includes truncated/difficult/confidence metadata

    Returns:
        Dictionary with localization data in JSONB-compatible format.

        Format:
        {
            "OD": {
                "type": "bbox",
                "coordinates": {...},
                "metadata": {...}
            },
            "fovea": {
                "type": "point",
                "coordinates": {...},
                "metadata": {...}
            }
        }

    Raises:
        FileNotFoundError: If XML file does not exist
        ValueError: If XML cannot be parsed or format is unknown

    Example:
        >>> data = extract_localization_from_xml(
        ...     Path("data/30_SUSTech-SYSU/odFoveaLabels/0400.xml")
        ... )
        {
            "OD": {
                "type": "bbox",
                "coordinates": {"xmin": 2269, "ymin": 936, ...},
                "metadata": {"truncated": false, "difficult": false}
            },
            "fovea": {
                "type": "point",
                "coordinates": {"x": 1421, "y": 1422},
                ...
            }
        }
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
    xml_format = _detect_localization_format(root)

    # Parse based on detected format
    if xml_format == "pascal_voc":
        result = _extract_pascal_voc_localizations(root, class_filter, include_metadata)
    elif xml_format == "imageret_circles":
        result = _extract_imageret_localizations(root, class_filter, include_metadata)
    else:
        raise ValueError(
            f"Unknown localization format in {xml_path}. "
            f"Supported: PASCAL VOC bounding boxes, ImageRet circle regions."
        )

    grouped = {}
    localizations = result.get("localizations", [])

    # Count occurrences of each class
    class_counts = {}
    for loc in localizations:
        cls = loc["class"]
        class_counts[cls] = class_counts.get(cls, 0) + 1

    # Group by class
    class_indices = {}
    for loc in localizations:
        cls = loc["class"]

        # Remove the "class" key from the localization
        loc_data = {k: v for k, v in loc.items() if k != "class"}

        # If multiple objects of same class, add index
        if class_counts[cls] > 1:
            if cls not in class_indices:
                class_indices[cls] = 0
            class_indices[cls] += 1
            key = f"{cls}_{class_indices[cls]}"
        else:
            key = cls

        grouped[key] = loc_data

    return grouped


def _detect_localization_format(root: ET.Element) -> str:
    """
    Auto-detect XML localization format.

    Returns:
        "pascal_voc" - PASCAL VOC style bounding boxes
        "imageret_circles" - DiaRetDB1 style circle annotations
    """
    # Check for PASCAL VOC format (bndbox)
    if root.find(".//bndbox") is not None:
        return "pascal_voc"

    # If annotation tag with object/bndbox
    if root.tag == "annotation" and root.find(".//object") is not None:
        return "pascal_voc"

    # Check for ImageRet circle format (DiaRetDB1)
    if root.find(".//circleregion") is not None:
        return "imageret_circles"

    # If imgannotooldata with markinglist
    if root.tag == "imgannotooldata" and root.find(".//markinglist") is not None:
        return "imageret_circles"

    return "unknown"


def _extract_pascal_voc_localizations(
    root: ET.Element,
    class_filter: Optional[Union[str, List[str]]],
    include_metadata: bool,
) -> Dict[str, Any]:
    """
    Extract bounding box localizations from PASCAL VOC format.

    Detects if bbox is actually a point (1-2 pixel size) and converts to point type.
    """
    # Normalize class filter
    if isinstance(class_filter, str):
        class_filter = [class_filter]

    localizations = []

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

        # Calculate width and height
        width = xmax - xmin
        height = ymax - ymin

        # Determine if this is a point or bbox
        # Points are typically 1-2 pixels (like fovea center)
        is_point = width <= 2 and height <= 2

        localization = {
            "class": class_name,
            "type": "point" if is_point else "bbox",
        }

        if is_point:
            # For points, use center coordinates
            localization["coordinates"] = {"x": (xmin + xmax) // 2, "y": (ymin + ymax) // 2}
        else:
            # For bboxes, include all coordinates
            localization["coordinates"] = {
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
                "width": width,
                "height": height,
                "center_x": (xmin + xmax) // 2,
                "center_y": (ymin + ymax) // 2,
            }

        # Add metadata if requested
        if include_metadata:
            metadata = {}

            truncated_elem = obj.find("truncated")
            if truncated_elem is not None:
                metadata["truncated"] = bool(int(truncated_elem.text))

            difficult_elem = obj.find("difficult")
            if difficult_elem is not None:
                metadata["difficult"] = bool(int(difficult_elem.text))

            if metadata:
                localization["metadata"] = metadata

        localizations.append(localization)

    return {"format": "pascal_voc", "localizations": localizations}


def _extract_imageret_localizations(
    root: ET.Element,
    class_filter: Optional[Union[str, List[str]]],
    include_metadata: bool,
) -> Dict[str, Any]:
    """
    Extract circle region localizations from ImageRet format (DiaRetDB1).
    """
    # Normalize class filter
    if isinstance(class_filter, str):
        class_filter = [class_filter]

    localizations = []

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
        class_name = marking_type_elem.text.strip() if marking_type_elem is not None else "unknown"

        # Filter by class
        if class_filter and class_name not in class_filter:
            continue

        localization = {
            "class": class_name,
            "type": "circle",
            "coordinates": {
                "center_x": cx,
                "center_y": cy,
                "radius": radius,
                # Also provide bbox equivalent for consistency
                "xmin": max(0, cx - radius),
                "ymin": max(0, cy - radius),
                "xmax": cx + radius,
                "ymax": cy + radius,
            },
        }

        # Add metadata if requested
        if include_metadata:
            metadata = {}

            # Extract confidence level
            confidence_elem = marking.find(".//confidencelevel")
            if confidence_elem is not None:
                metadata["confidence"] = confidence_elem.text.strip()

            # Extract representative point if available
            repr_point_elem = marking.find(".//representativepoint/coords2d")
            if repr_point_elem is not None:
                repr_coords = repr_point_elem.text.strip().split(",")
                if len(repr_coords) == 2:
                    metadata["representative_point"] = {
                        "x": int(repr_coords[0]),
                        "y": int(repr_coords[1]),
                    }

            if metadata:
                localization["metadata"] = metadata

        localizations.append(localization)

    return {"format": "imageret_circles", "localizations": localizations}
