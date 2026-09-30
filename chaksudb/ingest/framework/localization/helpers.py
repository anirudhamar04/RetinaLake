"""
Helper functions for coordinate normalization and manipulation.
"""

from typing import Any, Dict, Optional, Union


def normalize_bbox_coordinates(
    xmin: Union[int, float],
    ymin: Union[int, float],
    xmax: Union[int, float],
    ymax: Union[int, float],
) -> Dict[str, float]:
    """
    Normalize bounding box coordinates to standard format.

    Args:
        xmin, ymin, xmax, ymax: Bounding box coordinates

    Returns:
        Dictionary with normalized coordinates including width, height, and center
    """
    return {
        "xmin": float(xmin),
        "ymin": float(ymin),
        "xmax": float(xmax),
        "ymax": float(ymax),
        "width": float(xmax - xmin),
        "height": float(ymax - ymin),
        "center_x": float((xmin + xmax) / 2),
        "center_y": float((ymin + ymax) / 2),
    }


def normalize_keypoint_coordinates(
    x: Union[int, float],
    y: Union[int, float],
) -> Dict[str, float]:
    """
    Normalize keypoint coordinates to standard format.

    Args:
        x, y: Keypoint coordinates

    Returns:
        Dictionary with normalized coordinates
    """
    return {
        "x": float(x),
        "y": float(y),
    }


def normalize_center_point_coordinates(
    center_x: Union[int, float],
    center_y: Union[int, float],
    radius: Optional[Union[int, float]] = None,
) -> Dict[str, float]:
    """
    Normalize center point coordinates (for circles).

    Args:
        center_x, center_y: Center coordinates
        radius: Optional radius

    Returns:
        Dictionary with normalized coordinates
    """
    coords = {
        "center_x": float(center_x),
        "center_y": float(center_y),
    }
    if radius is not None:
        coords["radius"] = float(radius)
        # Also provide bbox equivalent for consistency
        coords["xmin"] = float(max(0, center_x - radius))
        coords["ymin"] = float(max(0, center_y - radius))
        coords["xmax"] = float(center_x + radius)
        coords["ymax"] = float(center_y + radius)
    return coords


def get_bbox_from_localization(localization: Dict[str, Any]) -> Optional[Dict[str, int]]:
    """
    Extract bounding box coordinates from any localization type.

    Args:
        localization: Single localization dict

    Returns:
        Dict with {xmin, ymin, xmax, ymax} or None if not applicable

    Example:
        >>> data = extract_localization_from_xml(xml_path)
        >>> bbox = get_bbox_from_localization(data["OD"])
        {"xmin": 2269, "ymin": 936, "xmax": 2688, "ymax": 1391}
    """
    coords = localization.get("coordinates", {})
    loc_type = localization.get("type")

    if loc_type == "bbox":
        return {
            "xmin": coords["xmin"],
            "ymin": coords["ymin"],
            "xmax": coords["xmax"],
            "ymax": coords["ymax"],
        }
    elif loc_type == "circle":
        return {
            "xmin": coords["xmin"],
            "ymin": coords["ymin"],
            "xmax": coords["xmax"],
            "ymax": coords["ymax"],
        }
    elif loc_type == "point":
        # For points, return a 1-pixel bbox
        x, y = coords["x"], coords["y"]
        return {
            "xmin": x,
            "ymin": y,
            "xmax": x,
            "ymax": y,
        }

    return None


def get_center_from_localization(localization: Dict[str, Any]) -> Optional[Dict[str, int]]:
    """
    Extract center point from any localization type.

    Args:
        localization: Single localization dict

    Returns:
        Dict with {x, y} or None if not applicable

    Example:
        >>> data = extract_localization_from_xml(xml_path)
        >>> center = get_center_from_localization(data["fovea"])
        {"x": 1421, "y": 1422}
    """
    coords = localization.get("coordinates", {})
    loc_type = localization.get("type")

    if loc_type == "point":
        return {
            "x": coords["x"],
            "y": coords["y"],
        }
    elif loc_type in ["bbox", "circle"]:
        return {
            "x": coords.get("center_x"),
            "y": coords.get("center_y"),
        }

    return None
