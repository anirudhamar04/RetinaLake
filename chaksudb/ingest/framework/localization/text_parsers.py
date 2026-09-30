"""
Text file parsers for localization annotations.

Supports:
- Tab-separated bounding boxes (DR1-2)
- Space-separated keypoints (Drishti-GS1)
- Space-separated boundary contours (Drishti-GS1)
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .helpers import normalize_bbox_coordinates, normalize_keypoint_coordinates

logger = logging.getLogger(__name__)


def parse_tsv_bounding_boxes(
    tsv_path: Path,
    class_filter: Optional[Union[str, List[str]]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Parse tab-separated text files with bounding box annotations.

    Format: lesion_id  lesion_type  xmin  xmax  ymin  ymax

    Args:
        tsv_path: Path to TSV file
        class_filter: Optional class name(s) to extract

    Returns:
        Dictionary mapping class names to lists of bounding boxes

    Example:
        >>> boxes = parse_tsv_bounding_boxes(Path("markings/IMG0001.txt"))
        {"exsudato-duro": [{"xmin": 214, "ymin": 203, ...}, ...]}
    """
    if not tsv_path.exists():
        raise FileNotFoundError(f"TSV file not found: {tsv_path}")

    # Normalize class_filter
    if class_filter and isinstance(class_filter, str):
        class_filter = [class_filter]

    localizations = {}

    with open(tsv_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) < 6:
                logger.warning(f"Skipping malformed line in {tsv_path}: {line}")
                continue

            lesion_id = parts[0].strip()
            lesion_type = parts[1].strip()

            try:
                xmin = int(parts[2].strip())
                xmax = int(parts[3].strip())
                ymin = int(parts[4].strip())
                ymax = int(parts[5].strip())
            except ValueError as e:
                logger.warning(f"Skipping line with invalid coordinates in {tsv_path}: {line} ({e})")
                continue

            # Filter by class
            if class_filter and lesion_type not in class_filter:
                continue

            # Normalize coordinates
            coordinates = normalize_bbox_coordinates(xmin, ymin, xmax, ymax)

            # Group by lesion type
            if lesion_type not in localizations:
                localizations[lesion_type] = []

            localizations[lesion_type].append({
                "type": "bounding_box",
                "coordinates": coordinates,
                "lesion_id": lesion_id,
            })

    return localizations


def parse_text_keypoint(
    txt_path: Path,
    structure_name: str = "optic_disc_center",
) -> Dict[str, Dict[str, Any]]:
    """
    Parse single-line space-separated x y coordinates.

    Args:
        txt_path: Path to text file
        structure_name: Name of the anatomical structure

    Returns:
        Dictionary with keypoint localization

    Example:
        >>> point = parse_text_keypoint(Path("drishtiGS_002_diskCenter.txt"))
        {"optic_disc_center": {"type": "keypoint", "coordinates": {"x": 1448, "y": 972}}}
    """
    if not txt_path.exists():
        raise FileNotFoundError(f"Text file not found: {txt_path}")

    with open(txt_path, "r") as f:
        line = f.readline().strip()

    if not line:
        raise ValueError(f"Empty file: {txt_path}")

    parts = line.split()
    if len(parts) != 2:
        raise ValueError(f"Expected 2 space-separated values, got {len(parts)}: {line}")

    try:
        x = float(parts[0])
        y = float(parts[1])
    except ValueError as e:
        raise ValueError(f"Invalid coordinates in {txt_path}: {line} ({e})")

    coordinates = normalize_keypoint_coordinates(x, y)

    return {
        structure_name: {
            "type": "keypoint",
            "coordinates": coordinates,
        }
    }


def parse_text_boundary(
    txt_path: Path,
    structure_name: str,
) -> Dict[str, Dict[str, Any]]:
    """
    Parse multi-line space-separated x y coordinates (polyline/contour).

    Args:
        txt_path: Path to text file with multiple x y coordinate pairs
        structure_name: Name of the anatomical structure

    Returns:
        Dictionary with contour localization

    Example:
        >>> boundary = parse_text_boundary(
        ...     Path("drishtiGS_002_ODAvgBoundary.txt"),
        ...     "optic_disc_boundary"
        ... )
        {
            "optic_disc_boundary": {
                "type": "contour",
                "coordinates": {
                    "points": [[1615.0, 982.0], [1614.0, 983.0], ...]
                }
            }
        }
    """
    if not txt_path.exists():
        raise FileNotFoundError(f"Text file not found: {txt_path}")

    points = []

    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 2:
                logger.warning(f"Skipping malformed line in {txt_path}: {line}")
                continue

            try:
                x = float(parts[0])
                y = float(parts[1])
                points.append([x, y])
            except ValueError as e:
                logger.warning(f"Skipping line with invalid coordinates in {txt_path}: {line} ({e})")
                continue

    if not points:
        raise ValueError(f"No valid coordinates found in {txt_path}")

    return {
        structure_name: {
            "type": "contour",
            "coordinates": {
                "points": points,
                "num_points": len(points),
            },
        }
    }


def parse_text_boundary_as_keypoints(
    txt_path: Path,
    structure_name: str,
) -> List[Dict[str, Any]]:
    """
    Parse multi-line space-separated x y coordinates as individual keypoints.

    This is useful when you want to treat each point in a boundary as a separate
    localization annotation rather than a single contour.

    Args:
        txt_path: Path to text file with multiple x y coordinate pairs
        structure_name: Name of the anatomical structure

    Returns:
        List of keypoint localizations

    Example:
        >>> keypoints = parse_text_boundary_as_keypoints(
        ...     Path("drishtiGS_002_ODAvgBoundary.txt"),
        ...     "optic_disc_boundary_point"
        ... )
        [
            {"type": "keypoint", "coordinates": {"x": 1615.0, "y": 982.0}, "index": 0},
            {"type": "keypoint", "coordinates": {"x": 1614.0, "y": 983.0}, "index": 1},
            ...
        ]
    """
    if not txt_path.exists():
        raise FileNotFoundError(f"Text file not found: {txt_path}")

    keypoints = []

    with open(txt_path, "r") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 2:
                logger.warning(f"Skipping malformed line in {txt_path}: {line}")
                continue

            try:
                x = float(parts[0])
                y = float(parts[1])

                coordinates = normalize_keypoint_coordinates(x, y)

                keypoints.append({
                    "type": "keypoint",
                    "coordinates": coordinates,
                    "index": idx,
                })
            except ValueError as e:
                logger.warning(f"Skipping line with invalid coordinates in {txt_path}: {line} ({e})")
                continue

    if not keypoints:
        raise ValueError(f"No valid coordinates found in {txt_path}")

    return keypoints
