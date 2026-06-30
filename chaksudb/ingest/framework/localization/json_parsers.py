"""
JSON parsers for localization annotations.

Supports:
- JSON keypoint coordinates (REFUGE, BRSET)
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .helpers import normalize_keypoint_coordinates

logger = logging.getLogger(__name__)


def parse_json_keypoints(
    json_path: Path,
    image_id_key: str = "ImgName",
    coordinate_mapping: Optional[Dict[str, Tuple[str, str]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Parse JSON files with keypoint coordinates.

    Args:
        json_path: Path to JSON file
        image_id_key: Key for image identifier in JSON
        coordinate_mapping: Dict mapping structure names to (x_key, y_key) tuples
                           Default: {"fovea": ("Fovea_X", "Fovea_Y")}

    Returns:
        Dictionary mapping structure names to localization data

    Example:
        >>> points = parse_json_keypoints(Path("index.json"))
        {"fovea": {"type": "keypoint", "coordinates": {"x": 1057.95, "y": 1076.52}}}

        >>> # Custom mapping for optic disc
        >>> points = parse_json_keypoints(
        ...     Path("metadata.json"),
        ...     coordinate_mapping={"optic_disc": ("OD_X", "OD_Y")}
        ... )
    """
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    # Default coordinate mapping
    if coordinate_mapping is None:
        coordinate_mapping = {"fovea": ("Fovea_X", "Fovea_Y")}

    with open(json_path, "r") as f:
        data = json.load(f)

    # Handle both single object and list/dict of objects
    if isinstance(data, list):
        # Take first entry
        entry = data[0] if data else {}
    elif isinstance(data, dict):
        # Check if it's a dict of dicts (indexed by number)
        if "0" in data:
            entry = data["0"]
        else:
            entry = data
    else:
        raise ValueError(f"Unexpected JSON structure in {json_path}")

    localizations = {}

    for structure_name, (x_key, y_key) in coordinate_mapping.items():
        if x_key in entry and y_key in entry:
            x = entry[x_key]
            y = entry[y_key]

            coordinates = normalize_keypoint_coordinates(x, y)

            localizations[structure_name] = {
                "type": "keypoint",
                "coordinates": coordinates,
            }

    return localizations


def parse_json_keypoints_batch(
    json_path: Path,
    coordinate_mapping: Optional[Dict[str, Tuple[str, str]]] = None,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Parse JSON files containing multiple images with keypoint coordinates.

    Args:
        json_path: Path to JSON file
        coordinate_mapping: Dict mapping structure names to (x_key, y_key) tuples
                           Default: {"fovea": ("Fovea_X", "Fovea_Y")}

    Returns:
        Dictionary mapping image names to structure localizations

    Example:
        >>> batch = parse_json_keypoints_batch(Path("index.json"))
        {
            "g0001.jpg": {
                "fovea": {"type": "keypoint", "coordinates": {"x": 1057.95, "y": 1076.52}}
            },
            "g0010.jpg": {
                "fovea": {"type": "keypoint", "coordinates": {"x": 1310.76, "y": 1074.7}}
            }
        }
    """
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    # Default coordinate mapping
    if coordinate_mapping is None:
        coordinate_mapping = {"fovea": ("Fovea_X", "Fovea_Y")}

    with open(json_path, "r") as f:
        data = json.load(f)

    results = {}

    # Handle dict of dicts (indexed by number)
    if isinstance(data, dict):
        for idx, entry in data.items():
            if not isinstance(entry, dict):
                continue

            # Get image name
            image_name = entry.get("ImgName", f"image_{idx}")

            # Extract keypoints
            localizations = {}
            for structure_name, (x_key, y_key) in coordinate_mapping.items():
                if x_key in entry and y_key in entry:
                    x = entry[x_key]
                    y = entry[y_key]

                    coordinates = normalize_keypoint_coordinates(x, y)

                    localizations[structure_name] = {
                        "type": "keypoint",
                        "coordinates": coordinates,
                    }

            if localizations:
                results[image_name] = localizations

    # Handle list of dicts
    elif isinstance(data, list):
        for idx, entry in enumerate(data):
            if not isinstance(entry, dict):
                continue

            # Get image name
            image_name = entry.get("ImgName", f"image_{idx}")

            # Extract keypoints
            localizations = {}
            for structure_name, (x_key, y_key) in coordinate_mapping.items():
                if x_key in entry and y_key in entry:
                    x = entry[x_key]
                    y = entry[y_key]

                    coordinates = normalize_keypoint_coordinates(x, y)

                    localizations[structure_name] = {
                        "type": "keypoint",
                        "coordinates": coordinates,
                    }

            if localizations:
                results[image_name] = localizations

    return results
