"""
Localization framework for extracting bounding box, keypoint, and contour coordinates.

This module provides parsers for various localization annotation formats:
- XML: Pascal VOC bounding boxes, ImageRet circle regions
- JSON: Keypoint coordinates
- Text: TSV bounding boxes, space-separated keypoints and boundaries

All parsers return JSONB-compatible dictionaries ready for database storage.
"""

# XML parsers
from .xml_parsers import extract_localization_from_xml

# JSON parsers
from .json_parsers import parse_json_keypoints, parse_json_keypoints_batch

# Text parsers
from .text_parsers import (
    parse_tsv_bounding_boxes,
    parse_text_keypoint,
    parse_text_boundary,
    parse_text_boundary_as_keypoints,
)

# Helper functions
from .helpers import (
    normalize_bbox_coordinates,
    normalize_keypoint_coordinates,
    normalize_center_point_coordinates,
    get_bbox_from_localization,
    get_center_from_localization,
)

__all__ = [
    # XML parsers
    "extract_localization_from_xml",
    # JSON parsers
    "parse_json_keypoints",
    "parse_json_keypoints_batch",
    # Text parsers
    "parse_tsv_bounding_boxes",
    "parse_text_keypoint",
    "parse_text_boundary",
    "parse_text_boundary_as_keypoints",
    # Helper functions
    "normalize_bbox_coordinates",
    "normalize_keypoint_coordinates",
    "normalize_center_point_coordinates",
    "get_bbox_from_localization",
    "get_center_from_localization",
]
