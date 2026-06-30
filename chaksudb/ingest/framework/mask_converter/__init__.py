"""
Mask format handling utilities.

Provides specific functions for handling different mask formats:
- Binary masks: validation and storage
- Contour/polygon: conversion to binary masks
- XML annotations: unified parser for all XML formats
- Soft maps: loading as probability maps (not converted to binary)
- Layer boundaries: loading as boundary arrays (not converted to binary)
"""

# Binary mask operations
from .binary_masks import (
    is_multiclass_mask,
    get_mask_classes,
    extract_class_from_mask,
    extract_classes_from_multiclass_mask,
    validate_binary_mask,
)

# Contour operations
from .contours import (
    convert_contour_to_binary_mask,
    convert_contour_to_binary_mask_async,
)

# XML annotation parsing
from .xml_annotations import (
    parse_xml_annotations,
    parse_xml_polygon_to_binary_mask,
)

# Soft maps
from .soft_maps import (
    load_soft_map,
)

# Layer boundaries
from .layer_boundaries import (
    load_layer_boundaries,
)

# HEI-MED gzip float32 exudate maps
from .gnd_maps import (
    load_exudate_map_gz,
    parse_gnd_blob_count,
    parse_meta_file,
)

__all__ = [
    # Binary masks
    "is_multiclass_mask",
    "get_mask_classes",
    "extract_class_from_mask",
    "extract_classes_from_multiclass_mask",
    "validate_binary_mask",
    # Contours
    "convert_contour_to_binary_mask",
    "convert_contour_to_binary_mask_async",
    # XML annotations
    "parse_xml_annotations",
    "parse_xml_polygon_to_binary_mask",
    # Soft maps
    "load_soft_map",
    # Layer boundaries
    "load_layer_boundaries",
    # HEI-MED gzip float32 exudate maps
    "load_exudate_map_gz",
    "parse_gnd_blob_count",
    "parse_meta_file",
]
