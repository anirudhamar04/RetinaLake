"""
Localization processor for bounding boxes, keypoints, and landmarks.

This module provides high-level processing functions for localization annotations
that handle various input formats, UUID generation, and model preparation.

Key features:
- Wraps existing localization parsers (XML, JSON, text)
- Converts coordinates to LocalizationAnnotation models
- Generates deterministic UUIDs
- Returns models ready for upsert
- Automatic provenance tracking via context variables
"""

import hashlib
import json
import logging
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from chaksudb.db.models import LocalizationAnnotation
from chaksudb.ingest.framework.gen_uuid import generate_localization_uuid
from chaksudb.ingest.framework.localization import (
    extract_localization_from_xml,
    parse_json_keypoints,
    parse_text_keypoint,
    parse_tsv_bounding_boxes,
)
from chaksudb.ingest.framework.provenance_context import get_current_provenance

logger = logging.getLogger(__name__)

# Canonical target_structure names. Maps raw XML/annotation class names
# (lowercased) to (target_structure, lesion_subtype|None).
_STRUCTURE_NORMALIZATION: Dict[str, Tuple[str, Optional[str]]] = {
    # Anatomical landmarks
    "od": ("optic_disc", None),
    "optic_disc": ("optic_disc", None),
    "optic_disc_center": ("optic_disc", None),
    "disc": ("optic_disc", None),
    "fovea": ("fovea", None),
    "fovea_center": ("fovea", None),
    "macula": ("fovea", None),
    # Lesion short codes (OIA-DDR, SUSTech-SYSU XML)
    "ex": ("lesions", "EX"),
    "he": ("lesions", "HE"),
    "ma": ("lesions", "MA"),
    "se": ("lesions", "SE"),
    # Lesion long names (DiaRetDB1 XML, legacy)
    "hard_exudates": ("lesions", "EX"),
    "hard exudates": ("lesions", "EX"),
    "soft_exudates": ("lesions", "SE"),
    "soft exudates": ("lesions", "SE"),
    "haemorrhages": ("lesions", "HE"),
    "hemorrhages": ("lesions", "HE"),
    "exudates": ("lesions", "EX"),
    "microaneurysms": ("lesions", "MA"),
    "red_small_dots": ("lesions", "MA"),
    "red small dots": ("lesions", "MA"),
    "irma": ("lesions", "IRMA"),
    "neovascularisation": ("lesions", "NV"),
}

_INDEX_SUFFIX_RE = re.compile(r"^(.+?)_(\d+)$")


def normalize_target_structure(
    raw_name: str,
) -> Tuple[str, Optional[str]]:
    """Resolve a raw XML/annotation class name to (target_structure, lesion_subtype).

    Strips trailing ``_N`` index suffixes added by the XML parser for
    multi-object classes before looking up the canonical name.
    Unknown names are returned as-is (lowercased) with ``lesion_subtype=None``.
    """
    stripped = raw_name.strip()
    # Strip _N index suffix (e.g. "ma_12" -> "ma")
    m = _INDEX_SUFFIX_RE.match(stripped)
    base = m.group(1) if m else stripped
    key = base.lower().replace("_", " ").strip()
    # Also try with underscores preserved
    key_underscore = base.lower().strip()

    if key in _STRUCTURE_NORMALIZATION:
        return _STRUCTURE_NORMALIZATION[key]
    if key_underscore in _STRUCTURE_NORMALIZATION:
        return _STRUCTURE_NORMALIZATION[key_underscore]
    return (stripped, None)


# ============================================
# Helper Functions
# ============================================


def compute_coordinates_hash(coordinates: Dict[str, Any]) -> str:
    """
    Compute a deterministic hash of coordinates dictionary.

    Args:
        coordinates: Dictionary of coordinates

    Returns:
        SHA256 hash of the JSON-serialized dictionary
    """
    # Sort keys for deterministic serialization
    json_str = json.dumps(coordinates, sort_keys=True)
    return hashlib.sha256(json_str.encode()).hexdigest()


# ============================================
# Main Processing Functions
# ============================================


async def process_localization_from_xml(
    xml_path: Path,
    image_id: uuid.UUID,
    image_size: Optional[Tuple[int, int]] = None,
    class_filter: Optional[Union[str, List[str]]] = None,
    raw_data_id: Optional[uuid.UUID] = None,
    expert_annotation_id: Optional[uuid.UUID] = None,
    consensus_id: Optional[uuid.UUID] = None,
    annotation_method: str = "manual",
    provenance_chain_id: Optional[uuid.UUID] = None,
) -> List[LocalizationAnnotation]:
    """
    Process localizations from XML and prepare for database upsert.

    Wraps extract_localization_from_xml() and converts output to
    LocalizationAnnotation models.

    Args:
        xml_path: Path to XML annotation file
        image_id: UUID of the image
        image_size: Optional (width, height) for validation
        class_filter: Optional class name(s) to extract
        raw_data_id: Optional raw annotation file UUID
        expert_annotation_id: Optional expert annotation UUID
        consensus_id: Optional consensus UUID
        annotation_method: Method ('manual', 'pseudo')
        provenance_chain_id: Optional provenance chain UUID

    Returns:
        List of LocalizationAnnotation models ready for upsert

    Example:
        ```python
        annotations = await process_localization_from_xml(
            xml_path=Path("odFoveaLabels/0400.xml"),
            image_id=image_id,
            raw_data_id=raw_file_id,
        )
        for annotation in annotations:
            await upsert_localization_annotation(annotation)
        ```
    """
    # Get provenance from context if not explicitly provided
    if raw_data_id is None or provenance_chain_id is None:
        context_raw_id, context_chain_id = get_current_provenance()
        raw_data_id = raw_data_id or context_raw_id
        provenance_chain_id = provenance_chain_id or context_chain_id
    
    # Validate annotation_method
    valid_methods = {"manual", "pseudo"}
    if annotation_method not in valid_methods:
        raise ValueError(
            f"Invalid annotation_method: {annotation_method}. "
            f"Must be one of {valid_methods}"
        )

    # Call existing function to extract localizations
    localizations = extract_localization_from_xml(
        xml_path,
        class_filter=class_filter,
        include_metadata=True,
    )

    # Convert each localization to a model
    annotations = []

    for raw_target_structure, data in localizations.items():
        # Determine localization_type from data["type"]
        type_mapping = {
            "bbox": "bounding_box",
            "point": "keypoint",
            "circle": "center_point",
        }
        localization_type = type_mapping.get(data["type"], data["type"])

        # Normalize target_structure and detect lesion subtype
        target_structure, lesion_subtype = normalize_target_structure(raw_target_structure)

        # Compute hash of coordinates
        coordinates_hash = compute_coordinates_hash(data["coordinates"])

        # Generate UUID (uses normalized target_structure for consistency)
        localization_id = generate_localization_uuid(
            image_id=image_id,
            localization_type=localization_type,
            target_structure=target_structure,
            expert_annotation_id=expert_annotation_id,
            consensus_id=consensus_id,
            raw_data_id=raw_data_id,
            coordinates_hash=coordinates_hash,
        )

        # Create model
        annotation = LocalizationAnnotation(
            localization_id=localization_id,
            image_id=image_id,
            localization_type=localization_type,
            target_structure=target_structure,
            coordinates=data["coordinates"],
            lesion_subtype=lesion_subtype,
            raw_data_id=raw_data_id,
            expert_annotation_id=expert_annotation_id,
            consensus_id=consensus_id,
            annotation_method=annotation_method,
            provenance_chain_id=provenance_chain_id,
        )
        annotations.append(annotation)

        logger.debug(
            f"Processed {localization_type} localization for {target_structure} "
            f"(raw: {raw_target_structure}) in image {image_id}"
        )

    return annotations


async def process_localization_from_tsv(
    tsv_path: Path,
    image_id: uuid.UUID,
    class_filter: Optional[Union[str, List[str]]] = None,
    raw_data_id: Optional[uuid.UUID] = None,
    annotation_method: str = "manual",
    provenance_chain_id: Optional[uuid.UUID] = None,
) -> List[LocalizationAnnotation]:
    """
    Process localizations from tab-separated text file.

    Args:
        tsv_path: Path to TSV file
        image_id: UUID of the image
        class_filter: Optional class name(s) to extract
        raw_data_id: Optional raw annotation file UUID
        annotation_method: Method ('manual', 'pseudo')
        provenance_chain_id: Optional provenance chain UUID

    Returns:
        List of LocalizationAnnotation models

    Example:
        ```python
        annotations = await process_localization_from_tsv(
            tsv_path=Path("markings/IMG0001.txt"),
            image_id=image_id,
            raw_data_id=raw_file_id,
        )
        ```
    """
    # Get provenance from context if not explicitly provided
    if raw_data_id is None or provenance_chain_id is None:
        context_raw_id, context_chain_id = get_current_provenance()
        raw_data_id = raw_data_id or context_raw_id
        provenance_chain_id = provenance_chain_id or context_chain_id
    
    # Parse TSV file
    localizations = parse_tsv_bounding_boxes(tsv_path, class_filter)

    annotations = []

    for raw_target_structure, bboxes in localizations.items():
        target_structure, lesion_subtype = normalize_target_structure(raw_target_structure)

        for bbox_data in bboxes:
            # Compute hash
            coordinates_hash = compute_coordinates_hash(bbox_data["coordinates"])

            # Generate UUID
            localization_id = generate_localization_uuid(
                image_id=image_id,
                localization_type="bounding_box",
                target_structure=target_structure,
                raw_data_id=raw_data_id,
                coordinates_hash=coordinates_hash,
            )

            # Create model
            annotation = LocalizationAnnotation(
                localization_id=localization_id,
                image_id=image_id,
                localization_type="bounding_box",
                target_structure=target_structure,
                coordinates=bbox_data["coordinates"],
                lesion_subtype=lesion_subtype,
                raw_data_id=raw_data_id,
                annotation_method=annotation_method,
                provenance_chain_id=provenance_chain_id,
            )
            annotations.append(annotation)

    logger.debug(
        f"Processed {len(annotations)} bounding box localizations from {tsv_path}"
    )

    return annotations


async def process_localization_from_json(
    json_path: Path,
    image_id: uuid.UUID,
    coordinate_mapping: Optional[Dict[str, Tuple[str, str]]] = None,
    raw_data_id: Optional[uuid.UUID] = None,
    annotation_method: str = "manual",
    provenance_chain_id: Optional[uuid.UUID] = None,
) -> List[LocalizationAnnotation]:
    """
    Process keypoint localizations from JSON file.

    Args:
        json_path: Path to JSON file
        image_id: UUID of the image
        coordinate_mapping: Dict mapping structure names to (x_key, y_key) tuples
        raw_data_id: Optional raw annotation file UUID
        annotation_method: Method ('manual', 'pseudo')
        provenance_chain_id: Optional provenance chain UUID

    Returns:
        List of LocalizationAnnotation models

    Example:
        ```python
        annotations = await process_localization_from_json(
            json_path=Path("index.json"),
            image_id=image_id,
            coordinate_mapping={"fovea": ("Fovea_X", "Fovea_Y")},
        )
        ```
    """
    # Get provenance from context if not explicitly provided
    if raw_data_id is None or provenance_chain_id is None:
        context_raw_id, context_chain_id = get_current_provenance()
        raw_data_id = raw_data_id or context_raw_id
        provenance_chain_id = provenance_chain_id or context_chain_id
    
    # Parse JSON file
    localizations = parse_json_keypoints(json_path, coordinate_mapping=coordinate_mapping)

    annotations = []

    for raw_target_structure, data in localizations.items():
        target_structure, lesion_subtype = normalize_target_structure(raw_target_structure)

        # Compute hash
        coordinates_hash = compute_coordinates_hash(data["coordinates"])

        # Generate UUID
        localization_id = generate_localization_uuid(
            image_id=image_id,
            localization_type="keypoint",
            target_structure=target_structure,
            raw_data_id=raw_data_id,
            coordinates_hash=coordinates_hash,
        )

        # Create model
        annotation = LocalizationAnnotation(
            localization_id=localization_id,
            image_id=image_id,
            localization_type="keypoint",
            target_structure=target_structure,
            coordinates=data["coordinates"],
            lesion_subtype=lesion_subtype,
            raw_data_id=raw_data_id,
            annotation_method=annotation_method,
            provenance_chain_id=provenance_chain_id,
        )
        annotations.append(annotation)

    logger.debug(
        f"Processed {len(annotations)} keypoint localizations from {json_path}"
    )

    return annotations


async def process_localization_from_text_keypoint(
    txt_path: Path,
    image_id: uuid.UUID,
    structure_name: str = "optic_disc_center",
    raw_data_id: Optional[uuid.UUID] = None,
    annotation_method: str = "manual",
    provenance_chain_id: Optional[uuid.UUID] = None,
) -> LocalizationAnnotation:
    """
    Process single keypoint localization from text file.

    Args:
        txt_path: Path to text file
        image_id: UUID of the image
        structure_name: Name of the anatomical structure
        raw_data_id: Optional raw annotation file UUID
        annotation_method: Method ('manual', 'pseudo')
        provenance_chain_id: Optional provenance chain UUID

    Returns:
        LocalizationAnnotation model

    Example:
        ```python
        annotation = await process_localization_from_text_keypoint(
            txt_path=Path("drishtiGS_002_diskCenter.txt"),
            image_id=image_id,
            structure_name="optic_disc_center",
        )
        ```
    """
    # Get provenance from context if not explicitly provided
    if raw_data_id is None or provenance_chain_id is None:
        context_raw_id, context_chain_id = get_current_provenance()
        raw_data_id = raw_data_id or context_raw_id
        provenance_chain_id = provenance_chain_id or context_chain_id
    
    # Parse text file
    localizations = parse_text_keypoint(txt_path, structure_name)

    data = localizations[structure_name]

    # Normalize structure name
    target_structure, lesion_subtype = normalize_target_structure(structure_name)

    # Compute hash
    coordinates_hash = compute_coordinates_hash(data["coordinates"])

    # Generate UUID
    localization_id = generate_localization_uuid(
        image_id=image_id,
        localization_type="keypoint",
        target_structure=target_structure,
        raw_data_id=raw_data_id,
        coordinates_hash=coordinates_hash,
    )

    # Create model
    annotation = LocalizationAnnotation(
        localization_id=localization_id,
        image_id=image_id,
        localization_type="keypoint",
        target_structure=target_structure,
        coordinates=data["coordinates"],
        lesion_subtype=lesion_subtype,
        raw_data_id=raw_data_id,
        annotation_method=annotation_method,
        provenance_chain_id=provenance_chain_id,
    )

    logger.debug(
        f"Processed keypoint localization for {target_structure} "
        f"(raw: {structure_name}) in image {image_id}"
    )

    return annotation


# ============================================
# Convenience Function (alias)
# ============================================


async def prepare_localizations_for_upsert(
    source_path: Path,
    image_id: uuid.UUID,
    **kwargs,
) -> List[LocalizationAnnotation]:
    """
    Auto-detect format and process localizations.

    Convenience function that auto-detects the file format and calls
    the appropriate processing function.

    Args:
        source_path: Path to annotation file
        image_id: UUID of the image
        **kwargs: Additional arguments for specific processors

    Returns:
        List of LocalizationAnnotation models

    Example:
        ```python
        # Auto-detect and process any supported format
        annotations = await prepare_localizations_for_upsert(
            source_path=Path("annotations/0400.xml"),
            image_id=image_id,
            raw_data_id=raw_file_id,
        )
        ```
    """
    suffix = source_path.suffix.lower()

    if suffix == ".xml":
        return await process_localization_from_xml(source_path, image_id, **kwargs)
    elif suffix == ".json":
        return await process_localization_from_json(source_path, image_id, **kwargs)
    elif suffix == ".txt":
        # Try to detect if it's TSV or single keypoint
        with open(source_path, "r") as f:
            first_line = f.readline().strip()

        if "\t" in first_line:
            return await process_localization_from_tsv(source_path, image_id, **kwargs)
        else:
            return [await process_localization_from_text_keypoint(source_path, image_id, **kwargs)]
    else:
        raise ValueError(f"Unsupported file format: {suffix}")
