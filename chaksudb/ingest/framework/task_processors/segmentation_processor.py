"""
Segmentation processor for mask annotations.

This module provides high-level processing functions for segmentation annotations
that handle various mask formats, UUID generation, and model preparation.

Key features:
- Wraps existing mask_converter functions (binary masks, contours, XML, soft maps, layer boundaries)
- Converts masks to SegmentationAnnotation models
- Generates deterministic UUIDs
- Returns models ready for upsert
- Handles annotation type registration
- Automatic provenance tracking via context variables
"""

import cv2
import hashlib
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from chaksudb.db.models import AnnotationType, SegmentationAnnotation
from chaksudb.db.queries.annotation_types import upsert_annotation_type
from chaksudb.ingest.framework.gen_uuid import (
    generate_annotation_type_uuid,
    generate_dataset_uuid,
    generate_segmentation_uuid,
)
from chaksudb.ingest.framework.raw_file_helpers import register_individual_file
from chaksudb.ingest.framework.mask_converter import (
    convert_contour_to_binary_mask,
    convert_contour_to_binary_mask_async,
    extract_class_from_mask,
    extract_classes_from_multiclass_mask,
    is_multiclass_mask,
    load_layer_boundaries,
    load_soft_map,
    parse_xml_annotations,
    parse_xml_polygon_to_binary_mask,
    validate_binary_mask,
)
from chaksudb.ingest.framework.provenance_context import get_current_provenance
from chaksudb.ingest.framework.transformations import log_and_link_transformation
from chaksudb.storage.paths import (
    generate_storage_path,
    compute_relative_path,
    get_storage_root,
)
from chaksudb.storage import create_local_locator
from chaksudb.config.config import get_data_root

logger = logging.getLogger(__name__)


# ============================================
# Helper Functions
# ============================================


async def get_or_create_annotation_type(
    annotation_type: str,
    annotation_description: Optional[str] = None,
) -> uuid.UUID:
    """
    Get or create an annotation type (idempotent).

    If the annotation type exists, returns its UUID. If it doesn't exist, creates it.

    Args:
        annotation_type: Name of the annotation type (e.g., 'optic_disc', 'microaneurysms')
        annotation_description: Optional description of the annotation type

    Returns:
        annotation_type_id UUID

    Example:
        ```python
        annotation_type_id = await get_or_create_annotation_type(
            annotation_type="optic_disc",
            annotation_description="Optic disc segmentation"
        )
        ```
    """
    # Generate deterministic UUID for the annotation type
    annotation_type_id = generate_annotation_type_uuid(annotation_type)

    # Create annotation type record (upsert is idempotent)
    annotation_type_obj = AnnotationType(
        annotation_type_id=annotation_type_id,
        annotation_type=annotation_type,
        annotation_description=annotation_description,
    )

    await upsert_annotation_type(annotation_type_obj)

    logger.debug(f"Registered annotation type: {annotation_type}")

    return annotation_type_id

from chaksudb.ingest.framework.hashing import compute_content_hash
def compute_mask_hash(mask: np.ndarray) -> str:
    """
    Compute a deterministic hash of a mask array.

    Args:
        mask: Binary mask array

    Returns:
        SHA256 hash of the mask data
    """
    return compute_content_hash(mask.tobytes())


def _build_locator_file_path(path: Path, root: Optional[Path] = None) -> str:
    """
    Build a locator-normalized file path string for storage.

    Always returns a path relative to a known root (never an absolute path).
    Tries, in order: ``root`` (if provided), data root, storage root, current
    working directory. Uses the first that contains ``path``. If none contain
    it (e.g. path on another drive), returns a fallback relative form and logs
    a warning.
    """
    resolved = path.resolve()
    roots_to_try: List[Path] = []
    if root is not None:
        roots_to_try.append(Path(root).resolve())
    roots_to_try.append(get_data_root().resolve())
    roots_to_try.append(get_storage_root().resolve())
    roots_to_try.append(Path.cwd().resolve())

    for candidate_root in roots_to_try:
        try:
            relative_str = compute_relative_path(resolved, candidate_root)
            target_path = Path(relative_str)
            locator = create_local_locator(target_path)
            return locator.file_path
        except ValueError:
            continue

    # Path is not under any known root (e.g. different drive); store a safe relative form
    logger.warning(
        "Mask path %s is not under data root, storage root, or cwd; "
        "storing relative fallback (filename only).",
        path,
    )
    fallback = Path("external") / path.name
    return create_local_locator(fallback).file_path


def save_processed_mask(
    mask: np.ndarray,
    dataset_name: str,
    annotation_type: str,
    segmentation_id: uuid.UUID,
    is_soft_map: bool = False,
) -> str:
    """
    Save a processed/converted mask to the processed directory.
    
    Only called when there's an actual transformation (contour→mask, XML→mask,
    class extraction, merge, or hole filling), not for simple validation.
    
    Args:
        mask: Mask array to save (binary uint8 0-255, or float32 0-1 for soft maps)
        dataset_name: Name of the dataset
        annotation_type: Type of annotation (e.g., 'optic_disc', 'attention_map')
        segmentation_id: UUID of the segmentation annotation
        is_soft_map: If True, saves as PNG with 0-1 values mapped to 0-255
    
    Returns:
        Locator-normalized path string suitable for DB storage. When the
        mask is saved under the configured storage root, the path will be
        relative to that root and use forward slashes.
        
    Example:
        >>> relative_path = save_processed_mask(
        ...     mask=binary_mask,
        ...     dataset_name="ORIGA",
        ...     annotation_type="optic_disc",
        ...     segmentation_id=uuid.uuid4(),
        ...     is_soft_map=False,
        ... )
        >>> # Returns: "ORIGA/masks/optic_disc/1a2b3c4d.png"
    """
    # ------------------------------------------------------------------
    # 1. Generate canonical output path via storage layer
    # ------------------------------------------------------------------
    output_path = generate_storage_path(
        dataset_name=dataset_name,
        subdirectory=f"masks/{annotation_type}",
        filename=f"{str(segmentation_id)[:8]}.png",
    )

    # ------------------------------------------------------------------
    # 2. Prepare mask for saving
    # ------------------------------------------------------------------
    if is_soft_map:
        # Soft map: ensure 0–1 range, convert to 0–255 for PNG
        if mask.dtype in (np.float32, np.float64):
            mask_to_save = (mask * 255).astype(np.uint8)
        else:
            mask_to_save = mask
    else:
        # Binary mask: ensure uint8
        mask_to_save = mask.astype(np.uint8)

    # ------------------------------------------------------------------
    # 3. Write file
    # ------------------------------------------------------------------
    success = cv2.imwrite(str(output_path), mask_to_save)
    if not success:
        logger.error(f"Failed to write mask to {output_path}")
        raise IOError(f"OpenCV failed to write mask: {output_path}")

    # ------------------------------------------------------------------
    # 4. Return DB-safe locator path (relative to storage root when possible)
    # ------------------------------------------------------------------
    mask_file_path = _build_locator_file_path(output_path, root=get_storage_root())
    logger.debug(f"Saved processed mask to {mask_file_path}")
    return mask_file_path


async def _ensure_provenance_from_source(
    source_path: Path,
    dataset_id: Optional[uuid.UUID],
    dataset_name: Optional[str],
    raw_data_id: Optional[uuid.UUID],
    provenance_chain_id: Optional[uuid.UUID],
) -> Tuple[Optional[uuid.UUID], Optional[uuid.UUID]]:
    """
    If provenance is missing but we have source_path and (dataset_id or dataset_name),
    register the source file as a raw annotation and return (raw_data_id, provenance_chain_id).
    Otherwise return the existing values unchanged.

    Prefer passing dataset_id from ingest scripts so the same UUID as the upserted dataset
    is used (satisfies raw_annotation_files.fk_raw_files_dataset).
    """
    if raw_data_id is not None and provenance_chain_id is not None:
        return raw_data_id, provenance_chain_id
    if not source_path.exists():
        return raw_data_id, provenance_chain_id
    resolved_dataset_id = dataset_id
    if resolved_dataset_id is None and dataset_name:
        resolved_dataset_id = generate_dataset_uuid(dataset_name)
    if resolved_dataset_id is None:
        return raw_data_id, provenance_chain_id
    raw_file_id, chain_id = await register_individual_file(
        file_path=source_path,
        dataset_id=resolved_dataset_id,
        unified_annotation_type="segmentation",
    )
    return raw_file_id, chain_id


# ============================================
# Main Processing Functions
# ============================================


async def process_segmentation_from_binary_mask(
    mask_path: Path,
    annotation_type: str,
    image_id: uuid.UUID,
    annotation_description: Optional[str] = None,
    lesion_subtype: Optional[str] = None,
    extract_class: Optional[int] = None,
    merge_nonzero: bool = False,
    fill_holes: bool = False,
    group_id: Optional[uuid.UUID] = None,
    raw_data_id: Optional[uuid.UUID] = None,
    expert_annotation_id: Optional[uuid.UUID] = None,
    consensus_id: Optional[uuid.UUID] = None,
    annotation_method: str = "manual",
    confidence_score: Optional[float] = None,
    provenance_chain_id: Optional[uuid.UUID] = None,
    dataset_name: Optional[str] = None,
    dataset_id: Optional[uuid.UUID] = None,
    original_source_path: Optional[Path] = None,
) -> SegmentationAnnotation:
    """
    Process segmentation from binary mask file and prepare for database upsert.

    Wraps validate_binary_mask() and converts output to SegmentationAnnotation model.

    Args:
        mask_path: Path to binary mask file
        annotation_type: Type of annotation (e.g., 'optic_disc', 'microaneurysms')
        image_id: UUID of the image
        annotation_description: Optional description for annotation type
        lesion_subtype: Optional lesion subtype
        extract_class: Optional class ID to extract from multi-class mask
        merge_nonzero: If True, merge all non-zero values into foreground
        fill_holes: If True, fill holes in the mask (WARNING: don't use for vessels!)
        group_id: Optional group UUID for related masks
        raw_data_id: Optional raw annotation file UUID
        expert_annotation_id: Optional expert annotation UUID
        consensus_id: Optional consensus UUID
        annotation_method: Method ('manual', 'semi_automatic', 'automatic', 'pseudo')
        confidence_score: Optional confidence score
        provenance_chain_id: Optional provenance chain UUID
        dataset_name: Optional dataset name for saving processed masks to processed/ directory
        dataset_id: Optional dataset UUID (used with source path to ensure provenance when not set)
        original_source_path: When mask_path is a temp/scratch file, pass the real source path
            here; it will be stored as original_file_path (provenance) instead of mask_path.

    Returns:
        SegmentationAnnotation model ready for upsert

    Example:
        ```python
        annotation = await process_segmentation_from_binary_mask(
            mask_path=Path("masks/optic_disc_001.png"),
            annotation_type="optic_disc",
            image_id=image_id,
            annotation_description="Optic disc segmentation",
            dataset_name="ORIGA",
        )
        await upsert_segmentation_annotation(annotation)
        ```
    """
    # Get provenance from context if not explicitly provided
    if raw_data_id is None or provenance_chain_id is None:
        context_raw_id, context_chain_id = get_current_provenance()
        raw_data_id = raw_data_id or context_raw_id
        provenance_chain_id = provenance_chain_id or context_chain_id

    # Ensure provenance when still missing but we have source path and dataset
    # Use original_source_path when provided so we register the real file, not a temp path
    provenance_source = original_source_path if original_source_path is not None else mask_path
    raw_data_id, provenance_chain_id = await _ensure_provenance_from_source(
        source_path=provenance_source,
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        raw_data_id=raw_data_id,
        provenance_chain_id=provenance_chain_id,
    )
    
    # Validate annotation_method
    valid_methods = {"manual", "semi_automatic", "automatic", "pseudo"}
    if annotation_method not in valid_methods:
        raise ValueError(
            f"Invalid annotation_method: {annotation_method}. "
            f"Must be one of {valid_methods}"
        )
    
    # Warning: fill_holes should not be used for vessel segmentation
    if fill_holes and "vessel" in annotation_type.lower():
        logger.warning(
            f"fill_holes=True used for annotation_type='{annotation_type}'. "
            "This may destroy vessel data by filling natural gaps in vessels!"
        )

    # Get or create annotation type
    annotation_type_id = await get_or_create_annotation_type(
        annotation_type=annotation_type,
        annotation_description=annotation_description,
    )

    # Load and validate binary mask
    mask = validate_binary_mask(
        mask_path=mask_path,
        extract_class=extract_class,
        merge_nonzero=merge_nonzero,
        fill_holes=fill_holes,
    )

    # Generate deterministic UUID
    segmentation_id = generate_segmentation_uuid(
        image_id=image_id,
        annotation_type_id=annotation_type_id,
        expert_annotation_id=expert_annotation_id,
        consensus_id=consensus_id,
        raw_data_id=raw_data_id,
        lesion_subtype=lesion_subtype,
    )

    # Check if any processing/transformation was applied
    processing_applied = extract_class is not None or merge_nonzero or fill_holes

    # Compute DB path for the original source file (e.g. .mat used to build a temp mask)
    path_for_provenance = original_source_path if original_source_path is not None else mask_path
    original_file_path_str = _build_locator_file_path(
        path_for_provenance,
        root=get_data_root(),
    )

    # Save to processed/ if dataset_name provided (for standardization to PNG)
    # OR if transformation occurred
    if dataset_name:
        final_mask_path = save_processed_mask(
            mask=mask,
            dataset_name=dataset_name,
            annotation_type=annotation_type,
            segmentation_id=segmentation_id,
            is_soft_map=False,
        )
        if processing_applied:
            logger.debug(f"Saved processed mask to {final_mask_path}")
        else:
            logger.debug(f"Saved standardized mask to {final_mask_path}")
    else:
        # No dataset_name - point mask_file_path at the original mask
        final_mask_path = _build_locator_file_path(
            mask_path,
            root=get_data_root(),
        )
        if processing_applied:
            logger.debug(
                "Processing applied on binary mask but not saved (no dataset_name "
                "provided); mask_file_path points to original mask."
            )

    # Determine format information (use source path suffix when provided for provenance)
    original_format = path_for_provenance.suffix.lstrip(".")
    unified_format = "binary_mask"

    # Create model
    annotation = SegmentationAnnotation(
        segmentation_id=segmentation_id,
        image_id=image_id,
        annotation_type_id=annotation_type_id,
        lesion_subtype=lesion_subtype,
        mask_file_path=final_mask_path,
        group_id=group_id,
        unified_format=unified_format,
        original_format=original_format,
        original_file_path=original_file_path_str,
        raw_data_id=raw_data_id,
        coordinate_system="pixel",
        expert_annotation_id=expert_annotation_id,
        consensus_id=consensus_id,
        annotation_method=annotation_method,
        confidence_score=confidence_score,
        provenance_chain_id=provenance_chain_id,
    )

    logger.debug(
        f"Processed binary mask segmentation for {annotation_type} in image {image_id}"
    )

    # Log transformation when processing was applied or mask was saved to processed/
    if provenance_chain_id and (processing_applied or dataset_name):
        await log_and_link_transformation(
            chain_id=provenance_chain_id,
            transformation_type="segmentation_binary_mask",
            input_data={
                "original_file_path": original_file_path_str,
                "dataset_name": dataset_name,
            },
            output_data={
                "mask_file_path": final_mask_path,
                "segmentation_id": str(segmentation_id),
                "annotation_type": annotation_type,
            },
            parameters={
                "extract_class": extract_class,
                "merge_nonzero": merge_nonzero,
                "fill_holes": fill_holes,
            },
            operator="segmentation_processor",
            notes="Binary mask validation and optional postprocessing (extract_class, merge_nonzero, fill_holes) or standardization to processed/.",
        )

    return annotation


async def process_segmentation_from_multiclass_mask(
    mask_path: Path,
    class_names: Dict[int, str],
    image_id: uuid.UUID,
    classes_to_extract: Optional[List[int]] = None,
    fill_holes: bool = False,
    group_id: Optional[uuid.UUID] = None,
    raw_data_id: Optional[uuid.UUID] = None,
    expert_annotation_id: Optional[uuid.UUID] = None,
    consensus_id: Optional[uuid.UUID] = None,
    annotation_method: str = "manual",
    confidence_score: Optional[float] = None,
    provenance_chain_id: Optional[uuid.UUID] = None,
    dataset_name: Optional[str] = None,
    dataset_id: Optional[uuid.UUID] = None,
    original_source_path: Optional[Path] = None,
) -> List[SegmentationAnnotation]:
    """
    Process segmentation from multi-class mask file.

    Extracts each class as a separate segmentation annotation.

    Args:
        mask_path: Path to multi-class mask file
        class_names: Dictionary mapping class IDs to annotation type names
                     (e.g., {1: "optic_disc", 2: "optic_cup"})
        image_id: UUID of the image
        classes_to_extract: Optional list of class IDs to extract
        fill_holes: If True, fill holes in extracted masks (WARNING: don't use for vessels!)
        group_id: Optional group UUID for related masks
        raw_data_id: Optional raw annotation file UUID
        expert_annotation_id: Optional expert annotation UUID
        consensus_id: Optional consensus UUID
        annotation_method: Method ('manual', 'semi_automatic', 'automatic', 'pseudo')
        confidence_score: Optional confidence score
        provenance_chain_id: Optional provenance chain UUID
        dataset_name: Optional dataset name for saving processed masks to processed/ directory
        dataset_id: Optional dataset UUID (used with source path to ensure provenance when not set)
        original_source_path: When mask_path is a temp/scratch file, pass the real source path
            here; it will be stored as original_file_path (provenance) instead of mask_path.

    Returns:
        List of SegmentationAnnotation models ready for upsert

    Example:
        ```python
        annotations = await process_segmentation_from_multiclass_mask(
            mask_path=Path("masks/disc_cup_001.png"),
            class_names={1: "optic_disc", 2: "optic_cup"},
            image_id=image_id,
            dataset_name="ORIGA",
        )
        for annotation in annotations:
            await upsert_segmentation_annotation(annotation)
        ```
    """
    # Get provenance from context if not explicitly provided
    if raw_data_id is None or provenance_chain_id is None:
        context_raw_id, context_chain_id = get_current_provenance()
        raw_data_id = raw_data_id or context_raw_id
        provenance_chain_id = provenance_chain_id or context_chain_id

    # Ensure provenance when still missing but we have source path and dataset
    # Use original_source_path when provided so we register the real file, not a temp path
    provenance_source = original_source_path if original_source_path is not None else mask_path
    raw_data_id, provenance_chain_id = await _ensure_provenance_from_source(
        source_path=provenance_source,
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        raw_data_id=raw_data_id,
        provenance_chain_id=provenance_chain_id,
    )

    # Validate annotation_method
    valid_methods = {"manual", "semi_automatic", "automatic", "pseudo"}
    if annotation_method not in valid_methods:
        raise ValueError(
            f"Invalid annotation_method: {annotation_method}. "
            f"Must be one of {valid_methods}"
        )
    
    # Warning: fill_holes should not be used for vessel segmentation
    if fill_holes:
        for class_id, class_name in class_names.items():
            if "vessel" in class_name.lower():
                logger.warning(
                    f"fill_holes=True used for class '{class_name}'. "
                    "This may destroy vessel data by filling natural gaps in vessels!"
                )

    # Extract classes from multi-class mask (TRANSFORMATION HAPPENING)
    masks = extract_classes_from_multiclass_mask(
        mask_path=mask_path,
        class_names=class_names,
        classes_to_extract=classes_to_extract,
        fill_holes=fill_holes,
    )

    annotations = []

    for annotation_type, mask in masks.items():
        # Get or create annotation type
        annotation_type_id = await get_or_create_annotation_type(
            annotation_type=annotation_type,
            annotation_description=f"{annotation_type} segmentation",
        )

        # Generate deterministic UUID
        segmentation_id = generate_segmentation_uuid(
            image_id=image_id,
            annotation_type_id=annotation_type_id,
            expert_annotation_id=expert_annotation_id,
            consensus_id=consensus_id,
            raw_data_id=raw_data_id,
        )
        
        # ALWAYS save extracted class if dataset_name provided (extraction = transformation)
        if dataset_name:
            final_mask_path = save_processed_mask(
                mask=mask,
                dataset_name=dataset_name,
                annotation_type=annotation_type,
                segmentation_id=segmentation_id,
                is_soft_map=False,
            )
            logger.debug(f"Saved extracted class '{annotation_type}' to {final_mask_path}")
        else:
            # No dataset_name – point mask_file_path at the original multi-class mask
            final_mask_path = _build_locator_file_path(
                mask_path,
                root=get_data_root(),
            )
            logger.warning(
                "Multi-class extraction done but not saved (no dataset_name); "
                "mask_file_path points to original multi-class mask."
            )

        # Determine format information (use source path when provided for provenance)
        path_for_provenance = original_source_path if original_source_path is not None else mask_path
        original_format = path_for_provenance.suffix.lstrip(".")
        unified_format = "binary_mask"
        
        # Compute locator-normalized path for original file under data root where possible
        original_file_path_str = _build_locator_file_path(
            path_for_provenance,
            root=get_data_root(),
        )

        # Create model
        annotation = SegmentationAnnotation(
            segmentation_id=segmentation_id,
            image_id=image_id,
            annotation_type_id=annotation_type_id,
            lesion_subtype=annotation_type,
            mask_file_path=final_mask_path,
            group_id=group_id,
            unified_format=unified_format,
            original_format=original_format,
            original_file_path=original_file_path_str,
            raw_data_id=raw_data_id,
            coordinate_system="pixel",
            expert_annotation_id=expert_annotation_id,
            consensus_id=consensus_id,
            annotation_method=annotation_method,
            confidence_score=confidence_score,
            provenance_chain_id=provenance_chain_id,
        )
        annotations.append(annotation)

    logger.debug(
        f"Processed {len(annotations)} segmentation annotations from multi-class mask {mask_path}"
    )

    # Log transformation (multi-class extraction is always a transformation)
    # Use same path as stored in annotations (original source when provided)
    path_for_log = original_source_path if original_source_path is not None else mask_path
    original_file_path_str = _build_locator_file_path(path_for_log, root=get_data_root())
    if provenance_chain_id:
        await log_and_link_transformation(
            chain_id=provenance_chain_id,
            transformation_type="segmentation_multiclass_extract",
            input_data={
                "original_file_path": original_file_path_str,
                "dataset_name": dataset_name,
            },
            output_data={
                "mask_file_paths": [a.mask_file_path for a in annotations],
                "segmentation_ids": [str(a.segmentation_id) for a in annotations],
                "annotation_types": list(masks.keys()),
            },
            parameters={
                "classes_to_extract": classes_to_extract,
                "fill_holes": fill_holes,
                "class_names": class_names,
            },
            operator="segmentation_processor",
            notes="Extracted per-class binary masks from multi-class mask.",
        )

    return annotations


async def process_segmentation_from_contour(
    contour_path: Path,
    annotation_type: str,
    image_id: uuid.UUID,
    image_size: Tuple[int, int],
    annotation_description: Optional[str] = None,
    lesion_subtype: Optional[str] = None,
    coordinate_format: Optional[str] = None,
    group_id: Optional[uuid.UUID] = None,
    raw_data_id: Optional[uuid.UUID] = None,
    expert_annotation_id: Optional[uuid.UUID] = None,
    consensus_id: Optional[uuid.UUID] = None,
    annotation_method: str = "manual",
    confidence_score: Optional[float] = None,
    provenance_chain_id: Optional[uuid.UUID] = None,
    dataset_name: Optional[str] = None,
    dataset_id: Optional[uuid.UUID] = None,
) -> SegmentationAnnotation:
    """
    Process segmentation from contour/polygon file.

    Converts contour coordinates to binary mask. Uses async operations to prevent
    blocking the event loop. Automatically uses GPU acceleration if available,
    otherwise falls back to CPU.

    Args:
        contour_path: Path to contour file (text or JSON)
        annotation_type: Type of annotation (e.g., 'optic_disc')
        image_id: UUID of the image
        image_size: (width, height) of target image
        annotation_description: Optional description for annotation type
        lesion_subtype: Optional lesion subtype
        coordinate_format: Optional format hint ('line_separated', 'space_separated', etc.)
        group_id: Optional group UUID for related masks
        raw_data_id: Optional raw annotation file UUID
        expert_annotation_id: Optional expert annotation UUID
        consensus_id: Optional consensus UUID
        annotation_method: Method ('manual', 'semi_automatic', 'automatic', 'pseudo')
        confidence_score: Optional confidence score
        provenance_chain_id: Optional provenance chain UUID
        dataset_name: Optional dataset name for saving processed masks to processed/ directory
        dataset_id: Optional dataset UUID (used with source path to ensure provenance when not set)

    Returns:
        SegmentationAnnotation model ready for upsert

    Example:
        ```python
        annotation = await process_segmentation_from_contour(
            contour_path=Path("contours/disc_001.txt"),
            annotation_type="optic_disc",
            image_id=image_id,
            image_size=(1152, 1500),
            dataset_name="Drishti-GS1",
        )
        await upsert_segmentation_annotation(annotation)
        ```
    """
    # Get provenance from context if not explicitly provided
    if raw_data_id is None or provenance_chain_id is None:
        context_raw_id, context_chain_id = get_current_provenance()
        raw_data_id = raw_data_id or context_raw_id
        provenance_chain_id = provenance_chain_id or context_chain_id

    # Ensure provenance when still missing but we have source path and dataset
    raw_data_id, provenance_chain_id = await _ensure_provenance_from_source(
        source_path=contour_path,
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        raw_data_id=raw_data_id,
        provenance_chain_id=provenance_chain_id,
    )

    # Validate annotation_method
    valid_methods = {"manual", "semi_automatic", "automatic", "pseudo"}
    if annotation_method not in valid_methods:
        raise ValueError(
            f"Invalid annotation_method: {annotation_method}. "
            f"Must be one of {valid_methods}"
        )

    # Get or create annotation type
    annotation_type_id = await get_or_create_annotation_type(
        annotation_type=annotation_type,
        annotation_description=annotation_description,
    )

    # Convert contour to binary mask (TRANSFORMATION HAPPENING)
    # Use async version to prevent blocking the event loop
    # GPU acceleration is automatically used if available
    mask = await convert_contour_to_binary_mask_async(
        contour_path=contour_path,
        image_size=image_size,
        coordinate_format=coordinate_format,
    )

    # Generate deterministic UUID
    segmentation_id = generate_segmentation_uuid(
        image_id=image_id,
        annotation_type_id=annotation_type_id,
        expert_annotation_id=expert_annotation_id,
        consensus_id=consensus_id,
        raw_data_id=raw_data_id,
        lesion_subtype=lesion_subtype,
    )

    # ALWAYS save converted mask if dataset_name provided (conversion = transformation)
    if dataset_name:
        final_mask_path = save_processed_mask(
            mask=mask,
            dataset_name=dataset_name,
            annotation_type=annotation_type,
            segmentation_id=segmentation_id,
            is_soft_map=False,
        )
        logger.debug(f"Saved converted contour→mask to {final_mask_path}")
    else:
        # Backwards compatibility - no path (mask only in memory)
        final_mask_path = None
        logger.warning(f"Contour converted to mask but not saved (no dataset_name)")

    # Determine format information
    original_format = contour_path.suffix.lstrip(".")
    unified_format = "binary_mask"

    # Compute locator-normalized path for original contour file
    original_file_path_str = _build_locator_file_path(
        contour_path,
        root=get_data_root(),
    )

    # Create model
    annotation = SegmentationAnnotation(
        segmentation_id=segmentation_id,
        image_id=image_id,
        annotation_type_id=annotation_type_id,
        lesion_subtype=lesion_subtype,
        mask_file_path=final_mask_path,
        group_id=group_id,
        unified_format=unified_format,
        original_format=f"contour_{original_format}",
        original_file_path=original_file_path_str,
        raw_data_id=raw_data_id,
        coordinate_system="pixel",
        expert_annotation_id=expert_annotation_id,
        consensus_id=consensus_id,
        annotation_method=annotation_method,
        confidence_score=confidence_score,
        provenance_chain_id=provenance_chain_id,
    )

    logger.debug(
        f"Processed contour segmentation for {annotation_type} in image {image_id}"
    )

    # Log transformation (contour→mask conversion)
    if provenance_chain_id:
        await log_and_link_transformation(
            chain_id=provenance_chain_id,
            transformation_type="segmentation_contour_to_mask",
            input_data={
                "original_file_path": original_file_path_str,
                "dataset_name": dataset_name,
            },
            output_data={
                "mask_file_path": final_mask_path,
                "segmentation_id": str(segmentation_id),
                "annotation_type": annotation_type,
            },
            parameters={
                "image_size": image_size,
                "coordinate_format": coordinate_format,
            },
            operator="segmentation_processor",
            notes="Converted contour/polygon coordinates to binary mask.",
        )

    return annotation


async def process_segmentation_from_xml(
    xml_path: Path,
    annotation_type: str,
    image_id: uuid.UUID,
    image_size: Tuple[int, int],
    annotation_description: Optional[str] = None,
    lesion_subtype: Optional[str] = None,
    class_filter: Optional[Union[str, List[str]]] = None,
    group_id: Optional[uuid.UUID] = None,
    raw_data_id: Optional[uuid.UUID] = None,
    expert_annotation_id: Optional[uuid.UUID] = None,
    consensus_id: Optional[uuid.UUID] = None,
    annotation_method: str = "manual",
    confidence_score: Optional[float] = None,
    provenance_chain_id: Optional[uuid.UUID] = None,
    dataset_name: Optional[str] = None,
    dataset_id: Optional[uuid.UUID] = None,
    **kwargs,
) -> List[SegmentationAnnotation]:
    """
    Process segmentation from XML annotation file.

    Supports multiple XML formats (PASCAL VOC, ImageRet, custom polygons).

    Args:
        xml_path: Path to XML annotation file
        annotation_type: Type of annotation (e.g., 'hemorrhages', 'exudates')
        image_id: UUID of the image
        image_size: (width, height) of target image
        annotation_description: Optional description for annotation type
        lesion_subtype: Optional lesion subtype
        class_filter: Optional class name(s) to filter
        group_id: Optional group UUID for related masks
        raw_data_id: Optional raw annotation file UUID
        expert_annotation_id: Optional expert annotation UUID
        consensus_id: Optional consensus UUID
        annotation_method: Method ('manual', 'semi_automatic', 'automatic', 'pseudo')
        confidence_score: Optional confidence score
        provenance_chain_id: Optional provenance chain UUID
        dataset_name: Optional dataset name for saving processed masks to processed/ directory
        dataset_id: Optional dataset UUID (used with source path to ensure provenance when not set)
        **kwargs: Additional format-specific options

    Returns:
        List of SegmentationAnnotation models ready for upsert

    Example:
        ```python
        annotations = await process_segmentation_from_xml(
            xml_path=Path("annotations/0400.xml"),
            annotation_type="lesions",
            image_id=image_id,
            image_size=(1152, 1500),
            class_filter=["Haemorrhages", "Hard_exudates"],
            dataset_name="ImageRet",
        )
        for annotation in annotations:
            await upsert_segmentation_annotation(annotation)
        ```
    """
    # Get provenance from context if not explicitly provided
    if raw_data_id is None or provenance_chain_id is None:
        context_raw_id, context_chain_id = get_current_provenance()
        raw_data_id = raw_data_id or context_raw_id
        provenance_chain_id = provenance_chain_id or context_chain_id

    # Ensure provenance when still missing but we have source path and dataset
    raw_data_id, provenance_chain_id = await _ensure_provenance_from_source(
        source_path=xml_path,
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        raw_data_id=raw_data_id,
        provenance_chain_id=provenance_chain_id,
    )

    # Validate annotation_method
    valid_methods = {"manual", "semi_automatic", "automatic", "pseudo"}
    if annotation_method not in valid_methods:
        raise ValueError(
            f"Invalid annotation_method: {annotation_method}. "
            f"Must be one of {valid_methods}"
        )

    # Parse XML to get masks by class (CONVERSION HAPPENING)
    masks = parse_xml_annotations(
        xml_path=xml_path,
        image_size=image_size,
        output_format="masks_by_class",
        class_filter=class_filter,
        **kwargs,
    )

    annotations = []

    for class_name, mask in masks.items():
        # Use class name as annotation type if no specific type provided
        ann_type = class_name if annotation_type == "lesions" else annotation_type
        ann_lesion_subtype = lesion_subtype or class_name

        # Get or create annotation type
        annotation_type_id = await get_or_create_annotation_type(
            annotation_type=ann_type,
            annotation_description=annotation_description or f"{ann_type} segmentation",
        )

        # Generate deterministic UUID. Include lesion_subtype: when annotation_type
        # is fixed (not "lesions"), annotation_type_id is identical across classes,
        # so lesion_subtype/class_name is what keeps each class's segmentation_id
        # distinct (otherwise different classes would collide and overwrite each
        # other on upsert).
        segmentation_id = generate_segmentation_uuid(
            image_id=image_id,
            annotation_type_id=annotation_type_id,
            expert_annotation_id=expert_annotation_id,
            consensus_id=consensus_id,
            raw_data_id=raw_data_id,
            lesion_subtype=ann_lesion_subtype,
        )
        
        # ALWAYS save converted mask if dataset_name provided (conversion = transformation)
        if dataset_name:
            final_mask_path = save_processed_mask(
                mask=mask,
                dataset_name=dataset_name,
                annotation_type=ann_type,
                segmentation_id=segmentation_id,
                is_soft_map=False,
            )
            logger.debug(f"Saved converted XML→mask to {final_mask_path}")
        else:
            final_mask_path = None
            logger.warning(f"XML converted to mask but not saved (no dataset_name)")

        # Compute locator-normalized path for original XML file
        original_file_path_str = _build_locator_file_path(
            xml_path,
            root=get_data_root(),
        )

        # Create model
        annotation = SegmentationAnnotation(
            segmentation_id=segmentation_id,
            image_id=image_id,
            annotation_type_id=annotation_type_id,
            lesion_subtype=ann_lesion_subtype,
            mask_file_path=final_mask_path,
            group_id=group_id,
            unified_format="binary_mask",
            original_format="xml",
            original_file_path=original_file_path_str,
            raw_data_id=raw_data_id,
            coordinate_system="pixel",
            expert_annotation_id=expert_annotation_id,
            consensus_id=consensus_id,
            annotation_method=annotation_method,
            confidence_score=confidence_score,
            provenance_chain_id=provenance_chain_id,
        )
        annotations.append(annotation)

    logger.debug(
        f"Processed {len(annotations)} segmentation annotations from XML {xml_path}"
    )

    # Log transformation (XML→mask conversion)
    xml_original_path_str = _build_locator_file_path(xml_path, root=get_data_root())
    if provenance_chain_id:
        await log_and_link_transformation(
            chain_id=provenance_chain_id,
            transformation_type="segmentation_xml_to_mask",
            input_data={
                "original_file_path": xml_original_path_str,
                "dataset_name": dataset_name,
            },
            output_data={
                "mask_file_paths": [a.mask_file_path for a in annotations],
                "segmentation_ids": [str(a.segmentation_id) for a in annotations],
                "annotation_types": list(masks.keys()),
            },
            parameters={
                "image_size": image_size,
                "class_filter": class_filter,
            },
            operator="segmentation_processor",
            notes="Converted XML annotations (polygons/regions) to binary masks by class.",
        )

    return annotations


async def process_segmentation_from_soft_map(
    soft_map_path: Path,
    annotation_type: str,
    image_id: uuid.UUID,
    annotation_description: Optional[str] = None,
    lesion_subtype: Optional[str] = None,
    group_id: Optional[uuid.UUID] = None,
    raw_data_id: Optional[uuid.UUID] = None,
    expert_annotation_id: Optional[uuid.UUID] = None,
    consensus_id: Optional[uuid.UUID] = None,
    annotation_method: str = "manual",
    confidence_score: Optional[float] = None,
    provenance_chain_id: Optional[uuid.UUID] = None,
) -> SegmentationAnnotation:
    """
    Process segmentation from soft probability map.

    Soft maps are stored as-is (not converted to binary).

    Args:
        soft_map_path: Path to soft map file
        annotation_type: Type of annotation (e.g., 'vessel_probability')
        image_id: UUID of the image
        annotation_description: Optional description for annotation type
        lesion_subtype: Optional lesion subtype
        group_id: Optional group UUID for related masks
        raw_data_id: Optional raw annotation file UUID
        expert_annotation_id: Optional expert annotation UUID
        consensus_id: Optional consensus UUID
        annotation_method: Method ('manual', 'semi_automatic', 'automatic', 'pseudo')
        confidence_score: Optional confidence score
        provenance_chain_id: Optional provenance chain UUID

    Returns:
        SegmentationAnnotation model ready for upsert

    Example:
        ```python
        annotation = await process_segmentation_from_soft_map(
            soft_map_path=Path("soft_maps/vessel_prob_001.png"),
            annotation_type="vessel_probability",
            image_id=image_id,
        )
        await upsert_segmentation_annotation(annotation)
        ```
    """
    # Validate annotation_method
    valid_methods = {"manual", "semi_automatic", "automatic", "pseudo"}
    if annotation_method not in valid_methods:
        raise ValueError(
            f"Invalid annotation_method: {annotation_method}. "
            f"Must be one of {valid_methods}"
        )

    # Get or create annotation type
    annotation_type_id = await get_or_create_annotation_type(
        annotation_type=annotation_type,
        annotation_description=annotation_description,
    )

    # Load soft map (validates it exists)
    soft_map = load_soft_map(soft_map_path)

    # Generate deterministic UUID
    segmentation_id = generate_segmentation_uuid(
        image_id=image_id,
        annotation_type_id=annotation_type_id,
        expert_annotation_id=expert_annotation_id,
        consensus_id=consensus_id,
        raw_data_id=raw_data_id,
        lesion_subtype=lesion_subtype,
    )

    # Determine format information
    original_format = soft_map_path.suffix.lstrip(".")
    unified_format = "soft_map"

    soft_map_file_path = _build_locator_file_path(
        soft_map_path,
        root=get_data_root(),
    )

    # Create model
    annotation = SegmentationAnnotation(
        segmentation_id=segmentation_id,
        image_id=image_id,
        annotation_type_id=annotation_type_id,
        lesion_subtype=lesion_subtype,
        mask_file_path=soft_map_file_path,
        group_id=group_id,
        unified_format=unified_format,
        original_format=original_format,
        original_file_path=soft_map_file_path,
        raw_data_id=raw_data_id,
        coordinate_system="pixel",
        expert_annotation_id=expert_annotation_id,
        consensus_id=consensus_id,
        annotation_method=annotation_method,
        confidence_score=confidence_score,
        provenance_chain_id=provenance_chain_id,
    )

    logger.debug(
        f"Processed soft map segmentation for {annotation_type} in image {image_id}"
    )

    return annotation


async def process_segmentation_from_layer_boundaries(
    boundary_path: Path,
    annotation_type: str,
    image_id: uuid.UUID,
    annotation_description: Optional[str] = None,
    lesion_subtype: Optional[str] = None,
    group_id: Optional[uuid.UUID] = None,
    raw_data_id: Optional[uuid.UUID] = None,
    expert_annotation_id: Optional[uuid.UUID] = None,
    consensus_id: Optional[uuid.UUID] = None,
    annotation_method: str = "manual",
    confidence_score: Optional[float] = None,
    provenance_chain_id: Optional[uuid.UUID] = None,
) -> SegmentationAnnotation:
    """
    Process segmentation from layer boundary file (OCT).

    Layer boundaries are stored as-is (not converted to binary masks).

    Args:
        boundary_path: Path to layer boundary file
        annotation_type: Type of annotation (e.g., 'retinal_layers')
        image_id: UUID of the image
        annotation_description: Optional description for annotation type
        lesion_subtype: Optional lesion subtype
        group_id: Optional group UUID for related masks
        raw_data_id: Optional raw annotation file UUID
        expert_annotation_id: Optional expert annotation UUID
        consensus_id: Optional consensus UUID
        annotation_method: Method ('manual', 'semi_automatic', 'automatic', 'pseudo')
        confidence_score: Optional confidence score
        provenance_chain_id: Optional provenance chain UUID

    Returns:
        SegmentationAnnotation model ready for upsert

    Example:
        ```python
        annotation = await process_segmentation_from_layer_boundaries(
            boundary_path=Path("boundaries/layers_001.txt"),
            annotation_type="retinal_layers",
            image_id=image_id,
        )
        await upsert_segmentation_annotation(annotation)
        ```
    """
    # Validate annotation_method
    valid_methods = {"manual", "semi_automatic", "automatic", "pseudo"}
    if annotation_method not in valid_methods:
        raise ValueError(
            f"Invalid annotation_method: {annotation_method}. "
            f"Must be one of {valid_methods}"
        )

    # Get or create annotation type
    annotation_type_id = await get_or_create_annotation_type(
        annotation_type=annotation_type,
        annotation_description=annotation_description,
    )

    # Load layer boundaries (validates it exists)
    boundaries = load_layer_boundaries(boundary_path)

    # Generate deterministic UUID
    segmentation_id = generate_segmentation_uuid(
        image_id=image_id,
        annotation_type_id=annotation_type_id,
        expert_annotation_id=expert_annotation_id,
        consensus_id=consensus_id,
        raw_data_id=raw_data_id,
        lesion_subtype=lesion_subtype,
    )

    # Determine format information
    original_format = boundary_path.suffix.lstrip(".")
    unified_format = "layer_boundaries"

    boundary_file_path = _build_locator_file_path(
        boundary_path,
        root=get_data_root(),
    )

    # Create model
    annotation = SegmentationAnnotation(
        segmentation_id=segmentation_id,
        image_id=image_id,
        annotation_type_id=annotation_type_id,
        lesion_subtype=lesion_subtype,
        mask_file_path=boundary_file_path,
        group_id=group_id,
        unified_format=unified_format,
        original_format=original_format,
        original_file_path=boundary_file_path,
        raw_data_id=raw_data_id,
        coordinate_system="pixel",
        expert_annotation_id=expert_annotation_id,
        consensus_id=consensus_id,
        annotation_method=annotation_method,
        confidence_score=confidence_score,
        provenance_chain_id=provenance_chain_id,
    )

    logger.debug(
        f"Processed layer boundary segmentation for {annotation_type} in image {image_id}"
    )

    return annotation


# ============================================
# Convenience Function (alias)
# ============================================


async def prepare_segmentation_for_upsert(
    source_path: Path,
    annotation_type: str,
    image_id: uuid.UUID,
    image_size: Optional[Tuple[int, int]] = None,
    **kwargs,
) -> Union[SegmentationAnnotation, List[SegmentationAnnotation]]:
    """
    Auto-detect format and process segmentation.

    Convenience function that auto-detects the file format and calls
    the appropriate processing function.

    Args:
        source_path: Path to annotation file
        annotation_type: Type of annotation
        image_id: UUID of the image
        image_size: Optional (width, height) for contour/XML conversion
        **kwargs: Additional arguments for specific processors

    Returns:
        SegmentationAnnotation model or list of models

    Example:
        ```python
        # Auto-detect and process any supported format
        annotations = await prepare_segmentation_for_upsert(
            source_path=Path("masks/optic_disc_001.png"),
            annotation_type="optic_disc",
            image_id=image_id,
        )
        # Returns single annotation or list depending on format
        if isinstance(annotations, list):
            for annotation in annotations:
                await upsert_segmentation_annotation(annotation)
        else:
            await upsert_segmentation_annotation(annotations)
        ```
    """
    suffix = source_path.suffix.lower()

    # Check if it's a multi-class mask
    if suffix in [".png", ".tif", ".tiff", ".jpg", ".jpeg"]:
        if is_multiclass_mask(source_path) and "class_names" in kwargs:
            return await process_segmentation_from_multiclass_mask(
                source_path, image_id=image_id, **kwargs
            )
        else:
            return await process_segmentation_from_binary_mask(
                source_path, annotation_type, image_id, **kwargs
            )

    elif suffix == ".xml":
        if image_size is None:
            raise ValueError("image_size is required for XML annotations")
        return await process_segmentation_from_xml(
            source_path, annotation_type, image_id, image_size, **kwargs
        )

    elif suffix in [".txt", ".json"]:
        # Could be contour, layer boundaries, or soft map
        # Try to detect based on content or kwargs
        if image_size is not None:
            # Assume contour
            return await process_segmentation_from_contour(
                source_path, annotation_type, image_id, image_size, **kwargs
            )
        else:
            # Could be layer boundaries - let the function handle it
            return await process_segmentation_from_layer_boundaries(
                source_path, annotation_type, image_id, **kwargs
            )

    else:
        raise ValueError(f"Unsupported file format: {suffix}")

