"""
Grading scale registration and mapping utilities.

This module provides utilities for:
- Registering grading scales
- Storing disease grades with automatic mapping to standard scales
- Updating existing grades when new mappings are added
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from chaksudb.common.progress import OperationStatistics, ProgressTracker
from chaksudb.db.models import DiseaseGrading, GradingScale, GradingScaleMapping
from chaksudb.db.queries import (
    find_grading_scale_by_id,
    find_grading_scale_mapping_to_standard,
    get_all_disease_gradings_with_original_grade,
    upsert_disease_grading,
    upsert_grading_scale,
    upsert_grading_scale_mapping,
)
from chaksudb.ingest.framework.gen_uuid import (
    generate_disease_grading_uuid,
    generate_grading_scale_mapping_uuid,
    generate_grading_scale_uuid,
)

logger = logging.getLogger(__name__)


# ============================================
# Standard Scale Names (commonly used scales)
# ============================================

STANDARD_SCALES = {
    "ETDRS": "ETDRS",  # Early Treatment Diabetic Retinopathy Study
    "ICDR": "ICDR",  # International Clinical Diabetic Retinopathy
    "AAO": "AAO",  # American Academy of Ophthalmology
}


async def register_grading_scale(
    scale_name: str,
    disease_type: str,
    scale_description: Optional[str] = None,
    min_value: Optional[int] = None,
    max_value: Optional[int] = None,
    value_labels: Optional[Dict[str, Any]] = None,
) -> uuid.UUID:
    """
    Register a new grading scale in the database.

    Args:
        scale_name: Name of the grading scale (e.g., 'ICDR', 'ETDRS', 'AAO')
        disease_type: Disease type ('DR', 'DME', 'Glaucoma', 'AMD')
        scale_description: Optional description of the scale
        min_value: Optional minimum value in the scale
        max_value: Optional maximum value in the scale
        value_labels: Optional dictionary mapping values to labels

    Returns:
        scale_id UUID of the registered scale

    Example:
        ```python
        scale_id = await register_grading_scale(
            scale_name="ICDR",
            disease_type="DR",
            scale_description="International Clinical Diabetic Retinopathy scale",
            min_value=0,
            max_value=4,
            value_labels={"0": "No DR", "1": "Mild", "2": "Moderate", "3": "Severe", "4": "Proliferative"}
        )
        ```
    """
    # Generate deterministic UUID for the scale
    scale_id = generate_grading_scale_uuid(scale_name=scale_name, disease_type=disease_type)

    # Create grading scale model
    scale = GradingScale(
        scale_id=scale_id,
        scale_name=scale_name,
        disease_type=disease_type,
        scale_description=scale_description,
        min_value=min_value,
        max_value=max_value,
        value_labels=value_labels,
    )

    # Store in database (idempotent upsert)
    await upsert_grading_scale(scale)

    logger.info(f"Registered grading scale {scale_name} for {disease_type} with scale_id {scale_id}")

    return scale_id


async def find_mapping_to_standard_scale(
    source_scale_id: uuid.UUID,
    source_value: str,
    target_scale_name: Optional[str] = None,
) -> Optional[GradingScaleMapping]:
    """
    Find a mapping from a source scale value to a standard scale.

    If target_scale_name is not provided, searches for mappings to any standard scale.
    Standard scales are: ETDRS, ICDR, AAO.

    Args:
        source_scale_id: UUID of the source grading scale
        source_value: Original grade value (string)
        target_scale_name: Optional target scale name to search for (if None, searches all standard scales)

    Returns:
        GradingScaleMapping if found, None otherwise
    """
    return await find_grading_scale_mapping_to_standard(
        source_scale_id=source_scale_id,
        source_value=source_value,
        target_scale_name=target_scale_name,
    )


async def find_scale_by_name(scale_name: str, disease_type: str) -> Optional[uuid.UUID]:
    """
    Find a grading scale by name and disease type.

    Args:
        scale_name: Name of the grading scale
        disease_type: Disease type

    Returns:
        scale_id UUID if found, None otherwise
    """
    scale_id = generate_grading_scale_uuid(scale_name=scale_name, disease_type=disease_type)
    scale = await find_grading_scale_by_id(scale_id=scale_id)
    return scale.scale_id if scale else None


async def store_disease_grading(
    image_id: uuid.UUID,
    disease_type: str,
    scale_id: uuid.UUID,
    original_grade: str,
    raw_data_id: Optional[uuid.UUID] = None,
    expert_annotation_id: Optional[uuid.UUID] = None,
    consensus_id: Optional[uuid.UUID] = None,
    annotation_method: str = "manual",
    confidence_score: Optional[float] = None,
    provenance_chain_id: Optional[uuid.UUID] = None,
    grade_label: Optional[str] = None,
) -> uuid.UUID:
    """
    Store a disease grading, automatically checking for mappings to standard scales.

    Before storing, this function queries grading_scale_mappings to see if there's
    a mapping to a standard scale. If a mapping exists, both original_grade and
    scaled_grade are stored. If no mapping exists, only original_grade is stored.

    Args:
        image_id: UUID of the image
        disease_type: Disease type ('DR', 'DME', 'Glaucoma', 'AMD')
        scale_id: UUID of the grading scale used
        original_grade: Original grade value (string)
        raw_data_id: Optional raw annotation file UUID
        expert_annotation_id: Optional expert annotation UUID
        consensus_id: Optional consensus annotation UUID
        annotation_method: Annotation method ('manual', 'adjudicated', 'consensus', 'pseudo')
        confidence_score: Optional confidence score
        provenance_chain_id: Optional provenance chain UUID
        grade_label: Optional grade label

    Returns:
        grading_id UUID of the stored grading

    Example:
        ```python
        grading_id = await store_disease_grading(
            image_id=image_id,
            disease_type="DR",
            scale_id=icdr_scale_id,
            original_grade="Mild",
            raw_data_id=raw_file_id,
            provenance_chain_id=chain_id
        )
        ```
    """
    # Check for mapping to standard scale
    mapping = await find_grading_scale_mapping_to_standard(
        source_scale_id=scale_id,
        source_value=original_grade,
    )

    scaled_grade = None
    if mapping and mapping.target_value is not None:
        scaled_grade = mapping.target_value
        logger.debug(
            f"Found mapping for {original_grade} from scale {scale_id} "
            f"to standard scale {mapping.target_scale_id} with value {scaled_grade}"
        )

    # Generate deterministic UUID for the grading
    grading_id = generate_disease_grading_uuid(
        image_id=image_id,
        disease_type=disease_type,
        scale_id=scale_id,
        expert_annotation_id=expert_annotation_id,
        consensus_id=consensus_id,
        raw_data_id=raw_data_id,
        original_grade=original_grade,
    )

    # Create disease grading model
    grading = DiseaseGrading(
        grading_id=grading_id,
        image_id=image_id,
        disease_type=disease_type,
        scale_id=scale_id,
        original_grade=original_grade,
        scaled_grade=scaled_grade,
        grade_label=grade_label,
        raw_data_id=raw_data_id,
        expert_annotation_id=expert_annotation_id,
        consensus_id=consensus_id,
        annotation_method=annotation_method,
        confidence_score=confidence_score,
        provenance_chain_id=provenance_chain_id,
        created_at=datetime.now(),
        updated_at=None,
    )

    # Store in database (idempotent upsert)
    await upsert_disease_grading(grading)

    if scaled_grade is not None:
        logger.debug(
            f"Stored disease grading {grading_id} with original_grade={original_grade} "
            f"and scaled_grade={scaled_grade}"
        )
    else:
        logger.debug(
            f"Stored disease grading {grading_id} with original_grade={original_grade} "
            f"(no mapping to standard scale found)"
        )

    return grading_id


async def update_all_grades(
    tracker: Optional[ProgressTracker] = None,
) -> Tuple[int, int]:
    """
    Update all stored disease_grades by checking mappings and recalculating scaled_grade.

    This function:
    1. Queries all disease_grades that have original_grade but may need updated scaled_grade
    2. For each grading, checks grading_scale_mappings for a mapping to a standard scale
    3. Updates the scaled_grade if a mapping is found
    4. Updates the updated_at timestamp

    Args:
        tracker: Optional ProgressTracker for monitoring progress

    Returns:
        Tuple of (updated_count, total_checked_count)

    Example:
        ```python
        # With tracking
        tracker = ProgressTracker(total=1000, description="Updating grades")
        updated, total = await update_all_grades(tracker=tracker)
        tracker.finish()
        
        # Without tracking
        updated, total = await update_all_grades()
        print(f"Updated {updated} out of {total} grades")
        ```
    """
    # Get all disease gradings with original_grade
    rows = await get_all_disease_gradings_with_original_grade()

    updated_count = 0
    total_checked = len(rows)

    # Create internal tracker if not provided
    internal_tracker = tracker or ProgressTracker(
        total=total_checked, description="Updating disease grades"
    )

    internal_tracker.log(f"Checking {total_checked} disease grades for mapping updates")

    for row in rows:
        grading_id = row["grading_id"]
        scale_id = row["scale_id"]
        original_grade = row["original_grade"]
        current_scaled_grade = row["scaled_grade"]

        try:
            disease_type = row["disease_type"]
            target_scale_id = generate_grading_scale_uuid("ICDR_0_4", disease_type)

            # Mirror the trigger: if the record is already on ICDR_0_4, set
            # scaled_grade = original_grade directly (no mapping row needed).
            if scale_id == target_scale_id:
                try:
                    new_scaled_grade = int(original_grade)
                except (ValueError, TypeError):
                    new_scaled_grade = None
            else:
                # Check for mapping to standard scale
                mapping = await find_grading_scale_mapping_to_standard(
                    source_scale_id=scale_id,
                    source_value=original_grade,
                )
                new_scaled_grade = mapping.target_value if (mapping and mapping.target_value is not None) else None

            # Only update if the scaled_grade has changed
            if new_scaled_grade != current_scaled_grade:
                # Recreate the DiseaseGrading model with updated values
                grading = DiseaseGrading(
                    grading_id=row["grading_id"],
                    image_id=row["image_id"],
                    disease_type=row["disease_type"],
                    scale_id=row["scale_id"],
                    original_grade=row["original_grade"],
                    scaled_grade=new_scaled_grade,  # Updated value
                    grade_label=row["grade_label"],
                    raw_data_id=row["raw_data_id"],
                    expert_annotation_id=row["expert_annotation_id"],
                    consensus_id=row["consensus_id"],
                    annotation_method=row["annotation_method"],
                    confidence_score=row["confidence_score"],
                    provenance_chain_id=row["provenance_chain_id"],
                    created_at=row["created_at"],
                    updated_at=datetime.now(),  # Update timestamp
                )

                # Upsert the updated grading
                await upsert_disease_grading(grading)
                updated_count += 1

                # Track successful update
                internal_tracker.record_success(item_type="grade_updated")
                internal_tracker.update()

                logger.debug(
                    f"Updated grading {grading_id}: scaled_grade changed from "
                    f"{current_scaled_grade} to {new_scaled_grade}"
                )
            else:
                # No update needed, record as skipped
                internal_tracker.record_skip(
                    item_type="grade_unchanged", reason="no_mapping_change"
                )
                internal_tracker.update()

        except Exception as e:
            # Record error and continue
            internal_tracker.record_error(
                error_type="update_failed",
                error_message=str(e),
                item_id=str(grading_id),
                details={
                    "scale_id": str(scale_id),
                    "original_grade": original_grade,
                },
            )
            internal_tracker.update(success=False)
            logger.error(f"Failed to update grading {grading_id}: {e}")

    # Get statistics before finishing
    stats = internal_tracker.get_statistics()

    # Finish tracking and log summary (only if we created the tracker internally)
    if tracker is None:
        internal_tracker.finish()

    # Log detailed summary
    logger.debug(
        f"Grade update complete: {updated_count} updated, "
        f"{stats.skipped_items} unchanged, {stats.failed_items} failed"
    )

    # Print statistics summary
    print("\n" + "=" * 60)
    print("GRADE UPDATE STATISTICS")
    print("=" * 60)
    print(f"Total grades checked:     {total_checked}")
    print(f"Successfully updated:     {stats.successful_items}")
    print(f"Unchanged (no mapping):   {stats.skipped_items}")
    print(f"Failed:                   {stats.failed_items}")
    print(
        f"Success rate:             {(stats.successful_items / total_checked * 100):.1f}%"
        if total_checked > 0
        else "N/A"
    )

    if stats.error_counts:
        print("\nError breakdown:")
        for error_type, count in stats.error_counts.items():
            print(f"  - {error_type}: {count}")

    print("=" * 60 + "\n")

    return updated_count, total_checked


async def create_mapping(
    source_scale_id: uuid.UUID,
    target_scale_id: uuid.UUID,
    source_value: str,
    target_value: Optional[int] = None,
    mapping_confidence: str = "exact",
) -> uuid.UUID:
    """
    Create a mapping between two grading scales.

    Args:
        source_scale_id: UUID of the source grading scale
        target_scale_id: UUID of the target grading scale
        source_value: Source grade value (string)
        target_value: Target grade value (integer, optional)
        mapping_confidence: Confidence level ('exact', 'approximate', 'manual_review_required')

    Returns:
        mapping_id UUID of the created mapping

    Example:
        ```python
        mapping_id = await create_mapping(
            source_scale_id=icdr_scale_id,
            target_scale_id=etdrs_scale_id,
            source_value="Mild",
            target_value=1,
            mapping_confidence="exact"
        )
        ```
    """
    # Generate deterministic UUID for the mapping
    mapping_id = generate_grading_scale_mapping_uuid(
        source_scale_id=source_scale_id,
        target_scale_id=target_scale_id,
        source_value=source_value,
    )

    # Create grading scale mapping model
    mapping = GradingScaleMapping(
        mapping_id=mapping_id,
        source_scale_id=source_scale_id,
        target_scale_id=target_scale_id,
        source_value=source_value,
        target_value=target_value,
        mapping_confidence=mapping_confidence,
    )

    # Store in database (idempotent upsert)
    await upsert_grading_scale_mapping(mapping)

    logger.info(
        f"Created mapping from {source_value} (scale {source_scale_id}) "
        f"to {target_value} (scale {target_scale_id}) with confidence {mapping_confidence}"
    )

    return mapping_id
