"""
Grading processor for disease grading annotations.

This module provides high-level processing functions for disease grading
that handle scale registration, validation, and model preparation.

Key features:
- Accepts various input formats (int, str, float)
- Validates against disease type constraints
- Auto-registers unknown scales with logging
- Returns typed DiseaseGrading models ready for upsert
- Delegates conversion logic to database triggers
- Helper functions for scale management
- Automatic provenance tracking via context variables
"""

import logging
import uuid
from typing import Any, Optional, Union

from chaksudb.config.config import constants
from chaksudb.db.models import DiseaseGrading, GradingScale
from chaksudb.db.queries.grading import (
    find_grading_scale_by_id,
    upsert_grading_scale,
)
from chaksudb.ingest.framework.gen_uuid import (
    generate_disease_grading_uuid,
    generate_grading_scale_uuid,
)
from chaksudb.ingest.framework.provenance_context import get_current_provenance

logger = logging.getLogger(__name__)


# ============================================
# Helper Functions
# ============================================


async def get_or_create_scale(
    scale_name: str,
    disease_type: str,
    scale_description: Optional[str] = None,
    min_value: Optional[int] = None,
    max_value: Optional[int] = None,
    value_labels: Optional[dict[str, Any]] = None,
) -> uuid.UUID:
    """
    Get or create a grading scale (idempotent).

    If the scale exists, returns its UUID. If it doesn't exist, creates it
    and logs a warning about the unmapped scale.

    Args:
        scale_name: Name of the grading scale (e.g., 'ICDR_0_4', 'AAO')
        disease_type: Disease type ('DR', 'DME', 'Glaucoma', 'AMD')
        scale_description: Optional description of the scale
        min_value: Optional minimum value in the scale
        max_value: Optional maximum value in the scale
        value_labels: Optional dictionary mapping values to labels

    Returns:
        scale_id UUID

    Example:
        ```python
        scale_id = await get_or_create_scale(
            scale_name="EYEPACS_0_4",
            disease_type="DR",
            scale_description="EYEPACS 5-level DR grading",
            min_value=0,
            max_value=4,
            value_labels={"0": "No DR", "1": "Mild", "2": "Moderate",
                         "3": "Severe", "4": "Proliferative"}
        )
        ```
    """
    # Generate deterministic UUID for the scale
    scale_id = generate_grading_scale_uuid(scale_name, disease_type)

    # Check if scale already exists
    existing_scale = await find_grading_scale_by_id(scale_id)

    if existing_scale:
        logger.debug(f"Scale {scale_name} for {disease_type} already exists")
        return scale_id

    # Create new scale
    logger.warning(
        f"Auto-registering unknown scale '{scale_name}' for {disease_type}. "
        f"No conversion mappings exist yet. Add mappings to enable normalization."
    )

    scale = GradingScale(
        scale_id=scale_id,
        scale_name=scale_name,
        disease_type=disease_type,
        scale_description=scale_description,
        min_value=min_value,
        max_value=max_value,
        value_labels=value_labels,
    )

    await upsert_grading_scale(scale)

    logger.info(
        f"Created new scale: {scale_name} ({disease_type}) with scale_id {scale_id}"
    )

    return scale_id


async def check_scale_mapping_exists(
    scale_id: uuid.UUID,
    target_scale_id: uuid.UUID,
) -> bool:
    """
    Check if a mapping exists between two grading scales.

    Note: This is a local helper that delegates to the query layer.
    """
    from chaksudb.db.queries.grading import check_scale_mapping_exists as db_check

    try:
        return await db_check(scale_id, target_scale_id)
    except Exception as e:
        logger.debug(f"Error checking scale mapping: {e}")
        return False


# ============================================
# Main Processing Function
# ============================================


async def process_disease_grade(
    grade_value: Union[int, float, str],
    disease_type: str,
    scale_name: str,
    image_id: uuid.UUID,
    scale_description: Optional[str] = None,
    min_value: Optional[int] = None,
    max_value: Optional[int] = None,
    value_labels: Optional[dict[str, Any]] = None,
    grade_label: Optional[str] = None,
    raw_data_id: Optional[uuid.UUID] = None,
    expert_annotation_id: Optional[uuid.UUID] = None,
    consensus_id: Optional[uuid.UUID] = None,
    annotation_method: str = "manual",
    confidence_score: Optional[float] = None,
    provenance_chain_id: Optional[uuid.UUID] = None,
) -> DiseaseGrading:
    """
    Process a disease grade and prepare it for upsert.

    This function:
    1. Validates disease_type against allowed values
    2. Normalizes grade_value to string format
    3. Gets or creates the grading scale (auto-registers if unknown)
    4. Generates deterministic UUID for the grading
    5. Returns a typed DiseaseGrading model ready for upsert

    The database trigger will handle automatic conversion to target scale.

    Args:
        grade_value: Grade value (can be int, float, or string)
        disease_type: Disease type ('DR', 'DME', 'Glaucoma', 'AMD')
        scale_name: Name of the grading scale (e.g., 'ICDR_0_4', 'EYEPACS')
        image_id: UUID of the image being graded
        scale_description: Optional scale description
        min_value: Optional minimum value
        max_value: Optional maximum value
        value_labels: Optional mapping of values to labels
        grade_label: Optional human-readable grade label
        raw_data_id: Optional raw annotation file UUID
        expert_annotation_id: Optional expert annotation UUID
        consensus_id: Optional consensus annotation UUID
        annotation_method: Method ('manual', 'adjudicated', 'consensus', 'pseudo')
        confidence_score: Optional confidence score (0.0 to 1.0)
        provenance_chain_id: Optional provenance chain UUID

    Returns:
        DiseaseGrading model ready for upsert

    Raises:
        ValueError: If disease_type is invalid or grade_value cannot be normalized

    Example:
        ```python
        # Process a DR grade from EYEPACS dataset
        grading = await process_disease_grade(
            grade_value=2,
            disease_type="DR",
            scale_name="EYEPACS_0_4",
            image_id=image_id,
            scale_description="EYEPACS 5-level DR grading",
            min_value=0,
            max_value=4,
            value_labels={
                "0": "No DR",
                "1": "Mild NPDR",
                "2": "Moderate NPDR",
                "3": "Severe NPDR",
                "4": "PDR"
            },
            raw_data_id=raw_file_id,
        )

        # Upsert to database
        await upsert_disease_grading(grading)
        ```
    """
    # Get provenance from context if not explicitly provided
    if raw_data_id is None or provenance_chain_id is None:
        context_raw_id, context_chain_id = get_current_provenance()
        raw_data_id = raw_data_id or context_raw_id
        provenance_chain_id = provenance_chain_id or context_chain_id
    
    # Validate disease_type
    if disease_type not in constants.DISEASE_TYPES:
        raise ValueError(
            f"Invalid disease_type: {disease_type}. "
            f"Must be one of {constants.DISEASE_TYPES}"
        )

    # Normalize grade_value to string
    try:
        if isinstance(grade_value, float):
            # Convert float to int first if it's a whole number
            if grade_value == int(grade_value):
                original_grade = str(int(grade_value))
            else:
                original_grade = str(grade_value)
        elif isinstance(grade_value, int):
            original_grade = str(grade_value)
        elif isinstance(grade_value, str):
            # Try to clean up the string
            original_grade = grade_value.strip()
        else:
            raise ValueError(
                f"Invalid grade_value type: {type(grade_value)}. "
                "Must be int, float, or string"
            )
    except Exception as e:
        raise ValueError(f"Could not normalize grade_value {grade_value}: {e}") from e

    # Validate annotation_method
    valid_methods = {"manual", "adjudicated", "consensus", "pseudo"}
    if annotation_method not in valid_methods:
        raise ValueError(
            f"Invalid annotation_method: {annotation_method}. "
            f"Must be one of {valid_methods}"
        )

    # Get or create scale (idempotent)
    scale_id = await get_or_create_scale(
        scale_name=scale_name,
        disease_type=disease_type,
        scale_description=scale_description,
        min_value=min_value,
        max_value=max_value,
        value_labels=value_labels,
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

    # Create DiseaseGrading model
    # Note: scaled_grade is left as None - the database trigger will populate it
    grading = DiseaseGrading(
        grading_id=grading_id,
        image_id=image_id,
        disease_type=disease_type,
        scale_id=scale_id,
        original_grade=original_grade,
        scaled_grade=None,  # Trigger will populate this
        grade_label=grade_label,
        raw_data_id=raw_data_id,
        expert_annotation_id=expert_annotation_id,
        consensus_id=consensus_id,
        annotation_method=annotation_method,
        confidence_score=confidence_score,
        provenance_chain_id=provenance_chain_id,
    )

    logger.debug(
        f"Processed disease grade: {disease_type} grade {original_grade} "
        f"(scale: {scale_name}) for image {image_id}"
    )

    return grading


# ============================================
# Convenience Function (alias)
# ============================================


async def prepare_grading_for_upsert(
    grade_value: Union[int, float, str],
    disease_type: str,
    scale_name: str,
    image_id: uuid.UUID,
    **kwargs,
) -> DiseaseGrading:
    """
    Alias for process_disease_grade() for consistency with other processors.

    See process_disease_grade() for full documentation.
    """
    return await process_disease_grade(
        grade_value=grade_value,
        disease_type=disease_type,
        scale_name=scale_name,
        image_id=image_id,
        **kwargs,
    )
