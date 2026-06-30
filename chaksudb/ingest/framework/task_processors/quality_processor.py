"""
Quality processor for image quality annotations.

This module provides high-level processing functions for quality annotations
that handle validation, normalization, and model preparation.

Key features:
- Validates quality_type constraints
- Handles DeepDRiD quality metrics (Overall, Clarity, Field definition, Artifact)
- Supports quality_score (numeric) and quality_label (categorical)
- Handles multiple quality scale types (0-1, 0-5, etc.)
- Returns typed QualityAnnotation models ready for upsert
- Automatic provenance tracking via context variables
"""

import logging
import uuid
from typing import Optional, Union

from chaksudb.db.models import QualityAnnotation
from chaksudb.ingest.framework.gen_uuid import generate_quality_uuid
from chaksudb.ingest.framework.provenance_context import get_current_provenance

logger = logging.getLogger(__name__)


# ============================================
# Helper Functions
# ============================================


def normalize_quality_score(
    score: Union[int, float, str],
    scale_min: Optional[float] = None,
    scale_max: Optional[float] = None,
    target_min: float = 0.0,
    target_max: float = 1.0,
) -> float:
    """
    Normalize a quality score to a target range.

    Args:
        score: Raw quality score
        scale_min: Minimum value of source scale (if known)
        scale_max: Maximum value of source scale (if known)
        target_min: Minimum value of target scale (default: 0.0)
        target_max: Maximum value of target scale (default: 1.0)

    Returns:
        Normalized quality score in [target_min, target_max]

    Raises:
        ValueError: If score cannot be converted or is out of range

    Example:
        >>> normalize_quality_score(3, scale_min=0, scale_max=5)
        0.6
        >>> normalize_quality_score(0.75, scale_min=0, scale_max=1)
        0.75
    """
    # Convert to float
    try:
        if isinstance(score, str):
            score_float = float(score.strip())
        else:
            score_float = float(score)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Cannot convert quality score '{score}' to float: {e}") from e

    # If no scale provided, assume score is already normalized
    if scale_min is None or scale_max is None:
        # Check if score is in [0, 1]
        if not 0.0 <= score_float <= 1.0:
            logger.warning(
                f"Quality score {score_float} outside [0, 1] range without scale info"
            )
        return score_float

    # Validate score is within source scale
    if not scale_min <= score_float <= scale_max:
        raise ValueError(
            f"Quality score {score_float} outside valid range [{scale_min}, {scale_max}]"
        )

    # Normalize to target range
    if scale_max == scale_min:
        # Degenerate case
        normalized = target_min
    else:
        normalized = (
            (score_float - scale_min) / (scale_max - scale_min)
        ) * (target_max - target_min) + target_min

    return float(normalized)


def parse_quality_label(
    label: Union[str, int, bool],
    quality_type: str,
) -> str:
    """
    Parse and normalize quality labels to standard format.

    Args:
        label: Quality label (can be string, int, or bool)
        quality_type: Type of quality annotation

    Returns:
        Normalized quality label as string

    Example:
        >>> parse_quality_label("Good", "overall")
        "good"
        >>> parse_quality_label(1, "gradability")
        "gradable"
        >>> parse_quality_label(True, "gradability")
        "gradable"
    """
    if isinstance(label, bool):
        # Common for gradability
        if quality_type == "gradability":
            return "gradable" if label else "ungradable"
        else:
            return "acceptable" if label else "unacceptable"

    elif isinstance(label, int):
        # Common for gradability (0/1)
        if quality_type == "gradability":
            return "gradable" if label == 1 else "ungradable"
        else:
            return str(label)

    elif isinstance(label, str):
        # Normalize string labels
        label_lower = label.strip().lower()

        # Common synonyms
        label_map = {
            "good": "good",
            "acceptable": "acceptable",
            "adequate": "acceptable",
            "excellent": "excellent",
            "poor": "poor",
            "unacceptable": "unacceptable",
            "inadequate": "unacceptable",
            "bad": "poor",
            "fair": "fair",
            "moderate": "moderate",
            "gradable": "gradable",
            "ungradable": "ungradable",
            "reject": "ungradable",
            "accept": "gradable",
        }

        return label_map.get(label_lower, label_lower)

    else:
        raise ValueError(f"Invalid quality label type: {type(label)}")


# ============================================
# Main Processing Function
# ============================================


async def process_quality_annotation(
    quality_type: str,
    image_id: uuid.UUID,
    quality_score: Optional[Union[int, float, str]] = None,
    quality_label: Optional[Union[str, int, bool]] = None,
    scale_description: Optional[str] = None,
    scale_min: Optional[float] = None,
    scale_max: Optional[float] = None,
    normalize_score: bool = True,
    raw_data_id: Optional[uuid.UUID] = None,
    expert_annotation_id: Optional[uuid.UUID] = None,
    provenance_chain_id: Optional[uuid.UUID] = None,
) -> QualityAnnotation:
    """
    Process a quality annotation and prepare it for upsert.

    This function:
    1. Validates quality_type against allowed values
    2. Normalizes quality_score to [0, 1] range (if normalize_score=True)
    3. Parses and normalizes quality_label
    4. Generates deterministic UUID for the quality annotation
    5. Returns a typed QualityAnnotation model ready for upsert

    Args:
        quality_type: Quality type ('overall', 'gradability', 'clarity',
                      'field_definition', 'artifact', 'contrast', 'blur', 'illumination')
        image_id: UUID of the image being assessed
        quality_score: Optional numeric quality score
        quality_label: Optional categorical quality label
        scale_description: Optional description of the quality scale used
        scale_min: Optional minimum value of source scale (for normalization)
        scale_max: Optional maximum value of source scale (for normalization)
        normalize_score: Whether to normalize score to [0, 1] (default: True)
        raw_data_id: Optional raw annotation file UUID
        expert_annotation_id: Optional expert annotation UUID
        provenance_chain_id: Optional provenance chain UUID

    Returns:
        QualityAnnotation model ready for upsert

    Raises:
        ValueError: If quality_type is invalid or both quality_score and quality_label are None

    Examples:
        DeepDRiD overall quality (0-2 scale):
        ```python
        quality = await process_quality_annotation(
            quality_type="overall",
            image_id=image_id,
            quality_score=2,
            quality_label="Excellent",
            scale_description="DeepDRiD Overall Quality (0=Poor, 1=Good, 2=Excellent)",
            scale_min=0,
            scale_max=2,
            raw_data_id=raw_file_id,
        )
        ```

        Gradability (binary):
        ```python
        quality = await process_quality_annotation(
            quality_type="gradability",
            image_id=image_id,
            quality_label="gradable",
            raw_data_id=raw_file_id,
        )
        ```

        Clarity score (0-5 scale):
        ```python
        quality = await process_quality_annotation(
            quality_type="clarity",
            image_id=image_id,
            quality_score=4,
            scale_description="Clarity (0-5)",
            scale_min=0,
            scale_max=5,
            raw_data_id=raw_file_id,
        )
        ```
    """
    # Get provenance from context if not explicitly provided
    if raw_data_id is None or provenance_chain_id is None:
        context_raw_id, context_chain_id = get_current_provenance()
        raw_data_id = raw_data_id or context_raw_id
        provenance_chain_id = provenance_chain_id or context_chain_id
    
    # Register the quality_type in the reference table if it's new (idempotent). This
    # replaces the old hard-coded allow-list: quality dimensions are now extensible, and
    # the FK on quality_annotations.quality_type enforces validity against this registry.
    if not quality_type or not quality_type.strip():
        raise ValueError("quality_type must be a non-empty string")
    from chaksudb.db.queries.annotation_types import get_or_create_quality_type
    await get_or_create_quality_type(quality_type)

    # Ensure at least one of quality_score or quality_label is provided
    if quality_score is None and quality_label is None:
        raise ValueError(
            "At least one of quality_score or quality_label must be provided"
        )

    # Process quality_score
    normalized_score: Optional[float] = None
    if quality_score is not None:
        if normalize_score:
            normalized_score = normalize_quality_score(
                score=quality_score,
                scale_min=scale_min,
                scale_max=scale_max,
            )
        else:
            # Just convert to float without normalization
            try:
                if isinstance(quality_score, str):
                    normalized_score = float(quality_score.strip())
                else:
                    normalized_score = float(quality_score)
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"Cannot convert quality_score '{quality_score}' to float: {e}"
                ) from e

    # Process quality_label
    parsed_label: Optional[str] = None
    if quality_label is not None:
        parsed_label = parse_quality_label(quality_label, quality_type)

    # Generate deterministic UUID
    quality_id = generate_quality_uuid(
        image_id=image_id,
        quality_type=quality_type,
        expert_annotation_id=expert_annotation_id,
        raw_data_id=raw_data_id,
        quality_score=normalized_score,
    )

    # Create QualityAnnotation model
    quality = QualityAnnotation(
        quality_id=quality_id,
        image_id=image_id,
        quality_type=quality_type,
        quality_score=normalized_score,
        quality_label=parsed_label,
        scale_description=scale_description,
        raw_data_id=raw_data_id,
        expert_annotation_id=expert_annotation_id,
        provenance_chain_id=provenance_chain_id,
    )

    logger.debug(
        f"Processed quality annotation: {quality_type} "
        f"(score={normalized_score}, label={parsed_label}) for image {image_id}"
    )

    return quality


# ============================================
# Convenience Functions
# ============================================


async def prepare_quality_for_upsert(
    quality_type: str,
    image_id: uuid.UUID,
    **kwargs,
) -> QualityAnnotation:
    """
    Alias for process_quality_annotation() for consistency with other processors.

    See process_quality_annotation() for full documentation.
    """
    return await process_quality_annotation(
        quality_type=quality_type,
        image_id=image_id,
        **kwargs,
    )


async def process_deepdrid_quality(
    image_id: uuid.UUID,
    overall_quality: Optional[int] = None,
    clarity: Optional[int] = None,
    field_definition: Optional[int] = None,
    artifact: Optional[int] = None,
    raw_data_id: Optional[uuid.UUID] = None,
) -> list[QualityAnnotation]:
    """
    Process DeepDRiD quality metrics (convenience function).

    DeepDRiD provides 4 quality metrics:
    - Overall Quality: 0 (poor), 1 (good), 2 (excellent)
    - Clarity: 0 (severe), 1 (mild), 2 (no blur)
    - Field Definition: 0 (inadequate), 1 (adequate)
    - Artifact: 0 (severe), 1 (mild), 2 (no artifact)

    Args:
        image_id: UUID of the image
        overall_quality: Overall quality score (0-2)
        clarity: Clarity score (0-2)
        field_definition: Field definition score (0-1)
        artifact: Artifact score (0-2)
        raw_data_id: Optional raw annotation file UUID

    Returns:
        List of QualityAnnotation models

    Example:
        ```python
        qualities = await process_deepdrid_quality(
            image_id=image_id,
            overall_quality=2,
            clarity=2,
            field_definition=1,
            artifact=1,
            raw_data_id=raw_file_id,
        )
        for quality in qualities:
            await upsert_quality_annotation(quality)
        ```
    """
    annotations = []

    if overall_quality is not None:
        label_map = {0: "poor", 1: "good", 2: "excellent"}
        annotations.append(
            await process_quality_annotation(
                quality_type="overall",
                image_id=image_id,
                quality_score=overall_quality,
                quality_label=label_map.get(overall_quality),
                scale_description="DeepDRiD Overall Quality (0=Poor, 1=Good, 2=Excellent)",
                scale_min=0,
                scale_max=2,
                raw_data_id=raw_data_id,
            )
        )

    if clarity is not None:
        label_map = {0: "severe_blur", 1: "mild_blur", 2: "no_blur"}
        annotations.append(
            await process_quality_annotation(
                quality_type="clarity",
                image_id=image_id,
                quality_score=clarity,
                quality_label=label_map.get(clarity),
                scale_description="DeepDRiD Clarity (0=Severe blur, 1=Mild blur, 2=No blur)",
                scale_min=0,
                scale_max=2,
                raw_data_id=raw_data_id,
            )
        )

    if field_definition is not None:
        label_map = {0: "inadequate", 1: "adequate"}
        annotations.append(
            await process_quality_annotation(
                quality_type="field_definition",
                image_id=image_id,
                quality_score=field_definition,
                quality_label=label_map.get(field_definition),
                scale_description="DeepDRiD Field Definition (0=Inadequate, 1=Adequate)",
                scale_min=0,
                scale_max=1,
                raw_data_id=raw_data_id,
            )
        )

    if artifact is not None:
        label_map = {0: "severe_artifact", 1: "mild_artifact", 2: "no_artifact"}
        annotations.append(
            await process_quality_annotation(
                quality_type="artifact",
                image_id=image_id,
                quality_score=artifact,
                quality_label=label_map.get(artifact),
                scale_description="DeepDRiD Artifact (0=Severe, 1=Mild, 2=No artifact)",
                scale_min=0,
                scale_max=2,
                raw_data_id=raw_data_id,
            )
        )

    return annotations
