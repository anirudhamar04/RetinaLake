"""
Classification processor for binary, multi-class, and multi-label classification annotations.

Every row is self-describing. The contract:
  - ``task_name``     identifies the dataset's task (e.g. 'glaucoma_screening',
                      'fives_disease'); export pivots on it.
  - ``task_type``     one of binary | multi_class | multi_label.
  - ``is_multilabel`` explicit boolean shape flag (== task_type=='multi_label').
  - ``class_index``   always populated (never NULL).
  - ``class_label``   the *real* label name ('RG', 'moderate', ...), never 'positive'.
  - ``concept``       canonical clinical concept for cross-task filtering (see concepts.py).

Storage convention:
  binary:      class_index=0|1, class_label=real labels, sub_key=NULL,
               concept=concept of class_name
  multi_class: class_index=N,   class_label=winning class, sub_key=NULL,
               concept=concept of the winning class
  multi_label: one row per sub-key; class_index=0|1, class_label=sub_key,
               sub_key=sub_key, concept=concept of the sub-key

Each ingest script is expected to declare task_name, task_type, and (for multi_class) a
class_labels map explicitly when calling process_classification.
"""

import json
import logging
import uuid
from typing import Any, Optional, Union

from chaksudb.db.models import ClassificationAnnotation
from chaksudb.ingest.framework.concepts import normalize_class_name, to_concept
from chaksudb.ingest.framework.gen_uuid import generate_classification_uuid
from chaksudb.ingest.framework.provenance_context import get_current_provenance

logger = logging.getLogger(__name__)


# ============================================
# Helper Functions
# ============================================


def compute_class_value_hash(class_value: Any) -> str:
    """Compute a deterministic hash of a class_value.

    Args:
        class_value: Value to hash (dict, bool, int, etc.)

    Returns:
        SHA256 hash of the JSON-serialized value
    """
    import hashlib

    if isinstance(class_value, dict):
        json_str = json.dumps(class_value, sort_keys=True)
    else:
        json_str = json.dumps(class_value)
    return hashlib.sha256(json_str.encode()).hexdigest()


def _to_bool(value: Union[bool, int, str]) -> bool:
    """Convert a value to boolean for binary classification."""
    if isinstance(value, bool):
        return value
    elif isinstance(value, int):
        if value not in (0, 1):
            raise ValueError(
                f"Binary classification int value must be 0 or 1, got {value}"
            )
        return bool(value)
    elif isinstance(value, str):
        value_lower = value.lower().strip()
        if value_lower in ("yes", "true", "1", "positive", "present"):
            return True
        elif value_lower in ("no", "false", "0", "negative", "absent"):
            return False
        else:
            raise ValueError(
                f"Cannot interpret '{value}' as binary classification value"
            )
    else:
        raise ValueError(f"Invalid type for binary classification: {type(value)}")


# ============================================
# Main Processing Function
# ============================================


async def process_classification(
    class_value: Union[bool, int, str, dict[str, Any]],
    task_type: str,
    class_name: str,
    image_id: uuid.UUID,
    task_name: Optional[str] = None,
    class_labels: Optional[dict[int, str]] = None,
    concept: Optional[str] = None,
    raw_data_id: Optional[uuid.UUID] = None,
    expert_annotation_id: Optional[uuid.UUID] = None,
    consensus_id: Optional[uuid.UUID] = None,
    annotation_method: str = "manual",
    confidence_score: Optional[float] = None,
    provenance_chain_id: Optional[uuid.UUID] = None,
) -> list[ClassificationAnnotation]:
    """
    Process a classification annotation and prepare it for upsert.

    Returns a list of ClassificationAnnotation models:
    - binary/multi_class: single-element list
    - multi_label: one element per sub-key

    Args:
        class_value: Classification value (format depends on task_type)
            - binary: bool, int (0/1), or str ('yes'/'no')
            - multi_class: int (class index) or str (class label)
            - multi_label: dict mapping sub-keys to bool/int/float
        task_type: Task type ('binary', 'multi_class', 'multi_label')
        class_name: Class name (e.g., 'glaucoma', 'disease_category')
        image_id: UUID of the image being classified
        task_name: Identity of the dataset's task (e.g. 'glaucoma_screening',
            'fives_disease'). Defaults to class_name when omitted.
        class_labels: Mapping from class indices to label names.
            Required for multi_class (so class_index is always populated); optional but
            recommended for binary (e.g. {1: 'RG', 0: 'NRG'}) to keep the real labels.
        concept: Canonical clinical concept for cross-task filtering. When omitted it is
            derived from the label via the shared concepts vocabulary.
        raw_data_id: Optional raw annotation file UUID
        expert_annotation_id: Optional expert annotation UUID
        consensus_id: Optional consensus annotation UUID
        annotation_method: Method ('manual', 'consensus', 'pseudo')
        confidence_score: Optional confidence score (0.0 to 1.0)
        provenance_chain_id: Optional provenance chain UUID

    Returns:
        List of ClassificationAnnotation models ready for upsert

    Raises:
        ValueError: If task_type is invalid or class_value format doesn't match task_type
    """
    # Get provenance from context if not explicitly provided
    if raw_data_id is None or provenance_chain_id is None:
        context_raw_id, context_chain_id = get_current_provenance()
        raw_data_id = raw_data_id or context_raw_id
        provenance_chain_id = provenance_chain_id or context_chain_id

    # Normalize class_name
    original_class_name = class_name
    class_name = normalize_class_name(class_name)
    if class_name != original_class_name:
        logger.debug(
            f"Normalized class_name: '{original_class_name}' -> '{class_name}'"
        )

    # task_name defaults to class_name so single-task datasets need not repeat themselves
    task_name = task_name or class_name

    # Validate task_type
    valid_task_types = {"binary", "multi_class", "multi_label"}
    if task_type not in valid_task_types:
        raise ValueError(
            f"Invalid task_type: {task_type}. Must be one of {valid_task_types}"
        )

    # Validate annotation_method
    valid_methods = {"manual", "consensus", "pseudo"}
    if annotation_method not in valid_methods:
        raise ValueError(
            f"Invalid annotation_method: {annotation_method}. "
            f"Must be one of {valid_methods}"
        )

    # Validate confidence_score
    if confidence_score is not None:
        if not 0.0 <= confidence_score <= 1.0:
            raise ValueError(
                f"confidence_score must be in [0, 1], got {confidence_score}"
            )

    common = dict(
        task_name=task_name,
        concept=concept,
        image_id=image_id,
        raw_data_id=raw_data_id,
        expert_annotation_id=expert_annotation_id,
        consensus_id=consensus_id,
        annotation_method=annotation_method,
        confidence_score=confidence_score,
        provenance_chain_id=provenance_chain_id,
    )

    # Build annotations based on task_type
    if task_type == "binary":
        return _process_binary(class_value, class_name, class_labels, **common)

    elif task_type == "multi_class":
        return _process_multi_class(class_value, class_name, class_labels, **common)

    elif task_type == "multi_label":
        if not isinstance(class_value, dict):
            raise ValueError(
                f"multi_label task_type requires dict class_value, got {type(class_value)}"
            )
        return _process_multi_label(class_value, class_name, **common)

    else:
        raise ValueError(f"Unsupported task_type: {task_type}")


def _process_binary(
    value: Union[bool, int, str, dict],
    class_name: str,
    class_labels: Optional[dict[int, str]],
    *,
    task_name: str,
    concept: Optional[str],
    image_id: uuid.UUID,
    raw_data_id: Optional[uuid.UUID],
    expert_annotation_id: Optional[uuid.UUID],
    consensus_id: Optional[uuid.UUID],
    annotation_method: str,
    confidence_score: Optional[float],
    provenance_chain_id: Optional[uuid.UUID],
) -> list[ClassificationAnnotation]:
    """Process binary classification into a single-row annotation.

    Honors ``class_labels`` (e.g. {1: 'RG', 0: 'NRG'}) so the real label names survive;
    falls back to positive/negative only when the caller supplies none.
    """
    if isinstance(value, dict):
        # Already-formatted dict: extract the boolean value
        bool_value = next(iter(value.values()))
        if not isinstance(bool_value, bool):
            bool_value = _to_bool(bool_value)
    else:
        bool_value = _to_bool(value)

    class_index = 1 if bool_value else 0
    labels = class_labels or {1: "positive", 0: "negative"}
    class_label = labels[class_index]

    classification_id = generate_classification_uuid(
        image_id=image_id,
        task_type="binary",
        class_name=class_name,
        expert_annotation_id=expert_annotation_id,
        consensus_id=consensus_id,
        raw_data_id=raw_data_id,
        class_value_hash=compute_class_value_hash(bool_value),
    )

    annotation = ClassificationAnnotation(
        classification_id=classification_id,
        image_id=image_id,
        task_type="binary",
        task_name=task_name,
        class_name=class_name,
        concept=concept or to_concept(class_name),
        is_multilabel=False,
        class_index=class_index,
        class_label=class_label,
        sub_key=None,
        class_value={class_name: bool_value},
        raw_data_id=raw_data_id,
        expert_annotation_id=expert_annotation_id,
        consensus_id=consensus_id,
        annotation_method=annotation_method,
        confidence_score=confidence_score,
        provenance_chain_id=provenance_chain_id,
    )

    logger.debug(f"Processed binary classification: {class_name}={class_index} for image {image_id}")
    return [annotation]


def _process_multi_class(
    value: Union[int, str, dict],
    class_name: str,
    class_labels: Optional[dict[int, str]],
    *,
    task_name: str,
    concept: Optional[str],
    image_id: uuid.UUID,
    raw_data_id: Optional[uuid.UUID],
    expert_annotation_id: Optional[uuid.UUID],
    consensus_id: Optional[uuid.UUID],
    annotation_method: str,
    confidence_score: Optional[float],
    provenance_chain_id: Optional[uuid.UUID],
) -> list[ClassificationAnnotation]:
    """Process multi-class classification into a single-row annotation.

    Requires a class_labels map so class_index is ALWAYS populated. Stores the winning
    class as both class_name and class_label; the task identity lives in task_name.
    """
    if class_labels is None:
        raise ValueError(
            f"multi_class task '{task_name}' requires a class_labels map "
            f"(index -> label) so class_index is never NULL"
        )

    if isinstance(value, int):
        if value not in class_labels:
            raise ValueError(
                f"multi_class index {value} not in class_labels {class_labels} "
                f"for task '{task_name}'"
            )
        class_index = value
        class_label = class_labels[value]
    elif isinstance(value, str):
        label = value.strip()
        inverse = {lbl: idx for idx, lbl in class_labels.items()}
        if label not in inverse:
            raise ValueError(
                f"multi_class label '{label}' not in class_labels {class_labels} "
                f"for task '{task_name}'"
            )
        class_index = inverse[label]
        class_label = label
    else:
        raise ValueError(f"Invalid type for multi-class classification: {type(value)}")

    jsonb_value = {"class_index": class_index, "class_label": class_label}

    classification_id = generate_classification_uuid(
        image_id=image_id,
        task_type="multi_class",
        class_name=task_name,
        expert_annotation_id=expert_annotation_id,
        consensus_id=consensus_id,
        raw_data_id=raw_data_id,
        class_value_hash=compute_class_value_hash(jsonb_value),
    )

    annotation = ClassificationAnnotation(
        classification_id=classification_id,
        image_id=image_id,
        task_type="multi_class",
        task_name=task_name,
        # the winning class is the meaningful label here
        class_name=class_label,
        concept=concept or to_concept(class_label),
        is_multilabel=False,
        class_index=class_index,
        class_label=class_label,
        sub_key=None,
        class_value={task_name: jsonb_value},
        raw_data_id=raw_data_id,
        expert_annotation_id=expert_annotation_id,
        consensus_id=consensus_id,
        annotation_method=annotation_method,
        confidence_score=confidence_score,
        provenance_chain_id=provenance_chain_id,
    )

    logger.debug(
        f"Processed multi_class classification: {task_name}={class_label} "
        f"(index={class_index}) for image {image_id}"
    )
    return [annotation]


def _process_multi_label(
    values: dict[str, Union[bool, int, float]],
    class_name: str,
    *,
    task_name: str,
    concept: Optional[str],
    image_id: uuid.UUID,
    raw_data_id: Optional[uuid.UUID],
    expert_annotation_id: Optional[uuid.UUID],
    consensus_id: Optional[uuid.UUID],
    annotation_method: str,
    confidence_score: Optional[float],
    provenance_chain_id: Optional[uuid.UUID],
) -> list[ClassificationAnnotation]:
    """Process multi-label classification by exploding into one row per sub-key.

    Each sub-key gets its own row with concept derived from the sub-key, so individual
    diseases within a multi-label vector participate in cross-task concept filtering.
    """
    annotations = []

    for sub_key, value in values.items():
        # Convert to class_index
        if isinstance(value, bool):
            class_index = 1 if value else 0
        elif isinstance(value, int):
            if value not in (0, 1):
                raise ValueError(
                    f"Multi-label int value for '{sub_key}' must be 0 or 1, got {value}"
                )
            class_index = value
        elif isinstance(value, float):
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"Multi-label float value for '{sub_key}' must be in [0, 1], got {value}"
                )
            # For soft labels, store raw float as class_index rounded and keep float in class_value
            class_index = round(value)
        else:
            raise ValueError(
                f"Invalid type for multi-label value '{sub_key}': {type(value)}"
            )

        classification_id = generate_classification_uuid(
            image_id=image_id,
            task_type="multi_label",
            class_name=class_name,
            sub_key=sub_key,
            expert_annotation_id=expert_annotation_id,
            consensus_id=consensus_id,
            raw_data_id=raw_data_id,
        )

        # Store the original value in class_value for lossless round-trip
        if isinstance(value, float):
            cv = {sub_key: value}
        elif isinstance(value, (bool, int)):
            cv = {sub_key: bool(value) if isinstance(value, int) else value}
        else:
            cv = {sub_key: value}

        annotation = ClassificationAnnotation(
            classification_id=classification_id,
            image_id=image_id,
            task_type="multi_label",
            task_name=task_name,
            class_name=class_name,
            concept=concept or to_concept(sub_key),
            is_multilabel=True,
            class_index=class_index,
            class_label=sub_key,
            sub_key=sub_key,
            class_value=cv,
            raw_data_id=raw_data_id,
            expert_annotation_id=expert_annotation_id,
            consensus_id=consensus_id,
            annotation_method=annotation_method,
            confidence_score=confidence_score,
            provenance_chain_id=provenance_chain_id,
        )
        annotations.append(annotation)

    logger.debug(
        f"Processed multi_label classification: {class_name} "
        f"({len(annotations)} sub-keys) for image {image_id}"
    )
    return annotations


# ============================================
# Backward-compatible formatting helpers
# ============================================
# These are kept for any code that calls them directly,
# but process_classification is the primary entry point.


def format_binary_classification(
    value: Union[bool, int, str],
    class_name: str,
) -> dict[str, Any]:
    """Format binary classification value as JSONB (backward compat)."""
    bool_value = _to_bool(value)
    return {class_name: bool_value}


def format_multi_class_classification(
    value: Union[int, str],
    class_name: str,
    class_labels: Optional[dict[int, str]] = None,
) -> dict[str, Any]:
    """Format multi-class classification value as JSONB (backward compat)."""
    result = {}
    if isinstance(value, int):
        result["class_index"] = value
        if class_labels and value in class_labels:
            result["class_label"] = class_labels[value]
    elif isinstance(value, str):
        result["class_label"] = value.strip()
    else:
        raise ValueError(f"Invalid type for multi-class classification: {type(value)}")
    return {class_name: result}


def format_multi_label_classification(
    values: dict[str, Union[bool, int, float]],
) -> dict[str, Any]:
    """Format multi-label classification as JSONB (backward compat)."""
    result = {}
    for label, value in values.items():
        if isinstance(value, bool):
            result[label] = value
        elif isinstance(value, int):
            if value not in (0, 1):
                raise ValueError(
                    f"Multi-label int value for '{label}' must be 0 or 1, got {value}"
                )
            result[label] = bool(value)
        elif isinstance(value, float):
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"Multi-label float value for '{label}' must be in [0, 1], got {value}"
                )
            result[label] = value
        else:
            raise ValueError(
                f"Invalid type for multi-label value '{label}': {type(value)}"
            )
    return result


# ============================================
# Convenience Function (alias)
# ============================================


async def prepare_classification_for_upsert(
    class_value: Union[bool, int, str, dict[str, Any]],
    task_type: str,
    class_name: str,
    image_id: uuid.UUID,
    **kwargs,
) -> list[ClassificationAnnotation]:
    """
    Alias for process_classification() for consistency with other processors.

    See process_classification() for full documentation.
    """
    return await process_classification(
        class_value=class_value,
        task_type=task_type,
        class_name=class_name,
        image_id=image_id,
        **kwargs,
    )
