"""
UUID v5 generation utilities for all database tables.

This module provides deterministic UUID v5 generation functions for all tables
in the database schema. UUIDs are generated from identifying fields to enable
hash-based lookups and natural idempotency.

All functions use UUID v5 (SHA-1 based) which ensures:
- Deterministic: same inputs always produce same UUID
- Idempotent: re-ingestion won't create duplicates
- Hash-based lookups: can compute UUID from known fields
"""

import uuid
from typing import Optional

from chaksudb.config.config import constants


def generate_uuid_v5(namespace: uuid.UUID, name: str) -> uuid.UUID:
    """
    Core UUID v5 generation function.

    Args:
        namespace: UUID namespace (from constants)
        name: String identifier to hash

    Returns:
        Deterministic UUID v5
    """
    return uuid.uuid5(namespace, name)


# ============================================
# Core Tables
# ============================================

def generate_dataset_uuid(dataset_name: str) -> uuid.UUID:
    """
    Generate UUID for datasets table.

    Args:
        dataset_name: Unique dataset name

    Returns:
        dataset_id UUID
    """
    return generate_uuid_v5(constants.NAMESPACE_DATASET, dataset_name)


def generate_model_uuid(model_name: str) -> uuid.UUID:
    """
    Generate UUID for models table.

    Args:
        model_name: Unique model name

    Returns:
        model_id UUID
    """
    return generate_uuid_v5(constants.NAMESPACE_MODEL, model_name)


def generate_expert_uuid(
    dataset_id: Optional[uuid.UUID],
    model_id: Optional[uuid.UUID],
    expert_name: Optional[str] = None
) -> uuid.UUID:
    """
    Generate UUID for experts table.

    Experts can be either dataset-based (real expert) or model-based (pseudo expert).
    At least one of dataset_id or model_id must be provided.

    Args:
        dataset_id: Dataset UUID if real expert
        model_id: Model UUID if pseudo expert
        expert_name: Optional expert name for additional uniqueness

    Returns:
        expert_id UUID
    """
    if dataset_id is not None:
        name = f"dataset:{dataset_id}"
    elif model_id is not None:
        name = f"model:{model_id}"
    else:
        raise ValueError("Either dataset_id or model_id must be provided")

    if expert_name:
        name = f"{name}:{expert_name}"

    return generate_uuid_v5(constants.NAMESPACE_EXPERT, name)


# ============================================
# Patient & Image Tables
# ============================================

def generate_patient_uuid(dataset_id: uuid.UUID, original_patient_id: str) -> uuid.UUID:
    """
    Generate UUID for patients table.

    Args:
        dataset_id: Dataset UUID
        original_patient_id: Original patient identifier from source dataset

    Returns:
        patient_id UUID
    """
    name = f"{dataset_id}:{original_patient_id}"
    return generate_uuid_v5(constants.NAMESPACE_PATIENT, name)


def generate_image_group_uuid(
    dataset_id: uuid.UUID,
    group_type: str,
    group_identifier: str
) -> uuid.UUID:
    """
    Generate UUID for image_groups table.

    Args:
        dataset_id: Dataset UUID
        group_type: Type of group ('oct_volume', 'video', 'sequence')
        group_identifier: Unique identifier for the group within the dataset

    Returns:
        group_id UUID
    """
    name = f"{dataset_id}:{group_type}:{group_identifier}"
    return generate_uuid_v5(constants.NAMESPACE_IMAGE_GROUP, name)


def generate_image_uuid(dataset_id: uuid.UUID, original_image_id: str) -> uuid.UUID:
    """
    Generate UUID for images table.

    Args:
        dataset_id: Dataset UUID
        original_image_id: Original image identifier from source dataset

    Returns:
        image_id UUID
    """
    name = f"{dataset_id}:{original_image_id}"
    return generate_uuid_v5(constants.NAMESPACE_IMAGE, name)


def generate_patient_image_uuid(patient_id: uuid.UUID, image_id: uuid.UUID) -> uuid.UUID:
    """
    Generate UUID for patient_images join table.

    Args:
        patient_id: Patient UUID
        image_id: Image UUID

    Returns:
        relationship_id UUID
    """
    name = f"{patient_id}:{image_id}"
    return generate_uuid_v5(constants.NAMESPACE_PATIENT_IMAGE, name)


# ============================================
# Raw Annotation Files
# ============================================

def generate_raw_file_uuid(dataset_id: uuid.UUID, file_hash: str) -> uuid.UUID:
    """
    Generate UUID for raw_annotation_files table.

    Args:
        dataset_id: Dataset UUID
        file_hash: SHA256 hash of the file content

    Returns:
        raw_file_id UUID
    """
    name = f"{dataset_id}:{file_hash}"
    return generate_uuid_v5(constants.NAMESPACE_RAW_FILE, name)


# ============================================
# Expert Annotations
# ============================================

def generate_expert_annotation_uuid(
    expert_id: uuid.UUID,
    annotation_task: str,
    raw_data_id: Optional[uuid.UUID],
    annotation_value_hash: Optional[str] = None
) -> uuid.UUID:
    """
    Generate UUID for expert_annotations table.

    Args:
        expert_id: Expert UUID
        annotation_task: Task type ('grading', 'segmentation', etc.)
        raw_data_id: Optional raw file UUID
        annotation_value_hash: Optional hash of annotation_value JSONB for uniqueness

    Returns:
        expert_annotation_id UUID
    """
    parts = [f"expert:{expert_id}", f"task:{annotation_task}"]
    if raw_data_id:
        parts.append(f"raw:{raw_data_id}")
    if annotation_value_hash:
        parts.append(f"hash:{annotation_value_hash}")
    name = ":".join(parts)
    return generate_uuid_v5(constants.NAMESPACE_EXPERT_ANNOTATION, name)


# ============================================
# Annotation Types & Scales
# ============================================

def generate_annotation_type_uuid(annotation_type: str) -> uuid.UUID:
    """
    Generate UUID for annotation_type table.

    Args:
        annotation_type: Type name (e.g., 'drusen', 'hemorrhage')

    Returns:
        annotation_type_id UUID
    """
    return generate_uuid_v5(constants.NAMESPACE_ANNOTATION_TYPE, annotation_type)


def generate_grading_scale_uuid(scale_name: str, disease_type: str) -> uuid.UUID:
    """
    Generate UUID for grading_scales table.

    Args:
        scale_name: Scale name (e.g., 'ICDR', 'ETDRS')
        disease_type: Disease type ('DR', 'DME', 'Glaucoma', 'AMD')

    Returns:
        scale_id UUID
    """
    name = f"{scale_name}:{disease_type}"
    return generate_uuid_v5(constants.NAMESPACE_GRADING_SCALE, name)


def generate_grading_scale_mapping_uuid(
    source_scale_id: uuid.UUID,
    target_scale_id: uuid.UUID,
    source_value: str
) -> uuid.UUID:
    """
    Generate UUID for grading_scale_mappings table.

    Args:
        source_scale_id: Source scale UUID
        target_scale_id: Target scale UUID
        source_value: Source scale value

    Returns:
        mapping_id UUID
    """
    name = f"{source_scale_id}:{target_scale_id}:{source_value}"
    return generate_uuid_v5(constants.NAMESPACE_GRADING_SCALE_MAPPING, name)


# ============================================
# Consensus Annotations
# ============================================

def generate_consensus_uuid(
    image_id: uuid.UUID,
    annotation_task: str,
    consensus_method: str,
    expert_annotation_ids: list[uuid.UUID]
) -> uuid.UUID:
    """
    Generate UUID for consensus_annotations table.

    Args:
        image_id: Image UUID
        annotation_task: Task type
        consensus_method: Method used ('majority_vote', 'mean', etc.)
        expert_annotation_ids: List of expert annotation UUIDs used

    Returns:
        consensus_id UUID
    """
    # Sort expert IDs for deterministic UUID
    sorted_ids = sorted(str(eid) for eid in expert_annotation_ids)
    expert_ids_str = ",".join(sorted_ids)
    name = f"{image_id}:{annotation_task}:{consensus_method}:{expert_ids_str}"
    return generate_uuid_v5(constants.NAMESPACE_CONSENSUS, name)


# ============================================
# Provenance & Transformations
# ============================================

def generate_provenance_chain_uuid(
    unified_annotation_type: str,
    source_type: str,
    root_source_raw_data_id: Optional[uuid.UUID],
    source_annotation_ids: list[uuid.UUID]
) -> uuid.UUID:
    """
    Generate UUID for provenance_chain table.

    Args:
        unified_annotation_type: Unified annotation type
        source_type: Source type ('original', 'transformed', etc.)
        root_source_raw_data_id: Optional root raw file UUID
        source_annotation_ids: List of source annotation UUIDs

    Returns:
        chain_id UUID
    """
    parts = [f"type:{unified_annotation_type}", f"source:{source_type}"]
    if root_source_raw_data_id:
        parts.append(f"root:{root_source_raw_data_id}")
    # Sort annotation IDs for deterministic UUID
    sorted_ids = sorted(str(aid) for aid in source_annotation_ids)
    if sorted_ids:
        parts.append(f"annotations:{','.join(sorted_ids)}")
    name = ":".join(parts)
    return generate_uuid_v5(constants.NAMESPACE_PROVENANCE_CHAIN, name)


def generate_transformation_uuid(
    operation_type: str,
    input_data_hash: Optional[str] = None,
    operation_parameters_hash: Optional[str] = None
) -> uuid.UUID:
    """
    Generate UUID for transformation_operations table.

    Args:
        operation_type: Type of transformation operation
        input_data_hash: Optional hash of input_data JSONB
        operation_parameters_hash: Optional hash of operation_parameters JSONB

    Returns:
        transformation_id UUID
    """
    parts = [f"op:{operation_type}"]
    if input_data_hash:
        parts.append(f"input:{input_data_hash}")
    if operation_parameters_hash:
        parts.append(f"params:{operation_parameters_hash}")
    name = ":".join(parts)
    return generate_uuid_v5(constants.NAMESPACE_TRANSFORMATION, name)


def generate_provenance_transformation_uuid(
    chain_id: uuid.UUID,
    transformation_id: uuid.UUID
) -> uuid.UUID:
    """
    Generate UUID for provenance_transformations join table.

    Args:
        chain_id: Provenance chain UUID
        transformation_id: Transformation operation UUID

    Returns:
        id UUID
    """
    name = f"{chain_id}:{transformation_id}"
    return generate_uuid_v5(constants.NAMESPACE_PROVENANCE_TRANSFORMATION, name)


# ============================================
# Annotation Tables
# ============================================

def generate_segmentation_uuid(
    image_id: uuid.UUID,
    annotation_type_id: uuid.UUID,
    expert_annotation_id: Optional[uuid.UUID] = None,
    consensus_id: Optional[uuid.UUID] = None,
    raw_data_id: Optional[uuid.UUID] = None,
    lesion_subtype: Optional[str] = None,
) -> uuid.UUID:
    """
    Generate UUID for segmentation_annotations table.

    Args:
        image_id: Image UUID
        annotation_type_id: Annotation type UUID
        expert_annotation_id: Optional expert annotation UUID
        consensus_id: Optional consensus UUID
        raw_data_id: Optional raw file UUID
        lesion_subtype: Optional lesion subtype (MA, HE, EX, SE, etc.)

    Returns:
        segmentation_id UUID
    """
    parts = [f"image:{image_id}", f"type:{annotation_type_id}"]
    if lesion_subtype:
        parts.append(f"subtype:{lesion_subtype}")
    if expert_annotation_id:
        parts.append(f"expert:{expert_annotation_id}")
    if consensus_id:
        parts.append(f"consensus:{consensus_id}")
    if raw_data_id:
        parts.append(f"raw:{raw_data_id}")
    name = ":".join(parts)
    return generate_uuid_v5(constants.NAMESPACE_SEGMENTATION, name)


def generate_disease_grading_uuid(
    image_id: uuid.UUID,
    disease_type: str,
    scale_id: uuid.UUID,
    expert_annotation_id: Optional[uuid.UUID] = None,
    consensus_id: Optional[uuid.UUID] = None,
    raw_data_id: Optional[uuid.UUID] = None,
    original_grade: Optional[str] = None
) -> uuid.UUID:
    """
    Generate UUID for disease_grading table.

    Args:
        image_id: Image UUID
        disease_type: Disease type ('DR', 'DME', 'Glaucoma', 'AMD')
        scale_id: Grading scale UUID
        expert_annotation_id: Optional expert annotation UUID
        consensus_id: Optional consensus UUID
        raw_data_id: Optional raw file UUID
        original_grade: Optional original grade value

    Returns:
        grading_id UUID
    """
    parts = [
        f"image:{image_id}",
        f"disease:{disease_type}",
        f"scale:{scale_id}"
    ]
    if expert_annotation_id:
        parts.append(f"expert:{expert_annotation_id}")
    if consensus_id:
        parts.append(f"consensus:{consensus_id}")
    if raw_data_id:
        parts.append(f"raw:{raw_data_id}")
    if original_grade:
        parts.append(f"grade:{original_grade}")
    name = ":".join(parts)
    return generate_uuid_v5(constants.NAMESPACE_DISEASE_GRADING, name)


def generate_localization_uuid(
    image_id: uuid.UUID,
    localization_type: str,
    target_structure: str,
    expert_annotation_id: Optional[uuid.UUID] = None,
    consensus_id: Optional[uuid.UUID] = None,
    raw_data_id: Optional[uuid.UUID] = None,
    coordinates_hash: Optional[str] = None
) -> uuid.UUID:
    """
    Generate UUID for localization_annotations table.

    Args:
        image_id: Image UUID
        localization_type: Type ('bounding_box', 'keypoint', 'center_point')
        target_structure: Target structure name
        expert_annotation_id: Optional expert annotation UUID
        consensus_id: Optional consensus UUID
        raw_data_id: Optional raw file UUID
        coordinates_hash: Optional hash of coordinates JSONB

    Returns:
        localization_id UUID
    """
    parts = [
        f"image:{image_id}",
        f"type:{localization_type}",
        f"target:{target_structure}"
    ]
    if expert_annotation_id:
        parts.append(f"expert:{expert_annotation_id}")
    if consensus_id:
        parts.append(f"consensus:{consensus_id}")
    if raw_data_id:
        parts.append(f"raw:{raw_data_id}")
    if coordinates_hash:
        parts.append(f"coords:{coordinates_hash}")
    name = ":".join(parts)
    return generate_uuid_v5(constants.NAMESPACE_LOCALIZATION, name)


def generate_classification_uuid(
    image_id: uuid.UUID,
    task_type: str,
    class_name: str,
    sub_key: Optional[str] = None,
    expert_annotation_id: Optional[uuid.UUID] = None,
    consensus_id: Optional[uuid.UUID] = None,
    raw_data_id: Optional[uuid.UUID] = None,
    class_value_hash: Optional[str] = None
) -> uuid.UUID:
    """
    Generate UUID for classification_annotations table.

    Args:
        image_id: Image UUID
        task_type: Task type ('binary', 'multi_class', 'multi_label')
        class_name: Class name
        sub_key: Multi-label sub-key (e.g., 'diabetes' within 'disease_indicators')
        expert_annotation_id: Optional expert annotation UUID
        consensus_id: Optional consensus UUID
        raw_data_id: Optional raw file UUID
        class_value_hash: Optional hash of class_value JSONB

    Returns:
        classification_id UUID
    """
    parts = [
        f"image:{image_id}",
        f"task:{task_type}",
        f"class:{class_name}"
    ]
    if sub_key:
        parts.append(f"sub:{sub_key}")
    if expert_annotation_id:
        parts.append(f"expert:{expert_annotation_id}")
    if consensus_id:
        parts.append(f"consensus:{consensus_id}")
    if raw_data_id:
        parts.append(f"raw:{raw_data_id}")
    if class_value_hash:
        parts.append(f"value:{class_value_hash}")
    name = ":".join(parts)
    return generate_uuid_v5(constants.NAMESPACE_CLASSIFICATION, name)


def generate_quality_uuid(
    image_id: uuid.UUID,
    quality_type: str,
    expert_annotation_id: Optional[uuid.UUID] = None,
    raw_data_id: Optional[uuid.UUID] = None,
    quality_score: Optional[float] = None
) -> uuid.UUID:
    """
    Generate UUID for quality_annotations table.

    Args:
        image_id: Image UUID
        quality_type: Quality type ('overall', 'gradability', etc.)
        expert_annotation_id: Optional expert annotation UUID
        raw_data_id: Optional raw file UUID
        quality_score: Optional quality score for uniqueness

    Returns:
        quality_id UUID
    """
    parts = [f"image:{image_id}", f"type:{quality_type}"]
    if expert_annotation_id:
        parts.append(f"expert:{expert_annotation_id}")
    if raw_data_id:
        parts.append(f"raw:{raw_data_id}")
    if quality_score is not None:
        parts.append(f"score:{quality_score}")
    name = ":".join(parts)
    return generate_uuid_v5(constants.NAMESPACE_QUALITY, name)


def generate_description_uuid(
    image_id: uuid.UUID,
    description_type: str,
    expert_id: Optional[uuid.UUID] = None,
    raw_data_id: Optional[uuid.UUID] = None,
    description_hash: Optional[str] = None
) -> uuid.UUID:
    """
    Generate UUID for clinical_descriptions table.

    Args:
        image_id: Image UUID
        description_type: Type ('clinical_caption', 'diagnosis_text', 'notes')
        expert_id: Optional expert UUID
        raw_data_id: Optional raw file UUID
        description_hash: Optional hash of description_text

    Returns:
        description_id UUID
    """
    parts = [f"image:{image_id}", f"type:{description_type}"]
    if expert_id:
        parts.append(f"expert:{expert_id}")
    if raw_data_id:
        parts.append(f"raw:{raw_data_id}")
    if description_hash:
        parts.append(f"text:{description_hash}")
    name = ":".join(parts)
    return generate_uuid_v5(constants.NAMESPACE_DESCRIPTION, name)


# ============================================
# Keywords
# ============================================

def generate_keyword_uuid(
    dataset_id: uuid.UUID,
    keyword_term: str,
    keyword_source: str
) -> uuid.UUID:
    """
    Generate UUID for keyword_vocabulary table.

    Args:
        dataset_id: Dataset UUID
        keyword_term: Keyword term
        keyword_source: Source ('diagnostic_keywords', 'clinical_description', etc.)

    Returns:
        keyword_id UUID
    """
    name = f"{dataset_id}:{keyword_term}:{keyword_source}"
    return generate_uuid_v5(constants.NAMESPACE_KEYWORD, name)


def generate_keyword_annotation_uuid(
    image_id: uuid.UUID,
    keyword_id: uuid.UUID,
    expert_id: Optional[uuid.UUID] = None,
    raw_data_id: Optional[uuid.UUID] = None
) -> uuid.UUID:
    """
    Generate UUID for keyword_annotations table.

    Args:
        image_id: Image UUID
        keyword_id: Keyword vocabulary UUID
        expert_id: Optional expert UUID
        raw_data_id: Optional raw file UUID

    Returns:
        keyword_annotation_id UUID
    """
    parts = [f"image:{image_id}", f"keyword:{keyword_id}"]
    if expert_id:
        parts.append(f"expert:{expert_id}")
    if raw_data_id:
        parts.append(f"raw:{raw_data_id}")
    name = ":".join(parts)
    return generate_uuid_v5(constants.NAMESPACE_KEYWORD_ANNOTATION, name)


# ============================================
# Dataset Splits
# ============================================

def generate_dataset_split_uuid(
    dataset_id: uuid.UUID,
    split_name: str,
    split_type: str,
    task_type: Optional[str] = None
) -> uuid.UUID:
    """
    Generate UUID for dataset_splits table.

    Args:
        dataset_id: Dataset UUID
        split_name: Split name (e.g., 'train', 'val', 'test')
        split_type: Split type ('explicit', 'metadata_defined', etc.)
        task_type: Optional task type

    Returns:
        split_id UUID
    """
    parts = [f"{dataset_id}:{split_name}:{split_type}"]
    if task_type:
        parts.append(task_type)
    name = ":".join(parts)
    return generate_uuid_v5(constants.NAMESPACE_DATASET_SPLIT, name)


def generate_image_split_uuid(
    image_id: uuid.UUID,
    split_id: uuid.UUID,
    task_type: Optional[str] = None
) -> uuid.UUID:
    """
    Generate UUID for image_splits table.

    Args:
        image_id: Image UUID
        split_id: Dataset split UUID
        task_type: Optional task type

    Returns:
        assignment_id UUID
    """
    parts = [f"image:{image_id}", f"split:{split_id}"]
    if task_type:
        parts.append(task_type)
    name = ":".join(parts)
    return generate_uuid_v5(constants.NAMESPACE_IMAGE_SPLIT, name)
