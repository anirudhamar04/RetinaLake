"""
Database query functions module.

This module provides idempotent upsert functions and bulk insert helpers
for all database tables. All functions are re-exported here for backward
compatibility with existing imports.

Example usage:
    from chaksudb.db.queries import upsert_dataset, bulk_upsert_images
    from chaksudb.db.queries import validate_dataset_exists
"""

# Dataset operations
from chaksudb.db.queries.datasets import (
    bulk_upsert_datasets,
    upsert_dataset,
    upsert_dataset_split,
    upsert_image_split,
)

# Model operations
from chaksudb.db.queries.models import upsert_model

# Expert operations
from chaksudb.db.queries.experts import (
    bulk_upsert_expert_annotations,
    upsert_expert,
    upsert_expert_annotation,
)

# Patient operations
from chaksudb.db.queries.patients import bulk_upsert_patients, upsert_patient

# Image operations
from chaksudb.db.queries.images import (
    bulk_upsert_image_groups,
    bulk_upsert_images,
    bulk_upsert_patient_images,
    upsert_image,
    upsert_image_group,
    upsert_patient_image,
)

# Raw annotation file operations
from chaksudb.db.queries.raw_annotations import (
    bulk_upsert_raw_annotation_files,
    get_raw_annotation_file,
    upsert_raw_annotation_file,
)

# Grading operations
from chaksudb.db.queries.grading import (
    bulk_upsert_disease_gradings,
    check_scale_mapping_exists,
    find_grading_scale_by_id,
    find_grading_scale_mapping_to_standard,
    get_all_disease_gradings_with_original_grade,
    upsert_disease_grading,
    upsert_grading_scale,
    upsert_grading_scale_mapping,
)

# Consensus operations
from chaksudb.db.queries.consensus import upsert_consensus_annotation

# Provenance operations
from chaksudb.db.queries.provenance import (
    fetch_grade_conversions_for_audit,
    find_orphan_transformations,
    get_chain,
    get_lineage_for_image,
    get_transformations_for_chain,
    upsert_provenance_chain,
    upsert_provenance_transformation,
    upsert_transformation_operation,
)

# Annotation type operations
from chaksudb.db.queries.annotation_types import (
    bulk_upsert_classification_annotations,
    bulk_upsert_localization_annotations,
    bulk_upsert_quality_annotations,
    upsert_annotation_type,
    upsert_classification_annotation,
    upsert_clinical_description,
    upsert_keyword_annotation,
    upsert_keyword_vocabulary,
    upsert_localization_annotation,
    upsert_quality_annotation,
    upsert_segmentation_annotation,
)

# Validation operations
from chaksudb.db.queries.validation import (
    validate_dataset_exists,
    validate_image_exists,
    validate_patient_exists,
)

__all__ = [
    # Dataset operations
    "upsert_dataset",
    "bulk_upsert_datasets",
    "upsert_dataset_split",
    "upsert_image_split",
    # Model operations
    "upsert_model",
    # Expert operations
    "upsert_expert",
    "upsert_expert_annotation",
    "bulk_upsert_expert_annotations",
    # Patient operations
    "upsert_patient",
    "bulk_upsert_patients",
    # Image operations
    "upsert_image_group",
    "bulk_upsert_image_groups",
    "upsert_image",
    "bulk_upsert_images",
    "upsert_patient_image",
    "bulk_upsert_patient_images",
    # Raw annotation file operations
    "upsert_raw_annotation_file",
    "bulk_upsert_raw_annotation_files",
    # Grading operations
    "upsert_grading_scale",
    "upsert_grading_scale_mapping",
    "upsert_disease_grading",
    "bulk_upsert_disease_gradings",
    "find_grading_scale_mapping_to_standard",
    "check_scale_mapping_exists",
    "find_grading_scale_by_id",
    "get_all_disease_gradings_with_original_grade",
    # Consensus operations
    "upsert_consensus_annotation",
    # Provenance operations
    "upsert_provenance_chain",
    "upsert_transformation_operation",
    "upsert_provenance_transformation",
    "get_chain",
    "get_transformations_for_chain",
    "get_lineage_for_image",
    "find_orphan_transformations",
    "fetch_grade_conversions_for_audit",
    # Annotation type operations
    "upsert_annotation_type",
    "upsert_segmentation_annotation",
    "upsert_localization_annotation",
    "bulk_upsert_localization_annotations",
    "upsert_classification_annotation",
    "bulk_upsert_classification_annotations",
    "upsert_quality_annotation",
    "bulk_upsert_quality_annotations",
    "upsert_clinical_description",
    "upsert_keyword_vocabulary",
    "upsert_keyword_annotation",
    # Validation operations
    "validate_dataset_exists",
    "validate_image_exists",
    "validate_patient_exists",
]
