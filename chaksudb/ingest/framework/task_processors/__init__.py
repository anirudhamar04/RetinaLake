"""
Task processors for ingestion framework.

Each task processor handles a specific annotation type and provides
functions to process and prepare data for database upsert.
"""

from chaksudb.ingest.framework.task_processors.grading_processor import (
    process_disease_grade,
    get_or_create_scale,
    check_scale_mapping_exists,
    prepare_grading_for_upsert,
)
from chaksudb.ingest.framework.task_processors.classification_processor import (
    process_classification,
    format_binary_classification,
    format_multi_class_classification,
    format_multi_label_classification,
    prepare_classification_for_upsert,
)
from chaksudb.ingest.framework.task_processors.quality_processor import (
    process_quality_annotation,
    normalize_quality_score,
    parse_quality_label,
    process_deepdrid_quality,
    prepare_quality_for_upsert,
)
from chaksudb.ingest.framework.task_processors.keyword_processor import (
    process_keyword_annotation,
    process_keywords_batch,
    get_or_create_keyword_vocabulary,
    parse_keyword_string,
    parse_deepeyenet_keywords,
    prepare_keywords_for_upsert,
)
from chaksudb.ingest.framework.task_processors.localization_processor import (
    process_localization_from_xml,
    process_localization_from_tsv,
    process_localization_from_json,
    process_localization_from_text_keypoint,
    prepare_localizations_for_upsert,
)
from chaksudb.ingest.framework.task_processors.segmentation_processor import (
    process_segmentation_from_binary_mask,
    process_segmentation_from_multiclass_mask,
    process_segmentation_from_contour,
    process_segmentation_from_xml,
    process_segmentation_from_soft_map,
    process_segmentation_from_layer_boundaries,
    get_or_create_annotation_type,
    prepare_segmentation_for_upsert,
)

__all__ = [
    # Grading processor
    "process_disease_grade",
    "get_or_create_scale",
    "check_scale_mapping_exists",
    "prepare_grading_for_upsert",
    # Classification processor
    "process_classification",
    "format_binary_classification",
    "format_multi_class_classification",
    "format_multi_label_classification",
    "prepare_classification_for_upsert",
    # Quality processor
    "process_quality_annotation",
    "normalize_quality_score",
    "parse_quality_label",
    "process_deepdrid_quality",
    "prepare_quality_for_upsert",
    # Keyword processor
    "process_keyword_annotation",
    "process_keywords_batch",
    "get_or_create_keyword_vocabulary",
    "parse_keyword_string",
    "parse_deepeyenet_keywords",
    "prepare_keywords_for_upsert",
    # Localization processor
    "process_localization_from_xml",
    "process_localization_from_tsv",
    "process_localization_from_json",
    "process_localization_from_text_keypoint",
    "prepare_localizations_for_upsert",
    # Segmentation processor
    "process_segmentation_from_binary_mask",
    "process_segmentation_from_multiclass_mask",
    "process_segmentation_from_contour",
    "process_segmentation_from_xml",
    "process_segmentation_from_soft_map",
    "process_segmentation_from_layer_boundaries",
    "get_or_create_annotation_type",
    "prepare_segmentation_for_upsert",
]
