"""
UnifiedExportRow: Type definition for export row structure.

Defines the structure of rows exported from the database, including core image fields
and optional annotation blocks (grading, segmentation, localization, etc.).
"""

from typing import Any, Optional
from typing_extensions import TypedDict
from uuid import UUID


class SegmentationMask(TypedDict, total=False):
    """Structure for a single segmentation mask annotation.
    
    Only includes training-relevant fields. IDs (segmentation_id, annotation_type_id)
    are excluded as they are not needed for model training.
    """

    annotation_type: Optional[str]  # from annotation_type table (e.g. 'vessel', 'optic_disc')
    lesion_subtype: Optional[str]
    mask_file_path: Optional[str]
    confidence_score: Optional[float]
    unified_format: Optional[str]  # 'binary_mask', 'soft_map', or 'layer_boundaries'


class LocalizationAnnotation(TypedDict, total=False):
    """Structure for a single localization annotation.
    
    Only includes training-relevant fields. ID (localization_id) is excluded
    as it is not needed for model training.
    """

    localization_type: str  # 'bounding_box', 'keypoint', 'center_point'
    target_structure: str
    coordinates: dict[str, Any]  # JSONB structure
    lesion_subtype: Optional[str]


class ClassificationAnnotation(TypedDict, total=False):
    """Structure for a single classification annotation.
    
    Only includes training-relevant fields. ID (classification_id) is excluded
    as it is not needed for model training.
    """

    task_type: str  # 'binary', 'multi_class', 'multi_label'
    class_name: str
    class_index: Optional[int]
    class_label: Optional[str]
    sub_key: Optional[str]
    class_value: Any  # JSONB value (kept for backward compat / soft labels)
    confidence_score: Optional[float]


class UnifiedExportRow(TypedDict, total=False):
    """
    Unified export row structure.

    This TypedDict defines the structure of rows exported from the database.
    All fields are optional (total=False) because different export specs will
    include different fields based on the requested annotation tasks.

    Core Fields (always present when ImageModule is used):
        - image_id: UUID of the image (only ID kept - all other IDs excluded for training)
        - dataset_name: Name of the dataset (dataset_id excluded - not needed for training)
        - file_path: Local file path (for storage_provider='local')
        - storage_provider: Storage provider ('local', 's3', 'gcs', 'azure', 'http')
        - object_key: Object key for cloud storage (for non-local providers)
        - modality: Image modality ('fundus', 'oct', 'fa', 'uwf')
        - eye_laterality: Eye laterality ('left', 'right', 'unknown')
        - group_id: UUID shared by all frames of an OCT volume; NULL for non-OCT/standalone images
        - frame_index: Position in the OCT volume; 0 = key frame (B-scan), 1+ = volume frames; NULL if not part of a volume

    Split Fields (present when SplitModule is used):
        - split_name: Name of the split (e.g., 'train', 'test', 'val')
        - task_type: Task type for the split

    Grading Fields (present when GradingModule is used):
        - dr_grade: Scaled DR grade (integer) if DR grading is included
        - dme_grade: Scaled DME grade (integer) if DME grading is included
        - glaucoma_grade: Scaled Glaucoma grade (integer) if Glaucoma grading is included
        - amd_grade: Scaled AMD grade (integer) if AMD grading is included
        - dr_original_grade: Original DR grade text (if include_original_grade=True)
        - dme_original_grade: Original DME grade text (if include_original_grade=True)
        - glaucoma_original_grade: Original Glaucoma grade text (if include_original_grade=True)
        - amd_original_grade: Original AMD grade text (if include_original_grade=True)
        - dr_scale_name: Name of the grading scale used for DR
        - dme_scale_name: Name of the grading scale used for DME
        - glaucoma_scale_name: Name of the grading scale used for Glaucoma
        - amd_scale_name: Name of the grading scale used for AMD
        - dr_annotation_source: Source of DR annotation ('expert' or 'consensus')
        - dme_annotation_source: Source of DME annotation ('expert' or 'consensus')
        - glaucoma_annotation_source: Source of Glaucoma annotation ('expert' or 'consensus')
        - amd_annotation_source: Source of AMD annotation ('expert' or 'consensus')

    Segmentation Fields (present when SegmentationModule is used):
        - segmentation_masks: List of SegmentationMask dicts (JSONB array)
          Each mask contains: annotation_type, lesion_subtype, mask_file_path, confidence_score
          (IDs like segmentation_id, annotation_type_id are excluded)

    Localization Fields (present when LocalizationModule is used):
        - localization_annotations: List of LocalizationAnnotation dicts (JSONB array)
          Each annotation contains: localization_type, target_structure, coordinates, lesion_subtype
          (ID like localization_id is excluded)

    Classification Fields (present when ClassificationModule is used):
        For each class_name in spec.classification_class_names, creates flat training-ready columns:
        - Binary classification:
            - {class_name}_label: int (0 or 1) or float (0.0 or 1.0)
            - {class_name}_class_label: str (e.g., 'positive', 'negative')
        - Multi-class classification:
            - {class_name}_label: int (class index) or float
            - {class_name}_class_label: str (class label text)
        - Multi-label classification (with spec.multi_label_keys):
            - {class_name}_{key}: int (0 or 1) or float for each sublabel key
        - Multi-label classification (without keys):
            - {class_name}_labels: str (JSON string of label dict)

    Quality Fields (present when QualityModule is used):
        Quality is exported in pivoted per-type columns (e.g. overall_quality_score,
        overall_quality_label, gradability_quality_score, gradability_quality_label)
        and as quality_annotations JSONB. The scalar fields below are a subset;
        see QualityModule for the full column set.
        - quality_type: Type of quality annotation (legacy / alternate representation)
        - quality_score: Quality score (float)
        - quality_label: Quality label (text)
        - quality_annotations: JSONB array of quality annotations (per QualityModule)

    Keywords Fields (present when KeywordsModule is used):
        - keywords: List of keyword strings (array)

    Clinical Fields (present when ClinicalModule is used):
        - clinical_description_text: Clinical description text
        - clinical_description_type: Type of description ('clinical_caption', 'diagnosis_text', 'notes')
        - clinical_word_count: Word count of description
    """

    # Core image fields
    image_id: UUID  # Only ID kept - all other IDs excluded for training
    dataset_name: str  # dataset_id excluded - not needed for training
    file_path: Optional[str]
    storage_provider: str
    object_key: Optional[str]
    modality: Optional[str]
    eye_laterality: Optional[str]
    health_status: Optional[str]  # 'normal' | 'abnormal' | None (cross-dataset derived)
    # OCT volume fields — NULL for non-OCT or standalone images
    group_id: Optional[UUID]   # shared across all frames of the same OCT volume
    frame_index: Optional[int] # 0 = key frame (B-scan), 1+ = volume frames

    # Split fields
    split_name: Optional[str]
    task_type: Optional[str]

    # Grading fields - DR
    dr_grade: Optional[int]
    dr_original_grade: Optional[str]
    dr_scale_name: Optional[str]
    dr_annotation_source: Optional[str]  # 'expert' or 'consensus'

    # Grading fields - DME
    dme_grade: Optional[int]
    dme_original_grade: Optional[str]
    dme_scale_name: Optional[str]
    dme_annotation_source: Optional[str]

    # Grading fields - Glaucoma
    glaucoma_grade: Optional[int]
    glaucoma_original_grade: Optional[str]
    glaucoma_scale_name: Optional[str]
    glaucoma_annotation_source: Optional[str]

    # Grading fields - AMD
    amd_grade: Optional[int]
    amd_original_grade: Optional[str]
    amd_scale_name: Optional[str]
    amd_annotation_source: Optional[str]

    # Segmentation fields
    segmentation_masks: Optional[list[SegmentationMask]]

    # Localization fields
    localization_annotations: Optional[list[LocalizationAnnotation]]

    # Classification fields (dynamic columns based on spec.classification_class_names)
    # For each class_name, the following columns are added:
    # - {class_name}_label: int or float (training label)
    # - {class_name}_class_label: str (human-readable label)
    # - {class_name}_{key}: int or float (for multi-label with keys)
    # - {class_name}_labels: str (for multi-label without keys, JSON string)
    # These are not explicitly typed here as they vary per export spec

    # Quality fields (per-type pivoted columns from QualityModule)
    # For each quality_type in [overall, gradability, clarity, field_definition,
    # artifact, contrast, blur, illumination]:
    #   {quality_type}_quality_score: Optional[float]
    #   {quality_type}_quality_label: Optional[str]
    # These are dynamic and not explicitly typed here.
    overall_quality_score: Optional[float]
    overall_quality_label: Optional[str]
    gradability_quality_score: Optional[float]
    gradability_quality_label: Optional[str]
    quality_annotations: Optional[Any]  # JSONB array from QualityModule

    # Fundus ROI fields (present when include_fundus_roi=True)
    fundus_roi_cx: Optional[float]
    fundus_roi_cy: Optional[float]
    fundus_roi_radius: Optional[float]
    fundus_roi_method: Optional[str]  # 'ransac' or 'fallback'

    # Keywords fields
    keywords: Optional[list[str]]

    # Clinical description fields
    clinical_description_text: Optional[str]
    clinical_description_type: Optional[str]
    clinical_word_count: Optional[int]

    # Patient fields (present when include_patient_data=True)
    patient_id: Optional[UUID]
    original_patient_id: Optional[str]
    age: Optional[int]
    sex: Optional[str]
    ethnicity: Optional[str]
    comorbidities: Optional[Any]  # JSONB

    # Caption fields (present when caption_mode is set)
    caption_clinical_text: Optional[str]
    caption_keywords: Optional[list[str]]
