"""
Pydantic models matching the PostgreSQL schema.

Provides type-safe data validation for all database tables with enum validation
and JSONB field handling.
"""

import uuid
from datetime import date, datetime
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from chaksudb.config.config import constants


# ============================================
# Core: Datasets
# ============================================

class Dataset(BaseModel):
    """Model for datasets table."""

    dataset_id: uuid.UUID
    dataset_name: str
    source_url: Optional[str] = None
    license: Optional[str] = None
    modality_types: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)

    @field_validator("modality_types")
    @classmethod
    def validate_modality_types(cls, v: list[str]) -> list[str]:
        """Validate modality types against allowed values."""
        if v:
            invalid = set(v) - constants.MODALITIES
            if invalid:
                raise ValueError(f"Invalid modality types: {invalid}")
        return v


# ============================================
# Core: Models
# ============================================

class Model(BaseModel):
    """Model for models table."""

    model_id: uuid.UUID
    model_name: str
    model_description: Optional[str] = None
    model_url: Optional[str] = None


# ============================================
# Core: Experts
# ============================================

class Expert(BaseModel):
    """Model for experts table."""

    expert_id: uuid.UUID
    expert_name: Optional[str] = None
    expertise_area: Optional[str] = None
    dataset_id: Optional[uuid.UUID] = None
    model_id: Optional[uuid.UUID] = None
    created_at: datetime = Field(default_factory=datetime.now)

    @model_validator(mode="after")
    def validate_source(self) -> "Expert":
        """Ensure at least one of dataset_id or model_id is provided."""
        if self.dataset_id is None and self.model_id is None:
            raise ValueError("Either dataset_id or model_id must be provided")
        return self


# ============================================
# Patients
# ============================================

class Patient(BaseModel):
    """Model for patients table."""

    patient_id: uuid.UUID
    dataset_id: uuid.UUID
    original_patient_id: str
    age: Optional[int] = None
    sex: Optional[str] = None
    ethnicity: Optional[str] = None
    nationality: Optional[str] = None
    comorbidities: Optional[dict[str, Any]] = None
    created_at: datetime = Field(default_factory=datetime.now)

    @field_validator("sex")
    @classmethod
    def validate_sex(cls, v: Optional[str]) -> Optional[str]:
        """Validate sex value."""
        if v is not None and v not in constants.SEX_VALUES:
            raise ValueError(f"Invalid sex value: {v}. Must be one of {constants.SEX_VALUES}")
        return v


# ============================================
# Image Groups
# ============================================

class ImageGroup(BaseModel):
    """Model for image_groups table."""

    group_id: uuid.UUID
    dataset_id: uuid.UUID
    group_type: str
    created_at: datetime = Field(default_factory=datetime.now)

    @field_validator("group_type")
    @classmethod
    def validate_group_type(cls, v: str) -> str:
        """Validate group type."""
        if v not in constants.GROUP_TYPES:
            raise ValueError(f"Invalid group_type: {v}. Must be one of {constants.GROUP_TYPES}")
        return v


# ============================================
# Images
# ============================================

class Image(BaseModel):
    """Model for images table."""

    image_id: uuid.UUID
    dataset_id: uuid.UUID
    original_image_id: Optional[str] = None
    storage_provider: str = "local"
    bucket: Optional[str] = None
    object_key: Optional[str] = None
    version_id: Optional[str] = None
    file_path: Optional[str] = None
    file_format: Optional[str] = None
    modality: Optional[str] = None
    file_hash: Optional[str] = None
    content_hash: Optional[str] = None
    phash: Optional[str] = None
    group_id: Optional[uuid.UUID] = None
    frame_index: Optional[int] = None
    resolution_width: Optional[int] = None
    resolution_height: Optional[int] = None
    field_of_view: Optional[int] = None
    eye_laterality: Optional[str] = None
    acquisition_date: Optional[date] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None

    @field_validator("file_format")
    @classmethod
    def validate_file_format(cls, v: Optional[str]) -> Optional[str]:
        """Validate file format."""
        if v is not None and v not in constants.FILE_FORMATS:
            raise ValueError(f"Invalid file_format: {v}. Must be one of {constants.FILE_FORMATS}")
        return v

    @field_validator("modality")
    @classmethod
    def validate_modality(cls, v: Optional[str]) -> Optional[str]:
        """Validate modality."""
        if v is not None and v not in constants.MODALITIES:
            raise ValueError(f"Invalid modality: {v}. Must be one of {constants.MODALITIES}")
        return v

    @field_validator("eye_laterality")
    @classmethod
    def validate_eye_laterality(cls, v: Optional[str]) -> Optional[str]:
        """Validate eye laterality."""
        if v is not None and v not in constants.EYE_LATERALITY:
            raise ValueError(f"Invalid eye_laterality: {v}. Must be one of {constants.EYE_LATERALITY}")
        return v

    @field_validator("storage_provider")
    @classmethod
    def validate_storage_provider(cls, v: str) -> str:
        """Validate storage provider."""
        if v not in constants.STORAGE_PROVIDERS:
            raise ValueError(f"Invalid storage_provider: {v}. Must be one of {constants.STORAGE_PROVIDERS}")
        return v


# ============================================
# Image <-> Dataset cross-membership
# ============================================

class ImageDatasetMembership(BaseModel):
    """Model for image_dataset_memberships join table.

    Records that a canonical image (owned by one dataset) is also a member of
    a secondary dataset, e.g. MAPLES-DR annotations overlaid on MESSIDOR images.
    """

    image_id: uuid.UUID
    dataset_id: uuid.UUID
    original_image_id: Optional[str] = None
    added_at: datetime = Field(default_factory=datetime.now)


# ============================================
# Patient <-> Image relationship
# ============================================

class PatientImage(BaseModel):
    """Model for patient_images join table."""

    relationship_id: uuid.UUID
    patient_id: uuid.UUID
    image_id: uuid.UUID
    exam_date: Optional[date] = None
    created_at: datetime = Field(default_factory=datetime.now)


# ============================================
# Raw Annotation Files
# ============================================

class RawAnnotationFile(BaseModel):
    """Model for raw_annotation_files table."""

    raw_file_id: uuid.UUID
    dataset_id: uuid.UUID
    storage_provider: str = "local"
    bucket: Optional[str] = None
    object_key: Optional[str] = None
    version_id: Optional[str] = None
    file_path: Optional[str] = None
    file_type: Optional[str] = None
    file_name: Optional[str] = None
    file_hash: Optional[str] = None
    file_size: Optional[int] = None
    encoding: Optional[str] = None
    parsed_status: str = "not_parsed"
    parse_errors: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None

    @field_validator("parsed_status")
    @classmethod
    def validate_parsed_status(cls, v: str) -> str:
        """Validate parsed status."""
        valid_statuses = {"not_parsed", "parsed", "error"}
        if v not in valid_statuses:
            raise ValueError(f"Invalid parsed_status: {v}. Must be one of {valid_statuses}")
        return v

    @field_validator("storage_provider")
    @classmethod
    def validate_storage_provider(cls, v: str) -> str:
        """Validate storage provider."""
        if v not in constants.STORAGE_PROVIDERS:
            raise ValueError(f"Invalid storage_provider: {v}. Must be one of {constants.STORAGE_PROVIDERS}")
        return v


# ============================================
# Expert Annotations
# ============================================

class ExpertAnnotation(BaseModel):
    """Model for expert_annotations table."""

    expert_annotation_id: uuid.UUID
    expert_id: uuid.UUID
    annotation_task: str
    raw_data_id: Optional[uuid.UUID] = None
    annotation_value: Optional[dict[str, Any]] = None
    confidence_level: Optional[str] = None
    annotation_timestamp: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.now)

    @field_validator("annotation_task")
    @classmethod
    def validate_annotation_task(cls, v: str) -> str:
        """Validate annotation task."""
        if v not in constants.ANNOTATION_TASKS:
            raise ValueError(f"Invalid annotation_task: {v}. Must be one of {constants.ANNOTATION_TASKS}")
        return v

    @field_validator("confidence_level")
    @classmethod
    def validate_confidence_level(cls, v: Optional[str]) -> Optional[str]:
        """Validate confidence level."""
        if v is not None and v not in constants.CONFIDENCE_LEVELS:
            raise ValueError(f"Invalid confidence_level: {v}. Must be one of {constants.CONFIDENCE_LEVELS}")
        return v


# ============================================
# Annotation Type Vocabulary
# ============================================

class AnnotationType(BaseModel):
    """Model for annotation_type table."""

    annotation_type_id: uuid.UUID
    annotation_type: str
    annotation_description: Optional[str] = None


# ============================================
# Grading Scales
# ============================================

class GradingScale(BaseModel):
    """Model for grading_scales table."""

    scale_id: uuid.UUID
    scale_name: str
    disease_type: str
    scale_description: Optional[str] = None
    min_value: Optional[int] = None
    max_value: Optional[int] = None
    value_labels: Optional[dict[str, Any]] = None

    @field_validator("disease_type")
    @classmethod
    def validate_disease_type(cls, v: str) -> str:
        """Validate disease type."""
        if v not in constants.DISEASE_TYPES:
            raise ValueError(f"Invalid disease_type: {v}. Must be one of {constants.DISEASE_TYPES}")
        return v


# ============================================
# Grading Scale Mappings
# ============================================

class GradingScaleMapping(BaseModel):
    """Model for grading_scale_mappings table."""

    mapping_id: uuid.UUID
    source_scale_id: uuid.UUID
    target_scale_id: uuid.UUID
    source_value: str
    target_value: Optional[int] = None
    mapping_confidence: str = "exact"

    @field_validator("mapping_confidence")
    @classmethod
    def validate_mapping_confidence(cls, v: str) -> str:
        """Validate mapping confidence."""
        valid_confidences = {"exact", "approximate", "manual_review_required"}
        if v not in valid_confidences:
            raise ValueError(f"Invalid mapping_confidence: {v}. Must be one of {valid_confidences}")
        return v


# ============================================
# Consensus Annotations
# ============================================

class ConsensusAnnotation(BaseModel):
    """Model for consensus_annotations table."""

    consensus_id: uuid.UUID
    image_id: uuid.UUID
    annotation_task: str
    consensus_method: str
    expert_annotation_ids: Optional[list[uuid.UUID]] = None
    consensus_value: Optional[dict[str, Any]] = None
    agreement_score: Optional[float] = None
    disagreement_details: Optional[dict[str, Any]] = None
    adjudicator_id: Optional[uuid.UUID] = None
    created_at: datetime = Field(default_factory=datetime.now)

    @field_validator("annotation_task")
    @classmethod
    def validate_annotation_task(cls, v: str) -> str:
        """Validate annotation task."""
        valid_tasks = {"grading", "segmentation", "classification", "localization", "quality"}
        if v not in valid_tasks:
            raise ValueError(f"Invalid annotation_task: {v}")
        return v

    @field_validator("consensus_method")
    @classmethod
    def validate_consensus_method(cls, v: str) -> str:
        """Validate consensus method."""
        valid_methods = {"majority_vote", "mean", "median", "staple", "adjudicated", "senior_review"}
        if v not in valid_methods:
            raise ValueError(f"Invalid consensus_method: {v}")
        return v


# ============================================
# Provenance Chain
# ============================================

class ProvenanceChain(BaseModel):
    """Model for provenance_chain table."""

    chain_id: uuid.UUID
    unified_annotation_type: str
    source_type: str
    root_source_raw_data_id: Optional[uuid.UUID] = None
    source_annotation_ids: Optional[list[uuid.UUID]] = None
    created_at: datetime = Field(default_factory=datetime.now)

    @field_validator("unified_annotation_type")
    @classmethod
    def validate_unified_annotation_type(cls, v: str) -> str:
        """Validate unified annotation type."""
        if v not in constants.ANNOTATION_TASKS:
            raise ValueError(f"Invalid unified_annotation_type: {v}")
        return v

    @field_validator("source_type")
    @classmethod
    def validate_source_type(cls, v: str) -> str:
        """Validate source type."""
        if v not in constants.PROVENANCE_SOURCE_TYPES:
            raise ValueError(f"Invalid source_type: {v}")
        return v


# ============================================
# Transformation Operations
# ============================================

class TransformationOperation(BaseModel):
    """Model for transformation_operations table."""

    transformation_id: uuid.UUID
    operation_type: str
    input_data: Optional[dict[str, Any]] = None
    output_data: Optional[dict[str, Any]] = None
    operation_parameters: Optional[dict[str, Any]] = None
    operation_timestamp: datetime = Field(default_factory=datetime.now)
    operator: Optional[str] = None
    notes: Optional[str] = None


# ============================================
# Provenance Transformations
# ============================================

class ProvenanceTransformation(BaseModel):
    """Model for provenance_transformations join table."""

    id: uuid.UUID
    chain_id: uuid.UUID
    transformation_id: uuid.UUID
    created_at: datetime = Field(default_factory=datetime.now)


# ============================================
# Segmentation Annotations
# ============================================

class SegmentationAnnotation(BaseModel):
    """Model for segmentation_annotations table."""

    segmentation_id: uuid.UUID
    image_id: uuid.UUID
    annotation_type_id: uuid.UUID
    lesion_subtype: Optional[str] = None
    mask_file_path: Optional[str] = None
    group_id: Optional[uuid.UUID] = None
    unified_format: Optional[str] = None
    original_format: Optional[str] = None
    original_file_path: Optional[str] = None
    raw_data_id: Optional[uuid.UUID] = None
    coordinate_system: Optional[str] = None
    expert_annotation_id: Optional[uuid.UUID] = None
    consensus_id: Optional[uuid.UUID] = None
    annotation_method: str = "manual"
    confidence_score: Optional[float] = None
    provenance_chain_id: Optional[uuid.UUID] = None
    created_at: datetime = Field(default_factory=datetime.now)

    @field_validator("annotation_method")
    @classmethod
    def validate_annotation_method(cls, v: str) -> str:
        """Validate annotation method."""
        valid_methods = {"manual", "semi_automatic", "automatic", "pseudo"}
        if v not in valid_methods:
            raise ValueError(f"Invalid annotation_method: {v}")
        return v


# ============================================
# Disease Grading
# ============================================

class DiseaseGrading(BaseModel):
    """Model for disease_grading table."""

    grading_id: uuid.UUID
    image_id: uuid.UUID
    disease_type: str
    scale_id: uuid.UUID
    original_grade: Optional[str] = None
    scaled_grade: Optional[int] = None
    grade_label: Optional[str] = None
    raw_data_id: Optional[uuid.UUID] = None
    expert_annotation_id: Optional[uuid.UUID] = None
    consensus_id: Optional[uuid.UUID] = None
    annotation_method: str = "manual"
    confidence_score: Optional[float] = None
    provenance_chain_id: Optional[uuid.UUID] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None

    @field_validator("disease_type")
    @classmethod
    def validate_disease_type(cls, v: str) -> str:
        """Validate disease type."""
        if v not in constants.DISEASE_TYPES:
            raise ValueError(f"Invalid disease_type: {v}")
        return v

    @field_validator("annotation_method")
    @classmethod
    def validate_annotation_method(cls, v: str) -> str:
        """Validate annotation method."""
        valid_methods = {"manual", "adjudicated", "consensus", "pseudo"}
        if v not in valid_methods:
            raise ValueError(f"Invalid annotation_method: {v}")
        return v


# ============================================
# Localization Annotations
# ============================================

class LocalizationAnnotation(BaseModel):
    """Model for localization_annotations table."""

    localization_id: uuid.UUID
    image_id: uuid.UUID
    localization_type: str
    target_structure: str
    coordinates: dict[str, Any]
    lesion_subtype: Optional[str] = None
    raw_data_id: Optional[uuid.UUID] = None
    expert_annotation_id: Optional[uuid.UUID] = None
    consensus_id: Optional[uuid.UUID] = None
    annotation_method: str = "manual"
    provenance_chain_id: Optional[uuid.UUID] = None
    created_at: datetime = Field(default_factory=datetime.now)

    @field_validator("localization_type")
    @classmethod
    def validate_localization_type(cls, v: str) -> str:
        """Validate localization type."""
        valid_types = {"bounding_box", "keypoint", "center_point"}
        if v not in valid_types:
            raise ValueError(f"Invalid localization_type: {v}")
        return v

    @field_validator("annotation_method")
    @classmethod
    def validate_annotation_method(cls, v: str) -> str:
        """Validate annotation method."""
        valid_methods = {"manual", "pseudo"}
        if v not in valid_methods:
            raise ValueError(f"Invalid annotation_method: {v}")
        return v


# ============================================
# Classification Annotations
# ============================================

class ClassificationAnnotation(BaseModel):
    """Model for classification_annotations table."""

    classification_id: uuid.UUID
    image_id: uuid.UUID
    task_type: str
    task_name: str
    class_name: str
    concept: Optional[str] = None
    is_multilabel: bool
    class_index: int
    class_label: str
    sub_key: Optional[str] = None
    class_value: Optional[dict[str, Any]] = None
    raw_data_id: Optional[uuid.UUID] = None
    expert_annotation_id: Optional[uuid.UUID] = None
    consensus_id: Optional[uuid.UUID] = None
    annotation_method: str = "manual"
    confidence_score: Optional[float] = None
    provenance_chain_id: Optional[uuid.UUID] = None
    created_at: datetime = Field(default_factory=datetime.now)

    @field_validator("task_type")
    @classmethod
    def validate_task_type(cls, v: str) -> str:
        """Validate task type."""
        valid_types = {"binary", "multi_class", "multi_label"}
        if v not in valid_types:
            raise ValueError(f"Invalid task_type: {v}")
        return v

    @field_validator("annotation_method")
    @classmethod
    def validate_annotation_method(cls, v: str) -> str:
        """Validate annotation method."""
        valid_methods = {"manual", "consensus", "pseudo"}
        if v not in valid_methods:
            raise ValueError(f"Invalid annotation_method: {v}")
        return v


# ============================================
# Quality Annotations
# ============================================

class QualityType(BaseModel):
    """Model for the quality_types reference table (extensible quality vocabulary)."""

    quality_type: str
    description: Optional[str] = None
    category: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)


class QualityAnnotation(BaseModel):
    """Model for quality_annotations table."""

    quality_id: uuid.UUID
    image_id: uuid.UUID
    quality_type: str
    quality_score: Optional[float] = None
    quality_label: Optional[str] = None
    scale_description: Optional[str] = None
    raw_data_id: Optional[uuid.UUID] = None
    expert_annotation_id: Optional[uuid.UUID] = None
    provenance_chain_id: Optional[uuid.UUID] = None
    created_at: datetime = Field(default_factory=datetime.now)

    @field_validator("quality_type")
    @classmethod
    def validate_quality_type(cls, v: str) -> str:
        """quality_type is validated against the quality_types reference table at the DB
        layer (FK). Here we only require it to be a non-empty identifier."""
        if not v or not v.strip():
            raise ValueError("quality_type must be a non-empty string")
        return v.strip()


# ============================================
# Clinical Descriptions
# ============================================

class ClinicalDescription(BaseModel):
    """Model for clinical_descriptions table."""

    description_id: uuid.UUID
    image_id: uuid.UUID
    description_text: str
    description_type: str
    raw_data_id: Optional[uuid.UUID] = None
    expert_id: Optional[uuid.UUID] = None
    word_count: Optional[int] = None
    created_at: datetime = Field(default_factory=datetime.now)

    @field_validator("description_type")
    @classmethod
    def validate_description_type(cls, v: str) -> str:
        """Validate description type."""
        valid_types = {"clinical_caption", "diagnosis_text", "notes"}
        if v not in valid_types:
            raise ValueError(f"Invalid description_type: {v}")
        return v


# ============================================
# Keywords
# ============================================

class KeywordVocabulary(BaseModel):
    """Model for keyword_vocabulary table."""

    keyword_id: uuid.UUID
    keyword_term: str
    keyword_source: str
    category: Optional[str] = None
    dataset_id: uuid.UUID
    created_at: datetime = Field(default_factory=datetime.now)

    @field_validator("keyword_source")
    @classmethod
    def validate_keyword_source(cls, v: str) -> str:
        """Validate keyword source."""
        valid_sources = {"diagnostic_keywords", "clinical_description", "diagnosis_text"}
        if v not in valid_sources:
            raise ValueError(f"Invalid keyword_source: {v}")
        return v


class KeywordAnnotation(BaseModel):
    """Model for keyword_annotations table."""

    keyword_annotation_id: uuid.UUID
    image_id: uuid.UUID
    keyword_id: uuid.UUID
    keyword_text: Optional[str] = None
    raw_data_id: Optional[uuid.UUID] = None
    expert_id: Optional[uuid.UUID] = None
    annotation_method: str = "manual"
    provenance_chain_id: Optional[uuid.UUID] = None
    created_at: datetime = Field(default_factory=datetime.now)

    @field_validator("annotation_method")
    @classmethod
    def validate_annotation_method(cls, v: str) -> str:
        """Validate annotation method."""
        valid_methods = {"manual", "extracted", "pseudo"}
        if v not in valid_methods:
            raise ValueError(f"Invalid annotation_method: {v}")
        return v


# ============================================
# Dataset Splits
# ============================================

class DatasetSplit(BaseModel):
    """Model for dataset_splits table."""

    split_id: uuid.UUID
    dataset_id: uuid.UUID
    split_name: str
    split_type: str
    task_type: Optional[str] = None
    image_count: Optional[int] = None
    created_at: datetime = Field(default_factory=datetime.now)

    @field_validator("split_type")
    @classmethod
    def validate_split_type(cls, v: str) -> str:
        """Validate split type."""
        if v not in constants.SPLIT_TYPES:
            raise ValueError(f"Invalid split_type: {v}")
        return v


class ImageSplit(BaseModel):
    """Model for image_splits table."""

    assignment_id: uuid.UUID
    image_id: uuid.UUID
    split_id: uuid.UUID
    task_type: Optional[str] = None
    is_primary: bool = True
    created_at: datetime = Field(default_factory=datetime.now)
