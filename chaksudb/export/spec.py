"""
ExportSpec: User intent specification for data exports.

Defines all query dimensions and filters for exporting image data and annotations
from the database to Parquet or PyTorch DataLoader formats.
"""

from typing import Any, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, model_validator

from chaksudb.config.config import constants


class AnnotationOrFilter(BaseModel):
    """
    Defines OR conditions within an annotation task's sub-filters.
    
    Each condition dict is evaluated separately, and the results are ORed together.
    When provided for a task, this replaces the flat filter fields for that task.
    
    Example:
        >>> # "lesion microaneurysm OR vessel segmentation"
        >>> AnnotationOrFilter(
        ...     task="segmentation",
        ...     conditions=[
        ...         {"segmentation_types": ["lesion"], "lesion_subtypes": ["microaneurysm"]},
        ...         {"segmentation_types": ["vessel"]},
        ...     ]
        ... )
    """
    task: str = Field(
        description="Annotation task name (e.g., 'segmentation', 'localization')"
    )
    conditions: list[dict[str, Any]] = Field(
        description="List of filter dicts to be ORed together. "
                    "Each dict contains filter key-value pairs for the task."
    )


class ExportSpec(BaseModel):
    """
    Specification for data export queries.

    Defines filters and options for exporting image data and annotations.
    All fields are optional, allowing flexible query construction.

    Examples:
        >>> # Export all images from a specific dataset
        >>> spec = ExportSpec(dataset_names=["EYEPACS"])
        >>>
        >>> # Export DR grading data for training
        >>> spec = ExportSpec(
        ...     dataset_names=["EYEPACS"],
        ...     split_names=["train"],
        ...     annotation_tasks=["grading"],
        ...     disease_types=["DR"],
        ...     annotation_source="prefer_consensus"
        ... )
        >>>
        >>> # Export with segmentation masks
        >>> spec = ExportSpec(
        ...     annotation_tasks=["grading", "segmentation"],
        ...     modalities=["fundus"],
        ...     require_annotations=True
        ... )
    """

    # Dataset filters
    dataset_ids: Optional[list[UUID]] = Field(
        default=None,
        description="Filter by specific dataset UUIDs"
    )
    dataset_names: Optional[list[str]] = Field(
        default=None,
        description="Filter by dataset names"
    )

    # Split filters
    split_names: Optional[list[str]] = Field(
        default=None,
        description="Filter by split names (e.g., ['train', 'test', 'val'])"
    )
    split_task_type: Optional[str] = Field(
        default=None,
        description="Filter by split task type"
    )
    split_type: Optional[str] = Field(
        default=None,
        description="Filter by dataset_splits.split_type "
                    "(e.g. 'explicit', 'user_defined', 'metadata_defined'). "
                    "Use 'user_defined' to get splits assigned by assign_splits.py, "
                    "which is the only type that includes a val split."
    )

    # Image scope filters
    modalities: Optional[list[str]] = Field(
        default=None,
        description="Filter by image modalities (e.g., ['fundus', 'oct','uwf','fa'])"
    )
    storage_provider: Optional[str] = Field(
        default="local",
        description="Filter by storage provider (e.g., 'local', 's3', 'gcs'). Default is 'local'."
    )
    oct_key_frames_only: bool = Field(
        default=False,
        description="When True, restrict OCT images to key frames only (frame_index = 0, i.e. the B-scan). "
                    "Volume frames (frame_index >= 1) are excluded. Has no effect on non-OCT images. "
                    "Combine with modalities=['oct'] to get only B-scans from OCT datasets."
    )

    # Annotation axis
    annotation_tasks: Optional[list[str]] = Field(
        default=None,
        description="List of annotation task types to include: "
                    "['grading', 'segmentation', 'classification', 'localization', 'quality', 'keyword', 'description']"
    )

    # Disease axis (for grading)
    disease_types: Optional[list[str]] = Field(
        default=None,
        description="Filter by disease types for grading (e.g., ['DR', 'DME', 'Glaucoma', 'AMD'])"
    )

    # Segmentation filters (when annotation_tasks includes 'segmentation')
    segmentation_types: Optional[list[str]] = Field(
        default=None,
        description="Filter segmentations by annotation type (e.g. ['vessel', 'optic_disc', 'lesion']). "
                    "Requires 'segmentation' in annotation_tasks."
    )
    lesion_subtypes: Optional[list[str]] = Field(
        default=None,
        description="Filter segmentations by lesion subtype (e.g. ['microaneurysm', 'hemorrhage']). "
                    "Requires 'segmentation' in annotation_tasks."
    )

    # Localization filters (when annotation_tasks includes 'localization')
    localization_types: Optional[list[str]] = Field(
        default=None,
        description="Filter localizations by localization type (e.g. ['bounding_box', 'keypoint', 'center_point']). "
                    "Requires 'localization' in annotation_tasks."
    )

    # Source axis (consensus/expert preference)
    annotation_source: Literal["expert_only", "consensus_only", "prefer_consensus", "both"] = Field(
        default="prefer_consensus",
        description="How to handle expert vs consensus annotations: "
                    "'expert_only' (only expert annotations), "
                    "'consensus_only' (only consensus annotations), "
                    "'prefer_consensus' (prefer consensus, fallback to expert), "
                    "'both' (include both when available)"
    )

    # Grading options
    grading_scale_name: Optional[str] = Field(
        default=None,
        description="Filter by specific grading scale name"
    )
    include_original_grade: bool = Field(
        default=True,
        description="Include original grade text in export"
    )
    include_scaled_grade: bool = Field(
        default=True,
        description="Include scaled grade integer in export"
    )

    # Filters
    grade_filter: Optional[dict[str, Any]] = Field(
        default=None,
        description="Filter by grade values. Supports multiple formats: "
                    "{'DR': {'min': 1, 'max': 3}} for range, "
                    "{'DR': {'values': [0, 1, 2]}} for specific values, "
                    "{'DR': [0, 1, 2]} as shorthand for values. "
                    "Applied in HAVING clause after aggregation"
    )
    classification_filter: Optional[dict[str, Any]] = Field(
        default=None,
        description="Filter classifications by class_name and/or class_value. "
                    "Format: {'class_names': ['glaucoma', 'normal'], 'class_values': [1, 2]} "
                    "or {'task_type': 'binary', 'class_name': 'glaucoma'}. "
                    "Requires 'classification' in annotation_tasks."
    )
    
    # Classification pivoting options
    classification_class_names: Optional[list[str]] = Field(
        default=None,
        description="List of class_names to pivot into flat columns. "
                    "Each class_name becomes {class_name}_label (int/float) and "
                    "{class_name}_class_label (str) columns. "
                    "REQUIRED when 'classification' is in annotation_tasks."
    )
    classification_label_type: Literal["int", "float"] = Field(
        default="int",
        description="Type for label columns: 'int' for CrossEntropyLoss/BCEWithLogitsLoss, "
                    "'float' for soft labels or BCELoss. Default 'int'."
    )
    classification_task_types: Optional[dict[str, str]] = Field(
        default=None,
        description="Map each class_name to its task_type: 'binary', 'multi_class', or 'multi_label'. "
                    "Format: {'glaucoma': 'binary', 'disease_category': 'multi_class'}. "
                    "If not provided, the module will auto-detect from the database."
    )
    multi_label_keys: Optional[dict[str, list[str]]] = Field(
        default=None,
        description="For multi-label class_names, the sublabel keys to flatten. "
                    "Format: {'disease_indicators': ['normal', 'diabetes', 'glaucoma', ...]}. "
                    "If not provided, multi-label annotations are kept as JSON string."
    )

    # Concept-centric classification (the recommended ML-task interface).
    # A concept (e.g. 'glaucoma') is retrievable as a unified binary column regardless of
    # whether a dataset stored it as binary, a multi_label sub-key, or a multi_class winner.
    classification_concepts: Optional[list[str]] = Field(
        default=None,
        description="Canonical concepts (e.g. ['glaucoma', 'DR', 'AMD']) to export as "
                    "per-concept binary presence columns ({concept}_present, 0/1), unioned "
                    "across all storage shapes via the classification 'concept' field. "
                    "This is the cross-dataset 'glaucoma binary classification' interface. "
                    "Requires 'classification' in annotation_tasks."
    )
    classification_positive_for: Optional[list[str]] = Field(
        default=None,
        description="Keep only images that are POSITIVE for at least one of these concepts, "
                    "regardless of storage shape (binary class_index=1, multi_label sub-key on, "
                    "or multi_class winning class mapping to the concept). "
                    "Requires 'classification' in annotation_tasks."
    )

    # Annotation requirement mode (replaces require_annotations)
    require_annotations_mode: Literal["none", "all", "any"] = Field(
        default="none",
        description="How to handle images without requested annotations: "
                    "'none' (include all images, LEFT JOIN annotations), "
                    "'all' (only images with ALL requested annotation tasks, INNER JOIN), "
                    "'any' (only images with at least ONE requested annotation task). "
                    "Default 'none'."
    )
    
    require_annotations: Optional[bool] = Field(
        default=None,
        description="DEPRECATED: Use require_annotations_mode instead. "
                    "If True, maps to 'all'. If False, maps to 'none'."
    )
    
    # OR filter groups for annotation tasks
    annotation_or_filters: Optional[list[AnnotationOrFilter]] = Field(
        default=None,
        description="Define OR conditions within annotation task sub-filters. "
                    "When provided for a task, replaces the flat filter fields for that task. "
                    "Example: [AnnotationOrFilter(task='segmentation', conditions=[...])]. "
                    "Requires annotation_tasks to include the specified tasks."
    )

    # Patient data
    include_patient_data: bool = Field(
        default=False,
        description="Include patient demographics (age, sex, ethnicity, comorbidities) in export"
    )

    # Caption generation
    caption_mode: Optional[Literal["clinical", "keyword", "grading", "classification", "synthetic", "all"]] = Field(
        default=None,
        description="Enable caption generation. "
                    "'clinical' uses clinical_descriptions text, "
                    "'keyword' uses keyword terms enriched by the definitions dictionary, "
                    "'grading' generates text from disease grading labels + definitions, "
                    "'classification' generates text from classification labels + definitions, "
                    "'synthetic' combines grading + classification + localization + segmentation structures, "
                    "'all' combines every available source into one rich caption."
    )

    # Detection format
    detection_format: Literal["nested", "coco"] = Field(
        default="nested",
        description="Format for localization export: 'nested' (JSONB array) or "
                    "'coco' (COCO-style bbox list with category_id)"
    )
    detection_category_map: Optional[dict[str, int]] = Field(
        default=None,
        description="Map target_structure names to integer category IDs for COCO format. "
                    "E.g., {'lesions': 1, 'optic_disc': 2}"
    )

    # Fundus ROI
    include_fundus_roi: bool = Field(
        default=False,
        description="Add flat fundus ROI circle columns to the export: "
                    "fundus_roi_cx, fundus_roi_cy, fundus_roi_radius, fundus_roi_method. "
                    "Requires ROI annotations stored by run_roi_iqa.py."
    )

    # Health status: a single cross-dataset normal/abnormal derived field.
    include_health_status: bool = Field(
        default=False,
        description="Add a 'health_status' column ('normal' | 'abnormal' | None) derived "
                    "across grading + disease classification: abnormal = any disease grade>=1 "
                    "or any disease concept positive; normal = assessed for disease with "
                    "nothing positive; None = never assessed."
    )
    health_status_filter: Optional[Literal["normal", "abnormal"]] = Field(
        default=None,
        description="Keep only 'normal' or only 'abnormal' images (unassessed/unknown "
                    "images are excluded). Implies include_health_status."
    )

    # Quality pivoting: which quality_type columns to emit. When None the module pivots a
    # default set; set this (e.g. via build_dataset_spec) to emit only the types a dataset
    # actually has and avoid all-NULL columns.
    quality_types: Optional[list[str]] = Field(
        default=None,
        description="quality_type values to pivot into {type}_quality_score/_label columns. "
                    "If None, a default set is used. Requires 'quality' in annotation_tasks."
    )

    # IQA quality filter
    iqa_min_quality_score: Optional[float] = Field(
        default=None,
        description="Only export images whose QuickQual overall quality_score >= this value "
                    "(0.0–1.0, where 1.0 = certainly good). Images with no IQA annotation "
                    "are excluded when this filter is set. Requires run_roi_iqa.py to have been run."
    )
    iqa_quality_labels: Optional[list[str]] = Field(
        default=None,
        description="Only export images whose QuickQual quality_label is in this list "
                    "(e.g. ['good'], ['good', 'usable']). Images with no IQA annotation "
                    "are excluded. Requires run_roi_iqa.py to have been run."
    )

    # Output format
    output_format: Optional[Literal["classification", "grading", "segmentation", "detection", "vision_language", "ssl"]] = Field(
        default=None,
        description="When set, __getitem__ returns a standard output for the given task "
                    "instead of the generic (image, dict) tuple. "
                    "'classification': (image, int) single class or (image, dict[str,int]) multi-class; "
                    "'grading': (image, int) single disease or (image, dict[str,int]) multi-disease; "
                    "'segmentation': (image, dict[str, PIL.Image]) mapping structure → mask; "
                    "'detection': (image, {'boxes': [...], 'labels': [...], 'keypoints': [...]}); "
                    "'vision_language': (image, str) using the synthesized caption column (requires caption_mode); "
                    "'ssl': image only, no label — for self-supervised pre-training. "
                    "None preserves existing behavior (backward-compatible)."
    )

    # Path handling
    base_path_for_paths: Optional[str] = Field(
        default=None,
        description="Base path prefix to prepend to file_path values. "
                    "Useful for relocating datasets or converting relative to absolute paths"
    )

    @field_validator("modalities")
    @classmethod
    def validate_modalities(cls, v: Optional[list[str]]) -> Optional[list[str]]:
        """Validate modality values against schema constraints."""
        if v is not None:
            invalid = set(v) - constants.MODALITIES
            if invalid:
                raise ValueError(
                    f"Invalid modalities: {invalid}. "
                    f"Must be one of {sorted(constants.MODALITIES)}"
                )
        return v

    @field_validator("storage_provider")
    @classmethod
    def validate_storage_provider(cls, v: Optional[str]) -> Optional[str]:
        """Validate storage provider against schema constraints."""
        if v is not None and v not in constants.STORAGE_PROVIDERS:
            raise ValueError(
                f"Invalid storage_provider: {v}. "
                f"Must be one of {sorted(constants.STORAGE_PROVIDERS)}"
            )
        return v

    @field_validator("annotation_tasks")
    @classmethod
    def validate_annotation_tasks(cls, v: Optional[list[str]]) -> Optional[list[str]]:
        """Validate annotation task types against schema constraints."""
        if v is not None:
            invalid = set(v) - constants.ANNOTATION_TASKS
            if invalid:
                raise ValueError(
                    f"Invalid annotation_tasks: {invalid}. "
                    f"Must be one of {sorted(constants.ANNOTATION_TASKS)}"
                )
        return v

    @field_validator("disease_types")
    @classmethod
    def validate_disease_types(cls, v: Optional[list[str]]) -> Optional[list[str]]:
        """Validate disease types against schema constraints."""
        if v is not None:
            invalid = set(v) - constants.DISEASE_TYPES
            if invalid:
                raise ValueError(
                    f"Invalid disease_types: {invalid}. "
                    f"Must be one of {sorted(constants.DISEASE_TYPES)}"
                )
        return v

    @model_validator(mode="after")
    def validate_spec_consistency(self) -> "ExportSpec":
        """
        Validate that spec combinations are consistent.

        - disease_types requires grading in annotation_tasks
        - grading_scale_name requires grading in annotation_tasks
        - grade_filter requires grading in annotation_tasks
        """
        has_grading = (
            self.annotation_tasks is not None
            and "grading" in self.annotation_tasks
        )

        if self.disease_types is not None and not has_grading:
            raise ValueError(
                "disease_types requires 'grading' in annotation_tasks. "
                "Add 'grading' to annotation_tasks to filter by disease types."
            )

        if self.grading_scale_name is not None and not has_grading:
            raise ValueError(
                "grading_scale_name requires 'grading' in annotation_tasks. "
                "Add 'grading' to annotation_tasks to filter by grading scale."
            )

        if self.grade_filter is not None and not has_grading:
            raise ValueError(
                "grade_filter requires 'grading' in annotation_tasks. "
                "Add 'grading' to annotation_tasks to use grade filtering."
            )

        # grade_filter keys must be among the requested disease_types, otherwise the
        # HAVING references an always-NULL column and silently drops every row.
        if self.grade_filter is not None and self.disease_types is not None:
            unknown = set(self.grade_filter) - set(self.disease_types)
            if unknown:
                raise ValueError(
                    f"grade_filter references diseases not in disease_types: {sorted(unknown)}. "
                    f"Add them to disease_types or remove them from grade_filter."
                )

        has_segmentation = (
            self.annotation_tasks is not None
            and "segmentation" in self.annotation_tasks
        )
        if self.segmentation_types is not None and not has_segmentation:
            raise ValueError(
                "segmentation_types requires 'segmentation' in annotation_tasks. "
                "Add 'segmentation' to annotation_tasks to filter by segmentation type."
            )
        if self.lesion_subtypes is not None and not has_segmentation:
            raise ValueError(
                "lesion_subtypes requires 'segmentation' in annotation_tasks. "
                "Add 'segmentation' to annotation_tasks to filter by lesion subtype."
            )

        has_quality = (
            self.annotation_tasks is not None
            and "quality" in self.annotation_tasks
        )
        if self.quality_types is not None and not has_quality:
            raise ValueError(
                "quality_types requires 'quality' in annotation_tasks."
            )

        has_localization = (
            self.annotation_tasks is not None
            and "localization" in self.annotation_tasks
        )
        if self.localization_types is not None and not has_localization:
            raise ValueError(
                "localization_types requires 'localization' in annotation_tasks. "
                "Add 'localization' to annotation_tasks to filter by localization type."
            )

        has_classification = (
            self.annotation_tasks is not None
            and "classification" in self.annotation_tasks
        )
        if self.classification_filter is not None and not has_classification:
            raise ValueError(
                "classification_filter requires 'classification' in annotation_tasks. "
                "Add 'classification' to annotation_tasks to use classification filtering."
            )
        
        if self.classification_class_names is not None and not has_classification:
            raise ValueError(
                "classification_class_names requires 'classification' in annotation_tasks. "
                "Add 'classification' to annotation_tasks to use classification pivoting."
            )

        if self.classification_concepts is not None and not has_classification:
            raise ValueError(
                "classification_concepts requires 'classification' in annotation_tasks."
            )
        if self.classification_positive_for is not None and not has_classification:
            raise ValueError(
                "classification_positive_for requires 'classification' in annotation_tasks."
            )

        # classification_class_names is no longer mandatory: when omitted (and no concepts
        # are requested) the export auto-discovers every task present for the datasets.

        if self.multi_label_keys is not None and self.classification_class_names is None:
            raise ValueError(
                "multi_label_keys requires classification_class_names to be set. "
                "Specify which class_names to pivot before providing multi-label keys."
            )

        # Validate detection_format requires localization
        has_localization = (
            self.annotation_tasks is not None
            and "localization" in self.annotation_tasks
        )
        if self.detection_format == "coco" and not has_localization:
            raise ValueError(
                "detection_format='coco' requires 'localization' in annotation_tasks."
            )
        if self.detection_category_map is not None and not has_localization:
            raise ValueError(
                "detection_category_map requires 'localization' in annotation_tasks."
            )

        # Handle backward compatibility for require_annotations
        if self.require_annotations is not None:
            # User provided deprecated field
            if self.require_annotations_mode != "none":
                # Both provided and non-default
                import warnings
                warnings.warn(
                    "Both 'require_annotations' and 'require_annotations_mode' are set. "
                    "Using 'require_annotations_mode' and ignoring 'require_annotations'. "
                    "Please migrate to 'require_annotations_mode' only.",
                    DeprecationWarning,
                    stacklevel=2
                )
            else:
                # Map old field to new field
                if self.require_annotations:
                    self.require_annotations_mode = "all"
                else:
                    self.require_annotations_mode = "none"

        # Validate annotation_or_filters
        if self.annotation_or_filters is not None:
            if self.annotation_tasks is None:
                raise ValueError(
                    "annotation_or_filters requires annotation_tasks to be set. "
                    "Specify which annotation tasks to include."
                )
            
            # Check that all OR filter tasks are in annotation_tasks
            or_filter_tasks = {f.task for f in self.annotation_or_filters}
            invalid_tasks = or_filter_tasks - set(self.annotation_tasks)
            if invalid_tasks:
                raise ValueError(
                    f"annotation_or_filters contains tasks not in annotation_tasks: {invalid_tasks}. "
                    f"All OR filter tasks must be in annotation_tasks."
                )
            
            # Validate task names against constants
            invalid = or_filter_tasks - constants.ANNOTATION_TASKS
            if invalid:
                raise ValueError(
                    f"Invalid task names in annotation_or_filters: {invalid}. "
                    f"Must be one of {sorted(constants.ANNOTATION_TASKS)}"
                )
            
            # Warn if both flat filters and OR filters target the same task
            import warnings
            for or_filter in self.annotation_or_filters:
                if or_filter.task == "segmentation":
                    if self.segmentation_types is not None or self.lesion_subtypes is not None:
                        warnings.warn(
                            f"Both annotation_or_filters and flat filters (segmentation_types/lesion_subtypes) "
                            f"are set for task 'segmentation'. The flat filters will be ignored in favor of OR filters.",
                            UserWarning,
                            stacklevel=2
                        )
                elif or_filter.task == "localization":
                    if self.localization_types is not None:
                        warnings.warn(
                            f"Both annotation_or_filters and flat filters (localization_types) "
                            f"are set for task 'localization'. The flat filters will be ignored in favor of OR filters.",
                            UserWarning,
                            stacklevel=2
                        )

        return self
