"""
Tests for ExportSpec validation and public export API.

Tests ExportSpec field validation, enum constraints, and consistency checks.
Public API: use "from chaksudb.export import ExportSpec, export" only.
"""

import pytest
from uuid import UUID
from pydantic import ValidationError

from chaksudb.config.config import constants
from chaksudb.export import ExportSpec, export


class TestExportSpecValidation:
    """Tests for ExportSpec field validation."""

    def test_empty_spec_is_valid(self):
        """Test that an empty ExportSpec is valid (all fields optional)."""
        spec = ExportSpec()
        assert spec is not None
        assert spec.dataset_ids is None
        assert spec.annotation_tasks is None

    def test_valid_modalities(self):
        """Test that valid modalities are accepted."""
        for modality in constants.MODALITIES:
            spec = ExportSpec(modalities=[modality])
            assert spec.modalities == [modality]

    def test_invalid_modalities(self):
        """Test that invalid modalities raise ValueError."""
        with pytest.raises(ValueError, match="Invalid modalities"):
            ExportSpec(modalities=["invalid_modality"])

    def test_multiple_modalities(self):
        """Test that multiple valid modalities are accepted."""
        spec = ExportSpec(modalities=["fundus", "oct"])
        assert set(spec.modalities) == {"fundus", "oct"}

    def test_valid_storage_provider(self):
        """Test that valid storage providers are accepted."""
        for provider in constants.STORAGE_PROVIDERS:
            spec = ExportSpec(storage_provider=provider)
            assert spec.storage_provider == provider

    def test_invalid_storage_provider(self):
        """Test that invalid storage provider raises ValueError."""
        with pytest.raises(ValueError, match="Invalid storage_provider"):
            ExportSpec(storage_provider="invalid_provider")

    def test_valid_annotation_tasks(self):
        """Test that valid annotation tasks are accepted."""
        for task in constants.ANNOTATION_TASKS:
            if task == "classification":
                # classification requires classification_class_names
                spec = ExportSpec(annotation_tasks=[task], classification_class_names=["disease"])
            else:
                spec = ExportSpec(annotation_tasks=[task])
            assert task in spec.annotation_tasks

    def test_invalid_annotation_tasks(self):
        """Test that invalid annotation tasks raise ValueError."""
        with pytest.raises(ValueError, match="Invalid annotation_tasks"):
            ExportSpec(annotation_tasks=["invalid_task"])

    def test_multiple_annotation_tasks(self):
        """Test that multiple valid annotation tasks are accepted."""
        spec = ExportSpec(annotation_tasks=["grading", "segmentation"])
        assert set(spec.annotation_tasks) == {"grading", "segmentation"}

    def test_valid_disease_types(self):
        """Test that valid disease types are accepted."""
        for disease in constants.DISEASE_TYPES:
            spec = ExportSpec(
                annotation_tasks=["grading"],
                disease_types=[disease]
            )
            assert disease in spec.disease_types

    def test_invalid_disease_types(self):
        """Test that invalid disease types raise ValueError."""
        with pytest.raises(ValueError, match="Invalid disease_types"):
            ExportSpec(
                annotation_tasks=["grading"],
                disease_types=["invalid_disease"]
            )

    def test_annotation_source_default(self):
        """Test that annotation_source defaults to 'prefer_consensus'."""
        spec = ExportSpec()
        assert spec.annotation_source == "prefer_consensus"

    def test_valid_annotation_source(self):
        """Test that valid annotation_source values are accepted."""
        for source in ["expert_only", "consensus_only", "prefer_consensus", "both"]:
            spec = ExportSpec(annotation_source=source)
            assert spec.annotation_source == source

    def test_invalid_annotation_source_raises(self):
        """Test that invalid annotation_source raises ValidationError."""
        with pytest.raises(ValidationError, match="annotation_source"):
            ExportSpec(annotation_source="invalid_source")

    def test_include_original_grade_default(self):
        """Test that include_original_grade defaults to True."""
        spec = ExportSpec()
        assert spec.include_original_grade is True

    def test_include_scaled_grade_default(self):
        """Test that include_scaled_grade defaults to True."""
        spec = ExportSpec()
        assert spec.include_scaled_grade is True

    def test_require_annotations_default(self):
        """Test that require_annotations (deprecated) defaults to None."""
        spec = ExportSpec()
        assert spec.require_annotations is None

    def test_dataset_ids_with_uuids(self):
        """Test that dataset_ids accepts list of UUIDs."""
        uuids = [
            UUID("11111111-1111-1111-1111-111111111111"),
            UUID("22222222-2222-2222-2222-222222222222"),
        ]
        spec = ExportSpec(dataset_ids=uuids)
        assert spec.dataset_ids == uuids

    def test_dataset_names(self):
        """Test that dataset_names accepts list of strings."""
        spec = ExportSpec(dataset_names=["EYEPACS", "MESSIDOR"])
        assert spec.dataset_names == ["EYEPACS", "MESSIDOR"]

    def test_split_names(self):
        """Test that split_names accepts list of strings."""
        spec = ExportSpec(split_names=["train", "test", "val"])
        assert spec.split_names == ["train", "test", "val"]

    def test_split_task_type(self):
        """Test that split_task_type accepts a string."""
        spec = ExportSpec(split_task_type="classification")
        assert spec.split_task_type == "classification"

    def test_base_path_for_paths(self):
        """Test that base_path_for_paths accepts string."""
        spec = ExportSpec(base_path_for_paths="/data/images")
        assert spec.base_path_for_paths == "/data/images"


class TestExportSpecConsistency:
    """Tests for ExportSpec consistency validation."""

    def test_disease_types_requires_grading(self):
        """Test that disease_types requires grading in annotation_tasks."""
        with pytest.raises(ValueError, match="disease_types requires 'grading' in annotation_tasks"):
            ExportSpec(disease_types=["DR"])

    def test_disease_types_with_grading_is_valid(self):
        """Test that disease_types is valid when grading is in annotation_tasks."""
        spec = ExportSpec(
            annotation_tasks=["grading"],
            disease_types=["DR", "DME"]
        )
        assert spec.disease_types == ["DR", "DME"]

    def test_grading_scale_name_requires_grading(self):
        """Test that grading_scale_name requires grading in annotation_tasks."""
        with pytest.raises(ValueError, match="grading_scale_name requires 'grading' in annotation_tasks"):
            ExportSpec(grading_scale_name="ETDRS")

    def test_grading_scale_name_with_grading_is_valid(self):
        """Test that grading_scale_name is valid when grading is in annotation_tasks."""
        spec = ExportSpec(
            annotation_tasks=["grading"],
            grading_scale_name="ETDRS"
        )
        assert spec.grading_scale_name == "ETDRS"

    def test_grade_filter_requires_grading(self):
        """Test that grade_filter requires grading in annotation_tasks."""
        with pytest.raises(ValueError, match="grade_filter requires 'grading' in annotation_tasks"):
            ExportSpec(grade_filter={"DR": {"min": 1, "max": 3}})

    def test_grade_filter_with_grading_is_valid(self):
        """Test that grade_filter is valid when grading is in annotation_tasks."""
        spec = ExportSpec(
            annotation_tasks=["grading"],
            grade_filter={"DR": {"min": 1, "max": 3}}
        )
        assert spec.grade_filter == {"DR": {"min": 1, "max": 3}}

    def test_multiple_grading_requirements_together(self):
        """Test that multiple grading-related fields can be used together."""
        spec = ExportSpec(
            annotation_tasks=["grading"],
            disease_types=["DR", "DME"],
            grading_scale_name="ETDRS",
            grade_filter={"DR": {"min": 1, "max": 3}},
            include_original_grade=True,
            include_scaled_grade=True
        )
        assert spec.disease_types == ["DR", "DME"]
        assert spec.grading_scale_name == "ETDRS"
        assert spec.grade_filter == {"DR": {"min": 1, "max": 3}}

    def test_segmentation_types_requires_segmentation(self):
        """Test that segmentation_types requires segmentation in annotation_tasks."""
        with pytest.raises(ValueError, match="segmentation_types requires 'segmentation' in annotation_tasks"):
            ExportSpec(segmentation_types=["vessel", "optic_disc"])

    def test_lesion_subtypes_requires_segmentation(self):
        """Test that lesion_subtypes requires segmentation in annotation_tasks."""
        with pytest.raises(ValueError, match="lesion_subtypes requires 'segmentation' in annotation_tasks"):
            ExportSpec(lesion_subtypes=["microaneurysm"])

    def test_segmentation_types_with_segmentation_is_valid(self):
        """Test that segmentation_types is valid when segmentation is in annotation_tasks."""
        spec = ExportSpec(
            annotation_tasks=["segmentation"],
            segmentation_types=["vessel", "optic_disc"]
        )
        assert spec.segmentation_types == ["vessel", "optic_disc"]

    def test_lesion_subtypes_with_segmentation_is_valid(self):
        """Test that lesion_subtypes is valid when segmentation is in annotation_tasks."""
        spec = ExportSpec(
            annotation_tasks=["grading", "segmentation"],
            lesion_subtypes=["microaneurysm", "hemorrhage"]
        )
        assert spec.lesion_subtypes == ["microaneurysm", "hemorrhage"]


class TestExportSpecDefaults:
    """Tests for ExportSpec default values."""

    def test_all_defaults(self):
        """Test that all optional fields default correctly."""
        spec = ExportSpec()
        assert spec.dataset_ids is None
        assert spec.dataset_names is None
        assert spec.split_names is None
        assert spec.split_task_type is None
        assert spec.modalities is None
        assert spec.storage_provider == "local"
        assert spec.annotation_tasks is None
        assert spec.disease_types is None
        assert spec.annotation_source == "prefer_consensus"
        assert spec.grading_scale_name is None
        assert spec.include_original_grade is True
        assert spec.include_scaled_grade is True
        assert spec.grade_filter is None
        assert spec.segmentation_types is None
        assert spec.lesion_subtypes is None
        assert spec.require_annotations is None
        assert spec.base_path_for_paths is None

    def test_empty_modalities_list_treated_as_no_filter(self):
        """Test that empty modalities list is accepted (no filter)."""
        spec = ExportSpec(modalities=[])
        assert spec.modalities == []

    def test_empty_dataset_names_list_accepted(self):
        """Test that empty dataset_names list is accepted."""
        spec = ExportSpec(dataset_names=[])
        assert spec.dataset_names == []

    def test_grade_filter_with_valid_disease_keys(self):
        """Test that grade_filter with valid disease keys is accepted."""
        spec = ExportSpec(
            annotation_tasks=["grading"],
            grade_filter={"DR": {"min": 0, "max": 4}, "DME": {"min": 0, "max": 2}},
        )
        assert spec.grade_filter["DR"] == {"min": 0, "max": 4}
        assert spec.grade_filter["DME"] == {"min": 0, "max": 2}

    def test_complex_spec_example(self):
        """Test a complex spec example similar to real-world usage."""
        spec = ExportSpec(
            dataset_names=["EYEPACS"],
            split_names=["train"],
            annotation_tasks=["grading", "segmentation"],
            disease_types=["DR"],
            annotation_source="prefer_consensus",
            grading_scale_name="ETDRS",
            include_original_grade=True,
            include_scaled_grade=True,
            require_annotations=True,
            modalities=["fundus"],
            storage_provider="local"
        )
        assert spec.dataset_names == ["EYEPACS"]
        assert spec.split_names == ["train"]
        assert "grading" in spec.annotation_tasks
        assert "segmentation" in spec.annotation_tasks
        assert spec.disease_types == ["DR"]
        assert spec.annotation_source == "prefer_consensus"
        assert spec.require_annotations is True


class TestRequireAnnotationsMode:
    """Tests for require_annotations_mode field."""

    def test_require_annotations_mode_default(self):
        """Test that require_annotations_mode defaults to 'none'."""
        spec = ExportSpec()
        assert spec.require_annotations_mode == "none"

    def test_require_annotations_mode_valid_values(self):
        """Test that valid require_annotations_mode values are accepted."""
        for mode in ["none", "all", "any"]:
            spec = ExportSpec(require_annotations_mode=mode)
            assert spec.require_annotations_mode == mode

    def test_require_annotations_mode_invalid_raises(self):
        """Test that invalid require_annotations_mode raises ValidationError."""
        with pytest.raises(ValidationError, match="require_annotations_mode"):
            ExportSpec(require_annotations_mode="invalid_mode")

    def test_backward_compat_require_annotations_false(self):
        """Test backward compatibility: require_annotations=False maps to mode='none'."""
        spec = ExportSpec(require_annotations=False)
        assert spec.require_annotations_mode == "none"

    def test_backward_compat_require_annotations_true(self):
        """Test backward compatibility: require_annotations=True maps to mode='all'."""
        spec = ExportSpec(require_annotations=True)
        assert spec.require_annotations_mode == "all"

    def test_require_annotations_mode_takes_precedence(self):
        """Test that require_annotations_mode takes precedence over deprecated require_annotations."""
        # Both provided: mode should win (with warning)
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            spec = ExportSpec(
                require_annotations=True,  # Would map to "all"
                require_annotations_mode="any"  # Should win
            )
            assert spec.require_annotations_mode == "any"
            # Should have deprecation warning
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)


class TestAnnotationOrFilters:
    """Tests for annotation_or_filters field."""

    def test_annotation_or_filters_default(self):
        """Test that annotation_or_filters defaults to None."""
        spec = ExportSpec()
        assert spec.annotation_or_filters is None

    def test_annotation_or_filters_requires_annotation_tasks(self):
        """Test that annotation_or_filters requires annotation_tasks to be set."""
        from chaksudb.export.spec import AnnotationOrFilter
        
        with pytest.raises(ValueError, match="annotation_or_filters requires annotation_tasks"):
            ExportSpec(
                annotation_or_filters=[
                    AnnotationOrFilter(
                        task="segmentation",
                        conditions=[{"segmentation_types": ["vessel"]}]
                    )
                ]
            )

    def test_annotation_or_filters_task_must_be_in_annotation_tasks(self):
        """Test that OR filter tasks must be in annotation_tasks."""
        from chaksudb.export.spec import AnnotationOrFilter
        
        with pytest.raises(ValueError, match="contains tasks not in annotation_tasks"):
            ExportSpec(
                annotation_tasks=["grading"],
                annotation_or_filters=[
                    AnnotationOrFilter(
                        task="segmentation",  # Not in annotation_tasks
                        conditions=[{"segmentation_types": ["vessel"]}]
                    )
                ]
            )

    def test_annotation_or_filters_valid(self):
        """Test that valid annotation_or_filters is accepted."""
        from chaksudb.export.spec import AnnotationOrFilter
        
        spec = ExportSpec(
            annotation_tasks=["segmentation"],
            annotation_or_filters=[
                AnnotationOrFilter(
                    task="segmentation",
                    conditions=[
                        {"segmentation_types": ["lesion"], "lesion_subtypes": ["microaneurysm"]},
                        {"segmentation_types": ["vessel"]},
                    ]
                )
            ]
        )
        assert len(spec.annotation_or_filters) == 1
        assert spec.annotation_or_filters[0].task == "segmentation"
        assert len(spec.annotation_or_filters[0].conditions) == 2

    def test_annotation_or_filters_warns_on_conflicting_flat_filters(self):
        """Test that warning is issued when both OR filters and flat filters are set."""
        from chaksudb.export.spec import AnnotationOrFilter
        import warnings
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            spec = ExportSpec(
                annotation_tasks=["segmentation"],
                segmentation_types=["vessel"],  # Flat filter
                annotation_or_filters=[
                    AnnotationOrFilter(
                        task="segmentation",
                        conditions=[{"segmentation_types": ["lesion"]}]
                    )
                ]
            )
            # Should have warning
            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
            assert "flat filters" in str(w[0].message).lower()

    def test_annotation_or_filters_task_not_in_annotation_tasks(self):
        """Test that OR filter tasks must be present in annotation_tasks."""
        from chaksudb.export.spec import AnnotationOrFilter

        with pytest.raises(ValueError, match="contains tasks not in annotation_tasks"):
            ExportSpec(
                annotation_tasks=["segmentation"],
                annotation_or_filters=[
                    AnnotationOrFilter(
                        task="grading",  # valid task but not in annotation_tasks
                        conditions=[{"some_filter": ["value"]}]
                    )
                ]
            )


class TestPublicExportAPI:
    """Verify simplified public API: ExportSpec and export only."""

    def test_public_api_export_spec_and_export_importable(self):
        """from chaksudb.export import ExportSpec, export is the intended public API."""
        from chaksudb.export import __all__
        assert "ExportSpec" in __all__
        assert "export" in __all__
        assert callable(export)

    def test_export_accepts_spec_returns_none_without_target(self):
        """export(spec) with no parquet_path or torch returns None."""
        spec = ExportSpec()
        result = export(spec)
        assert result is None
