"""Tests for export presets."""

import pytest

from chaksudb.export import presets
from chaksudb.export.spec import ExportSpec


class TestPresets:
    """Smoke tests: each preset returns a valid ExportSpec."""

    def test_dr_classification(self):
        spec = presets.dr_classification()
        assert isinstance(spec, ExportSpec)
        assert "grading" in spec.annotation_tasks
        assert "DR" in spec.disease_types

    def test_dr_classification_with_dataset(self):
        spec = presets.dr_classification(datasets=["EYEPACS"], split="train")
        assert spec.dataset_names == ["EYEPACS"]
        assert spec.split_names == ["train"]

    def test_glaucoma_detection(self):
        spec = presets.glaucoma_detection()
        assert isinstance(spec, ExportSpec)
        assert "classification" in spec.annotation_tasks
        assert "glaucoma" in spec.classification_class_names

    def test_lesion_segmentation(self):
        spec = presets.lesion_segmentation()
        assert isinstance(spec, ExportSpec)
        assert "segmentation" in spec.annotation_tasks
        assert "lesion" in spec.segmentation_types

    def test_optic_disc_segmentation(self):
        spec = presets.optic_disc_segmentation()
        assert isinstance(spec, ExportSpec)
        assert "optic_disc" in spec.segmentation_types

    def test_lesion_detection_coco(self):
        spec = presets.lesion_detection_coco()
        assert isinstance(spec, ExportSpec)
        assert spec.detection_format == "coco"
        assert "localization" in spec.annotation_tasks

    def test_fundus_captioning(self):
        spec = presets.fundus_captioning()
        assert isinstance(spec, ExportSpec)
        assert spec.caption_mode == "all"

    def test_quality_assessment(self):
        spec = presets.quality_assessment()
        assert isinstance(spec, ExportSpec)
        assert "quality" in spec.annotation_tasks

    def test_multi_label_disease(self):
        spec = presets.multi_label_disease()
        assert isinstance(spec, ExportSpec)
        assert "classification" in spec.annotation_tasks

    def test_landmark_detection(self):
        spec = presets.landmark_detection()
        assert isinstance(spec, ExportSpec)
        assert "localization" in spec.annotation_tasks
        assert "keypoint" in spec.localization_types

    def test_multi_task(self):
        spec = presets.multi_task()
        assert isinstance(spec, ExportSpec)
        assert len(spec.annotation_tasks) == 3

    def test_all_presets_produce_valid_sql(self):
        """Every preset should produce renderable SQL via QueryBuilder."""
        from chaksudb.export.query_builder import QueryBuilder

        builder = QueryBuilder()
        preset_fns = [
            presets.dr_classification,
            presets.glaucoma_detection,
            presets.lesion_segmentation,
            presets.optic_disc_segmentation,
            presets.lesion_detection_coco,
            presets.fundus_captioning,
            presets.quality_assessment,
            presets.multi_label_disease,
            presets.landmark_detection,
            presets.multi_task,
        ]
        for fn in preset_fns:
            spec = fn()
            plan = builder.build_query(spec)
            sql = plan.render_sql()
            assert "SELECT" in sql, f"{fn.__name__} produced invalid SQL"
            assert "ORDER BY" in sql, f"{fn.__name__} missing ORDER BY"
