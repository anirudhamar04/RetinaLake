"""Tests for CaptionModule."""

import pytest

from chaksudb.export.modules.caption import CaptionModule
from chaksudb.export.query_builder import QueryBuilder, QueryPlan
from chaksudb.export.spec import ExportSpec


class TestCaptionModule:
    def test_apply_clinical_mode(self):
        module = CaptionModule()
        plan = QueryPlan()
        plan.from_tables.append("images i")
        spec = ExportSpec(caption_mode="clinical")
        module.apply(plan, spec)

        join_text = " ".join(plan.joins)
        assert "clinical_descriptions" in join_text
        select_text = " ".join(plan.select)
        assert "caption_clinical_text" in select_text

    def test_apply_keyword_mode(self):
        module = CaptionModule()
        plan = QueryPlan()
        plan.from_tables.append("images i")
        spec = ExportSpec(caption_mode="keyword")
        module.apply(plan, spec)

        join_text = " ".join(plan.joins)
        assert "keyword_annotations" in join_text
        select_text = " ".join(plan.select)
        assert "caption_keywords" in select_text

    def test_apply_grading_mode(self):
        module = CaptionModule()
        plan = QueryPlan()
        plan.from_tables.append("images i")
        spec = ExportSpec(caption_mode="grading")
        module.apply(plan, spec)

        join_text = " ".join(plan.joins)
        assert "disease_grading" in join_text
        select_text = " ".join(plan.select)
        assert "caption_grade_data" in select_text
        # Should NOT include clinical or keyword
        assert "clinical_descriptions" not in join_text
        assert "keyword_annotations" not in join_text

    def test_apply_classification_mode(self):
        module = CaptionModule()
        plan = QueryPlan()
        plan.from_tables.append("images i")
        spec = ExportSpec(caption_mode="classification")
        module.apply(plan, spec)

        join_text = " ".join(plan.joins)
        assert "classification_annotations" in join_text
        select_text = " ".join(plan.select)
        assert "caption_class_data" in select_text

    def test_apply_synthetic_mode(self):
        module = CaptionModule()
        plan = QueryPlan()
        plan.from_tables.append("images i")
        spec = ExportSpec(caption_mode="synthetic")
        module.apply(plan, spec)

        join_text = " ".join(plan.joins)
        assert "disease_grading" in join_text
        assert "classification_annotations" in join_text
        assert "localization_annotations" in join_text
        assert "segmentation_annotations" in join_text

        select_text = " ".join(plan.select)
        assert "caption_grade_data" in select_text
        assert "caption_class_data" in select_text
        assert "caption_loc_structures" in select_text
        assert "caption_seg_structures" in select_text

        # synthetic should NOT include clinical or keyword
        assert "clinical_descriptions" not in join_text
        assert "keyword_annotations" not in join_text

    def test_apply_all_mode(self):
        module = CaptionModule()
        plan = QueryPlan()
        plan.from_tables.append("images i")
        spec = ExportSpec(caption_mode="all")
        module.apply(plan, spec)

        join_text = " ".join(plan.joins)
        assert "clinical_descriptions" in join_text
        assert "keyword_annotations" in join_text
        assert "disease_grading" in join_text
        assert "classification_annotations" in join_text
        assert "localization_annotations" in join_text
        assert "segmentation_annotations" in join_text

    def test_caption_module_in_query_builder_all(self):
        builder = QueryBuilder()
        spec = ExportSpec(caption_mode="all")
        plan = builder.build_query(spec)
        sql = plan.render_sql()

        assert "clinical_descriptions" in sql
        assert "keyword_annotations" in sql
        assert "disease_grading" in sql
        assert "classification_annotations" in sql
        assert "localization_annotations" in sql
        assert "segmentation_annotations" in sql

    def test_caption_module_in_query_builder_synthetic(self):
        builder = QueryBuilder()
        spec = ExportSpec(caption_mode="synthetic")
        plan = builder.build_query(spec)
        sql = plan.render_sql()

        assert "caption_grade_data" in sql
        assert "caption_class_data" in sql
        assert "caption_loc_structures" in sql
        assert "caption_seg_structures" in sql

    def test_caption_not_included_without_mode(self):
        builder = QueryBuilder()
        spec = ExportSpec()
        plan = builder.build_query(spec)
        sql = plan.render_sql()

        assert "caption_clinical_text" not in sql

    def test_get_output_fields(self):
        module = CaptionModule()
        fields = module.get_output_fields()
        assert "caption_clinical_text" in fields
        assert "caption_keywords" in fields
        assert "caption_grade_data" in fields
        assert "caption_class_data" in fields
        assert "caption_loc_structures" in fields
        assert "caption_seg_structures" in fields
