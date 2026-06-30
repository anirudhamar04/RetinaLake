"""
Tests for SegmentationModule.
"""

import pytest

from chaksudb.export.modules.segmentation import SegmentationModule
from chaksudb.export.query_builder import QueryPlan
from chaksudb.export.spec import ExportSpec


class TestSegmentationModule:
    """Tests for SegmentationModule."""

    def test_apply_adds_join(self):
        """Test that SegmentationModule adds segmentation JOIN."""
        module = SegmentationModule()
        plan = QueryPlan()
        plan.from_tables.append("images i")
        spec = ExportSpec(annotation_tasks=["segmentation"])
        module.apply(plan, spec)
        # Should add JOIN for segmentation_annotations
        assert len(plan.joins) > 0
        join_sql = " ".join(plan.joins).lower()
        assert "segmentation" in join_sql

    def test_apply_adds_segmentation_aggregation(self):
        """Test that SegmentationModule adds segmentation_masks aggregation."""
        module = SegmentationModule()
        plan = QueryPlan()
        plan.from_tables.append("images i")
        spec = ExportSpec(annotation_tasks=["segmentation"])
        module.apply(plan, spec)
        # Should add jsonb_agg for segmentation_masks
        select_sql = " ".join(plan.select).lower()
        assert "segmentation" in select_sql or "jsonb_agg" in select_sql

    def test_apply_adds_group_by(self):
        """Test that SegmentationModule adds GROUP BY image_id."""
        module = SegmentationModule()
        plan = QueryPlan()
        plan.from_tables.append("images i")
        spec = ExportSpec(annotation_tasks=["segmentation"])
        module.apply(plan, spec)
        # Should add GROUP BY for aggregation
        assert "i.image_id" in plan.group_bys

    def test_get_output_fields(self):
        """Test that get_output_fields returns correct field names."""
        module = SegmentationModule()
        fields = module.get_output_fields()
        assert "segmentation_masks" in fields
        assert len(fields) == 1

    def test_apply_includes_annotation_type_in_aggregation(self):
        """Test that segmentation_masks aggregation includes annotation_type from annotation_type table."""
        module = SegmentationModule()
        plan = QueryPlan()
        plan.from_tables.append("images i")
        spec = ExportSpec(annotation_tasks=["segmentation"])
        module.apply(plan, spec)
        select_sql = " ".join(plan.select)
        assert "annotation_type" in select_sql
        assert "at.annotation_type" in select_sql

    def test_apply_adds_type_subtype_filters_when_specified(self):
        """Test that segmentation_types and lesion_subtypes add WHERE and params."""
        module = SegmentationModule()
        plan = QueryPlan()
        plan.from_tables.append("images i")
        spec = ExportSpec(
            annotation_tasks=["segmentation"],
            segmentation_types=["vessel", "optic_disc"],
            lesion_subtypes=["microaneurysm"],
        )
        module.apply(plan, spec)
        where_sql = " ".join(plan.wheres).lower()
        assert "segmentation_id is null" in where_sql
        assert "annotation_type" in where_sql or "lesion_subtype" in where_sql
        assert len(plan.params) >= 2  # segmentation_types and lesion_subtypes
