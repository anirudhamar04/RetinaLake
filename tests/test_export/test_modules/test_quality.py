"""
Tests for QualityModule.
"""

import pytest

from chaksudb.export.modules.quality import QualityModule
from chaksudb.export.query_builder import QueryPlan
from chaksudb.export.spec import ExportSpec


class TestQualityModule:
    """Tests for QualityModule."""

    def test_apply_adds_join(self):
        """Test that QualityModule adds quality JOIN."""
        module = QualityModule()
        plan = QueryPlan()
        plan.from_tables.append("images i")
        spec = ExportSpec(annotation_tasks=["quality"])
        module.apply(plan, spec)
        # Should add JOIN for quality_annotations
        assert len(plan.joins) > 0
        join_sql = " ".join(plan.joins).lower()
        assert "quality" in join_sql

    def test_apply_adds_quality_fields(self):
        """Test that QualityModule adds quality fields to SELECT."""
        module = QualityModule()
        plan = QueryPlan()
        plan.from_tables.append("images i")
        spec = ExportSpec(annotation_tasks=["quality"])
        module.apply(plan, spec)
        # Should add quality-related fields
        select_sql = " ".join(plan.select).lower()
        assert "quality" in select_sql

    def test_get_output_fields(self):
        """Test that get_output_fields returns correct field names."""
        module = QualityModule()
        fields = module.get_output_fields()
        # Should include quality-related fields
        assert len(fields) > 0
        quality_fields = [f for f in fields if "quality" in f.lower()]
        assert len(quality_fields) > 0
