"""
Tests for LocalizationModule.
"""

import pytest

from chaksudb.export.modules.localization import LocalizationModule
from chaksudb.export.query_builder import QueryPlan
from chaksudb.export.spec import ExportSpec


class TestLocalizationModule:
    """Tests for LocalizationModule."""

    def test_apply_adds_join(self):
        """Test that LocalizationModule adds localization JOIN."""
        module = LocalizationModule()
        plan = QueryPlan()
        plan.from_tables.append("images i")
        spec = ExportSpec(annotation_tasks=["localization"])
        module.apply(plan, spec)
        # Should add JOIN for localization_annotations
        assert len(plan.joins) > 0
        join_sql = " ".join(plan.joins).lower()
        assert "localization" in join_sql

    def test_apply_adds_localization_aggregation(self):
        """Test that LocalizationModule adds localization_annotations aggregation."""
        module = LocalizationModule()
        plan = QueryPlan()
        plan.from_tables.append("images i")
        spec = ExportSpec(annotation_tasks=["localization"])
        module.apply(plan, spec)
        # Should add jsonb_agg for localization_annotations
        select_sql = " ".join(plan.select).lower()
        assert "localization" in select_sql or "jsonb_agg" in select_sql

    def test_apply_adds_group_by(self):
        """Test that LocalizationModule adds GROUP BY image_id."""
        module = LocalizationModule()
        plan = QueryPlan()
        plan.from_tables.append("images i")
        spec = ExportSpec(annotation_tasks=["localization"])
        module.apply(plan, spec)
        # Should add GROUP BY for aggregation
        assert "i.image_id" in plan.group_bys

    def test_get_output_fields(self):
        """Test that get_output_fields returns correct field names."""
        module = LocalizationModule()
        fields = module.get_output_fields()
        assert "localization_annotations" in fields
        assert len(fields) == 1
