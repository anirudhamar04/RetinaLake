"""
Tests for KeywordsModule.
"""

import pytest

from chaksudb.export.modules.keywords import KeywordsModule
from chaksudb.export.query_builder import QueryPlan
from chaksudb.export.spec import ExportSpec


class TestKeywordsModule:
    """Tests for KeywordsModule."""

    def test_apply_adds_joins(self):
        """Test that KeywordsModule adds keyword-related JOINs."""
        module = KeywordsModule()
        plan = QueryPlan()
        plan.from_tables.append("images i")
        spec = ExportSpec(annotation_tasks=["keyword"])
        module.apply(plan, spec)
        # Should add JOINs for keyword_annotations and keyword_vocabulary
        assert len(plan.joins) > 0
        join_sql = " ".join(plan.joins).lower()
        assert "keyword" in join_sql

    def test_apply_adds_keywords_aggregation(self):
        """Test that KeywordsModule adds keywords array aggregation."""
        module = KeywordsModule()
        plan = QueryPlan()
        plan.from_tables.append("images i")
        spec = ExportSpec(annotation_tasks=["keyword"])
        module.apply(plan, spec)
        # Should add array_agg for keywords
        select_sql = " ".join(plan.select).lower()
        assert "keyword" in select_sql or "array_agg" in select_sql

    def test_apply_adds_group_by(self):
        """Test that KeywordsModule adds GROUP BY image_id."""
        module = KeywordsModule()
        plan = QueryPlan()
        plan.from_tables.append("images i")
        spec = ExportSpec(annotation_tasks=["keyword"])
        module.apply(plan, spec)
        # Should add GROUP BY for aggregation
        assert "i.image_id" in plan.group_bys

    def test_get_output_fields(self):
        """Test that get_output_fields returns correct field names."""
        module = KeywordsModule()
        fields = module.get_output_fields()
        assert "keywords" in fields
        assert len(fields) == 1
