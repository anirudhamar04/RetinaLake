"""
Tests for SplitModule.
"""

import pytest

from chaksudb.export.modules.split import SplitModule
from chaksudb.export.query_builder import QueryPlan
from chaksudb.export.spec import ExportSpec


class TestSplitModule:
    """Tests for SplitModule."""

    def test_apply_adds_joins(self):
        """Test that SplitModule adds split-related JOINs."""
        module = SplitModule()
        plan = QueryPlan()
        plan.from_tables.append("images i")
        spec = ExportSpec(split_names=["train"])
        module.apply(plan, spec)
        # Should add JOINs for image_splits and dataset_splits
        assert len(plan.joins) > 0
        join_sql = " ".join(plan.joins).lower()
        assert "split" in join_sql

    def test_apply_adds_split_fields(self):
        """Test that SplitModule adds split_name and task_type to SELECT."""
        module = SplitModule()
        plan = QueryPlan()
        plan.from_tables.append("images i")
        spec = ExportSpec(split_names=["train"])
        module.apply(plan, spec)
        # Should add split_name and task_type
        select_sql = " ".join(plan.select).lower()
        assert "split_name" in select_sql
        assert "task_type" in select_sql

    def test_apply_with_split_names_filter(self):
        """Test that SplitModule filters by split_names inside the LATERAL join (not a
        post-join WHERE), so the filter selects the matching row instead of excluding a
        fanned-out duplicate."""
        module = SplitModule()
        plan = QueryPlan()
        plan.from_tables.append("images i")
        spec = ExportSpec(split_names=["train", "test"])
        module.apply(plan, spec)
        join_sql = " ".join(plan.joins).lower()
        assert "split_name" in join_sql

    def test_apply_with_split_task_type_filter(self):
        """Test that SplitModule filters by split_task_type inside the LATERAL join."""
        module = SplitModule()
        plan = QueryPlan()
        plan.from_tables.append("images i")
        spec = ExportSpec(split_task_type="classification")
        module.apply(plan, spec)
        join_sql = " ".join(plan.joins).lower()
        assert "task_type" in join_sql

    def test_apply_returns_at_most_one_row_per_image(self):
        """Regression test: the JOIN must be a LATERAL subquery capped with LIMIT 1, so an
        image with multiple overlapping image_splits rows (e.g. stale explicit + user_defined
        rows) can't fan out into duplicate export rows."""
        module = SplitModule()
        plan = QueryPlan()
        plan.from_tables.append("images i")
        spec = ExportSpec(split_names=["train"])
        module.apply(plan, spec)
        join_sql = " ".join(plan.joins).lower()
        assert "lateral" in join_sql
        assert "limit 1" in join_sql

    def test_apply_with_require_annotations_inner_join(self):
        """Test that SplitModule uses INNER JOIN when require_annotations is True."""
        module = SplitModule()
        plan = QueryPlan()
        plan.from_tables.append("images i")
        spec = ExportSpec(split_names=["train"], require_annotations=True)
        module.apply(plan, spec)
        # Should use INNER JOIN
        join_sql = " ".join(plan.joins).lower()
        assert "inner join" in join_sql

    def test_apply_with_require_annotations_false_left_join(self):
        """Test that SplitModule uses LEFT JOIN when require_annotations is False."""
        module = SplitModule()
        plan = QueryPlan()
        plan.from_tables.append("images i")
        spec = ExportSpec(split_names=["train"], require_annotations=False)
        module.apply(plan, spec)
        # Should use LEFT JOIN
        join_sql = " ".join(plan.joins).lower()
        assert "left join" in join_sql

    def test_get_output_fields(self):
        """Test that get_output_fields returns correct field names."""
        module = SplitModule()
        fields = module.get_output_fields()
        assert "split_name" in fields
        assert "task_type" in fields
        assert len(fields) == 2
