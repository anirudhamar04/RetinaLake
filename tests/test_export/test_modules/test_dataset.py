"""
Tests for DatasetModule.
"""

import pytest
from uuid import UUID

from chaksudb.export.modules.dataset import DatasetModule
from chaksudb.export.query_builder import QueryPlan
from chaksudb.export.spec import ExportSpec


class TestDatasetModule:
    """Tests for DatasetModule."""

    def test_apply_adds_join(self):
        """Test that DatasetModule adds datasets JOIN."""
        module = DatasetModule()
        plan = QueryPlan()
        plan.from_tables.append("images i")  # ImageModule should have added this
        spec = ExportSpec()
        module.apply(plan, spec)
        assert len(plan.joins) == 1
        assert "datasets" in plan.joins[0].lower()
        assert "d.dataset_id" in plan.joins[0]

    def test_apply_adds_dataset_name(self):
        """Test that DatasetModule adds dataset_name to SELECT."""
        module = DatasetModule()
        plan = QueryPlan()
        plan.from_tables.append("images i")
        spec = ExportSpec()
        module.apply(plan, spec)
        assert "d.dataset_name" in plan.select

    def test_apply_with_dataset_ids_filter(self):
        """Test that DatasetModule adds dataset_ids filter when specified."""
        module = DatasetModule()
        plan = QueryPlan()
        plan.from_tables.append("images i")
        dataset_ids = [
            UUID("11111111-1111-1111-1111-111111111111"),
            UUID("22222222-2222-2222-2222-222222222222"),
        ]
        spec = ExportSpec(dataset_ids=dataset_ids)
        module.apply(plan, spec)
        # Should add WHERE condition for dataset_ids
        dataset_where = [w for w in plan.wheres if "dataset_id" in w.lower()]
        assert len(dataset_where) > 0

    def test_apply_with_dataset_names_filter(self):
        """Test that DatasetModule adds dataset_names filter when specified."""
        module = DatasetModule()
        plan = QueryPlan()
        plan.from_tables.append("images i")
        spec = ExportSpec(dataset_names=["EYEPACS", "MESSIDOR"])
        module.apply(plan, spec)
        # Should add WHERE condition for dataset_names
        dataset_where = [w for w in plan.wheres if "dataset_name" in w.lower()]
        assert len(dataset_where) > 0

    def test_get_output_fields(self):
        """Test that get_output_fields returns correct field names."""
        module = DatasetModule()
        fields = module.get_output_fields()
        assert "dataset_name" in fields
        assert len(fields) == 1
