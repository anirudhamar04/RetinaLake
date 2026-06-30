"""
Tests for ClinicalModule.
"""

import pytest

from chaksudb.export.modules.clinical import ClinicalModule
from chaksudb.export.query_builder import QueryPlan
from chaksudb.export.spec import ExportSpec


class TestClinicalModule:
    """Tests for ClinicalModule."""

    def test_apply_adds_join(self):
        """Test that ClinicalModule adds clinical descriptions JOIN."""
        module = ClinicalModule()
        plan = QueryPlan()
        plan.from_tables.append("images i")
        spec = ExportSpec(annotation_tasks=["description"])
        module.apply(plan, spec)
        # Should add JOIN for clinical_descriptions
        assert len(plan.joins) > 0
        join_sql = " ".join(plan.joins).lower()
        assert "clinical" in join_sql or "description" in join_sql

    def test_apply_adds_clinical_fields(self):
        """Test that ClinicalModule adds clinical fields to SELECT."""
        module = ClinicalModule()
        plan = QueryPlan()
        plan.from_tables.append("images i")
        spec = ExportSpec(annotation_tasks=["description"])
        module.apply(plan, spec)
        # Should add clinical-related fields
        select_sql = " ".join(plan.select).lower()
        assert "clinical" in select_sql or "description" in select_sql

    def test_get_output_fields(self):
        """Test that get_output_fields returns correct field names."""
        module = ClinicalModule()
        fields = module.get_output_fields()
        # Should include clinical-related fields
        assert len(fields) > 0
        clinical_fields = [f for f in fields if "clinical" in f.lower() or "description" in f.lower()]
        assert len(clinical_fields) > 0
