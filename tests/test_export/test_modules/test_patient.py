"""Tests for PatientModule."""

import pytest

from chaksudb.export.modules.patient import PatientModule
from chaksudb.export.query_builder import QueryPlan
from chaksudb.export.spec import ExportSpec


class TestPatientModule:
    """Tests for PatientModule."""

    def test_apply_adds_joins(self):
        """Test that PatientModule adds patient_images and patients JOINs."""
        module = PatientModule()
        plan = QueryPlan()
        plan.from_tables.append("images i")
        spec = ExportSpec(include_patient_data=True)
        module.apply(plan, spec)

        join_text = " ".join(plan.joins)
        assert "patient_images" in join_text
        assert "patients" in join_text
        assert len(plan.joins) == 2

    def test_apply_adds_select_fields(self):
        """Test that PatientModule adds demographic fields to SELECT."""
        module = PatientModule()
        plan = QueryPlan()
        plan.from_tables.append("images i")
        spec = ExportSpec(include_patient_data=True)
        module.apply(plan, spec)

        select_text = " ".join(plan.select)
        assert "p.patient_id" in select_text
        assert "p.age" in select_text
        assert "p.sex" in select_text
        assert "p.ethnicity" in select_text
        assert "p.comorbidities" in select_text

    def test_get_output_fields(self):
        """Test output field names."""
        module = PatientModule()
        fields = module.get_output_fields()
        assert "patient_id" in fields
        assert "age" in fields
        assert "sex" in fields
        assert "ethnicity" in fields

    def test_patient_module_registered_in_query_builder(self):
        """Test that include_patient_data triggers PatientModule in query builder."""
        from chaksudb.export.query_builder import QueryBuilder

        builder = QueryBuilder()
        spec = ExportSpec(include_patient_data=True)
        plan = builder.build_query(spec)
        sql = plan.render_sql()

        assert "patient_images" in sql
        assert "patients" in sql

    def test_patient_module_not_included_by_default(self):
        """Test that PatientModule is not included when include_patient_data=False."""
        from chaksudb.export.query_builder import QueryBuilder

        builder = QueryBuilder()
        spec = ExportSpec()
        plan = builder.build_query(spec)
        sql = plan.render_sql()

        assert "patient_images" not in sql
        assert "patients p" not in sql
