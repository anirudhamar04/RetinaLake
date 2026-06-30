"""
Tests for ImageModule.
"""

import pytest

from chaksudb.export.modules.image import ImageModule
from chaksudb.export.query_builder import QueryPlan
from chaksudb.export.spec import ExportSpec


class TestImageModule:
    """Tests for ImageModule."""

    def test_apply_adds_from_table(self):
        """Test that ImageModule adds images table to FROM."""
        module = ImageModule()
        plan = QueryPlan()
        spec = ExportSpec()
        module.apply(plan, spec)
        assert len(plan.from_tables) == 1
        assert "images i" in plan.from_tables[0]

    def test_apply_adds_core_fields(self):
        """Test that ImageModule adds core image fields to SELECT."""
        module = ImageModule()
        plan = QueryPlan()
        spec = ExportSpec()
        module.apply(plan, spec)
        assert "i.image_id" in plan.select
        assert "i.file_path" in plan.select
        assert "i.storage_provider" in plan.select
        assert "i.object_key" in plan.select
        assert "i.modality" in plan.select
        assert "i.eye_laterality" in plan.select

    def test_apply_with_base_path_for_paths(self):
        """Test that ImageModule transforms file_path when base_path_for_paths is set."""
        module = ImageModule()
        plan = QueryPlan()
        spec = ExportSpec(base_path_for_paths="/data/images")
        module.apply(plan, spec)
        # Should use CONCAT for path transformation
        file_path_select = [s for s in plan.select if "file_path" in s][0]
        assert "CONCAT" in file_path_select or "CASE" in file_path_select
        assert "base_path" in str(plan.params)

    def test_apply_with_modalities_filter(self):
        """Test that ImageModule adds modality filter when modalities are specified."""
        module = ImageModule()
        plan = QueryPlan()
        spec = ExportSpec(modalities=["fundus", "oct"])
        module.apply(plan, spec)
        # Should add WHERE condition for modalities
        modality_where = [w for w in plan.wheres if "modality" in w.lower()]
        assert len(modality_where) > 0

    def test_apply_with_storage_provider_filter(self):
        """Test that ImageModule adds storage_provider filter when specified."""
        module = ImageModule()
        plan = QueryPlan()
        spec = ExportSpec(storage_provider="local")
        module.apply(plan, spec)
        # Should add WHERE condition for storage_provider
        provider_where = [w for w in plan.wheres if "storage_provider" in w.lower()]
        assert len(provider_where) > 0

    def test_get_output_fields(self):
        """Test that get_output_fields returns correct field names."""
        module = ImageModule()
        fields = module.get_output_fields()
        assert "image_id" in fields
        assert "file_path" in fields
        assert "storage_provider" in fields
        assert "object_key" in fields
        assert "modality" in fields
        assert "eye_laterality" in fields
        assert "resolution_width" in fields
        assert "resolution_height" in fields
