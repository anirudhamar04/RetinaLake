"""
Tests for ClassificationModule.

Verifies that the module produces the correct SQL using scalar columns
(class_index, class_label, sub_key) instead of JSONB extraction.
"""

import pytest

from chaksudb.export.modules.classification import ClassificationModule
from chaksudb.export.query_builder import QueryPlan
from chaksudb.export.spec import ExportSpec


class TestClassificationModule:
    """Tests for ClassificationModule."""

    def test_apply_adds_join(self):
        """Test that ClassificationModule adds classification JOIN."""
        module = ClassificationModule()
        plan = QueryPlan()
        plan.from_tables.append("images i")
        spec = ExportSpec(
            annotation_tasks=["classification"],
            classification_class_names=["glaucoma"],
        )
        module.apply(plan, spec)
        assert len(plan.joins) > 0
        join_sql = " ".join(plan.joins).lower()
        assert "classification_annotations" in join_sql

    def test_apply_adds_classification_fields(self):
        """Test that ClassificationModule adds SELECT columns."""
        module = ClassificationModule()
        plan = QueryPlan()
        plan.from_tables.append("images i")
        spec = ExportSpec(
            annotation_tasks=["classification"],
            classification_class_names=["glaucoma"],
        )
        module.apply(plan, spec)
        select_sql = " ".join(plan.select).lower()
        assert "glaucoma_label" in select_sql
        assert "glaucoma_class_label" in select_sql

    def test_binary_uses_scalar_columns(self):
        """Test that binary classification uses c.class_index, not JSONB."""
        module = ClassificationModule()
        plan = QueryPlan()
        plan.from_tables.append("images i")
        spec = ExportSpec(
            annotation_tasks=["classification"],
            classification_class_names=["glaucoma"],
            classification_task_types={"glaucoma": "binary"},
        )
        module.apply(plan, spec)
        select_sql = " ".join(plan.select)
        assert "c.class_index" in select_sql
        assert "c.class_label" in select_sql
        # Should NOT contain JSONB extraction
        assert "class_value" not in select_sql

    def test_multi_class_uses_scalar_columns(self):
        """Test that multi-class classification uses c.class_index, not JSONB."""
        module = ClassificationModule()
        plan = QueryPlan()
        plan.from_tables.append("images i")
        spec = ExportSpec(
            annotation_tasks=["classification"],
            classification_class_names=["disease_type"],
            classification_task_types={"disease_type": "multi_class"},
        )
        module.apply(plan, spec)
        select_sql = " ".join(plan.select)
        assert "c.class_index" in select_sql
        assert "c.class_label" in select_sql
        assert "class_value" not in select_sql

    def test_multi_label_with_keys_uses_sub_key(self):
        """Test that multi-label with keys uses c.sub_key, not JSONB extraction."""
        module = ClassificationModule()
        plan = QueryPlan()
        plan.from_tables.append("images i")
        spec = ExportSpec(
            annotation_tasks=["classification"],
            classification_class_names=["disease_indicators"],
            classification_task_types={"disease_indicators": "multi_label"},
            multi_label_keys={"disease_indicators": ["diabetes", "glaucoma", "amd"]},
        )
        module.apply(plan, spec)
        select_sql = " ".join(plan.select)
        assert "c.sub_key" in select_sql
        assert "c.class_index" in select_sql
        assert "disease_indicators_diabetes" in select_sql
        assert "disease_indicators_glaucoma" in select_sql
        assert "disease_indicators_amd" in select_sql

    def test_multi_label_without_keys_fallback(self):
        """Test multi-label without keys falls back to JSON aggregation."""
        module = ClassificationModule()
        plan = QueryPlan()
        plan.from_tables.append("images i")
        spec = ExportSpec(
            annotation_tasks=["classification"],
            classification_class_names=["disease_indicators"],
            classification_task_types={"disease_indicators": "multi_label"},
        )
        module.apply(plan, spec)
        select_sql = " ".join(plan.select)
        assert "disease_indicators_labels" in select_sql

    def test_prefer_consensus_subquery_includes_sub_key(self):
        """Test prefer_consensus partitions by sub_key."""
        module = ClassificationModule()
        plan = QueryPlan()
        plan.from_tables.append("images i")
        spec = ExportSpec(
            annotation_tasks=["classification"],
            classification_class_names=["glaucoma"],
            annotation_source="prefer_consensus",
        )
        module.apply(plan, spec)
        join_sql = " ".join(plan.joins)
        assert "c_inner.sub_key" in join_sql

    def test_get_output_fields_returns_empty(self):
        """Test that get_output_fields returns empty (fields are dynamic)."""
        module = ClassificationModule()
        assert module.get_output_fields() == []

    def test_get_primary_id_column(self):
        """Test primary ID column for HAVING clause."""
        module = ClassificationModule()
        assert module.get_primary_id_column() == "c.classification_id"

    def test_mixed_task_types(self):
        """Test multiple class_names with different task types."""
        module = ClassificationModule()
        plan = QueryPlan()
        plan.from_tables.append("images i")
        spec = ExportSpec(
            annotation_tasks=["classification"],
            classification_class_names=["glaucoma", "disease_type", "disease_indicators"],
            classification_task_types={
                "glaucoma": "binary",
                "disease_type": "multi_class",
                "disease_indicators": "multi_label",
            },
            multi_label_keys={"disease_indicators": ["diabetes", "amd"]},
        )
        module.apply(plan, spec)
        select_sql = " ".join(plan.select)
        assert "glaucoma_label" in select_sql
        assert "disease_type_label" in select_sql
        assert "disease_indicators_diabetes" in select_sql
        assert "disease_indicators_amd" in select_sql
