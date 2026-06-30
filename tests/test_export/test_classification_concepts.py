"""Tests for the concept-centric classification export interface.

Covers the contract changes: classification_class_names is optional, concepts produce
per-concept presence columns, and the positive-for filter is applied across storage shapes.
"""

import pytest

from chaksudb.export.modules.classification import ClassificationModule
from chaksudb.export.query_builder import QueryPlan
from chaksudb.export.spec import ExportSpec


def _plan_for(spec: ExportSpec) -> QueryPlan:
    plan = QueryPlan()
    plan.from_tables.append("images i")
    ClassificationModule().apply(plan, spec)
    return plan


def test_classification_no_longer_requires_class_names():
    # Previously raised; now valid (export auto-discovers / uses concepts).
    spec = ExportSpec(annotation_tasks=["classification"])
    assert spec.classification_class_names is None


def test_concept_presence_column_emitted():
    spec = ExportSpec(
        annotation_tasks=["classification"],
        classification_concepts=["glaucoma"],
    )
    plan = _plan_for(spec)
    selects = " ".join(plan.select)
    assert "glaucoma_present" in selects
    # presence unifies binary positives and multi_class winners
    assert "multi_class" in selects and "class_index = 1" in selects


def test_positive_for_filter_is_cross_shape():
    spec = ExportSpec(
        annotation_tasks=["classification"],
        classification_positive_for=["glaucoma"],
    )
    plan = _plan_for(spec)
    where = " ".join(plan.wheres)
    assert "EXISTS" in where
    assert "classification_annotations" in where
    assert "task_type = 'multi_class'" in where or "class_index = 1" in where


def test_concept_fields_require_classification_task():
    with pytest.raises(ValueError):
        ExportSpec(classification_concepts=["glaucoma"])  # no annotation_tasks
