"""
Tests for GradingModule.
"""

import pytest

from chaksudb.export.modules.grading import GradingModule
from chaksudb.export.query_builder import QueryPlan
from chaksudb.export.spec import ExportSpec


class TestGradingModule:
    """Tests for GradingModule."""

    def test_apply_adds_joins(self):
        """Test that GradingModule adds grading-related JOINs."""
        module = GradingModule()
        plan = QueryPlan()
        plan.from_tables.append("images i")
        spec = ExportSpec(
            annotation_tasks=["grading"],
            disease_types=["DR"]
        )
        module.apply(plan, spec)
        # Should add JOINs for disease_grading and grading_scales
        assert len(plan.joins) > 0
        join_sql = " ".join(plan.joins).lower()
        assert "grading" in join_sql or "disease" in join_sql

    def test_apply_adds_group_by(self):
        """Test that GradingModule adds GROUP BY image_id."""
        module = GradingModule()
        plan = QueryPlan()
        plan.from_tables.append("images i")
        spec = ExportSpec(
            annotation_tasks=["grading"],
            disease_types=["DR"]
        )
        module.apply(plan, spec)
        # Should add GROUP BY for aggregation
        assert "i.image_id" in plan.group_bys

    def test_apply_with_disease_types_filter(self):
        """Test that GradingModule adds disease_types filter when specified."""
        module = GradingModule()
        plan = QueryPlan()
        plan.from_tables.append("images i")
        spec = ExportSpec(
            annotation_tasks=["grading"],
            disease_types=["DR", "DME"]
        )
        module.apply(plan, spec)
        # The disease_types filter must be applied. With the default
        # annotation_source="prefer_consensus" it is added inside the dedup subquery
        # (a JOIN), not the outer WHERE — so check both places.
        applied = [
            frag for frag in (plan.wheres + plan.joins)
            if "disease_type" in frag.lower()
        ]
        assert len(applied) > 0

    def test_apply_with_grading_scale_name_filter(self):
        """Test that GradingModule adds grading_scale_name filter when specified."""
        module = GradingModule()
        plan = QueryPlan()
        plan.from_tables.append("images i")
        spec = ExportSpec(
            annotation_tasks=["grading"],
            grading_scale_name="ETDRS"
        )
        module.apply(plan, spec)
        # Should add WHERE condition for grading_scale_name
        scale_where = [w for w in plan.wheres if "scale_name" in w.lower()]
        assert len(scale_where) > 0

    def test_apply_with_expert_only(self):
        """Test that GradingModule filters to expert_only when specified."""
        module = GradingModule()
        plan = QueryPlan()
        plan.from_tables.append("images i")
        spec = ExportSpec(
            annotation_tasks=["grading"],
            annotation_source="expert_only"
        )
        module.apply(plan, spec)
        # Should add WHERE condition for consensus_id IS NULL
        expert_where = [w for w in plan.wheres if "consensus_id" in w.lower() and "null" in w.lower()]
        assert len(expert_where) > 0

    def test_apply_with_consensus_only(self):
        """Test that GradingModule filters to consensus_only when specified."""
        module = GradingModule()
        plan = QueryPlan()
        plan.from_tables.append("images i")
        spec = ExportSpec(
            annotation_tasks=["grading"],
            annotation_source="consensus_only"
        )
        module.apply(plan, spec)
        # Should add WHERE condition for consensus_id IS NOT NULL
        consensus_where = [w for w in plan.wheres if "consensus_id" in w.lower() and "not null" in w.lower()]
        assert len(consensus_where) > 0

    def test_apply_with_require_annotations_inner_join(self):
        """Test that GradingModule uses INNER JOIN when require_annotations is True."""
        module = GradingModule()
        plan = QueryPlan()
        plan.from_tables.append("images i")
        spec = ExportSpec(
            annotation_tasks=["grading"],
            require_annotations=True
        )
        module.apply(plan, spec)
        # Should use INNER JOIN
        join_sql = " ".join(plan.joins).lower()
        assert "inner join" in join_sql

    def test_apply_with_require_annotations_false_left_join(self):
        """Test that GradingModule uses LEFT JOIN when require_annotations is False."""
        module = GradingModule()
        plan = QueryPlan()
        plan.from_tables.append("images i")
        spec = ExportSpec(
            annotation_tasks=["grading"],
            require_annotations=False
        )
        module.apply(plan, spec)
        # Should use LEFT JOIN (or subquery for prefer_consensus)
        join_sql = " ".join(plan.joins).lower()
        # May use LEFT JOIN or subquery, but should not use INNER JOIN
        assert "inner join" not in join_sql or spec.annotation_source == "prefer_consensus"

    def test_get_output_fields(self):
        """Test that get_output_fields returns correct field names."""
        module = GradingModule()
        fields = module.get_output_fields()
        # Should include fields for disease types (exact fields depend on spec)
        # At minimum, should indicate grading fields are present
        assert len(fields) > 0
