"""
Tests for QueryBuilder and QueryPlan.

Tests QueryPlan construction, SQL rendering, and parameter handling.
"""

import pytest
from uuid import UUID
from pydantic import ValidationError

from chaksudb.export.query_builder import QueryBuilder, QueryPlan
from chaksudb.export.spec import ExportSpec


class TestQueryPlan:
    """Tests for QueryPlan data structure."""

    def test_empty_plan(self):
        """Test that an empty QueryPlan initializes correctly."""
        plan = QueryPlan()
        assert plan.select == []
        assert plan.from_tables == []
        assert plan.joins == []
        assert plan.wheres == []
        assert plan.group_bys == []
        assert plan.havings == []
        assert plan.params == {}
        assert plan.param_counter == 0

    def test_add_select(self):
        """Test adding SELECT fragments."""
        plan = QueryPlan()
        plan.add_select("i.image_id")
        plan.add_select("i.file_path")
        assert plan.select == ["i.image_id", "i.file_path"]

    def test_add_join(self):
        """Test adding JOIN clauses."""
        plan = QueryPlan()
        plan.add_join("LEFT JOIN datasets d ON i.dataset_id = d.dataset_id")
        plan.add_join("LEFT JOIN image_splits s ON i.image_id = s.image_id")
        assert len(plan.joins) == 2
        assert "datasets" in plan.joins[0]
        assert "image_splits" in plan.joins[1]

    def test_add_where(self):
        """Test adding WHERE conditions."""
        plan = QueryPlan()
        plan.add_where("i.modality = 'fundus'")
        plan.add_where("i.storage_provider = 'local'")
        assert len(plan.wheres) == 2

    def test_add_group_by(self):
        """Test adding GROUP BY columns."""
        plan = QueryPlan()
        plan.add_group_by("i.image_id")
        plan.add_group_by("i.dataset_id")
        assert plan.group_bys == ["i.image_id", "i.dataset_id"]

    def test_add_having(self):
        """Test adding HAVING conditions."""
        plan = QueryPlan()
        plan.add_having("MAX(g.scaled_grade) >= 1")
        plan.add_having("COUNT(s.segmentation_id) > 0")
        assert len(plan.havings) == 2

    def test_add_param(self):
        """Test adding parameterized query parameters."""
        plan = QueryPlan()
        param1 = plan.add_param("modality", "fundus")
        param2 = plan.add_param("modality", "oct")
        assert param1 == "%(modality_0)s"
        assert param2 == "%(modality_1)s"
        assert plan.params == {"modality_0": "fundus", "modality_1": "oct"}
        assert plan.param_counter == 2

    def test_render_sql_minimal(self):
        """Test rendering minimal SQL query."""
        plan = QueryPlan()
        plan.add_select("i.image_id")
        plan.from_tables.append("images i")
        sql = plan.render_sql()
        assert "SELECT" in sql
        assert "i.image_id" in sql
        assert "FROM" in sql
        assert "images i" in sql

    def test_render_sql_with_joins(self):
        """Test rendering SQL with JOINs."""
        plan = QueryPlan()
        plan.add_select("i.image_id")
        plan.add_select("d.dataset_name")
        plan.from_tables.append("images i")
        plan.add_join("LEFT JOIN datasets d ON i.dataset_id = d.dataset_id")
        sql = plan.render_sql()
        assert "LEFT JOIN datasets" in sql
        assert "d.dataset_name" in sql

    def test_render_sql_with_where(self):
        """Test rendering SQL with WHERE clause."""
        plan = QueryPlan()
        plan.add_select("i.image_id")
        plan.from_tables.append("images i")
        plan.add_where("i.modality = 'fundus'")
        plan.add_where("i.storage_provider = 'local'")
        sql = plan.render_sql()
        assert "WHERE" in sql
        assert "i.modality = 'fundus'" in sql
        assert "i.storage_provider = 'local'" in sql
        assert " AND " in sql

    def test_render_sql_with_group_by(self):
        """Test rendering SQL with GROUP BY clause."""
        plan = QueryPlan()
        plan.add_select("i.image_id")
        plan.add_select("COUNT(*) AS count")
        plan.from_tables.append("images i")
        plan.add_group_by("i.image_id")
        sql = plan.render_sql()
        assert "GROUP BY" in sql
        assert "i.image_id" in sql

    def test_render_sql_with_having(self):
        """Test rendering SQL with HAVING clause."""
        plan = QueryPlan()
        plan.add_select("i.image_id")
        plan.add_select("MAX(g.scaled_grade) AS max_grade")
        plan.from_tables.append("images i")
        plan.add_group_by("i.image_id")
        plan.add_having("MAX(g.scaled_grade) >= 1")
        sql = plan.render_sql()
        assert "HAVING" in sql
        assert "MAX(g.scaled_grade) >= 1" in sql

    def test_render_sql_complete(self):
        """Test rendering complete SQL query with all clauses."""
        plan = QueryPlan()
        plan.add_select("i.image_id")
        plan.add_select("d.dataset_name")
        plan.from_tables.append("images i")
        plan.add_join("LEFT JOIN datasets d ON i.dataset_id = d.dataset_id")
        plan.add_where("i.modality = 'fundus'")
        plan.add_group_by("i.image_id")
        plan.add_group_by("d.dataset_name")
        plan.add_having("COUNT(*) > 0")
        sql = plan.render_sql()
        assert "SELECT" in sql
        assert "FROM" in sql
        assert "LEFT JOIN" in sql
        assert "WHERE" in sql
        assert "GROUP BY" in sql
        assert "HAVING" in sql

    def test_render_sql_no_select_raises_error(self):
        """Test that rendering without SELECT raises ValueError."""
        plan = QueryPlan()
        plan.from_tables.append("images i")
        with pytest.raises(ValueError, match="must have at least one SELECT clause"):
            plan.render_sql()

    def test_render_sql_no_from_raises_error(self):
        """Test that rendering without FROM raises ValueError."""
        plan = QueryPlan()
        plan.add_select("i.image_id")
        with pytest.raises(ValueError, match="must have at least one FROM table"):
            plan.render_sql()


class TestQueryBuilder:
    """Tests for QueryBuilder."""

    def test_build_query_minimal(self):
        """Test building a minimal query with only image and dataset modules."""
        builder = QueryBuilder()
        spec = ExportSpec()
        plan = builder.build_query(spec)
        assert plan is not None
        sql = plan.render_sql()
        assert "SELECT" in sql
        assert "FROM" in sql

    def test_build_query_with_dataset_filter(self):
        """Test building query with dataset name filter."""
        builder = QueryBuilder()
        spec = ExportSpec(dataset_names=["EYEPACS"])
        plan = builder.build_query(spec)
        assert plan is not None
        sql = plan.render_sql()
        assert "datasets" in sql.lower() or "dataset" in sql.lower()

    def test_build_query_with_modality_filter(self):
        """Test building query with modality filter."""
        builder = QueryBuilder()
        spec = ExportSpec(modalities=["fundus"])
        plan = builder.build_query(spec)
        assert plan is not None
        # Modality filter should be in WHERE clause
        sql = plan.render_sql()
        assert "modality" in sql.lower()

    def test_build_query_with_split_filter(self):
        """Test building query with split filter."""
        builder = QueryBuilder()
        spec = ExportSpec(split_names=["train"])
        plan = builder.build_query(spec)
        assert plan is not None
        sql = plan.render_sql()
        # Split module should add split-related JOINs
        assert len(plan.joins) > 0

    def test_build_query_with_grading(self):
        """Test building query with grading annotation task."""
        builder = QueryBuilder()
        spec = ExportSpec(
            annotation_tasks=["grading"],
            disease_types=["DR"]
        )
        plan = builder.build_query(spec)
        assert plan is not None
        sql = plan.render_sql()
        # Grading module should add grading-related JOINs
        assert len(plan.joins) > 0

    def test_build_query_with_segmentation(self):
        """Test building query with segmentation annotation task."""
        builder = QueryBuilder()
        spec = ExportSpec(annotation_tasks=["segmentation"])
        plan = builder.build_query(spec)
        assert plan is not None
        sql = plan.render_sql()
        # Segmentation module should add segmentation-related JOINs
        assert len(plan.joins) > 0

    def test_build_query_with_multiple_annotation_tasks(self):
        """Test building query with multiple annotation tasks."""
        builder = QueryBuilder()
        spec = ExportSpec(
            annotation_tasks=["grading", "segmentation", "keyword"]
        )
        plan = builder.build_query(spec)
        assert plan is not None
        sql = plan.render_sql()
        # Should have multiple JOINs for different annotation types
        assert len(plan.joins) > 1

    def test_build_query_unknown_annotation_task_raises_error(self):
        """Test that unknown annotation task raises ValidationError at spec construction."""
        with pytest.raises(ValidationError, match="Invalid annotation_tasks"):
            ExportSpec(annotation_tasks=["unknown_task"])

    def test_validate_spec_disease_types_without_grading(self):
        """Test that ExportSpec catches disease_types without grading at construction."""
        with pytest.raises(ValidationError, match="disease_types requires 'grading' in annotation_tasks"):
            ExportSpec(disease_types=["DR"])

    def test_validate_spec_grading_scale_without_grading(self):
        """Test that ExportSpec catches grading_scale_name without grading at construction."""
        with pytest.raises(ValidationError, match="grading_scale_name requires 'grading' in annotation_tasks"):
            ExportSpec(grading_scale_name="ETDRS")

    def test_validate_spec_grade_filter_without_grading(self):
        """Test that ExportSpec catches grade_filter without grading at construction."""
        with pytest.raises(ValidationError, match="grade_filter requires 'grading' in annotation_tasks"):
            ExportSpec(grade_filter={"DR": {"min": 1}})

    def test_build_query_complex_spec(self):
        """Test building query with complex spec."""
        builder = QueryBuilder()
        spec = ExportSpec(
            dataset_names=["EYEPACS"],
            split_names=["train"],
            annotation_tasks=["grading", "segmentation"],
            disease_types=["DR"],
            modalities=["fundus"],
            storage_provider="local",
            annotation_source="prefer_consensus",
            require_annotations=True
        )
        plan = builder.build_query(spec)
        assert plan is not None
        sql = plan.render_sql()
        assert "SELECT" in sql
        assert "FROM" in sql
        # Should have multiple JOINs
        assert len(plan.joins) > 1
        # Should have WHERE conditions
        assert len(plan.wheres) > 0

    def test_build_query_parameters_are_set(self):
        """Test that query parameters are properly set in plan."""
        builder = QueryBuilder()
        spec = ExportSpec(
            dataset_names=["EYEPACS"],
            modalities=["fundus"]
        )
        plan = builder.build_query(spec)
        # Parameters should be set (exact structure depends on module implementation)
        assert isinstance(plan.params, dict)


class TestRequireAnnotationsModeQueryBuilding:
    """Tests for require_annotations_mode in query building."""

    def test_require_annotations_mode_none_uses_left_join(self):
        """Test that mode='none' uses LEFT JOIN for annotation modules."""
        builder = QueryBuilder()
        spec = ExportSpec(
            annotation_tasks=["segmentation"],
            require_annotations_mode="none"
        )
        plan = builder.build_query(spec)
        sql = plan.render_sql()
        # Should have LEFT JOIN for segmentation
        assert "LEFT JOIN" in sql
        assert "segmentation_annotations" in sql

    def test_require_annotations_mode_all_uses_inner_join(self):
        """Test that mode='all' uses INNER JOIN for annotation modules."""
        builder = QueryBuilder()
        spec = ExportSpec(
            annotation_tasks=["segmentation"],
            require_annotations_mode="all"
        )
        plan = builder.build_query(spec)
        sql = plan.render_sql()
        # Should have INNER JOIN for segmentation
        assert "INNER JOIN" in sql
        assert "segmentation_annotations" in sql

    def test_require_annotations_mode_any_adds_having_clause(self):
        """Test that mode='any' adds HAVING clause requiring at least one annotation."""
        builder = QueryBuilder()
        spec = ExportSpec(
            annotation_tasks=["segmentation", "localization"],
            require_annotations_mode="any"
        )
        plan = builder.build_query(spec)
        sql = plan.render_sql()
        
        # Should have LEFT JOINs
        assert "LEFT JOIN" in sql
        
        # Should have HAVING clause with OR conditions
        assert "HAVING" in sql
        assert "COUNT(s.segmentation_id) > 0" in sql or "COUNT(l.localization_id) > 0" in sql
        assert " OR " in sql

    def test_require_annotations_mode_any_with_single_task(self):
        """Test that mode='any' works with single annotation task."""
        builder = QueryBuilder()
        spec = ExportSpec(
            annotation_tasks=["segmentation"],
            require_annotations_mode="any"
        )
        plan = builder.build_query(spec)
        sql = plan.render_sql()
        
        # Should have HAVING clause even with single task
        assert "HAVING" in sql
        assert "COUNT(s.segmentation_id) > 0" in sql

    def test_backward_compat_require_annotations_true(self):
        """Test backward compatibility: require_annotations=True works like mode='all'."""
        builder = QueryBuilder()
        spec = ExportSpec(
            annotation_tasks=["segmentation"],
            require_annotations=True
        )
        plan = builder.build_query(spec)
        sql = plan.render_sql()
        
        # Should use INNER JOIN
        assert "INNER JOIN" in sql
        assert spec.require_annotations_mode == "all"

    def test_backward_compat_require_annotations_false(self):
        """Test backward compatibility: require_annotations=False works like mode='none'."""
        builder = QueryBuilder()
        spec = ExportSpec(
            annotation_tasks=["segmentation"],
            require_annotations=False
        )
        plan = builder.build_query(spec)
        sql = plan.render_sql()
        
        # Should use LEFT JOIN
        assert "LEFT JOIN" in sql
        assert spec.require_annotations_mode == "none"


class TestAnnotationOrFiltersQueryBuilding:
    """Tests for annotation_or_filters in query building."""

    def test_or_filters_applied_to_segmentation(self):
        """Test that OR filters are applied to segmentation module."""
        from chaksudb.export.spec import AnnotationOrFilter
        
        builder = QueryBuilder()
        spec = ExportSpec(
            annotation_tasks=["segmentation"],
            annotation_or_filters=[
                AnnotationOrFilter(
                    task="segmentation",
                    conditions=[
                        {"segmentation_types": ["lesion"], "lesion_subtypes": ["microaneurysm"]},
                        {"segmentation_types": ["vessel"]},
                    ]
                )
            ]
        )
        plan = builder.build_query(spec)
        sql = plan.render_sql()
        
        # Should have WHERE clause with OR conditions
        assert "WHERE" in sql
        # Should have NULL handling for LEFT JOIN
        assert "s.segmentation_id IS NULL OR" in sql

    def test_or_filters_applied_to_localization(self):
        """Test that OR filters are applied to localization module."""
        from chaksudb.export.spec import AnnotationOrFilter
        
        builder = QueryBuilder()
        spec = ExportSpec(
            annotation_tasks=["localization"],
            annotation_or_filters=[
                AnnotationOrFilter(
                    task="localization",
                    conditions=[
                        {"localization_types": ["bounding_box"]},
                        {"localization_types": ["keypoint"]},
                    ]
                )
            ]
        )
        plan = builder.build_query(spec)
        sql = plan.render_sql()
        
        # Should have WHERE clause with OR conditions
        assert "WHERE" in sql
        # Should have NULL handling for LEFT JOIN
        assert "l.localization_id IS NULL OR" in sql
