"""
QueryBuilder and QueryPlan: SQL query construction for exports.

Provides a composable way to build SQL queries incrementally through modules.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from chaksudb.export.spec import ExportSpec

from chaksudb.export.modules.caption import CaptionModule
from chaksudb.export.modules.classification import ClassificationModule
from chaksudb.export.modules.clinical import ClinicalModule
from chaksudb.export.modules.dataset import DatasetModule
from chaksudb.export.modules.fundus_roi import FundusROIModule
from chaksudb.export.modules.grading import GradingModule
from chaksudb.export.modules.health import HealthStatusModule
from chaksudb.export.modules.image import ImageModule
from chaksudb.export.modules.keywords import KeywordsModule
from chaksudb.export.modules.localization import LocalizationModule
from chaksudb.export.modules.patient import PatientModule
from chaksudb.export.modules.quality import QualityModule
from chaksudb.export.modules.segmentation import SegmentationModule
from chaksudb.export.modules.split import SplitModule


@dataclass
class QueryPlan:
    """
    Data structure for building SQL queries incrementally.

    Modules add fragments to this plan, which is then rendered into a complete SQL query.
    Uses parameterized queries for safety.

    Attributes:
        select: List of SELECT clause fragments (e.g., ["i.image_id", "i.file_path"])
        from_tables: List of base FROM tables (typically just one, e.g., ["images i"])
        joins: List of JOIN clauses (e.g., ["LEFT JOIN datasets d ON i.dataset_id = d.dataset_id"])
        wheres: List of WHERE clause fragments (combined with AND)
        group_bys: List of GROUP BY columns
        havings: List of HAVING clause fragments (combined with AND)
        params: Dictionary of parameterized query parameters (for psycopg style: %(name)s)
        param_counter: Counter for generating unique parameter names
        group_by_expressions_from_select: When GROUP BY is used, these expressions are added to group_bys
    """

    select: list[str] = field(default_factory=list)
    from_tables: list[str] = field(default_factory=list)
    joins: list[str] = field(default_factory=list)
    wheres: list[str] = field(default_factory=list)
    group_bys: list[str] = field(default_factory=list)
    havings: list[str] = field(default_factory=list)
    order_bys: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    param_counter: int = field(default=0)
    group_by_expressions_from_select: list[str] = field(default_factory=list)

    def add_select(
        self, fragment: str, group_by_expression: str | None = None
    ) -> None:
        """
        Add a SELECT clause fragment.

        Args:
            fragment: SQL fragment to add (e.g., "i.image_id", "COUNT(*) AS count")
            group_by_expression: When aggregation (GROUP BY) is used, this expression is added to
                group_bys so the column is valid. Use the same expression as in SELECT (or the part
                before AS alias). Omit for aggregate expressions (e.g. MAX(...), jsonb_agg(...)).
        """
        self.select.append(fragment)
        if group_by_expression is not None:
            self.group_by_expressions_from_select.append(group_by_expression)

    def add_join(self, join_clause: str) -> None:
        """
        Add a JOIN clause.

        Args:
            join_clause: Complete JOIN clause (e.g., "LEFT JOIN datasets d ON i.dataset_id = d.dataset_id")
        """
        self.joins.append(join_clause)

    def add_where(self, condition: str) -> None:
        """
        Add a WHERE clause fragment.

        Args:
            condition: SQL condition (e.g., "i.modality = %(modality)s")
        """
        self.wheres.append(condition)

    def add_group_by(self, column: str) -> None:
        """
        Add a GROUP BY column.

        Args:
            column: Column name or expression (e.g., "i.image_id")
        """
        self.group_bys.append(column)

    def add_having(self, condition: str) -> None:
        """
        Add a HAVING clause fragment.

        Args:
            condition: SQL condition (e.g., "MAX(g.scaled_grade) >= %(min_grade)s")
        """
        self.havings.append(condition)

    def add_order_by(self, expression: str) -> None:
        """
        Add an ORDER BY expression.

        Args:
            expression: SQL expression (e.g., "i.image_id")
        """
        self.order_bys.append(expression)

    def add_param(self, name: str, value: Any) -> str:
        """
        Add a parameterized query parameter and return the parameter placeholder.

        Args:
            name: Base name for the parameter (will be made unique if needed)
            value: Parameter value

        Returns:
            Parameter placeholder string (e.g., "%(modality_0)s")
        """
        # Make parameter name unique
        param_name = f"{name}_{self.param_counter}"
        self.param_counter += 1
        self.params[param_name] = value
        return f"%({param_name})s"

    def render_sql(self) -> str:
        """
        Render the complete SQL query from the plan.

        Combines all fragments into a valid SQL SELECT statement.

        Returns:
            Complete SQL query string with parameterized placeholders

        Raises:
            ValueError: If required components (SELECT, FROM) are missing
        """
        if not self.select:
            raise ValueError("QueryPlan must have at least one SELECT clause")
        if not self.from_tables:
            raise ValueError("QueryPlan must have at least one FROM table")

        # Build SELECT clause
        select_clause = "SELECT " + ", ".join(self.select)

        # Build FROM clause
        from_clause = "FROM " + ", ".join(self.from_tables)

        # Build JOIN clauses
        join_clause = "\n".join(self.joins) if self.joins else ""

        # Build WHERE clause
        where_clause = ""
        if self.wheres:
            where_clause = "WHERE " + " AND ".join(self.wheres)

        # Build GROUP BY clause
        group_by_clause = ""
        if self.group_bys:
            group_by_clause = "GROUP BY " + ", ".join(self.group_bys)

        # Build HAVING clause
        having_clause = ""
        if self.havings:
            having_clause = "HAVING " + " AND ".join(self.havings)

        # Build ORDER BY clause
        order_by_clause = ""
        if self.order_bys:
            order_by_clause = "ORDER BY " + ", ".join(self.order_bys)

        # Combine all clauses
        parts = [
            select_clause,
            from_clause,
            join_clause,
            where_clause,
            group_by_clause,
            having_clause,
            order_by_clause,
        ]

        # Filter out empty parts and join with newlines
        sql = "\n".join(part for part in parts if part)

        return sql


class QueryBuilder:
    """
    Builder for constructing SQL queries from ExportSpec.

    Composes modules to build complete SQL queries based on user requirements.
    Handles module discovery, composition order, and SQL rendering.

    Example:
        >>> builder = QueryBuilder()
        >>> spec = ExportSpec(
        ...     dataset_names=["EYEPACS"],
        ...     annotation_tasks=["grading"],
        ...     disease_types=["DR"]
        ... )
        >>> plan = builder.build_query(spec)
        >>> sql = plan.render_sql()
        >>> params = plan.params
    """

    # Mapping from annotation task names to module classes
    _ANNOTATION_MODULE_MAP: dict[str, type] = {
        "grading": GradingModule,
        "segmentation": SegmentationModule,
        "classification": ClassificationModule,
        "localization": LocalizationModule,
        "quality": QualityModule,
        "keyword": KeywordsModule,
        "description": ClinicalModule,
    }

    def build_query(self, spec: "ExportSpec") -> QueryPlan:
        """
        Build a QueryPlan from an ExportSpec.

        Composes modules based on the spec requirements:
        - Always includes: ImageModule, DatasetModule
        - Conditionally includes: SplitModule (if split filters specified)
        - Includes annotation modules based on annotation_tasks

        Args:
            spec: The ExportSpec containing user requirements

        Returns:
            QueryPlan with all SQL fragments and parameters

        Raises:
            ValueError: If spec contains invalid annotation tasks
        """
        # Create new QueryPlan
        plan = QueryPlan()

        # Always include core modules (order matters for JOIN dependencies)
        image_module = ImageModule()
        image_module.apply(plan, spec)

        dataset_module = DatasetModule()
        dataset_module.apply(plan, spec)

        # Conditionally include SplitModule if split filters are specified
        if spec.split_names or spec.split_task_type or spec.split_type:
            split_module = SplitModule()
            split_module.apply(plan, spec)

        # Conditionally include PatientModule
        if spec.include_patient_data:
            patient_module = PatientModule()
            patient_module.apply(plan, spec)

        # Conditionally include CaptionModule
        if spec.caption_mode is not None:
            caption_module = CaptionModule()
            caption_module.apply(plan, spec)

        # Conditionally include FundusROIModule (ROI flat columns + IQA filter)
        if spec.include_fundus_roi or spec.iqa_min_quality_score is not None or spec.iqa_quality_labels:
            roi_module = FundusROIModule()
            roi_module.apply(plan, spec)

        # Conditionally include HealthStatusModule (normal/abnormal derived field + filter)
        if spec.include_health_status or spec.health_status_filter is not None:
            HealthStatusModule().apply(plan, spec)

        # Include annotation modules based on annotation_tasks
        annotation_modules = []
        if spec.annotation_tasks:
            # Build a mapping of task name to OR filter conditions (if any)
            or_filter_map = {}
            if spec.annotation_or_filters:
                for or_filter in spec.annotation_or_filters:
                    or_filter_map[or_filter.task] = or_filter.conditions
            
            for task in spec.annotation_tasks:
                module_class = self._ANNOTATION_MODULE_MAP.get(task)
                if module_class is None:
                    raise ValueError(
                        f"Unknown annotation task: {task}. "
                        f"Valid tasks: {sorted(self._ANNOTATION_MODULE_MAP.keys())}"
                    )
                module = module_class()
                
                # Check if there are OR filters for this task
                if task in or_filter_map:
                    # Apply OR filters instead of the standard flat filters
                    module.apply_or_filters(plan, or_filter_map[task], task)
                
                # Apply the module's standard logic
                module.apply(plan, spec)
                
                # Store module for potential HAVING clause in 'any' mode
                annotation_modules.append(module)

        # When require_annotations_mode="any", add HAVING clause requiring at least one annotation
        if spec.require_annotations_mode == "any" and annotation_modules:
            # Collect primary ID columns from all annotation modules
            id_columns = []
            for module in annotation_modules:
                id_col = module.get_primary_id_column()
                if id_col:
                    id_columns.append(id_col)
            
            if id_columns:
                # Build HAVING clause: COUNT(id1) > 0 OR COUNT(id2) > 0 OR ...
                having_parts = [f"COUNT({col}) > 0" for col in id_columns]
                having_clause = " OR ".join(having_parts)
                plan.add_having(having_clause)

        # When aggregation is used (GROUP BY), every non-aggregated SELECT column must
        # appear in GROUP BY. Add all expressions registered by modules via add_select(..., group_by_expression=...).
        if plan.group_bys:
            for expr in plan.group_by_expressions_from_select:
                if expr not in plan.group_bys:
                    plan.add_group_by(expr)

        # Add deterministic ORDER BY for reproducible results
        plan.add_order_by("i.image_id")

        return plan
