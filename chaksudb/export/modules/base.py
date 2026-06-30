"""
BaseModule: Abstract interface for export query modules.

Each module is responsible for adding specific parts of the SQL query (SELECT, JOIN, WHERE)
based on the ExportSpec requirements. Modules are composable and can be combined to build
complex queries.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from chaksudb.export.query_builder import QueryPlan
    from chaksudb.export.spec import ExportSpec


class BaseModule(ABC):
    """
    Abstract base class for export query modules.

    Each module implements the contract:
    1. `apply()`: Modifies the QueryPlan to add SELECT, JOIN, WHERE clauses
    2. `get_output_fields()`: Returns the list of field names this module adds to the output

    Modules are composed together by the QueryBuilder to construct complete queries.
    The order of module application matters for JOIN dependencies.

    Example:
        >>> class ImageModule(BaseModule):
        ...     def apply(self, plan: QueryPlan, spec: ExportSpec) -> None:
        ...         plan.from_tables.append("images i")
        ...         plan.add_select("i.image_id")
        ...         plan.add_select("i.file_path")
        ...
        ...     def get_output_fields(self) -> list[str]:
        ...         return ["image_id", "file_path"]
    """

    @abstractmethod
    def apply(self, plan: "QueryPlan", spec: "ExportSpec") -> None:
        """
        Apply this module to the query plan.

        Modifies the QueryPlan by adding SELECT fragments, JOINs, WHERE conditions,
        GROUP BY columns, or HAVING conditions as needed.

        Args:
            plan: The QueryPlan to modify
            spec: The ExportSpec containing user requirements
        """
        pass

    @abstractmethod
    def get_output_fields(self) -> list[str]:
        """
        Get the list of output field names this module adds.

        Returns:
            List of field names that will be present in the output rows when this module is used.
            These should match the keys in UnifiedExportRow (or be documented if they don't).

        Example:
            >>> module = ImageModule()
            >>> fields = module.get_output_fields()
            >>> print(fields)
            ['image_id', 'dataset_id', 'file_path', 'storage_provider', 'modality', 'eye_laterality']
        """
        pass

    def get_primary_id_column(self) -> str | None:
        """
        Get the primary ID column for this module's annotation table.
        
        Used by QueryBuilder when require_annotations_mode="any" to construct
        a HAVING clause that requires at least one annotation type to be present.
        
        Returns:
            The fully qualified primary ID column (e.g., "s.segmentation_id") if this
            is an annotation module, or None if this is a non-annotation module (like
            ImageModule, DatasetModule).
            
        Example:
            >>> module = SegmentationModule()
            >>> id_col = module.get_primary_id_column()
            >>> print(id_col)
            's.segmentation_id'
        """
        return None

    def apply_or_filters(
        self,
        plan: "QueryPlan",
        conditions: list[dict[str, Any]],
        task_name: str,
    ) -> None:
        """
        Apply OR-grouped filter conditions for this module.
        
        This method is called when annotation_or_filters contains an entry for this
        module's task. Each condition dict in the list is converted to a WHERE clause,
        and the clauses are ORed together.
        
        Default implementation does nothing. Annotation modules should override this
        to support OR filter logic.
        
        Args:
            plan: The QueryPlan to modify
            conditions: List of filter dicts to be ORed together
            task_name: The task name from annotation_or_filters (for error messages)
            
        Example:
            >>> # Called for: AnnotationOrFilter(
            >>> #   task="segmentation",
            >>> #   conditions=[
            >>> #       {"segmentation_types": ["lesion"], "lesion_subtypes": ["microaneurysm"]},
            >>> #       {"segmentation_types": ["vessel"]},
            >>> #   ]
            >>> # )
            >>> module.apply_or_filters(plan, conditions, "segmentation")
            >>> # Adds: WHERE (s.segmentation_id IS NULL OR ((cond1) OR (cond2)))
        """
        pass

    def add_parameterized_where(
        self,
        plan: "QueryPlan",
        condition_template: str,
        param_name: str,
        value: Any,
    ) -> None:
        """
        Helper method to add a parameterized WHERE condition.

        Creates a unique parameter name and adds both the condition and parameter value.

        Args:
            plan: The QueryPlan to modify
            condition_template: SQL condition template with %s placeholder
                               (e.g., "i.modality = %s")
            param_name: Base name for the parameter
            value: Parameter value

        Example:
            >>> self.add_parameterized_where(
            ...     plan,
            ...     "i.modality = %s",
            ...     "modality",
            ...     "fundus"
            ... )
            >>> # Adds: "i.modality = %(modality_0)s" to WHERE
            >>> # Adds: {"modality_0": "fundus"} to params
        """
        param_placeholder = plan.add_param(param_name, value)
        condition = condition_template.replace("%s", param_placeholder)
        plan.add_where(condition)

    def add_in_clause(
        self,
        plan: "QueryPlan",
        column: str,
        param_name: str,
        values: list[Any],
    ) -> None:
        """
        Helper method to add an IN clause for filtering by multiple values.

        Args:
            plan: The QueryPlan to modify
            column: Column name (e.g., "i.modality")
            param_name: Base name for the parameter
            values: List of values to filter by

        Example:
            >>> self.add_in_clause(plan, "i.modality", "modalities", ["fundus", "oct"])
            >>> # Adds: "i.modality = ANY(%(modalities_0)s)" to WHERE
            >>> # Adds: {"modalities_0": ["fundus", "oct"]} to params
        """
        if not values:
            return

        param_placeholder = plan.add_param(param_name, values)
        # Use ANY() for array parameter in PostgreSQL
        condition = f"{column} = ANY({param_placeholder})"
        plan.add_where(condition)
