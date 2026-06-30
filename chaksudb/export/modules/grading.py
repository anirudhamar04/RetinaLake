"""
GradingModule: Disease grading annotations.

Adds disease_grading table JOIN and includes grading fields in the output.
Handles consensus/expert preference logic and disease type filtering.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chaksudb.export.query_builder import QueryPlan
    from chaksudb.export.spec import ExportSpec

from chaksudb.export.modules.base import BaseModule


class GradingModule(BaseModule):
    """
    Module for adding disease grading annotations to export queries.

    This module adds JOINs to disease_grading and grading_scales tables and includes
    grading fields in the output. It handles consensus/expert preference logic and
    filters by disease types and grading scales.

    The JOIN type (LEFT vs INNER) depends on require_annotations_mode:
    - If require_annotations_mode is "none" or "any": LEFT JOIN (include images without grades)
    - If require_annotations_mode is "all": INNER JOIN (only include images with grades)

    Consensus preference modes:
    - expert_only: Only include expert annotations (consensus_id IS NULL)
    - consensus_only: Only include consensus annotations (consensus_id IS NOT NULL)
    - prefer_consensus: Prefer consensus, fallback to expert (uses LATERAL subquery)
    - both: Include both when available (requires aggregation)

    Output fields (varies by disease types and options):
        - For each disease type: {disease_type}_grade (scaled grade)
        - For each disease type: {disease_type}_original_grade (if include_original_grade)
        - For each disease type: {disease_type}_scale_name (scale name)
        - For each disease type: {disease_type}_source (expert/consensus)
    """

    def apply(self, plan: "QueryPlan", spec: "ExportSpec") -> None:
        """
        Apply grading module to the query plan.

        Adds:
        - JOIN to disease_grading table (LEFT or INNER based on require_annotations_mode)
        - JOIN to grading_scales table
        - Grading fields to SELECT with disease pivoting (conditional aggregation)
        - WHERE filters for disease_types, grading_scale_name, and consensus preference
        - GROUP BY image_id (required for aggregation)
        - HAVING filters for grade_filter if specified

        Args:
            plan: The QueryPlan to modify
            spec: The ExportSpec containing user requirements
        """
        # Determine JOIN type based on require_annotations_mode
        join_type = "INNER" if spec.require_annotations_mode == "all" else "LEFT"

        # For disease pivoting, we need to aggregate by image_id
        # Use a subquery to handle consensus preference per disease type
        if spec.annotation_source == "prefer_consensus":
            # Use subquery with window function to select best grading per disease type
            self._add_prefer_consensus_subquery(plan, join_type, spec)
        else:
            # For other modes, use direct JOIN with filters
            self._add_simple_grading_join(plan, join_type, spec)

        # Add JOIN to grading_scales for scale name lookup
        plan.add_join(
            "LEFT JOIN grading_scales gs ON g.scale_id = gs.scale_id"
        )

        # Add filters for disease_types.
        # For prefer_consensus, disease_types are already filtered inside the subquery —
        # adding an outer WHERE would drop LEFT-JOIN null rows (images with no grading).
        # For simple joins with LEFT JOIN, allow NULLs so unannotated images aren't lost.
        if spec.disease_types and spec.annotation_source != "prefer_consensus":
            param_placeholder = plan.add_param("disease_types", spec.disease_types)
            if join_type == "LEFT":
                plan.add_where(
                    f"(g.disease_type = ANY({param_placeholder}) OR g.disease_type IS NULL)"
                )
            else:
                plan.add_where(f"g.disease_type = ANY({param_placeholder})")

        # Add filter for grading_scale_name (only if not already applied in subquery)
        # For prefer_consensus mode, the scale filter is already in the subquery
        # For other modes, apply it here
        if spec.grading_scale_name and spec.annotation_source != "prefer_consensus":
            self.add_parameterized_where(
                plan,
                "gs.scale_name = %s",
                "grading_scale_name",
                spec.grading_scale_name,
            )
        elif spec.grading_scale_name and spec.annotation_source == "prefer_consensus":
            # For prefer_consensus, we still need to filter the final result by scale
            # (though it should already be filtered in subquery, this ensures consistency)
            self.add_parameterized_where(
                plan,
                "gs.scale_name = %s",
                "grading_scale_name",
                spec.grading_scale_name,
            )

        # Add consensus/expert filters for simple modes
        if spec.annotation_source == "expert_only":
            plan.add_where("g.consensus_id IS NULL")
        elif spec.annotation_source == "consensus_only":
            plan.add_where("g.consensus_id IS NOT NULL")

        # Add GROUP BY image_id for aggregation (required for pivoting)
        # Note: Other modules may add additional GROUP BY columns
        plan.add_group_by("i.image_id")

        # Add grading fields with disease pivoting
        self._add_grading_fields(plan, spec)

        # Add grade_filter to HAVING clause (applied after aggregation)
        if spec.grade_filter:
            self._add_grade_filters(plan, spec)

    def _add_simple_grading_join(
        self, plan: "QueryPlan", join_type: str, spec: "ExportSpec"
    ) -> None:
        """
        Add simple grading JOIN (for expert_only or consensus_only modes).

        Args:
            plan: The QueryPlan to modify
            join_type: "LEFT" or "INNER"
            spec: The ExportSpec
        """
        plan.add_join(
            f"{join_type} JOIN disease_grading g ON i.image_id = g.image_id"
        )

    def _add_prefer_consensus_subquery(
        self, plan: "QueryPlan", join_type: str, spec: "ExportSpec"
    ) -> None:
        """
        Add grading JOIN with prefer_consensus logic using subquery with window function.

        Uses a subquery with ROW_NUMBER() window function to select the best grading
        per disease type (consensus preferred, then expert, ordered by created_at DESC).

        When grading_scale_name is specified, filters to that scale in the subquery.
        When not specified, includes all scales for the disease type.

        Args:
            plan: The QueryPlan to modify
            join_type: "LEFT" or "INNER" (affects the join)
            spec: The ExportSpec
        """
        # Build subquery that ranks gradings per disease type
        # Filter by disease_types if specified
        where_conditions = []
        if spec.disease_types:
            param_placeholder = plan.add_param("disease_types_subq", spec.disease_types)
            where_conditions.append(f"g_inner.disease_type = ANY({param_placeholder})")

        # If grading_scale_name is specified, filter by scale in the subquery
        # This ensures we only get gradings with the specified scale
        if spec.grading_scale_name:
            # Need to join grading_scales in the subquery to filter by scale_name
            scale_param = plan.add_param("grading_scale_name_subq", spec.grading_scale_name)
            # Build WHERE clause properly
            where_parts = []
            if where_conditions:
                where_parts.extend(where_conditions)
            where_parts.append(f"gs_inner.scale_name = {scale_param}")
            where_clause = f"WHERE {' AND '.join(where_parts)}"
            
            subquery = (
                f"{join_type} JOIN ("
                "  SELECT g_inner.*, "
                "    ROW_NUMBER() OVER ("
                "      PARTITION BY g_inner.image_id, g_inner.disease_type "
                "      ORDER BY (g_inner.consensus_id IS NOT NULL) DESC, "
                "               g_inner.created_at DESC, g_inner.grading_id DESC"
                "    ) AS rn "
                "  FROM disease_grading g_inner "
                "  INNER JOIN grading_scales gs_inner ON g_inner.scale_id = gs_inner.scale_id "
                f"  {where_clause}"
                ") g ON i.image_id = g.image_id AND g.rn = 1"
            )
        else:
            # No scale filter - get all gradings for the disease type
            where_clause = ""
            if where_conditions:
                where_clause = f"WHERE {' AND '.join(where_conditions)}"
            
            subquery = (
                f"{join_type} JOIN ("
                "  SELECT g_inner.*, "
                "    ROW_NUMBER() OVER ("
                "      PARTITION BY g_inner.image_id, g_inner.disease_type "
                "      ORDER BY (g_inner.consensus_id IS NOT NULL) DESC, "
                "               g_inner.created_at DESC, g_inner.grading_id DESC"
                "    ) AS rn "
                "  FROM disease_grading g_inner "
                f"  {where_clause}"
                ") g ON i.image_id = g.image_id AND g.rn = 1"
            )

        plan.add_join(subquery)


    def _add_grading_fields(self, plan: "QueryPlan", spec: "ExportSpec") -> None:
        """
        Add grading fields to SELECT with disease pivoting using conditional aggregation.

        Pivots grading data by disease type, creating separate columns for each disease
        (dr_grade, dme_grade, etc.). Uses MAX() aggregation with CASE WHEN to handle
        multiple gradings per image.

        Args:
            plan: The QueryPlan to modify
            spec: The ExportSpec
        """
        # Define disease types to pivot (use spec.disease_types if provided, otherwise all)
        disease_types = spec.disease_types or ["DR", "DME", "Glaucoma", "AMD"]

        # For each disease type, add pivoted fields
        for disease_type in disease_types:
            # Normalize disease type for column name (lowercase, handle special cases)
            col_prefix = disease_type.lower()

            # Scaled grade (integer)
            # Always include scaled_grade when a specific scale is specified
            # Otherwise, include based on include_scaled_grade flag
            if spec.grading_scale_name or spec.include_scaled_grade:
                plan.add_select(
                    f"MAX(CASE WHEN g.disease_type = '{disease_type}' "
                    f"THEN g.scaled_grade END) AS {col_prefix}_grade"
                )

            # Original grade (text)
            if spec.include_original_grade:
                plan.add_select(
                    f"MAX(CASE WHEN g.disease_type = '{disease_type}' "
                    f"THEN g.original_grade END) AS {col_prefix}_original_grade"
                )

            # Scale name
            plan.add_select(
                f"MAX(CASE WHEN g.disease_type = '{disease_type}' "
                f"THEN gs.scale_name END) AS {col_prefix}_scale_name"
            )

            # Annotation source (expert/consensus)
            plan.add_select(
                f"MAX(CASE WHEN g.disease_type = '{disease_type}' "
                f"THEN CASE WHEN g.consensus_id IS NOT NULL THEN 'consensus' "
                f"ELSE 'expert' END END) AS {col_prefix}_annotation_source"
            )

    def _add_grade_filters(self, plan: "QueryPlan", spec: "ExportSpec") -> None:
        """
        Add grade_filter conditions to HAVING clause.

        Supports multiple formats:
        - {'DR': {'min': 1, 'max': 3}} - range filter
        - {'DR': {'values': [0, 1, 2]}} - specific values
        - {'DR': [0, 1, 2]} - shorthand for values

        Args:
            plan: The QueryPlan to modify
            spec: The ExportSpec with grade_filter dict
        """
        if not spec.grade_filter:
            return

        # grade_filter format: {'DR': {'min': 1, 'max': 3}, 'DME': {'min': 0}}
        # or {'DR': {'values': [0, 1, 2]}} or {'DR': [0, 1, 2]}
        for disease_type, filters in spec.grade_filter.items():
            # Handle shorthand format: {'DR': [0, 1, 2]}
            if isinstance(filters, list):
                filters = {"values": filters}
            
            # Handle range filters (min/max)
            if "min" in filters:
                param_placeholder = plan.add_param(
                    f"{disease_type}_min_grade", filters["min"]
                )
                plan.add_having(
                    f"MAX(CASE WHEN g.disease_type = '{disease_type}' "
                    f"THEN g.scaled_grade END) >= {param_placeholder}"
                )

            if "max" in filters:
                param_placeholder = plan.add_param(
                    f"{disease_type}_max_grade", filters["max"]
                )
                plan.add_having(
                    f"MAX(CASE WHEN g.disease_type = '{disease_type}' "
                    f"THEN g.scaled_grade END) <= {param_placeholder}"
                )

            # Handle specific values filter
            if "values" in filters:
                param_placeholder = plan.add_param(
                    f"{disease_type}_grade_values", filters["values"]
                )
                plan.add_having(
                    f"MAX(CASE WHEN g.disease_type = '{disease_type}' "
                    f"THEN g.scaled_grade END) = ANY({param_placeholder})"
                )

    def get_output_fields(self) -> list[str]:
        """
        Get the list of output field names this module adds.

        Returns:
            List of field names with disease type prefixes:
            - For each disease type: {disease_type}_grade, {disease_type}_original_grade,
              {disease_type}_scale_name, {disease_type}_annotation_source
        """
        # Return dynamic list based on disease types
        # This is a simplified version - actual fields depend on spec.disease_types
        # and spec.include_scaled_grade/include_original_grade
        fields = []
        disease_types = ["dr", "dme", "glaucoma", "amd"]
        for dt in disease_types:
            fields.extend([
                f"{dt}_grade",
                f"{dt}_original_grade",
                f"{dt}_scale_name",
                f"{dt}_annotation_source",
            ])
        return fields

    def get_primary_id_column(self) -> str | None:
        """
        Get the primary ID column for grading annotations.
        
        Returns:
            "g.grading_id" - used for require_annotations_mode="any" HAVING clause
        """
        return "g.grading_id"
