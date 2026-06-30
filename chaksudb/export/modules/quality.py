"""
QualityModule: Quality assessment annotations.

Adds quality_annotations table JOIN and includes quality fields in the output.
Handles pivoting by quality_type or aggregation as JSONB.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chaksudb.export.query_builder import QueryPlan
    from chaksudb.export.spec import ExportSpec

from chaksudb.export.modules.base import BaseModule


class QualityModule(BaseModule):
    """
    Module for adding quality assessment annotations to export queries.

    This module adds a JOIN to quality_annotations table and includes quality
    assessment fields in the output. It can pivot by quality_type (creating
    separate columns for each quality type) or aggregate as JSONB.

    The JOIN type (LEFT vs INNER) depends on require_annotations_mode:
    - If require_annotations_mode is "none" or "any": LEFT JOIN (include images without quality annotations)
    - If require_annotations_mode is "all": INNER JOIN (only include images with quality annotations)

    Note: quality_annotations table does not have consensus_id, so consensus preference
    logic is not applicable. Only expert annotations are available.

    Output fields (when pivoting by quality_type):
        - For each quality type: {quality_type}_quality_score, {quality_type}_quality_label
        - Or aggregated: quality_annotations (JSONB array)

    For simplicity, this implementation pivots by quality_type similar to grading.
    Common quality types: 'overall', 'gradability', 'clarity', 'field_definition',
    'artifact', 'contrast', 'blur', 'illumination'
    """

    def apply(self, plan: "QueryPlan", spec: "ExportSpec") -> None:
        """
        Apply quality module to the query plan.

        Adds:
        - JOIN to quality_annotations table (LEFT or INNER based on require_annotations_mode)
        - Quality fields to SELECT with quality_type pivoting (conditional aggregation)
        - GROUP BY image_id (required for aggregation)

        Args:
            plan: The QueryPlan to modify
            spec: The ExportSpec containing user requirements
        """
        # Determine JOIN type based on require_annotations_mode
        join_type = "INNER" if spec.require_annotations_mode == "all" else "LEFT"

        # Add JOIN to quality_annotations table
        plan.add_join(
            f"{join_type} JOIN quality_annotations q ON i.image_id = q.image_id"
        )

        # Add GROUP BY image_id for aggregation (required for pivoting)
        # Note: Other modules may add additional GROUP BY columns
        plan.add_group_by("i.image_id")

        # Add quality fields with quality_type pivoting
        self._add_quality_fields(plan, spec)

    def _add_quality_fields(self, plan: "QueryPlan", spec: "ExportSpec") -> None:
        """
        Add quality fields to SELECT with quality_type pivoting using conditional aggregation.

        Pivots quality data by quality_type, creating separate columns for each quality type
        (overall_quality_score, gradability_quality_score, etc.). Uses MAX() aggregation
        with CASE WHEN to handle multiple quality annotations per image.

        Args:
            plan: The QueryPlan to modify
            spec: The ExportSpec
        """
        # Pivot only the requested quality types when provided (build_dataset_spec fills
        # this with the types a dataset actually has, avoiding all-NULL columns); otherwise
        # fall back to the common registered set.
        quality_types = spec.quality_types or [
            "overall",
            "gradability",
            "clarity",
            "field_definition",
            "artifact",
            "contrast",
            "blur",
            "illumination",
        ]

        # For each quality type, add pivoted fields
        for quality_type in quality_types:
            # Normalize quality type for column name (replace underscores if needed)
            col_prefix = quality_type

            # Quality score (float)
            plan.add_select(
                f"MAX(CASE WHEN q.quality_type = '{quality_type}' "
                f"THEN q.quality_score END) AS {col_prefix}_quality_score"
            )

            # Quality label (text)
            plan.add_select(
                f"MAX(CASE WHEN q.quality_type = '{quality_type}' "
                f"THEN q.quality_label END) AS {col_prefix}_quality_label"
            )

        # Also add a simple aggregated version for backward compatibility.
        # DISTINCT collapses duplicates produced when this join is multiplied by another
        # one-to-many task in the same query (the per-type pivot columns above use MAX and
        # are unaffected; this JSONB column would otherwise repeat each entry).
        quality_obj = (
            "jsonb_build_object("
            "  'quality_type', q.quality_type,"
            "  'quality_score', q.quality_score,"
            "  'quality_label', q.quality_label"
            ")"
        )
        plan.add_select(
            f"jsonb_agg(DISTINCT {quality_obj} ORDER BY {quality_obj}) "
            "FILTER (WHERE q.quality_id IS NOT NULL) AS quality_annotations"
        )

    def get_output_fields(self) -> list[str]:
        """
        Get the list of output field names this module adds.

        Returns:
            List of field names with quality type prefixes:
            - For each quality type: {quality_type}_quality_score, {quality_type}_quality_label
            - quality_annotations: JSONB array of all quality annotations
        """
        quality_types = [
            "overall",
            "gradability",
            "clarity",
            "field_definition",
            "artifact",
            "contrast",
            "blur",
            "illumination",
        ]
        fields = []
        for qt in quality_types:
            fields.extend([
                f"{qt}_quality_score",
                f"{qt}_quality_label",
            ])
        fields.append("quality_annotations")
        return fields

    def get_primary_id_column(self) -> str | None:
        """
        Get the primary ID column for quality annotations.
        
        Returns:
            "q.quality_id" - used for require_annotations_mode="any" HAVING clause
        """
        return "q.quality_id"
