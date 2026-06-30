"""
KeywordsModule: Keyword annotations.

Adds keyword_annotations and keyword_vocabulary JOINs and aggregates keywords
using array_agg() to create a list of keyword terms per image.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chaksudb.export.query_builder import QueryPlan
    from chaksudb.export.spec import ExportSpec

from chaksudb.export.modules.base import BaseModule


class KeywordsModule(BaseModule):
    """
    Module for adding keyword annotations to export queries.

    This module adds JOINs to keyword_annotations and keyword_vocabulary tables
    and aggregates keywords using array_agg() to create a list of keyword terms per image.

    The JOIN type (LEFT vs INNER) depends on require_annotations_mode:
    - If require_annotations_mode is "none" or "any": LEFT JOIN (include images without keywords)
    - If require_annotations_mode is "all": INNER JOIN (only include images with keywords)

    Output fields:
        - keywords: Array of keyword strings (distinct keyword terms)
    """

    def apply(self, plan: "QueryPlan", spec: "ExportSpec") -> None:
        """
        Apply keywords module to the query plan.

        Adds:
        - JOIN to keyword_annotations table (LEFT or INNER based on require_annotations_mode)
        - JOIN to keyword_vocabulary table
        - Aggregated keywords field to SELECT using array_agg()
        - GROUP BY image_id (required for aggregation)

        Args:
            plan: The QueryPlan to modify
            spec: The ExportSpec containing user requirements
        """
        # Determine JOIN type based on require_annotations_mode
        join_type = "INNER" if spec.require_annotations_mode == "all" else "LEFT"

        # Add JOIN to keyword_annotations table
        plan.add_join(
            f"{join_type} JOIN keyword_annotations ka ON i.image_id = ka.image_id"
        )

        # Add JOIN to keyword_vocabulary table to get keyword_term
        plan.add_join(
            "LEFT JOIN keyword_vocabulary kv ON ka.keyword_id = kv.keyword_id"
        )

        # Add GROUP BY image_id for aggregation (required for array_agg)
        # Note: Other modules may add additional GROUP BY columns
        plan.add_group_by("i.image_id")

        # Add aggregated keywords field
        self._add_keywords_fields(plan)

    def _add_keywords_fields(self, plan: "QueryPlan") -> None:
        """
        Add aggregated keywords field to SELECT using array_agg().

        Aggregates keyword terms into an array, using DISTINCT to avoid duplicates
        and FILTER to return NULL (not empty array) when no keywords exist.

        Args:
            plan: The QueryPlan to modify
        """
        # Aggregate keywords using array_agg() with DISTINCT
        # FILTER clause ensures NULL is returned (not empty array) when no keywords exist
        plan.add_select(
            "array_agg(DISTINCT kv.keyword_term) "
            "FILTER (WHERE kv.keyword_term IS NOT NULL) AS keywords"
        )

    def get_output_fields(self) -> list[str]:
        """
        Get the list of output field names this module adds.

        Returns:
            List of field names: keywords
        """
        return ["keywords"]

    def get_primary_id_column(self) -> str | None:
        """
        Get the primary ID column for keyword annotations.
        
        Returns:
            "ka.keyword_annotation_id" - used for require_annotations_mode="any" HAVING clause
        """
        return "ka.keyword_annotation_id"
