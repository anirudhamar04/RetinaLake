"""
SplitModule: Split joins and filtering.

Adds image_splits and dataset_splits JOINs and includes split_name and task_type.
Handles filtering by split_names and split_task_type.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chaksudb.export.query_builder import QueryPlan
    from chaksudb.export.spec import ExportSpec

from chaksudb.export.modules.base import BaseModule


class SplitModule(BaseModule):
    """
    Module for adding split information to export queries.

    This module adds JOINs to image_splits and dataset_splits tables and includes
    split_name and task_type in the output. It handles filtering by split names
    and task types.

    The JOIN type (LEFT vs INNER) depends on require_annotations:
    - If require_annotations is False: LEFT JOIN (include images without splits)
    - If require_annotations is True: INNER JOIN (only include images with splits)

    An image can legitimately have more than one matching `image_splits` row (e.g.
    overlapping `explicit`/`user_defined` assignments, or stale rows from a re-run split
    assignment). A plain JOIN would fan those out into duplicate output rows per image, so
    this module joins via a LATERAL subquery that applies all split filters *inside* the
    subquery and caps the result to one row per image (`LIMIT 1`), same pattern as
    FundusROIModule's ROI lookup.

    Output fields:
        - split_name: Name of the split (e.g., 'train', 'test', 'val')
        - task_type: Task type for the split
    """

    def apply(self, plan: "QueryPlan", spec: "ExportSpec") -> None:
        """
        Apply split module to the query plan.

        Adds:
        - LATERAL JOIN (LEFT or INNER based on require_annotations) selecting at most one
          image_splits/dataset_splits row per image, with split_names/split_task_type/
          split_type filters applied inside the subquery so they select the matching row
          rather than excluding it after a fan-out join.
        - split_name and task_type to SELECT

        Args:
            plan: The QueryPlan to modify
            spec: The ExportSpec containing user requirements
        """
        # Use the current require_annotations_mode (the old `require_annotations` field is
        # deprecated and was always falsy here, so splits never dropped unannotated images).
        lateral_join = "INNER JOIN" if spec.require_annotations_mode == "all" else "LEFT JOIN"

        conditions = ["isp.image_id = i.image_id"]

        if spec.split_names:
            ph = plan.add_param("split_names", spec.split_names)
            conditions.append(f"ds.split_name = ANY({ph})")

        if spec.split_task_type:
            ph = plan.add_param("split_task_type", spec.split_task_type)
            conditions.append(f"isp.task_type = {ph}")

        if spec.split_type:
            ph = plan.add_param("split_type", spec.split_type)
            conditions.append(f"ds.split_type = {ph}")

        where_clause = " AND ".join(conditions)

        plan.add_join(
            f"{lateral_join} LATERAL ("
            "  SELECT ds.split_name AS split_name, isp.task_type AS task_type"
            "  FROM image_splits isp"
            "  JOIN dataset_splits ds ON isp.split_id = ds.split_id"
            f"  WHERE {where_clause}"
            "  ORDER BY isp.is_primary DESC, isp.created_at DESC"
            "  LIMIT 1"
            ") isp ON true"
        )

        plan.add_select("isp.split_name", group_by_expression="isp.split_name")
        plan.add_select("isp.task_type AS task_type", group_by_expression="isp.task_type")

    def get_output_fields(self) -> list[str]:
        """
        Get the list of output field names this module adds.

        Returns:
            List of field names: split_name, task_type
        """
        return ["split_name", "task_type"]
