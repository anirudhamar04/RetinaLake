"""
DatasetModule: Dataset join and name filtering.

Adds the datasets table JOIN and includes dataset_name in the output.
Handles filtering by dataset_ids or dataset_names.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chaksudb.export.query_builder import QueryPlan
    from chaksudb.export.spec import ExportSpec

from chaksudb.export.modules.base import BaseModule


class DatasetModule(BaseModule):
    """
    Module for adding dataset information to export queries.

    This module adds a JOIN to the datasets table and includes dataset_name in the output.
    It handles filtering by dataset UUIDs or names.

    Output fields:
        - dataset_name: Name of the dataset
    """

    def apply(self, plan: "QueryPlan", spec: "ExportSpec") -> None:
        """
        Apply dataset module to the query plan.

        Adds:
        - JOIN to datasets table: LEFT JOIN datasets d ON i.dataset_id = d.dataset_id
        - dataset_name to SELECT
        - WHERE filters for dataset_ids or dataset_names if specified

        Args:
            plan: The QueryPlan to modify
            spec: The ExportSpec containing user requirements
        """
        # Add JOIN to datasets table
        plan.add_join("LEFT JOIN datasets d ON i.dataset_id = d.dataset_id")

        # Add dataset_name to SELECT
        plan.add_select("d.dataset_name", group_by_expression="d.dataset_name")

        # Filters are membership-aware: an image matches if it is owned by a requested
        # dataset OR it is a cross-dataset member of one (image_dataset_memberships). This
        # lets a shared image (e.g. MAPLES-DR annotations on MESSIDOR images) surface when
        # the user filters by the secondary dataset name.
        if spec.dataset_ids:
            ph = plan.add_param("dataset_ids", spec.dataset_ids)
            self._add_membership_aware_filter(plan, "i.dataset_id", "idm.dataset_id", ph)

        if spec.dataset_names:
            ph = plan.add_param("dataset_names", spec.dataset_names)
            self._add_membership_aware_filter(
                plan,
                "d.dataset_name",
                "d2.dataset_name",
                ph,
                membership_join="JOIN datasets d2 ON idm.dataset_id = d2.dataset_id",
            )

    @staticmethod
    def _add_membership_aware_filter(
        plan,
        owner_col: str,
        membership_col: str,
        ph: str,
        membership_join: str = "",
    ) -> None:
        """WHERE clause matching images owned by, OR cross-dataset members of, the requested
        datasets. ``membership_join`` resolves the membership row to the column being matched
        (needed when filtering by name rather than id)."""
        plan.add_where(
            f"({owner_col} = ANY({ph}) OR EXISTS ("
            f"  SELECT 1 FROM image_dataset_memberships idm {membership_join}"
            f"  WHERE idm.image_id = i.image_id AND {membership_col} = ANY({ph})))"
        )

    def get_output_fields(self) -> list[str]:
        """
        Get the list of output field names this module adds.

        Returns:
            List of field names: dataset_name
        """
        return ["dataset_name"]
