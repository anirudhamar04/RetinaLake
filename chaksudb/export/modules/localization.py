"""
LocalizationModule: Localization annotations.

Adds localization_annotations table JOIN and aggregates localization data
using jsonb_agg() to create a list of localization annotations per image.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from chaksudb.export.query_builder import QueryPlan
    from chaksudb.export.spec import ExportSpec

from chaksudb.export.modules.base import BaseModule


class LocalizationModule(BaseModule):
    """
    Module for adding localization annotations to export queries.

    This module adds a LEFT JOIN to localization_annotations table and aggregates
    localization data using jsonb_agg() to create a list of localization annotations
    per image. It handles consensus/expert source preference.

    The JOIN type (LEFT vs INNER) depends on require_annotations_mode:
    - If require_annotations_mode is "none" or "any": LEFT JOIN (include images without localizations)
    - If require_annotations_mode is "all": INNER JOIN (only include images with localizations)

    Consensus preference modes:
    - expert_only: Only include expert annotations (consensus_id IS NULL)
    - consensus_only: Only include consensus annotations (consensus_id IS NOT NULL)
    - prefer_consensus: Prefer consensus, fallback to expert (uses subquery with window function)
    - both: Include both when available

    Localization types supported:
    - bounding_box: Bounding box coordinates
    - keypoint: Keypoint coordinates
    - center_point: Center point coordinates

    Output fields:
        - localization_annotations: JSONB array of LocalizationAnnotation objects, each containing:
            - localization_id: UUID
            - localization_type: Type of localization (bounding_box, keypoint, center_point)
            - target_structure: Target structure being localized
            - coordinates: JSONB structure containing coordinate data
            - lesion_subtype: Optional string
    """

    def apply(self, plan: "QueryPlan", spec: "ExportSpec") -> None:
        """
        Apply localization module to the query plan.

        Adds:
        - JOIN to localization_annotations table (LEFT or INNER based on require_annotations_mode)
        - Aggregated localization_annotations field to SELECT using jsonb_agg()
        - WHERE filters for consensus preference
        - GROUP BY image_id (required for aggregation)

        Args:
            plan: The QueryPlan to modify
            spec: The ExportSpec containing user requirements
        """
        # Determine JOIN type based on require_annotations_mode
        join_type = "INNER" if spec.require_annotations_mode == "all" else "LEFT"

        # Handle consensus preference logic
        if spec.annotation_source == "prefer_consensus":
            # Use subquery with window function to select best localization per image
            self._add_prefer_consensus_subquery(plan, join_type, spec)
        else:
            # For other modes, use direct JOIN with filters
            self._add_simple_localization_join(plan, join_type, spec)

        # Add consensus/expert filters for simple modes
        if spec.annotation_source == "expert_only":
            plan.add_where("l.consensus_id IS NULL")
        elif spec.annotation_source == "consensus_only":
            plan.add_where("l.consensus_id IS NOT NULL")

        # Filter by localization_type when specified
        if spec.localization_types:
            self.add_in_clause(plan, "l.localization_type", "localization_types", spec.localization_types)

        # Add GROUP BY image_id for aggregation (required for jsonb_agg)
        # Note: Other modules may add additional GROUP BY columns
        plan.add_group_by("i.image_id")

        # Add aggregated localization_annotations field
        self._add_localization_fields(plan, spec)

    def _add_simple_localization_join(
        self, plan: "QueryPlan", join_type: str, spec: "ExportSpec"
    ) -> None:
        """
        Add simple localization JOIN (for expert_only, consensus_only, or both modes).

        Args:
            plan: The QueryPlan to modify
            join_type: "LEFT" or "INNER"
            spec: The ExportSpec
        """
        plan.add_join(
            f"{join_type} JOIN localization_annotations l ON i.image_id = l.image_id"
        )

    def _add_prefer_consensus_subquery(
        self, plan: "QueryPlan", join_type: str, spec: "ExportSpec"
    ) -> None:
        """
        Add localization JOIN with prefer_consensus logic using subquery with window function.

        Uses a subquery with ROW_NUMBER() window function to select the best localization
        per image (consensus preferred, then expert, ordered by created_at DESC).

        Args:
            plan: The QueryPlan to modify
            join_type: "LEFT" or "INNER" (affects the join)
            spec: The ExportSpec
        """
        # Build WHERE clause for localization_type filter if specified
        where_clause = ""
        if spec.localization_types:
            param_placeholder = plan.add_param("localization_types_subq", spec.localization_types)
            where_clause = f"WHERE l_inner.localization_type = ANY({param_placeholder})"

        # Use window function to rank localizations per image
        # Prefer consensus over expert, then by created_at DESC
        subquery = (
            f"{join_type} JOIN ("
            "  SELECT l_inner.*, "
            "    ROW_NUMBER() OVER ("
            "      PARTITION BY l_inner.image_id "
            "      ORDER BY (l_inner.consensus_id IS NOT NULL) DESC, "
            "               l_inner.created_at DESC, l_inner.localization_id DESC"
            "    ) AS rn "
            "  FROM localization_annotations l_inner"
            f"  {where_clause}"
            ") l ON i.image_id = l.image_id AND l.rn = 1"
        )

        plan.add_join(subquery)

    def _add_localization_fields(self, plan: "QueryPlan", spec: "ExportSpec") -> None:
        """
        Add aggregated localization fields to SELECT using jsonb_agg().

        When detection_format='coco', includes category_id derived from
        detection_category_map.  Otherwise uses the standard nested format.

        Args:
            plan: The QueryPlan to modify
            spec: The ExportSpec
        """
        if spec.detection_format == "coco":
            self._add_coco_fields(plan, spec)
        else:
            # DISTINCT collapses duplicate localization objects produced when this join is
            # multiplied by another one-to-many task in the same query (see segmentation).
            loc_obj = (
                "jsonb_build_object("
                "  'localization_type', l.localization_type,"
                "  'target_structure', l.target_structure,"
                "  'coordinates', l.coordinates,"
                "  'lesion_subtype', l.lesion_subtype"
                ")"
            )
            plan.add_select(
                f"jsonb_agg(DISTINCT {loc_obj} ORDER BY {loc_obj}) "
                "FILTER (WHERE l.localization_id IS NOT NULL) AS localization_annotations"
            )

    def _add_coco_fields(self, plan: "QueryPlan", spec: "ExportSpec") -> None:
        """Add COCO-style bbox annotations with category_id mapping."""
        cat_map = spec.detection_category_map or {}

        # Build a SQL CASE expression for category_id mapping
        if cat_map:
            case_parts = []
            for structure, cat_id in cat_map.items():
                s_param = plan.add_param(f"coco_struct_{structure}", structure)
                id_param = plan.add_param(f"coco_catid_{structure}", cat_id)
                case_parts.append(f"WHEN l.target_structure = {s_param} THEN {id_param}")
            case_expr = "CASE " + " ".join(case_parts) + " ELSE 0 END"
        else:
            case_expr = "0"

        coco_obj = (
            "jsonb_build_object("
            "  'localization_type', l.localization_type,"
            "  'target_structure', l.target_structure,"
            "  'coordinates', l.coordinates,"
            "  'lesion_subtype', l.lesion_subtype,"
            f" 'category_id', {case_expr}"
            ")"
        )
        plan.add_select(
            f"jsonb_agg(DISTINCT {coco_obj} ORDER BY {coco_obj}) "
            "FILTER (WHERE l.localization_id IS NOT NULL) AS localization_annotations"
        )

    def get_output_fields(self) -> list[str]:
        """
        Get the list of output field names this module adds.

        Returns:
            List of field names: localization_annotations
        """
        return ["localization_annotations"]

    def get_primary_id_column(self) -> str | None:
        """
        Get the primary ID column for localization annotations.
        
        Returns:
            "l.localization_id" - used for require_annotations_mode="any" HAVING clause
        """
        return "l.localization_id"

    def apply_or_filters(
        self,
        plan: "QueryPlan",
        conditions: list[dict[str, Any]],
        task_name: str,
    ) -> None:
        """
        Apply OR-grouped filter conditions for localization.
        
        Each condition dict can contain:
        - localization_types: list of localization types
        
        The conditions are ORed together in a single WHERE clause.
        
        Args:
            plan: The QueryPlan to modify
            conditions: List of filter dicts to be ORed together
            task_name: Task name (for error messages)
        """
        if not conditions:
            return
        
        # Build OR groups
        or_groups = []
        for i, cond_dict in enumerate(conditions):
            # Handle localization_types
            if "localization_types" in cond_dict:
                loc_types = cond_dict["localization_types"]
                if loc_types:
                    ph = plan.add_param(f"or_loc_types_{i}", loc_types)
                    or_groups.append(f"l.localization_type = ANY({ph})")
        
        # Combine all OR groups
        if or_groups:
            # Keep NULL localizations (when using LEFT JOIN)
            or_clause = " OR ".join(or_groups)
            plan.add_where(f"(l.localization_id IS NULL OR ({or_clause}))")
