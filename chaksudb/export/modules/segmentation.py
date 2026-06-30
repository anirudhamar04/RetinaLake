"""
SegmentationModule: Segmentation mask annotations.

Adds segmentation_annotations table JOIN and aggregates segmentation masks
using jsonb_agg() to create a list of mask annotations per image.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from chaksudb.export.query_builder import QueryPlan
    from chaksudb.export.spec import ExportSpec

from chaksudb.export.modules.base import BaseModule


class SegmentationModule(BaseModule):
    """
    Module for adding segmentation mask annotations to export queries.

    This module adds a LEFT JOIN to segmentation_annotations table and aggregates
    segmentation data using jsonb_agg() to create a list of mask annotations per image.
    It handles consensus/expert source preference similar to grading.

    The JOIN type (LEFT vs INNER) depends on require_annotations_mode:
    - If require_annotations_mode is "none" or "any": LEFT JOIN (include images without segmentations)
    - If require_annotations_mode is "all": INNER JOIN (only include images with segmentations)

    Consensus preference modes:
    - expert_only: Only include expert annotations (consensus_id IS NULL)
    - consensus_only: Only include consensus annotations (consensus_id IS NOT NULL)
    - prefer_consensus: Prefer consensus, fallback to expert (uses subquery with window function)
    - both: Include both when available

    Output fields:
        - segmentation_masks: JSONB array of SegmentationMask objects, each containing:
            - segmentation_id: UUID
            - annotation_type_id: UUID
            - annotation_type: Optional string (from annotation_type table, e.g. 'vessel', 'optic_disc')
            - lesion_subtype: Optional string
            - mask_file_path: Optional string
            - confidence_score: Optional float
    """

    def apply(self, plan: "QueryPlan", spec: "ExportSpec") -> None:
        """
        Apply segmentation module to the query plan.

        Adds:
        - JOIN to segmentation_annotations table (LEFT or INNER based on require_annotations_mode)
        - JOIN to annotation_type table for annotation type information
        - Aggregated segmentation_masks field to SELECT using jsonb_agg()
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
            # Use subquery with window function to select best segmentation per image
            self._add_prefer_consensus_subquery(plan, join_type, spec)
        else:
            # For other modes, use direct JOIN with filters
            self._add_simple_segmentation_join(plan, join_type, spec)

        # Add JOIN to annotation_type for annotation type information
        plan.add_join(
            "LEFT JOIN annotation_type at ON s.annotation_type_id = at.annotation_type_id"
        )

        # Add consensus/expert filters for simple modes
        if spec.annotation_source == "expert_only":
            plan.add_where("s.consensus_id IS NULL")
        elif spec.annotation_source == "consensus_only":
            plan.add_where("s.consensus_id IS NOT NULL")

        # Filter by segmentation type and/or lesion subtype when specified
        self._add_segmentation_type_filters(plan, spec)

        # Add GROUP BY image_id for aggregation (required for jsonb_agg)
        # Note: Other modules may add additional GROUP BY columns
        plan.add_group_by("i.image_id")

        # Add aggregated segmentation_masks field
        self._add_segmentation_fields(plan, spec)

    def _add_simple_segmentation_join(
        self, plan: "QueryPlan", join_type: str, spec: "ExportSpec"
    ) -> None:
        """
        Add simple segmentation JOIN (for expert_only, consensus_only, or both modes).

        Args:
            plan: The QueryPlan to modify
            join_type: "LEFT" or "INNER"
            spec: The ExportSpec
        """
        plan.add_join(
            f"{join_type} JOIN segmentation_annotations s ON i.image_id = s.image_id"
        )

    def _add_prefer_consensus_subquery(
        self, plan: "QueryPlan", join_type: str, spec: "ExportSpec"
    ) -> None:
        """
        Add segmentation JOIN with prefer_consensus logic using subquery with window function.

        Uses a subquery with ROW_NUMBER() window function to select the best segmentation
        per (image, annotation_type) pair. Consensus annotations are preferred over expert
        annotations; ties are broken by created_at DESC.

        Partitioning by (image_id, annotation_type_id, lesion_subtype) ensures that all
        annotation types and lesion subtypes for an image are preserved (e.g. optic_disc,
        optic_cup, vessels, and MA/HE/EX/SE lesions) while still deduplicating consensus
        vs expert for the same type+subtype combination. Without lesion_subtype in the
        partition, all lesion subtypes sharing annotation_type="lesions" would collapse
        to a single row per image.

        Args:
            plan: The QueryPlan to modify
            join_type: "LEFT" or "INNER" (affects the join)
            spec: The ExportSpec
        """
        subquery = (
            f"{join_type} JOIN ("
            "  SELECT s_inner.*, "
            "    ROW_NUMBER() OVER ("
            "      PARTITION BY s_inner.image_id, s_inner.annotation_type_id,"
            "                   COALESCE(s_inner.lesion_subtype, '') "
            "      ORDER BY (s_inner.consensus_id IS NOT NULL) DESC, "
            "               s_inner.created_at DESC, s_inner.segmentation_id DESC"
            "    ) AS rn "
            "  FROM segmentation_annotations s_inner"
            ") s ON i.image_id = s.image_id AND s.rn = 1"
        )

        plan.add_join(subquery)

    def _add_segmentation_type_filters(self, plan: "QueryPlan", spec: "ExportSpec") -> None:
        """
        Add WHERE conditions for segmentation_types and/or lesion_subtypes when set.

        Keeps rows with no segmentation (s.segmentation_id IS NULL) so images without
        matching segmentations are still included when using LEFT JOIN; only
        segmentations that match the type/subtype lists are included in the aggregation.

        When require_annotations_mode="all" and segmentation_types is set, also adds a
        HAVING clause requiring every requested type to be present on the image — the
        INNER JOIN alone only guarantees at least one segmentation exists.
        """
        if not spec.segmentation_types and not spec.lesion_subtypes:
            return
        conditions = []
        if spec.segmentation_types:
            ph = plan.add_param("segmentation_types", spec.segmentation_types)
            conditions.append(f"at.annotation_type = ANY({ph})")
        if spec.lesion_subtypes:
            ph = plan.add_param("lesion_subtypes", spec.lesion_subtypes)
            conditions.append(f"s.lesion_subtype = ANY({ph})")
        combined = " AND ".join(conditions)
        plan.add_where(f"(s.segmentation_id IS NULL OR ({combined}))")

        if spec.require_annotations_mode == "all" and spec.segmentation_types:
            ph2 = plan.add_param("segmentation_types_all", spec.segmentation_types)
            plan.add_having(
                f"COUNT(DISTINCT CASE WHEN at.annotation_type = ANY({ph2})"
                f" THEN at.annotation_type END) = cardinality({ph2})"
            )

    def _add_segmentation_fields(self, plan: "QueryPlan", spec: "ExportSpec") -> None:
        """
        Add aggregated segmentation fields to SELECT using jsonb_agg().

        Aggregates segmentation annotations into a JSONB array, with each element
        containing segmentation_id, annotation_type_id, lesion_subtype, mask_file_path,
        and confidence_score.

        Args:
            plan: The QueryPlan to modify
            spec: The ExportSpec
        """
        # Aggregate segmentation masks using jsonb_agg()
        # Include only training-relevant fields (drop IDs like segmentation_id, annotation_type_id)
        # FILTER clause ensures NULL is returned (not empty array) when no segmentations exist
        # DISTINCT collapses the duplicate mask objects produced when this LEFT JOIN is
        # multiplied by another one-to-many task (grading/classification/localization) in
        # the same query — without it the mask array is repeated once per joined row.
        # The objects are byte-identical per real mask (mask_file_path is unique), so the
        # dedup is safe; ORDER BY the object keeps the array order deterministic (stable
        # channel assignment) and satisfies the DISTINCT-aggregate ordering rule.
        mask_obj = (
            "jsonb_build_object("
            "  'annotation_type', at.annotation_type,"
            "  'lesion_subtype', s.lesion_subtype,"
            "  'mask_file_path', s.mask_file_path,"
            "  'confidence_score', s.confidence_score,"
            "  'unified_format', s.unified_format"
            ")"
        )
        plan.add_select(
            f"jsonb_agg(DISTINCT {mask_obj} ORDER BY {mask_obj}) "
            "FILTER (WHERE s.segmentation_id IS NOT NULL) AS segmentation_masks"
        )

    def get_output_fields(self) -> list[str]:
        """
        Get the list of output field names this module adds.

        Returns:
            List of field names: segmentation_masks
        """
        return ["segmentation_masks"]

    def get_primary_id_column(self) -> str | None:
        """
        Get the primary ID column for segmentation annotations.
        
        Returns:
            "s.segmentation_id" - used for require_annotations_mode="any" HAVING clause
        """
        return "s.segmentation_id"

    def apply_or_filters(
        self,
        plan: "QueryPlan",
        conditions: list[dict[str, Any]],
        task_name: str,
    ) -> None:
        """
        Apply OR-grouped filter conditions for segmentation.
        
        Each condition dict can contain:
        - segmentation_types: list of annotation types
        - lesion_subtypes: list of lesion subtypes
        
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
            and_parts = []
            
            # Handle segmentation_types
            if "segmentation_types" in cond_dict:
                seg_types = cond_dict["segmentation_types"]
                if seg_types:
                    ph = plan.add_param(f"or_seg_types_{i}", seg_types)
                    and_parts.append(f"at.annotation_type = ANY({ph})")
            
            # Handle lesion_subtypes
            if "lesion_subtypes" in cond_dict:
                lesion_sub = cond_dict["lesion_subtypes"]
                if lesion_sub:
                    ph = plan.add_param(f"or_lesion_subtypes_{i}", lesion_sub)
                    and_parts.append(f"s.lesion_subtype = ANY({ph})")
            
            # Combine with AND within this condition
            if and_parts:
                or_groups.append("(" + " AND ".join(and_parts) + ")")
        
        # Combine all OR groups
        if or_groups:
            # Keep NULL segmentations (when using LEFT JOIN)
            or_clause = " OR ".join(or_groups)
            plan.add_where(f"(s.segmentation_id IS NULL OR ({or_clause}))")
