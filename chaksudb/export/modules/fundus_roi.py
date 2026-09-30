"""
FundusROIModule: Flat fundus ROI circle columns + IQA quality threshold filter.

When spec.include_fundus_roi is True, adds:
  fundus_roi_cx, fundus_roi_cy, fundus_roi_radius, fundus_roi_method

When spec.iqa_min_quality_score or spec.iqa_quality_labels is set, filters
images by their QuickQual pseudo quality annotation (quality_type='overall',
scale_description LIKE 'QuickQual%').  Images with no IQA annotation are
excluded when a threshold is set.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chaksudb.export.query_builder import QueryPlan
    from chaksudb.export.spec import ExportSpec

from chaksudb.export.modules.base import BaseModule


class FundusROIModule(BaseModule):
    """
    Adds flat ROI circle columns and/or IQA quality filtering.

    ROI columns (when include_fundus_roi=True):
        fundus_roi_cx     — float, circle centre x in original image pixels
        fundus_roi_cy     — float, circle centre y in original image pixels
        fundus_roi_radius — float, circle radius in original image pixels
        fundus_roi_method — str, 'ransac' or 'fallback'

    IQA filter (when iqa_min_quality_score or iqa_quality_labels is set):
        Only images whose QuickQual overall quality_score >= threshold
        and/or quality_label is in the allowed label set are returned.
        Images with no IQA annotation are excluded.
    """

    def apply(self, plan: "QueryPlan", spec: "ExportSpec") -> None:
        if spec.include_fundus_roi:
            self._add_roi_columns(plan)

        if spec.iqa_min_quality_score is not None or spec.iqa_quality_labels:
            self._add_iqa_filter(plan, spec)

    def _add_roi_columns(self, plan: "QueryPlan") -> None:
        """LEFT JOIN LATERAL to pull the most-recent fundus_roi circle as flat columns."""
        plan.add_join(
            "LEFT JOIN LATERAL ("
            "  SELECT"
            "    (la.coordinates->>'center_x')::float AS fundus_roi_cx,"
            "    (la.coordinates->>'center_y')::float AS fundus_roi_cy,"
            "    (la.coordinates->>'radius')::float   AS fundus_roi_radius,"
            "    la.coordinates->>'method'            AS fundus_roi_method"
            "  FROM localization_annotations la"
            "  WHERE la.image_id = i.image_id"
            "    AND la.target_structure = 'fundus_roi'"
            "    AND la.localization_type = 'center_point'"
            "  ORDER BY la.created_at DESC"
            "  LIMIT 1"
            ") roi ON true"
        )
        # Register with group_by_expression so they remain valid when other modules add GROUP BY
        plan.add_select("roi.fundus_roi_cx", group_by_expression="roi.fundus_roi_cx")
        plan.add_select("roi.fundus_roi_cy", group_by_expression="roi.fundus_roi_cy")
        plan.add_select("roi.fundus_roi_radius", group_by_expression="roi.fundus_roi_radius")
        plan.add_select("roi.fundus_roi_method", group_by_expression="roi.fundus_roi_method")

    def _add_iqa_filter(self, plan: "QueryPlan", spec: "ExportSpec") -> None:
        """INNER JOIN filter subquery on quality_annotations for IQA scores."""
        conditions = ["qa_iqa.image_id = i.image_id",
                      "qa_iqa.quality_type = 'overall'",
                      "qa_iqa.scale_description LIKE 'QuickQual%%'"]

        if spec.iqa_min_quality_score is not None:
            ph = plan.add_param("iqa_min_score", spec.iqa_min_quality_score)
            conditions.append(f"qa_iqa.quality_score >= {ph}")

        if spec.iqa_quality_labels:
            ph = plan.add_param("iqa_labels", spec.iqa_quality_labels)
            conditions.append(f"qa_iqa.quality_label = ANY({ph})")

        where_clause = " AND ".join(conditions)
        plan.add_join(
            f"JOIN LATERAL ("
            f"  SELECT 1 FROM quality_annotations qa_iqa"
            f"  WHERE {where_clause}"
            f"  LIMIT 1"
            f") iqa_filter ON true"
        )

    def get_output_fields(self) -> list[str]:
        return ["fundus_roi_cx", "fundus_roi_cy", "fundus_roi_radius", "fundus_roi_method"]

    def get_primary_id_column(self) -> str | None:
        return None
