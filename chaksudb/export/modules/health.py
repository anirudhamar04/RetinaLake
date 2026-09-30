"""
HealthStatusModule: a cross-dataset normal/abnormal derived field.

Collapses every disease signal an image has — DR/DME/glaucoma/AMD grading, disease-concept
classification (binary / multi_label / multi_class), and explicit "normal" labels — into a
single ``health_status`` column:

  'abnormal'  positive disease evidence (any grade >= 1, any disease concept positive, or an
              explicit normal indicator turned off)
  'normal'    assessed for disease and nothing positive (incl. an explicit normal label on)
  NULL        the image was never assessed for disease (unknown — not assumed healthy)

This lets you export "all normal images" or "all diseased images" across datasets with one
field, regardless of how each dataset recorded disease. Uses a pre-aggregated per-image
subquery (bool_or) so it never multiplies the main query's rows.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chaksudb.export.query_builder import QueryPlan
    from chaksudb.export.spec import ExportSpec

from chaksudb.export.modules.base import BaseModule

# A row counts as positive disease evidence when:
#   - a normalized disease grade is >= 1, OR
#   - a disease-concept classification is positive (multi_class winner is a disease, or a
#     binary/multi_label disease row with class_index = 1), OR
#   - an explicit (binary/multi_label) "normal" indicator is OFF (class_index = 0).
_ABNORMAL_ROW = (
    "(dg.scaled_grade >= 1)"
    " OR (c.concept IS NOT NULL AND c.concept <> 'normal'"
    "     AND (c.task_type = 'multi_class' OR c.class_index = 1))"
    " OR (c.concept = 'normal' AND c.task_type <> 'multi_class' AND c.class_index = 0)"
)

# A row counts as a disease assessment when there's a grade or a concept-bearing
# classification (disease or normal). Meta/anatomical classifications (concept IS NULL) do
# not make an image "assessed for disease".
_ASSESSED_ROW = "(dg.grading_id IS NOT NULL) OR (c.concept IS NOT NULL)"


class HealthStatusModule(BaseModule):
    """Adds the ``health_status`` column and optional normal/abnormal filtering."""

    def apply(self, plan: "QueryPlan", spec: "ExportSpec") -> None:
        # Per-image health verdict. grading × classification inside the subquery may
        # multiply rows, but only bool_or aggregates are used, so the verdict is unaffected.
        plan.add_join(
            "LEFT JOIN ("
            "  SELECT img.image_id,"
            f"    CASE WHEN bool_or({_ABNORMAL_ROW}) THEN 'abnormal'"
            f"         WHEN bool_or({_ASSESSED_ROW}) THEN 'normal'"
            "          ELSE NULL END AS health_status"
            "  FROM images img"
            "  LEFT JOIN disease_grading dg ON dg.image_id = img.image_id"
            "  LEFT JOIN classification_annotations c ON c.image_id = img.image_id"
            "  GROUP BY img.image_id"
            ") health ON health.image_id = i.image_id"
        )
        plan.add_group_by("i.image_id")
        plan.add_select("MIN(health.health_status) AS health_status")

        # Optional filter: keep only normal or only abnormal images (NULL/unknown excluded).
        if spec.health_status_filter:
            ph = plan.add_param("health_status_filter", spec.health_status_filter)
            plan.add_where(f"health.health_status = {ph}")

    def get_output_fields(self) -> list[str]:
        return ["health_status"]
