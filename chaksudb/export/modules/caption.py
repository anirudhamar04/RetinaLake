"""
CaptionModule: Synthesised text captions for VLM / captioning tasks.

Pulls raw text from clinical_descriptions, keyword_vocabulary,
disease_grading, classification_annotations, localization_annotations,
and segmentation_annotations, then exposes them as SQL columns that
CaptionEngine synthesizes into final captions in the export pipeline.

The heavy caption synthesis (template formatting, dictionary look-ups)
happens *after* the SQL query via CaptionEngine, not inside SQL.
This module's job is to ensure the required source columns are present
in the query output.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chaksudb.export.query_builder import QueryPlan
    from chaksudb.export.spec import ExportSpec

from chaksudb.export.modules.base import BaseModule


class CaptionModule(BaseModule):
    """
    Module that adds caption-source columns to the query.

    Depending on ``spec.caption_mode`` it joins the relevant tables:

    - ``"clinical"``: clinical_descriptions text
    - ``"keyword"``: aggregated keyword terms
    - ``"grading"``: disease_grading labels (original_grade, disease_type)
    - ``"classification"``: classification_annotations class_name + class_value
    - ``"synthetic"``: grading + classification + localization structures + segmentation structures
    - ``"all"`` (default): all of the above

    The actual caption string is synthesised post-query by CaptionEngine,
    which enriches labels using the definitions dictionary.

    Output fields (per mode):
        - caption_clinical_text: Text from clinical_descriptions (nullable)
        - caption_keywords: Array of keyword terms (nullable)
        - caption_grade_data: JSONB array of {disease_type, original_grade, grade_label} (nullable)
        - caption_class_data: JSONB object of {class_name: class_value} (nullable)
        - caption_loc_structures: Text array of distinct target_structure values (nullable)
        - caption_seg_structures: Text array of distinct annotation type / lesion subtype values (nullable)
    """

    def apply(self, plan: "QueryPlan", spec: "ExportSpec") -> None:
        mode = spec.caption_mode or "all"

        if mode in ("clinical", "all"):
            self._add_clinical_caption(plan, spec)

        if mode in ("keyword", "all"):
            self._add_keyword_caption(plan, spec)

        if mode in ("grading", "synthetic", "all"):
            self._add_grading_caption(plan, spec)

        if mode in ("classification", "synthetic", "all"):
            self._add_classification_caption(plan, spec)

        if mode in ("synthetic", "all"):
            self._add_localization_structures(plan, spec)
            self._add_segmentation_structures(plan, spec)

    # ------------------------------------------------------------------
    # Clinical / keyword (existing)
    # ------------------------------------------------------------------

    def _add_clinical_caption(self, plan: "QueryPlan", spec: "ExportSpec") -> None:
        """Join clinical_descriptions and expose the best description."""
        subquery = (
            "LEFT JOIN ("
            "  SELECT cd_inner.image_id, cd_inner.description_text, cd_inner.description_type, "
            "    ROW_NUMBER() OVER ("
            "      PARTITION BY cd_inner.image_id "
            "      ORDER BY "
            "        CASE cd_inner.description_type "
            "          WHEN 'diagnosis_text' THEN 1 "
            "          WHEN 'clinical_caption' THEN 2 "
            "          WHEN 'notes' THEN 3 "
            "          ELSE 4 "
            "        END, "
            "        cd_inner.created_at DESC"
            "    ) AS rn "
            "  FROM clinical_descriptions cd_inner"
            ") cap_cd ON i.image_id = cap_cd.image_id AND cap_cd.rn = 1"
        )
        plan.add_join(subquery)
        plan.add_select(
            "cap_cd.description_text AS caption_clinical_text",
            group_by_expression="cap_cd.description_text",
        )

    def _add_keyword_caption(self, plan: "QueryPlan", spec: "ExportSpec") -> None:
        """Join keyword tables and aggregate terms."""
        plan.add_join(
            "LEFT JOIN keyword_annotations cap_ka ON i.image_id = cap_ka.image_id"
        )
        plan.add_join(
            "LEFT JOIN keyword_vocabulary cap_kv ON cap_ka.keyword_id = cap_kv.keyword_id"
        )
        plan.add_group_by("i.image_id")
        plan.add_select(
            "array_agg(DISTINCT cap_kv.keyword_term) "
            "FILTER (WHERE cap_kv.keyword_term IS NOT NULL) AS caption_keywords"
        )

    # ------------------------------------------------------------------
    # New: grading / classification / structures
    # ------------------------------------------------------------------

    def _add_grading_caption(self, plan: "QueryPlan", spec: "ExportSpec") -> None:
        """Pull all disease grading labels as a JSONB array for caption synthesis."""
        plan.add_join(
            "LEFT JOIN ("
            "  SELECT g_cap.image_id,"
            "    jsonb_agg(jsonb_build_object("
            "      'disease_type', g_cap.disease_type,"
            "      'original_grade', g_cap.original_grade,"
            "      'grade_label', g_cap.grade_label"
            "    )) AS caption_grade_data"
            "  FROM disease_grading g_cap"
            "  GROUP BY g_cap.image_id"
            ") cap_grading ON i.image_id = cap_grading.image_id"
        )
        plan.add_group_by("i.image_id")
        plan.add_select(
            "MIN(cap_grading.caption_grade_data::text)::jsonb AS caption_grade_data"
        )

    def _add_classification_caption(self, plan: "QueryPlan", spec: "ExportSpec") -> None:
        """Pull class names and labels as a JSONB object for caption synthesis.

        Uses scalar class_index/class_label columns. For multi-label (exploded
        rows), keys become class_name_subkey so the CaptionEngine sees flat
        entries it already knows how to parse.
        """
        plan.add_join(
            "LEFT JOIN ("
            "  SELECT ca_cap.image_id,"
            "    jsonb_object_agg("
            "      CASE WHEN ca_cap.sub_key IS NOT NULL"
            "        THEN ca_cap.class_name || '_' || ca_cap.sub_key"
            "        ELSE ca_cap.class_name"
            "      END,"
            "      CASE"
            "        WHEN ca_cap.task_type = 'binary' THEN to_jsonb(ca_cap.class_index = 1)"
            "        WHEN ca_cap.task_type = 'multi_class' THEN to_jsonb(ca_cap.class_label)"
            "        WHEN ca_cap.task_type = 'multi_label' THEN to_jsonb(ca_cap.class_index = 1)"
            "      END"
            "    ) FILTER (WHERE ca_cap.class_name IS NOT NULL) AS caption_class_data"
            "  FROM classification_annotations ca_cap"
            "  GROUP BY ca_cap.image_id"
            ") cap_classification ON i.image_id = cap_classification.image_id"
        )
        plan.add_group_by("i.image_id")
        plan.add_select(
            "MIN(cap_classification.caption_class_data::text)::jsonb AS caption_class_data"
        )

    def _add_localization_structures(self, plan: "QueryPlan", spec: "ExportSpec") -> None:
        """Pull distinct target structures and lesion subtypes from localizations."""
        plan.add_join(
            "LEFT JOIN ("
            "  SELECT l_cap.image_id,"
            "    array_agg(DISTINCT COALESCE(l_cap.lesion_subtype, l_cap.target_structure))"
            "    FILTER (WHERE COALESCE(l_cap.lesion_subtype, l_cap.target_structure) IS NOT NULL)"
            "    AS caption_loc_structures"
            "  FROM localization_annotations l_cap"
            "  GROUP BY l_cap.image_id"
            ") cap_loc ON i.image_id = cap_loc.image_id"
        )
        plan.add_group_by("i.image_id")
        plan.add_select(
            "MIN(ARRAY_TO_STRING(cap_loc.caption_loc_structures, ','))::text AS caption_loc_structures"
        )

    def _add_segmentation_structures(self, plan: "QueryPlan", spec: "ExportSpec") -> None:
        """Pull distinct annotation types / lesion subtypes from segmentation masks."""
        plan.add_join(
            "LEFT JOIN ("
            "  SELECT sa_cap.image_id,"
            "    array_agg(DISTINCT COALESCE(sa_cap.lesion_subtype, at_cap.annotation_type))"
            "    FILTER (WHERE COALESCE(sa_cap.lesion_subtype, at_cap.annotation_type) IS NOT NULL)"
            "    AS caption_seg_structures"
            "  FROM segmentation_annotations sa_cap"
            "  LEFT JOIN annotation_type at_cap ON sa_cap.annotation_type_id = at_cap.annotation_type_id"
            "  GROUP BY sa_cap.image_id"
            ") cap_seg ON i.image_id = cap_seg.image_id"
        )
        plan.add_group_by("i.image_id")
        plan.add_select(
            "MIN(ARRAY_TO_STRING(cap_seg.caption_seg_structures, ','))::text AS caption_seg_structures"
        )

    # ------------------------------------------------------------------

    def get_output_fields(self) -> list[str]:
        return [
            "caption_clinical_text",
            "caption_keywords",
            "caption_grade_data",
            "caption_class_data",
            "caption_loc_structures",
            "caption_seg_structures",
        ]
