"""
ClinicalModule: Clinical description annotations.

Adds clinical_descriptions table JOIN and includes clinical description fields.
Handles selection of primary description when multiple descriptions exist per image.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chaksudb.export.query_builder import QueryPlan
    from chaksudb.export.spec import ExportSpec

from chaksudb.export.modules.base import BaseModule


class ClinicalModule(BaseModule):
    """
    Module for adding clinical description annotations to export queries.

    This module adds a JOIN to clinical_descriptions table and includes
    clinical description fields in the output. When multiple descriptions
    exist per image, it selects the primary one based on description_type
    preference (diagnosis_text > clinical_caption > notes).

    The JOIN type (LEFT vs INNER) depends on require_annotations_mode:
    - If require_annotations_mode is "none" or "any": LEFT JOIN (include images without descriptions)
    - If require_annotations_mode is "all": INNER JOIN (only include images with descriptions)

    Output fields:
        - clinical_description_text: Primary clinical description text
        - clinical_description_type: Type of the selected description
        - clinical_word_count: Word count of the selected description
    """

    def apply(self, plan: "QueryPlan", spec: "ExportSpec") -> None:
        """
        Apply clinical module to the query plan.

        Adds:
        - JOIN to clinical_descriptions table (LEFT or INNER based on require_annotations_mode)
        - Clinical description fields to SELECT
        - Uses subquery with window function to select primary description per image

        Args:
            plan: The QueryPlan to modify
            spec: The ExportSpec containing user requirements
        """
        # Determine JOIN type based on require_annotations_mode
        join_type = "INNER" if spec.require_annotations_mode == "all" else "LEFT"

        # Use subquery with window function to select primary description per image
        # Preference order: diagnosis_text > clinical_caption > notes
        # Then by created_at DESC for tie-breaking
        subquery = (
            f"{join_type} JOIN ("
            "  SELECT cd_inner.*, "
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
            ") cd ON i.image_id = cd.image_id AND cd.rn = 1"
        )

        plan.add_join(subquery)

        # Add clinical description fields to SELECT
        plan.add_select(
            "cd.description_text AS clinical_description_text",
            group_by_expression="cd.description_text",
        )
        plan.add_select(
            "cd.description_type AS clinical_description_type",
            group_by_expression="cd.description_type",
        )
        plan.add_select(
            "cd.word_count AS clinical_word_count",
            group_by_expression="cd.word_count",
        )

    def get_output_fields(self) -> list[str]:
        """
        Get the list of output field names this module adds.

        Returns:
            List of field names: clinical_description_text, clinical_description_type,
            clinical_word_count
        """
        return [
            "clinical_description_text",
            "clinical_description_type",
            "clinical_word_count",
        ]

    def get_primary_id_column(self) -> str | None:
        """
        Get the primary ID column for clinical descriptions.
        
        Returns:
            "cd.description_id" - used for require_annotations_mode="any" HAVING clause
        """
        return "cd.description_id"
