"""
PatientModule: Patient demographics and bilateral image linking.

Adds patient_images and patients table JOINs to include patient demographics
(age, sex, ethnicity, comorbidities) and link bilateral image pairs.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chaksudb.export.query_builder import QueryPlan
    from chaksudb.export.spec import ExportSpec

from chaksudb.export.modules.base import BaseModule


class PatientModule(BaseModule):
    """
    Module for adding patient demographics to export queries.

    Joins through patient_images to patients to expose patient-level fields.
    Always uses LEFT JOIN so images without patient data are still included.

    Output fields:
        - patient_id: UUID of the patient
        - original_patient_id: Original patient identifier from the source dataset
        - age: Patient age (integer, nullable)
        - sex: Patient sex ('male', 'female', 'unknown', nullable)
        - ethnicity: Patient ethnicity (text, nullable)
        - comorbidities: Patient comorbidities (JSONB, nullable)
    """

    def apply(self, plan: "QueryPlan", spec: "ExportSpec") -> None:
        """
        Apply patient module to the query plan.

        Adds LEFT JOINs to patient_images and patients tables, plus
        patient demographic fields to SELECT.

        Args:
            plan: The QueryPlan to modify
            spec: The ExportSpec containing user requirements
        """
        plan.add_join(
            "LEFT JOIN patient_images pi ON i.image_id = pi.image_id"
        )
        plan.add_join(
            "LEFT JOIN patients p ON pi.patient_id = p.patient_id"
        )

        plan.add_select("p.patient_id", group_by_expression="p.patient_id")
        plan.add_select(
            "p.original_patient_id",
            group_by_expression="p.original_patient_id",
        )
        plan.add_select("p.age", group_by_expression="p.age")
        plan.add_select("p.sex", group_by_expression="p.sex")
        plan.add_select("p.ethnicity", group_by_expression="p.ethnicity")
        plan.add_select("p.comorbidities", group_by_expression="p.comorbidities")

    def get_output_fields(self) -> list[str]:
        return [
            "patient_id",
            "original_patient_id",
            "age",
            "sex",
            "ethnicity",
            "comorbidities",
        ]
