"""
Dataset-faithful export: build an ExportSpec that flattens *everything* a dataset has.

`build_dataset_spec(["BRSET"])` introspects which annotation families the dataset actually
contains (grading, the full classification panel, quality params, segmentation, localization,
patient data) and returns a spec whose columns mirror the source — e.g. BRSET comes out
looking like labels_brset.csv, FIVES comes out with its disease multi_class columns plus its
segmentation masks. Useful for single-dataset testing without hand-listing every label.

    spec = await build_dataset_spec(["BRSET"])
    export(spec, parquet_path="brset.parquet")
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from chaksudb.db.connection import get_connection
from chaksudb.db.queries.annotation_types import (
    list_classification_tasks,
    list_quality_types,
)
from chaksudb.export.spec import ExportSpec


async def _distinct(conn, sql: str, dataset_names: Sequence[str]) -> list:
    async with conn.cursor() as cur:
        await cur.execute(sql, (list(dataset_names),))
        return [r[0] for r in await cur.fetchall() if r[0] is not None]


async def build_dataset_spec(
    dataset_names: Sequence[str],
    *,
    modalities: Optional[list[str]] = None,
    **overrides: Any,
) -> ExportSpec:
    """Introspect the given dataset(s) and return a spec that flattens all their labels.

    Args:
        dataset_names: dataset name(s) to export.
        modalities: optional modality filter.
        **overrides: any ExportSpec field to override the discovered defaults
            (e.g. split_names=["test"], require_annotations_mode="any").
    """
    names = list(dataset_names)

    async with get_connection() as conn:
        disease_types = await _distinct(
            conn,
            "SELECT DISTINCT g.disease_type FROM disease_grading g "
            "JOIN images i ON g.image_id = i.image_id "
            "JOIN datasets d ON i.dataset_id = d.dataset_id "
            "WHERE d.dataset_name = ANY(%s)",
            names,
        )
        segmentation_types = await _distinct(
            conn,
            "SELECT DISTINCT at.annotation_type FROM segmentation_annotations s "
            "JOIN annotation_type at ON s.annotation_type_id = at.annotation_type_id "
            "JOIN images i ON s.image_id = i.image_id "
            "JOIN datasets d ON i.dataset_id = d.dataset_id "
            "WHERE d.dataset_name = ANY(%s)",
            names,
        )
        localization_types = await _distinct(
            conn,
            "SELECT DISTINCT l.localization_type FROM localization_annotations l "
            "JOIN images i ON l.image_id = i.image_id "
            "JOIN datasets d ON i.dataset_id = d.dataset_id "
            "WHERE d.dataset_name = ANY(%s)",
            names,
        )

        async def _exists(table: str) -> bool:
            rows = await _distinct(
                conn,
                f"SELECT 1 FROM {table} t "
                "JOIN images i ON t.image_id = i.image_id "
                "JOIN datasets d ON i.dataset_id = d.dataset_id "
                "WHERE d.dataset_name = ANY(%s) LIMIT 1",
                names,
            )
            return bool(rows)

        has_quality = await _exists("quality_annotations")
        quality_types = await list_quality_types(names) if has_quality else []
        has_keywords = await _exists("keyword_annotations")
        has_clinical = await _exists("clinical_descriptions")
        has_patients = await _exists("patient_images")

    # Classification: discover every task and flatten it (binary/multi_class -> label
    # columns; multi_label -> one column per sub_key, exactly like the source CSV).
    cls_tasks = await list_classification_tasks(names)
    classification_class_names = [t["task_name"] for t in cls_tasks] or None
    classification_task_types = {t["task_name"]: t["task_type"] for t in cls_tasks} or None
    multi_label_keys = {
        t["task_name"]: list(t["sub_keys"])
        for t in cls_tasks
        if t["task_type"] == "multi_label" and t["sub_keys"]
    } or None

    annotation_tasks: list[str] = []
    if disease_types:
        annotation_tasks.append("grading")
    if cls_tasks:
        annotation_tasks.append("classification")
    if segmentation_types:
        annotation_tasks.append("segmentation")
    if localization_types:
        annotation_tasks.append("localization")
    if has_quality:
        annotation_tasks.append("quality")
    if has_keywords:
        annotation_tasks.append("keyword")
    if has_clinical:
        annotation_tasks.append("description")

    fields: dict[str, Any] = dict(
        dataset_names=names,
        modalities=modalities,
        annotation_tasks=annotation_tasks or None,
        disease_types=disease_types or None,
        classification_class_names=classification_class_names,
        classification_task_types=classification_task_types,
        multi_label_keys=multi_label_keys,
        segmentation_types=segmentation_types or None,
        localization_types=localization_types or None,
        quality_types=quality_types or None,
        include_patient_data=has_patients,
        include_original_grade=True,  # keep the dataset's native grade alongside scaled
    )
    fields.update(overrides)
    return ExportSpec(**fields)
