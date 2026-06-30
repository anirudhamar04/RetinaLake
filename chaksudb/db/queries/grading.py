"""
Grading scale and disease grading database operations.
"""

import uuid
from typing import Optional, Sequence

import psycopg
from psycopg.rows import dict_row

from chaksudb.db.connection import get_connection
from chaksudb.db.models import DiseaseGrading, GradingScale, GradingScaleMapping
from chaksudb.db.queries.helpers import _prepare_upsert_query, _serialize_jsonb


async def upsert_grading_scale(grading_scale: GradingScale) -> None:
    """Upsert a grading scale record."""
    async with get_connection() as conn:
        columns = [
            "scale_id",
            "scale_name",
            "disease_type",
            "scale_description",
            "min_value",
            "max_value",
            "value_labels",
        ]
        values = (
            grading_scale.scale_id,
            grading_scale.scale_name,
            grading_scale.disease_type,
            grading_scale.scale_description,
            grading_scale.min_value,
            grading_scale.max_value,
            _serialize_jsonb(grading_scale.value_labels),
        )
        query = _prepare_upsert_query("grading_scales", columns, conflict_target=["scale_id"])
        async with conn.cursor() as cur:
            await cur.execute(query, values)


async def upsert_grading_scale_mapping(mapping: GradingScaleMapping) -> None:
    """Upsert a grading scale mapping record."""
    async with get_connection() as conn:
        columns = [
            "mapping_id",
            "source_scale_id",
            "target_scale_id",
            "source_value",
            "target_value",
            "mapping_confidence",
        ]
        values = (
            mapping.mapping_id,
            mapping.source_scale_id,
            mapping.target_scale_id,
            mapping.source_value,
            mapping.target_value,
            mapping.mapping_confidence,
        )
        query = _prepare_upsert_query(
            "grading_scale_mappings",
            columns,
            conflict_target=["mapping_id"],
        )
        async with conn.cursor() as cur:
            await cur.execute(query, values)


async def upsert_disease_grading(grading: DiseaseGrading) -> None:
    """Upsert a disease grading record."""
    async with get_connection() as conn:
        columns = [
            "grading_id",
            "image_id",
            "disease_type",
            "scale_id",
            "original_grade",
            "scaled_grade",
            "grade_label",
            "raw_data_id",
            "expert_annotation_id",
            "consensus_id",
            "annotation_method",
            "confidence_score",
            "provenance_chain_id",
            "created_at",
            "updated_at",
        ]
        values = (
            grading.grading_id,
            grading.image_id,
            grading.disease_type,
            grading.scale_id,
            grading.original_grade,
            grading.scaled_grade,
            grading.grade_label,
            grading.raw_data_id,
            grading.expert_annotation_id,
            grading.consensus_id,
            grading.annotation_method,
            grading.confidence_score,
            grading.provenance_chain_id,
            grading.created_at,
            grading.updated_at,
        )
        query = _prepare_upsert_query(
            "disease_grading",
            columns,
            conflict_target=["grading_id"],
        )
        async with conn.cursor() as cur:
            await cur.execute(query, values)


async def bulk_upsert_disease_gradings(
    gradings: Sequence[DiseaseGrading], batch_size: int = 1000
) -> int:
    """Bulk upsert disease grading records."""
    async with get_connection() as conn:
        columns = [
            "grading_id",
            "image_id",
            "disease_type",
            "scale_id",
            "original_grade",
            "scaled_grade",
            "grade_label",
            "raw_data_id",
            "expert_annotation_id",
            "consensus_id",
            "annotation_method",
            "confidence_score",
            "provenance_chain_id",
            "created_at",
            "updated_at",
        ]
        values_list = [
            (
                g.grading_id,
                g.image_id,
                g.disease_type,
                g.scale_id,
                g.original_grade,
                g.scaled_grade,
                g.grade_label,
                g.raw_data_id,
                g.expert_annotation_id,
                g.consensus_id,
                g.annotation_method,
                g.confidence_score,
                g.provenance_chain_id,
                g.created_at,
                g.updated_at,
            )
            for g in gradings
        ]
        
        query = _prepare_upsert_query(
            "disease_grading",
            columns,
            conflict_target=["grading_id"],
        )
        
        total_rows = 0
        for i in range(0, len(values_list), batch_size):
            batch = values_list[i : i + batch_size]
            async with conn.cursor() as cur:
                await cur.executemany(query, batch)
                total_rows += len(batch)
        
        return total_rows


async def find_grading_scale_mapping_to_standard(
    source_scale_id: uuid.UUID,
    source_value: str,
    target_scale_name: Optional[str] = None,
) -> Optional[GradingScaleMapping]:
    """
    Find a mapping from a source scale value to a standard scale.

    If target_scale_name is not provided, searches for mappings to any standard scale.
    Standard scales are: ETDRS, ICDR, AAO.

    Args:
        source_scale_id: UUID of the source grading scale
        source_value: Original grade value (string)
        target_scale_name: Optional target scale name to search for (if None, searches all standard scales)

    Returns:
        GradingScaleMapping if found, None otherwise
    """
    async with get_connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            if target_scale_name:
                # Search for specific target scale
                await cur.execute(
                    """
                    SELECT m.mapping_id, m.source_scale_id, m.target_scale_id, 
                           m.source_value, m.target_value, m.mapping_confidence,
                           t.scale_name as target_scale_name
                    FROM grading_scale_mappings m
                    JOIN grading_scales t ON m.target_scale_id = t.scale_id
                    WHERE m.source_scale_id = %s 
                      AND m.source_value = %s
                      AND t.scale_name = %s
                    LIMIT 1
                    """,
                    (source_scale_id, source_value, target_scale_name),
                )
            else:
                # Search for any standard scale
                await cur.execute(
                    """
                    SELECT m.mapping_id, m.source_scale_id, m.target_scale_id, 
                           m.source_value, m.target_value, m.mapping_confidence,
                           t.scale_name as target_scale_name
                    FROM grading_scale_mappings m
                    JOIN grading_scales t ON m.target_scale_id = t.scale_id
                    WHERE m.source_scale_id = %s 
                      AND m.source_value = %s
                      AND t.scale_name IN ('ETDRS', 'ICDR', 'AAO')
                    ORDER BY 
                        CASE t.scale_name
                            WHEN 'ETDRS' THEN 1
                            WHEN 'ICDR' THEN 2
                            WHEN 'AAO' THEN 3
                            ELSE 4
                        END
                    LIMIT 1
                    """,
                    (source_scale_id, source_value),
                )

            row = await cur.fetchone()
            if row:
                return GradingScaleMapping(
                    mapping_id=row["mapping_id"],
                    source_scale_id=row["source_scale_id"],
                    target_scale_id=row["target_scale_id"],
                    source_value=row["source_value"],
                    target_value=row["target_value"],
                    mapping_confidence=row["mapping_confidence"],
                )
            return None


async def check_scale_mapping_exists(
    scale_id: uuid.UUID,
    target_scale_id: uuid.UUID,
) -> bool:
    """
    Check if a mapping exists between two grading scales.

    Args:
        scale_id: Source scale UUID
        target_scale_id: Target scale UUID

    Returns:
        True if mapping exists, False otherwise
    """
    # If same scale, no mapping needed
    if scale_id == target_scale_id:
        return True

    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT EXISTS(
                    SELECT 1
                    FROM grading_scale_mappings
                    WHERE source_scale_id = %s
                      AND target_scale_id = %s
                    LIMIT 1
                )
                """,
                (scale_id, target_scale_id),
            )
            result = await cur.fetchone()
            return result[0] if result else False


async def find_grading_scale_by_id(scale_id: uuid.UUID) -> Optional[GradingScale]:
    """
    Find a grading scale by its UUID.

    Args:
        scale_id: UUID of the grading scale

    Returns:
        GradingScale if found, None otherwise
    """
    async with get_connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """
                SELECT scale_id, scale_name, disease_type, scale_description,
                       min_value, max_value, value_labels
                FROM grading_scales
                WHERE scale_id = %s
                """,
                (scale_id,),
            )
            row = await cur.fetchone()
            if row:
                return GradingScale(
                    scale_id=row["scale_id"],
                    scale_name=row["scale_name"],
                    disease_type=row["disease_type"],
                    scale_description=row["scale_description"],
                    min_value=row["min_value"],
                    max_value=row["max_value"],
                    value_labels=row["value_labels"],
                )
            return None


async def get_all_disease_gradings_with_original_grade() -> list[dict]:
    """
    Get all disease_grading records that have original_grade.

    Returns:
        List of dictionaries containing all grading fields
    """
    async with get_connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """
                SELECT grading_id, image_id, disease_type, scale_id, original_grade, scaled_grade,
                       grade_label, raw_data_id, expert_annotation_id, consensus_id,
                       annotation_method, confidence_score, provenance_chain_id, created_at, updated_at
                FROM disease_grading
                WHERE original_grade IS NOT NULL
                """
            )
            return await cur.fetchall()
