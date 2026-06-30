"""
Expert-related database operations.
"""

import psycopg

from chaksudb.db.connection import get_connection
from chaksudb.db.models import Expert, ExpertAnnotation
from chaksudb.db.queries.helpers import _prepare_upsert_query, _serialize_jsonb


async def upsert_expert(expert: Expert) -> None:
    """Upsert an expert record."""
    async with get_connection() as conn:
        columns = [
            "expert_id",
            "expert_name",
            "expertise_area",
            "dataset_id",
            "model_id",
            "created_at",
        ]
        values = (
            expert.expert_id,
            expert.expert_name,
            expert.expertise_area,
            expert.dataset_id,
            expert.model_id,
            expert.created_at,
        )
        query = _prepare_upsert_query("experts", columns, conflict_target=["expert_id"])
        async with conn.cursor() as cur:
            await cur.execute(query, values)


async def upsert_expert_annotation(expert_annotation: ExpertAnnotation) -> None:
    """Upsert an expert annotation record."""
    async with get_connection() as conn:
        columns = [
            "expert_annotation_id",
            "expert_id",
            "annotation_task",
            "raw_data_id",
            "annotation_value",
            "confidence_level",
            "annotation_timestamp",
            "created_at",
        ]
        values = (
            expert_annotation.expert_annotation_id,
            expert_annotation.expert_id,
            expert_annotation.annotation_task,
            expert_annotation.raw_data_id,
            _serialize_jsonb(expert_annotation.annotation_value),
            expert_annotation.confidence_level,
            expert_annotation.annotation_timestamp,
            expert_annotation.created_at,
        )
        query = _prepare_upsert_query(
            "expert_annotations",
            columns,
            conflict_target=["expert_annotation_id"],
        )
        async with conn.cursor() as cur:
            await cur.execute(query, values)


async def bulk_upsert_expert_annotations(
    expert_annotations: list[ExpertAnnotation], batch_size: int = 1000
) -> int:
    """Bulk upsert expert annotation records."""
    async with get_connection() as conn:
        columns = [
            "expert_annotation_id",
            "expert_id",
            "annotation_task",
            "raw_data_id",
            "annotation_value",
            "confidence_level",
            "annotation_timestamp",
            "created_at",
        ]
        values_list = [
            (
                ea.expert_annotation_id,
                ea.expert_id,
                ea.annotation_task,
                ea.raw_data_id,
                _serialize_jsonb(ea.annotation_value),
                ea.confidence_level,
                ea.annotation_timestamp,
                ea.created_at,
            )
            for ea in expert_annotations
        ]
        
        query = _prepare_upsert_query(
            "expert_annotations",
            columns,
            conflict_target=["expert_annotation_id"],
        )
        
        total_rows = 0
        for i in range(0, len(values_list), batch_size):
            batch = values_list[i : i + batch_size]
            async with conn.cursor() as cur:
                await cur.executemany(query, batch)
                total_rows += len(batch)
        
        return total_rows
