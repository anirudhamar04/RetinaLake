"""
Dataset-related database operations.
"""

from typing import Sequence

import psycopg

from chaksudb.db.connection import get_connection
from chaksudb.db.models import Dataset, DatasetSplit, ImageSplit
from chaksudb.db.queries.helpers import _prepare_upsert_query


async def _execute_upsert(
    conn: psycopg.AsyncConnection,
    table_name: str,
    columns: list[str],
    values: tuple,
    conflict_target: list[str],
    update_columns: list[str] | None = None,
) -> None:
    """Execute a single upsert operation."""
    query = _prepare_upsert_query(table_name, columns, conflict_target, update_columns)
    async with conn.cursor() as cur:
        await cur.execute(query, values)


async def _bulk_upsert(
    conn: psycopg.AsyncConnection,
    table_name: str,
    columns: list[str],
    values_list: Sequence[tuple],
    conflict_target: list[str],
    update_columns: list[str] | None = None,
    batch_size: int = 1000,
) -> int:
    """Execute bulk upsert operations in batches."""
    query = _prepare_upsert_query(table_name, columns, conflict_target, update_columns)
    total_rows = 0

    for i in range(0, len(values_list), batch_size):
        batch = values_list[i : i + batch_size]
        async with conn.cursor() as cur:
            await cur.executemany(query, batch)
            total_rows += len(batch)

    return total_rows


async def upsert_dataset(dataset: Dataset) -> None:
    """Upsert a dataset record."""
    async with get_connection() as conn:
        columns = [
            "dataset_id",
            "dataset_name",
            "source_url",
            "license",
            "modality_types",
            "created_at",
        ]
        values = (
            dataset.dataset_id,
            dataset.dataset_name,
            dataset.source_url,
            dataset.license,
            dataset.modality_types,
            dataset.created_at,
        )
        await _execute_upsert(
            conn, "datasets", columns, values, conflict_target=["dataset_id"]
        )


async def bulk_upsert_datasets(
    datasets: Sequence[Dataset], batch_size: int = 1000
) -> int:
    """Bulk upsert dataset records."""
    async with get_connection() as conn:
        columns = [
            "dataset_id",
            "dataset_name",
            "source_url",
            "license",
            "modality_types",
            "created_at",
        ]
        values_list = [
            (
                d.dataset_id,
                d.dataset_name,
                d.source_url,
                d.license,
                d.modality_types,
                d.created_at,
            )
            for d in datasets
        ]
        return await _bulk_upsert(
            conn, "datasets", columns, values_list, conflict_target=["dataset_id"], batch_size=batch_size
        )


async def upsert_dataset_split(split: DatasetSplit) -> None:
    """Upsert a dataset split record."""
    async with get_connection() as conn:
        columns = [
            "split_id",
            "dataset_id",
            "split_name",
            "split_type",
            "task_type",
            "image_count",
            "created_at",
        ]
        values = (
            split.split_id,
            split.dataset_id,
            split.split_name,
            split.split_type,
            split.task_type,
            split.image_count,
            split.created_at,
        )
        await _execute_upsert(
            conn,
            "dataset_splits",
            columns,
            values,
            conflict_target=["split_id"],
        )


async def upsert_image_split(image_split: ImageSplit) -> None:
    """Upsert an image split assignment record."""
    async with get_connection() as conn:
        columns = [
            "assignment_id",
            "image_id",
            "split_id",
            "task_type",
            "is_primary",
            "created_at",
        ]
        values = (
            image_split.assignment_id,
            image_split.image_id,
            image_split.split_id,
            image_split.task_type,
            image_split.is_primary,
            image_split.created_at,
        )
        await _execute_upsert(
            conn,
            "image_splits",
            columns,
            values,
            conflict_target=["assignment_id"],  # Must use PK because unique constraint has nullable column
        )
