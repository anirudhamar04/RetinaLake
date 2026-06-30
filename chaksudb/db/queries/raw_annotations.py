"""
Raw annotation file database operations.
"""

import uuid
from typing import Optional, Sequence

import psycopg

from chaksudb.db.connection import get_connection
from chaksudb.db.models import RawAnnotationFile
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


async def get_raw_annotation_file(raw_file_id: uuid.UUID) -> Optional[RawAnnotationFile]:
    """
    Get a raw annotation file record by ID.
    
    Args:
        raw_file_id: UUID of the raw annotation file
        
    Returns:
        RawAnnotationFile model if found, None otherwise
    """
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT
                    raw_file_id, dataset_id, storage_provider, bucket, object_key,
                    version_id, file_path, file_type, file_name, file_hash,
                    file_size, encoding, parsed_status, parse_errors,
                    created_at, updated_at
                FROM raw_annotation_files
                WHERE raw_file_id = %s
                """,
                (raw_file_id,)
            )
            row = await cur.fetchone()
            
            if row is None:
                return None
            
            return RawAnnotationFile(
                raw_file_id=row[0],
                dataset_id=row[1],
                storage_provider=row[2],
                bucket=row[3],
                object_key=row[4],
                version_id=row[5],
                file_path=row[6],
                file_type=row[7],
                file_name=row[8],
                file_hash=row[9],
                file_size=row[10],
                encoding=row[11],
                parsed_status=row[12],
                parse_errors=row[13],
                created_at=row[14],
                updated_at=row[15],
            )


async def upsert_raw_annotation_file(raw_file: RawAnnotationFile) -> None:
    """Upsert a raw annotation file record."""
    async with get_connection() as conn:
        columns = [
            "raw_file_id",
            "dataset_id",
            "storage_provider",
            "bucket",
            "object_key",
            "version_id",
            "file_path",
            "file_type",
            "file_name",
            "file_hash",
            "file_size",
            "encoding",
            "parsed_status",
            "parse_errors",
            "created_at",
            "updated_at",
        ]
        values = (
            raw_file.raw_file_id,
            raw_file.dataset_id,
            raw_file.storage_provider,
            raw_file.bucket,
            raw_file.object_key,
            raw_file.version_id,
            raw_file.file_path,
            raw_file.file_type,
            raw_file.file_name,
            raw_file.file_hash,
            raw_file.file_size,
            raw_file.encoding,
            raw_file.parsed_status,
            raw_file.parse_errors,
            raw_file.created_at,
            raw_file.updated_at,
        )
        # Use primary key for conflict detection (works with deterministic UUIDs)
        await _execute_upsert(
            conn,
            "raw_annotation_files",
            columns,
            values,
            conflict_target=["raw_file_id"],
            update_columns=[col for col in columns if col not in ["raw_file_id", "created_at"]],
        )


async def bulk_upsert_raw_annotation_files(
    raw_files: Sequence[RawAnnotationFile], batch_size: int = 1000
) -> int:
    """Bulk upsert raw annotation file records."""
    async with get_connection() as conn:
        columns = [
            "raw_file_id",
            "dataset_id",
            "storage_provider",
            "bucket",
            "object_key",
            "version_id",
            "file_path",
            "file_type",
            "file_name",
            "file_hash",
            "file_size",
            "encoding",
            "parsed_status",
            "parse_errors",
            "created_at",
            "updated_at",
        ]
        values_list = [
            (
                rf.raw_file_id,
                rf.dataset_id,
                rf.storage_provider,
                rf.bucket,
                rf.object_key,
                rf.version_id,
                rf.file_path,
                rf.file_type,
                rf.file_name,
                rf.file_hash,
                rf.file_size,
                rf.encoding,
                rf.parsed_status,
                rf.parse_errors,
                rf.created_at,
                rf.updated_at,
            )
            for rf in raw_files
        ]
        return await _bulk_upsert(
            conn,
            "raw_annotation_files",
            columns,
            values_list,
            conflict_target=["raw_file_id"],
            update_columns=[col for col in columns if col not in ["raw_file_id", "created_at"]],
            batch_size=batch_size,
        )
