"""
Image-related database operations.
"""

from typing import Sequence

import psycopg

from chaksudb.db.connection import get_connection
from chaksudb.db.models import Image, ImageDatasetMembership, ImageGroup, PatientImage
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


async def upsert_image_group(image_group: ImageGroup) -> None:
    """Upsert an image group record."""
    async with get_connection() as conn:
        columns = ["group_id", "dataset_id", "group_type", "created_at"]
        values = (
            image_group.group_id,
            image_group.dataset_id,
            image_group.group_type,
            image_group.created_at,
        )
        await _execute_upsert(
            conn, "image_groups", columns, values, conflict_target=["group_id"]
        )


async def bulk_upsert_image_groups(
    image_groups: Sequence[ImageGroup], batch_size: int = 1000
) -> int:
    """Bulk upsert image group records."""
    async with get_connection() as conn:
        columns = ["group_id", "dataset_id", "group_type", "created_at"]
        values_list = [
            (
                ig.group_id,
                ig.dataset_id,
                ig.group_type,
                ig.created_at,
            )
            for ig in image_groups
        ]
        return await _bulk_upsert(
            conn,
            "image_groups",
            columns,
            values_list,
            conflict_target=["group_id"],
            batch_size=batch_size,
        )


async def upsert_image(image: Image) -> None:
    """Upsert an image record."""
    async with get_connection() as conn:
        columns = [
            "image_id",
            "dataset_id",
            "original_image_id",
            "storage_provider",
            "bucket",
            "object_key",
            "version_id",
            "file_path",
            "file_format",
            "modality",
            "file_hash",
            "content_hash",
            "phash",
            "group_id",
            "frame_index",
            "resolution_width",
            "resolution_height",
            "field_of_view",
            "eye_laterality",
            "acquisition_date",
            "created_at",
            "updated_at",
        ]
        values = (
            image.image_id,
            image.dataset_id,
            image.original_image_id,
            image.storage_provider,
            image.bucket,
            image.object_key,
            image.version_id,
            image.file_path,
            image.file_format,
            image.modality,
            image.file_hash,
            image.content_hash,
            image.phash,
            image.group_id,
            image.frame_index,
            image.resolution_width,
            image.resolution_height,
            image.field_of_view,
            image.eye_laterality,
            image.acquisition_date,
            image.created_at,
            image.updated_at,
        )
        # Use primary key for conflict detection (works with deterministic UUIDs)
        await _execute_upsert(
            conn,
            "images",
            columns,
            values,
            conflict_target=["image_id"],
            update_columns=[col for col in columns if col not in ["image_id", "created_at"]],
        )


async def bulk_upsert_images(
    images: Sequence[Image], batch_size: int = 1000
) -> int:
    """Bulk upsert image records."""
    async with get_connection() as conn:
        columns = [
            "image_id",
            "dataset_id",
            "original_image_id",
            "storage_provider",
            "bucket",
            "object_key",
            "version_id",
            "file_path",
            "file_format",
            "modality",
            "file_hash",
            "content_hash",
            "phash",
            "group_id",
            "frame_index",
            "resolution_width",
            "resolution_height",
            "field_of_view",
            "eye_laterality",
            "acquisition_date",
            "created_at",
            "updated_at",
        ]
        values_list = [
            (
                img.image_id,
                img.dataset_id,
                img.original_image_id,
                img.storage_provider,
                img.bucket,
                img.object_key,
                img.version_id,
                img.file_path,
                img.file_format,
                img.modality,
                img.file_hash,
                img.content_hash,
                img.phash,
                img.group_id,
                img.frame_index,
                img.resolution_width,
                img.resolution_height,
                img.field_of_view,
                img.eye_laterality,
                img.acquisition_date,
                img.created_at,
                img.updated_at,
            )
            for img in images
        ]
        return await _bulk_upsert(
            conn,
            "images",
            columns,
            values_list,
            conflict_target=["image_id"],
            update_columns=[col for col in columns if col not in ["image_id", "created_at"]],
            batch_size=batch_size,
        )


async def upsert_patient_image(patient_image: PatientImage) -> None:
    """Upsert a patient-image relationship record."""
    async with get_connection() as conn:
        columns = [
            "relationship_id",
            "patient_id",
            "image_id",
            "exam_date",
            "created_at",
        ]
        values = (
            patient_image.relationship_id,
            patient_image.patient_id,
            patient_image.image_id,
            patient_image.exam_date,
            patient_image.created_at,
        )
        await _execute_upsert(
            conn,
            "patient_images",
            columns,
            values,
            conflict_target=["patient_id", "image_id"],
        )


async def bulk_upsert_patient_images(
    patient_images: Sequence[PatientImage], batch_size: int = 1000
) -> int:
    """Bulk upsert patient-image relationship records."""
    async with get_connection() as conn:
        columns = [
            "relationship_id",
            "patient_id",
            "image_id",
            "exam_date",
            "created_at",
        ]
        values_list = [
            (
                pi.relationship_id,
                pi.patient_id,
                pi.image_id,
                pi.exam_date,
                pi.created_at,
            )
            for pi in patient_images
        ]
        return await _bulk_upsert(
            conn,
            "patient_images",
            columns,
            values_list,
            conflict_target=["patient_id", "image_id"],
            batch_size=batch_size,
        )


# ============================================
# Cross-dataset duplicate detection / membership
# ============================================


async def find_image_by_content_hash(content_hash: str):
    """Return the canonical (image_id, dataset_id) for a decoded-pixel content hash, or None.

    content_hash is encoding-invariant (same pixels under any lossless container match), so
    this is the safe key for cross-dataset dedup. The canonical row is the earliest-ingested
    image with this content.
    """
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT image_id, dataset_id FROM images "
                "WHERE content_hash = %s ORDER BY created_at ASC LIMIT 1",
                (content_hash,),
            )
            return await cur.fetchone()


async def find_image_by_file_hash(file_hash: str):
    """Return the canonical (image_id, dataset_id) for an exact-bytes file hash, or None."""
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT image_id, dataset_id FROM images "
                "WHERE file_hash = %s ORDER BY created_at ASC LIMIT 1",
                (file_hash,),
            )
            return await cur.fetchone()


async def add_image_dataset_membership(
    image_id, dataset_id, original_image_id: str | None = None
) -> None:
    """Record that a secondary dataset contains an already-ingested (canonical) image.

    Idempotent: re-recording the same membership is a no-op.
    """
    membership = ImageDatasetMembership(
        image_id=image_id, dataset_id=dataset_id, original_image_id=original_image_id
    )
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "INSERT INTO image_dataset_memberships "
                "(image_id, dataset_id, original_image_id) VALUES (%s, %s, %s) "
                "ON CONFLICT (image_id, dataset_id) DO NOTHING",
                (membership.image_id, membership.dataset_id, membership.original_image_id),
            )
