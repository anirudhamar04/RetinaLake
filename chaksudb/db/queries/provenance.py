"""
Provenance chain and transformation database operations.
"""

import uuid
from typing import Any, Optional

import psycopg
from psycopg.rows import dict_row

from chaksudb.db.connection import get_connection
from chaksudb.db.models import (
    ProvenanceChain,
    ProvenanceTransformation,
    TransformationOperation,
)
from chaksudb.db.queries.helpers import _prepare_upsert_query, _serialize_jsonb


async def upsert_provenance_chain(chain: ProvenanceChain) -> None:
    """Upsert a provenance chain record."""
    async with get_connection() as conn:
        columns = [
            "chain_id",
            "unified_annotation_type",
            "source_type",
            "root_source_raw_data_id",
            "source_annotation_ids",
            "created_at",
        ]
        values = (
            chain.chain_id,
            chain.unified_annotation_type,
            chain.source_type,
            chain.root_source_raw_data_id,
            [str(aid) for aid in (chain.source_annotation_ids or [])],
            chain.created_at,
        )
        query = _prepare_upsert_query(
            "provenance_chain",
            columns,
            conflict_target=["chain_id"],
        )
        async with conn.cursor() as cur:
            await cur.execute(query, values)


async def upsert_transformation_operation(transformation: TransformationOperation) -> None:
    """Upsert a transformation operation record."""
    async with get_connection() as conn:
        columns = [
            "transformation_id",
            "operation_type",
            "input_data",
            "output_data",
            "operation_parameters",
            "operation_timestamp",
            "operator",
            "notes",
        ]
        values = (
            transformation.transformation_id,
            transformation.operation_type,
            _serialize_jsonb(transformation.input_data),
            _serialize_jsonb(transformation.output_data),
            _serialize_jsonb(transformation.operation_parameters),
            transformation.operation_timestamp,
            transformation.operator,
            transformation.notes,
        )
        query = _prepare_upsert_query(
            "transformation_operations",
            columns,
            conflict_target=["transformation_id"],
        )
        async with conn.cursor() as cur:
            await cur.execute(query, values)


async def upsert_provenance_transformation(prov_trans: ProvenanceTransformation) -> None:
    """Upsert a provenance-transformation link record."""
    async with get_connection() as conn:
        columns = ["id", "chain_id", "transformation_id", "created_at"]
        values = (
            prov_trans.id,
            prov_trans.chain_id,
            prov_trans.transformation_id,
            prov_trans.created_at,
        )
        query = _prepare_upsert_query(
            "provenance_transformations",
            columns,
            conflict_target=["chain_id", "transformation_id"],
        )
        async with conn.cursor() as cur:
            await cur.execute(query, values)


# ============================================
# Read / audit helpers
# ============================================


async def get_chain(chain_id: uuid.UUID) -> Optional[ProvenanceChain]:
    """Fetch a single provenance chain by id, or None if absent."""
    async with get_connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                "SELECT * FROM provenance_chain WHERE chain_id = %s",
                (chain_id,),
            )
            row = await cur.fetchone()
            return ProvenanceChain(**row) if row else None


async def get_transformations_for_chain(
    chain_id: uuid.UUID,
) -> list[TransformationOperation]:
    """Fetch all transformation operations linked to a provenance chain.

    Ordered by ``operation_timestamp`` so the lineage reads chronologically.
    """
    async with get_connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """
                SELECT t.*
                FROM transformation_operations t
                JOIN provenance_transformations pt
                  ON pt.transformation_id = t.transformation_id
                WHERE pt.chain_id = %s
                ORDER BY t.operation_timestamp
                """,
                (chain_id,),
            )
            return [TransformationOperation(**row) for row in await cur.fetchall()]


# Annotation tables that carry a provenance_chain_id FK. (clinical_descriptions has
# image_id but no provenance_chain_id, so it is intentionally excluded.)
_ANNOTATION_CHAIN_SOURCES = (
    ("disease_grading", "grading_id"),
    ("segmentation_annotations", "segmentation_id"),
    ("localization_annotations", "localization_id"),
    ("classification_annotations", "classification_id"),
    ("quality_annotations", "quality_id"),
    ("keyword_annotations", "keyword_id"),
)


async def get_lineage_for_image(image_id: uuid.UUID) -> list[dict[str, Any]]:
    """Return the full lineage for an image: each chain touching the image plus its
    transformations.

    Answers "what was done to this image's raw data" in one call. Returns a list of
    ``{"chain": ProvenanceChain, "transformations": [TransformationOperation, ...]}``.
    """
    chain_ids: set[uuid.UUID] = set()
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            for table, _pk in _ANNOTATION_CHAIN_SOURCES:
                await cur.execute(
                    f"SELECT DISTINCT provenance_chain_id FROM {table} "
                    f"WHERE image_id = %s AND provenance_chain_id IS NOT NULL",
                    (image_id,),
                )
                for (cid,) in await cur.fetchall():
                    chain_ids.add(cid)

    lineage: list[dict[str, Any]] = []
    for cid in chain_ids:
        chain = await get_chain(cid)
        if chain is None:
            continue
        lineage.append(
            {
                "chain": chain,
                "transformations": await get_transformations_for_chain(cid),
            }
        )
    return lineage


async def find_orphan_transformations() -> list[TransformationOperation]:
    """Return transformation_operations rows not linked to any provenance chain.

    A non-empty result means audit rows exist with no traceable lineage — exactly what
    the old ``gen_random_uuid()`` triggers produced. Used by validation and as a
    regression guard.
    """
    async with get_connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """
                SELECT t.*
                FROM transformation_operations t
                LEFT JOIN provenance_transformations pt
                  ON pt.transformation_id = t.transformation_id
                WHERE pt.transformation_id IS NULL
                ORDER BY t.operation_timestamp
                """
            )
            return [TransformationOperation(**row) for row in await cur.fetchall()]


async def fetch_grade_conversions_for_audit() -> list[dict[str, Any]]:
    """Fetch every converted disease_grading row joined to its ICDR_0_4 target scale.

    Source for ``reconcile_grade_conversions``: each row is shaped like the NOTIFY event
    the trigger emits (values as strings/ints) so the reconciliation sweep produces
    byte-identical, deterministically-keyed audit rows to the live listener.
    """
    async with get_connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """
                SELECT
                    dg.grading_id,
                    dg.image_id,
                    dg.scale_id,
                    dg.original_grade,
                    dg.disease_type,
                    dg.scaled_grade,
                    dg.provenance_chain_id,
                    ts.scale_id   AS target_scale_id,
                    ts.scale_name AS target_scale_name
                FROM disease_grading dg
                JOIN grading_scales ts
                  ON ts.scale_name = 'ICDR_0_4'
                 AND ts.disease_type = dg.disease_type
                WHERE dg.scaled_grade IS NOT NULL
                  AND dg.provenance_chain_id IS NOT NULL
                """
            )
            rows = await cur.fetchall()

    events: list[dict[str, Any]] = []
    for r in rows:
        same_scale = r["scale_id"] == r["target_scale_id"]
        events.append(
            {
                "mode": "same_scale" if same_scale else "mapped",
                "grading_id": str(r["grading_id"]),
                "image_id": str(r["image_id"]),
                "scale_id": str(r["scale_id"]),
                "original_grade": r["original_grade"],
                "disease_type": r["disease_type"],
                "scaled_grade": r["scaled_grade"],
                "target_scale_id": str(r["target_scale_id"]),
                "target_scale_name": r["target_scale_name"],
                "provenance_chain_id": str(r["provenance_chain_id"]),
            }
        )
    return events
