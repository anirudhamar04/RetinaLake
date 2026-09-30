"""
Normalize class_name 'Glaucoma' → 'glaucoma' and recompute classification_id UUIDs.

Affected datasets: fundus_avseg (50), HRF-v1 (51), HRF-v2 (52), LES-AV (53).

For each row where class_name = 'Glaucoma':
  1. Recompute classification_id using class_name='glaucoma'
  2. INSERT the corrected row
  3. DELETE the old row

Done inside a single transaction — rolls back entirely on any error.

Usage:
    uv run python schema/migrations/normalize_glaucoma_class_name.py
    uv run python schema/migrations/normalize_glaucoma_class_name.py --dry-run
"""

import argparse
import asyncio
import hashlib
import json
import logging
import uuid

from psycopg.rows import dict_row

from chaksudb.db.connection import get_connection, close_pool
from chaksudb.ingest.framework.gen_uuid import generate_classification_uuid

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _class_value_hash(class_value: dict | None) -> str | None:
    if class_value is None:
        return None
    serialized = json.dumps(class_value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


async def migrate(dry_run: bool = False) -> None:
    async with get_connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """
                SELECT classification_id, image_id, task_type, class_name, sub_key,
                       class_index, class_label, class_value,
                       expert_annotation_id, consensus_id, raw_data_id,
                       annotation_method, confidence_score, provenance_chain_id, created_at
                FROM classification_annotations
                WHERE class_name = 'Glaucoma'
                """
            )
            rows = await cur.fetchall()

        if not rows:
            logger.info("No rows with class_name = 'Glaucoma' found — nothing to do.")
            await close_pool()
            return

        logger.info("Found %d rows to migrate.", len(rows))

        updates: list[tuple[uuid.UUID, uuid.UUID]] = []  # (old_id, new_id)
        for row in rows:
            new_id = generate_classification_uuid(
                image_id=row["image_id"],
                task_type=row["task_type"],
                class_name="glaucoma",
                sub_key=row["sub_key"],
                expert_annotation_id=row["expert_annotation_id"],
                consensus_id=row["consensus_id"],
                raw_data_id=row["raw_data_id"],
                class_value_hash=_class_value_hash(row["class_value"]),
            )
            updates.append((row["classification_id"], new_id, row))

        if dry_run:
            for old_id, new_id, row in updates:
                logger.info("  %s → %s  (image_id=%s)", old_id, new_id, row["image_id"])
            logger.info("Dry run — no changes written.")
            await close_pool()
            return

        async with get_connection() as conn:
            async with conn.transaction():
                async with conn.cursor() as cur:
                    for old_id, new_id, row in updates:
                        # Insert corrected row
                        await cur.execute(
                            """
                            INSERT INTO classification_annotations (
                                classification_id, image_id, task_type, class_name, sub_key,
                                class_index, class_label, class_value,
                                expert_annotation_id, consensus_id, raw_data_id,
                                annotation_method, confidence_score, provenance_chain_id, created_at
                            ) VALUES (
                                %s, %s, %s, %s, %s,
                                %s, %s, %s,
                                %s, %s, %s,
                                %s, %s, %s, %s
                            )
                            ON CONFLICT (classification_id) DO NOTHING
                            """,
                            (
                                new_id,
                                row["image_id"],
                                row["task_type"],
                                "glaucoma",
                                row["sub_key"],
                                row["class_index"],
                                row["class_label"],
                                json.dumps(row["class_value"]) if row["class_value"] is not None else None,
                                row["expert_annotation_id"],
                                row["consensus_id"],
                                row["raw_data_id"],
                                row["annotation_method"],
                                row["confidence_score"],
                                row["provenance_chain_id"],
                                row["created_at"],
                            ),
                        )

                    # Delete all old rows in one shot
                    old_ids = [old_id for old_id, _, _ in updates]
                    await cur.execute(
                        "DELETE FROM classification_annotations WHERE classification_id = ANY(%s)",
                        (old_ids,),
                    )

                    # Verify none remain
                    await cur.execute(
                        "SELECT COUNT(*) FROM classification_annotations WHERE class_name = 'Glaucoma'"
                    )
                    remaining = (await cur.fetchone())[0]
                    if remaining:
                        raise RuntimeError(f"{remaining} rows still have class_name='Glaucoma' after migration")

        logger.info("Migration complete: %d rows updated.", len(updates))

    await close_pool()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="Print planned changes without writing to DB")
    args = parser.parse_args()
    asyncio.run(migrate(dry_run=args.dry_run))


if __name__ == "__main__":
    main()
