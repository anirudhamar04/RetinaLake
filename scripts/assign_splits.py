"""
Standalone script to assign train/val/test splits to all datasets that don't have
complete splits, without re-running ingestion.

Three cases are handled automatically per dataset:
  - No splits at all  → 90/10 train+test, then 90/10 train+val  (≈81/9/10)
  - train + test only → 90/10 split on train to create val        (≈81/9/10)
  - train + val + test → already complete; re-registered as user_defined with the
    exact same image membership (so the export's user_defined filter retrieves them too)

Splits are stratified by classification or grading labels where available,
otherwise fall back to random.

Usage:
    # Process all datasets in the DB that aren't fully split
    uv run python scripts/assign_splits.py

    # Only specific datasets
    uv run python scripts/assign_splits.py --datasets G1020 CHASEDB1 HRF

    # Re-assign even datasets that already have full train/val/test
    uv run python scripts/assign_splits.py --force
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import uuid
from typing import Optional

from psycopg.rows import dict_row

from chaksudb.db import close_pool, get_connection
from chaksudb.ingest.framework.split_assigner import (
    auto_stratified_splits,
    delete_splits_for_dataset,
)

logger = logging.getLogger(__name__)


async def _fetch_all_datasets(names: Optional[list[str]]) -> list[dict]:
    """Return all {dataset_id, dataset_name} rows, optionally filtered by name."""
    async with get_connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            if names:
                await cur.execute(
                    """
                    SELECT dataset_id, dataset_name
                    FROM datasets
                    WHERE dataset_name = ANY(%s)
                    ORDER BY dataset_name
                    """,
                    (names,),
                )
            else:
                await cur.execute(
                    "SELECT dataset_id, dataset_name FROM datasets ORDER BY dataset_name"
                )
            return await cur.fetchall()


async def _fetch_existing_splits(dataset_id: uuid.UUID) -> dict[str, list[uuid.UUID]]:
    """Return {split_name: [image_id, ...]} for all splits that already exist."""
    async with get_connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """
                SELECT ds.split_name, isp.image_id
                FROM dataset_splits ds
                JOIN image_splits isp ON ds.split_id = isp.split_id
                WHERE ds.dataset_id = %s
                ORDER BY ds.split_name
                """,
                (dataset_id,),
            )
            rows = await cur.fetchall()

    splits: dict[str, list[uuid.UUID]] = {}
    for row in rows:
        splits.setdefault(row["split_name"], []).append(row["image_id"])
    return splits


async def _fetch_all_image_ids(dataset_id: uuid.UUID) -> list[uuid.UUID]:
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT image_id FROM images WHERE dataset_id = %s ORDER BY image_id",
                (dataset_id,),
            )
            return [r[0] for r in await cur.fetchall()]


async def _fetch_classification_labels(dataset_id: uuid.UUID) -> dict[uuid.UUID, str]:
    """Return {image_id: class_value} using the most frequent class_name in this dataset."""
    async with get_connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """
                SELECT ca.class_name, COUNT(*) AS cnt
                FROM classification_annotations ca
                JOIN images i ON ca.image_id = i.image_id
                WHERE i.dataset_id = %s
                GROUP BY ca.class_name
                ORDER BY cnt DESC
                LIMIT 1
                """,
                (dataset_id,),
            )
            row = await cur.fetchone()
            if not row:
                return {}
            class_name = row["class_name"]

            await cur.execute(
                """
                SELECT DISTINCT ON (ca.image_id) ca.image_id, ca.class_value::text AS label
                FROM classification_annotations ca
                JOIN images i ON ca.image_id = i.image_id
                WHERE i.dataset_id = %s AND ca.class_name = %s
                ORDER BY ca.image_id
                """,
                (dataset_id, class_name),
            )
            return {r["image_id"]: r["label"] for r in await cur.fetchall()}


async def _fetch_grading_labels(dataset_id: uuid.UUID) -> dict[uuid.UUID, str]:
    """Return {image_id: grade_value_str} using the most frequent disease_type."""
    async with get_connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """
                SELECT dg.disease_type, COUNT(*) AS cnt
                FROM disease_grading dg
                JOIN images i ON dg.image_id = i.image_id
                WHERE i.dataset_id = %s
                GROUP BY dg.disease_type
                ORDER BY cnt DESC
                LIMIT 1
                """,
                (dataset_id,),
            )
            row = await cur.fetchone()
            if not row:
                return {}
            disease_type = row["disease_type"]

            await cur.execute(
                """
                SELECT DISTINCT ON (dg.image_id) dg.image_id,
                       COALESCE(dg.grade_label, dg.scaled_grade::text, dg.original_grade) AS label
                FROM disease_grading dg
                JOIN images i ON dg.image_id = i.image_id
                WHERE i.dataset_id = %s AND dg.disease_type = %s
                  AND COALESCE(dg.grade_label, dg.scaled_grade::text, dg.original_grade) IS NOT NULL
                ORDER BY dg.image_id
                """,
                (dataset_id, disease_type),
            )
            return {r["image_id"]: r["label"] for r in await cur.fetchall()}


async def _get_labels(dataset_id: uuid.UUID) -> tuple[dict[uuid.UUID, str], str]:
    """Try classification → grading → none. Returns (labels, source_description)."""
    labels = await _fetch_classification_labels(dataset_id)
    if labels:
        return labels, "classification"
    labels = await _fetch_grading_labels(dataset_id)
    if labels:
        return labels, "grading"
    return {}, "none (random)"


async def assign_splits_for_dataset(
    dataset_id: uuid.UUID,
    dataset_name: str,
    force: bool = False,
) -> None:
    existing = await _fetch_existing_splits(dataset_id)
    existing_keys = set(existing.keys())

    labels, label_source = await _get_labels(dataset_id)

    # Determine what work needs to be done
    if existing_keys >= {"train", "val", "test"} and not force:
        # Already complete — do NOT re-randomize. Re-register the exact existing
        # membership under split_type="user_defined" (auto_stratified_splits Case 1
        # derives nothing when all splits are known, so boundaries are preserved).
        # This is what lets the export's user_defined filter retrieve these datasets,
        # which were previously stored as explicit and skipped entirely.
        split_assignments = {k: existing[k] for k in existing_keys}
        case = "complete → relabel as user_defined (boundaries preserved)"

    elif not existing_keys or force:
        # No splits at all (or forced): fetch all images and do the full 90/10/10
        image_ids = await _fetch_all_image_ids(dataset_id)
        if not image_ids:
            logger.warning(f"[{dataset_name}] No images found — skipping")
            return
        split_assignments = {"train": image_ids}
        case = "no splits → train/val/test"

    elif existing_keys == {"train", "test"} or existing_keys == {"train"}:
        # Has train (and maybe test) but no val — pass what exists so auto_stratified_splits
        # only derives the missing val split from train
        split_assignments = {k: existing[k] for k in existing_keys}
        case = f"{'+'.join(sorted(existing_keys))} → adding val"

    else:
        # Partial or unexpected split config — treat all images as train pool
        image_ids = await _fetch_all_image_ids(dataset_id)
        split_assignments = {"train": image_ids}
        case = f"partial splits {existing_keys} → reassigning"

    total_images = sum(len(v) for v in split_assignments.values())
    label_coverage = f"{len(labels)}/{total_images}" if labels else "0"
    logger.info(
        f"[{dataset_name}] {case} | {total_images} images | "
        f"labels: {label_source} ({label_coverage})"
    )

    # Remove all pre-existing splits before re-assigning. user_defined splits
    # are NOT idempotent if the image pool changed between runs (different random
    # boundaries → stale rows accumulate). Always purge all types first.
    await delete_splits_for_dataset(
        dataset_id=dataset_id,
        split_types=["explicit", "metadata_defined", "undefined", "user_defined"],
    )

    _, split_counts = await auto_stratified_splits(
        dataset_id=dataset_id,
        split_assignments=split_assignments,
        labels=labels or None,
        split_type="user_defined",
    )

    logger.info(
        f"[{dataset_name}] Done: "
        + ", ".join(f"{k}={v}" for k, v in sorted(split_counts.items()))
    )


async def run(names: Optional[list[str]], force: bool) -> None:
    datasets = await _fetch_all_datasets(names)

    if names:
        found = {ds["dataset_name"] for ds in datasets}
        missing = [n for n in names if n not in found]
        if missing:
            logger.warning(f"Not found in DB (not yet ingested?): {', '.join(missing)}")

    if not datasets:
        logger.error("No datasets found in DB.")
        await close_pool()
        return

    logger.info(f"Checking {len(datasets)} dataset(s)...")
    for ds in datasets:
        try:
            await assign_splits_for_dataset(
                dataset_id=ds["dataset_id"],
                dataset_name=ds["dataset_name"],
                force=force,
            )
        except Exception as e:
            logger.error(f"[{ds['dataset_name']}] Failed: {e}", exc_info=True)

    await close_pool()
    logger.info("All done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assign stratified splits to all datasets that aren't fully split"
    )
    parser.add_argument(
        "--datasets", nargs="+", default=None, metavar="NAME",
        help="Only process these datasets (default: all datasets in DB)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-assign splits even for datasets that already have train/val/test"
    )
    args = parser.parse_args()

    asyncio.run(run(names=args.datasets, force=args.force))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()
