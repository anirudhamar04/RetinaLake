"""
Report duplicate images across datasets at three levels of strictness.

  1. exact bytes      — same file_hash (identical file).
  2. same content     — same content_hash but NOT same bytes: the same image stored under
                        different (lossless) encodings / re-saves.
  3. perceptual       — same phash (or within --hamming bits): visually the same image,
                        including across lossy re-encoding (JPEG) or a resize.

Levels 1-2 are exact and safe to auto-merge; level 3 is advisory (review before merging).

Usage:
    uv run python scripts/find_duplicate_images.py            # exact + content groups
    uv run python scripts/find_duplicate_images.py --perceptual          # + phash groups
    uv run python scripts/find_duplicate_images.py --perceptual --hamming 4
"""

import argparse
import asyncio
import csv
import logging
from collections import defaultdict

from chaksudb.db.connection import get_connection
from chaksudb.ingest.framework.hashing import hamming_distance

logger = logging.getLogger(__name__)

# Accumulates one row per duplicate group for optional CSV export. Read-only run;
# nothing here writes to the DB.
_CSV_ROWS: list[dict] = []


async def _groups_by(column: str, extra_where: str = "") -> list[tuple]:
    sql = f"""
        SELECT i.{column},
               count(*) AS n,
               array_agg(d.dataset_name ORDER BY i.created_at) AS datasets,
               array_agg(i.original_image_id ORDER BY i.created_at) AS ids
        FROM images i
        JOIN datasets d ON i.dataset_id = d.dataset_id
        WHERE i.{column} IS NOT NULL {extra_where}
        GROUP BY i.{column}
        HAVING count(*) > 1
        ORDER BY n DESC
    """
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(sql)
            return await cur.fetchall()


def _print_groups(title: str, rows: list[tuple], level: str) -> None:
    if not rows:
        logger.info("%s: none", title)
        return
    logger.info("%s: %d groups (%d rows)", title, len(rows), sum(r[1] for r in rows))
    for value, n, datasets, ids in rows:
        logger.info("  %s… x%d", str(value)[:12], n)
        logger.info("    datasets: %s", ", ".join(datasets))
        logger.info("    ids:      %s", ", ".join(str(i) for i in ids))
        _CSV_ROWS.append({
            "level": level,
            "hash": value,
            "count": n,
            "datasets": ", ".join(datasets),
            "image_ids": ", ".join(str(i) for i in ids),
        })


async def _perceptual_near_dups(max_hamming: int) -> None:
    """Cluster images whose perceptual hashes are within max_hamming bits.

    O(n^2) over distinct phashes; fine for auditing. max_hamming=0 is exact phash equality.
    """
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT i.phash, d.dataset_name, i.original_image_id "
                "FROM images i JOIN datasets d ON i.dataset_id = d.dataset_id "
                "WHERE i.phash IS NOT NULL ORDER BY i.phash"
            )
            rows = await cur.fetchall()

    by_hash: dict[str, list[str]] = defaultdict(list)
    for phash, dataset, oid in rows:
        by_hash[phash].append(f"{dataset}:{oid}")
    distinct = list(by_hash)

    seen: set[str] = set()
    clusters: list[list[str]] = []
    for i, ha in enumerate(distinct):
        if ha in seen:
            continue
        members = list(by_hash[ha])
        seen.add(ha)
        for hb in distinct[i + 1:]:
            if hb not in seen and hamming_distance(ha, hb) <= max_hamming:
                members.extend(by_hash[hb])
                seen.add(hb)
        if len(members) > 1:
            clusters.append(members)

    if not clusters:
        logger.info("Perceptual near-duplicates (<=%d bits): none", max_hamming)
        return
    logger.info("Perceptual near-duplicates (<=%d bits): %d clusters", max_hamming, len(clusters))
    for members in sorted(clusters, key=len, reverse=True):
        logger.info("  cluster x%d: %s", len(members), ", ".join(members))
        _CSV_ROWS.append({
            "level": f"perceptual<=({max_hamming})",
            "hash": "",
            "count": len(members),
            "datasets": "",
            "image_ids": ", ".join(members),
        })


async def main() -> int:
    parser = argparse.ArgumentParser(description="Find duplicate images across datasets.")
    parser.add_argument("--perceptual", action="store_true",
                        help="Also report perceptual (encoding/resize-invariant) near-duplicates.")
    parser.add_argument("--hamming", type=int, default=0,
                        help="Max perceptual Hamming distance to treat as a near-duplicate (default 0).")
    parser.add_argument("--csv", nargs="?", const="duplication_report.csv", default=None,
                        help="Write the duplicate groups to a CSV (default duplication_report.csv). "
                             "Read-only: this never modifies the DB.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    logger.info("=" * 80)
    _print_groups("Exact byte duplicates (file_hash)", await _groups_by("file_hash"), "exact_bytes")
    logger.info("-" * 80)
    # same decoded pixels but different bytes => same image, different encoding
    _print_groups(
        "Same-content, different-encoding (content_hash)",
        await _groups_by(
            "content_hash",
            "AND i.image_id NOT IN ("
            "  SELECT i2.image_id FROM images i2 WHERE i2.file_hash IN ("
            "    SELECT file_hash FROM images WHERE file_hash IS NOT NULL "
            "    GROUP BY file_hash HAVING count(*) > 1))",
        ),
        "same_content",
    )
    if args.perceptual:
        logger.info("-" * 80)
        await _perceptual_near_dups(args.hamming)
    logger.info("=" * 80)

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["level", "hash", "count", "datasets", "image_ids"])
            writer.writeheader()
            writer.writerows(_CSV_ROWS)
        logger.info("Wrote %d duplicate groups -> %s", len(_CSV_ROWS), args.csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
