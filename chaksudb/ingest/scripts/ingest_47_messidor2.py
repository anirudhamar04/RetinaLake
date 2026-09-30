"""
Ingestion script for MESSIDOR2 dataset.

Dataset: MESSIDOR-2 — Diabetic Retinopathy Multi-site Re-annotation
Structure: Single CSV (annotations.csv) + Images/ flat folder
  CSV columns: Image name, Ophthalmologic department, Retinopathy grade,
               Risk of macular edema, source_file
  1200 images from 12 acquisition bases (100 images each) across 3 French
  ophthalmologic departments.

Relationship to MESSIDOR (02):
  - 1058 images share filenames with MESSIDOR (02) — SAME dataset_id / image_id
    (deterministic UUID v5). Only annotations are added for these; the Image row
    and split assignment already exist from the MESSIDOR ingest and are not touched.
  - 142 images are new to MESSIDOR2 — Image rows are created and splits assigned.

Annotations added:
  - DR grading: 0–4 (ICDR scale; MESSIDOR2 adjudicated grades 0–3 map directly;
    grade 4 PDR is absent from this dataset). Stored under ICDR_0_4, superseding
    any MESSIDOR (02) grade for shared images.
  - DME grading: 0–2 (0=no risk, 1=low risk, 2=high risk), scale MESSIDOR2_DME_0_2
    (3-level, distinct from MESSIDOR's binary DME classification)
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Set
from uuid import UUID

from chaksudb.common.progress import ProgressTracker, OperationStatistics
from chaksudb.config.config import get_data_root
from chaksudb.db.models import Dataset, DiseaseGrading, Image
from chaksudb.db.queries import (
    bulk_upsert_disease_gradings,
    bulk_upsert_images,
    upsert_dataset,
)
from chaksudb.ingest.framework import get_image_metadata_dict, process_csv, read_csv_auto
from chaksudb.ingest.framework.gen_uuid import generate_dataset_uuid, generate_image_uuid
from chaksudb.ingest.framework.task_processors.grading_processor import process_disease_grade
from chaksudb.ingest.framework.split_assigner import auto_stratified_splits

logger = logging.getLogger(__name__)

# Same dataset as MESSIDOR — this is an extension, not a new dataset.
DATASET_NAME = "MESSIDOR"
DATASET_URL = "https://www.adcis.net/en/third-party/messidor2/"
DATASET_LICENSE = "Custom - Educational and research use"

DR_SCALE_NAME = "ICDR_0_4"
DR_SCALE_DESCRIPTION = (
    "International Clinical Diabetic Retinopathy scale: "
    "0=no DR, 1=mild NPDR, 2=moderate NPDR, 3=severe NPDR, 4=PDR"
)
DR_VALUE_LABELS = {
    "0": "no_DR",
    "1": "mild_NPDR",
    "2": "moderate_NPDR",
    "3": "severe_NPDR",
    "4": "PDR",
}

DME_SCALE_NAME = "MESSIDOR2_DME_0_2"
DME_SCALE_DESCRIPTION = "MESSIDOR-2 macular edema risk: 0=no risk, 1=low risk, 2=high risk"
DME_VALUE_LABELS = {"0": "no_risk", "1": "low_risk", "2": "high_risk"}


def _load_messidor1_filenames() -> Set[str]:
    """Return the set of image filenames already ingested by MESSIDOR (02)."""
    messidor1_csv = get_data_root() / "02_MESSIDOR" / "messidor_data.csv"
    if not messidor1_csv.exists():
        logger.warning(
            "MESSIDOR original CSV not found at %s — treating all MESSIDOR2 images as new",
            messidor1_csv,
        )
        return set()
    rows = read_csv_auto(messidor1_csv)
    return {str(r["image_id"]).strip() for r in rows}


async def ingest_messidor2() -> OperationStatistics:
    data_root = get_data_root() / "47_MESSIDOR2"
    images_dir = data_root / "Images"
    csv_path = data_root / "annotations.csv"
    dataset_id = generate_dataset_uuid(DATASET_NAME)

    logger.info("=" * 80)
    logger.info(f"Starting ingestion: MESSIDOR2 (extending {DATASET_NAME})")
    logger.info(f"Data root: {data_root}")
    logger.info("=" * 80)

    # Ensure dataset record exists (idempotent — no-op if already present).
    dataset = Dataset(
        dataset_id=dataset_id,
        dataset_name=DATASET_NAME,
        source_url=DATASET_URL,
        license=DATASET_LICENSE,
        modality_types=["fundus"],
    )
    await upsert_dataset(dataset)

    # Filenames that MESSIDOR (02) already ingested — skip Image row + split for these.
    known_filenames: Set[str] = await asyncio.to_thread(_load_messidor1_filenames)
    logger.info(
        f"Loaded {len(known_filenames)} known MESSIDOR filenames — "
        "Image rows and splits will only be created for images not in this set."
    )

    csv_rows = await asyncio.to_thread(read_csv_auto, csv_path)
    total_count = len(csv_rows)
    logger.info(f"Found {total_count} rows in MESSIDOR2 annotations CSV")

    tracker = ProgressTracker(total=total_count, description="Ingesting MESSIDOR2")

    new_images: List[Image] = []          # 142 images not in MESSIDOR1
    all_dr_gradings: List[DiseaseGrading] = []
    all_dme_gradings: List[DiseaseGrading] = []
    new_image_ids: List[UUID] = []        # only for split assignment
    new_image_labels: Dict[UUID, int] = {}

    async def process_row(row, idx):
        try:
            image_name = str(row["Image name"]).strip()
            image_id = generate_image_uuid(dataset_id, image_name)
            is_new = image_name not in known_filenames

            if is_new:
                image_path = images_dir / image_name
                if not await asyncio.to_thread(image_path.exists):
                    # Try case variations for new images only.
                    found = False
                    for ext in [".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG"]:
                        candidate = images_dir / f"{Path(image_name).stem}{ext}"
                        if await asyncio.to_thread(candidate.exists):
                            image_path = candidate
                            found = True
                            break
                    if not found:
                        tracker.record_error(
                            error_type="file_not_found",
                            error_message=f"New image not found: {image_name}",
                            item_id=image_name,
                        )
                        tracker.update(count=1, success=False)
                        return

                new_images.append(Image(
                    image_id=image_id,
                    dataset_id=dataset_id,
                    original_image_id=image_name,
                    **get_image_metadata_dict(image_path),
                    modality="fundus",
                ))
                new_image_ids.append(image_id)

            # DR grading — ICDR 0-4. For shared images this supersedes the MESSIDOR (02)
            # grade stored under the same scale via upsert.
            dr_raw = row.get("Retinopathy grade")
            if dr_raw is not None and str(dr_raw).strip() not in ("", "nan"):
                dr_grade = int(float(str(dr_raw).strip()))
                if is_new:
                    new_image_labels[image_id] = dr_grade
                dr_grading = await process_disease_grade(
                    grade_value=dr_grade,
                    disease_type="DR",
                    scale_name=DR_SCALE_NAME,
                    image_id=image_id,
                    scale_description=DR_SCALE_DESCRIPTION,
                    min_value=0,
                    max_value=4,
                    value_labels=DR_VALUE_LABELS,
                    annotation_method="manual",
                )
                all_dr_gradings.append(dr_grading)

            # DME grading — 3-level (distinct from MESSIDOR's binary DME classification).
            dme_col = "Risk of macular edema "   # trailing space as in the CSV header
            dme_raw = row.get(dme_col) or row.get(dme_col.strip())
            if dme_raw is not None and str(dme_raw).strip() not in ("", "nan"):
                dme_grade = int(float(str(dme_raw).strip()))
                dme_grading = await process_disease_grade(
                    grade_value=dme_grade,
                    disease_type="DME",
                    scale_name=DME_SCALE_NAME,
                    image_id=image_id,
                    scale_description=DME_SCALE_DESCRIPTION,
                    min_value=0,
                    max_value=2,
                    value_labels=DME_VALUE_LABELS,
                    annotation_method="manual",
                )
                all_dme_gradings.append(dme_grading)

            tracker.update(count=1, success=True)

        except Exception as e:
            tracker.update(count=1, success=False)
            tracker.record_error(
                error_type="processing",
                error_message=str(e),
                item_id=str(row.get("Image name", idx)),
            )
            logger.error(f"Failed to process row {idx}: {e}")

    await process_csv(csv_path, dataset_id, "grading", process_row)

    logger.info(
        f"New images: {len(new_images)} | "
        f"DR gradings: {len(all_dr_gradings)} | "
        f"DME gradings: {len(all_dme_gradings)}"
    )

    if new_images:
        await bulk_upsert_images(new_images, batch_size=1000)

    await asyncio.gather(
        bulk_upsert_disease_gradings(all_dr_gradings, batch_size=1000),
        bulk_upsert_disease_gradings(all_dme_gradings, batch_size=1000),
    )

    # Assign splits only for the 142 images not already split by MESSIDOR ingest.
    if new_image_ids:
        logger.info(f"Assigning stratified splits for {len(new_image_ids)} new images...")
        await auto_stratified_splits(
            dataset_id=dataset_id,
            split_assignments={"train": new_image_ids},
            labels=new_image_labels,
            split_type="explicit",
        )

    tracker.finish()
    stats = tracker.get_statistics()
    logger.info("=" * 80)
    logger.info("Ingestion Summary:")
    logger.info(
        f"  Total: {stats.total_items}, "
        f"Success: {stats.successful_items}, "
        f"Failed: {stats.failed_items}"
    )
    logger.info(
        f"  New images created: {len(new_images)} | "
        f"Shared images (annotations only): {total_count - len(new_images)}"
    )
    logger.info("=" * 80)
    return stats


async def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    try:
        stats = await ingest_messidor2()
        if stats.failed_items > 0:
            logger.error(f"Ingestion completed with {stats.failed_items} errors")
            return 1
        logger.info("Ingestion completed successfully!")
        return 0
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        return 1


if __name__ == "__main__":
    exit(asyncio.run(main()))
