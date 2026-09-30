"""
Ingestion script for MuReD dataset.

Dataset: MuReD (Multi-label Retinal Diseases) - Large-scale multi-label fundus
         classification across 20 retinal disease/condition categories.
Structure:
  images/          2,451 fundus images (PNG + TIF, mixed sources)
  train_data.csv   1,764 annotated training samples
  val_data.csv       444 annotated validation samples
  (243 images have no CSV entry — unlabeled holdout/test set)
Annotations: Multi-label classification — 20 binary flags per image
Tasks: Multi-label retinal disease detection

Disease classes:
  DR    Diabetic Retinopathy
  NORMAL Normal/Healthy
  MH    Media Haze
  ODC   Optic Disc Change
  TSLN  Tessellated Lesions
  ARMD  Age-Related Macular Degeneration
  DN    Diabetic Neuropathy (Neovascularization)
  MYA   Myopia
  BRVO  Branch Retinal Vein Occlusion
  ODP   Optic Disc Pallor
  CRVO  Central Retinal Vein Occlusion
  CNV   Choroidal Neovascularization
  RS    Retinal Scar
  ODE   Optic Disc Edema
  LS    Laser Scar (Localized Sclerosis)
  CSR   Central Serous Retinopathy
  HTR   Hypertensive Retinopathy
  ASR   Arteriosclerotic Retinopathy
  CRS   Cotton-Wool Spots / Retinal Striae
  OTHER Other Abnormalities
"""

import asyncio
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Set
from uuid import UUID

from chaksudb.common.progress import ProgressTracker, OperationStatistics
from chaksudb.config.config import get_data_root
from chaksudb.db.models import Dataset, Image, ClassificationAnnotation
from chaksudb.db.queries import (
    upsert_dataset,
    bulk_upsert_images,
    bulk_upsert_classification_annotations,
)
from chaksudb.ingest.framework import get_image_metadata_dict
from chaksudb.ingest.framework.gen_uuid import (
    generate_dataset_uuid,
    generate_image_uuid,
)
from chaksudb.ingest.framework.ingestion_helpers import process_csv
from chaksudb.ingest.framework.split_assigner import (
    register_standard_splits,
    bulk_assign_images_to_split,
)
from chaksudb.ingest.framework.task_processors.classification_processor import (
    process_classification,
)

logger = logging.getLogger(__name__)

DATASET_NAME = "MuReD"
DATASET_URL = "https://data.mendeley.com/datasets/pc4mb3h8hz/1"
DATASET_LICENSE = "Research/Academic Use"

# Ordered list of disease columns in the CSV (preserves original MuReD naming)
DISEASE_COLUMNS = [
    "DR", "NORMAL", "MH", "ODC", "TSLN", "ARMD", "DN", "MYA", "BRVO",
    "ODP", "CRVO", "CNV", "RS", "ODE", "LS", "CSR", "HTR", "ASR", "CRS", "OTHER",
]


def find_image_file(image_dir: Path, image_id: str) -> Optional[Path]:
    """Return the image file for *image_id*, trying common extensions."""
    for ext in (".png", ".PNG", ".tif", ".TIF", ".jpg", ".JPG", ".jpeg", ".JPEG"):
        candidate = image_dir / f"{image_id}{ext}"
        if candidate.exists():
            return candidate
    return None


async def ingest_mured() -> OperationStatistics:
    """Main ingestion function for MuReD dataset."""
    data_root = get_data_root() / "46_MuReD"
    image_dir = data_root / "images"
    dataset_id = generate_dataset_uuid(DATASET_NAME)

    logger.info("=" * 80)
    logger.info(f"Starting ingestion: {DATASET_NAME}")
    logger.info(f"Data root: {data_root}")
    logger.info("=" * 80)

    # Step 1: Register dataset
    dataset = Dataset(
        dataset_id=dataset_id,
        dataset_name=DATASET_NAME,
        source_url=DATASET_URL,
        license=DATASET_LICENSE,
        modality_types=["fundus"],
        task_types=["classification"],
        description=(
            "MuReD (Multi-label Retinal Diseases) is a large-scale fundus image "
            "dataset with 20 disease/condition labels per image, covering Diabetic "
            "Retinopathy, ARMD, glaucoma-related changes, vein occlusions, myopia, "
            "and more. Images originate from multiple sources (ARIA subset included). "
            "Labels are binary multi-label annotations (presence/absence per class)."
        ),
    )
    await upsert_dataset(dataset)

    # Step 2: Determine annotated image IDs (to identify the unlabeled holdout set)
    annotated_ids: Set[str] = set()
    for csv_path in [data_root / "train_data.csv", data_root / "val_data.csv"]:
        if csv_path.exists():
            import csv as _csv
            with open(csv_path, newline="", encoding="utf-8") as f:
                for row in _csv.DictReader(f):
                    if row.get("ID"):
                        annotated_ids.add(row["ID"].strip())

    # All image files present on disk
    all_image_files = list(image_dir.glob("*"))
    all_image_files = [
        p for p in all_image_files
        if p.suffix.lower() in (".png", ".tif", ".jpg", ".jpeg")
    ]

    # Unlabeled test images (not in either CSV)
    unlabeled_ids = [
        p.stem for p in all_image_files if p.stem not in annotated_ids
    ]

    total_annotated = len(annotated_ids)
    total_unlabeled = len(unlabeled_ids)
    # Each annotated image: 1 image + 1 multi-label classification = 2 items
    tracker = ProgressTracker(
        total=total_annotated * 2 + total_unlabeled,
        description=f"Ingesting {DATASET_NAME}",
    )
    logger.info(f"Annotated images: {total_annotated} (train + val), unlabeled: {total_unlabeled}")

    # Step 3: Shared collections (asyncio is single-threaded; no locking needed)
    all_images: List[Image] = []
    all_classifications: List[ClassificationAnnotation] = []
    split_image_ids: Dict[str, List[UUID]] = {"train": [], "val": [], "test": []}

    # Step 4: Row processor for train and val CSVs
    def make_row_handler(split_name: str):
        async def process_row(row: dict, idx: int) -> None:
            image_id_str = row.get("ID", "").strip()
            if not image_id_str:
                tracker.update(success=False)
                tracker.record_error(
                    error_type="missing_id",
                    error_message="Empty ID in CSV row",
                    item_id=f"row_{idx}",
                )
                return

            image_path = find_image_file(image_dir, image_id_str)
            if image_path is None:
                tracker.update(success=False)
                tracker.record_error(
                    error_type="image_not_found",
                    error_message=f"No image file for ID: {image_id_str}",
                    item_id=image_id_str,
                )
                return

            image_id = generate_image_uuid(dataset_id, image_id_str)
            image = Image(
                image_id=image_id,
                dataset_id=dataset_id,
                original_image_id=image_id_str,
                **get_image_metadata_dict(image_path),
                modality="fundus",
            )
            all_images.append(image)
            split_image_ids[split_name].append(image_id)
            tracker.update(success=True)
            tracker.record_success("image")

            # Build multi-label dict — skip rows with missing/invalid labels
            labels: Dict[str, int] = {}
            for col in DISEASE_COLUMNS:
                val = row.get(col, "").strip()
                if val == "" or (isinstance(val, float) and math.isnan(val)):
                    continue
                try:
                    labels[col] = int(float(val))
                except (ValueError, TypeError):
                    logger.warning(f"Row {idx}: Invalid value for {col}: {val!r}")

            if labels:
                classifications = await process_classification(
                    class_value=labels,
                    task_type="multi_label",
                    task_name="disease_panel",
                    class_name="disease_panel",
                    image_id=image_id,
                    annotation_method="manual",
                )
                all_classifications.extend(classifications)
                tracker.update(success=True)
                tracker.record_success("classification")
            else:
                logger.warning(f"Row {idx}: No valid labels for {image_id_str}")
                tracker.update(success=False)
                tracker.record_error(
                    error_type="no_labels",
                    error_message="No valid label values",
                    item_id=image_id_str,
                )

        return process_row

    # Step 5: Process both CSVs in parallel (provenance auto-tracked per CSV)
    train_csv = data_root / "train_data.csv"
    val_csv = data_root / "val_data.csv"

    await asyncio.gather(
        process_csv(
            csv_path=train_csv,
            dataset_id=dataset_id,
            unified_annotation_type="classification",
            process_row_fn=make_row_handler("train"),
            progress_tracker=tracker,
            skip_errors=True,
        ),
        process_csv(
            csv_path=val_csv,
            dataset_id=dataset_id,
            unified_annotation_type="classification",
            process_row_fn=make_row_handler("val"),
            progress_tracker=tracker,
            skip_errors=True,
        ),
    )

    # Step 6: Collect unlabeled test images (image records only, no annotations)
    logger.info(f"Processing {total_unlabeled} unlabeled test images...")
    for stem in unlabeled_ids:
        image_path = next(
            (p for p in all_image_files if p.stem == stem), None
        )
        if image_path is None:
            continue
        image_id = generate_image_uuid(dataset_id, stem)
        image = Image(
            image_id=image_id,
            dataset_id=dataset_id,
            original_image_id=stem,
            **get_image_metadata_dict(image_path),
            modality="fundus",
        )
        all_images.append(image)
        split_image_ids["test"].append(image_id)
        tracker.update(success=True)
        tracker.record_success("image_unlabeled")

    # Step 7: Bulk upserts (images first)
    logger.info(f"Upserting {len(all_images)} images...")
    await bulk_upsert_images(all_images, batch_size=1000)

    logger.info(f"Upserting {len(all_classifications)} classification annotations...")
    await bulk_upsert_classification_annotations(all_classifications, batch_size=1000)

    # Step 8: Register splits and assign
    splits = await register_standard_splits(
        dataset_id=dataset_id,
        split_type="explicit",
        train_count=len(split_image_ids["train"]),
        val_count=len(split_image_ids["val"]),
        test_count=len(split_image_ids["test"]),
    )
    await asyncio.gather(
        bulk_assign_images_to_split(split_image_ids["train"], splits["train"]),
        bulk_assign_images_to_split(split_image_ids["val"], splits["val"]),
        bulk_assign_images_to_split(split_image_ids["test"], splits["test"]),
    )

    tracker.finish()
    stats = tracker.get_statistics()
    logger.info("=" * 80)
    logger.info("Ingestion Summary:")
    logger.info(f"  Total items: {stats.total_items}")
    logger.info(f"  Successful: {stats.successful_items}")
    logger.info(f"  Failed: {stats.failed_items}")
    logger.info(f"  Skipped: {stats.skipped_items}")
    if stats.item_counts:
        logger.info("  Breakdown:")
        for item_type, count in sorted(stats.item_counts.items()):
            logger.info(f"    {item_type}: {count}")
    logger.info("=" * 80)

    return stats


async def main():
    """Entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    try:
        stats = await ingest_mured()
        if stats.failed_items > 0:
            logger.error(f"Ingestion completed with {stats.failed_items} errors")
            return 1
        logger.info("Ingestion completed successfully!")
        return 0
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
