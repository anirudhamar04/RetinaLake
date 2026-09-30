"""
Ingestion script for RIM-ONE DL dataset.

Dataset: RIM-ONE DL - Retinal IMage database for Optic Nerve Evaluation (Deep Learning edition)
Structure: Two partitioning schemes (by_hospital, randomly); images in train/test × glaucoma/normal
           folders. Flat segmentation directory per class with Cup and Disc binary masks.
Annotations:
  - Binary glaucoma classification (from folder name)
  - Optic disc and optic cup segmentation (binary PNG masks, one expert per image)
Tasks: Binary glaucoma classification, Segmentation (optic disc, optic cup)
Split: partitioned_by_hospital used as primary (avoids hospital-level data leakage)
"""

import asyncio
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional
from uuid import UUID

from chaksudb.common.progress import ProgressTracker, OperationStatistics
from chaksudb.config.config import get_data_root
from chaksudb.db.models import (
    ClassificationAnnotation,
    Dataset,
    Image,
    SegmentationAnnotation,
)
from chaksudb.db.queries import (
    bulk_upsert_classification_annotations,
    bulk_upsert_images,
    upsert_dataset,
    upsert_segmentation_annotation,
)
from chaksudb.ingest.framework import get_image_metadata_dict
from chaksudb.ingest.framework.gen_uuid import (
    generate_dataset_uuid,
    generate_image_uuid,
)
from chaksudb.ingest.framework.raw_file_helpers import register_individual_file
from chaksudb.ingest.framework.split_assigner import (
    bulk_assign_images_to_split,
    register_standard_splits,
)
from chaksudb.ingest.framework.task_processors.classification_processor import (
    process_classification,
)
from chaksudb.ingest.framework.task_processors.segmentation_processor import (
    process_segmentation_from_binary_mask,
)

logger = logging.getLogger(__name__)

DATASET_NAME = "RIM-ONE-DL"
DATASET_URL = "https://github.com/miag-ull/rim-one-dl"
DATASET_LICENSE = "Creative Commons Attribution 4.0 International (CC BY 4.0)"

# Segmentation structures: folder key → annotation_type
SEGMENTATION_STRUCTURES = {
    "Cup": "optic_cup",
    "Disc": "optic_disc",
}


async def ingest_rim_one() -> OperationStatistics:
    """Main ingestion function for RIM-ONE DL dataset."""
    data_root = get_data_root() / "44_RIM-ONE"
    dataset_id = generate_dataset_uuid(DATASET_NAME)

    logger.info("=" * 80)
    logger.info(f"Starting ingestion: {DATASET_NAME}")
    logger.info(f"Data root: {data_root}")
    logger.info("=" * 80)

    await upsert_dataset(
        Dataset(
            dataset_id=dataset_id,
            dataset_name=DATASET_NAME,
            source_url=DATASET_URL,
            license=DATASET_LICENSE,
            modality_types=["fundus"],
            task_types=["classification", "segmentation"],
            description=(
                "RIM-ONE DL is a retinal fundus image dataset for glaucoma detection "
                "with 485 images (172 glaucoma, 313 normal) from multiple hospitals. "
                "Each image has optic disc and optic cup binary segmentation masks "
                "from one expert annotator. Two partitioning schemes are provided; "
                "this ingestion uses partitioned_by_hospital for train/test splits."
            ),
        )
    )

    # Discover images via partitioned_by_hospital
    partition_dir = data_root / "RIM-ONE_DL_images" / "partitioned_by_hospital"
    seg_root = data_root / "RIM-ONE_DL_reference_segmentations"

    # Collect all images per split, preserving class label
    # structure: {split: [(image_path, class_label)]}
    split_items: Dict[str, List[tuple]] = {"train": [], "test": []}
    split_dir_map = {"train": "training_set", "test": "test_set"}

    for split_key, dir_name in split_dir_map.items():
        for class_label in ["glaucoma", "normal"]:
            class_dir = partition_dir / dir_name / class_label
            if not class_dir.exists():
                continue
            for img_path in sorted(class_dir.glob("*.png")):
                split_items[split_key].append((img_path, class_label))

    total_images = sum(len(v) for v in split_items.values())
    # Each image: 1 classification + up to 2 segmentation masks
    total_count = total_images * 3
    logger.info(
        f"Found {total_images} images "
        f"(train: {len(split_items['train'])}, test: {len(split_items['test'])})"
    )

    tracker = ProgressTracker(total=total_count, description=f"Ingesting {DATASET_NAME}")

    all_images: List[Image] = []
    all_classifications: List[ClassificationAnnotation] = []
    all_segmentations: List[SegmentationAnnotation] = []
    train_ids: List[UUID] = []
    test_ids: List[UUID] = []

    for split_key, items in split_items.items():
        for image_path, class_label in items:
            stem = image_path.stem  # e.g. "r2_Im347"
            original_id = stem

            # ── Image ──────────────────────────────────────────────────
            image_id = generate_image_uuid(dataset_id, original_id)
            image = Image(
                image_id=image_id,
                dataset_id=dataset_id,
                original_image_id=original_id,
                **get_image_metadata_dict(image_path),
                modality="fundus",
            )
            all_images.append(image)
            if split_key == "train":
                train_ids.append(image_id)
            else:
                test_ids.append(image_id)

            # ── Classification ─────────────────────────────────────────
            try:
                clfs = await process_classification(
                    class_value=(class_label == "glaucoma"),
                    task_type="binary",
                    class_name="glaucoma",
                    image_id=image_id,
                    annotation_method="manual",
                )
                all_classifications.extend(clfs)
                tracker.update(success=True)
                tracker.record_success("classification")
            except Exception as e:
                tracker.update(success=False)
                tracker.record_error(
                    error_type="classification",
                    error_message=str(e),
                    item_id=original_id,
                )
                logger.error(f"Classification failed for {stem}: {e}")

            # ── Segmentation masks ─────────────────────────────────────
            seg_class_dir = seg_root / class_label
            for structure_key, annotation_type in SEGMENTATION_STRUCTURES.items():
                mask_path = seg_class_dir / f"{stem}-1-{structure_key}-T.png"
                if not mask_path.exists():
                    tracker.update(success=False)
                    tracker.record_error(
                        error_type="mask_not_found",
                        error_message=f"Missing {structure_key} mask: {mask_path.name}",
                        item_id=original_id,
                    )
                    continue

                try:
                    raw_id, chain = await register_individual_file(
                        file_path=mask_path,
                        dataset_id=dataset_id,
                        unified_annotation_type="segmentation",
                        auto_detect_type=False,
                    )
                    seg = await process_segmentation_from_binary_mask(
                        mask_path=mask_path,
                        annotation_type=annotation_type,
                        image_id=image_id,
                        annotation_description=f"Optic {'cup' if structure_key == 'Cup' else 'disc'} segmentation",
                        raw_data_id=raw_id,
                        provenance_chain_id=chain,
                        annotation_method="manual",
                        dataset_name=DATASET_NAME,
                    )
                    all_segmentations.append(seg)
                    tracker.update(success=True)
                    tracker.record_success(f"seg_{annotation_type}")
                except Exception as e:
                    tracker.update(success=False)
                    tracker.record_error(
                        error_type=f"seg_{annotation_type}",
                        error_message=str(e),
                        item_id=original_id,
                    )
                    logger.error(f"Segmentation ({structure_key}) failed for {stem}: {e}")

    # ── Upsert (FK order) ─────────────────────────────────────────────
    logger.info(f"Upserting {len(all_images)} images...")
    await bulk_upsert_images(all_images, batch_size=500)

    logger.info(f"Upserting {len(all_classifications)} classifications and {len(all_segmentations)} segmentations...")
    await bulk_upsert_classification_annotations(all_classifications, batch_size=500)

    for seg in all_segmentations:
        await upsert_segmentation_annotation(seg)

    # ── Split assignment ──────────────────────────────────────────────
    splits = await register_standard_splits(
        dataset_id=dataset_id,
        split_type="explicit",
        train_count=len(train_ids),
        test_count=len(test_ids),
    )
    await asyncio.gather(
        bulk_assign_images_to_split(train_ids, splits["train"]),
        bulk_assign_images_to_split(test_ids, splits["test"]),
    )

    # ── Summary ───────────────────────────────────────────────────────
    tracker.finish()
    stats = tracker.get_statistics()
    logger.info("=" * 80)
    logger.info("Ingestion Summary:")
    logger.info(f"  Total items: {stats.total_items}")
    logger.info(f"  Successful: {stats.successful_items}")
    logger.info(f"  Failed: {stats.failed_items}")
    logger.info(f"  Skipped: {stats.skipped_items}")
    logger.info(f"  Images: {len(all_images)}")
    logger.info(f"  Classifications: {len(all_classifications)}")
    logger.info(f"  Segmentations: {len(all_segmentations)}")
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
        stats = await ingest_rim_one()
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
