"""
Ingestion script for RITE dataset.

Dataset: RITE (Retinal Images vessel Tree Extraction)
Structure: Folder-based train/test split with paired TIF images and PNG masks
  - training/images/   20 TIF fundus photographs (IDs 21-40, from DRIVE training set)
  - training/vessel/   20 PNG binary vessel masks
  - training/av/       20 PNG color-coded artery/vein masks
  - test/images/       20 TIF fundus photographs (IDs 01-20, from DRIVE test set)
  - test/vessel/       20 PNG binary vessel masks
  - test/av/           20 PNG color-coded artery/vein masks
Annotations (stored identically to the other AV datasets via process_av_color_mask):
  - av      — color RGB mask (R=arteries, G=overlap, B=veins), unified_format="color_mask"
  - vessels — binary union of all vessel pixels, unified_format="binary_mask"
Source raw AV colours: red=arteries, blue=veins, green=overlap, white=uncertain.
Tasks: Retinal vessel segmentation and artery/vein classification

Source: Based on DRIVE database (Digital Retinal Images for Vessel Extraction)
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List
from uuid import UUID

from chaksudb.common.progress import ProgressTracker, OperationStatistics
from chaksudb.config.config import get_data_root
from chaksudb.db.models import Dataset, Image, SegmentationAnnotation
from chaksudb.db.queries import (
    upsert_dataset,
    bulk_upsert_images,
    upsert_segmentation_annotation,
)
from chaksudb.ingest.framework import get_image_metadata_dict
from chaksudb.ingest.framework.gen_uuid import (
    generate_dataset_uuid,
    generate_image_uuid,
)
from chaksudb.ingest.framework.split_assigner import (
    register_standard_splits,
    bulk_assign_images_to_split,
)
from chaksudb.ingest.framework.task_processors.av_segmentation import process_av_color_mask

logger = logging.getLogger(__name__)

DATASET_NAME = "RITE"
DATASET_URL = "https://medicine.uiowa.edu/eye/rite-dataset"
DATASET_LICENSE = "Research/Academic Use"


async def process_image(
    image_path: Path,
    av_mask_path: Path,
    split_name: str,
    dataset_id: UUID,
    tracker: ProgressTracker,
    all_images: List[Image],
    all_segmentations: List[SegmentationAnnotation],
    split_image_ids: Dict[str, List[UUID]],
) -> None:
    """Process a single image with its color AV mask.

    The color AV mask is the vessel map with arteries/veins/overlap/uncertain
    colour-coded, so process_av_color_mask derives both the 'av' colour mask and
    the binary 'vessels' union from it — the separate training/vessel/ binary file
    is redundant with that union and no longer ingested.
    """
    stem = image_path.stem  # e.g. "21_training"

    try:
        image_id = generate_image_uuid(dataset_id, stem)

        image = Image(
            image_id=image_id,
            dataset_id=dataset_id,
            original_image_id=stem,
            **get_image_metadata_dict(image_path),
            modality="fundus",
        )
        all_images.append(image)
        split_image_ids[split_name].append(image_id)
        tracker.update(success=True)
        tracker.record_success("image")

        # --- AV mask (color-coded RGB) -> 'av' (color) + 'vessels' (binary) ---
        if av_mask_path.exists():
            av_segs = await process_av_color_mask(
                av_mask_path=av_mask_path,
                image_id=image_id,
                dataset_id=dataset_id,
                dataset_name=DATASET_NAME,
                group_identifier=stem,
            )
            all_segmentations.extend(av_segs)
            for _ in av_segs:
                tracker.update(success=True)
                tracker.record_success("av_segmentation")
        else:
            logger.warning(f"AV mask not found: {av_mask_path}")
            tracker.record_error(
                error_type="mask_not_found",
                error_message=f"AV mask not found: {av_mask_path}",
                item_id=stem,
            )

    except Exception as e:
        tracker.update(success=False)
        tracker.record_error(
            error_type="image_processing",
            error_message=str(e),
            item_id=stem,
        )
        logger.error(f"Failed to process {stem}: {e}", exc_info=True)


async def ingest_rite() -> OperationStatistics:
    """Main ingestion function for RITE dataset."""
    data_root = get_data_root() / "45_RITE"
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
        task_types=["segmentation"],
        description=(
            "RITE (Retinal Images vessel Tree Extraction) enables comparative "
            "studies on segmentation and classification of arteries and veins on "
            "retinal fundus images. Built from DRIVE (Digital Retinal Images for "
            "Vessel Extraction), it provides vessel segmentation masks and "
            "color-coded artery/vein classification masks (red=arteries, "
            "blue=veins, green=overlapping, white=uncertain) for 40 images split "
            "equally into training and test subsets."
        ),
    )
    await upsert_dataset(dataset)

    # Step 2: Discover images per split
    splits_config = {
        "train": {
            "images": sorted((data_root / "training" / "images").glob("*.tif")),
            "av_dir": data_root / "training" / "av",
        },
        "test": {
            "images": sorted((data_root / "test" / "images").glob("*.tif")),
            "av_dir": data_root / "test" / "av",
        },
    }

    total_images = sum(len(v["images"]) for v in splits_config.values())
    # Each image: 1 image + 2 AV-derived segs (av color + vessels binary) = 3 items
    tracker = ProgressTracker(
        total=total_images * 3,
        description=f"Ingesting {DATASET_NAME}",
    )

    # Step 3: Collect all records
    all_images: List[Image] = []
    all_segmentations: List[SegmentationAnnotation] = []
    split_image_ids: Dict[str, List[UUID]] = {"train": [], "test": []}

    for split_name, config in splits_config.items():
        image_paths = config["images"]
        av_dir = config["av_dir"]
        logger.info(f"Processing {len(image_paths)} {split_name} images...")

        for image_path in image_paths:
            stem = image_path.stem  # e.g. "21_training" or "01_test"
            av_mask_path = av_dir / f"{stem}.png"

            await process_image(
                image_path=image_path,
                av_mask_path=av_mask_path,
                split_name=split_name,
                dataset_id=dataset_id,
                tracker=tracker,
                all_images=all_images,
                all_segmentations=all_segmentations,
                split_image_ids=split_image_ids,
            )

    # Step 4: Bulk upsert images (FK dependency for annotations)
    logger.info(f"Upserting {len(all_images)} images...")
    await bulk_upsert_images(all_images, batch_size=500)

    # Step 5: Upsert segmentations
    logger.info(f"Upserting {len(all_segmentations)} segmentation annotations...")
    for seg in all_segmentations:
        try:
            await upsert_segmentation_annotation(seg)
        except Exception as e:
            tracker.record_error(
                error_type="segmentation_upsert",
                error_message=str(e),
                item_id=str(seg.segmentation_id),
            )
            logger.error(f"Failed to upsert segmentation: {e}")

    # Step 6: Register explicit train/test splits
    splits = await register_standard_splits(
        dataset_id=dataset_id,
        split_type="explicit",
        train_count=len(split_image_ids["train"]),
        test_count=len(split_image_ids["test"]),
    )
    await asyncio.gather(
        bulk_assign_images_to_split(split_image_ids["train"], splits["train"]),
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
        stats = await ingest_rite()
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
