"""
Ingestion script for AV-DRIVE dataset.

Dataset: AV-DRIVE — Artery/vein classification masks for DRIVE fundus images.
Structure: train/test split mirroring DRIVE (IDs 21-40 train, 01-20 test).
    training/images/*.tif  — 20 fundus photographs
    training/av/*.png      — 20 color-coded AV masks
    test/images/*.tif      — 20 fundus photographs
    test/av/*.png          — 19 color-coded AV masks (one image has no mask)
Annotations:
    Color-coded AV masks (red=arteries, blue=veins, green=overlap, white=uncertain).
    Derived annotation types per image (shared group_id):
        arteries, veins, vessels_overlapping, vessels_uncertain,
        arteries_inclusive, veins_inclusive, vessels (binary union).
Tasks: Retinal artery/vein segmentation
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List
from uuid import UUID

from chaksudb.common.progress import OperationStatistics, ProgressTracker
from chaksudb.config.config import get_data_root
from chaksudb.db.models import Dataset, Image, SegmentationAnnotation
from chaksudb.db.queries import bulk_upsert_images, upsert_dataset, upsert_segmentation_annotation
from chaksudb.ingest.framework import get_image_metadata_dict
from chaksudb.ingest.framework.gen_uuid import generate_dataset_uuid, generate_image_uuid
from chaksudb.ingest.framework.split_assigner import (
    bulk_assign_images_to_split,
    register_standard_splits,
)
from chaksudb.ingest.framework.task_processors.av_segmentation import process_av_color_mask

logger = logging.getLogger(__name__)

DATASET_NAME = "AV-DRIVE"
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
    stem = image_path.stem
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
        tracker.record_success("image")

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
                tracker.record_success("av_segmentation")
        else:
            logger.warning("AV mask not found: %s", av_mask_path)
            tracker.record_error(
                error_type="mask_not_found",
                error_message=f"AV mask not found: {av_mask_path}",
                item_id=stem,
            )

        tracker.update(success=True)
    except Exception as exc:
        tracker.update(success=False)
        tracker.record_error(
            error_type="image_processing",
            error_message=str(exc),
            item_id=stem,
        )
        logger.error("Failed to process %s: %s", stem, exc, exc_info=True)


async def ingest_av_drive() -> OperationStatistics:
    data_root = get_data_root() / "49_AV_DRIVE"
    dataset_id = generate_dataset_uuid(DATASET_NAME)

    logger.info("=" * 80)
    logger.info("Starting ingestion: %s", DATASET_NAME)
    logger.info("Data root: %s", data_root)
    logger.info("=" * 80)

    dataset = Dataset(
        dataset_id=dataset_id,
        dataset_name=DATASET_NAME,
        source_url=DATASET_URL,
        license=DATASET_LICENSE,
        modality_types=["fundus"],
        task_types=["segmentation"],
        description=(
            "AV-DRIVE provides color-coded artery/vein classification masks for the "
            "DRIVE fundus image set. Each image is annotated with red=arteries, "
            "blue=veins, green=overlapping vessels, and white=uncertain vessels. "
            "Supports both AV segmentation and binary vessel segmentation at export time."
        ),
    )
    await upsert_dataset(dataset)

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
    tracker = ProgressTracker(total=total_images * 8, description=f"Ingesting {DATASET_NAME}")

    all_images: List[Image] = []
    all_segmentations: List[SegmentationAnnotation] = []
    split_image_ids: Dict[str, List[UUID]] = {"train": [], "test": []}

    for split_name, config in splits_config.items():
        for image_path in config["images"]:
            stem = image_path.stem
            av_mask_path = config["av_dir"] / f"{stem}.png"
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

    logger.info("Upserting %d images…", len(all_images))
    await bulk_upsert_images(all_images, batch_size=500)

    logger.info("Upserting %d segmentation annotations…", len(all_segmentations))
    for seg in all_segmentations:
        try:
            await upsert_segmentation_annotation(seg)
        except Exception as exc:
            logger.error("Failed to upsert segmentation %s: %s", seg.segmentation_id, exc)

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
    logger.info(
        "Ingestion summary — total: %d, success: %d, failed: %d",
        stats.total_items, stats.successful_items, stats.failed_items,
    )
    logger.info("=" * 80)
    return stats


async def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    try:
        stats = await ingest_av_drive()
        return 1 if stats.failed_items > 0 else 0
    except Exception as exc:
        logger.exception("Fatal error: %s", exc)
        return 1


if __name__ == "__main__":
    exit(asyncio.run(main()))
