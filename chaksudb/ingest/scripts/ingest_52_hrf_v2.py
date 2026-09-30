"""
Ingestion script for HRF-v2 dataset.

Dataset: HRF-v2 — High-Resolution Fundus (version 2) with AV segmentation.
Structure: train/test split; disease type encoded in filename stem.
    training/images/*.png  — 29 fundus photographs (01_dr missing vs v1)
    training/av/*.png      — 27 color-coded AV masks
    test/images/*.png      — 15 fundus photographs
    test/av/*.png          — 15 color-coded AV masks
Filename convention: NN_XX.png where XX in {dr, g, h}
    dr = diabetic retinopathy, g = glaucoma, h = healthy
Annotations:
    Color-coded AV masks (red=arteries, blue=veins, green=overlap, white=uncertain).
    Binary disease classification from filename code.
    Derived annotation types per image (shared group_id):
        arteries, veins, vessels_overlapping, vessels_uncertain,
        arteries_inclusive, veins_inclusive, vessels (binary union).
Tasks: Retinal AV segmentation; disease classification.
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List
from uuid import UUID

from chaksudb.common.progress import OperationStatistics, ProgressTracker
from chaksudb.config.config import get_data_root
from chaksudb.db.models import ClassificationAnnotation, Dataset, Image, SegmentationAnnotation
from chaksudb.db.queries import (
    bulk_upsert_classification_annotations,
    bulk_upsert_images,
    upsert_dataset,
    upsert_segmentation_annotation,
)
from chaksudb.ingest.framework import get_image_metadata_dict
from chaksudb.ingest.framework.gen_uuid import generate_dataset_uuid, generate_image_uuid
from chaksudb.ingest.framework.split_assigner import (
    bulk_assign_images_to_split,
    register_standard_splits,
)
from chaksudb.ingest.framework.task_processors.av_segmentation import process_av_color_mask
from chaksudb.ingest.framework.task_processors.classification_processor import process_classification

logger = logging.getLogger(__name__)

DATASET_NAME = "HRF-v2"
DATASET_URL = "https://github.com/rubenhx/av-segmentation"
DATASET_LICENSE = "Research/Academic Use"

_DISEASE_SUFFIX: Dict[str, List] = {
    "dr": [("DR", True)],
    "g":  [("glaucoma", True)],
    "h":  [],
}


def _disease_suffix(stem: str) -> str:
    return stem.rsplit("_", 1)[-1].lower()


async def _process_image(
    image_path: Path,
    av_dir: Path,
    split_name: str,
    dataset_id: UUID,
    tracker: ProgressTracker,
    all_images: List[Image],
    all_segmentations: List[SegmentationAnnotation],
    all_classifications: List[ClassificationAnnotation],
    split_image_ids: Dict[str, List[UUID]],
) -> None:
    stem = image_path.stem
    av_mask_path = av_dir / image_path.name

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

        suffix = _disease_suffix(stem)
        for cls_name, value in _DISEASE_SUFFIX.get(suffix, []):
            cls_anns = await process_classification(
                class_value=value,
                task_type="binary",
                class_name=cls_name,
                image_id=image_id,
            )
            all_classifications.extend(cls_anns)

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


async def ingest_hrf_v2() -> OperationStatistics:
    data_root = get_data_root() / "52_HRF-v2"
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
        task_types=["segmentation", "classification"],
        description=(
            "HRF-v2 (High-Resolution Fundus version 2) provides 44 fundus images "
            "(29 training, 15 test) with color-coded artery/vein segmentation masks. "
            "Disease categories: DR (diabetic retinopathy), glaucoma, and healthy. "
            "AV masks use red=arteries, blue=veins, green=overlap, white=uncertain."
        ),
    )
    await upsert_dataset(dataset)

    splits_config = {
        "train": {
            "images": sorted((data_root / "training" / "images").glob("*.png")),
            "av_dir": data_root / "training" / "av",
        },
        "test": {
            "images": sorted((data_root / "test" / "images").glob("*.png")),
            "av_dir": data_root / "test" / "av",
        },
    }

    total_images = sum(len(v["images"]) for v in splits_config.values())
    tracker = ProgressTracker(total=total_images * 9, description=f"Ingesting {DATASET_NAME}")

    all_images: List[Image] = []
    all_segmentations: List[SegmentationAnnotation] = []
    all_classifications: List[ClassificationAnnotation] = []
    split_image_ids: Dict[str, List[UUID]] = {"train": [], "test": []}

    for split_name, config in splits_config.items():
        for image_path in config["images"]:
            await _process_image(
                image_path=image_path,
                av_dir=config["av_dir"],
                split_name=split_name,
                dataset_id=dataset_id,
                tracker=tracker,
                all_images=all_images,
                all_segmentations=all_segmentations,
                all_classifications=all_classifications,
                split_image_ids=split_image_ids,
            )

    logger.info("Upserting %d images…", len(all_images))
    await bulk_upsert_images(all_images, batch_size=500)

    logger.info("Upserting %d segmentation annotations…", len(all_segmentations))
    for seg in all_segmentations:
        try:
            await upsert_segmentation_annotation(seg)
        except Exception as exc:
            logger.error("Failed to upsert segmentation: %s", exc)

    if all_classifications:
        logger.info("Upserting %d classification annotations…", len(all_classifications))
        await bulk_upsert_classification_annotations(all_classifications, batch_size=500)

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
        stats = await ingest_hrf_v2()
        return 1 if stats.failed_items > 0 else 0
    except Exception as exc:
        logger.exception("Fatal error: %s", exc)
        return 1


if __name__ == "__main__":
    exit(asyncio.run(main()))
