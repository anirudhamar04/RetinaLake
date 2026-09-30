"""
Ingestion script for Fundus-AVSeg dataset.

Dataset: Fundus-AVSeg — 100 fundus images with AV segmentation and disease labels.
Structure: flat directories with split defined in text files.
    images/NNN_X.png       — fundus photographs (X = N/D/G/A disease code)
    annotation/NNN_X.png   — matching color-coded AV masks
    training.txt           — 80 image filenames in training set
    testing.txt            — 20 image filenames in test set
Annotations:
    Color-coded AV masks (red=arteries, blue=veins, green=overlap).
    Disease classification from filename code:
        N → Normal, D → DR, G → Glaucoma, A → AMD
    Derived annotation types per image (shared group_id):
        arteries, veins, vessels_overlapping,
        arteries_inclusive, veins_inclusive, vessels (binary union).
Tasks: Retinal AV segmentation; multi-disease classification.
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional
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

DATASET_NAME = "Fundus-AVSeg"
DATASET_URL = "https://figshare.com/projects/Fundus-AVSeg_A_Fundus_Image_Dataset_for_AI-based_Artery-Vein_Vessel_Segmentation/229986"
DATASET_LICENSE = "Research/Academic Use"

# Disease code in filename → (class_name, bool_value for presence)
_DISEASE_CODES = {
    "N": [],  # Normal — no disease present
    "D": [("DR", True)],
    "G": [("glaucoma", True)],
    "A": [("AMD", True)],
}


def _parse_disease_code(filename_stem: str) -> Optional[str]:
    """Extract the single-letter disease code from a stem like '093_N'."""
    parts = filename_stem.split("_")
    return parts[-1] if parts else None


async def process_image(
    image_path: Path,
    annotation_dir: Path,
    split_name: str,
    dataset_id: UUID,
    tracker: ProgressTracker,
    all_images: List[Image],
    all_segmentations: List[SegmentationAnnotation],
    all_classifications: List[ClassificationAnnotation],
    split_image_ids: Dict[str, List[UUID]],
) -> None:
    stem = image_path.stem  # e.g. "093_N"
    av_mask_path = annotation_dir / image_path.name
    code = _parse_disease_code(stem)

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
        if split_name:
            split_image_ids[split_name].append(image_id)
        tracker.record_success("image")

        # Classification from filename disease code
        if code in _DISEASE_CODES:
            for disease_annotations in _DISEASE_CODES[code]:
                cls_name, present = disease_annotations
                cls_anns = await process_classification(
                    class_value=present,
                    task_type="binary",
                    class_name=cls_name,
                    image_id=image_id,
                )
                all_classifications.extend(cls_anns)

        # AV segmentation
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
            logger.warning("AV mask not found for %s", stem)
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


async def ingest_fundus_avseg() -> OperationStatistics:
    data_root = get_data_root() / "50_Fundus-AVSeg"
    dataset_id = generate_dataset_uuid(DATASET_NAME)
    images_dir = data_root / "images"
    annotation_dir = data_root / "annotation"

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
            "Fundus-AVSeg provides 100 fundus images with color-coded artery/vein "
            "segmentation masks and disease labels (Normal, DR, Glaucoma, AMD). "
            "AV masks use red=arteries, blue=veins, green=overlap convention. "
            "80 training / 20 test split."
        ),
    )
    await upsert_dataset(dataset)

    # Read split files
    def _read_names(txt_path: Path) -> List[str]:
        return [line.strip() for line in txt_path.read_text().splitlines() if line.strip()]

    train_names = set(_read_names(data_root / "training.txt"))
    test_names = set(_read_names(data_root / "testing.txt"))

    all_image_paths = sorted(images_dir.glob("*.png"))
    total = len(all_image_paths)
    tracker = ProgressTracker(total=total * 9, description=f"Ingesting {DATASET_NAME}")

    all_images: List[Image] = []
    all_segmentations: List[SegmentationAnnotation] = []
    all_classifications: List[ClassificationAnnotation] = []
    split_image_ids: Dict[str, List[UUID]] = {"train": [], "test": []}

    for image_path in all_image_paths:
        fname = image_path.name
        if fname in train_names:
            split_name = "train"
        elif fname in test_names:
            split_name = "test"
        else:
            split_name = ""  # no explicit split — skip split assignment

        await process_image(
            image_path=image_path,
            annotation_dir=annotation_dir,
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

    if split_image_ids["train"] or split_image_ids["test"]:
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
        stats = await ingest_fundus_avseg()
        return 1 if stats.failed_items > 0 else 0
    except Exception as exc:
        logger.exception("Fatal error: %s", exc)
        return 1


if __name__ == "__main__":
    exit(asyncio.run(main()))
