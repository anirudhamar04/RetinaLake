"""
Ingestion script for DRIVE dataset.

Dataset: DRIVE - Digital Retinal Images for Vessel Extraction
Structure: train/test split with images (.tif), vessel masks (.gif), and FOV masks (.gif)
Annotations: Vessel segmentation (binary masks, 2 human observers on test, 1 on training)
Tasks: Binary vessel segmentation with multi-expert annotations
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List
from uuid import UUID

from chaksudb.common.progress import ProgressTracker, OperationStatistics
from chaksudb.config.config import get_data_root
from chaksudb.db.models import (
    Dataset,
    Image,
    SegmentationAnnotation,
    Expert,
    ExpertAnnotation,
)
from chaksudb.db.queries import (
    upsert_dataset,
    bulk_upsert_images,
    upsert_segmentation_annotation,
    upsert_expert,
    upsert_expert_annotation,
)
from chaksudb.ingest.framework import get_image_metadata_dict
from chaksudb.ingest.framework.gen_uuid import (
    generate_dataset_uuid,
    generate_image_uuid,
    generate_expert_uuid,
    generate_expert_annotation_uuid,
)
from chaksudb.ingest.framework.task_processors.segmentation_processor import (
    process_segmentation_from_binary_mask,
)
from chaksudb.ingest.framework.raw_file_helpers import register_individual_file
from chaksudb.ingest.framework.split_assigner import (
    register_standard_splits,
    bulk_assign_images_to_split,
)

logger = logging.getLogger(__name__)

# Dataset metadata
DATASET_NAME = "DRIVE"
DATASET_URL = "https://www.kaggle.com/datasets/andrewmvd/drive-digital-retinal-images-for-vessel-extraction"
DATASET_LICENSE = "Research/Academic Use"

# Expert metadata for vessel segmentation
VESSEL_EXPERTS = {
    "1st_observer": {
        "name": "1st Human Observer",
        "expertise": "Blood Vessel Segmentation",
    },
    "2nd_observer": {
        "name": "2nd Human Observer",
        "expertise": "Blood Vessel Segmentation",
    },
}

# Split directories and their naming patterns
SPLITS = {
    "train": {
        "dir": "training",
        "image_suffix": "_training.tif",
        "mask_suffix": "_training_mask.gif",
        "manual1_suffix": "_manual1.gif",
    },
    "test": {
        "dir": "test",
        "image_suffix": "_test.tif",
        "mask_suffix": "_test_mask.gif",
        "manual1_suffix": "_manual1.gif",
        "manual2_suffix": "_manual2.gif",
    },
}


async def register_experts(dataset_id: UUID) -> Dict[str, UUID]:
    """Register vessel segmentation experts."""
    expert_ids = {}
    for key, info in VESSEL_EXPERTS.items():
        expert_id = generate_expert_uuid(
            dataset_id=dataset_id,
            model_id=None,
            expert_name=info["name"],
        )
        expert = Expert(
            expert_id=expert_id,
            expert_name=info["name"],
            dataset_id=dataset_id,
            model_id=None,
            expertise_area=info["expertise"],
        )
        await upsert_expert(expert)
        expert_ids[key] = expert_id
        logger.info(f"Registered expert: {info['name']} ({key})")
    return expert_ids


async def ingest_drive() -> OperationStatistics:
    """Main ingestion function for DRIVE dataset."""
    data_root = get_data_root() / "42_DRIVE"
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
            "DRIVE (Digital Retinal Images for Vessel Extraction) contains 40 retinal "
            "fundus images with manual blood vessel segmentation masks. The dataset is "
            "divided into training (20 images, 1 observer) and test (20 images, 2 observers) "
            "sets. Images are from a diabetic retinopathy screening program."
        ),
    )
    await upsert_dataset(dataset)

    # Step 2: Register experts
    expert_ids = await register_experts(dataset_id)

    # Step 3: Discover and process images from both splits
    all_images: List[Image] = []
    image_id_map: Dict[str, UUID] = {}  # original_id -> image_id
    train_ids: List[UUID] = []
    test_ids: List[UUID] = []

    # Count total items for progress tracker (images + masks)
    # Train: 20 images + ~19 manual1 masks = ~39
    # Test: 20 images + 20 manual1 masks + ~19 manual2 masks = ~59
    tracker = ProgressTracker(
        total=98,  # approximate; updated as we go
        description=f"Ingesting {DATASET_NAME}",
    )

    for split_name, split_info in SPLITS.items():
        split_dir = data_root / split_info["dir"]
        image_dir = split_dir / "images"

        # Find all images in this split
        image_files = sorted(image_dir.glob("*.tif"))
        logger.info(f"Found {len(image_files)} images in {split_name} split")

        for image_path in image_files:
            try:
                # Extract numeric ID from filename: "21_training.tif" -> "21"
                stem = image_path.stem
                numeric_id = stem.split("_")[0]
                original_id = f"{numeric_id}_{split_name}"

                image_id = generate_image_uuid(dataset_id, original_id)
                image = Image(
                    image_id=image_id,
                    dataset_id=dataset_id,
                    original_image_id=original_id,
                    **get_image_metadata_dict(image_path),
                    modality="fundus",
                    eye_laterality="unknown",
                )
                all_images.append(image)
                image_id_map[numeric_id] = image_id

                if split_name == "train":
                    train_ids.append(image_id)
                else:
                    test_ids.append(image_id)

                tracker.update(success=True)
                tracker.record_success("image")
            except Exception as e:
                tracker.update(success=False)
                tracker.record_error(
                    error_type="image_processing",
                    error_message=str(e),
                    item_id=image_path.name,
                )
                logger.error(f"Failed to process image {image_path.name}: {e}")

    # Bulk upsert images first (FK constraint)
    if all_images:
        logger.info(f"Upserting {len(all_images)} images...")
        await bulk_upsert_images(all_images, batch_size=500)

    # Step 4: Process vessel segmentation masks
    all_segmentations: List[SegmentationAnnotation] = []

    for split_name, split_info in SPLITS.items():
        split_dir = data_root / split_info["dir"]

        # 1st observer masks (both train and test)
        manual1_dir = split_dir / "1st_manual"
        manual1_masks = sorted(manual1_dir.glob("*_manual1.gif"))
        logger.info(f"Found {len(manual1_masks)} 1st observer masks in {split_name}")

        for mask_path in manual1_masks:
            try:
                # "22_manual1.gif" -> "22"
                numeric_id = mask_path.stem.split("_")[0]

                if numeric_id not in image_id_map:
                    tracker.update(success=False)
                    tracker.record_error(
                        error_type="image_not_found",
                        error_message=f"No image for mask: {mask_path.name}",
                        item_id=mask_path.name,
                    )
                    continue

                image_id = image_id_map[numeric_id]
                expert_id = expert_ids["1st_observer"]

                raw_file_id, chain_id = await register_individual_file(
                    file_path=mask_path,
                    dataset_id=dataset_id,
                    unified_annotation_type="segmentation",
                    auto_detect_type=False,
                )

                expert_annotation_id = generate_expert_annotation_uuid(
                    expert_id=expert_id,
                    annotation_task="segmentation",
                    raw_data_id=raw_file_id,
                    annotation_value_hash=None,
                )
                expert_annotation = ExpertAnnotation(
                    expert_annotation_id=expert_annotation_id,
                    expert_id=expert_id,
                    annotation_task="segmentation",
                    raw_data_id=raw_file_id,
                    annotation_value=None,
                    confidence_level=None,
                    annotation_timestamp=None,
                )
                await upsert_expert_annotation(expert_annotation)

                segmentation = await process_segmentation_from_binary_mask(
                    mask_path=mask_path,
                    annotation_type="vessels",
                    image_id=image_id,
                    annotation_description="Blood vessel segmentation by 1st Human Observer",
                    raw_data_id=raw_file_id,
                    expert_annotation_id=expert_annotation_id,
                    annotation_method="manual",
                    provenance_chain_id=chain_id,
                    dataset_name=DATASET_NAME,
                )
                all_segmentations.append(segmentation)
                tracker.update(success=True)
                tracker.record_success("vessel_seg_1st")
            except Exception as e:
                tracker.update(success=False)
                tracker.record_error(
                    error_type="vessel_seg_1st",
                    error_message=str(e),
                    item_id=mask_path.name,
                )
                logger.error(f"Failed to process 1st observer mask {mask_path.name}: {e}")

        # 2nd observer masks (test split only)
        if "manual2_suffix" in split_info:
            manual2_dir = split_dir / "2nd_manual"
            manual2_masks = sorted(manual2_dir.glob("*_manual2.gif"))
            logger.info(f"Found {len(manual2_masks)} 2nd observer masks in {split_name}")

            for mask_path in manual2_masks:
                try:
                    numeric_id = mask_path.stem.split("_")[0]

                    if numeric_id not in image_id_map:
                        tracker.update(success=False)
                        tracker.record_error(
                            error_type="image_not_found",
                            error_message=f"No image for mask: {mask_path.name}",
                            item_id=mask_path.name,
                        )
                        continue

                    image_id = image_id_map[numeric_id]
                    expert_id = expert_ids["2nd_observer"]

                    raw_file_id, chain_id = await register_individual_file(
                        file_path=mask_path,
                        dataset_id=dataset_id,
                        unified_annotation_type="segmentation",
                        auto_detect_type=False,
                    )

                    expert_annotation_id = generate_expert_annotation_uuid(
                        expert_id=expert_id,
                        annotation_task="segmentation",
                        raw_data_id=raw_file_id,
                        annotation_value_hash=None,
                    )
                    expert_annotation = ExpertAnnotation(
                        expert_annotation_id=expert_annotation_id,
                        expert_id=expert_id,
                        annotation_task="segmentation",
                        raw_data_id=raw_file_id,
                        annotation_value=None,
                        confidence_level=None,
                        annotation_timestamp=None,
                    )
                    await upsert_expert_annotation(expert_annotation)

                    segmentation = await process_segmentation_from_binary_mask(
                        mask_path=mask_path,
                        annotation_type="vessels",
                        image_id=image_id,
                        annotation_description="Blood vessel segmentation by 2nd Human Observer",
                        raw_data_id=raw_file_id,
                        expert_annotation_id=expert_annotation_id,
                        annotation_method="manual",
                        provenance_chain_id=chain_id,
                        dataset_name=DATASET_NAME,
                    )
                    all_segmentations.append(segmentation)
                    tracker.update(success=True)
                    tracker.record_success("vessel_seg_2nd")
                except Exception as e:
                    tracker.update(success=False)
                    tracker.record_error(
                        error_type="vessel_seg_2nd",
                        error_message=str(e),
                        item_id=mask_path.name,
                    )
                    logger.error(f"Failed to process 2nd observer mask {mask_path.name}: {e}")

    # Step 5: Upsert segmentations
    logger.info(f"Upserting {len(all_segmentations)} vessel segmentation annotations...")
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

    # Step 6: Register splits and assign images
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

    # Step 7: Summary
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
        stats = await ingest_drive()
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
