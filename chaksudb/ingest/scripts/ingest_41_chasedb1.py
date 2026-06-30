"""
Ingestion script for CHASE_DB1 dataset.

Dataset: CHASE_DB1 - Child Heart and Health Study in England retinal vessel segmentation
Structure: Flat directory with 28 fundus images and paired binary vessel masks from 2 observers
Annotations: Vessel segmentation (binary masks, 2 human observers: 1stHO, 2ndHO)
Tasks: Binary vessel segmentation with multi-expert annotations
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
from chaksudb.ingest.framework.split_assigner import auto_stratified_splits

logger = logging.getLogger(__name__)

# Dataset metadata
DATASET_NAME = "CHASEDB1"
DATASET_URL = "https://www.kaggle.com/datasets/khoongweihao/chasedb1"
DATASET_LICENSE = "Research/Academic Use"

# Expert metadata for vessel segmentation
VESSEL_EXPERTS = {
    "1stHO": {
        "name": "1st Human Observer",
        "expertise": "Blood Vessel Segmentation",
    },
    "2ndHO": {
        "name": "2nd Human Observer",
        "expertise": "Blood Vessel Segmentation",
    },
}

# Image naming pattern: Image_NNX.jpg where NN=patient(01-14), X=L/R
IMAGE_PATTERN = re.compile(r"^Image_(\d{2})(L|R)$")


def parse_image_stem(stem: str) -> Optional[tuple[str, str]]:
    """Parse image stem into (patient_id, laterality). Returns None if not matched."""
    m = IMAGE_PATTERN.match(stem)
    if m:
        return m.group(1), "left" if m.group(2) == "L" else "right"
    return None


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


async def ingest_chasedb1() -> OperationStatistics:
    """Main ingestion function for CHASE_DB1 dataset."""
    data_root = get_data_root() / "41_CHASEDB1"
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
            "CHASE_DB1 (Child Heart and Health Study in England) contains 28 retinal "
            "fundus images from 14 children (left and right eyes). Blood vessel "
            "segmentation masks are provided by two independent human observers."
        ),
    )
    await upsert_dataset(dataset)

    # Step 2: Register experts
    expert_ids = await register_experts(dataset_id)

    # Step 3: Discover images
    image_paths = sorted(data_root.glob("Image_*.jpg"))
    # 28 images + 56 masks = 84 items
    tracker = ProgressTracker(
        total=len(image_paths) * 3,  # each image + 2 masks
        description=f"Ingesting {DATASET_NAME}",
    )

    # Step 4: Process images
    all_images: List[Image] = []
    image_id_map: Dict[str, UUID] = {}
    all_image_ids: List[UUID] = []

    for image_path in image_paths:
        try:
            parsed = parse_image_stem(image_path.stem)
            if not parsed:
                tracker.update(success=False)
                tracker.record_error(
                    error_type="filename_parse",
                    error_message=f"Cannot parse filename: {image_path.name}",
                    item_id=image_path.name,
                )
                continue

            patient_num, laterality = parsed
            original_id = image_path.stem  # e.g. "Image_01L"

            image_id = generate_image_uuid(dataset_id, original_id)
            image = Image(
                image_id=image_id,
                dataset_id=dataset_id,
                original_image_id=original_id,
                **get_image_metadata_dict(image_path),
                modality="fundus",
                eye_laterality=laterality,
            )
            all_images.append(image)
            image_id_map[original_id] = image_id
            all_image_ids.append(image_id)
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

    # Step 5: Process vessel segmentation masks
    all_segmentations: List[SegmentationAnnotation] = []

    for expert_key in ["1stHO", "2ndHO"]:
        mask_paths = sorted(data_root.glob(f"Image_*_{expert_key}.png"))
        logger.info(f"Found {len(mask_paths)} vessel masks for {expert_key}")

        for mask_path in mask_paths:
            try:
                # Extract image identifier: Image_01L_1stHO.png -> Image_01L
                image_stem = mask_path.stem.replace(f"_{expert_key}", "")

                if image_stem not in image_id_map:
                    tracker.update(success=False)
                    tracker.record_error(
                        error_type="image_not_found",
                        error_message=f"No image for mask: {mask_path.name}",
                        item_id=mask_path.name,
                    )
                    continue

                image_id = image_id_map[image_stem]
                expert_id = expert_ids[expert_key]

                # Register mask file for provenance
                raw_file_id, chain_id = await register_individual_file(
                    file_path=mask_path,
                    dataset_id=dataset_id,
                    unified_annotation_type="segmentation",
                    auto_detect_type=False,
                )

                # Create expert annotation record
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

                # Process binary vessel mask
                segmentation = await process_segmentation_from_binary_mask(
                    mask_path=mask_path,
                    annotation_type="vessels",
                    image_id=image_id,
                    annotation_description=f"Blood vessel segmentation by {VESSEL_EXPERTS[expert_key]['name']}",
                    raw_data_id=raw_file_id,
                    expert_annotation_id=expert_annotation_id,
                    annotation_method="manual",
                    provenance_chain_id=chain_id,
                    dataset_name=DATASET_NAME,
                )
                all_segmentations.append(segmentation)
                tracker.update(success=True)
                tracker.record_success("vessel_segmentation")
            except Exception as e:
                tracker.update(success=False)
                tracker.record_error(
                    error_type="vessel_segmentation_processing",
                    error_message=str(e),
                    item_id=mask_path.name,
                )
                logger.error(f"Failed to process vessel mask {mask_path.name}: {e}")

    # Step 6: Upsert segmentations
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

    # Step 7: Register splits — random 90/10 train+test, then 90/10 train+val
    if all_image_ids:
        await auto_stratified_splits(
            dataset_id=dataset_id,
            split_assignments={"train": all_image_ids},
            split_type="undefined",
        )

    # Step 8: Summary
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
        stats = await ingest_chasedb1()
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
