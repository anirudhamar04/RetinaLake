"""
Ingestion script for AIROGS dataset.

Dataset: AIROGS - Artificial Intelligence for Referable Glaucoma Screening
Structure: train_labels.csv with challenge_id and class (RG/NRG)
Annotations: Binary glaucoma classification (Referable Glaucoma vs Non-Referable Glaucoma)
Tasks: Binary classification (glaucoma: RG = True, NRG = False)
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List
from uuid import UUID

from chaksudb.common.progress import ProgressTracker, OperationStatistics
from chaksudb.config.config import get_data_root
from chaksudb.db.models import Dataset, Image
from chaksudb.db.queries import (
    upsert_dataset,
    bulk_upsert_images,
    bulk_upsert_classification_annotations,
)
from chaksudb.ingest.framework import (
    process_csv,
    read_csv_auto,
    get_image_metadata_dict,
)
from chaksudb.ingest.framework.gen_uuid import (
    generate_dataset_uuid,
    generate_image_uuid,
)
from chaksudb.ingest.framework.task_processors.classification_processor import (
    process_classification,
)
from chaksudb.ingest.framework.split_assigner import auto_stratified_splits

logger = logging.getLogger(__name__)

# Dataset metadata
DATASET_NAME = "AIROGS"
DATASET_URL = "https://zenodo.org/records/5793241"
DATASET_LICENSE = "Research/Academic Use"  # Placeholder - update if known


async def ingest_airogs() -> OperationStatistics:
    """
    Main ingestion function for AIROGS dataset.
    
    Strategy:
    - Process train_labels.csv with challenge_id and class (RG/NRG)
    - Map RG (Referable Glaucoma) -> True, NRG (Non-Referable Glaucoma) -> False
    - Images are in documents/ folder with .JPG extension
    - All images are in train split (no explicit test split)
    
    Returns:
        OperationStatistics with success/error counts
    """
    data_root = get_data_root() / "29_AIROGS"
    dataset_id = generate_dataset_uuid(DATASET_NAME)
    
    logger.info("=" * 80)
    logger.info(f"Starting ingestion: {DATASET_NAME}")
    logger.info(f"Data root: {data_root}")
    logger.info("=" * 80)
    
    # Step 1: Register dataset
    logger.info(f"Registering dataset: {DATASET_NAME}")
    dataset = Dataset(
        dataset_id=dataset_id,
        dataset_name=DATASET_NAME,
        source_url=DATASET_URL,
        license=DATASET_LICENSE,
        modality_types=["fundus"],
        description=(
            "AIROGS (Artificial Intelligence for Referable Glaucoma Screening) dataset "
            "with binary glaucoma classification. RG = Referable Glaucoma (positive), "
            "NRG = Non-Referable Glaucoma (negative). Highly imbalanced dataset with 101K+ images."
        ),
    )
    await upsert_dataset(dataset)
    
    # Step 2: Count total rows for progress tracking
    logger.info("Counting CSV rows...")
    csv_path = data_root / "train_labels.csv"
    
    rows = await asyncio.to_thread(read_csv_auto, csv_path)
    total_count = len(rows)
    
    logger.info(f"Found {total_count} images in train_labels.csv")
    
    # Step 3: Setup progress tracker
    tracker = ProgressTracker(
        total=total_count,
        description=f"Ingesting {DATASET_NAME}"
    )
    
    # Collect items for bulk upsert
    all_images: List[Image] = []
    all_classifications: List = []
    all_image_ids: List[UUID] = []
    image_labels: dict = {}  # image_id → class label for stratified splitting
    
    # Image directories - try documents/ first, fallback to documents_org/
    image_dirs = [
        data_root / "documents",
        data_root / "documents_org",
    ]
    
    async def process_row(row, idx):
        """Process a single CSV row with binary glaucoma classification."""
        try:
            challenge_id = row["challenge_id"]
            class_label = row["class"].strip().upper()
            
            # Validate class label
            if class_label not in ["RG", "NRG"]:
                logger.warning(f"Row {idx}: Invalid class label: {class_label}")
                tracker.record_error(
                    error_type="invalid_class_label",
                    error_message=f"Invalid class label: {class_label} (expected RG or NRG)",
                    item_id=challenge_id,
                )
                tracker.update(count=1, success=False)
                return
            
            # Generate image ID
            image_id = generate_image_uuid(dataset_id, challenge_id)
            
            # Find image file - try both directories and both extensions
            image_path = None
            for image_dir in image_dirs:
                if not await asyncio.to_thread(image_dir.exists):
                    continue
                # Try both .JPG and .jpg extensions
                for ext in [".JPG", ".jpg"]:
                    candidate = image_dir / f"{challenge_id}{ext}"
                    if await asyncio.to_thread(candidate.exists):
                        image_path = candidate
                        break
                if image_path:
                    break
            
            if not image_path:
                tracker.record_error(
                    error_type="file_not_found",
                    error_message=f"Image not found: {challenge_id}",
                    item_id=challenge_id,
                )
                tracker.update(count=1, success=False)
                return
            
            # Create image with automatic metadata extraction
            image = Image(
                image_id=image_id,
                dataset_id=dataset_id,
                original_image_id=challenge_id,
                **get_image_metadata_dict(image_path),
                modality="fundus",
            )
            all_images.append(image)
            all_image_ids.append(image_id)
            image_labels[image_id] = class_label

            # Map class label to binary value
            # RG = Referable Glaucoma = True (positive case)
            # NRG = Non-Referable Glaucoma = False (negative case)
            is_referable_glaucoma = class_label == "RG"
            
            # Process classification using task processor
            # Provenance is automatically retrieved from context by process_classification()
            classifications = await process_classification(
                class_value=is_referable_glaucoma,
                task_type="binary",
                task_name="glaucoma",  # ML-concept-aligned: unifies with all glaucoma binary tasks
                class_name="glaucoma",
                image_id=image_id,
                class_labels={
                    True: "RG",
                    False: "NRG",
                },
                annotation_method="manual",
            )
            all_classifications.extend(classifications)
            
            tracker.update(count=1, success=True)
            
        except Exception as e:
            tracker.update(count=1, success=False)
            tracker.record_error(
                error_type="processing",
                error_message=str(e),
                item_id=row.get("challenge_id", "unknown"),
            )
            logger.error(f"Failed to process row {idx}: {e}", exc_info=True)
    
    # Step 4: Process CSV with automatic provenance tracking
    logger.info("Processing annotations...")
    
    # Process CSV with classification as primary annotation type
    csv_stats, raw_file_id, chain_id = await process_csv(
        csv_path,
        dataset_id,
        "classification",  # Primary annotation type
        process_row,
        progress_tracker=tracker,
    )
    
    logger.info(f"CSV registered: raw_file_id={raw_file_id}, chain_id={chain_id}")
    
    # Step 5: Bulk upsert - images first, then classifications (due to foreign key constraint)
    logger.info(f"Upserting {len(all_images)} images...")
    if all_images:
        try:
            await bulk_upsert_images(all_images, batch_size=1000)
            logger.info(f"Successfully upserted {len(all_images)} images")
        except Exception as e:
            logger.error(f"Failed to bulk upsert images: {e}")
            raise
    
    logger.info(f"Upserting {len(all_classifications)} classification annotations...")
    if all_classifications:
        try:
            await bulk_upsert_classification_annotations(all_classifications, batch_size=1000)
            logger.info(f"Successfully upserted {len(all_classifications)} classification annotations")
        except Exception as e:
            logger.error(f"Failed to bulk upsert classifications: {e}")
            raise
    
    # Step 6: Register splits — stratified 90/10 train+test, then 90/10 train+val
    logger.info("Registering dataset splits...")
    if all_image_ids:
        await auto_stratified_splits(
            dataset_id=dataset_id,
            split_assignments={"train": all_image_ids},
            labels=image_labels,
            split_type="explicit",
        )
    
    tracker.finish()
    stats = tracker.get_statistics()
    
    # Calculate class distribution for summary
    rg_count = sum(1 for c in all_classifications if c.class_value.get("referable_glaucoma") is True)
    nrg_count = len(all_classifications) - rg_count
    
    # Final summary
    logger.info("=" * 80)
    logger.info("Ingestion Summary:")
    logger.info(f"  Total items: {stats.total_items}")
    logger.info(f"  Successful: {stats.successful_items}")
    logger.info(f"  Failed: {stats.failed_items}")
    logger.info(f"  Skipped: {stats.skipped_items}")
    logger.info(f"  Images registered: {len(all_images)}")
    logger.info(f"  Classification annotations: {len(all_classifications)}")
    logger.info(f"    - RG (Referable Glaucoma): {rg_count}")
    logger.info(f"    - NRG (Non-Referable Glaucoma): {nrg_count}")
    if stats.errors:
        logger.warning(f"  Total errors: {len(stats.errors)}")
        for error_type, count in stats.error_counts.items():
            logger.warning(f"    {error_type}: {count}")
    logger.info("=" * 80)
    
    return stats


async def main():
    """Entry point for script execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    try:
        stats = await ingest_airogs()
        
        if stats.failed_items > 0:
            logger.error(f"Ingestion completed with {stats.failed_items} errors")
            return 1
        else:
            logger.info("Ingestion completed successfully!")
            return 0
            
    except Exception as e:
        logger.exception(f"Fatal error during ingestion: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
