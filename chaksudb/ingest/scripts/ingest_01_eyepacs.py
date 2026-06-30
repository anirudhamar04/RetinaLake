"""
Ingestion script for EYEPACS dataset.

Dataset: Diabetic Retinopathy Detection
Structure: train.csv, test.csv with image filenames and DR grading (0-4)
Annotations: Disease grading (DR)
Tasks: Grading (DR, 5-level scale)
"""

import asyncio
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from uuid import UUID

from chaksudb.common.progress import ProgressTracker, OperationStatistics
from chaksudb.config.config import get_data_root
from chaksudb.db.models import Dataset, Image, DiseaseGrading
from chaksudb.db.queries import (
    upsert_dataset,
    bulk_upsert_images,
    bulk_upsert_disease_gradings,
)
from chaksudb.ingest.framework.provenance_context import (
    get_current_provenance,
    set_provenance_context,
    reset_provenance_context,
)
from chaksudb.ingest.framework import (
    find_images,
    find_matching_file,
    process_csv,
    read_csv_auto,
    get_image_metadata_dict,
)
from chaksudb.ingest.framework.gen_uuid import (
    generate_dataset_uuid,
    generate_image_uuid,
)
from chaksudb.ingest.framework.task_processors.grading_processor import process_disease_grade
from chaksudb.ingest.framework.split_assigner import (
    register_standard_splits,
    bulk_assign_images_to_split,
)

logger = logging.getLogger(__name__)

# Dataset metadata
DATASET_NAME = "EYEPACS"
DATASET_URL = "https://www.kaggle.com/c/diabetic-retinopathy-detection"
DATASET_LICENSE = "CC-BY-4.0"


async def ingest_eyepacs() -> OperationStatistics:
    """
    Main ingestion function for EYEPACS dataset.
    
    Returns:
        OperationStatistics with success/error counts
    """
    data_root = get_data_root() / "01_EYEPACS"
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
    )
    await upsert_dataset(dataset)
    
    # Step 2: Count total rows for progress tracking
    logger.info("Counting CSV rows...")
    train_csv_path = data_root / "train.csv"
    test_csv_path = data_root / "test.csv"
    
    train_rows = await asyncio.to_thread(read_csv_auto, train_csv_path)
    test_rows = await asyncio.to_thread(read_csv_auto, test_csv_path)
    total_count = len(train_rows) + len(test_rows)
    
    logger.info(f"Found {len(train_rows)} training images and {len(test_rows)} test images")
    
    # Step 3: Setup progress tracker
    tracker = ProgressTracker(
        total=total_count,
        description=f"Ingesting {DATASET_NAME}"
    )
    
    # Collect items for bulk upsert
    all_images: List[Image] = []
    all_gradings: List[DiseaseGrading] = []
    image_to_split: Dict[UUID, str] = {}  # For split assignment
    
    async def process_row(row, idx, split_name):
        """Process a single CSV row."""
        try:
            image_name = row["image"]
            image_id = generate_image_uuid(dataset_id, image_name)
            
            # Find image file - try different extensions
            image_dir = data_root / split_name
            image_path = None
            
            for ext in [".jpeg", ".jpg", ".png"]:
                candidate = image_dir / f"{image_name}{ext}"
                if await asyncio.to_thread(candidate.exists):
                    image_path = candidate
                    break
            
            if not image_path:
                tracker.record_error(
                    error_type="file_not_found",
                    error_message=f"Image not found: {image_name}",
                    item_id=image_name,
                )
                tracker.update(success=False)
                return
            
            # Extract laterality from filename (format: <id>_<left/right>)
            laterality = None
            laterality_match = re.search(r"_(left|right)$", image_name, re.IGNORECASE)
            if laterality_match:
                laterality = laterality_match.group(1).lower()
            
            # Create image with automatic metadata extraction
            image = Image(
                image_id=image_id,
                dataset_id=dataset_id,
                original_image_id=image_name,
                **get_image_metadata_dict(image_path),
                modality="fundus",
                eye_laterality=laterality,
            )
            all_images.append(image)
            image_to_split[image_id] = split_name
            
            # Process grading using task processor
            # EYEPACS uses standard ICDR scale (0-4): No DR, Mild, Moderate, Severe, PDR
            # Task processor automatically handles: provenance, UUID generation, timestamps
            grading = await process_disease_grade(
                grade_value=int(row["level"]),
                disease_type="DR",
                scale_name="ICDR_0_4",
                image_id=image_id,
                annotation_method="manual",
            )
            all_gradings.append(grading)
            
            tracker.update(count=1, success=True)
            
        except Exception as e:
            tracker.update(count=1, success=False)
            tracker.record_error(
                error_type="processing",
                error_message=str(e),
                item_id=row.get("image"),
            )
            logger.error(f"Failed to process row {idx}: {e}")
    
    # Step 4: Process train and test CSVs in parallel
    logger.info("Processing annotations...")
    
    # Create separate handlers for train and test
    async def train_handler(row, idx):
        await process_row(row, idx, "train")
    
    async def test_handler(row, idx):
        await process_row(row, idx, "test")
    
    # Process both CSVs in parallel with provenance tracking
    # EYEPACS primary annotation type is "grading" (DR grading)
    train_results, test_results = await asyncio.gather(
        process_csv(
            train_csv_path,
            dataset_id,
            "grading",  # Primary annotation type
            train_handler
        ),
        process_csv(
            test_csv_path,
            dataset_id,
            "grading",  # Primary annotation type
            test_handler
        ),
    )
    
    # Log provenance information
    train_stats, train_raw_id, train_chain_id = train_results
    test_stats, test_raw_id, test_chain_id = test_results
    logger.info(f"Train CSV registered: raw_file_id={train_raw_id}, chain_id={train_chain_id}")
    logger.info(f"Test CSV registered: raw_file_id={test_raw_id}, chain_id={test_chain_id}")
    
    # Step 5: Bulk upsert - images first, then gradings (due to foreign key constraint)
    logger.info(f"Upserting {len(all_images)} images...")
    await bulk_upsert_images(all_images, batch_size=1000)
    
    logger.info(f"Upserting {len(all_gradings)} gradings...")
    await bulk_upsert_disease_gradings(all_gradings, batch_size=1000)
    
    # Step 6: Register splits and assign images
    logger.info("Registering dataset splits...")
    train_image_ids = [img_id for img_id, split in image_to_split.items() if split == "train"]
    test_image_ids = [img_id for img_id, split in image_to_split.items() if split == "test"]
    
    splits = await register_standard_splits(
        dataset_id=dataset_id,
        split_type="explicit",
        train_count=len(train_image_ids),
        test_count=len(test_image_ids),
    )
    
    # Assign images to splits
    await asyncio.gather(
        bulk_assign_images_to_split(train_image_ids, splits["train"]),
        bulk_assign_images_to_split(test_image_ids, splits["test"]),
    )
    
    tracker.finish()
    stats = tracker.get_statistics()
    
    # Final summary
    logger.info("=" * 80)
    logger.info("Ingestion Summary:")
    logger.info(f"  Total items: {stats.total_items}")
    logger.info(f"  Successful: {stats.successful_items}")
    logger.info(f"  Failed: {stats.failed_items}")
    logger.info(f"  Skipped: {stats.skipped_items}")
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
        stats = await ingest_eyepacs()
        
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
