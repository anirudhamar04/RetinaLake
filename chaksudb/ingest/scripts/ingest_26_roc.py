"""
Ingestion script for ROC dataset.

Dataset: ROC - Retinal Image Classification dataset
Structure: Single folder with images, split encoded in filename
Annotations: None (images only)

Key Features:
  - Images with train/test splits encoded in filename
  - Filename format: `image{0-49}_test.jpg` and `image{0-49}_training.jpg`
  - 50 image pairs (100 total images)
"""

import asyncio
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional
from uuid import UUID

from chaksudb.common.progress import ProgressTracker, OperationStatistics
from chaksudb.config.config import get_data_root
from chaksudb.db.models import Dataset, Image
from chaksudb.db.queries import (
    upsert_dataset,
    bulk_upsert_images,
)
from chaksudb.ingest.framework import (
    get_image_metadata_dict,
    find_images,
)
from chaksudb.ingest.framework.gen_uuid import (
    generate_dataset_uuid,
    generate_image_uuid,
)
from chaksudb.ingest.framework.split_assigner import (
    register_standard_splits,
    assign_images_by_split_dict,
)

logger = logging.getLogger(__name__)

# Dataset metadata
DATASET_NAME = "ROC"
DATASET_URL = "http://webeye.ophth.uiowa.edu/ROC/"
DATASET_LICENSE = "Research/Academic Use"  # Placeholder - update if known


def extract_split_from_filename(filename: str) -> Optional[str]:
    """
    Extract split name from ROC filename.
    
    Args:
        filename: Image filename (e.g., "image0_test.jpg", "image0_training.jpg")
    
    Returns:
        Split name ("train" or "test") or None if pattern doesn't match
    """
    # Pattern: image{number}_{test|training}.{ext}
    pattern = r"^image\d+_(test|training)\.(jpg|jpeg|png)$"
    match = re.match(pattern, filename, re.IGNORECASE)
    
    if not match:
        return None
    
    split_suffix = match.group(1).lower()
    
    # Map "training" to "train" for consistency
    if split_suffix == "training":
        return "train"
    elif split_suffix == "test":
        return "test"
    
    return None


async def ingest_roc() -> OperationStatistics:
    """
    Main ingestion function for ROC dataset.
    
    Strategy:
    - Find all images in data root
    - Extract split from filename (test vs training)
    - Create Image objects with metadata
    - Bulk upsert all images
    - Register splits and assign images based on filename
    
    Returns:
        OperationStatistics with success/error counts
    """
    data_root = get_data_root() / "26_ROC"
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
            "ROC dataset containing fundus images with train/test splits encoded in filenames. "
            "Images are named as image{number}_test.jpg or image{number}_training.jpg. "
            "No annotations provided - images only."
        ),
    )
    await upsert_dataset(dataset)
    
    # Step 2: Find all images
    logger.info("Finding images...")
    if not data_root.exists():
        raise FileNotFoundError(f"Data directory not found: {data_root}")
    
    all_image_paths = await asyncio.to_thread(find_images, data_root, recursive=False)
    total_images = len(all_image_paths)
    
    logger.info(f"Found {total_images} images in {data_root}")
    
    # Step 3: Setup progress tracker
    tracker = ProgressTracker(
        total=total_images,
        description=f"Ingesting {DATASET_NAME}"
    )
    
    # Step 4: Process all images and group by split
    logger.info("Processing images and extracting splits from filenames...")
    all_images: List[Image] = []
    image_ids_by_split: Dict[str, List[UUID]] = {
        "train": [],
        "test": [],
    }
    
    for image_path in all_image_paths:
        try:
            image_filename = image_path.name
            image_stem = image_path.stem
            
            # Extract split from filename
            split_name = extract_split_from_filename(image_filename)
            if not split_name:
                logger.warning(f"Could not extract split from filename: {image_filename}")
                tracker.record_error(
                    error_type="split_extraction",
                    error_message=f"Could not extract split from filename: {image_filename}",
                    item_id=image_stem,
                    item_path=str(image_path),
                )
                tracker.update(success=False)
                continue
            
            # Generate image ID
            image_id = generate_image_uuid(dataset_id, image_stem)
            
            # Create image with automatic metadata extraction
            image = Image(
                image_id=image_id,
                dataset_id=dataset_id,
                original_image_id=image_stem,
                **get_image_metadata_dict(image_path),
                modality="fundus",
            )
            
            all_images.append(image)
            image_ids_by_split[split_name].append(image_id)
            
            tracker.update(success=True)
            tracker.record_success("image")
            
        except Exception as e:
            logger.error(f"Failed to process {image_path}: {e}", exc_info=True)
            tracker.update(success=False)
            tracker.record_error(
                error_type="image_processing",
                error_message=str(e),
                item_id=image_path.stem,
                item_path=str(image_path),
            )
    
    # Log split distribution
    train_count = len(image_ids_by_split["train"])
    test_count = len(image_ids_by_split["test"])
    logger.info(f"Split distribution - Train: {train_count}, Test: {test_count}")
    
    # Step 5: Bulk upsert images
    logger.info(f"Upserting {len(all_images)} images...")
    if all_images:
        try:
            await bulk_upsert_images(all_images, batch_size=1000)
            logger.info(f"Successfully upserted {len(all_images)} images")
        except Exception as e:
            logger.error(f"Failed to bulk upsert images: {e}")
            raise
    
    # Step 6: Register splits and assign images
    logger.info("Registering dataset splits...")
    splits = await register_standard_splits(
        dataset_id=dataset_id,
        split_type="explicit",  # Split is explicitly encoded in filename
        train_count=train_count,
        test_count=test_count,
    )
    
    # Assign images to splits
    if image_ids_by_split["train"] or image_ids_by_split["test"]:
        logger.info("Assigning images to splits...")
        assignment_counts = await assign_images_by_split_dict(
            split_assignments=image_ids_by_split,
            split_ids=splits,
        )
        logger.info(f"Assigned {assignment_counts.get('train', 0)} images to train split")
        logger.info(f"Assigned {assignment_counts.get('test', 0)} images to test split")
    
    # Finish tracking
    tracker.finish()
    final_stats = tracker.get_statistics()
    
    # Final summary
    logger.info("=" * 80)
    logger.info("Ingestion Summary:")
    logger.info(f"  Total items: {final_stats.total_items}")
    logger.info(f"  Successful: {final_stats.successful_items}")
    logger.info(f"  Failed: {final_stats.failed_items}")
    logger.info(f"  Skipped: {final_stats.skipped_items}")
    logger.info(f"  Images registered: {len(all_images)}")
    logger.info(f"  Train images: {train_count}")
    logger.info(f"  Test images: {test_count}")
    if final_stats.errors:
        logger.warning(f"  Errors encountered: {len(final_stats.errors)}")
        for error in final_stats.errors[:10]:  # Show first 10 errors
            logger.warning(f"    - {error.error_type}: {error.error_message}")
    logger.info("=" * 80)
    
    return final_stats


async def main():
    """Entry point for script execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    try:
        stats = await ingest_roc()
        
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
