"""
Ingestion script for AGAR300 dataset.

Dataset: AGAR300 - Image registration only (no annotations)
Structure: Single folder with fundus images
Annotations: None (images only)

Key Features:
  - 300 fundus images for image registration tasks
  - No annotations provided
  - Images stored in img/ folder
"""

import asyncio
import logging
from pathlib import Path
from typing import List
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
from chaksudb.ingest.framework.split_assigner import auto_stratified_splits

logger = logging.getLogger(__name__)

# Dataset metadata
DATASET_NAME = "AGAR300"
DATASET_URL = "https://ieee-dataport.org/open-access/diabetic-retinopathy-fundus-image-datasetagar300"
DATASET_LICENSE = "Research/Academic Use"  # Placeholder - update if known


async def ingest_agar300() -> OperationStatistics:
    """
    Main ingestion function for AGAR300 dataset.
    
    Strategy:
    - Find all images in img/ folder
    - Create Image objects with metadata
    - Bulk upsert all images
    - Assign all images to a single "train" split
    
    Returns:
        OperationStatistics with success/error counts
    """
    data_root = get_data_root() / "14_AGAR300"
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
            "AGAR300 dataset containing 300 fundus images for image registration tasks. "
            "No annotations provided - images only."
        ),
    )
    await upsert_dataset(dataset)
    
    # Step 2: Find all images
    logger.info("Finding images...")
    img_dir = data_root / "img"
    if not img_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    
    all_image_paths = await asyncio.to_thread(find_images, img_dir)
    total_images = len(all_image_paths)
    
    logger.info(f"Found {total_images} images in {img_dir}")
    
    # Step 3: Setup progress tracker
    tracker = ProgressTracker(
        total=total_images,
        description=f"Ingesting {DATASET_NAME}"
    )
    
    # Step 4: Process all images
    logger.info("Processing images...")
    all_images: List[Image] = []
    image_ids_for_split: List[UUID] = []
    
    for image_path in all_image_paths:
        try:
            image_stem = image_path.stem
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
            image_ids_for_split.append(image_id)
            
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
    # AGAR300 has no annotations — random 90/10 train+test, then 90/10 train+val
    logger.info("Registering dataset splits...")
    if image_ids_for_split:
        await auto_stratified_splits(
            dataset_id=dataset_id,
            split_assignments={"train": image_ids_for_split},
            split_type="explicit",
        )
    
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
        stats = await ingest_agar300()
        
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
