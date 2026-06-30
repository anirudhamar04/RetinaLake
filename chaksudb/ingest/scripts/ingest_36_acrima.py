"""
Ingestion script for ACRIMA dataset.

Dataset: ACRIMA - Glaucoma classification dataset
Structure: Two folders (G/ for glaucoma, noG/ for normal) with filename-based classification
Annotations: Binary glaucoma classification from filename encoding

Key Features:
  - Glaucoma images: `Im###_g_ACRIMA.png` (e.g., `Im686_g_ACRIMA.png`)
  - Normal images: `Im###_ACRIMA.png` (e.g., `Im001_ACRIMA.png`)
  - Images are in PNG format (though some may be JPG)
  - 396 glaucoma images and 309 normal images (705 total)
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Optional, Tuple
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
    get_image_metadata_dict,
    find_images,
)
from chaksudb.ingest.framework.gen_uuid import (
    generate_dataset_uuid,
    generate_image_uuid,
)
from chaksudb.ingest.framework.provenance_context import (
    get_current_provenance,
    set_provenance_context,
    reset_provenance_context,
)
from chaksudb.ingest.framework.raw_file_helpers import register_mask_directory
from chaksudb.ingest.framework.split_assigner import auto_stratified_splits
from chaksudb.ingest.framework.task_processors.classification_processor import (
    process_classification,
)

logger = logging.getLogger(__name__)

# Dataset metadata
DATASET_NAME = "ACRIMA"
DATASET_URL = "https://figshare.com/articles/dataset/CNNs_for_Automatic_Glaucoma_Assessment_using_Fundus_Images_An_Extensive_Validation/7613135"
DATASET_LICENSE = "Research/Academic Use"  # Placeholder - update if known


def parse_classification_from_filename(filename: str) -> Optional[bool]:
    """
    Parse glaucoma classification from ACRIMA filename.
    
    Args:
        filename: Image filename (e.g., "Im686_g_ACRIMA.png", "Im001_ACRIMA.png")
    
    Returns:
        True if glaucoma (has "_g_" in filename), False if normal, None if pattern doesn't match
    """
    # Pattern: Im###_g_ACRIMA.{ext} (glaucoma) or Im###_ACRIMA.{ext} (normal)
    # Check for "_g_" pattern for glaucoma
    if "_g_" in filename:
        return True  # Glaucoma
    elif "_ACRIMA" in filename:
        return False  # Normal
    else:
        return None  # Pattern doesn't match


async def process_folder(
    folder_path: Path,
    dataset_id: UUID,
    tracker: ProgressTracker,
) -> Tuple[List[Image], List, List[UUID]]:
    """
    Process images from a folder (G/ or noG/).
    
    Args:
        folder_path: Path to folder (G/ or noG/)
        dataset_id: Dataset UUID
        tracker: Progress tracker
    
    Returns:
        Tuple of (images, classifications, image_ids)
    """
    all_images: List[Image] = []
    all_classifications: List = []
    image_ids: List[UUID] = []
    
    if not folder_path.exists():
        logger.warning(f"Folder not found: {folder_path}")
        return all_images, all_classifications, image_ids
    
    # Find all images in folder (PNG and JPG)
    image_paths = await asyncio.to_thread(
        find_images, folder_path, recursive=False
    )
    logger.info(f"Found {len(image_paths)} images in {folder_path.name} folder")
    
    for image_path in image_paths:
        try:
            image_filename = image_path.name
            image_stem = image_path.stem
            
            # Parse classification from filename
            is_glaucoma = parse_classification_from_filename(image_filename)
            if is_glaucoma is None:
                logger.warning(
                    f"Could not parse classification from filename: {image_filename}"
                )
                tracker.update(success=False)
                tracker.record_error(
                    error_type="filename_parsing",
                    error_message=f"Could not parse classification from filename: {image_filename}",
                    item_id=image_stem,
                    item_path=str(image_path),
                )
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
            image_ids.append(image_id)
            
            # Get provenance from context (set at folder structure level)
            raw_data_id, provenance_chain_id = get_current_provenance()
            
            # Process binary glaucoma classification
            # Provenance is automatically available from context
            classifications = await process_classification(
                class_value=is_glaucoma,
                task_type="binary",
                class_name="glaucoma",
                image_id=image_id,
                raw_data_id=raw_data_id,  # From provenance context
                provenance_chain_id=provenance_chain_id,  # From provenance context
                annotation_method="manual",
            )
            all_classifications.extend(classifications)
            
            tracker.update(success=True)
            tracker.record_success("image")
            
        except Exception as e:
            logger.error(
                f"Failed to process image {image_path}: {e}", exc_info=True
            )
            tracker.update(success=False)
            tracker.record_error(
                error_type="image_processing",
                error_message=str(e),
                item_id=image_path.stem,
                item_path=str(image_path),
            )
    
    return all_images, all_classifications, image_ids


async def ingest_acrima() -> OperationStatistics:
    """
    Main ingestion function for ACRIMA dataset.
    
    Strategy:
    - Process G/ folder: Glaucoma images (Im###_g_ACRIMA.*)
    - Process noG/ folder: Normal images (Im###_ACRIMA.*)
    - Extract classification from filename pattern
    - Bulk upsert all data
    - Assign all images to a single "train" split (no explicit splits provided)
    
    Returns:
        OperationStatistics with success/error counts
    """
    data_root = get_data_root() / "36_ACRIMA"
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
            "ACRIMA dataset with binary glaucoma classification. "
            "Glaucoma images: Im###_g_ACRIMA.* (396 images), "
            "Normal images: Im###_ACRIMA.* (309 images). "
            "Classification extracted from filename encoding."
        ),
    )
    await upsert_dataset(dataset)
    
    # Step 2: Register folder structure for provenance tracking
    # For ACRIMA, the folder structure (G/ and noG/) encodes the classification.
    # We register the root data directory as a raw annotation source to track provenance.
    logger.info("ACRIMA uses folder structure for annotations - registering folder structure for provenance")
    
    # Register the data root directory as a raw annotation source
    # This represents the folder structure that encodes the annotations
    folder_raw_file_id, folder_chain_id = await register_mask_directory(
        directory_path=data_root,
        dataset_id=dataset_id,
        unified_annotation_type="classification",
    )
    logger.info(f"Registered folder structure: raw_file_id={folder_raw_file_id}, chain_id={folder_chain_id}")
    
    # Set provenance context for the entire folder structure
    token_raw, token_chain = set_provenance_context(folder_raw_file_id, folder_chain_id)
    
    # Step 3: Count total images for progress tracking
    logger.info("Counting images...")
    g_folder = data_root / "G"
    nog_folder = data_root / "noG"
    
    g_count = 0
    nog_count = 0
    
    if g_folder.exists():
        g_count = len(list(g_folder.glob("*.png"))) + len(list(g_folder.glob("*.jpg"))) + len(list(g_folder.glob("*.PNG"))) + len(list(g_folder.glob("*.JPG")))
    
    if nog_folder.exists():
        nog_count = len(list(nog_folder.glob("*.png"))) + len(list(nog_folder.glob("*.jpg"))) + len(list(nog_folder.glob("*.PNG"))) + len(list(nog_folder.glob("*.JPG")))
    
    total_images = g_count + nog_count
    logger.info(f"  G/ folder: {g_count} images")
    logger.info(f"  noG/ folder: {nog_count} images")
    logger.info(f"  Total images: {total_images}")
    
    # Step 4: Setup progress tracker
    tracker = ProgressTracker(
        total=total_images,
        description=f"Ingesting {DATASET_NAME}"
    )
    
    # Step 5: Process G/ folder (glaucoma images)
    logger.info("=" * 80)
    logger.info("Processing G/ folder (glaucoma images)...")
    logger.info("=" * 80)
    try:
        g_images, g_classifications, g_image_ids = await process_folder(
            g_folder, dataset_id, tracker
        )
        logger.info(f"Processed {len(g_images)} images from G/ folder")
        
        # Step 6: Process noG/ folder (normal images)
        logger.info("=" * 80)
        logger.info("Processing noG/ folder (normal images)...")
        logger.info("=" * 80)
        nog_images, nog_classifications, nog_image_ids = await process_folder(
            nog_folder, dataset_id, tracker
        )
        logger.info(f"Processed {len(nog_images)} images from noG/ folder")
    finally:
        # Always reset provenance context
        reset_provenance_context(token_raw, token_chain)
    
    # Combine all results
    all_images = g_images + nog_images
    all_classifications = g_classifications + nog_classifications
    all_image_ids = g_image_ids + nog_image_ids
    image_labels = (
        {img_id: "glaucoma" for img_id in g_image_ids}
        | {img_id: "non_glaucoma" for img_id in nog_image_ids}
    )
    
    # Step 7: Bulk upsert images
    logger.info(f"Upserting {len(all_images)} images...")
    if all_images:
        try:
            await bulk_upsert_images(all_images, batch_size=1000)
            logger.info(f"Successfully upserted {len(all_images)} images")
        except Exception as e:
            logger.error(f"Failed to bulk upsert images: {e}")
            raise
    
    # Step 8: Bulk upsert classifications
    logger.info(f"Upserting {len(all_classifications)} classification annotations...")
    if all_classifications:
        try:
            await bulk_upsert_classification_annotations(
                all_classifications, batch_size=1000
            )
            logger.info(
                f"Successfully upserted {len(all_classifications)} classification annotations"
            )
        except Exception as e:
            logger.error(f"Failed to bulk upsert classifications: {e}")
            raise
    
    # Step 9: Register splits — stratified 90/10 train+test, then 90/10 train+val
    logger.info("Registering dataset splits...")
    if all_image_ids:
        await auto_stratified_splits(
            dataset_id=dataset_id,
            split_assignments={"train": all_image_ids},
            labels=image_labels,
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
    logger.info(f"    - G/ folder (glaucoma): {len(g_images)}")
    logger.info(f"    - noG/ folder (normal): {len(nog_images)}")
    logger.info(f"  Classification annotations: {len(all_classifications)}")
    logger.info(f"    - Glaucoma images: {len(g_images)}")
    logger.info(f"    - Normal images: {len(nog_images)}")
    if final_stats.errors:
        logger.warning(f"  Errors encountered: {len(final_stats.errors)}")
        for error in final_stats.errors[:10]:  # Show first 10 errors
            # Handle both dict and object error formats
            if isinstance(error, dict):
                error_type = error.get("error_type", "unknown")
                error_message = error.get("error_message", str(error))
            else:
                error_type = getattr(error, "error_type", "unknown")
                error_message = getattr(error, "error_message", str(error))
            logger.warning(f"    - {error_type}: {error_message}")
    logger.info("=" * 80)
    
    return final_stats


async def main():
    """Entry point for script execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    try:
        stats = await ingest_acrima()
        
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
