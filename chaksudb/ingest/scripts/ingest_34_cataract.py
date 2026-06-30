"""
Ingestion script for Cataract dataset.

Dataset: Retina dataset containing four categories: normal, cataract, glaucoma, retina disease
Structure: Folder-based multi-class classification
Annotations: Multi-class classification (normal, cataract, glaucoma, retina_disease)
Tasks: Multi-class classification (normal, cataract, glaucoma, retina_disease)
"""

import asyncio
import logging
from pathlib import Path
from typing import List
from uuid import UUID

from chaksudb.common.progress import ProgressTracker, OperationStatistics
from chaksudb.config.config import get_data_root
from chaksudb.db.models import Dataset, Image, ClassificationAnnotation
from chaksudb.db.queries import (
    upsert_dataset,
    bulk_upsert_images,
    bulk_upsert_classification_annotations,
)
from chaksudb.ingest.framework import (
    process_folder_tree,
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
DATASET_NAME = "Cataract"
DATASET_URL = "https://github.com/cvblab/retina_dataset"
DATASET_LICENSE = "Unknown"  # License not specified in dataset

# Folder name to class name mapping
FOLDER_TO_CLASS = {
    "1_normal": "normal",
    "2_cataract": "cataract",
    "2_glaucoma": "glaucoma",
    "3_retina_disease": "retina_disease",
}

# Class labels for multi-class classification
CLASS_LABELS = {
    0: "normal",
    1: "cataract",
    2: "glaucoma",
    3: "retina_disease",
}

# Class name to index mapping
CLASS_NAME_TO_INDEX = {
    "normal": 0,
    "cataract": 1,
    "glaucoma": 2,
    "retina_disease": 3,
}


async def ingest_cataract() -> OperationStatistics:
    """
    Main ingestion function for Cataract dataset.
    
    The Cataract dataset contains fundus images organized into four folders:
    - 1_normal/: Normal images
    - 2_cataract/: Cataract images
    - 2_glaucoma/: Glaucoma images
    - 3_retina_disease/: Retina disease images
    
    Returns:
        OperationStatistics with success/error counts
    """
    data_root = get_data_root() / "34_Cataract"
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
    
    # Step 2: Count total images for progress tracking
    logger.info("Counting images...")
    total_images = 0
    for folder_name in FOLDER_TO_CLASS.keys():
        folder_path = data_root / folder_name
        if folder_path.exists() and folder_path.is_dir():
            image_count = len(list(folder_path.glob("*.png")))
            total_images += image_count
            logger.info(f"  {folder_name}: {image_count} images")
    
    logger.info(f"Total images: {total_images}")
    
    # Step 3: Setup progress tracker
    tracker = ProgressTracker(
        total=total_images,
        description=f"Ingesting {DATASET_NAME}"
    )
    
    # Collect items for bulk upsert
    all_images: List[Image] = []
    all_classifications: List[ClassificationAnnotation] = []
    image_ids_for_split: List[UUID] = []
    image_labels: dict = {}  # image_id → class name for stratified splitting
    
    # Step 4: Process images from folder structure
    async def handle_image(file_path: Path, rel_path: Path, depth: int):
        """
        Process each image with multi-class classification.
        
        Args:
            file_path: Absolute path to the image file
            rel_path: Path relative to data_root
            depth: Directory depth (0 = root)
        """
        # Get folder name from parent directory
        folder_name = rel_path.parent.name
        
        # Skip if folder not in mapping
        if folder_name not in FOLDER_TO_CLASS:
            logger.warning(f"Unknown folder: {folder_name} (file: {file_path.name})")
            tracker.update(success=False)
            tracker.record_error(
                error_type="unknown_folder",
                error_message=f"Folder not in FOLDER_TO_CLASS mapping",
                item_id=file_path.stem,
                item_path=str(file_path),
            )
            return
        
        try:
            # Get class name from folder
            class_name = FOLDER_TO_CLASS[folder_name]
            class_index = CLASS_NAME_TO_INDEX[class_name]
            
            # Generate image ID
            image_id = generate_image_uuid(dataset_id, file_path.stem)
            
            # Create image with automatic metadata extraction
            image = Image(
                image_id=image_id,
                dataset_id=dataset_id,
                original_image_id=file_path.stem,
                **get_image_metadata_dict(file_path),
                modality="fundus",
            )
            all_images.append(image)
            image_ids_for_split.append(image_id)
            image_labels[image_id] = class_name

            # Process multi-class classification
            # Provenance is automatically available from process_folder_tree()
            classifications = await process_classification(
                class_value=class_index,  # Use class index (0-3)
                task_type="multi_class",
                task_name="disease_category",
                class_name="disease_category",
                image_id=image_id,
                class_labels=CLASS_LABELS,
                annotation_method="manual",  # Folder structure manually curated
            )
            all_classifications.extend(classifications)
            
            tracker.update(success=True)
            tracker.record_success("image")
            
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}", exc_info=True)
            tracker.update(success=False)
            tracker.record_error(
                error_type="processing",
                error_message=str(e),
                item_id=file_path.stem,
                item_path=str(file_path),
            )
    
    # Process folder tree with automatic per-file provenance
    logger.info("Processing images from folder structure...")
    stats = await process_folder_tree(
        root_dir=data_root,
        dataset_id=dataset_id,
        unified_annotation_type="classification",  # Primary annotation type
        process_file_fn=handle_image,
        file_extensions={".png", ".PNG"},
        recursive=True,
        include_dirs=False,
        progress_tracker=tracker,
        skip_errors=True,
    )
    
    # Step 5: Bulk upsert - images first, then classifications
    # Images must be inserted before classifications due to foreign key constraint
    logger.info(f"Upserting {len(all_images)} images...")
    await bulk_upsert_images(all_images, batch_size=1000)
    
    logger.info(f"Upserting {len(all_classifications)} classifications...")
    await bulk_upsert_classification_annotations(all_classifications, batch_size=1000)
    
    # Step 6: Register splits — stratified 90/10 train+test, then 90/10 train+val
    logger.info("Registering dataset splits...")
    await auto_stratified_splits(
        dataset_id=dataset_id,
        split_assignments={"train": image_ids_for_split},
        labels=image_labels,
        split_type="explicit",
    )
    
    tracker.finish()
    final_stats = tracker.get_statistics()
    
    # Final summary
    logger.info("=" * 80)
    logger.info("Ingestion Summary:")
    logger.info(f"  Total items: {final_stats.total_items}")
    logger.info(f"  Successful: {final_stats.successful_items}")
    logger.info(f"  Failed: {final_stats.failed_items}")
    logger.info(f"  Skipped: {final_stats.skipped_items}")
    logger.info(f"  Images: {len(all_images)}")
    logger.info(f"  Classifications: {len(all_classifications)}")
    if final_stats.errors:
        logger.warning(f"  Total errors: {len(final_stats.errors)}")
        for error_type, count in final_stats.error_counts.items():
            logger.warning(f"    {error_type}: {count}")
    logger.info("=" * 80)
    
    return final_stats


async def main():
    """Entry point for script execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    try:
        stats = await ingest_cataract()
        
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
