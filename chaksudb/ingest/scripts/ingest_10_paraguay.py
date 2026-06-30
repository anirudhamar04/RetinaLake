"""
Ingestion script for PARAGUAY dataset.

Dataset: Paraguay Diabetic Retinopathy Dataset
Structure: Hierarchical folders with DR grades (7 levels)
Annotations: Disease grading (DR)
Tasks: Grading (DR, 7-level scale)
"""

import asyncio
import logging
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
from chaksudb.ingest.framework import get_image_metadata_dict
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
from chaksudb.ingest.framework.task_processors.grading_processor import process_disease_grade
from chaksudb.ingest.framework.split_assigner import auto_stratified_splits

logger = logging.getLogger(__name__)

# Dataset metadata
DATASET_NAME = "PARAGUAY"
DATASET_URL = "https://zenodo.org/record/4647952"
DATASET_LICENSE = "CC-BY-4.0"

# Folder name to DR grade mapping
GRADE_MAP = {
    "1. No DR signs": 0,
    "2. Mild (or early) NPDR": 1,
    "3. Moderate NPDR": 2,
    "4. Severe NPDR": 3,
    "5. Very Severe NPDR": 4,
    "6. PDR": 4,
    "7. Advanced PDR": 4,
}

# Folder name to grade label mapping (for database storage)
GRADE_LABELS = {
    "1. No DR signs": "No DR signs",
    "2. Mild (or early) NPDR": "Mild NPDR",
    "3. Moderate NPDR": "Moderate NPDR",
    "4. Severe NPDR": "Severe NPDR",
    "5. Very Severe NPDR": "Very Severe NPDR",
    "6. PDR": "PDR",
    "7. Advanced PDR": "Advanced PDR",
}


async def ingest_paraguay() -> OperationStatistics:
    """
    Main ingestion function for PARAGUAY dataset.
    
    Returns:
        OperationStatistics with success/error counts
    """
    data_root = get_data_root() / "10_PARAGUAY"
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
    
    # Step 2: Register folder structure for provenance tracking
    # For PARAGUAY, the folder structure itself is the annotation source.
    # We register the root data directory as a raw annotation source to track provenance.
    # The grading scale will be auto-registered by task processor on first call.
    logger.info("PARAGUAY uses folder structure for annotations - registering folder structure for provenance")
    
    # Register the data root directory as a raw annotation source
    # This represents the folder structure that encodes the annotations
    folder_raw_file_id, folder_chain_id = await register_mask_directory(
        directory_path=data_root,
        dataset_id=dataset_id,
        unified_annotation_type="grading",
    )
    logger.info(f"Registered folder structure: raw_file_id={folder_raw_file_id}, chain_id={folder_chain_id}")
    
    # Set provenance context for the entire folder structure
    token_raw, token_chain = set_provenance_context(folder_raw_file_id, folder_chain_id)
    
    # Step 3: Count total images for progress tracking
    logger.info("Counting images in folder structure...")
    total_images = 0
    for folder_name in GRADE_MAP.keys():
        folder_path = data_root / folder_name
        if folder_path.exists() and folder_path.is_dir():
            image_count = len(list(folder_path.glob("*.jpg")))
            total_images += image_count
            logger.info(f"  {folder_name}: {image_count} images")
    
    logger.info(f"Total images: {total_images}")
    
    # Step 4: Setup progress tracker
    tracker = ProgressTracker(
        total=total_images,
        description=f"Ingesting {DATASET_NAME}"
    )
    
    # Collect items for bulk upsert
    all_images: List[Image] = []
    all_gradings: List[DiseaseGrading] = []
    image_ids_for_split: List[UUID] = []
    image_labels: dict = {}  # image_id → DR grade for stratified splitting
    
    # Step 5: Process each folder and its images
    logger.info("Processing images from folder structure...")
    try:
        for folder_name in GRADE_MAP.keys():
            folder_path = data_root / folder_name
            
            if not folder_path.exists() or not folder_path.is_dir():
                logger.warning(f"Folder not found: {folder_name}")
                continue
            
            grade_value = GRADE_MAP[folder_name]
            grade_label = GRADE_LABELS[folder_name]
            
            # Process all images in this folder
            for image_file in folder_path.glob("*.jpg"):
                try:
                    # Generate image ID based on filename
                    image_id = generate_image_uuid(dataset_id, image_file.stem)
                    
                    # Create image with automatic metadata extraction
                    image = Image(
                        image_id=image_id,
                        dataset_id=dataset_id,
                        original_image_id=image_file.stem,
                        **get_image_metadata_dict(image_file),
                        modality="fundus",
                    )
                    all_images.append(image)
                    image_ids_for_split.append(image_id)
                    image_labels[image_id] = grade_value
                    
                    # Get provenance from context (set at folder structure level)
                    raw_data_id, provenance_chain_id = get_current_provenance()
                    
                    # Process DR grading using task processor
                    # Use folder name as grade value to preserve detailed level info
                    # Provenance is automatically available from context
                    # Scale will be auto-registered on first call
                    grading = await process_disease_grade(
                        grade_value=grade_value,
                        disease_type="DR",
                        scale_name="PARAGUAY_DR_7_level",
                        image_id=image_id,
                        scale_description="PARAGUAY 7-level DR grading (0-4, with detailed NPDR stages)",
                        min_value=0,
                        max_value=4,
                        value_labels={
                            "0": "No DR signs",
                            "1": "Mild (or early) NPDR",
                            "2": "Moderate NPDR",
                            "3": "Severe NPDR",
                            "4": "Very Severe NPDR / PDR / Advanced PDR",
                        },
                        grade_label=grade_label,
                        raw_data_id=raw_data_id,  # From provenance context
                        provenance_chain_id=provenance_chain_id,  # From provenance context
                        annotation_method="manual",
                    )
                    all_gradings.append(grading)
                    
                    tracker.update(count=1, success=True)
                    
                except Exception as e:
                    tracker.update(count=1, success=False)
                    tracker.record_error(
                        error_type="processing",
                        error_message=str(e),
                        item_id=image_file.stem,
                        item_path=str(image_file),
                    )
                    logger.error(f"Failed to process {image_file}: {e}")
    finally:
        # Always reset provenance context
        reset_provenance_context(token_raw, token_chain)
    
    # Step 6: Bulk upsert - images first, then gradings in parallel
    logger.info(f"Upserting {len(all_images)} images...")
    await bulk_upsert_images(all_images, batch_size=1000)
    
    logger.info(f"Upserting {len(all_gradings)} gradings...")
    await bulk_upsert_disease_gradings(all_gradings, batch_size=1000)
    
    # Step 7: Register splits — stratified 90/10 train+test, then 90/10 train+val
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
        stats = await ingest_paraguay()
        
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
