"""
Ingestion script for LAG dataset.

Dataset: Large-scale Attention-based Glaucoma (LAG) Dataset
Structure: Two folders (suspicious_glaucoma, non_glaucoma) with images and classification labels
Annotations: Binary classification (glaucoma) + attention soft maps
Tasks: Classification (glaucoma/non-glaucoma), Segmentation (attention maps as soft maps)
"""

import asyncio
import logging
from pathlib import Path
from typing import List
from uuid import UUID

from chaksudb.common.progress import ProgressTracker, OperationStatistics
from chaksudb.config.config import get_data_root
from chaksudb.db.models import Dataset, Image, ClassificationAnnotation, SegmentationAnnotation
from chaksudb.db.queries import (
    upsert_dataset,
    bulk_upsert_images,
    bulk_upsert_classification_annotations,
    upsert_segmentation_annotation,
)
from chaksudb.ingest.framework import get_image_metadata_dict
from chaksudb.ingest.framework.gen_uuid import (
    generate_dataset_uuid,
    generate_image_uuid,
)
from chaksudb.ingest.framework.task_processors.classification_processor import process_classification
from chaksudb.ingest.framework.task_processors.segmentation_processor import process_segmentation_from_soft_map
from chaksudb.ingest.framework.split_assigner import auto_stratified_splits

logger = logging.getLogger(__name__)

# Dataset metadata
DATASET_NAME = "LAG"
DATASET_URL = "https://github.com/smilell/AG-CNN"
DATASET_LICENSE = "Unknown"  # License not specified in dataset


async def ingest_lag() -> OperationStatistics:
    """
    Main ingestion function for LAG dataset.
    
    The LAG dataset contains fundus images organized into two folders:
    - suspicious_glaucoma: Images with glaucoma indicators
    - non_glaucoma: Images without glaucoma
    
    Each folder has:
    - image/: Original fundus images (.jpg)
    - label/: Text files with 0/1 classification labels
    - attention_map/: Model-generated attention heatmaps (112x112 grayscale JPEG, saved as soft maps)
    
    Returns:
        OperationStatistics with success/error counts
    """
    data_root = get_data_root() / "07_LAG"
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
    suspicious_dir = data_root / "suspicious_glaucoma" / "image"
    non_glaucoma_dir = data_root / "non_glaucoma" / "image"
    
    suspicious_images = list(suspicious_dir.glob("*.jpg")) if suspicious_dir.exists() else []
    non_glaucoma_images = list(non_glaucoma_dir.glob("*.jpg")) if non_glaucoma_dir.exists() else []
    
    total_images = len(suspicious_images) + len(non_glaucoma_images)
    logger.info(f"Found {len(suspicious_images)} suspicious glaucoma images")
    logger.info(f"Found {len(non_glaucoma_images)} non-glaucoma images")
    logger.info(f"Total images: {total_images}")
    
    # Step 3: Setup progress tracker
    tracker = ProgressTracker(
        total=total_images,
        description=f"Ingesting {DATASET_NAME}"
    )
    
    # Collect items for bulk upsert
    all_images: List[Image] = []
    all_classifications: List[ClassificationAnnotation] = []
    all_segmentations: List[SegmentationAnnotation] = []
    image_ids_for_split: List[UUID] = []
    image_labels: dict = {}  # image_id → glaucoma bool for stratified splitting
    
    # Step 4: Process images from both folders
    logger.info("Processing images...")
    
    async def process_image_file(image_path: Path, is_glaucoma: bool, folder_type: str):
        """
        Process a single image file and its classification label.
        
        Args:
            image_path: Path to the image file
            is_glaucoma: True if from suspicious_glaucoma folder, False if from non_glaucoma
            folder_type: 'suspicious_glaucoma' or 'non_glaucoma'
        """
        try:
            image_stem = image_path.stem
            image_id = generate_image_uuid(dataset_id, f"{folder_type}_{image_stem}")
            
            # Verify label file exists and matches expected value
            label_dir = data_root / folder_type / "label"
            label_file = label_dir / f"{image_stem}.txt"
            
            expected_label = 1 if is_glaucoma else 0
            if label_file.exists():
                label_content = await asyncio.to_thread(label_file.read_text)
                label_value = int(label_content.strip())
                if label_value != expected_label:
                    logger.warning(
                        f"Label mismatch for {image_stem}: "
                        f"folder={folder_type} (expected {expected_label}), "
                        f"label file={label_value}"
                    )
            else:
                logger.warning(f"Label file not found for {image_stem}")
            
            # Create image with automatic metadata extraction
            image = Image(
                image_id=image_id,
                dataset_id=dataset_id,
                original_image_id=f"{folder_type}_{image_stem}",
                **get_image_metadata_dict(image_path),
                modality="fundus",
            )
            all_images.append(image)
            image_ids_for_split.append(image_id)
            image_labels[image_id] = is_glaucoma
            
            # Process glaucoma classification
            # Note: No explicit annotation file to register for folder-based structure
            # Setting raw_data_id and provenance_chain_id to None
            classification = await process_classification(
                class_value=is_glaucoma,
                task_type="binary",
                class_name="glaucoma",
                image_id=image_id,
                class_labels={0: "non_glaucoma", 1: "suspicious_glaucoma"},
                raw_data_id=None,  # Folder structure is implicit annotation
                provenance_chain_id=None,  # No explicit provenance chain
                annotation_method="manual",
            )
            all_classifications.extend(classification)
            
            # Process attention map as soft map segmentation
            attention_dir = data_root / folder_type / "attention_map"
            attention_file = attention_dir / f"{image_stem}.jpg"
            
            if attention_file.exists():
                segmentation = await process_segmentation_from_soft_map(
                    soft_map_path=attention_file,
                    annotation_type="attention_map",
                    image_id=image_id,
                    annotation_description="Model-generated attention heatmap from AG-CNN",
                    raw_data_id=None,  # Model-generated, not from raw annotation
                    provenance_chain_id=None,
                    annotation_method="automatic",  # Model-generated
                    confidence_score=None,
                )
                all_segmentations.append(segmentation)
            else:
                logger.warning(f"Attention map not found for {image_stem}"            )
            
            tracker.update(count=1, success=True)
            
        except Exception as e:
            tracker.update(count=1, success=False)
            tracker.record_error(
                error_type="processing",
                error_message=str(e),
                item_id=image_path.stem,
                item_path=str(image_path),
            )
            logger.error(f"Failed to process {image_path}: {e}")
    
    # Process suspicious glaucoma images
    logger.info("Processing suspicious glaucoma images...")
    for image_path in suspicious_images:
        await process_image_file(image_path, is_glaucoma=True, folder_type="suspicious_glaucoma")
    
    # Process non-glaucoma images
    logger.info("Processing non-glaucoma images...")
    for image_path in non_glaucoma_images:
        await process_image_file(image_path, is_glaucoma=False, folder_type="non_glaucoma")
    
    # Step 5: Bulk upsert - images first, then classifications and segmentations
    logger.info(f"Upserting {len(all_images)} images...")
    await bulk_upsert_images(all_images, batch_size=1000)
    
    logger.info(f"Upserting {len(all_classifications)} classifications...")
    await bulk_upsert_classification_annotations(all_classifications, batch_size=1000)
    
    # Note: Segmentations don't have bulk upsert yet, so upsert individually
    logger.info(f"Upserting {len(all_segmentations)} attention map segmentations...")
    for segmentation in all_segmentations:
        await upsert_segmentation_annotation(segmentation)
    
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
    logger.info(f"  Glaucoma images: {len(suspicious_images)}")
    logger.info(f"  Non-glaucoma images: {len(non_glaucoma_images)}")
    logger.info(f"  Classifications: {len(all_classifications)}")
    logger.info(f"  Attention map segmentations: {len(all_segmentations)}")
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
        stats = await ingest_lag()
        
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
