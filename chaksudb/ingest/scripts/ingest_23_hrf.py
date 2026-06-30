"""
Ingestion script for HRF dataset.

Dataset: HRF - High-Resolution Fundus dataset
Structure: Two separate tasks
  1. Noise folder: Meta classification task (good vs bad quality images)
  2. Documents folder: Binary disease classifications (healthy, glaucoma, diabetic_retinopathy)

Key Features:
  - Noise folder: `*_good.JPG` and `*_bad.JPG` (pairs of same images, good and blurry versions)
  - Documents folder: `##_h.jpg` (healthy), `##_g.jpg` (glaucoma), `##_dr.JPG` (diabetic retinopathy)
  - Both JPG and jpg extensions exist
"""

import asyncio
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from uuid import UUID

from chaksudb.common.progress import ProgressTracker, OperationStatistics
from chaksudb.config.config import get_data_root
from chaksudb.db.models import Dataset, Image
from chaksudb.db.queries import (
    upsert_dataset,
    bulk_upsert_images,
    bulk_upsert_classification_annotations,
    upsert_quality_annotation,
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
from chaksudb.ingest.framework.task_processors.classification_processor import (
    process_classification,
)
from chaksudb.ingest.framework.task_processors.quality_processor import (
    process_quality_annotation,
)

logger = logging.getLogger(__name__)

# Dataset metadata
DATASET_NAME = "HRF"
DATASET_URL = "https://www5.cs.fau.de/research/data/fundus-images/"
DATASET_LICENSE = "Research/Academic Use"  # Placeholder - update if known


def parse_quality_from_filename(filename: str) -> Optional[Tuple[str, str]]:
    """
    Parse quality annotation from Noise folder filename.
    
    Args:
        filename: Quality filename (e.g., "5_good.JPG", "5_bad.JPG")
    
    Returns:
        Tuple of (number_prefix, quality_label) or None if pattern doesn't match
        quality_label: "good" or "bad"
    """
    # Pattern: {number}_{good|bad}.{JPG|jpg}
    pattern = r"^(\d+)_(good|bad)\.(JPG|jpg)$"
    match = re.match(pattern, filename, re.IGNORECASE)
    
    if not match:
        return None
    
    number_prefix = match.group(1)
    quality_label = match.group(2).lower()
    
    return (number_prefix, quality_label)


def parse_classification_from_filename(filename: str) -> Optional[Tuple[str, str]]:
    """
    Parse classification from HRF documents filename.
    
    Args:
        filename: Image filename (e.g., "05_h.jpg", "05_g.jpg", "05_dr.JPG")
    
    Returns:
        Tuple of (number_prefix, classification_label) or None if pattern doesn't match
        classification_label: "healthy", "glaucoma", or "diabetic_retinopathy"
    """
    # Pattern: {number}_{h|g|dr}.{ext} - case insensitive
    pattern = r"^(\d+)_(h|g|dr)\.(jpg|JPG)$"
    match = re.match(pattern, filename, re.IGNORECASE)
    
    if not match:
        return None
    
    number_prefix = match.group(1)
    label_code = match.group(2).lower()
    
    # Map label codes to classification labels
    label_map = {
        "h": "healthy",
        "g": "glaucoma",
        "dr": "diabetic_retinopathy",
    }
    
    classification_label = label_map.get(label_code)
    if not classification_label:
        return None
    
    return (number_prefix, classification_label)


async def process_noise_folder(
    noise_dir: Path,
    dataset_id: UUID,
    tracker: ProgressTracker,
) -> Tuple[List[Image], List, List[UUID], dict]:
    """
    Process Noise folder images for quality annotations (good vs bad).
    
    Args:
        noise_dir: Path to Noise/ folder
        dataset_id: Dataset UUID
        tracker: Progress tracker
    
    Returns:
        Tuple of (images, quality_annotations, image_ids)
    """
    all_images: List[Image] = []
    all_quality_annotations: List = []
    image_ids: List[UUID] = []
    label_dict: dict = {}

    if not noise_dir.exists():
        logger.warning(f"Noise directory not found: {noise_dir}")
        return all_images, all_quality_annotations, image_ids, label_dict
    
    # Find all images in Noise folder (both .JPG and .jpg)
    image_paths = await asyncio.to_thread(find_images, noise_dir, recursive=False)
    logger.info(f"Found {len(image_paths)} images in Noise folder")
    
    for image_path in image_paths:
        try:
            image_filename = image_path.name
            image_stem = image_path.stem
            
            # Parse quality from filename
            quality_result = parse_quality_from_filename(image_filename)
            if not quality_result:
                logger.warning(f"Could not parse quality from filename: {image_filename}")
                tracker.update(success=False)
                tracker.record_error(
                    error_type="filename_parsing",
                    error_message=f"Could not parse quality from filename: {image_filename}",
                    item_id=image_stem,
                    item_path=str(image_path),
                )
                continue
            
            number_prefix, quality_label = quality_result
            
            # Generate image ID
            image_id = generate_image_uuid(dataset_id, f"noise_{image_stem}")
            
            # Create image with automatic metadata extraction
            image = Image(
                image_id=image_id,
                dataset_id=dataset_id,
                original_image_id=f"noise_{image_stem}",
                **get_image_metadata_dict(image_path),
                modality="fundus",
            )
            all_images.append(image)
            image_ids.append(image_id)
            label_dict[image_id] = quality_label

            quality_annotation = await process_quality_annotation(
                quality_type="overall",
                image_id=image_id,
                quality_label=quality_label,
                scale_description="HRF Noise folder quality (good vs bad)",
            )
            all_quality_annotations.append(quality_annotation)
            
            tracker.update(success=True)
            tracker.record_success("image")
            
        except Exception as e:
            logger.error(f"Failed to process Noise image {image_path}: {e}", exc_info=True)
            tracker.update(success=False)
            tracker.record_error(
                error_type="image_processing",
                error_message=str(e),
                item_id=image_path.stem,
                item_path=str(image_path),
            )
    
    return all_images, all_quality_annotations, image_ids, label_dict


async def process_documents_folder(
    documents_dir: Path,
    dataset_id: UUID,
    tracker: ProgressTracker,
) -> Tuple[List[Image], List, List[UUID], dict]:
    """
    Process Documents folder images for binary disease classifications.
    
    Args:
        documents_dir: Path to documents/ folder
        dataset_id: Dataset UUID
        tracker: Progress tracker
    
    Returns:
        Tuple of (images, classifications, image_ids)
    """
    all_images: List[Image] = []
    all_classifications: List = []
    image_ids: List[UUID] = []
    label_dict: dict = {}

    if not documents_dir.exists():
        logger.warning(f"Documents directory not found: {documents_dir}")
        return all_images, all_classifications, image_ids, label_dict
    
    # Find all images in documents folder (both .JPG and .jpg)
    image_paths = await asyncio.to_thread(find_images, documents_dir, recursive=False)
    logger.info(f"Found {len(image_paths)} images in Documents folder")
    
    for image_path in image_paths:
        try:
            image_filename = image_path.name
            image_stem = image_path.stem
            
            # Parse classification from filename
            classification_result = parse_classification_from_filename(image_filename)
            if not classification_result:
                logger.warning(f"Could not parse classification from filename: {image_filename}")
                tracker.update(success=False)
                tracker.record_error(
                    error_type="filename_parsing",
                    error_message=f"Could not parse classification from filename: {image_filename}",
                    item_id=image_stem,
                    item_path=str(image_path),
                )
                continue
            
            number_prefix, classification_label = classification_result
            
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
            label_dict[image_id] = classification_label

            # Process as binary classifications for each disease
            is_dr = classification_label == "diabetic_retinopathy"
            is_glaucoma = classification_label == "glaucoma"
            
            dr_classifications = await process_classification(
                class_value=is_dr,
                task_type="binary",
                class_name="DR",
                image_id=image_id,
                annotation_method="manual",
            )
            all_classifications.extend(dr_classifications)
            
            glaucoma_classifications = await process_classification(
                class_value=is_glaucoma,
                task_type="binary",
                class_name="glaucoma",
                image_id=image_id,
                annotation_method="manual",
            )
            all_classifications.extend(glaucoma_classifications)
            
            tracker.update(success=True)
            tracker.record_success("image")
            
        except Exception as e:
            logger.error(f"Failed to process Documents image {image_path}: {e}", exc_info=True)
            tracker.update(success=False)
            tracker.record_error(
                error_type="image_processing",
                error_message=str(e),
                item_id=image_path.stem,
                item_path=str(image_path),
            )

    return all_images, all_classifications, image_ids, label_dict


async def ingest_hrf() -> OperationStatistics:
    """
    Main ingestion function for HRF dataset.
    
    Strategy:
    - Process Noise folder: Meta classification task (good vs bad quality)
    - Process Documents folder: Binary disease classifications (h/g/dr)
    - Bulk upsert all data
    - Assign all images to a single "train" split
    
    Returns:
        OperationStatistics with success/error counts
    """
    data_root = get_data_root() / "23_HRF"
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
            "HRF (High-Resolution Fundus) dataset with two tasks: "
            "(1) Meta classification for image quality (good vs bad) from Noise folder, "
            "(2) Binary disease classifications (healthy, glaucoma, diabetic_retinopathy) from Documents folder."
        ),
    )
    await upsert_dataset(dataset)
    
    # Step 2: Count items first (lightweight - just count images)
    noise_dir = data_root / "Noise"
    documents_dir = data_root / "documents"
    
    noise_count = 0
    if noise_dir.exists():
        noise_image_paths = await asyncio.to_thread(find_images, noise_dir, recursive=False)
        noise_count = len(noise_image_paths)
        logger.info(f"  Noise folder: {noise_count} images")
    
    documents_count = 0
    if documents_dir.exists():
        documents_image_paths = await asyncio.to_thread(find_images, documents_dir, recursive=False)
        documents_count = len(documents_image_paths)
        logger.info(f"  Documents folder: {documents_count} images")
    
    total_count = noise_count + documents_count
    logger.info(f"Total items to process: {total_count}")
    
    # Step 3: Setup progress tracker with actual count
    tracker = ProgressTracker(
        total=total_count,
        description=f"Ingesting {DATASET_NAME}"
    )
    
    # Step 4: Process Noise folder (meta classification task)
    logger.info("=" * 80)
    logger.info("Processing Noise folder (meta classification: good vs bad quality)...")
    logger.info("=" * 80)
    noise_dir = data_root / "Noise"
    noise_images, noise_quality_annotations, noise_image_ids, noise_labels = await process_noise_folder(
        noise_dir, dataset_id, tracker
    )
    logger.info(f"Processed {len(noise_images)} images from Noise folder")

    # Step 5: Process Documents folder (binary disease classifications)
    logger.info("=" * 80)
    logger.info("Processing Documents folder (binary disease classifications: h/g/dr)...")
    logger.info("=" * 80)
    documents_dir = data_root / "documents"
    doc_images, doc_classifications, doc_image_ids, doc_labels = await process_documents_folder(
        documents_dir, dataset_id, tracker
    )
    logger.info(f"Processed {len(doc_images)} images from Documents folder")
    
    # Combine all results
    all_images = noise_images + doc_images
    all_classifications = doc_classifications
    image_labels = {**noise_labels, **doc_labels}
    all_quality_annotations = noise_quality_annotations
    all_image_ids = noise_image_ids + doc_image_ids
    
    # Step 6: Bulk upsert images
    logger.info(f"Upserting {len(all_images)} images...")
    if all_images:
        try:
            await bulk_upsert_images(all_images, batch_size=1000)
            logger.info(f"Successfully upserted {len(all_images)} images")
        except Exception as e:
            logger.error(f"Failed to bulk upsert images: {e}")
            raise
    
    # Step 7: Bulk upsert classifications
    logger.info(f"Upserting {len(all_classifications)} classification annotations...")
    if all_classifications:
        try:
            await bulk_upsert_classification_annotations(all_classifications, batch_size=1000)
            logger.info(f"Successfully upserted {len(all_classifications)} classification annotations")
        except Exception as e:
            logger.error(f"Failed to bulk upsert classifications: {e}")
            raise
    
    # Step 7b: Upsert quality annotations from Noise folder
    logger.info(f"Upserting {len(all_quality_annotations)} quality annotations...")
    for qa in all_quality_annotations:
        try:
            await upsert_quality_annotation(qa)
        except Exception as e:
            logger.error(f"Failed to upsert quality annotation: {e}")
    
    # Step 8: Register splits — stratified 90/10 train+test, then 90/10 train+val
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
    logger.info(f"    - Noise folder: {len(noise_images)}")
    logger.info(f"    - Documents folder: {len(doc_images)}")
    logger.info(f"  Classification annotations: {len(all_classifications)}")
    logger.info(f"    - Documents (disease): {len(doc_classifications)}")
    logger.info(f"  Quality annotations: {len(all_quality_annotations)}")
    logger.info(f"    - Noise (quality): {len(noise_quality_annotations)}")
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
        stats = await ingest_hrf()
        
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
