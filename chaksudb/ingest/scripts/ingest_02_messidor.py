"""
Ingestion script for MESSIDOR dataset.

Dataset: MESSIDOR - Diabetic Retinopathy and Diabetic Macular Edema
Structure: Single CSV with adjudicated DR grade, DME, and gradability
Annotations: Disease grading (DR), Classification (DME), Quality (gradability)
Tasks: Grading (DR, 0-3 scale), Classification (DME binary), Quality (gradability binary)
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from uuid import UUID

from chaksudb.common.progress import ProgressTracker, OperationStatistics
from chaksudb.config.config import get_data_root
from chaksudb.db.models import (
    Dataset,
    Image,
    DiseaseGrading,
    ClassificationAnnotation,
    QualityAnnotation,
)
from chaksudb.db.queries import (
    upsert_dataset,
    bulk_upsert_images,
    bulk_upsert_disease_gradings,
    bulk_upsert_classification_annotations,
    bulk_upsert_quality_annotations,
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
from chaksudb.ingest.framework.task_processors.classification_processor import process_classification
from chaksudb.ingest.framework.task_processors.quality_processor import process_quality_annotation
from chaksudb.ingest.framework.split_assigner import auto_stratified_splits

logger = logging.getLogger(__name__)

# Dataset metadata
DATASET_NAME = "MESSIDOR"
DATASET_URL = "https://www.adcis.net/en/third-party/messidor/"
DATASET_LICENSE = "Custom - Educational and research use"


async def ingest_messidor() -> OperationStatistics:
    """
    Main ingestion function for MESSIDOR dataset.
    
    Returns:
        OperationStatistics with success/error counts
    """
    data_root = get_data_root() / "02_MESSIDOR"
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
    csv_path = data_root / "messidor_data.csv"
    
    csv_rows = await asyncio.to_thread(read_csv_auto, csv_path)
    total_count = len(csv_rows)
    
    logger.info(f"Found {total_count} images")
    
    # Step 3: Setup progress tracker
    tracker = ProgressTracker(
        total=total_count,
        description=f"Ingesting {DATASET_NAME}"
    )
    
    # Collect items for bulk upsert
    all_images: List[Image] = []
    all_dr_gradings: List[DiseaseGrading] = []
    all_dme_classifications: List[ClassificationAnnotation] = []
    all_quality_annotations: List[QualityAnnotation] = []
    image_ids_for_split: List[UUID] = []
    image_labels: dict = {}  # image_id → DR grade for stratified splitting
    
    async def process_row(row, idx):
        """Process a single CSV row."""
        try:
            image_name = row["image_id"]
            image_id = generate_image_uuid(dataset_id, image_name)
            
            # Find image file in train_org directory
            # Handle case-insensitive file extensions (.jpg vs .JPG)
            image_dir = data_root / "train_org"
            image_path = image_dir / image_name
            
            # Try exact match first
            if not await asyncio.to_thread(image_path.exists):
                # Try with different case for extension
                stem = image_path.stem
                suffix = image_path.suffix
                if suffix.lower() == ".jpg":
                    # Try uppercase
                    alt_path = image_dir / f"{stem}.JPG"
                    if await asyncio.to_thread(alt_path.exists):
                        image_path = alt_path
                    else:
                        tracker.record_error(
                            error_type="file_not_found",
                            error_message=f"Image not found: {image_name}",
                            item_id=image_name,
                        )
                        tracker.update(success=False)
                        return
                elif suffix.lower() == ".png":
                    # Try uppercase
                    alt_path = image_dir / f"{stem}.PNG"
                    if await asyncio.to_thread(alt_path.exists):
                        image_path = alt_path
                    else:
                        tracker.record_error(
                            error_type="file_not_found",
                            error_message=f"Image not found: {image_name}",
                            item_id=image_name,
                        )
                        tracker.update(success=False)
                        return
                else:
                    tracker.record_error(
                        error_type="file_not_found",
                        error_message=f"Image not found: {image_name}",
                        item_id=image_name,
                    )
                    tracker.update(success=False)
                    return
            
            # Create image with automatic metadata extraction
            image = Image(
                image_id=image_id,
                dataset_id=dataset_id,
                original_image_id=image_name,
                **get_image_metadata_dict(image_path),
                modality="fundus",
            )
            all_images.append(image)
            image_ids_for_split.append(image_id)
            image_labels[image_id] = int(row["adjudicated_dr_grade"])

            # Process annotations using task processors
            # Task processors automatically handle: provenance, UUID generation, timestamps
            
            # 1. Process DR grading (0-3 scale) - scale auto-registered on first call
            dr_grading = await process_disease_grade(
                grade_value=int(row["adjudicated_dr_grade"]),
                disease_type="DR",
                scale_name="ICDR_0_4",
                image_id=image_id,
                annotation_method="manual",
            )
            all_dr_gradings.append(dr_grading)
            
            # 2. Process DME classification (binary: 0/1)
            dme_classification = await process_classification(
                class_value=bool(int(row["adjudicated_dme"])),
                task_type="binary",
                class_name="DME",
                image_id=image_id,
                annotation_method="manual",
            )
            all_dme_classifications.extend(dme_classification)
            
            # 3. Process gradability quality annotation (binary: 0/1)
            gradable_value = int(row["adjudicated_gradable"])
            quality_annotation = await process_quality_annotation(
                quality_type="gradability",
                image_id=image_id,
                quality_score=float(gradable_value),
                quality_label="gradable" if gradable_value == 1 else "not_gradable",
            )
            all_quality_annotations.append(quality_annotation)
            
            tracker.update(count=1, success=True)
            
        except Exception as e:
            tracker.update(count=1, success=False)
            tracker.record_error(
                error_type="processing",
                error_message=str(e),
                item_id=row.get("image_id"),
            )
            logger.error(f"Failed to process row {idx}: {e}")
    
    # Step 3: Process CSV with provenance tracking
    logger.info("Processing annotations...")
    # MESSIDOR primary annotation type is "grading" (DR grading is the main task)
    # It also has DME classification and quality, but grading is primary
    # Grading scale will be auto-registered by task processor on first call
    stats, raw_file_id, chain_id = await process_csv(
        csv_path,
        dataset_id,
        "grading",  # Primary annotation type
        process_row
    )
    
    # Log provenance information
    logger.info(f"CSV registered: raw_file_id={raw_file_id}, chain_id={chain_id}")
    
    # Step 4: Bulk upsert - images first, then annotations in parallel
    logger.info(f"Upserting {len(all_images)} images...")
    await bulk_upsert_images(all_images, batch_size=1000)
    
    logger.info(f"Upserting {len(all_dr_gradings)} DR gradings, {len(all_dme_classifications)} DME classifications, and {len(all_quality_annotations)} quality annotations...")
    await asyncio.gather(
        bulk_upsert_disease_gradings(all_dr_gradings, batch_size=1000),
        bulk_upsert_classification_annotations(all_dme_classifications, batch_size=1000),
        bulk_upsert_quality_annotations(all_quality_annotations, batch_size=1000),
    )
    
    # Step 5: Register splits — stratified 90/10 train+test, then 90/10 train+val
    logger.info("Registering dataset splits...")
    await auto_stratified_splits(
        dataset_id=dataset_id,
        split_assignments={"train": image_ids_for_split},
        labels=image_labels,
        split_type="explicit",
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
        stats = await ingest_messidor()
        
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
