"""
Ingestion script for JICHI dataset.

Dataset: JICHI Fundus Image Dataset
Structure: 
  - list.csv: CSV file with DR grading using custom labels
  - documents/: Fundus images (.jpg)
Annotations: 
  - DR grading (custom scale: ndr, sdr, ppdr, pdr mapped to ICDR values 0, 1, 3, 4)
Tasks: DR grading (custom scale mapping)
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List
from uuid import UUID

from chaksudb.common.progress import ProgressTracker, OperationStatistics
from chaksudb.config.config import get_data_root
from chaksudb.db.models import Dataset, Expert, ExpertAnnotation, Image, DiseaseGrading
from chaksudb.db.queries import (
    bulk_upsert_disease_gradings,
    bulk_upsert_expert_annotations,
    bulk_upsert_images,
    upsert_dataset,
    upsert_expert,
)
from chaksudb.ingest.framework import (
    get_image_metadata_dict,
    process_csv,
    read_csv_auto,
)
from chaksudb.ingest.framework.gen_uuid import (
    generate_dataset_uuid,
    generate_expert_annotation_uuid,
    generate_expert_uuid,
    generate_grading_scale_uuid,
    generate_image_uuid,
)
from chaksudb.ingest.framework.provenance_context import get_current_provenance
from chaksudb.ingest.framework.task_processors.grading_processor import (
    get_or_create_scale,
    process_disease_grade,
)
from chaksudb.ingest.framework.grading_scales import create_mapping, find_scale_by_name
from chaksudb.ingest.framework.split_assigner import auto_stratified_splits

logger = logging.getLogger(__name__)

# Dataset metadata
DATASET_NAME = "JICHI"
DATASET_URL = "https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0179790"
DATASET_LICENSE = "Unknown"

# Scale name for Davis grading
DAVIS_SCALE = "Davis_DR"
ICDR_SCALE = "ICDR_0_4"

# Mapping from custom labels to ICDR scale values
LABEL_TO_GRADE = {
    "ndr": 0,   # No diabetic retinopathy
    "sdr": 1,   # Slight diabetic retinopathy (mild NPDR)
    "ppdr": 3,  # Pre-proliferative diabetic retinopathy (severe NPDR)
    "pdr": 4,   # Proliferative diabetic retinopathy
}

# Expert names for the two grading methods
EXPERT_CONCATENATED = "Davis_Concatenated"
EXPERT_ONE_FIGURE = "Davis_OneFigure"


async def register_experts(dataset_id: UUID) -> Dict[str, UUID]:
    """Register the two Davis grading experts (concatenated vs one figure)."""
    expert_ids = {}
    
    for expert_name in [EXPERT_CONCATENATED, EXPERT_ONE_FIGURE]:
        expert_id = generate_expert_uuid(
            dataset_id=dataset_id,
            model_id=None,
            expert_name=expert_name,
        )
        
        expert = Expert(
            expert_id=expert_id,
            expert_name=expert_name,
            dataset_id=dataset_id,
            model_id=None,
            expertise_area="DR grading",
        )
        
        await upsert_expert(expert)
        expert_ids[expert_name] = expert_id
        logger.info(f"Registered expert: {expert_name}")
    
    return expert_ids


async def ingest_jichi() -> OperationStatistics:
    """
    Main ingestion function for JICHI dataset.
    
    Returns:
        OperationStatistics with success/error counts
    """
    data_root = get_data_root() / "31_JICHI"
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
    
    # Step 1.5: Register experts
    logger.info("Registering experts...")
    expert_ids = await register_experts(dataset_id)
    
    # Step 1.6: Register Davis scale and mappings to ICDR_0_4
    logger.info("Registering Davis scale...")
    davis_scale_id = await get_or_create_scale(
        scale_name=DAVIS_SCALE,
        disease_type="DR",
        scale_description="Davis DR grading scale (mapped to ICDR values)",
        min_value=0,
        max_value=4,
        value_labels={
            "0": "ndr (No DR)",
            "1": "sdr (Mild NPDR)",
            "3": "ppdr (Severe NPDR)",
            "4": "pdr (PDR)",
        },
    )
    logger.info(f"Davis scale registered with scale_id: {davis_scale_id}")
    
    # Get ICDR scale ID (should already exist, but get_or_create will handle it)
    logger.info("Getting ICDR_0_4 scale...")
    icdr_scale_id = await get_or_create_scale(
        scale_name=ICDR_SCALE,
        disease_type="DR",
    )
    
    # Register mappings: Davis scale values -> ICDR values
    logger.info("Registering scale mappings to ICDR_0_4...")
    for label, icdr_value in LABEL_TO_GRADE.items():
        # Map the numeric grade value (as string) to ICDR
        grade_value_str = str(icdr_value)
        await create_mapping(
            source_scale_id=davis_scale_id,
            target_scale_id=icdr_scale_id,
            source_value=grade_value_str,
            target_value=icdr_value,
            mapping_confidence="exact",
        )
        logger.debug(f"Registered mapping: Davis_DR:{grade_value_str} -> ICDR_0_4:{icdr_value}")
    
    logger.info("Scale mappings registered successfully")
    
    # Step 2: Count total rows for progress tracking
    logger.info("Counting CSV rows...")
    csv_path = data_root / "list.csv"
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
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
    all_gradings: List[DiseaseGrading] = []
    all_expert_annotations: List[ExpertAnnotation] = []
    image_ids_for_split: List[UUID] = []
    image_labels: dict = {}  # image_id → DR grade for stratified splitting
    image_id_map: Dict[str, UUID] = {}  # Track image IDs by filename
    
    image_dir = data_root / "documents"
    
    async def process_row(row, idx):
        """Process a single CSV row."""
        try:
            image_filename = row["Image"].strip()
            image_id_str = image_filename
            
            # Generate or get image ID
            if image_id_str not in image_id_map:
                image_id = generate_image_uuid(dataset_id, image_id_str)
                image_id_map[image_id_str] = image_id
            else:
                image_id = image_id_map[image_id_str]
            
            # Find image file
            image_path = image_dir / image_filename
            if not await asyncio.to_thread(image_path.exists):
                # Try case variations
                stem = image_path.stem
                for ext in [".jpg", ".JPG", ".jpeg", ".JPEG"]:
                    candidate = image_dir / f"{stem}{ext}"
                    if await asyncio.to_thread(candidate.exists):
                        image_path = candidate
                        break
                else:
                    tracker.record_error(
                        error_type="file_not_found",
                        error_message=f"Image not found: {image_filename}",
                        item_id=image_filename,
                    )
                    tracker.update(success=False)
                    return
            
            # Create image if not already created
            if image_id not in [img.image_id for img in all_images]:
                image = Image(
                    image_id=image_id,
                    dataset_id=dataset_id,
                    original_image_id=image_filename,
                    **get_image_metadata_dict(image_path),
                    modality="fundus",
                )
                all_images.append(image)
                image_ids_for_split.append(image_id)
            
            # Get provenance
            raw_data_id, provenance_chain_id = get_current_provenance()
            
            # Process both Davis grading columns using the same scale but different experts
            # This allows both gradings to be stored separately even if they have the same value
            
            # Process Davis_grading_of_concatenated_figures
            concatenated_grade_label = row.get("Davis_grading_of_concatenated_figures", "").strip().lower()
            if concatenated_grade_label and concatenated_grade_label in LABEL_TO_GRADE:
                grade_value = LABEL_TO_GRADE[concatenated_grade_label]
                image_labels[image_id] = grade_value
                expert_id = expert_ids[EXPERT_CONCATENATED]
                
                # Create expert annotation
                expert_annotation_id = generate_expert_annotation_uuid(
                    expert_id=expert_id,
                    annotation_task="grading",
                    raw_data_id=raw_data_id,
                )
                
                expert_annotation = ExpertAnnotation(
                    expert_annotation_id=expert_annotation_id,
                    expert_id=expert_id,
                    annotation_task="grading",
                    raw_data_id=raw_data_id,
                    annotation_value={"grade": concatenated_grade_label, "method": "concatenated"},
                )
                all_expert_annotations.append(expert_annotation)
                
                # Process grading with expert_annotation_id
                grading = await process_disease_grade(
                    grade_value=grade_value,
                    disease_type="DR",
                    scale_name=DAVIS_SCALE,
                    image_id=image_id,
                    scale_description="Davis DR grading scale (mapped to ICDR values)",
                    min_value=0,
                    max_value=4,
                    value_labels={
                        "0": "ndr (No DR)",
                        "1": "sdr (Mild NPDR)",
                        "3": "ppdr (Severe NPDR)",
                        "4": "pdr (PDR)",
                    },
                    grade_label=concatenated_grade_label,
                    expert_annotation_id=expert_annotation_id,
                    raw_data_id=raw_data_id,
                    provenance_chain_id=provenance_chain_id,
                    annotation_method="manual",
                )
                all_gradings.append(grading)
            
            # Process Davis_grading_of_one_figure
            one_figure_grade_label = row.get("Davis_grading_of_one_figure", "").strip().lower()
            if one_figure_grade_label and one_figure_grade_label in LABEL_TO_GRADE:
                grade_value = LABEL_TO_GRADE[one_figure_grade_label]
                expert_id = expert_ids[EXPERT_ONE_FIGURE]
                
                # Create expert annotation
                expert_annotation_id = generate_expert_annotation_uuid(
                    expert_id=expert_id,
                    annotation_task="grading",
                    raw_data_id=raw_data_id,
                )
                
                expert_annotation = ExpertAnnotation(
                    expert_annotation_id=expert_annotation_id,
                    expert_id=expert_id,
                    annotation_task="grading",
                    raw_data_id=raw_data_id,
                    annotation_value={"grade": one_figure_grade_label, "method": "one_figure"},
                )
                all_expert_annotations.append(expert_annotation)
                
                # Process grading with expert_annotation_id
                grading = await process_disease_grade(
                    grade_value=grade_value,
                    disease_type="DR",
                    scale_name=DAVIS_SCALE,
                    image_id=image_id,
                    scale_description="Davis DR grading scale (mapped to ICDR values)",
                    min_value=0,
                    max_value=4,
                    value_labels={
                        "0": "ndr (No DR)",
                        "1": "sdr (Mild NPDR)",
                        "3": "ppdr (Severe NPDR)",
                        "4": "pdr (PDR)",
                    },
                    grade_label=one_figure_grade_label,
                    expert_annotation_id=expert_annotation_id,
                    raw_data_id=raw_data_id,
                    provenance_chain_id=provenance_chain_id,
                    annotation_method="manual",
                )
                all_gradings.append(grading)
            
            # Log warning if either label is unknown
            if concatenated_grade_label and concatenated_grade_label not in LABEL_TO_GRADE:
                logger.warning(f"Unknown grade label '{concatenated_grade_label}' for {image_filename}")
            if one_figure_grade_label and one_figure_grade_label not in LABEL_TO_GRADE:
                logger.warning(f"Unknown grade label '{one_figure_grade_label}' for {image_filename}")
            
            tracker.update(success=True)
            tracker.record_success("image")
            
        except Exception as e:
            tracker.update(success=False)
            tracker.record_error(
                error_type="processing",
                error_message=str(e),
                item_id=row.get("Image", "unknown"),
            )
            logger.error(f"Failed to process row {idx}: {e}")
    
    # Step 4: Process CSV with provenance tracking
    logger.info("Processing annotations...")
    stats, raw_file_id, chain_id = await process_csv(
        csv_path,
        dataset_id,
        "grading",  # Primary annotation type
        process_row,
        progress_tracker=tracker,
    )
    
    logger.info(f"Processed {stats.successful_items} rows from list.csv")
    logger.info(f"CSV registered: raw_file_id={raw_file_id}, chain_id={chain_id}")
    
    # Step 5: Bulk upsert - images first, then expert annotations, then gradings
    logger.info(f"Upserting {len(all_images)} images...")
    await bulk_upsert_images(all_images, batch_size=1000)
    
    logger.info(f"Upserting {len(all_expert_annotations)} expert annotations...")
    await bulk_upsert_expert_annotations(all_expert_annotations, batch_size=1000)
    
    logger.info(f"Upserting {len(all_gradings)} DR gradings...")
    await bulk_upsert_disease_gradings(all_gradings, batch_size=1000)
    
    # Step 6: Register splits — stratified 90/10 train+test, then 90/10 train+val
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
        stats = await ingest_jichi()
        
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
