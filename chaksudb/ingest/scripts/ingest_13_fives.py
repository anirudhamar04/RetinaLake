"""
Ingestion script for FIVES dataset.

Dataset: FIVES - Fundus Image Vessel Segmentation
Structure: Train/test splits with Original images and Ground truth masks
Annotations:
  - Vessel segmentation (binary masks)
  - Disease classification (from filename: A=AMD, D=DR, G=Glaucoma, N=Normal)
  - Quality assessment (from Excel: IC=contrast, Blur, LC=illumination)

Key Features:
  - 800 high-resolution multi-disease color fundus photographs
  - Pixelwise manual vessel annotations
  - Quality assessment via Excel file (binary indicators)
  - Explicit train/test splits
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from uuid import UUID

from chaksudb.common.progress import ProgressTracker, OperationStatistics
from chaksudb.config.config import get_data_root
from chaksudb.db.models import (
    Dataset,
    Image,
    SegmentationAnnotation,
    ClassificationAnnotation,
    QualityAnnotation,
)
from chaksudb.db.queries import (
    upsert_dataset,
    bulk_upsert_images,
    upsert_segmentation_annotation,
    bulk_upsert_classification_annotations,
    bulk_upsert_quality_annotations,
)
from chaksudb.ingest.framework import (
    get_image_metadata_dict,
    find_images,
    process_excel,
)
from chaksudb.ingest.framework.gen_uuid import (
    generate_dataset_uuid,
    generate_image_uuid,
)
from chaksudb.ingest.framework.task_processors.segmentation_processor import (
    process_segmentation_from_binary_mask,
)
from chaksudb.ingest.framework.task_processors.classification_processor import (
    process_classification,
)
from chaksudb.ingest.framework.task_processors.quality_processor import (
    process_quality_annotation,
)
from chaksudb.ingest.framework.provenance_context import get_current_provenance
from chaksudb.ingest.framework.split_assigner import (
    register_standard_splits,
    bulk_assign_images_to_split,
)

logger = logging.getLogger(__name__)

# Dataset metadata
DATASET_NAME = "FIVES"
DATASET_URL = "https://figshare.com/articles/figure/FIVES_A_Fundus_Image_Dataset_for_AI-based_Vessel_Segmentation/19688169"
DATASET_LICENSE = "Research/Academic Use"

# Disease code mapping
DISEASE_MAP = {
    "A": "AMD",  # Age-related Macular Degeneration
    "D": "DR",   # Diabetic Retinopathy
    "G": "Glaucoma",
    "N": "Normal",
}

# Each FIVES image is exactly one disease (mutually exclusive) -> a multi_class task.
# Deterministic index->label map so class_index is stable across runs.
FIVES_DISEASE_LABELS = {i: name for i, name in enumerate(DISEASE_MAP.values())}


def parse_fives_filename(filename: str) -> Dict[str, str]:
    """
    Parse FIVES filename: {number}_{disease}.png
    
    Examples:
        5_A.png -> {number: "5", disease_code: "A", disease_name: "AMD", original_id: "5_A"}
        64_D.png -> {number: "64", disease_code: "D", disease_name: "DR", original_id: "64_D"}
    
    Args:
        filename: Filename (with or without extension)
    
    Returns:
        Dictionary with parsed components
    """
    # Remove extension
    stem = Path(filename).stem
    
    # Split by underscore
    parts = stem.split("_")
    if len(parts) != 2:
        raise ValueError(f"Unexpected filename format: {filename}. Expected: {{number}}_{{disease}}.png")
    
    number, disease_code = parts
    
    if disease_code not in DISEASE_MAP:
        raise ValueError(f"Unknown disease code: {disease_code}. Expected: A, D, G, or N")
    
    return {
        "number": number,
        "disease_code": disease_code,
        "disease_name": DISEASE_MAP[disease_code],
        "original_image_id": f"{number}_{disease_code}",
    }


async def process_quality_excel(
    excel_path: Path,
    dataset_id: UUID,
    sheet_name: str,
    tracker: ProgressTracker,
) -> Tuple[Dict[str, Dict], UUID, UUID]:
    """
    Process quality assessment Excel sheet and build lookup map.
    
    Args:
        excel_path: Path to Quality Assessment.xlsx
        dataset_id: Dataset UUID
        sheet_name: Sheet name ("train" or "test")
        tracker: Progress tracker
    
    Returns:
        Tuple of (quality_lookup, raw_file_id, chain_id)
        quality_lookup: Dictionary mapping "{number}_{disease}" -> {
            contrast: 0/1, blur: 0/1, illumination: 0/1,
            raw_file_id: UUID, chain_id: UUID
        }
    """
    quality_lookup: Dict[str, Dict] = {}
    
    async def process_quality_row(row: Dict, idx: int) -> None:
        """Process a single row from quality Excel."""
        try:
            # Get provenance from context (set by process_excel)
            raw_file_id, chain_id = get_current_provenance()
            
            # Extract data
            disease = str(row.get("Disease", "")).strip().upper()
            number = row.get("Number")
            ic = row.get("IC")  # Image Contrast
            blur = row.get("Blur")
            lc = row.get("LC")  # Lighting Condition
            
            # Validate
            if not disease or disease not in DISEASE_MAP:
                logger.warning(f"Row {idx}: Invalid disease code: {disease}")
                return
            
            if number is None:
                logger.warning(f"Row {idx}: Missing number")
                return
            
            # Convert to int if needed
            if isinstance(number, str):
                try:
                    number = int(number.strip())
                except ValueError:
                    logger.warning(f"Row {idx}: Invalid number: {number}")
                    return
            
            # Build lookup key
            key = f"{number}_{disease}"
            
            # Validate quality values (should be 0 or 1)
            def validate_binary(value, name: str) -> Optional[int]:
                if value is None:
                    return None
                if isinstance(value, str):
                    value = value.strip()
                    if value.lower() in ["0", "false", "no", ""]:
                        return 0
                    if value.lower() in ["1", "true", "yes"]:
                        return 1
                    try:
                        return int(value)
                    except ValueError:
                        logger.warning(f"Row {idx}: Invalid {name} value: {value}")
                        return None
                try:
                    val = int(value)
                    if val not in [0, 1]:
                        logger.warning(f"Row {idx}: {name} should be 0 or 1, got: {val}")
                    return val
                except (ValueError, TypeError):
                    logger.warning(f"Row {idx}: Invalid {name} value: {value}")
                    return None
            
            ic_val = validate_binary(ic, "IC")
            blur_val = validate_binary(blur, "Blur")
            lc_val = validate_binary(lc, "LC")
            
            # Store in lookup (provenance will be added after Excel processing)
            quality_lookup[key] = {
                "contrast": ic_val if ic_val is not None else 1,  # Default to 1 if missing
                "blur": blur_val if blur_val is not None else 1,
                "illumination": lc_val if lc_val is not None else 1,
            }
            
            tracker.update(success=True)
            tracker.record_success("quality_row")
            
        except Exception as e:
            tracker.update(success=False)
            tracker.record_error(
                error_type="quality_row_processing",
                error_message=str(e),
                item_id=f"row_{idx}",
            )
            logger.error(f"Failed to process quality row {idx}: {e}")
    
    # Process Excel with automatic provenance
    stats, raw_file_id, chain_id = await process_excel(
        excel_path=excel_path,
        dataset_id=dataset_id,
        unified_annotation_type="quality",
        process_row_fn=process_quality_row,
        sheet_name=sheet_name,
        progress_tracker=tracker,
        skip_errors=True,
    )
    
    logger.info(
        f"Processed {sheet_name} quality sheet: {stats.successful_items} successful, "
        f"{stats.failed_items} failed. Built lookup for {len(quality_lookup)} images."
    )
    
    # Add provenance to all entries
    for key in quality_lookup:
        quality_lookup[key]["raw_file_id"] = raw_file_id
        quality_lookup[key]["chain_id"] = chain_id
    
    return quality_lookup, raw_file_id, chain_id


async def process_split(
    data_root: Path,
    dataset_id: UUID,
    split_name: str,
    quality_lookup: Dict[str, Dict],
    excel_raw_file_id: UUID,
    excel_chain_id: UUID,
    tracker: ProgressTracker,
) -> Tuple[List[Image], List[SegmentationAnnotation], List[ClassificationAnnotation], List[QualityAnnotation], List[UUID]]:
    """
    Process a single split (train or test).
    
    Args:
        data_root: Dataset root directory
        dataset_id: Dataset UUID
        split_name: "train" or "test"
        quality_lookup: Quality data lookup map (contains raw_file_id and chain_id)
        excel_raw_file_id: Raw file ID from Excel (for provenance)
        excel_chain_id: Provenance chain ID from Excel
        tracker: Progress tracker
    
    Returns:
        Tuple of (images, segmentations, classifications, qualities, image_ids)
    """
    all_images: List[Image] = []
    all_segmentations: List[SegmentationAnnotation] = []
    all_classifications: List[ClassificationAnnotation] = []
    all_qualities: List[QualityAnnotation] = []
    image_ids: List[UUID] = []
    
    # Paths
    original_dir = data_root / split_name / "Original"
    mask_dir = data_root / split_name / "Ground truth"
    
    if not original_dir.exists():
        logger.warning(f"Original directory not found: {original_dir}")
        return all_images, all_segmentations, all_classifications, all_qualities, image_ids
    
    if not mask_dir.exists():
        logger.warning(f"Ground truth directory not found: {mask_dir}")
        return all_images, all_segmentations, all_classifications, all_qualities, image_ids
    
    # Find all images
    image_paths = await asyncio.to_thread(find_images, original_dir)
    logger.info(f"Found {len(image_paths)} images in {split_name} split")
    
    for image_path in image_paths:
        try:
            # Parse filename
            parsed = parse_fives_filename(image_path.name)
            original_image_id = parsed["original_image_id"]
            disease_name = parsed["disease_name"]
            disease_code = parsed["disease_code"]
            
            # Generate image UUID
            image_id = generate_image_uuid(dataset_id, original_image_id)
            
            # Create image model
            image = Image(
                image_id=image_id,
                dataset_id=dataset_id,
                original_image_id=original_image_id,
                **get_image_metadata_dict(image_path),
                modality="fundus",
                acquisition_date=None,
                image_quality=None,
                # Store disease info in comorbidities
                comorbidities={"disease_code": disease_code, "disease_name": disease_name},
            )
            
            all_images.append(image)
            image_ids.append(image_id)
            
            # Find corresponding mask
            mask_path = mask_dir / image_path.name
            if not mask_path.exists():
                tracker.record_error(
                    error_type="mask_not_found",
                    error_message=f"Mask not found for image: {image_path.name}",
                    item_id=original_image_id,
                    item_path=str(image_path),
                )
                logger.warning(f"Mask not found: {mask_path}")
                continue
            
            # Process vessel segmentation
            segmentation = await process_segmentation_from_binary_mask(
                mask_path=mask_path,
                annotation_type="vessels",
                image_id=image_id,
                annotation_description="Blood vessel segmentation",
                fill_holes=False,  # Don't fill holes for vessels
                raw_data_id=None,  # Mask file will be registered by processor
                expert_annotation_id=None,
                annotation_method="manual",
                provenance_chain_id=None,
                dataset_name=DATASET_NAME,
            )
            all_segmentations.append(segmentation)
            
            # One multi_class row: the image's single disease (mutually exclusive).
            # Export pivots task_name='fives_disease' into class_index/class_name columns.
            cls_annotations = await process_classification(
                class_value=disease_name,
                task_type="multi_class",
                task_name="fives_disease",
                class_name="fives_disease",
                image_id=image_id,
                class_labels=FIVES_DISEASE_LABELS,
                raw_data_id=excel_raw_file_id,
                expert_annotation_id=None,
                annotation_method="manual",
                provenance_chain_id=excel_chain_id,
            )
            all_classifications.extend(cls_annotations)
            
            # Process quality annotations (from Excel)
            quality_key = original_image_id
            if quality_key in quality_lookup:
                quality_data = quality_lookup[quality_key]
                
                # Use Excel provenance (same file)
                raw_file_id = excel_raw_file_id
                chain_id = excel_chain_id
                
                # Process contrast quality
                contrast_value = quality_data.get("contrast", 1)
                contrast_quality = await process_quality_annotation(
                    quality_type="contrast",
                    image_id=image_id,
                    quality_label="good" if contrast_value == 1 else "poor",
                    quality_score=contrast_value,
                    scale_description="FIVES Image Contrast (0=poor, 1=good)",
                    scale_min=0,
                    scale_max=1,
                    raw_data_id=raw_file_id,
                    expert_annotation_id=None,
                    provenance_chain_id=chain_id,
                )
                all_qualities.append(contrast_quality)
                
                # Process blur quality
                blur_value = quality_data.get("blur", 1)
                blur_quality = await process_quality_annotation(
                    quality_type="blur",
                    image_id=image_id,
                    quality_label="good" if blur_value == 1 else "poor",
                    quality_score=blur_value,
                    scale_description="FIVES Blur Quality (0=poor, 1=good)",
                    scale_min=0,
                    scale_max=1,
                    raw_data_id=raw_file_id,
                    expert_annotation_id=None,
                    provenance_chain_id=chain_id,
                )
                all_qualities.append(blur_quality)
                
                # Process illumination quality
                illumination_value = quality_data.get("illumination", 1)
                illumination_quality = await process_quality_annotation(
                    quality_type="illumination",
                    image_id=image_id,
                    quality_label="good" if illumination_value == 1 else "poor",
                    quality_score=illumination_value,
                    scale_description="FIVES Lighting Condition (0=poor, 1=good)",
                    scale_min=0,
                    scale_max=1,
                    raw_data_id=raw_file_id,
                    expert_annotation_id=None,
                    provenance_chain_id=chain_id,
                )
                all_qualities.append(illumination_quality)
            else:
                logger.warning(f"No quality data found for image: {quality_key}")
            
            tracker.update(success=True)
            tracker.record_success("image")
            
        except Exception as e:
            tracker.update(success=False)
            tracker.record_error(
                error_type="image_processing",
                error_message=str(e),
                item_id=image_path.stem,
                item_path=str(image_path),
            )
            logger.error(f"Failed to process image {image_path}: {e}")
    
    return all_images, all_segmentations, all_classifications, all_qualities, image_ids


async def ingest_fives() -> OperationStatistics:
    """
    Main ingestion function for FIVES dataset.
    
    Returns:
        OperationStatistics with success/error counts
    """
    data_root = get_data_root() / "13_FIVES"
    dataset_id = generate_dataset_uuid(DATASET_NAME)
    
    logger.info("=" * 80)
    logger.info(f"Starting FIVES dataset ingestion")
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
        task_types=["segmentation", "classification", "quality"],
        description=(
            "FIVES (Fundus Image Vessel Segmentation) dataset contains 800 high-resolution "
            "multi-disease color fundus photographs with pixelwise manual vessel annotations. "
            "The dataset includes quality assessment indicators (contrast, blur, lighting) "
            "and disease classification (AMD, DR, Glaucoma, Normal) derived from filenames."
        ),
    )
    await upsert_dataset(dataset)
    
    # Step 2: Setup progress tracker
    # Estimate: ~80 images + ~80 segmentations + ~80 classifications + ~240 qualities + ~160 Excel rows
    tracker = ProgressTracker(
        total=640,  # Rough estimate
        description=f"Ingesting {DATASET_NAME}"
    )
    
    # Step 3: Process quality Excel files (both sheets)
    logger.info("Processing quality assessment Excel file...")
    excel_path = data_root / "Quality Assessment.xlsx"
    
    if not excel_path.exists():
        logger.error(f"Quality Assessment Excel file not found: {excel_path}")
        raise FileNotFoundError(f"Quality Assessment Excel file not found: {excel_path}")
    
    # Process both sheets in parallel
    (quality_train, train_raw_id, train_chain_id), (quality_test, test_raw_id, test_chain_id) = await asyncio.gather(
        process_quality_excel(excel_path, dataset_id, "Train", tracker),
        process_quality_excel(excel_path, dataset_id, "Test", tracker),
    )
    
    logger.info(f"Quality lookup - Train: {len(quality_train)} entries, Test: {len(quality_test)} entries")
    
    # Step 4: Process train and test splits
    logger.info("Processing train split...")
    train_images, train_segmentations, train_classifications, train_qualities, train_image_ids = await process_split(
        data_root=data_root,
        dataset_id=dataset_id,
        split_name="train",
        quality_lookup=quality_train,
        excel_raw_file_id=train_raw_id,
        excel_chain_id=train_chain_id,
        tracker=tracker,
    )
    
    logger.info("Processing test split...")
    test_images, test_segmentations, test_classifications, test_qualities, test_image_ids = await process_split(
        data_root=data_root,
        dataset_id=dataset_id,
        split_name="test",
        quality_lookup=quality_test,
        excel_raw_file_id=test_raw_id,
        excel_chain_id=test_chain_id,
        tracker=tracker,
    )
    
    # Combine all data
    all_images = train_images + test_images
    all_segmentations = train_segmentations + test_segmentations
    all_classifications = train_classifications + test_classifications
    all_qualities = train_qualities + test_qualities
    
    logger.info(f"Total: {len(all_images)} images, {len(all_segmentations)} segmentations, "
                f"{len(all_classifications)} classifications, {len(all_qualities)} quality annotations")
    
    # Step 5: Bulk upsert images
    logger.info(f"Upserting {len(all_images)} images...")
    await bulk_upsert_images(all_images, batch_size=1000)
    
    # Step 6: Upsert segmentations (individual, no bulk operation)
    logger.info(f"Upserting {len(all_segmentations)} segmentation annotations...")
    for segmentation in all_segmentations:
        try:
            await upsert_segmentation_annotation(segmentation)
        except Exception as e:
            tracker.record_error(
                error_type="segmentation_upsert",
                error_message=str(e),
                item_id=str(segmentation.segmentation_id),
            )
            logger.error(f"Failed to upsert segmentation: {e}")
    
    # Step 7: Bulk upsert classifications and qualities in parallel
    logger.info(f"Upserting {len(all_classifications)} classification annotations...")
    logger.info(f"Upserting {len(all_qualities)} quality annotations...")
    await asyncio.gather(
        bulk_upsert_classification_annotations(all_classifications, batch_size=1000),
        bulk_upsert_quality_annotations(all_qualities, batch_size=1000),
    )
    
    # Step 8: Register splits and assign images
    logger.info("Registering dataset splits...")
    splits = await register_standard_splits(
        dataset_id=dataset_id,
        split_type="explicit",
        train_count=len(train_image_ids),
        test_count=len(test_image_ids),
    )
    
    logger.info(f"Assigning {len(train_image_ids)} images to train split...")
    logger.info(f"Assigning {len(test_image_ids)} images to test split...")
    await asyncio.gather(
        bulk_assign_images_to_split(train_image_ids, splits["train"]) if train_image_ids else asyncio.sleep(0),
        bulk_assign_images_to_split(test_image_ids, splits["test"]) if test_image_ids else asyncio.sleep(0),
    )
    
    tracker.finish()
    return tracker.get_statistics()


async def main():
    """Entry point for script execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    try:
        stats = await ingest_fives()
        
        logger.info("=" * 80)
        logger.info("Ingestion Summary:")
        logger.info(f"  Total items: {stats.total_items}")
        logger.info(f"  Successful: {stats.successful_items}")
        logger.info(f"  Failed: {stats.failed_items}")
        logger.info(f"  Skipped: {stats.skipped_items}")
        
        # Show breakdown by type
        if hasattr(stats, 'success_counts') and stats.success_counts:
            logger.info("")
            logger.info("  Breakdown by type:")
            for item_type, count in sorted(stats.success_counts.items()):
                logger.info(f"    {item_type}: {count}")
        
        if stats.errors:
            logger.warning(f"  Total errors: {len(stats.errors)}")
            # Group errors by type
            error_types = {}
            for error in stats.errors:
                error_type = error.get("error_type", "unknown")
                error_types[error_type] = error_types.get(error_type, 0) + 1
            
            for error_type, count in sorted(error_types.items()):
                logger.warning(f"    {error_type}: {count}")
        else:
            logger.info("  No errors encountered")
        
        logger.info("=" * 80)
        
        # Exit with error code if there were failures
        if stats.failed_items > 0:
            logger.error("Ingestion completed with errors")
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
