"""
Ingestion script for IDRID (Indian Diabetic Retinopathy Image Dataset).

Dataset: IDRID - Multi-task dataset with segmentation, grading, and localization
Structure: 3 sub-tasks with different image sets
- Task A (Segmentation): 7 images with 5 lesion types
- Task B (Disease Grading): 51 images with DR + DME grading
- Task C (Localization): Same 51 images as Task B with OD + Fovea centers

Annotations:
- Disease grading (DR 0-4, DME 0-2)
- Localization (OD center, Fovea center)
- Segmentation (Microaneurysms, Haemorrhages, Hard Exudates, Soft Exudates, Optic Disc)

Key Implementation Notes:
- Tasks B and C share the same images - register images ONCE
- Task A has a different set of images
- Comprehensive error handling with rollback at bulk operation level
- Parallelize CSV processing and mask conversion

Transaction Handling:
- Each bulk operation (bulk_upsert_images, bulk_upsert_disease_gradings) is atomic
- If any bulk operation fails, it will rollback that specific operation
- The script uses try/except to catch errors and provide detailed logging
- Failed items are tracked in the progress tracker for review

Note: For full transactional support where ALL operations succeed or ALL rollback
atomically, the query functions would need to accept optional connection parameters.
This is a future enhancement. Current behavior provides operation-level atomicity
which is suitable for most ingestion scenarios.
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set
from uuid import UUID

from chaksudb.common.progress import ProgressTracker, OperationStatistics
from chaksudb.config.config import get_data_root
from chaksudb.db.models import (
    Dataset,
    Image,
    DiseaseGrading,
    LocalizationAnnotation,
    SegmentationAnnotation,
)
from chaksudb.db.queries import (
    upsert_dataset,
    bulk_upsert_images,
    bulk_upsert_disease_gradings,
    upsert_localization_annotation,
    upsert_segmentation_annotation,
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
    generate_localization_uuid,
)
from chaksudb.ingest.framework.task_processors.grading_processor import process_disease_grade
from chaksudb.ingest.framework.task_processors.segmentation_processor import (
    process_segmentation_from_binary_mask,
)
from chaksudb.ingest.framework.split_assigner import (
    register_standard_splits,
    bulk_assign_images_to_split,
)
from chaksudb.ingest.framework.raw_file_helpers import register_individual_file
from chaksudb.ingest.framework.provenance_context import get_current_provenance

logger = logging.getLogger(__name__)

# Dataset metadata
DATASET_NAME = "IDRID"
DATASET_URL = "https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid"
DATASET_LICENSE = "CC BY 4.0"

# Lesion type mappings for segmentation
LESION_TYPE_MAP = {
    "1. Microaneurysms": {"annotation_type": "lesions", "lesion_subtype": "MA", "suffix": "_MA"},
    "2. Haemorrhages": {"annotation_type": "lesions", "lesion_subtype": "HE", "suffix": "_HE"},
    "3. Hard Exudates": {"annotation_type": "lesions", "lesion_subtype": "EX", "suffix": "_EX"},
    "4. Soft Exudates": {"annotation_type": "lesions", "lesion_subtype": "SE", "suffix": "_SE"},
    "5. Optic Disc": {"annotation_type": "optic_disc", "lesion_subtype": None, "suffix": "_OD"},
}


async def ingest_disease_grading_and_localization(
    data_root: Path,
    dataset_id: UUID,
    tracker: ProgressTracker,
) -> tuple[List[Image], List[DiseaseGrading], List[LocalizationAnnotation], Dict[UUID, str]]:
    """
    Phase 1: Ingest Task B (Disease Grading) and Task C (Localization).
    
    These tasks share the same 51 images (41 training, 10 testing).
    
    Returns:
        Tuple of (images, gradings, localizations, image_to_split_map)
    """
    logger.info("=" * 80)
    logger.info("Phase 1: Processing Disease Grading + Localization (51 images)")
    logger.info("=" * 80)
    
    grading_root = data_root / "B. Disease Grading"
    localization_root = data_root / "C. Localization"
    
    # Discover all grading images
    logger.info("Discovering images for grading/localization...")
    grading_train_images = list((grading_root / "1. Original Images" / "a. Training Set").glob("*.jpg"))
    grading_test_images = list((grading_root / "1. Original Images" / "b. Testing Set").glob("*.jpg"))
    
    logger.info(f"Found {len(grading_train_images)} training images")
    logger.info(f"Found {len(grading_test_images)} testing images")
    
    # Collections for bulk upsert
    all_images: List[Image] = []
    all_gradings: List[DiseaseGrading] = []
    all_localizations: List[LocalizationAnnotation] = []
    image_to_split: Dict[UUID, str] = {}
    processed_images: Set[str] = set()
    
    # First pass: Process disease grading CSVs
    logger.info("Processing disease grading CSVs...")
    
    async def process_grading_row(row, idx, split_name: str):
        """Process a single grading CSV row."""
        try:
            # Handle column name variations (some have trailing spaces)
            image_col = next((k for k in row.keys() if "image" in k.lower() and "name" in k.lower()), "Image name")
            image_name = row[image_col].strip()
            
            # Skip if already processed
            if image_name in processed_images:
                tracker.update(success=True)
                return
            
            processed_images.add(image_name)
            image_id = generate_image_uuid(dataset_id, image_name)
            
            # Find image file
            if split_name == "train":
                image_dir = grading_root / "1. Original Images" / "a. Training Set"
            else:
                image_dir = grading_root / "1. Original Images" / "b. Testing Set"
            
            image_path = image_dir / f"{image_name}.jpg"
            
            if not image_path.exists():
                tracker.record_error(
                    error_type="file_not_found",
                    error_message=f"Image not found: {image_name}.jpg",
                    item_id=image_name,
                )
                tracker.update(success=False)
                return
            
            # Create image model
            image = Image(
                image_id=image_id,
                dataset_id=dataset_id,
                original_image_id=image_name,
                **get_image_metadata_dict(image_path),
                modality="fundus",
            )
            all_images.append(image)
            image_to_split[image_id] = split_name
            
            # Process DR grading (0-4)
            # IDRID uses standard ICDR scale (same as EYEPACS)
            dr_col_name = next((k for k in row.keys() if "retinopathy" in k.lower() and "grade" in k.lower()), "Retinopathy grade")
            dr_grade_value = int(row[dr_col_name])
            dr_grading = await process_disease_grade(
                grade_value=dr_grade_value,
                disease_type="DR",
                scale_name="ICDR_0_4",  # Standard International Clinical DR scale
                image_id=image_id,
                annotation_method="manual",
            )
            all_gradings.append(dr_grading)
            
            # Process DME grading (0-2)
            # Note: CSV column has trailing space in some files
            dme_col_name = next((k for k in row.keys() if "risk of macular edema" in k.lower()), None)
            if not dme_col_name:
                raise KeyError("Could not find DME column (expected 'Risk of macular edema')")
            
            dme_grade_value = int(row[dme_col_name].strip())
            dme_grading = await process_disease_grade(
                grade_value=dme_grade_value,
                disease_type="DME",
                scale_name="IDRID_DME_0_2",
                scale_description="IDRID DME risk (0-2)",
                min_value=0,
                max_value=2,
                value_labels={
                    "0": "No DME",
                    "1": "Mild/Moderate DME",
                    "2": "Severe DME",
                },
                image_id=image_id,
                annotation_method="manual",
            )
            all_gradings.append(dme_grading)
            
            tracker.update(success=True)
            tracker.record_success("image_with_grading")
            
        except Exception as e:
            tracker.update(success=False)
            tracker.record_error(
                error_type="processing",
                error_message=str(e),
                item_id=row.get("Image name", "unknown"),
            )
            logger.exception(f"Failed to process grading row {idx}: {e}")
    
    # Process training and testing grading CSVs
    train_grading_csv = grading_root / "2. Groundtruths" / "a. IDRiD_Disease Grading_Training Labels.csv"
    test_grading_csv = grading_root / "2. Groundtruths" / "b. IDRiD_Disease Grading_Testing Labels.csv"
    
    train_handler = lambda row, idx: process_grading_row(row, idx, "train")
    test_handler = lambda row, idx: process_grading_row(row, idx, "test")
    
    # Process both CSVs in parallel
    (train_stats, _, _), (test_stats, _, _) = await asyncio.gather(
        process_csv(
            csv_path=train_grading_csv,
            dataset_id=dataset_id,
            unified_annotation_type="grading",
            process_row_fn=train_handler,
            progress_tracker=tracker,
            skip_errors=True,
        ),
        process_csv(
            csv_path=test_grading_csv,
            dataset_id=dataset_id,
            unified_annotation_type="grading",
            process_row_fn=test_handler,
            progress_tracker=tracker,
            skip_errors=True,
        ),
    )
    
    # Second pass: Process localization CSVs
    logger.info("Processing localization CSVs...")
    
    async def process_localization_row(row, idx, structure_type: str, split_name: str):
        """Process a single localization CSV row."""
        try:
            # Handle column name variations (some have trailing spaces)
            image_col = next((k for k in row.keys() if "image" in k.lower() and "no" in k.lower()), "Image No")
            x_col = next((k for k in row.keys() if "x" in k.lower() and "coordinate" in k.lower()), "X- Coordinate")
            y_col = next((k for k in row.keys() if "y" in k.lower() and "coordinate" in k.lower()), "Y - Coordinate")
            
            # Safe get + strip (handles trailing commas, missing keys, and rows that are just commas)
            image_name = (row.get(image_col) or "").strip()
            x_val = (row.get(x_col) or "").strip()
            y_val = (row.get(y_col) or "").strip()
            
            # Skip empty rows (just commas), rows with missing image id, or missing coordinates
            if not image_name or not x_val or not y_val:
                logger.debug(
                    "Skipping localization row %s (%s): empty row or missing required field (image=%r, x=%r, y=%r)",
                    idx, structure_type, image_name or "(empty)", x_val or "(empty)", y_val or "(empty)",
                )
                tracker.update(success=True)
                return
            
            image_id = generate_image_uuid(dataset_id, image_name)
            
            # Extract coordinates
            x_coord = float(x_val)
            y_coord = float(y_val)
            
            # Get provenance from context
            raw_data_id, provenance_chain_id = get_current_provenance()
            
            # Create localization annotation
            coordinates = {"x": x_coord, "y": y_coord}
            
            # Compute deterministic hash for UUID
            import hashlib
            import json
            coordinates_hash = hashlib.sha256(
                json.dumps(coordinates, sort_keys=True).encode()
            ).hexdigest()
            
            localization_id = generate_localization_uuid(
                image_id=image_id,
                localization_type="center_point",
                target_structure=structure_type,
                raw_data_id=raw_data_id,
                coordinates_hash=coordinates_hash,
            )
            
            localization = LocalizationAnnotation(
                localization_id=localization_id,
                image_id=image_id,
                localization_type="center_point",
                target_structure=structure_type,
                coordinates=coordinates,
                raw_data_id=raw_data_id,
                annotation_method="manual",
                provenance_chain_id=provenance_chain_id,
            )
            all_localizations.append(localization)
            
            tracker.update(success=True)
            tracker.record_success(f"localization_{structure_type}")
            
        except Exception as e:
            tracker.update(success=False)
            tracker.record_error(
                error_type="processing",
                error_message=str(e),
                item_id=row.get("Image No", "unknown"),
            )
            logger.exception(f"Failed to process localization row {idx}: {e}")
    
    # Process all 4 localization CSVs in parallel
    od_train_csv = localization_root / "2. Groundtruths" / "1. Optic Disc Center Location" / "a. IDRiD_OD_Center_Training Set_Markups.csv"
    od_test_csv = localization_root / "2. Groundtruths" / "1. Optic Disc Center Location" / "b. IDRiD_OD_Center_Testing Set_Markups.csv"
    fovea_train_csv = localization_root / "2. Groundtruths" / "2. Fovea Center Location" / "IDRiD_Fovea_Center_Training Set_Markups.csv"
    fovea_test_csv = localization_root / "2. Groundtruths" / "2. Fovea Center Location" / "IDRiD_Fovea_Center_Testing Set_Markups.csv"
    
    od_train_handler = lambda row, idx: process_localization_row(row, idx, "optic_disc", "train")
    od_test_handler = lambda row, idx: process_localization_row(row, idx, "optic_disc", "test")
    fovea_train_handler = lambda row, idx: process_localization_row(row, idx, "fovea", "train")
    fovea_test_handler = lambda row, idx: process_localization_row(row, idx, "fovea", "test")
    
    await asyncio.gather(
        process_csv(
            csv_path=od_train_csv,
            dataset_id=dataset_id,
            unified_annotation_type="localization",
            process_row_fn=od_train_handler,
            progress_tracker=tracker,
            skip_errors=True,
        ),
        process_csv(
            csv_path=od_test_csv,
            dataset_id=dataset_id,
            unified_annotation_type="localization",
            process_row_fn=od_test_handler,
            progress_tracker=tracker,
            skip_errors=True,
        ),
        process_csv(
            csv_path=fovea_train_csv,
            dataset_id=dataset_id,
            unified_annotation_type="localization",
            process_row_fn=fovea_train_handler,
            progress_tracker=tracker,
            skip_errors=True,
        ),
        process_csv(
            csv_path=fovea_test_csv,
            dataset_id=dataset_id,
            unified_annotation_type="localization",
            process_row_fn=fovea_test_handler,
            progress_tracker=tracker,
            skip_errors=True,
        ),
    )
    
    logger.info(f"Collected {len(all_images)} images")
    logger.info(f"Collected {len(all_gradings)} gradings (DR + DME)")
    logger.info(f"Collected {len(all_localizations)} localizations (OD + Fovea)")
    
    return all_images, all_gradings, all_localizations, image_to_split


async def ingest_segmentation(
    data_root: Path,
    dataset_id: UUID,
    tracker: ProgressTracker,
) -> tuple[List[Image], List[SegmentationAnnotation], Dict[UUID, str]]:
    """
    Phase 2: Ingest Task A (Segmentation).
    
    This task has a different set of 7 images (5 training, 2 testing).
    Each image has up to 5 lesion segmentation masks.
    
    Returns:
        Tuple of (images, segmentations, image_to_split_map)
    """
    logger.info("=" * 80)
    logger.info("Phase 2: Processing Segmentation (7 images, 5 lesion types)")
    logger.info("=" * 80)
    
    seg_root = data_root / "A. Segmentation"
    
    # Discover all segmentation images
    logger.info("Discovering images for segmentation...")
    seg_train_images = list((seg_root / "1. Original Images" / "a. Training Set").glob("*.jpg"))
    seg_test_images = list((seg_root / "1. Original Images" / "b. Testing Set").glob("*.jpg"))
    
    logger.info(f"Found {len(seg_train_images)} training images")
    logger.info(f"Found {len(seg_test_images)} testing images")
    
    # Collections for bulk upsert
    all_images: List[Image] = []
    all_segmentations: List[SegmentationAnnotation] = []
    image_to_split: Dict[UUID, str] = {}
    
    # Process each split
    for split_name, split_images in [("train", seg_train_images), ("test", seg_test_images)]:
        split_label = "a. Training Set" if split_name == "train" else "b. Testing Set"
        
        for image_path in split_images:
            try:
                image_name = image_path.stem  # e.g., "IDRiD_02"
                image_id = generate_image_uuid(dataset_id, image_name)
                
                # Register image
                image = Image(
                    image_id=image_id,
                    dataset_id=dataset_id,
                    original_image_id=image_name,
                    **get_image_metadata_dict(image_path),
                    modality="fundus",
                )
                all_images.append(image)
                image_to_split[image_id] = split_name
                
                tracker.update(success=True)
                tracker.record_success("segmentation_image")
                
                # Process each lesion type
                masks_root = seg_root / "2. All Segmentation Groundtruths" / split_label
                
                for lesion_folder, lesion_info in LESION_TYPE_MAP.items():
                    annotation_type = lesion_info["annotation_type"]
                    lesion_subtype = lesion_info["lesion_subtype"]
                    suffix = lesion_info["suffix"]
                    
                    # Find mask file
                    mask_path = masks_root / lesion_folder / f"{image_name}{suffix}.tif"
                    
                    if not mask_path.exists():
                        # Some images don't have all lesion types (e.g., Soft Exudates)
                        logger.debug(f"Mask not found: {mask_path} (skipping)")
                        continue
                    
                    # Register individual mask file for provenance tracking
                    # Note: file_type=NULL for binary mask images (not structured annotation files)
                    raw_file_id, chain_id = await register_individual_file(
                        file_path=mask_path,
                        dataset_id=dataset_id,
                        unified_annotation_type="segmentation",
                        file_type=None,  # Will be NULL in database
                        auto_detect_type=False,  # Don't auto-detect 'tif' extension
                    )
                    
                    # Process segmentation mask (will standardize to PNG in processed/)
                    segmentation = await process_segmentation_from_binary_mask(
                        mask_path=mask_path,
                        annotation_type=annotation_type,
                        image_id=image_id,
                        lesion_subtype=lesion_subtype,
                        raw_data_id=raw_file_id,
                        provenance_chain_id=chain_id,
                        annotation_method="manual",
                        dataset_name=DATASET_NAME,
                    )
                    
                    all_segmentations.append(segmentation)
                    
                    tracker.update(success=True)
                    tracker.record_success(f"segmentation_{annotation_type}")
                
            except Exception as e:
                tracker.update(success=False)
                tracker.record_error(
                    error_type="processing",
                    error_message=str(e),
                    item_id=image_path.stem,
                    item_path=str(image_path),
                )
                logger.exception(f"Failed to process segmentation image {image_path.stem}: {e}")
    
    logger.info(f"Collected {len(all_images)} images")
    logger.info(f"Collected {len(all_segmentations)} segmentations")
    
    return all_images, all_segmentations, image_to_split


async def ingest_idrid() -> OperationStatistics:
    """
    Main ingestion function for IDRID dataset.
    
    Returns:
        OperationStatistics with success/error counts
    """
    data_root = get_data_root() / "03_IDRID"
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
        description="Indian Diabetic Retinopathy Image Dataset with segmentation, grading, and localization tasks",
    )
    await upsert_dataset(dataset)
    
    # Step 2: Setup progress tracker
    # Estimate total items:
    # - Phase 1: 51 images + 102 gradings + 204 localizations (4 CSVs) = 357 items
    # - Phase 2: 7 images + ~35 segmentations (5 types × 7 images) = 42 items
    total_estimate = 400
    
    tracker = ProgressTracker(
        total=total_estimate,
        description=f"Ingesting {DATASET_NAME}",
    )
    
    # Step 3: Phase 1 - Disease Grading + Localization (same images)
    grading_images, gradings, localizations, grading_image_to_split = await ingest_disease_grading_and_localization(
        data_root=data_root,
        dataset_id=dataset_id,
        tracker=tracker,
    )
    
    # Step 4: Phase 2 - Segmentation (different images)
    seg_images, segmentations, seg_image_to_split = await ingest_segmentation(
        data_root=data_root,
        dataset_id=dataset_id,
        tracker=tracker,
    )
    
    # Step 5: Bulk upsert images from both phases
    logger.info("=" * 80)
    logger.info("Upserting data to database...")
    logger.info("=" * 80)
    
    try:
        all_images = grading_images + seg_images
        logger.info(f"Upserting {len(all_images)} images...")
        await bulk_upsert_images(all_images, batch_size=1000)
        logger.info("✓ Images upserted successfully")
        
    except Exception as e:
        logger.error(f"Failed to upsert images: {e}")
        tracker.record_error(
            error_type="bulk_upsert_images",
            error_message=str(e),
            item_id="bulk_images",
        )
        raise RuntimeError(f"Image upsert failed, aborting ingestion: {e}") from e
    
    # Step 6: Bulk upsert gradings
    try:
        logger.info(f"Upserting {len(gradings)} gradings...")
        await bulk_upsert_disease_gradings(gradings, batch_size=1000)
        logger.info("✓ Gradings upserted successfully")
        
    except Exception as e:
        logger.error(f"Failed to upsert gradings: {e}")
        tracker.record_error(
            error_type="bulk_upsert_gradings",
            error_message=str(e),
            item_id="bulk_gradings",
        )
        raise RuntimeError(f"Grading upsert failed, aborting ingestion: {e}") from e
    
    # Step 7: Upsert localizations (no bulk operation yet)
    try:
        logger.info(f"Upserting {len(localizations)} localizations...")
        localization_errors = 0
        for idx, localization in enumerate(localizations):
            try:
                await upsert_localization_annotation(localization)
            except Exception as e:
                localization_errors += 1
                logger.warning(f"Failed to upsert localization {idx}: {e}")
                tracker.record_error(
                    error_type="upsert_localization",
                    error_message=str(e),
                    item_id=str(localization.image_id),
                )
        
        if localization_errors == 0:
            logger.info("✓ All localizations upserted successfully")
        else:
            logger.warning(f"⚠ {localization_errors}/{len(localizations)} localizations failed")
            
    except Exception as e:
        logger.error(f"Critical error during localization upsert: {e}")
        raise RuntimeError(f"Localization upsert failed: {e}") from e
    
    # Step 8: Upsert segmentations (no bulk operation yet)
    try:
        logger.info(f"Upserting {len(segmentations)} segmentations...")
        segmentation_errors = 0
        for idx, segmentation in enumerate(segmentations):
            try:
                await upsert_segmentation_annotation(segmentation)
            except Exception as e:
                segmentation_errors += 1
                logger.warning(f"Failed to upsert segmentation {idx}: {e}")
                tracker.record_error(
                    error_type="upsert_segmentation",
                    error_message=str(e),
                    item_id=str(segmentation.image_id),
                )
        
        if segmentation_errors == 0:
            logger.info("✓ All segmentations upserted successfully")
        else:
            logger.warning(f"⚠ {segmentation_errors}/{len(segmentations)} segmentations failed")
            
    except Exception as e:
        logger.error(f"Critical error during segmentation upsert: {e}")
        raise RuntimeError(f"Segmentation upsert failed: {e}") from e
    
    # Step 9: Register splits and assign images
    try:
        logger.info("Registering dataset splits...")
        # Calculate split counts
        grading_train_count = sum(1 for split in grading_image_to_split.values() if split == "train")
        grading_test_count = sum(1 for split in grading_image_to_split.values() if split == "test")
        seg_train_count = sum(1 for split in seg_image_to_split.values() if split == "train")
        seg_test_count = sum(1 for split in seg_image_to_split.values() if split == "test")
        
        total_train = grading_train_count + seg_train_count
        total_test = grading_test_count + seg_test_count
        
        splits = await register_standard_splits(
            dataset_id=dataset_id,
            split_type="explicit",
            train_count=total_train,
            test_count=total_test,
        )
        
        train_split = splits["train"]
        test_split = splits["test"]
        
        # Assign grading/localization images to splits
        grading_train_images = [img_id for img_id, split in grading_image_to_split.items() if split == "train"]
        grading_test_images = [img_id for img_id, split in grading_image_to_split.items() if split == "test"]
        
        # Assign segmentation images to splits
        seg_train_images = [img_id for img_id, split in seg_image_to_split.items() if split == "train"]
        seg_test_images = [img_id for img_id, split in seg_image_to_split.items() if split == "test"]
        
        await asyncio.gather(
            bulk_assign_images_to_split(grading_train_images + seg_train_images, train_split),
            bulk_assign_images_to_split(grading_test_images + seg_test_images, test_split),
        )
        logger.info("✓ Dataset splits registered and assigned successfully")
        
    except Exception as e:
        logger.error(f"Failed to register splits: {e}")
        tracker.record_error(
            error_type="register_splits",
            error_message=str(e),
            item_id="splits",
        )
        raise RuntimeError(f"Split registration failed: {e}") from e
    
    tracker.finish()
    logger.info("=" * 80)
    logger.info("✓ IDRID ingestion completed successfully")
    logger.info("=" * 80)
    
    return tracker.get_statistics()


async def main():
    """Entry point for script execution."""
    import sys
    from pathlib import Path
    log_file = Path("./logs/ingest_03_idrid.log")
    log_file.touch(exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
        logging.FileHandler(log_file, mode='w'), 
        logging.StreamHandler(sys.stdout),          
        ],
    )
    
    try:
        stats = await ingest_idrid()
        
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
        
    except Exception as e:
        logger.exception(f"Fatal error during ingestion: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
