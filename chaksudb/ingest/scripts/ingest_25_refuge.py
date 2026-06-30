"""
Ingestion script for REFUGE dataset.

Dataset: REFUGE - Retinal Fundus Glaucoma Challenge dataset
Structure: JSON files with metadata, images, and segmentation masks
Annotations:
  - Classification: Glaucoma classification (0=normal, 1=glaucoma) from JSON Label field
  - Segmentation: Optic disc/cup segmentation masks (PNG format) in Masks/ folder
  - Localization: Fovea coordinates (Fovea_X, Fovea_Y) from JSON

Key Features:
  - Three splits: train, val, test
  - JSON files: train/index.json, val/index.json, test/index.json
  - Images in Images/ folder for each split
  - Masks in Masks/ folder for each split (PNG format)
  - Test set has no labels or fovea coordinates
"""

import asyncio
import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from uuid import UUID

from chaksudb.common.progress import ProgressTracker, OperationStatistics
from chaksudb.config.config import get_data_root
from chaksudb.db.models import (
    Dataset,
    Image,
    ClassificationAnnotation,
    SegmentationAnnotation,
    LocalizationAnnotation,
)
from chaksudb.db.queries import (
    upsert_dataset,
    bulk_upsert_images,
    bulk_upsert_classification_annotations,
    bulk_upsert_localization_annotations,
    upsert_segmentation_annotation,
)
from chaksudb.ingest.framework import (
    get_image_metadata_dict,
)
from chaksudb.ingest.framework.gen_uuid import (
    generate_dataset_uuid,
    generate_image_uuid,
    generate_localization_uuid,
)
from chaksudb.ingest.framework.ingestion_helpers import process_json
from chaksudb.ingest.framework.provenance_context import get_current_provenance
from chaksudb.ingest.framework.raw_file_helpers import register_individual_file
from chaksudb.ingest.framework.split_assigner import (
    register_standard_splits,
    bulk_assign_images_to_split,
)
from chaksudb.ingest.framework.task_processors.classification_processor import (
    process_classification,
)
from chaksudb.ingest.framework.task_processors.segmentation_processor import (
    process_segmentation_from_binary_mask,
    process_segmentation_from_multiclass_mask,
)
from chaksudb.ingest.framework.localization.helpers import normalize_keypoint_coordinates

logger = logging.getLogger(__name__)

# Dataset metadata
DATASET_NAME = "REFUGE"
DATASET_URL = "https://ieee-dataport.org/documents/refuge-retinal-fundus-glaucoma-challenge"
DATASET_LICENSE = "Research/Academic Use"  # Placeholder - update if known


async def process_json_entry(
    entry: Tuple[str, dict],
    idx: int,
    split_name: str,
    data_root: Path,
    dataset_id: UUID,
    tracker: ProgressTracker,
) -> Tuple[
    Optional[Image],
    List[ClassificationAnnotation],
    List[SegmentationAnnotation],
    List[LocalizationAnnotation],
    Optional[UUID],
]:
    """
    Process a single entry from REFUGE JSON file.
    
    Args:
        entry: Tuple of (key, value) from JSON object
        idx: Entry index
        split_name: Split name (train, val, test)
        data_root: Dataset root directory
        dataset_id: Dataset UUID
        tracker: Progress tracker
    
    Returns:
        Tuple of (Image, List[ClassificationAnnotation], List[SegmentationAnnotation], 
                 List[LocalizationAnnotation], image_id)
        or (None, [], [], [], None) on error
    """
    try:
        # Get provenance from context (set by process_json)
        json_raw_file_id, json_chain_id = get_current_provenance()
        
        # Extract entry data
        key, entry_data = entry
        
        # Extract image information from JSON
        img_name = entry_data.get("ImgName", "").strip()
        if not img_name:
            logger.warning(f"Entry {idx}: Missing ImgName")
            tracker.update(success=False)
            tracker.record_error(
                error_type="missing_image_name",
                error_message="Missing ImgName in JSON entry",
                item_id=f"entry_{idx}",
            )
            return None, [], [], [], None
        
        # Find image file
        split_dir = data_root / split_name
        image_path = split_dir / "Images" / img_name
        if not image_path.exists():
            logger.warning(f"Entry {idx}: Image not found: {image_path}")
            tracker.update(success=False)
            tracker.record_error(
                error_type="image_not_found",
                error_message=f"Image file not found: {image_path}",
                item_id=img_name,
                item_path=str(image_path),
            )
            return None, [], [], [], None
        
        # Generate image UUID
        original_image_id = Path(img_name).stem
        image_id = generate_image_uuid(dataset_id, f"{split_name}_{original_image_id}")
        
        # Create image model
        image = Image(
            image_id=image_id,
            dataset_id=dataset_id,
            original_image_id=original_image_id,
            **get_image_metadata_dict(image_path),
            modality="fundus",
            acquisition_date=None,
            image_quality=None,
        )
        
        # Process classification from Label field (only in train/val)
        classifications: List[ClassificationAnnotation] = []
        label = entry_data.get("Label")
        
        if label is not None:
            try:
                label_int = int(label)
                if label_int not in [0, 1]:
                    logger.warning(f"Entry {idx}: Invalid Label value: {label_int}")
                    tracker.record_error(
                        error_type="invalid_label",
                        error_message=f"Invalid Label value: {label_int}",
                        item_id=img_name,
                    )
                else:
                    # 0 = normal, 1 = glaucoma
                    classifications_list = await process_classification(
                        class_value=label_int,
                        task_type="binary",
                        class_name="glaucoma",
                        image_id=image_id,
                        class_labels={
                            0: "normal",
                            1: "glaucoma",
                        },
                        raw_data_id=json_raw_file_id,
                        provenance_chain_id=json_chain_id,
                        annotation_method="manual",
                    )
                    classifications.extend(classifications_list)
            except (ValueError, TypeError):
                logger.warning(f"Entry {idx}: Could not parse Label as integer: {label}")
                tracker.record_error(
                    error_type="invalid_label_format",
                    error_message=f"Could not parse Label: {label}",
                    item_id=img_name,
                )
        
        # Process fovea localization (only in train/val)
        localizations: List[LocalizationAnnotation] = []
        fovea_x = entry_data.get("Fovea_X")
        fovea_y = entry_data.get("Fovea_Y")
        
        if fovea_x is not None and fovea_y is not None:
            try:
                fovea_x_float = float(fovea_x)
                fovea_y_float = float(fovea_y)
                
                # Normalize coordinates
                coordinates = normalize_keypoint_coordinates(fovea_x_float, fovea_y_float)
                
                # Compute hash for deterministic UUID
                coords_hash = hashlib.sha256(
                    json.dumps(coordinates, sort_keys=True).encode()
                ).hexdigest()
                
                # Generate localization UUID
                localization_id = generate_localization_uuid(
                    image_id=image_id,
                    localization_type="keypoint",
                    target_structure="fovea",
                    raw_data_id=json_raw_file_id,
                    coordinates_hash=coords_hash,
                )
                
                # Create localization annotation
                localization = LocalizationAnnotation(
                    localization_id=localization_id,
                    image_id=image_id,
                    localization_type="keypoint",
                    target_structure="fovea",
                    coordinates=coordinates,
                    raw_data_id=json_raw_file_id,
                    annotation_method="manual",
                    provenance_chain_id=json_chain_id,
                )
                localizations.append(localization)
            except (ValueError, TypeError) as e:
                logger.warning(f"Entry {idx}: Could not parse fovea coordinates: {e}")
                tracker.record_error(
                    error_type="invalid_fovea_coordinates",
                    error_message=f"Could not parse fovea coordinates: {e}",
                    item_id=img_name,
                )
        
        # Process segmentation mask (if available)
        segmentations: List[SegmentationAnnotation] = []
        
        # Find mask file - try PNG first (Masks/ folder), then BMP (gts/ folder)
        mask_path = None
        mask_stem = Path(img_name).stem
        
        # Try Masks/ folder first (PNG format)
        mask_path_png = split_dir / "Masks" / f"{mask_stem}.png"
        if mask_path_png.exists():
            mask_path = mask_path_png
        else:
            # Try gts/ folder (BMP format)
            mask_path_bmp = split_dir / "gts" / f"{mask_stem}.bmp"
            if mask_path_bmp.exists():
                mask_path = mask_path_bmp
        
        if mask_path and mask_path.exists():
            # Register mask file for provenance
            mask_raw_file_id, mask_chain_id = await register_individual_file(
                file_path=mask_path,
                dataset_id=dataset_id,
                unified_annotation_type="segmentation",
                file_type=None,  # ✅ Binary mask files should have NULL file_type
                auto_detect_type=False,
            )
            
            # Process mask - REFUGE masks contain disc and cup segmentation
            # The mask typically has multiple classes: background (0), disc (1), cup (2)
            # We'll extract disc and cup separately using multiclass mask processor
            try:
                # Try multiclass mask processing first (if mask contains both disc and cup)
                multiclass_segmentations = await process_segmentation_from_multiclass_mask(
                    mask_path=mask_path,
                    class_names={1: "optic_disc", 2: "optic_cup"},
                    image_id=image_id,
                    classes_to_extract=[1, 2],  # Extract both disc and cup
                    fill_holes=False,  # Don't fill holes for disc/cup boundaries
                    raw_data_id=mask_raw_file_id,
                    annotation_method="manual",
                    provenance_chain_id=mask_chain_id,
                    dataset_name=DATASET_NAME,
                )
                segmentations.extend(multiclass_segmentations)
            except Exception as multiclass_error:
                # Fallback: Try binary mask processing if multiclass fails
                # This handles cases where masks might be separate binary masks
                logger.debug(f"Multiclass processing failed, trying binary: {multiclass_error}")
                try:
                    # Try to extract disc (class 1)
                    disc_segmentation = await process_segmentation_from_binary_mask(
                        mask_path=mask_path,
                        annotation_type="optic_disc",
                        image_id=image_id,
                        annotation_description="Optic disc segmentation from REFUGE",
                        extract_class=1,
                        merge_nonzero=False,
                        fill_holes=False,
                        raw_data_id=mask_raw_file_id,
                        annotation_method="manual",
                        provenance_chain_id=mask_chain_id,
                        dataset_name=DATASET_NAME,
                    )
                    segmentations.append(disc_segmentation)
                except Exception as e:
                    logger.warning(f"Failed to process disc segmentation: {e}")
            except Exception as e:
                logger.warning(f"Entry {idx}: Failed to process segmentation mask: {e}")
                tracker.record_error(
                    error_type="segmentation_processing_error",
                    error_message=str(e),
                    item_id=img_name,
                    item_path=str(mask_path),
                )
        else:
            # Log missing mask but don't fail - not all images may have masks
            logger.debug(f"Entry {idx}: Mask not found for {img_name}")
        
        # Consider entry successful if we have at least the image
        # Classification, localization, and segmentation are optional
        tracker.update(success=True)
        tracker.record_success("image")
        if classifications:
            tracker.record_success("classification")
        if localizations:
            tracker.record_success("localization")
        for _ in segmentations:
            tracker.record_success("segmentation")
        
        return image, classifications, segmentations, localizations, image_id
        
    except Exception as e:
        logger.error(f"Entry {idx}: Error processing entry: {e}", exc_info=True)
        tracker.update(success=False)
        tracker.record_error(
            error_type="entry_processing_error",
            error_message=str(e),
            item_id=f"entry_{idx}",
        )
        return None, [], [], [], None


async def process_split(
    split_name: str,
    data_root: Path,
    dataset_id: UUID,
    tracker: ProgressTracker,
) -> Tuple[
    List[Image],
    List[ClassificationAnnotation],
    List[SegmentationAnnotation],
    List[LocalizationAnnotation],
    List[UUID],
]:
    """
    Process a single split (train, val, or test).
    
    Args:
        split_name: Split name (train, val, test)
        data_root: Dataset root directory
        dataset_id: Dataset UUID
        tracker: Progress tracker
    
    Returns:
        Tuple of (images, classifications, segmentations, localizations, image_ids)
    """
    split_dir = data_root / split_name
    json_path = split_dir / "index.json"
    
    if not json_path.exists():
        logger.warning(f"JSON file not found for split {split_name}: {json_path}")
        return [], [], [], [], []
    
    # Collect results
    all_images: List[Image] = []
    all_classifications: List[ClassificationAnnotation] = []
    all_segmentations: List[SegmentationAnnotation] = []
    all_localizations: List[LocalizationAnnotation] = []
    all_image_ids: List[UUID] = []
    
    async def process_entry(entry: Tuple[str, dict], idx: int) -> None:
        """Wrapper to process JSON entry and collect results."""
        image, classifications, segmentations, localizations, image_id = (
            await process_json_entry(
                entry=entry,
                idx=idx,
                split_name=split_name,
                data_root=data_root,
                dataset_id=dataset_id,
                tracker=tracker,
            )
        )
        if image:
            all_images.append(image)
            if image_id:
                all_image_ids.append(image_id)
        all_classifications.extend(classifications)
        all_segmentations.extend(segmentations)
        all_localizations.extend(localizations)
    
    # Process JSON with automatic provenance
    stats, json_raw_file_id, json_chain_id = await process_json(
        json_path=json_path,
        dataset_id=dataset_id,
        unified_annotation_type="classification",  # Primary task
        process_entry_fn=process_entry,
        progress_tracker=tracker,
        skip_errors=True,
    )
    
    logger.info(
        f"Processed {split_name} split: {stats.successful_items} successful, "
        f"{stats.failed_items} failed"
    )
    logger.info(
        f"Collected {len(all_images)} images, {len(all_classifications)} classifications, "
        f"{len(all_segmentations)} segmentations, {len(all_localizations)} localizations"
    )
    
    return all_images, all_classifications, all_segmentations, all_localizations, all_image_ids


async def ingest_refuge() -> OperationStatistics:
    """
    Main ingestion function for REFUGE dataset.
    
    Returns:
        OperationStatistics with success/error counts
    """
    data_root = get_data_root() / "25_REFUGE"
    dataset_id = generate_dataset_uuid(DATASET_NAME)
    
    logger.info("=" * 80)
    logger.info(f"Starting {DATASET_NAME} dataset ingestion")
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
        task_types=["classification", "segmentation", "localization"],
        description=(
            "REFUGE (Retinal Fundus Glaucoma Challenge) dataset contains fundus images "
            "with glaucoma classification, optic disc/cup segmentation, and fovea "
            "localization annotations. The dataset includes train, validation, and test splits."
        ),
    )
    await upsert_dataset(dataset)
    
    # Step 2: Count items first (lightweight - just read JSON structure)
    splits = ["train", "val", "test"]
    total_count = 0
    
    for split_name in splits:
        split_dir = data_root / split_name
        json_file = split_dir / "index.json"
        if json_file.exists():
            with open(json_file, "r") as f:
                data = json.load(f)
                count = len(data) if isinstance(data, dict) else 0
                total_count += count
                logger.info(f"  {split_name}: {count} entries")
    
    logger.info(f"Total items to process: {total_count}")
    
    # Step 3: Setup progress tracker with actual count
    tracker = ProgressTracker(
        total=total_count,
        description=f"Ingesting {DATASET_NAME}"
    )
    
    # Step 4: Process each split
    all_images: List[Image] = []
    all_classifications: List[ClassificationAnnotation] = []
    all_segmentations: List[SegmentationAnnotation] = []
    all_localizations: List[LocalizationAnnotation] = []
    split_image_ids: Dict[str, List[UUID]] = {}
    
    for split_name in splits:
        logger.info("=" * 80)
        logger.info(f"Processing {split_name} split...")
        logger.info("=" * 80)
        
        images, classifications, segmentations, localizations, image_ids = (
            await process_split(
                split_name=split_name,
                data_root=data_root,
                dataset_id=dataset_id,
                tracker=tracker,
            )
        )
        
        all_images.extend(images)
        all_classifications.extend(classifications)
        all_segmentations.extend(segmentations)
        all_localizations.extend(localizations)
        split_image_ids[split_name] = image_ids
    
    # Step 5: Bulk upsert images
    logger.info(f"Upserting {len(all_images)} images...")
    if all_images:
        await bulk_upsert_images(all_images, batch_size=1000)
    
    # Step 6: Bulk upsert classifications
    logger.info(f"Upserting {len(all_classifications)} classification annotations...")
    if all_classifications:
        await bulk_upsert_classification_annotations(all_classifications, batch_size=1000)
    
    # Step 7: Bulk upsert localizations
    logger.info(f"Upserting {len(all_localizations)} localization annotations...")
    if all_localizations:
        await bulk_upsert_localization_annotations(all_localizations, batch_size=1000)
    
    # Step 8: Upsert segmentations (individual, no bulk operation available)
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
    
    # Step 9: Register splits and assign images
    logger.info("Registering dataset splits...")
    registered_splits = await register_standard_splits(
        dataset_id=dataset_id,
        split_type="explicit",
        train_count=len(split_image_ids.get("train", [])),
        val_count=len(split_image_ids.get("val", [])),
        test_count=len(split_image_ids.get("test", [])),
    )
    
    # Assign images to splits
    for split_name, image_ids in split_image_ids.items():
        if image_ids and split_name in registered_splits:
            logger.info(f"Assigning {len(image_ids)} images to {split_name} split...")
            await bulk_assign_images_to_split(image_ids, registered_splits[split_name])
    
    tracker.finish()
    return tracker.get_statistics()


async def main():
    """Entry point for script execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    try:
        stats = await ingest_refuge()
        
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
                error_type = error.get("error_type", "unknown") if isinstance(error, dict) else getattr(error, "error_type", "unknown")
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
