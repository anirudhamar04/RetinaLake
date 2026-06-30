"""
Ingestion script for ORIGA dataset.

Dataset: ORIGA - Glaucoma screening dataset with optic disc/cup segmentation
Structure: CSV files + images + masks + semi-automatic annotations
Annotations:
  - Classification: Glaucoma (binary) from OrigaList.csv with CDR, Ecc-Cup, Ecc-Disc metadata
  - Segmentation: Optic disc and cup from Masks/ folder (manual) and .mat files (pseudo)

Key Features:
  - OrigaList.csv: Eye laterality, ExpCDR, Set (split), Glaucoma (0/1)
  - origa_info.csv: CDR, Ecc-Cup, Ecc-Disc, Label (redundant with Glaucoma)
  - Masks/: Multi-class masks (0=background, 1=OD, 2=cup) - manual annotations
  - Semi-automatic-annotations/: .mat files with same structure - pseudo annotations
  - Images/: Original fundus images
  - Only process original images/masks (not cropped/square variants)
"""

import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from uuid import UUID

import cv2
import numpy as np
import scipy.io

from chaksudb.common.progress import ProgressTracker, OperationStatistics
from chaksudb.config.config import get_data_root
from chaksudb.db.models import (
    Dataset,
    Image,
    SegmentationAnnotation,
    ClassificationAnnotation,
)
from chaksudb.db.queries import (
    upsert_dataset,
    bulk_upsert_images,
    bulk_upsert_classification_annotations,
    upsert_segmentation_annotation,
)
from chaksudb.ingest.framework import (
    get_image_metadata_dict,
)
from chaksudb.ingest.framework.gen_uuid import (
    generate_dataset_uuid,
    generate_image_uuid,
)
from chaksudb.ingest.framework.ingestion_helpers import process_csv
from chaksudb.ingest.framework.provenance_context import get_current_provenance
from chaksudb.ingest.framework.raw_file_helpers import register_individual_file
from chaksudb.ingest.framework.split_assigner import (
    register_standard_splits,
    bulk_assign_images_to_split,
)
from chaksudb.ingest.framework.task_processors.segmentation_processor import (
    process_segmentation_from_multiclass_mask,
)
from chaksudb.ingest.framework.task_processors.classification_processor import (
    process_classification,
)

logger = logging.getLogger(__name__)

# Dataset metadata
DATASET_NAME = "ORIGA"
DATASET_URL = "https://www.kaggle.com/datasets/arnavjain1/glaucoma-datasets"
DATASET_LICENSE = "Research/Academic Use"  # Placeholder - update if known


def parse_eye_laterality(eye_str: str) -> Optional[str]:
    """
    Parse eye laterality from CSV Eye column.
    
    Args:
        eye_str: Eye value from CSV (e.g., "OS", "OD")
    
    Returns:
        "left", "right", or None if invalid
    """
    eye_str = eye_str.strip().upper()
    if eye_str == "OS":
        return "left"
    elif eye_str == "OD":
        return "right"
    else:
        return None


async def process_csv_entry(
    row: dict,
    idx: int,
    data_root: Path,
    dataset_id: UUID,
    origa_info_lookup: Dict[str, dict],
    tracker: ProgressTracker,
) -> Tuple[Optional[Image], List[ClassificationAnnotation], List[SegmentationAnnotation]]:
    """
    Process a single row from OrigaList.csv.
    
    Args:
        row: CSV row with Eye, Filename, ExpCDR, Set, Glaucoma
        idx: Row index
        data_root: Dataset root directory
        dataset_id: Dataset UUID
        origa_info_lookup: Lookup dict mapping filename to origa_info.csv data
        tracker: Progress tracker
    
    Returns:
        Tuple of (Image, List[ClassificationAnnotation], List[SegmentationAnnotation])
        or (None, [], []) on error
    """
    try:
        # Get provenance from context (set by process_csv)
        csv_raw_file_id, csv_chain_id = get_current_provenance()
        
        # Extract image information from CSV
        filename = row.get("Filename", "").strip()
        if not filename:
            logger.warning(f"Row {idx}: Missing Filename")
            tracker.update(success=False)
            tracker.record_error(
                error_type="missing_filename",
                error_message="Missing Filename in CSV row",
                item_id=f"row_{idx}",
            )
            return None, [], []
        
        # Extract base name without extension for matching
        base_name = Path(filename).stem  # e.g., "007"
        
        # Find image file (only original, not cropped/square)
        image_path = data_root / "Images" / filename
        if not image_path.exists():
            logger.warning(f"Row {idx}: Image not found: {image_path}")
            tracker.update(success=False)
            tracker.record_error(
                error_type="image_not_found",
                error_message=f"Image file not found: {image_path}",
                item_id=filename,
                item_path=str(image_path),
            )
            return None, [], []
        
        # Extract eye laterality
        eye_str = row.get("Eye", "").strip()
        eye_laterality = parse_eye_laterality(eye_str)
        
        # Generate image UUID
        original_image_id = base_name
        image_id = generate_image_uuid(dataset_id, original_image_id)
        
        # Create image model
        image = Image(
            image_id=image_id,
            dataset_id=dataset_id,
            original_image_id=original_image_id,
            **get_image_metadata_dict(image_path),
            modality="fundus",
            eye_laterality=eye_laterality,
            acquisition_date=None,
        )
        
        # Process classification with metadata from both CSVs
        classifications: List[ClassificationAnnotation] = []
        
        # Get glaucoma value from OrigaList.csv
        glaucoma_str = row.get("Glaucoma", "").strip()
        glaucoma_value = None
        if glaucoma_str:
            try:
                glaucoma_value = int(glaucoma_str)
                if glaucoma_value not in [0, 1]:
                    logger.warning(f"Row {idx}: Invalid Glaucoma value: {glaucoma_value}")
                    glaucoma_value = None
            except ValueError:
                logger.warning(f"Row {idx}: Could not parse Glaucoma: {glaucoma_str}")
        
        # Get additional metadata from origa_info.csv
        origa_info_data = origa_info_lookup.get(filename, {})
        cdr_value = origa_info_data.get("CDR")
        ecc_cup_value = origa_info_data.get("Ecc-Cup")
        ecc_disc_value = origa_info_data.get("Ecc-Disc")
        
        # Build classification value with metadata
        if glaucoma_value is not None:
            class_value = {
                "glaucoma": bool(glaucoma_value),
            }
            
            # Add CDR and eccentricity metrics if available
            if cdr_value is not None:
                try:
                    class_value["cdr"] = float(cdr_value)
                except (ValueError, TypeError):
                    pass
            
            if ecc_cup_value is not None:
                try:
                    class_value["ecc_cup"] = float(ecc_cup_value)
                except (ValueError, TypeError):
                    pass
            
            if ecc_disc_value is not None:
                try:
                    class_value["ecc_disc"] = float(ecc_disc_value)
                except (ValueError, TypeError):
                    pass
            
            # Process classification (use binary task_type but with extended JSONB)
            classifications_list = await process_classification(
                class_value=class_value,
                task_type="binary",  # Still binary classification
                class_name="glaucoma",
                image_id=image_id,
                raw_data_id=csv_raw_file_id,
                provenance_chain_id=csv_chain_id,
                annotation_method="manual",
            )
            classifications.extend(classifications_list)
        else:
            logger.warning(f"Row {idx}: Missing or invalid Glaucoma value for {filename}")
            tracker.record_error(
                error_type="missing_glaucoma",
                error_message="Missing or invalid Glaucoma value",
                item_id=filename,
            )
        
        # Process primary segmentation masks (manual annotations)
        segmentations: List[SegmentationAnnotation] = []
        
        # Find primary mask (only original, not cropped/square)
        primary_mask_path = data_root / "Masks" / f"{base_name}.png"
        
        if primary_mask_path.exists():
            # Register primary mask file for provenance
            mask_raw_file_id, mask_chain_id = await register_individual_file(
                file_path=primary_mask_path,
                dataset_id=dataset_id,
                unified_annotation_type="segmentation",
                file_type=None,  # Binary mask image, not structured file
                auto_detect_type=False,
            )
            
            # Process multi-class mask (extracts OD and cup separately)
            primary_segmentations = await process_segmentation_from_multiclass_mask(
                mask_path=primary_mask_path,
                class_names={1: "optic_disc", 2: "optic_cup"},
                image_id=image_id,
                classes_to_extract=[1, 2],
                raw_data_id=mask_raw_file_id,
                provenance_chain_id=mask_chain_id,
                annotation_method="manual",
                dataset_name=DATASET_NAME,
            )
            segmentations.extend(primary_segmentations)
        else:
            logger.warning(f"Row {idx}: Primary mask not found: {primary_mask_path}")
            tracker.record_error(
                error_type="mask_not_found",
                error_message=f"Primary mask not found: {primary_mask_path}",
                item_id=filename,
                item_path=str(primary_mask_path),
            )
        
        # Process semi-automatic annotations (.mat files) as pseudo masks
        mat_path = data_root / "Semi-automatic-annotations" / f"{base_name}.mat"
        
        if mat_path.exists():
            try:
                # Register .mat file for provenance
                mat_raw_file_id, mat_chain_id = await register_individual_file(
                    file_path=mat_path,
                    dataset_id=dataset_id,
                    unified_annotation_type="segmentation",
                    file_type="mat",
                    auto_detect_type=True,
                )
                
                # Load .mat file and extract mask
                mat_data = scipy.io.loadmat(str(mat_path))
                if 'mask' not in mat_data:
                    logger.warning(f"Row {idx}: .mat file missing 'mask' key: {mat_path}")
                    tracker.record_error(
                        error_type="mat_missing_mask",
                        error_message=".mat file missing 'mask' key",
                        item_id=filename,
                        item_path=str(mat_path),
                    )
                else:
                    mask_array = mat_data['mask']  # Shape: (2048, 3072), values: 0, 1, 2
                    
                    # Save mask array to temporary file for processing
                    # The processor expects a file path, not an array
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                        tmp_mask_path = Path(tmp_file.name)
                        # Convert to uint8 and save as PNG
                        mask_uint8 = mask_array.astype(np.uint8)
                        cv2.imwrite(str(tmp_mask_path), mask_uint8)
                        
                        try:
                            # Process multi-class mask (extracts OD and cup separately)
                            pseudo_segmentations = await process_segmentation_from_multiclass_mask(
                                mask_path=tmp_mask_path,
                                class_names={1: "optic_disc", 2: "optic_cup"},
                                image_id=image_id,
                                classes_to_extract=[1, 2],
                                raw_data_id=mat_raw_file_id,
                                provenance_chain_id=mat_chain_id,
                                annotation_method="pseudo",  # KEY: Mark as pseudo
                                dataset_name=DATASET_NAME,
                                original_source_path=mat_path,  # store .mat as provenance, not temp path
                            )
                            segmentations.extend(pseudo_segmentations)
                        finally:
                            # Clean up temporary file
                            if tmp_mask_path.exists():
                                tmp_mask_path.unlink()
                                
            except Exception as e:
                logger.error(f"Row {idx}: Failed to process .mat file {mat_path}: {e}", exc_info=True)
                tracker.record_error(
                    error_type="mat_processing_error",
                    error_message=str(e),
                    item_id=filename,
                    item_path=str(mat_path),
                )
        else:
            # .mat file not found is not an error (not all images may have semi-automatic annotations)
            logger.debug(f"Row {idx}: Semi-automatic annotation not found: {mat_path}")
        
        # Track split assignment (Set A/B from CSV)
        split_name = row.get("Set", "").strip().upper()
        if split_name in ["A", "B"]:
            # Will be assigned later in bulk
            pass
        
        # Consider row successful if we have at least classification
        if classifications:
            tracker.update(success=True)
            tracker.record_success("classification")
            for _ in segmentations:
                tracker.record_success("segmentation")
        else:
            logger.warning(f"Row {idx}: No valid annotations processed for {filename}")
            tracker.update(success=False)
            tracker.record_error(
                error_type="no_annotations",
                error_message="No valid annotations processed",
                item_id=filename,
            )
        
        return image, classifications, segmentations
        
    except Exception as e:
        logger.error(f"Row {idx}: Error processing row: {e}", exc_info=True)
        tracker.update(success=False)
        tracker.record_error(
            error_type="row_processing_error",
            error_message=str(e),
            item_id=f"row_{idx}",
        )
        return None, [], []


def load_origa_info_csv(data_root: Path) -> Dict[str, dict]:
    """
    Load origa_info.csv and create lookup dictionary by filename.
    
    Args:
        data_root: Dataset root directory
    
    Returns:
        Dictionary mapping filename to row data
    """
    import csv
    
    origa_info_path = data_root / "origa_info.csv"
    lookup: Dict[str, dict] = {}
    
    if not origa_info_path.exists():
        logger.warning(f"origa_info.csv not found: {origa_info_path}")
        return lookup
    
    try:
        with open(origa_info_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Extract filename from Image column (full path)
                image_path_str = row.get("Image", "").strip()
                if image_path_str:
                    # Extract just the filename from the path
                    filename = Path(image_path_str).name
                    lookup[filename] = row
    except Exception as e:
        logger.error(f"Failed to load origa_info.csv: {e}", exc_info=True)
    
    logger.info(f"Loaded {len(lookup)} entries from origa_info.csv")
    return lookup


async def ingest_origa() -> OperationStatistics:
    """
    Main ingestion function for ORIGA dataset.
    
    Returns:
        OperationStatistics with success/error counts
    """
    data_root = get_data_root() / "24_ORIGA"
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
        task_types=["classification", "segmentation"],
        description=(
            "ORIGA dataset contains fundus images with glaucoma classification "
            "and optic disc/cup segmentation annotations. The dataset includes "
            "binary classification (normal vs glaucoma) with CDR and eccentricity "
            "metrics, manual segmentation masks, and semi-automatic annotations."
        ),
    )
    await upsert_dataset(dataset)
    
    # Step 2: Load origa_info.csv for metadata lookup
    logger.info("Loading origa_info.csv for metadata...")
    origa_info_lookup = await asyncio.to_thread(load_origa_info_csv, data_root)
    
    # Step 3: Setup progress tracker
    # Estimate: ~131 images + ~131 classifications + ~262 primary segmentations + ~262 pseudo segmentations
    tracker = ProgressTracker(
        total=786,  # Rough estimate
        description=f"Ingesting {DATASET_NAME}"
    )
    
    # Step 4: Process OrigaList.csv
    logger.info("Processing OrigaList.csv...")
    csv_path = data_root / "OrigaList.csv"
    
    if not csv_path.exists():
        logger.error(f"CSV file not found: {csv_path}")
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Collect images, classifications, segmentations, and split assignments
    all_images: List[Image] = []
    all_classifications: List[ClassificationAnnotation] = []
    all_segmentations: List[SegmentationAnnotation] = []
    split_assignments: Dict[str, List[UUID]] = {"A": [], "B": []}  # Set A/B -> image_ids
    
    async def process_row(row: dict, idx: int) -> None:
        """Wrapper to process CSV row and collect results."""
        image, classifications, segmentations = await process_csv_entry(
            row=row,
            idx=idx,
            data_root=data_root,
            dataset_id=dataset_id,
            origa_info_lookup=origa_info_lookup,
            tracker=tracker,
        )
        if image:
            all_images.append(image)
            # Track split assignment
            split_name = row.get("Set", "").strip().upper()
            if split_name in ["A", "B"]:
                split_assignments[split_name].append(image.image_id)
        all_classifications.extend(classifications)
        all_segmentations.extend(segmentations)
    
    # Process CSV with automatic provenance
    stats, csv_raw_file_id, csv_chain_id = await process_csv(
        csv_path=csv_path,
        dataset_id=dataset_id,
        unified_annotation_type="classification",  # Primary task
        process_row_fn=process_row,
        progress_tracker=tracker,
        skip_errors=True,
    )
    
    logger.info(
        f"Processed OrigaList.csv: {stats.successful_items} successful, "
        f"{stats.failed_items} failed"
    )
    logger.info(
        f"Collected {len(all_images)} images, {len(all_classifications)} classifications, "
        f"and {len(all_segmentations)} segmentations"
    )
    
    # Step 5: Bulk upsert images
    logger.info(f"Upserting {len(all_images)} images...")
    await bulk_upsert_images(all_images, batch_size=1000)
    
    # Step 6: Bulk upsert classifications
    logger.info(f"Upserting {len(all_classifications)} classification annotations...")
    await bulk_upsert_classification_annotations(all_classifications, batch_size=1000)
    
    # Step 7: Upsert segmentations (individual, no bulk operation available)
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
    
    # Step 8: Register splits and assign images
    logger.info("Registering dataset splits...")
    splits = await register_standard_splits(
        dataset_id=dataset_id,
        split_type="explicit",  # Split is explicitly encoded in CSV (Set A/B)
    )
    
    # Map Set A -> train, Set B -> test
    if split_assignments["A"]:
        logger.info(f"Assigning {len(split_assignments['A'])} images to train split (Set A)")
        await bulk_assign_images_to_split(split_assignments["A"], splits["train"])
    
    if split_assignments["B"]:
        logger.info(f"Assigning {len(split_assignments['B'])} images to test split (Set B)")
        await bulk_assign_images_to_split(split_assignments["B"], splits["test"])
    
    tracker.finish()
    return tracker.get_statistics()


async def main():
    """Entry point for script execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    try:
        stats = await ingest_origa()
        
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
