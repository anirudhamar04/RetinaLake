"""
Ingestion script for E-ophta dataset.

Dataset: E-ophta - Microaneurysms and Exudates Detection
Structure: Patient folders with classification subfolders and binary masks
Annotations:
  - Microaneurysm (MA) segmentation (binary masks)
  - Exudate (EX) segmentation (binary masks)
  - Disease classification (MA, healthy, EX) from folder structure

Key Features:
  - Two subsets: e_optha_MA (microaneurysms) and e_optha_EX (exudates)
  - Patient-organized structure with classification folders
  - Binary masks for lesion segmentation
  - Folder-based classification (MA/healthy/EX)
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
)
from chaksudb.db.queries import (
    upsert_dataset,
    bulk_upsert_images,
    upsert_segmentation_annotation,
    bulk_upsert_classification_annotations,
)
from chaksudb.ingest.framework import (
    get_image_metadata_dict,
    find_images,
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
from chaksudb.ingest.framework.split_assigner import auto_stratified_splits

logger = logging.getLogger(__name__)

# Dataset metadata
DATASET_NAME = "E-ophta"
DATASET_URL = "https://www.adcis.net/en/third-party/e-ophtha/"
DATASET_LICENSE = "Research/Academic Use"

# lesion_type multi-class labels (index -> label) so class_index is always populated
EOPHTA_LESION_LABELS = {0: "normal", 1: "microaneurysm", 2: "exudate"}

# Mask extensions to search for
MASK_EXTENSIONS = {".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG"}


def extract_patient_id_from_path(image_path: Path) -> str:
    """
    Extract patient ID from image path.
    
    Path structure: e_optha_MA/MA/E0000043/DS000DGS.JPG
    Returns: E0000043
    """
    # Get the parent directory name (patient folder)
    return image_path.parent.name


def extract_classification_from_path(image_path: Path, subset: str) -> str:
    """
    Extract classification from image path.
    
    Path structure: e_optha_MA/MA/E0000043/DS000DGS.JPG
    Returns: "MA" (or "healthy" or "EX")
    """
    # Get the grandparent directory name (classification folder)
    parent_name = image_path.parent.parent.name
    
    # Normalize classification names
    if parent_name.lower() == "ma":
        return "MA"
    elif parent_name.lower() == "ex":
        return "EX"
    elif parent_name.lower() == "healthy":
        return "healthy"
    else:
        # Fallback: use the folder name as-is
        return parent_name


def find_mask_path(image_path: Path, annotation_dir: Path, subset_name: str) -> Optional[Path]:
    """
    Find corresponding mask file for an image.
    
    Args:
        image_path: Path to the image file
        annotation_dir: Directory containing annotation masks
        subset_name: "e_optha_MA" or "e_optha_EX" to determine suffix pattern
    
    Returns:
        Path to mask file if found, None otherwise
    
    Note:
        - MA masks: Same stem as image (e.g., DS000DGS.JPG -> DS000DGS.png)
        - EX masks: Stem with _EX suffix (e.g., DS000FGD.JPG -> DS000FGD_EX.png)
        - Healthy images: No masks (this function returns None)
    """
    # Get patient folder name
    patient_id = extract_patient_id_from_path(image_path)
    
    # Look for mask in patient's annotation folder
    patient_annotation_dir = annotation_dir / patient_id
    if not patient_annotation_dir.exists():
        return None
    
    # Get image stem (filename without extension)
    image_stem = image_path.stem
    
    # Determine suffix pattern based on subset
    # EX masks have _EX suffix, MA masks don't have suffix
    if subset_name == "e_optha_EX":
        # Try with _EX suffix first (EX pattern)
        suffixes_to_try = [f"{image_stem}_EX", image_stem]
    elif subset_name == "e_optha_MA":
        # Try without suffix first (MA pattern), then with _MA if needed
        suffixes_to_try = [image_stem, f"{image_stem}_MA"]
    else:
        # Fallback: try both patterns
        suffixes_to_try = [image_stem, f"{image_stem}_EX", f"{image_stem}_MA"]
    
    # Try to find mask with different suffix patterns
    for suffix in suffixes_to_try:
        for ext in MASK_EXTENSIONS:
            mask_path = patient_annotation_dir / f"{suffix}{ext}"
            if mask_path.exists():
                return mask_path
    
    return None


async def process_subset(
    data_root: Path,
    dataset_id: UUID,
    subset_name: str,
    tracker: ProgressTracker,
) -> Tuple[
    List[Image],
    List[SegmentationAnnotation],
    List[ClassificationAnnotation],
    List[UUID],
    Dict[str, List[UUID]],  # patient_id -> image_ids
    dict,  # image_id -> classification label
]:
    """
    Process a single subset (MA or EX).
    
    Args:
        data_root: Dataset root directory
        dataset_id: Dataset UUID
        subset_name: "e_optha_MA" or "e_optha_EX"
        tracker: Progress tracker
    
    Returns:
        Tuple of (images, segmentations, classifications, image_ids, patient_image_map)
        Note: patient_image_map is used for logging only, not for patient registration
    """
    all_images: List[Image] = []
    all_segmentations: List[SegmentationAnnotation] = []
    all_classifications: List[ClassificationAnnotation] = []
    image_ids: List[UUID] = []
    patient_image_map: Dict[str, List[UUID]] = {}  # patient_id -> list of image_ids
    label_dict: dict = {}  # image_id -> classification label

    subset_dir = data_root / subset_name

    if not subset_dir.exists():
        logger.warning(f"Subset directory not found: {subset_dir}")
        return all_images, all_segmentations, all_classifications, image_ids, patient_image_map, label_dict
    
    # Determine annotation directory name and lesion subtype code
    if subset_name == "e_optha_MA":
        annotation_dir_name = "Annotation_MA"
        lesion_subtype_code = "MA"
    elif subset_name == "e_optha_EX":
        annotation_dir_name = "Annotation_EX"
        lesion_subtype_code = "EX"
    else:
        logger.error(f"Unknown subset: {subset_name}")
        return all_images, all_segmentations, all_classifications, image_ids, patient_image_map
    
    annotation_dir = subset_dir / annotation_dir_name
    
    # Find all classification folders (MA, healthy, EX)
    classification_folders = []
    for item in subset_dir.iterdir():
        if item.is_dir() and item.name not in [annotation_dir_name, "Annotation_MA", "Annotation_EX"]:
            classification_folders.append(item)
    
    logger.info(f"Found {len(classification_folders)} classification folders in {subset_name}")
    
    # Process each classification folder
    for classification_folder in classification_folders:
        classification_name = classification_folder.name
        logger.info(f"Processing classification folder: {classification_name}")
        
        # Find all images in this classification folder (recursively in patient subfolders)
        image_paths = []
        for patient_folder in classification_folder.iterdir():
            if patient_folder.is_dir():
                patient_images = await asyncio.to_thread(find_images, patient_folder, recursive=True)
                image_paths.extend(patient_images)
        
        logger.info(f"Found {len(image_paths)} images in {classification_name} folder")
        
        # Process each image
        for image_path in image_paths:
            try:
                # Extract metadata
                patient_id = extract_patient_id_from_path(image_path)
                classification = extract_classification_from_path(image_path, subset_name)
                
                # Generate image UUID
                # Use patient_id + filename (without subset) to allow same image in both subsets
                # This enables multi-segmentation: one image can have both MA and EX masks
                image_filename = image_path.name
                original_image_id = f"{patient_id}/{image_filename}"
                image_id = generate_image_uuid(dataset_id, original_image_id)
                
                # Create image model
                # Note: If same image appears in both subsets, we'll deduplicate later
                image = Image(
                    image_id=image_id,
                    dataset_id=dataset_id,
                    original_image_id=original_image_id,
                    **get_image_metadata_dict(image_path),
                    modality="fundus",
                    acquisition_date=None,
                )
                
                all_images.append(image)
                image_ids.append(image_id)
                label_dict[image_id] = classification

                # Track patient-image relationship
                if patient_id not in patient_image_map:
                    patient_image_map[patient_id] = []
                patient_image_map[patient_id].append(image_id)
                
                # Find and process mask if available
                # Note: Healthy images don't have masks, so this will return None for them
                mask_path = find_mask_path(image_path, annotation_dir, subset_name)
                if mask_path and mask_path.exists():
                    # Process segmentation annotation
                    segmentation = await process_segmentation_from_binary_mask(
                        mask_path=mask_path,
                        annotation_type="lesions",
                        image_id=image_id,
                        lesion_subtype=lesion_subtype_code,
                        annotation_description="Lesion segmentation",
                        fill_holes=False,
                        raw_data_id=None,  # Mask file will be registered by processor
                        merge_nonzero=True,
                        expert_annotation_id=None,
                        annotation_method="manual",
                        provenance_chain_id=None,
                        dataset_name=DATASET_NAME,
                    )
                    all_segmentations.append(segmentation)
                else:
                    logger.debug(f"No mask found for image: {image_path}")
                
                # Process classification annotation
                # Map classification to standard labels
                if classification.lower() == "ma":
                    class_value = "microaneurysm"
                    class_name = "lesion_type"
                elif classification.lower() == "ex":
                    class_value = "exudate"
                    class_name = "lesion_type"
                elif classification.lower() == "healthy":
                    class_value = "normal"
                    class_name = "lesion_type"
                else:
                    class_value = classification
                    class_name = "lesion_type"
                
                classification_anns = await process_classification(
                    class_value=class_value,
                    task_type="multi_class",
                    task_name="lesion_type",
                    class_name=class_name,
                    image_id=image_id,
                    class_labels=EOPHTA_LESION_LABELS,
                    raw_data_id=None,  # Folder structure is implicit
                    expert_annotation_id=None,
                    annotation_method="manual",  # Classification from folder structure
                    provenance_chain_id=None,
                )
                all_classifications.extend(classification_anns)
                
                tracker.update(success=True)
                tracker.record_success("image")
                
            except Exception as e:
                tracker.update(success=False)
                tracker.record_error(
                    error_type="image_processing",
                    error_message=str(e),
                    item_id=str(image_path),
                    item_path=str(image_path),
                )
                logger.error(f"Failed to process image {image_path}: {e}")

    return all_images, all_segmentations, all_classifications, image_ids, patient_image_map, label_dict


async def ingest_eophta() -> OperationStatistics:
    """
    Main ingestion function for E-ophta dataset.
    
    Returns:
        OperationStatistics with success/error counts
    """
    data_root = get_data_root() / "20_E-ophta"
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
        task_types=["segmentation", "classification"],
        description=(
            "E-ophta dataset contains fundus images organized by patients with "
            "microaneurysm (MA) and exudate (EX) annotations. The dataset includes "
            "two subsets: e_optha_MA for microaneurysm detection and e_optha_EX for "
            "exudate detection. Images are organized in patient folders with "
            "classification subfolders (MA, healthy, EX) and corresponding binary "
            "segmentation masks."
        ),
    )
    await upsert_dataset(dataset)
    
    # Step 2: Setup progress tracker
    # Estimate: ~200 images per subset, ~200 segmentations, ~200 classifications
    tracker = ProgressTracker(
        total=600,  # Rough estimate
        description=f"Ingesting {DATASET_NAME}"
    )
    
    # Step 3: Process both subsets in parallel
    logger.info("Processing e_optha_MA subset...")
    logger.info("Processing e_optha_EX subset...")
    
    (ma_images, ma_segmentations, ma_classifications, ma_image_ids, ma_patient_map, ma_labels), \
    (ex_images, ex_segmentations, ex_classifications, ex_image_ids, ex_patient_map, ex_labels) = await asyncio.gather(
        process_subset(data_root, dataset_id, "e_optha_MA", tracker),
        process_subset(data_root, dataset_id, "e_optha_EX", tracker),
    )
    
    # Combine all data
    # Deduplicate images (same image might appear in both subsets for multi-segmentation)
    image_dict: Dict[UUID, Image] = {}
    for image in ma_images + ex_images:
        # If same image_id appears in both subsets, keep the first one
        # (they should be identical - same patient + filename)
        if image.image_id not in image_dict:
            image_dict[image.image_id] = image
    
    all_images = list(image_dict.values())
    all_segmentations = ma_segmentations + ex_segmentations
    all_classifications = ma_classifications + ex_classifications
    all_image_ids = list(set(ma_image_ids + ex_image_ids))  # Deduplicate
    image_labels = {**ma_labels, **ex_labels}  # ex_labels overwrite on overlap (both classify same image)
    
    # Combine patient maps (handle overlapping patient IDs)
    all_patient_map: Dict[str, List[UUID]] = {}
    for patient_id, image_ids_list in ma_patient_map.items():
        if patient_id not in all_patient_map:
            all_patient_map[patient_id] = []
        all_patient_map[patient_id].extend(image_ids_list)
    for patient_id, image_ids_list in ex_patient_map.items():
        if patient_id not in all_patient_map:
            all_patient_map[patient_id] = []
        all_patient_map[patient_id].extend(image_ids_list)
    
    logger.info(
        f"Total: {len(all_images)} images, {len(all_segmentations)} segmentations, "
        f"{len(all_classifications)} classifications, {len(all_patient_map)} patients"
    )
    
    # Step 4: Bulk upsert images
    logger.info(f"Upserting {len(all_images)} images...")
    await bulk_upsert_images(all_images, batch_size=1000)
    
    # Step 5: Patient information is stored in image metadata (comorbidities.patient_id)
    # We don't register patients since we only have patient IDs from folder structure,
    # not actual demographic metadata. This follows the patient_register documentation.
    logger.info(f"Patient IDs stored in image metadata for {len(all_patient_map)} unique patients")
    
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
    
    # Step 7: Bulk upsert classifications
    logger.info(f"Upserting {len(all_classifications)} classification annotations...")
    await bulk_upsert_classification_annotations(all_classifications, batch_size=1000)
    
    # Register splits — stratified 90/10 train+test, then 90/10 train+val
    if all_image_ids:
        logger.info("Registering dataset splits...")
        await auto_stratified_splits(
            dataset_id=dataset_id,
            split_assignments={"train": all_image_ids},
            labels=image_labels,
            split_type="explicit",
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
        stats = await ingest_eophta()
        
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
