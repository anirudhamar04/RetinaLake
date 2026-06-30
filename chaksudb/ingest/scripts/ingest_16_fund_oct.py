"""
Ingestion script for FUND-OCT dataset.

Dataset: FUND-OCT (Fundus and OCT Images Database)
Structure: Complex OCT dataset with patient folders organized by disease category
Annotations:
  - Disease category classification (from parent folder: Macula/OD categories)
  - Patient registration (from patient folder names: P_1, P_2, etc.)
  - Laterality tracking (from "Left Eye" / "Right Eye" folders)

Key Features:
  - Two main categories: Macula (with 6 disease types) and OD (with 2 disease types)
  - Patient folders (P_1, P_2, etc.) with Left Eye and Right Eye subfolders
  - Multiple image types:
    * B-scan: Key frame of OCT volume (frame_index=0)
    * VolumeFrames: OCT volume frames (frame_index=1, 2, 3, ...)
    * Color/Red-free: Corresponding fundus images (modality="fundus", standalone)
  - OCT volumes: B-scan + VolumeFrames grouped together with image_group
  - Both OCT and fundus modalities
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set
from uuid import UUID

from chaksudb.common.progress import ProgressTracker, OperationStatistics
from chaksudb.config.config import get_data_root
from chaksudb.db.models import (
    Dataset,
    Image,
    ImageGroup,
    Patient,
    PatientImage,
    ClassificationAnnotation,
)
from chaksudb.db.queries import (
    upsert_dataset,
    bulk_upsert_image_groups,
    bulk_upsert_images,
    bulk_upsert_patients,
    bulk_upsert_patient_images,
    bulk_upsert_classification_annotations,
)
from chaksudb.ingest.framework import (
    get_image_metadata_dict,
    process_folder_tree,
)
from chaksudb.ingest.framework.gen_uuid import (
    generate_dataset_uuid,
    generate_image_group_uuid,
    generate_image_uuid,
    generate_patient_uuid,
    generate_patient_image_uuid,
)
from chaksudb.ingest.framework.task_processors.classification_processor import (
    process_classification,
)
from chaksudb.ingest.framework.provenance_context import get_current_provenance
from chaksudb.ingest.framework.split_assigner import auto_stratified_splits

logger = logging.getLogger(__name__)

# Dataset metadata
DATASET_NAME = "FUND-OCT"
DATASET_URL = "https://www.kaggle.com/datasets/paultimothymooney/fundus-and-oct-images-database"
DATASET_LICENSE = "Unknown"

# Disease category mapping
# Macula categories
MACULA_CATEGORIES = {
    "acute CSR": "acute_csr",
    "chronic CSR": "chronic_csr",
    "ci-DME": "ci_dme",
    "geographic_AMD": "geographic_amd",
    "Healthy": "healthy",
    "neovascular_AMD": "neovascular_amd",
}

# OD (Optic Disc) categories
OD_CATEGORIES = {
    "Glaucoma": "glaucoma",
    "Healthy": "healthy",
}

# Image file extensions to process
IMAGE_EXTENSIONS = {".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"}


def is_bscan_image(filename: str) -> bool:
    """Check if filename indicates a B-scan key frame."""
    return "B-scan" in filename or "b-scan" in filename.lower()


def is_volumeframe(rel_path: Path) -> bool:
    """Check if image is in a VolumeFrames folder."""
    return "VolumeFrames" in rel_path.parts


def is_fundus_image(filename: str) -> bool:
    """Check if image is a fundus image (Color or Red-free)."""
    filename_lower = filename.lower()
    return "color" in filename_lower or "red-free" in filename_lower or "redfree" in filename_lower


def extract_frame_index_from_volumeframe(filename: str) -> Optional[int]:
    """Extract frame index from VolumeFrames filename (e.g., '1.jpg' -> 1)."""
    try:
        # Remove extension and parse as integer
        stem = Path(filename).stem
        return int(stem)
    except ValueError:
        return None


def get_oct_volume_key(patient_id: str, laterality: str, eye_folder_path: Path) -> str:
    """Generate a unique key for an OCT volume (B-scan + VolumeFrames)."""
    # Use the eye folder path as identifier (e.g., "Macula/acute CSR/P_1/Left Eye")
    return f"{patient_id}:{laterality}:{str(eye_folder_path)}"


def extract_disease_category(relative_path: Path, root_type: str = "macula") -> Optional[str]:
    """
    Extract disease category from folder path.
    
    Path structure (relative to Macula/ or OD/):
    {category}/P_X/{Left Eye|Right Eye}/...
    
    Args:
        relative_path: Path relative to Macula/ or OD/ root
        root_type: "macula" or "od" to determine category mapping
    
    Returns normalized category name or None.
    """
    parts = relative_path.parts
    
    # Category folder should be the first part
    if len(parts) < 1:
        return None
    
    category_name = parts[0]
    
    # Map based on root type
    if root_type == "macula":
        return MACULA_CATEGORIES.get(category_name)
    elif root_type == "od":
        return OD_CATEGORIES.get(category_name)
    
    return None


def extract_patient_id(relative_path: Path) -> Optional[str]:
    """
    Extract patient ID from folder path.
    
    Path structure: .../P_X/{Left Eye|Right Eye}/...
    
    Returns patient ID (e.g., "P_1") or None.
    """
    parts = relative_path.parts
    
    # Find the patient folder (should be named P_X)
    for part in parts:
        if part.startswith("P_") and len(part) > 2:
            return part
    
    return None


def extract_laterality(relative_path: Path) -> Optional[str]:
    """
    Extract laterality from folder path.
    
    Path structure: .../P_X/{Left Eye|Right Eye}/...
    
    Returns 'left', 'right', or None.
    """
    parts = relative_path.parts
    
    # Check for "Left Eye" or "Right Eye" folder
    for part in parts:
        part_lower = part.lower()
        if "left" in part_lower and "eye" in part_lower:
            return "left"
        elif "right" in part_lower and "eye" in part_lower:
            return "right"
    
    return None


async def ingest_fund_oct() -> OperationStatistics:
    """
    Main ingestion function for FUND-OCT dataset.
    
    Strategy:
    - Use process_folder_tree() to walk patient folders
    - Extract disease category from parent folder (Macula/OD categories)
    - Extract patient ID from folder name (P_1, P_2, etc.)
    - Extract laterality from folder name (Left Eye/Right Eye)
    - Register patients (even if only ID is available, per plan requirements)
    - Group OCT volumes: B-scan (frame_index=0) + VolumeFrames (frame_index=1,2,3...)
    - Create image_groups for OCT volumes
    - Process fundus images (Color, Red-free) as standalone fundus modality
    - Create classifications for disease categories (OCT images only)
    
    Returns:
        OperationStatistics with success/error counts
    """
    data_root = get_data_root() / "16_FUND-OCT"
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
        modality_types=["oct", "fundus"],  # Both OCT and fundus images
    )
    await upsert_dataset(dataset)
    
    # Step 2: Count total images for progress tracking
    logger.info("Counting images across all folders...")
    total_images = 0
    for root_dir in [data_root / "Macula", data_root / "OD"]:
        if root_dir.exists():
            for item in root_dir.rglob("*"):
                if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS:
                    total_images += 1
    
    logger.info(f"Total images found: {total_images}")
    
    # Step 3: Setup progress tracker
    tracker = ProgressTracker(
        total=total_images,
        description=f"Ingesting {DATASET_NAME}"
    )
    
    # Collections for bulk upsert
    all_patients: Dict[str, Patient] = {}  # patient_id_str -> Patient
    all_image_groups: Dict[str, ImageGroup] = {}  # volume_key -> ImageGroup
    all_images: List[Image] = []
    all_patient_images: List[PatientImage] = []
    all_classifications: List[ClassificationAnnotation] = []
    image_labels: dict = {}  # image_id → disease category for stratified splitting

    # Track OCT volumes: volume_key -> group_id
    oct_volume_groups: Dict[str, UUID] = {}
    
    def create_image_handler(root_type: str):
        """
        Create an image handler function that captures the root type.
        
        Args:
            root_type: "macula" or "od" to determine category mapping
        """
        async def handle_image(file_path: Path, rel_path: Path, depth: int):
            """
            Process each image file.
            
            Extracts:
            - Disease category from folder path
            - Patient ID from folder path
            - Laterality from folder path
            """
            try:
                # Extract metadata from path
                disease_category = extract_disease_category(rel_path, root_type)
                patient_id_str = extract_patient_id(rel_path)
                laterality = extract_laterality(rel_path)
                
                # Validate required fields
                if not disease_category:
                    logger.warning(
                        f"Could not extract disease category from path: {rel_path}"
                    )
                    tracker.record_error(
                        error_type="path_parse",
                        error_message=f"Cannot extract disease category from path",
                        item_id=file_path.stem,
                        item_path=str(file_path),
                    )
                    tracker.update(success=False)
                    return
                
                if not patient_id_str:
                    logger.warning(
                        f"Could not extract patient ID from path: {rel_path}"
                    )
                    tracker.record_error(
                        error_type="path_parse",
                        error_message=f"Cannot extract patient ID from path",
                        item_id=file_path.stem,
                        item_path=str(file_path),
                    )
                    tracker.update(success=False)
                    return
                
                # Register patient if not already registered
                if patient_id_str not in all_patients:
                    patient_id = generate_patient_uuid(dataset_id, patient_id_str)
                    patient = Patient(
                        patient_id=patient_id,
                        dataset_id=dataset_id,
                        original_patient_id=patient_id_str,
                        age=None,
                        sex=None,
                        ethnicity=None,
                        nationality=None,
                        comorbidities=None,
                    )
                    all_patients[patient_id_str] = patient
                
                patient = all_patients[patient_id_str]
                
                # Determine image type and modality
                filename = file_path.name
                is_bscan = is_bscan_image(filename)
                is_volumeframe_file = is_volumeframe(rel_path)
                is_fundus = is_fundus_image(filename)
                
                # Determine modality and grouping
                if is_fundus:
                    # Fundus images (Color, Red-free) are standalone
                    modality = "fundus"
                    group_id = None
                    frame_index = None
                elif is_bscan or is_volumeframe_file:
                    # OCT images: B-scan or VolumeFrames
                    modality = "oct"
                    
                    # Get the eye folder path (parent of VolumeFrames or parent of B-scan)
                    if is_volumeframe_file:
                        # VolumeFrames/1.jpg -> parent is "Left Eye" or "Right Eye"
                        eye_folder_path = rel_path.parent.parent  # Go up from VolumeFrames
                    else:
                        # B-scan is directly in eye folder
                        eye_folder_path = rel_path.parent
                    
                    volume_key = get_oct_volume_key(
                        patient_id_str, 
                        laterality or "unknown", 
                        eye_folder_path
                    )
                    
                    # Create or get image group for this OCT volume
                    if volume_key not in oct_volume_groups:
                        group_id = generate_image_group_uuid(
                            dataset_id=dataset_id,
                            group_type="oct_volume",
                            group_identifier=volume_key
                        )
                        image_group = ImageGroup(
                            group_id=group_id,
                            dataset_id=dataset_id,
                            group_type="oct_volume",
                        )
                        all_image_groups[volume_key] = image_group
                        oct_volume_groups[volume_key] = group_id
                    else:
                        group_id = oct_volume_groups[volume_key]
                    
                    # Set frame_index
                    if is_bscan:
                        frame_index = 0  # Key frame
                    elif is_volumeframe_file:
                        frame_index = extract_frame_index_from_volumeframe(filename)
                        if frame_index is None:
                            logger.warning(
                                f"Could not extract frame index from {filename}, skipping"
                            )
                            tracker.update(success=False)
                            tracker.record_error(
                                error_type="frame_index_parse",
                                error_message=f"Cannot extract frame index from filename",
                                item_id=file_path.stem,
                                item_path=str(file_path),
                            )
                            return
                    else:
                        frame_index = None
                else:
                    # Other image types (e.g., fluid.jpg) - treat as standalone OCT
                    modality = "oct"
                    group_id = None
                    frame_index = None
                
                # Generate image UUID
                # Use relative path to ensure uniqueness across patients
                image_id = generate_image_uuid(dataset_id, str(rel_path))
                
                # Create image with automatic metadata extraction
                image = Image(
                    image_id=image_id,
                    dataset_id=dataset_id,
                    original_image_id=str(rel_path),
                    **get_image_metadata_dict(file_path),
                    modality=modality,
                    eye_laterality=laterality,
                    group_id=group_id,
                    frame_index=frame_index,
                )
                all_images.append(image)
                if disease_category:
                    image_labels[image_id] = disease_category

                # Link patient to image
                relationship_id = generate_patient_image_uuid(patient.patient_id, image_id)
                patient_image = PatientImage(
                    relationship_id=relationship_id,
                    patient_id=patient.patient_id,
                    image_id=image_id,
                    exam_date=None,
                )
                all_patient_images.append(patient_image)
                
                # Create classification for disease category (only for OCT images, not fundus)
                if modality == "oct":
                    classifications = await process_classification(
                        class_value=True,
                        task_type="binary",
                        class_name=disease_category,
                        image_id=image_id,
                        annotation_method="manual",  # Folder structure manually curated
                    )
                    all_classifications.extend(classifications)
                
                tracker.update(success=True)
                tracker.record_success("image")
                
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}", exc_info=True)
                tracker.update(success=False)
                tracker.record_error(
                    error_type="processing",
                    error_message=str(e),
                    item_id=file_path.stem,
                    item_path=str(file_path),
                )
        
        return handle_image
    
    # Step 4: Process folder tree for Macula images
    logger.info("Processing Macula images...")
    macula_dir = data_root / "Macula"
    if macula_dir.exists():
        macula_handler = create_image_handler("macula")
        macula_stats = await process_folder_tree(
            root_dir=macula_dir,
            dataset_id=dataset_id,
            unified_annotation_type="classification",
            process_file_fn=macula_handler,
            file_extensions=IMAGE_EXTENSIONS,
            recursive=True,
            include_dirs=False,
            progress_tracker=tracker,
            skip_errors=True,
        )
        logger.info(
            f"Macula: {macula_stats.successful_items} successful, "
            f"{macula_stats.failed_items} failed"
        )
    
    # Step 5: Process folder tree for OD images
    logger.info("Processing OD images...")
    od_dir = data_root / "OD"
    if od_dir.exists():
        od_handler = create_image_handler("od")
        od_stats = await process_folder_tree(
            root_dir=od_dir,
            dataset_id=dataset_id,
            unified_annotation_type="classification",
            process_file_fn=od_handler,
            file_extensions=IMAGE_EXTENSIONS,
            recursive=True,
            include_dirs=False,
            progress_tracker=tracker,
            skip_errors=True,
        )
        logger.info(
            f"OD: {od_stats.successful_items} successful, "
            f"{od_stats.failed_items} failed"
        )
    
    # Step 6: Bulk upsert in correct order (groups -> patients -> images -> links -> annotations)
    logger.info(
        f"Upserting {len(all_image_groups)} image groups, {len(all_patients)} patients, "
        f"{len(all_images)} images, {len(all_patient_images)} patient links, "
        f"{len(all_classifications)} classifications..."
    )
    
    # Upsert image groups first (before images that reference them)
    if all_image_groups:
        await bulk_upsert_image_groups(list(all_image_groups.values()), batch_size=1000)
    
    # Upsert patients
    if all_patients:
        await bulk_upsert_patients(list(all_patients.values()), batch_size=1000)
    
    # Upsert images
    if all_images:
        await bulk_upsert_images(all_images, batch_size=1000)
    
    # Upsert patient links and classifications in parallel (both depend on images)
    await asyncio.gather(
        bulk_upsert_patient_images(all_patient_images, batch_size=1000),
        bulk_upsert_classification_annotations(all_classifications, batch_size=1000),
    )
    
    # Step 7: Register splits — stratified 90/10 train+test, then 90/10 train+val
    all_image_ids_for_split = [img.image_id for img in all_images]
    if all_image_ids_for_split:
        logger.info("Registering dataset splits...")
        await auto_stratified_splits(
            dataset_id=dataset_id,
            split_assignments={"train": all_image_ids_for_split},
            labels=image_labels,
            split_type="explicit",
        )

    # Step 8: Finish and return statistics
    tracker.finish()
    stats = tracker.get_statistics()
    
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
    logger.info(f"Ingested {len(all_image_groups)} OCT volume groups")
    logger.info(f"Ingested {len(all_patients)} patients")
    logger.info(f"Ingested {len(all_images)} images")
    logger.info(f"Ingested {len(all_classifications)} classifications")
    logger.info("=" * 80)
    
    return stats


async def main():
    """Entry point for script execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    try:
        stats = await ingest_fund_oct()
        
        logger.info("=" * 80)
        logger.info(f"FUND-OCT ingestion completed!")
        logger.info(f"Total: {stats.total_items}")
        logger.info(f"Successful: {stats.successful_items}")
        logger.info(f"Failed: {stats.failed_items}")
        logger.info("=" * 80)
        
        return 0 if stats.failed_items == 0 else 1
        
    except Exception as e:
        logger.exception(f"Fatal error during FUND-OCT ingestion: {e}")
        raise


if __name__ == "__main__":
    exit(asyncio.run(main()))
