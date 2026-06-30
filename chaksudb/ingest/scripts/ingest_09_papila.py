"""
Ingestion script for PAPILA dataset.

Dataset: PAPILA (Papilla Database for Glaucoma Research)
Structure: Excel patient data + fundus images + multi-expert contour segmentations
Annotations:
  - Patient-level: Age, gender, diagnosis, clinical measurements (IOP, pachymetry, etc.)
  - Image-level: Multi-label diagnosis classification (normal, glaucoma, suspicious)
  - Image-level: Multi-expert optic disc/cup segmentations (2 experts per structure)

Key Features:
  - 244 patients with both eyes (OD=right, OS=left)
  - 488 fundus images (2576*1934 pixels)
  - Clinical measurements stored in patient.comorbidities JSONB
  - 2 expert annotators for optic disc and optic cup segmentations
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from uuid import UUID

from chaksudb.common.progress import ProgressTracker, OperationStatistics
from chaksudb.config.config import get_data_root
from chaksudb.db.models import (
    Dataset,
    Image,
    Patient,
    PatientImage,
    ClassificationAnnotation,
    SegmentationAnnotation,
    Expert,
    ExpertAnnotation,
)
from chaksudb.db.queries import (
    upsert_dataset,
    bulk_upsert_images,
    bulk_upsert_patients,
    bulk_upsert_patient_images,
    bulk_upsert_classification_annotations,
    upsert_segmentation_annotation,
    upsert_expert,
    upsert_expert_annotation,
)
from chaksudb.ingest.framework import (
    get_image_metadata_dict,
    process_excel,
    find_images,
)
from chaksudb.ingest.framework.gen_uuid import (
    generate_dataset_uuid,
    generate_image_uuid,
    generate_patient_uuid,
    generate_patient_image_uuid,
    generate_expert_uuid,
    generate_expert_annotation_uuid,
)
from chaksudb.ingest.framework.task_processors.classification_processor import process_classification
from chaksudb.ingest.framework.task_processors.segmentation_processor import process_segmentation_from_contour
from chaksudb.ingest.framework.raw_file_helpers import register_individual_file
from chaksudb.ingest.framework.split_assigner import auto_stratified_splits

logger = logging.getLogger(__name__)

# Dataset metadata
DATASET_NAME = "PAPILA"
DATASET_URL = "https://figshare.com/articles/dataset/PAPILA/14798004"
DATASET_LICENSE = "CC-BY-4.0"

# Excel has multi-row headers - data starts at row 4 (0-indexed row 3)
EXCEL_HEADER_ROW = 3  # Skip first 3 rows
EXCEL_DATA_START_ROW = 4  # Data starts at row 4

# Diagnosis code mapping
DIAGNOSIS_MAP = {
    0: "normal",      # Healthy
    1: "glaucoma",    # Glaucoma
    2: "suspicious",  # Suspicious
}


async def ingest_papila() -> OperationStatistics:
    """
    Main ingestion function for PAPILA dataset.
    
    The PAPILA dataset contains:
    - Excel files with patient demographics and clinical measurements (OD and OS separate)
    - Paired left/right fundus images per patient
    - Multi-expert segmentation contours for optic disc and optic cup (2 experts)
    
    Processing:
    - Clinical measurements → patient.comorbidities JSONB
    - Diagnosis code → multi-label classification (normal/glaucoma/suspicious)
    - Contour files → segmentation annotations with expert tracking
    
    Returns:
        OperationStatistics with success/error counts
    """
    data_root = get_data_root() / "09_PAPILA"
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
    
    # Step 2: Check paths
    clinical_data_dir = data_root / "ClinicalData"
    images_dir = data_root / "FundusImages"
    contours_dir = data_root / "ExpertsSegmentations" / "Contours"
    
    excel_od = clinical_data_dir / "patient_data_od.xlsx"
    excel_os = clinical_data_dir / "patient_data_os.xlsx"
    
    if not excel_od.exists() or not excel_os.exists():
        raise FileNotFoundError(f"Excel files not found in {clinical_data_dir}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not contours_dir.exists():
        raise FileNotFoundError(f"Contours directory not found: {contours_dir}")
    
    logger.info(f"Clinical data (OD): {excel_od}")
    logger.info(f"Clinical data (OS): {excel_os}")
    logger.info(f"Images directory: {images_dir}")
    logger.info(f"Contours directory: {contours_dir}")
    
    # Step 3: Discover images
    logger.info("Discovering fundus images...")
    image_files = await asyncio.to_thread(find_images, images_dir)
    logger.info(f"Found {len(image_files)} fundus images")
    
    # Step 4: Build image path map for quick lookup
    image_path_map: Dict[str, Path] = {}
    for img_path in image_files:
        image_path_map[img_path.stem] = img_path
    
    # Step 5: Process patient data and create image annotations
    # Total items: patients from both Excel files + images + segmentations
    # Estimate: ~244 patients + 488 images + ~1952 segmentations = ~2684 items
    total_images = len(image_files)
    total_contours = len(list(contours_dir.glob("*.txt")))
    total_items = total_images + total_contours
    
    tracker = ProgressTracker(
        total=total_items,
        description=f"Ingesting {DATASET_NAME}"
    )
    
    # Collections for bulk upsert
    all_patients: Dict[str, Patient] = {}  # patient_id_str -> Patient
    all_images: List[Image] = []
    all_patient_images: List[PatientImage] = []
    all_classifications: List[ClassificationAnnotation] = []
    image_labels: dict = {}  # image_id → diagnosis code for stratified splitting
    
    # Track image dimensions for segmentation processing
    image_dimensions: Dict[UUID, Tuple[int, int]] = {}  # image_id -> (width, height)
    
    # Step 6: Process both Excel files (OD and OS)
    logger.info("Processing patient data from Excel files...")
    
    async def process_patient_excel(
        excel_path: Path,
        laterality: str,
    ):
        """Process single Excel file (OD or OS)."""
        
        async def handle_patient_row(row, idx):
            patient_id_str = "unknown"  # Default for error reporting
            try:
                # Excel structure: First row is used as headers by read_excel_sheet()
                # Row will be a dict with column names as keys
                # But PAPILA has multi-row headers, so we need to access by actual column names
                # The reader will use row 1 as headers: ['Unnamed: 0', 'Age', 'Gender', ...]
                
                # Get first column (patient ID) - could be under 'Unnamed: 0' or 'ID'
                first_col_value = None
                for key in ['Unnamed: 0', 'ID']:
                    if key in row and row[key] is not None:
                        first_col_value = row[key]
                        break
                
                # Skip header rows, empty rows, or rows with 'ID' as value
                if first_col_value is None or str(first_col_value).strip() in ['ID', 'None', 'nan', '']:
                    return
                
                patient_id_str = str(first_col_value).strip()  # e.g., "#002", "#004"
                
                # Access Excel columns by header names
                age = row.get('Age') if row.get('Age') is not None else None
                gender_code = row.get('Gender')  # 0=male, 1=female
                diagnosis_code = row.get('Diagnosis') if row.get('Diagnosis') is not None else 0
                
                # Build clinical measurements for comorbidities JSONB
                comorbidities = {
                    "refractive_defect": {
                        "dioptre_1": row.get('Refractive_Defect'),
                        "dioptre_2": row.get('Unnamed: 5'),
                        "astigmatism": row.get('Unnamed: 6'),
                    },
                    "phakic_pseudophakic": row.get('Phakic/Pseudophakic'),
                    "iop": {
                        "pneumatic": row.get('IOP'),
                        "perkins": row.get('Unnamed: 9'),
                    },
                    "pachymetry": row.get('Pachymetry'),
                    "axial_length": row.get('Axial_Length'),
                    "vf_md": row.get('VF_MD'),
                }
                
                # Map gender code to sex
                sex = None
                if gender_code is not None:
                    sex = "male" if gender_code == 0 else "female"
                
                # Create or update patient (deduplicate across OD/OS)
                patient_id = generate_patient_uuid(dataset_id, patient_id_str)
                
                if patient_id_str not in all_patients:
                    patient = Patient(
                        patient_id=patient_id,
                        dataset_id=dataset_id,
                        original_patient_id=patient_id_str,
                        age=age,
                        sex=sex,
                        comorbidities=comorbidities,
                    )
                    all_patients[patient_id_str] = patient
                
                # Find corresponding image file
                # Image naming: RET{patient_number}{OD|OS}.jpg
                # Excel ID: "#002" -> Image: "RET002OD" or "RET002OS"
                patient_number = patient_id_str.lstrip("#")
                laterality_suffix = "OD" if laterality == "right" else "OS"
                image_stem = f"RET{patient_number}{laterality_suffix}"
                
                if image_stem not in image_path_map:
                    logger.warning(f"Image not found for patient {patient_id_str} ({laterality}): {image_stem}")
                    tracker.record_error(
                        error_type="file_not_found",
                        error_message=f"Image file not found: {image_stem}",
                        item_id=patient_id_str,
                    )
                    return
                
                image_path = image_path_map[image_stem]
                image_id = generate_image_uuid(dataset_id, image_stem)
                
                # Create image with auto-extracted metadata
                image = Image(
                    image_id=image_id,
                    dataset_id=dataset_id,
                    original_image_id=image_stem,
                    eye_laterality=laterality,
                    **get_image_metadata_dict(image_path),
                    modality="fundus",
                )
                all_images.append(image)
                image_labels[image_id] = diagnosis_code

                # Store dimensions for segmentation processing
                image_dimensions[image_id] = (image.resolution_width, image.resolution_height)
                
                # Link patient to image
                relationship_id = generate_patient_image_uuid(patient_id, image_id)
                patient_image = PatientImage(
                    relationship_id=relationship_id,
                    patient_id=patient_id,
                    image_id=image_id,
                )
                all_patient_images.append(patient_image)
                
                # Create multi-label classification for diagnosis
                # Map diagnosis code to boolean flags
                diagnosis_labels = {
                    "normal": diagnosis_code == 0,
                    "glaucoma": diagnosis_code == 1,
                    "glaucoma_suspicious": diagnosis_code == 2,
                }
                
                classifications = await process_classification(
                    class_value=diagnosis_labels,
                    task_type="multi_label",
                    task_name="papila_diagnosis",
                    class_name="glaucoma",
                    image_id=image_id,
                    annotation_method="manual",
                )
                all_classifications.extend(classifications)
                
                tracker.update(success=True)
                tracker.record_success("image")
                
            except Exception as e:
                tracker.update(success=False)
                tracker.record_error(
                    error_type="processing",
                    error_message=str(e),
                    item_id=patient_id_str if patient_id_str else "unknown",
                )
                logger.exception(f"Failed to process patient row {idx}: {e}")
        
        # Process Excel with automatic provenance
        # Note: Excel has multi-row headers, we need to handle this
        stats, raw_file_id, chain_id = await process_excel(
            excel_path=excel_path,
            dataset_id=dataset_id,
            unified_annotation_type="classification",
            process_row_fn=handle_patient_row,
            sheet_name=0,  # Use first sheet (default)
            progress_tracker=tracker,
            skip_errors=True,
        )
        
        return stats
    
    # Process both Excel files in parallel
    od_stats, os_stats = await asyncio.gather(
        process_patient_excel(excel_od, "right"),
        process_patient_excel(excel_os, "left"),
    )
    
    logger.info(f"Processed OD Excel: {od_stats.successful_items} successful, {od_stats.failed_items} failed")
    logger.info(f"Processed OS Excel: {os_stats.successful_items} successful, {os_stats.failed_items} failed")
    
    # Step 7: Bulk upsert patients, images, patient links, and classifications
    # NOTE: Must insert in correct order due to FK constraints
    logger.info(f"Upserting {len(all_patients)} patients...")
    await bulk_upsert_patients(list(all_patients.values()), batch_size=1000)
    
    logger.info(f"Upserting {len(all_images)} images...")
    await bulk_upsert_images(all_images, batch_size=1000)
    
    # Now insert patient links and classifications (both depend on images being inserted first)
    logger.info(f"Upserting {len(all_patient_images)} patient links and {len(all_classifications)} classifications...")
    await asyncio.gather(
        bulk_upsert_patient_images(all_patient_images, batch_size=1000),
        bulk_upsert_classification_annotations(all_classifications, batch_size=1000),
    )
    
    # Step 8: Register experts
    logger.info("Registering expert annotators...")
    expert_1_id = generate_expert_uuid(dataset_id, "Expert_1")
    expert_2_id = generate_expert_uuid(dataset_id, "Expert_2")
    
    expert_1 = Expert(
        expert_id=expert_1_id,
        expert_name="Expert_1",
        expertise_area="ophthalmology",
        dataset_id=dataset_id,
    )
    expert_2 = Expert(
        expert_id=expert_2_id,
        expert_name="Expert_2",
        expertise_area="ophthalmology",
        dataset_id=dataset_id,
    )
    
    await asyncio.gather(
        upsert_expert(expert_1),
        upsert_expert(expert_2),
    )
    
    # Step 9: Process multi-expert segmentation contours
    logger.info("Processing multi-expert segmentation contours...")
    logger.info(f"Found {total_contours} contour files")
    
    # Find all contour files
    contour_files = list(contours_dir.glob("*.txt"))
    
    # Process contours with limited concurrency to avoid memory issues
    semaphore = asyncio.Semaphore(10)  # Limit to 10 concurrent mask conversions
    
    async def process_contour_file(contour_path: Path):
        """Process single contour file."""
        async with semaphore:
            try:
                # Parse filename: RET{patient_number}{OD|OS}_{structure}_{expert}.txt
                # Example: RET004OD_disc_exp1.txt
                filename = contour_path.stem
                parts = filename.split("_")
                
                if len(parts) != 3:
                    logger.warning(f"Unexpected contour filename format: {filename}")
                    tracker.update(success=False)
                    tracker.record_error(
                        error_type="filename_parse",
                        error_message=f"Cannot parse filename: {filename}",
                        item_path=str(contour_path),
                    )
                    return
                
                image_stem = parts[0]  # "RET004OD"
                structure = parts[1]    # "disc" or "cup"
                expert_str = parts[2]   # "exp1" or "exp2"
                
                # Map to annotation type
                annotation_type = f"optic_{structure}"  # "optic_disc" or "optic_cup"
                
                # Map to expert ID
                expert_id = expert_1_id if expert_str == "exp1" else expert_2_id
                
                # Find corresponding image
                if image_stem not in image_path_map:
                    logger.warning(f"Image not found for contour: {filename}")
                    tracker.update(success=False)
                    tracker.record_error(
                        error_type="image_not_found",
                        error_message=f"Image not found for contour: {image_stem}",
                        item_path=str(contour_path),
                    )
                    return
                
                image_id = generate_image_uuid(dataset_id, image_stem)
                
                # Get image dimensions (extracted earlier)
                if image_id not in image_dimensions:
                    logger.warning(f"Image dimensions not found for {image_stem}")
                    tracker.update(success=False)
                    tracker.record_error(
                        error_type="missing_metadata",
                        error_message=f"Image dimensions not available: {image_stem}",
                        item_path=str(contour_path),
                    )
                    return
                
                image_size = image_dimensions[image_id]
                
                # Register contour file for provenance
                raw_file_id, chain_id = await register_individual_file(
                    file_path=contour_path,
                    dataset_id=dataset_id,
                    unified_annotation_type="segmentation",
                    file_type="txt",
                    
                )
                
                # Process contour to segmentation annotation first to get the segmentation details
                seg = await process_segmentation_from_contour(
                    contour_path=contour_path,
                    annotation_type=annotation_type,
                    image_id=image_id,
                    image_size=image_size,  # Use actual image dimensions
                    expert_annotation_id=None,  # Will set after creating expert annotation
                    annotation_method="manual",
                    raw_data_id=raw_file_id,
                    provenance_chain_id=chain_id,
                    dataset_name=DATASET_NAME
                )
                
                # Create expert annotation record with annotation_value containing segmentation info
                annotation_value = {
                    "image_id": str(image_id),
                    "segmentation_id": str(seg.segmentation_id),
                    "annotation_type": annotation_type,
                    "structure": structure,  # "disc" or "cup"
                    "image_stem": image_stem,
                    "contour_file": contour_path.name,
                }
                
                expert_annotation_id = generate_expert_annotation_uuid(
                    expert_id=expert_id,
                    annotation_task="segmentation",
                    raw_data_id=raw_file_id,
                    annotation_value_hash=None,
                )
                
                expert_annotation = ExpertAnnotation(
                    expert_annotation_id=expert_annotation_id,
                    expert_id=expert_id,
                    annotation_task="segmentation",
                    raw_data_id=raw_file_id,
                    annotation_value=annotation_value,  # Connect to the segmentation
                    confidence_level=None,
                    annotation_timestamp=None,
                )
                await upsert_expert_annotation(expert_annotation)
                
                # Update segmentation with expert_annotation_id
                seg.expert_annotation_id = expert_annotation_id
                
                # Upsert individual segmentation (no bulk operation available yet)
                await upsert_segmentation_annotation(seg)
                
                tracker.update(success=True)
                tracker.record_success("segmentation")
                
            except Exception as e:
                tracker.update(success=False)
                tracker.record_error(
                    error_type="segmentation_processing",
                    error_message=str(e),
                    item_path=str(contour_path),
                )
                logger.exception(f"Failed to process contour {contour_path.name}: {e}")
    
    # Process all contours in parallel (with semaphore limiting concurrency)
    await asyncio.gather(*[process_contour_file(cf) for cf in contour_files])
    
    # Step 10: Register splits — stratified 90/10 train+test, then 90/10 train+val
    all_image_ids_for_split = [img.image_id for img in all_images]
    if all_image_ids_for_split:
        logger.info("Registering dataset splits...")
        await auto_stratified_splits(
            dataset_id=dataset_id,
            split_assignments={"train": all_image_ids_for_split},
            labels=image_labels,
            split_type="explicit",
        )

    # Step 11: Finish and return statistics
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
    logger.info(f"Ingested {len(all_patients)} patients")
    logger.info(f"Ingested {len(all_images)} images")
    logger.info(f"Ingested {len(all_classifications)} classifications")
    logger.info(f"Ingested {total_contours} segmentation annotations (2 experts × 2 structures)")
    logger.info("=" * 80)
    
    return stats


async def main():
    """Entry point for script execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    try:
        stats = await ingest_papila()
        
        logger.info("=" * 80)
        logger.info(f"PAPILA ingestion completed successfully!")
        logger.info(f"Total: {stats.total_items}")
        logger.info(f"Successful: {stats.successful_items}")
        logger.info(f"Failed: {stats.failed_items}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.exception(f"Fatal error during PAPILA ingestion: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
