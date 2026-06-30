"""
Ingestion script for DeepDRiD dataset.

Dataset: DeepDRiD - Deep Diabetic Retinopathy Image Dataset
Structure: Complex multi-task dataset with CSV files
  - regular_fundus_images/ - Regular fundus images
    - regular-fundus-training/ - Training set with CSV
    - regular-fundus-validation/ - Validation set with CSV
    - Online-Challenge1&2-Evaluation/ - Challenge set
  - ultra-widefield_images/ - Ultra-widefield images
    - ultra-widefield-training/ - Training set with CSV
    - ultra-widefield-validation/ - Validation set with CSV
    - Online-Challenge3-Evaluation/ - Challenge set
  - CSV columns (regular): patient_id, image_id, image_path, Overall quality, 
    left_eye_DR_Level, right_eye_DR_Level, patient_DR_Level, Clarity, 
    Field definition, Artifact
  - CSV columns (ultra-widefield): patient_id, image_id, image_path, DR_level, position
  - Patient folders with 4 images per patient (l1, l2, r1, r2 for regular; 
    l1, l2, r1, r2 or variations for ultra-widefield)
Annotations: DR grading (per eye and patient level), Quality annotation, Laterality tracking
Tasks: Patient registration, DR grading (per eye and patient level), Quality annotation, Laterality tracking
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional
from uuid import UUID

from chaksudb.common.progress import ProgressTracker, OperationStatistics
from chaksudb.config.config import get_data_root
from chaksudb.db.models import Dataset, Image, DiseaseGrading, QualityAnnotation, Patient, PatientImage
from chaksudb.db.queries import (
    bulk_upsert_images,
    bulk_upsert_disease_gradings,
    bulk_upsert_quality_annotations,
    bulk_upsert_patients,
    bulk_upsert_patient_images,
    upsert_dataset,
)
from chaksudb.ingest.framework import (
    get_image_metadata_dict,
    process_csv,
    extract_laterality,
    read_csv_auto,
)
from chaksudb.ingest.framework.annotation_io import read_excel_sheet
from chaksudb.ingest.framework.gen_uuid import (
    generate_dataset_uuid,
    generate_image_uuid,
    generate_patient_uuid,
    generate_patient_image_uuid,
)
from chaksudb.ingest.framework.provenance_context import get_current_provenance
from chaksudb.ingest.framework.split_assigner import (
    register_standard_splits,
    bulk_assign_images_to_split,
)
from chaksudb.ingest.framework.task_processors.grading_processor import process_disease_grade
from chaksudb.ingest.framework.task_processors.quality_processor import process_quality_annotation

logger = logging.getLogger(__name__)

# Dataset metadata
DATASET_NAME = "DeepDRiD"
DATASET_URL = "https://github.com/deepdrdoc/DeepDRiD"
DATASET_LICENSE = "CC0: Public Domain"

# DR grading scale (DeepDRiD uses ICDR 0-4 scale)
DR_SCALE = "ICDR_0_4"


def parse_float_or_none(value: str) -> Optional[float]:
    """Parse a string value to float, returning None if empty or invalid."""
    if not value or value.strip() == "":
        return None
    try:
        return float(value.strip())
    except (ValueError, TypeError):
        return None


def parse_int_or_none(value: str) -> Optional[int]:
    """Parse a string value to int, returning None if empty or invalid."""
    if not value or value.strip() == "":
        return None
    try:
        return int(float(value.strip()))
    except (ValueError, TypeError):
        return None


async def process_deepdrid_quality_custom(
    image_id: UUID,
    overall_quality: Optional[int] = None,
    clarity: Optional[int] = None,
    field_definition: Optional[int] = None,
    artifact: Optional[int] = None,
    raw_data_id: Optional[UUID] = None,
) -> List[QualityAnnotation]:
    """
    Process DeepDRiD quality metrics with correct scales.
    
    DeepDRiD actual scales (from CSV data):
    - Overall Quality: 0 (poor), 1 (good), 2 (excellent) - scale 0-2
    - Clarity: 0-10 scale (higher = better)
    - Field Definition: 0-10 scale (higher = better)
    - Artifact: 0-10 scale (lower = better, 0=no artifact, 10=severe artifact)
    
    Args:
        image_id: UUID of the image
        overall_quality: Overall quality score (0-2)
        clarity: Clarity score (0-10)
        field_definition: Field definition score (0-10)
        artifact: Artifact score (0-10, inverted: lower is worse)
        raw_data_id: Optional raw annotation file UUID
    
    Returns:
        List of QualityAnnotation models
    """
    annotations = []
    
    if overall_quality is not None:
        label_map = {0: "poor", 1: "good", 2: "excellent"}
        annotations.append(
            await process_quality_annotation(
                quality_type="overall",
                image_id=image_id,
                quality_score=overall_quality,
                quality_label=label_map.get(overall_quality),
                scale_description="DeepDRiD Overall Quality (0=Poor, 1=Good, 2=Excellent)",
                scale_min=0,
                scale_max=2,
                raw_data_id=raw_data_id,
            )
        )
    
    if clarity is not None:
        annotations.append(
            await process_quality_annotation(
                quality_type="clarity",
                image_id=image_id,
                quality_score=clarity,
                scale_description="DeepDRiD Clarity (0-10, higher=better)",
                scale_min=0,
                scale_max=10,
                raw_data_id=raw_data_id,
            )
        )
    
    if field_definition is not None:
        annotations.append(
            await process_quality_annotation(
                quality_type="field_definition",
                image_id=image_id,
                quality_score=field_definition,
                scale_description="DeepDRiD Field Definition (0-10, higher=better)",
                scale_min=0,
                scale_max=10,
                raw_data_id=raw_data_id,
            )
        )
    
    if artifact is not None:
        # Note: Artifact scale is 0-10 where lower is better (0=no artifact, 10=severe)
        # Store raw value - can normalize later if needed
        annotations.append(
            await process_quality_annotation(
                quality_type="artifact",
                image_id=image_id,
                quality_score=artifact,
                scale_description="DeepDRiD Artifact (0-10, lower=better, 0=no artifact, 10=severe)",
                scale_min=0,
                scale_max=10,
                raw_data_id=raw_data_id,
            )
        )
    
    return annotations


async def ingest_deepdrid() -> OperationStatistics:
    """
    Main ingestion function for DeepDRiD dataset.
    
    The dataset contains:
    - Regular fundus images with DR grading (per eye and patient level) and quality annotations
    - Ultra-widefield images with DR grading (per image) and position information
    - Patient folders with multiple images per patient (4 images for regular, 2 for ultra-widefield)
    
    Returns:
        OperationStatistics with success/error counts
    """
    data_root = get_data_root() / "37_DeepDRiD"
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
        modality_types=["fundus", "uwf"],  # Regular fundus and ultra-widefield images
    )
    await upsert_dataset(dataset)
    
    # Step 2: Count total CSV rows for progress tracking
    logger.info("Counting CSV rows...")
    total_count = 0
    
    regular_data_root = data_root / "regular_fundus_images"
    ultra_data_root = data_root / "ultra-widefield_images"
    
    # Count regular fundus training CSV
    regular_train_csv = regular_data_root / "regular-fundus-training" / "regular-fundus-training.csv"
    if regular_train_csv.exists():
        rows = await asyncio.to_thread(read_csv_auto, regular_train_csv)
        total_count += len(rows)
        logger.info(f"  regular-fundus-training: {len(rows)} rows")
    
    # Count regular fundus validation CSV
    regular_val_csv = regular_data_root / "regular-fundus-validation" / "regular-fundus-validation.csv"
    if regular_val_csv.exists():
        rows = await asyncio.to_thread(read_csv_auto, regular_val_csv)
        total_count += len(rows)
        logger.info(f"  regular-fundus-validation: {len(rows)} rows")
    
    # Count ultra-widefield training CSV
    ultra_train_csv = ultra_data_root / "ultra-widefield-training" / "ultra-widefield-training.csv"
    if ultra_train_csv.exists():
        rows = await asyncio.to_thread(read_csv_auto, ultra_train_csv)
        total_count += len(rows)
        logger.info(f"  ultra-widefield-training: {len(rows)} rows")
    
    # Count ultra-widefield validation CSV
    ultra_val_csv = ultra_data_root / "ultra-widefield-validation" / "ultra-widefield-validation.csv"
    if ultra_val_csv.exists():
        rows = await asyncio.to_thread(read_csv_auto, ultra_val_csv)
        total_count += len(rows)
        logger.info(f"  ultra-widefield-validation: {len(rows)} rows")
    
    logger.info(f"Total CSV rows to process: {total_count}")
    
    # Step 3: Setup progress tracker
    tracker = ProgressTracker(
        total=total_count,
        description=f"Ingesting {DATASET_NAME}"
    )
    
    # Step 4: Collect items for bulk upsert
    all_images: List[Image] = []
    all_dr_gradings: List[DiseaseGrading] = []
    all_quality_annotations: List[QualityAnnotation] = []
    all_patients: List[Patient] = []
    all_patient_image_pairs: List[tuple[UUID, UUID]] = []
    image_to_split: Dict[UUID, str] = {}
    
    # Track patients to avoid duplicates
    patient_lookup: Dict[str, UUID] = {}  # original_patient_id -> patient_id
    
    # Step 5: Process regular fundus images
    logger.info("Processing regular fundus images...")
    
    regular_data_root = data_root / "regular_fundus_images"
    
    # Process regular fundus training set
    regular_train_csv = regular_data_root / "regular-fundus-training" / "regular-fundus-training.csv"
    if regular_train_csv.exists():
        logger.info(f"Processing regular fundus training CSV: {regular_train_csv}")
        
        async def process_regular_row(row, idx):
            """Process a single row from regular fundus CSV."""
            try:
                # Get provenance from context
                raw_data_id, provenance_chain_id = get_current_provenance()
                
                # Extract patient ID
                original_patient_id = str(row.get("patient_id", "")).strip()
                if not original_patient_id:
                    logger.warning(f"Row {idx}: Missing patient_id")
                    tracker.update(success=False)
                    tracker.record_error(
                        error_type="missing_patient_id",
                        error_message="Missing patient_id in CSV row",
                        item_id=f"row_{idx}",
                    )
                    return
                
                # Register patient if not already registered
                patient_id = None
                if original_patient_id not in patient_lookup:
                    # DeepDRiD doesn't provide patient metadata in CSV, so we only register
                    # the patient ID (no age, sex, etc.)
                    patient_id = generate_patient_uuid(dataset_id, original_patient_id)
                    patient = Patient(
                        patient_id=patient_id,
                        dataset_id=dataset_id,
                        original_patient_id=original_patient_id,
                    )
                    all_patients.append(patient)
                    patient_lookup[original_patient_id] = patient_id
                else:
                    patient_id = patient_lookup[original_patient_id]
                
                # Extract image information
                image_id_str = str(row.get("image_id", "")).strip()
                if not image_id_str:
                    logger.warning(f"Row {idx}: Missing image_id")
                    tracker.update(success=False)
                    tracker.record_error(
                        error_type="missing_image_id",
                        error_message="Missing image_id in CSV row",
                        item_id=f"row_{idx}",
                    )
                    return
                
                # Find image file
                image_path_str = str(row.get("image_path", "")).strip()
                # Handle Windows path separators
                image_path_str = image_path_str.replace("\\", "/")
                # Remove leading slash if present
                if image_path_str.startswith("/"):
                    image_path_str = image_path_str[1:]
                
                # Try multiple path resolutions
                image_path = None
                
                # Try 1: Path from CSV (relative to regular_data_root)
                candidate = regular_data_root / image_path_str
                if candidate.exists():
                    image_path = candidate
                
                # Try 2: Path with Images subdirectory (actual structure)
                if image_path is None:
                    # Extract directory and filename from CSV path
                    # CSV path format: regular-fundus-training\3\3_l1.jpg
                    parts = image_path_str.split("/")
                    if len(parts) >= 2:
                        # parts[0] = "regular-fundus-training", parts[-1] = "3_l1.jpg"
                        # Try: regular-fundus-training/Images/3/3_l1.jpg
                        split_name = parts[0]  # e.g., "regular-fundus-training"
                        filename = parts[-1]  # e.g., "3_l1.jpg"
                        candidate = regular_data_root / split_name / "Images" / original_patient_id / filename
                        if candidate.exists():
                            image_path = candidate
                
                # Try 3: Direct path in Images subdirectory
                if image_path is None:
                    candidate = regular_data_root / "regular-fundus-training" / "Images" / original_patient_id / f"{image_id_str}.jpg"
                    if candidate.exists():
                        image_path = candidate
                
                if image_path is None:
                    logger.warning(f"Row {idx}: Image not found: {image_path_str} (tried multiple paths)")
                    tracker.update(success=False)
                    tracker.record_error(
                        error_type="file_not_found",
                        error_message=f"Image not found: {image_path_str}",
                        item_id=image_id_str,
                    )
                    return
                
                # Generate relative path from data root for UUID and original_image_id
                # Use forward slashes for consistency across platforms
                rel_path = image_path.relative_to(data_root)
                original_image_id = str(rel_path).replace("\\", "/")
                
                # Generate image ID from full relative path
                image_id = generate_image_uuid(dataset_id, original_image_id)
                
                # Extract laterality from image_id (format: {patient_id}_l1, {patient_id}_l2, {patient_id}_r1, {patient_id}_r2)
                laterality = extract_laterality(image_id_str)
                
                # Create image with automatic metadata extraction
                image = Image(
                    image_id=image_id,
                    dataset_id=dataset_id,
                    original_image_id=original_image_id,
                    **get_image_metadata_dict(image_path),
                    modality="fundus",
                    eye_laterality=laterality,
                )
                all_images.append(image)
                image_to_split[image_id] = "train"
                
                # Link patient to image
                all_patient_image_pairs.append((patient_id, image_id))
                
                # Process DR grading for left eye
                left_eye_dr = parse_float_or_none(str(row.get("left_eye_DR_Level", "")))
                if left_eye_dr is not None and laterality == "left":
                    dr_grading = await process_disease_grade(
                        grade_value=int(left_eye_dr),
                        disease_type="DR",
                        scale_name=DR_SCALE,
                        image_id=image_id,
                        raw_data_id=raw_data_id,
                        provenance_chain_id=provenance_chain_id,
                        annotation_method="manual",
                    )
                    all_dr_gradings.append(dr_grading)
                
                # Process DR grading for right eye
                right_eye_dr = parse_float_or_none(str(row.get("right_eye_DR_Level", "")))
                if right_eye_dr is not None and laterality == "right":
                    dr_grading = await process_disease_grade(
                        grade_value=int(right_eye_dr),
                        disease_type="DR",
                        scale_name=DR_SCALE,
                        image_id=image_id,
                        raw_data_id=raw_data_id,
                        provenance_chain_id=provenance_chain_id,
                        annotation_method="manual",
                    )
                    all_dr_gradings.append(dr_grading)
                
                # Process quality annotations
                overall_quality = parse_int_or_none(str(row.get("Overall quality", "")))
                clarity = parse_int_or_none(str(row.get("Clarity", "")))
                field_definition = parse_int_or_none(str(row.get("Field definition", "")))
                artifact = parse_int_or_none(str(row.get("Artifact", "")))
                
                if overall_quality is not None or clarity is not None or field_definition is not None or artifact is not None:
                    quality_annotations = await process_deepdrid_quality_custom(
                        image_id=image_id,
                        overall_quality=overall_quality,
                        clarity=clarity,
                        field_definition=field_definition,
                        artifact=artifact,
                        raw_data_id=raw_data_id,
                    )
                    all_quality_annotations.extend(quality_annotations)
                
                tracker.update(success=True)
                tracker.record_success("image")
                
            except Exception as e:
                logger.error(f"Row {idx}: Error processing row: {e}", exc_info=True)
                tracker.update(success=False)
                tracker.record_error(
                    error_type="row_processing_error",
                    error_message=str(e),
                    item_id=f"row_{idx}",
                )
        
        await process_csv(
            csv_path=regular_train_csv,
            dataset_id=dataset_id,
            unified_annotation_type="grading",
            process_row_fn=process_regular_row,
            progress_tracker=tracker,
            skip_errors=True,
        )
    
    # Process regular fundus validation set
    regular_val_csv = regular_data_root / "regular-fundus-validation" / "regular-fundus-validation.csv"
    if regular_val_csv.exists():
        logger.info(f"Processing regular fundus validation CSV: {regular_val_csv}")
        
        async def process_regular_val_row(row, idx):
            """Process a single row from regular fundus validation CSV."""
            try:
                # Get provenance from context
                raw_data_id, provenance_chain_id = get_current_provenance()
                
                # Extract patient ID
                original_patient_id = str(row.get("patient_id", "")).strip()
                if not original_patient_id:
                    logger.warning(f"Row {idx}: Missing patient_id")
                    tracker.update(success=False)
                    tracker.record_error(
                        error_type="missing_patient_id",
                        error_message="Missing patient_id in CSV row",
                        item_id=f"row_{idx}",
                    )
                    return
                
                # Register patient if not already registered
                patient_id = None
                if original_patient_id not in patient_lookup:
                    patient_id = generate_patient_uuid(dataset_id, original_patient_id)
                    patient = Patient(
                        patient_id=patient_id,
                        dataset_id=dataset_id,
                        original_patient_id=original_patient_id,
                    )
                    all_patients.append(patient)
                    patient_lookup[original_patient_id] = patient_id
                else:
                    patient_id = patient_lookup[original_patient_id]
                
                # Extract image information
                image_id_str = str(row.get("image_id", "")).strip()
                if not image_id_str:
                    logger.warning(f"Row {idx}: Missing image_id")
                    tracker.update(success=False)
                    tracker.record_error(
                        error_type="missing_image_id",
                        error_message="Missing image_id in CSV row",
                        item_id=f"row_{idx}",
                    )
                    return
                
                # Find image file
                image_path_str = str(row.get("image_path", "")).strip()
                # Handle Windows path separators
                image_path_str = image_path_str.replace("\\", "/")
                # Remove leading slash if present
                if image_path_str.startswith("/"):
                    image_path_str = image_path_str[1:]
                
                # Try multiple path resolutions
                image_path = None
                
                # Try 1: Path from CSV (relative to regular_data_root)
                candidate = regular_data_root / image_path_str
                if candidate.exists():
                    image_path = candidate
                
                # Try 2: Path with Images subdirectory (actual structure)
                if image_path is None:
                    # Extract directory and filename from CSV path
                    parts = image_path_str.split("/")
                    if len(parts) >= 2:
                        split_name = parts[0]  # e.g., "regular-fundus-validation"
                        filename = parts[-1]  # e.g., "309_l1.jpg"
                        candidate = regular_data_root / split_name / "Images" / original_patient_id / filename
                        if candidate.exists():
                            image_path = candidate
                
                # Try 3: Direct path in Images subdirectory
                if image_path is None:
                    candidate = regular_data_root / "regular-fundus-validation" / "Images" / original_patient_id / f"{image_id_str}.jpg"
                    if candidate.exists():
                        image_path = candidate
                
                if image_path is None:
                    logger.warning(f"Row {idx}: Image not found: {image_path_str} (tried multiple paths)")
                    tracker.update(success=False)
                    tracker.record_error(
                        error_type="file_not_found",
                        error_message=f"Image not found: {image_path_str}",
                        item_id=image_id_str,
                    )
                    return
                
                # Generate relative path from data root for UUID and original_image_id
                # Use forward slashes for consistency across platforms
                rel_path = image_path.relative_to(data_root)
                original_image_id = str(rel_path).replace("\\", "/")
                
                # Generate image ID from full relative path
                image_id = generate_image_uuid(dataset_id, original_image_id)
                
                # Extract laterality from image_id (format: {patient_id}_l1, {patient_id}_l2, {patient_id}_r1, {patient_id}_r2)
                laterality = extract_laterality(image_id_str)
                
                # Create image with automatic metadata extraction
                image = Image(
                    image_id=image_id,
                    dataset_id=dataset_id,
                    original_image_id=original_image_id,
                    **get_image_metadata_dict(image_path),
                    modality="fundus",
                    eye_laterality=laterality,
                )
                all_images.append(image)
                image_to_split[image_id] = "val"  # Validation split
                
                # Link patient to image
                all_patient_image_pairs.append((patient_id, image_id))
                
                # Process DR grading for left eye
                left_eye_dr = parse_float_or_none(str(row.get("left_eye_DR_Level", "")))
                if left_eye_dr is not None and laterality == "left":
                    dr_grading = await process_disease_grade(
                        grade_value=int(left_eye_dr),
                        disease_type="DR",
                        scale_name=DR_SCALE,
                        image_id=image_id,
                        raw_data_id=raw_data_id,
                        provenance_chain_id=provenance_chain_id,
                        annotation_method="manual",
                    )
                    all_dr_gradings.append(dr_grading)
                
                # Process DR grading for right eye
                right_eye_dr = parse_float_or_none(str(row.get("right_eye_DR_Level", "")))
                if right_eye_dr is not None and laterality == "right":
                    dr_grading = await process_disease_grade(
                        grade_value=int(right_eye_dr),
                        disease_type="DR",
                        scale_name=DR_SCALE,
                        image_id=image_id,
                        raw_data_id=raw_data_id,
                        provenance_chain_id=provenance_chain_id,
                        annotation_method="manual",
                    )
                    all_dr_gradings.append(dr_grading)
                
                # Process quality annotations
                overall_quality = parse_int_or_none(str(row.get("Overall quality", "")))
                clarity = parse_int_or_none(str(row.get("Clarity", "")))
                field_definition = parse_int_or_none(str(row.get("Field definition", "")))
                artifact = parse_int_or_none(str(row.get("Artifact", "")))
                
                if overall_quality is not None or clarity is not None or field_definition is not None or artifact is not None:
                    quality_annotations = await process_deepdrid_quality_custom(
                        image_id=image_id,
                        overall_quality=overall_quality,
                        clarity=clarity,
                        field_definition=field_definition,
                        artifact=artifact,
                        raw_data_id=raw_data_id,
                    )
                    all_quality_annotations.extend(quality_annotations)
                
                tracker.update(success=True)
                tracker.record_success("image")
                
            except Exception as e:
                logger.error(f"Row {idx}: Error processing row: {e}", exc_info=True)
                tracker.update(success=False)
                tracker.record_error(
                    error_type="row_processing_error",
                    error_message=str(e),
                    item_id=f"row_{idx}",
                )
        
        await process_csv(
            csv_path=regular_val_csv,
            dataset_id=dataset_id,
            unified_annotation_type="grading",
            process_row_fn=process_regular_val_row,
            progress_tracker=tracker,
            skip_errors=True,
        )
    
    # Step 6: Process ultra-widefield images
    logger.info("Processing ultra-widefield images...")
    
    ultra_data_root = data_root / "ultra-widefield_images"
    
    # Process ultra-widefield training set
    ultra_train_csv = ultra_data_root / "ultra-widefield-training" / "ultra-widefield-training.csv"
    if ultra_train_csv.exists():
        logger.info(f"Processing ultra-widefield training CSV: {ultra_train_csv}")
        
        async def process_ultra_row(row, idx):
            """Process a single row from ultra-widefield CSV."""
            try:
                # Get provenance from context
                raw_data_id, provenance_chain_id = get_current_provenance()
                
                # Extract patient ID
                original_patient_id = str(row.get("patient_id", "")).strip()
                if not original_patient_id:
                    logger.warning(f"Row {idx}: Missing patient_id")
                    tracker.update(success=False)
                    tracker.record_error(
                        error_type="missing_patient_id",
                        error_message="Missing patient_id in CSV row",
                        item_id=f"row_{idx}",
                    )
                    return
                
                # Register patient if not already registered
                patient_id = None
                if original_patient_id not in patient_lookup:
                    patient_id = generate_patient_uuid(dataset_id, original_patient_id)
                    patient = Patient(
                        patient_id=patient_id,
                        dataset_id=dataset_id,
                        original_patient_id=original_patient_id,
                    )
                    all_patients.append(patient)
                    patient_lookup[original_patient_id] = patient_id
                else:
                    patient_id = patient_lookup[original_patient_id]
                
                # Extract image information
                image_id_str = str(row.get("image_id", "")).strip()
                if not image_id_str:
                    logger.warning(f"Row {idx}: Missing image_id")
                    tracker.update(success=False)
                    tracker.record_error(
                        error_type="missing_image_id",
                        error_message="Missing image_id in CSV row",
                        item_id=f"row_{idx}",
                    )
                    return
                
                # Find image file
                image_path_str = str(row.get("image_path", "")).strip()
                # Handle Windows path separators
                image_path_str = image_path_str.replace("\\", "/")
                # Remove leading slash if present
                if image_path_str.startswith("/"):
                    image_path_str = image_path_str[1:]
                
                # Try multiple path resolutions
                image_path = None
                
                # Try 1: Path from CSV (relative to ultra_data_root)
                candidate = ultra_data_root / image_path_str
                if candidate.exists():
                    image_path = candidate
                
                # Try 2: Path with Images subdirectory (actual structure)
                if image_path is None:
                    # Extract directory and filename from CSV path
                    parts = image_path_str.split("/")
                    if len(parts) >= 2:
                        # parts[0] = "ultra-widefield-training", parts[-1] = "2_r1.jpg"
                        split_name = parts[0]  # e.g., "ultra-widefield-training"
                        filename = parts[-1]  # e.g., "2_r1.jpg"
                        candidate = ultra_data_root / split_name / "Images" / original_patient_id / filename
                        if candidate.exists():
                            image_path = candidate
                
                # Try 3: Direct path in Images subdirectory
                if image_path is None:
                    candidate = ultra_data_root / "ultra-widefield-training" / "Images" / original_patient_id / f"{image_id_str}.jpg"
                    if candidate.exists():
                        image_path = candidate
                
                if image_path is None:
                    logger.warning(f"Row {idx}: Image not found: {image_path_str} (tried multiple paths)")
                    tracker.update(success=False)
                    tracker.record_error(
                        error_type="file_not_found",
                        error_message=f"Image not found: {image_path_str}",
                        item_id=image_id_str,
                    )
                    return
                
                # Generate relative path from data root for UUID and original_image_id
                # Use forward slashes for consistency across platforms
                rel_path = image_path.relative_to(data_root)
                original_image_id = str(rel_path).replace("\\", "/")
                
                # Generate image ID from full relative path
                image_id = generate_image_uuid(dataset_id, original_image_id)
                
                # Extract laterality from position column or image_id
                laterality = None
                position = str(row.get("position", "")).strip().lower()
                if position == "left_eye":
                    laterality = "left"
                elif position == "right_eye":
                    laterality = "right"
                else:
                    # Fallback to extracting from image_id
                    laterality = extract_laterality(image_id_str)
                
                # Create image with automatic metadata extraction
                image = Image(
                    image_id=image_id,
                    dataset_id=dataset_id,
                    original_image_id=original_image_id,
                    **get_image_metadata_dict(image_path),
                    modality="uwf",  # Ultra-widefield modality
                    eye_laterality=laterality,
                )
                all_images.append(image)
                image_to_split[image_id] = "train"
                
                # Link patient to image
                all_patient_image_pairs.append((patient_id, image_id))
                
                # Process DR grading (ultra-widefield has single DR_level per image)
                dr_level = parse_float_or_none(str(row.get("DR_level", "")))
                if dr_level is not None:
                    dr_grading = await process_disease_grade(
                        grade_value=int(dr_level),
                        disease_type="DR",
                        scale_name=DR_SCALE,
                        image_id=image_id,
                        raw_data_id=raw_data_id,
                        provenance_chain_id=provenance_chain_id,
                        annotation_method="manual",
                    )
                    all_dr_gradings.append(dr_grading)
                
                tracker.update(success=True)
                tracker.record_success("image")
                
            except Exception as e:
                logger.error(f"Row {idx}: Error processing row: {e}", exc_info=True)
                tracker.update(success=False)
                tracker.record_error(
                    error_type="row_processing_error",
                    error_message=str(e),
                    item_id=f"row_{idx}",
                )
        
        await process_csv(
            csv_path=ultra_train_csv,
            dataset_id=dataset_id,
            unified_annotation_type="grading",
            process_row_fn=process_ultra_row,
            progress_tracker=tracker,
            skip_errors=True,
        )
    
    # Process ultra-widefield validation set
    ultra_val_csv = ultra_data_root / "ultra-widefield-validation" / "ultra-widefield-validation.csv"
    if ultra_val_csv.exists():
        logger.info(f"Processing ultra-widefield validation CSV: {ultra_val_csv}")
        
        async def process_ultra_val_row(row, idx):
            """Process a single row from ultra-widefield validation CSV."""
            try:
                # Get provenance from context
                raw_data_id, provenance_chain_id = get_current_provenance()
                
                # Extract patient ID
                original_patient_id = str(row.get("patient_id", "")).strip()
                if not original_patient_id:
                    logger.warning(f"Row {idx}: Missing patient_id")
                    tracker.update(success=False)
                    tracker.record_error(
                        error_type="missing_patient_id",
                        error_message="Missing patient_id in CSV row",
                        item_id=f"row_{idx}",
                    )
                    return
                
                # Register patient if not already registered
                patient_id = None
                if original_patient_id not in patient_lookup:
                    patient_id = generate_patient_uuid(dataset_id, original_patient_id)
                    patient = Patient(
                        patient_id=patient_id,
                        dataset_id=dataset_id,
                        original_patient_id=original_patient_id,
                    )
                    all_patients.append(patient)
                    patient_lookup[original_patient_id] = patient_id
                else:
                    patient_id = patient_lookup[original_patient_id]
                
                # Extract image information
                image_id_str = str(row.get("image_id", "")).strip()
                if not image_id_str:
                    logger.warning(f"Row {idx}: Missing image_id")
                    tracker.update(success=False)
                    tracker.record_error(
                        error_type="missing_image_id",
                        error_message="Missing image_id in CSV row",
                        item_id=f"row_{idx}",
                    )
                    return
                
                # Find image file
                image_path_str = str(row.get("image_path", "")).strip()
                # Handle Windows path separators
                image_path_str = image_path_str.replace("\\", "/")
                # Remove leading slash if present
                if image_path_str.startswith("/"):
                    image_path_str = image_path_str[1:]
                
                # Try multiple path resolutions
                image_path = None
                
                # Try 1: Path from CSV (relative to ultra_data_root)
                candidate = ultra_data_root / image_path_str
                if candidate.exists():
                    image_path = candidate
                
                # Try 2: Path with Images subdirectory (actual structure)
                if image_path is None:
                    # Extract directory and filename from CSV path
                    parts = image_path_str.split("/")
                    if len(parts) >= 2:
                        split_name = parts[0]  # e.g., "ultra-widefield-validation"
                        filename = parts[-1]  # e.g., "35_r1.jpg"
                        candidate = ultra_data_root / split_name / "Images" / original_patient_id / filename
                        if candidate.exists():
                            image_path = candidate
                
                # Try 3: Direct path in Images subdirectory
                if image_path is None:
                    candidate = ultra_data_root / "ultra-widefield-validation" / "Images" / original_patient_id / f"{image_id_str}.jpg"
                    if candidate.exists():
                        image_path = candidate
                
                if image_path is None:
                    logger.warning(f"Row {idx}: Image not found: {image_path_str} (tried multiple paths)")
                    tracker.update(success=False)
                    tracker.record_error(
                        error_type="file_not_found",
                        error_message=f"Image not found: {image_path_str}",
                        item_id=image_id_str,
                    )
                    return
                
                # Generate relative path from data root for UUID and original_image_id
                # Use forward slashes for consistency across platforms
                rel_path = image_path.relative_to(data_root)
                original_image_id = str(rel_path).replace("\\", "/")
                
                # Generate image ID from full relative path
                image_id = generate_image_uuid(dataset_id, original_image_id)
                
                # Extract laterality from position column or image_id
                laterality = None
                position = str(row.get("position", "")).strip().lower()
                if position == "left_eye":
                    laterality = "left"
                elif position == "right_eye":
                    laterality = "right"
                else:
                    # Fallback to extracting from image_id
                    laterality = extract_laterality(image_id_str)
                
                # Create image with automatic metadata extraction
                image = Image(
                    image_id=image_id,
                    dataset_id=dataset_id,
                    original_image_id=original_image_id,
                    **get_image_metadata_dict(image_path),
                    modality="uwf",  # Ultra-widefield modality
                    eye_laterality=laterality,
                )
                all_images.append(image)
                image_to_split[image_id] = "val"  # Validation split
                
                # Link patient to image
                all_patient_image_pairs.append((patient_id, image_id))
                
                # Process DR grading (ultra-widefield has single DR_level per image)
                dr_level = parse_float_or_none(str(row.get("DR_level", "")))
                if dr_level is not None:
                    dr_grading = await process_disease_grade(
                        grade_value=int(dr_level),
                        disease_type="DR",
                        scale_name=DR_SCALE,
                        image_id=image_id,
                        raw_data_id=raw_data_id,
                        provenance_chain_id=provenance_chain_id,
                        annotation_method="manual",
                    )
                    all_dr_gradings.append(dr_grading)
                
                tracker.update(success=True)
                tracker.record_success("image")
                
            except Exception as e:
                logger.error(f"Row {idx}: Error processing row: {e}", exc_info=True)
                tracker.update(success=False)
                tracker.record_error(
                    error_type="row_processing_error",
                    error_message=str(e),
                    item_id=f"row_{idx}",
                )
        
        await process_csv(
            csv_path=ultra_val_csv,
            dataset_id=dataset_id,
            unified_annotation_type="grading",
            process_row_fn=process_ultra_val_row,
            progress_tracker=tracker,
            skip_errors=True,
        )
    
    # Step 6b: Process regular fundus Online-Challenge1&2-Evaluation (labels from xlsx)
    eval_regular_root = regular_data_root / "Online-Challenge1&2-Evaluation"
    eval_regular_images_dir = eval_regular_root / "Images"
    challenge1_xlsx = eval_regular_root / "Challenge1_labels.xlsx"
    challenge2_xlsx = eval_regular_root / "Challenge2_labels.xlsx"
    if eval_regular_images_dir.exists() and challenge1_xlsx.exists():
        logger.info("Processing regular fundus Online-Challenge1&2-Evaluation...")
        c1_rows = await asyncio.to_thread(read_excel_sheet, challenge1_xlsx, sheet=0)
        c2_rows = await asyncio.to_thread(read_excel_sheet, challenge2_xlsx, sheet=0) if challenge2_xlsx.exists() else []
        dr_by_image: Dict[str, int] = {str(r.get("image_id", "")).strip(): int(r.get("DR_Levels", 0)) for r in c1_rows if r.get("image_id") and str(r.get("DR_Levels", "")).strip() != ""}
        quality_by_image: Dict[str, Dict] = {}
        for r in c2_rows:
            iid = str(r.get("image_id", "")).strip()
            if not iid:
                continue
            quality_by_image[iid] = {
                "Overall quality": parse_int_or_none(str(r.get("Overall quality", ""))),
                "Clarity": parse_int_or_none(str(r.get("Clarity", ""))),
                "Field definition": parse_int_or_none(str(r.get("Field definition", ""))),
                "Artifact": parse_int_or_none(str(r.get("Artifact", ""))),
            }
        for patient_dir in sorted(eval_regular_images_dir.iterdir()):
            if not patient_dir.is_dir():
                continue
            original_patient_id = patient_dir.name
            if original_patient_id not in patient_lookup:
                patient_id = generate_patient_uuid(dataset_id, original_patient_id)
                all_patients.append(Patient(patient_id=patient_id, dataset_id=dataset_id, original_patient_id=original_patient_id))
                patient_lookup[original_patient_id] = patient_id
            patient_id = patient_lookup[original_patient_id]
            for jpg_path in sorted(patient_dir.glob("*.jpg")):
                image_id_str = jpg_path.stem
                if image_id_str not in dr_by_image and image_id_str not in quality_by_image:
                    continue
                rel_path = jpg_path.relative_to(data_root)
                original_image_id = str(rel_path).replace("\\", "/")
                image_id = generate_image_uuid(dataset_id, original_image_id)
                laterality = extract_laterality(image_id_str)
                image = Image(
                    image_id=image_id,
                    dataset_id=dataset_id,
                    original_image_id=original_image_id,
                    **get_image_metadata_dict(jpg_path),
                    modality="fundus",
                    eye_laterality=laterality,
                )
                all_images.append(image)
                image_to_split[image_id] = "test"
                all_patient_image_pairs.append((patient_id, image_id))
                raw_data_id, provenance_chain_id = get_current_provenance()
                if image_id_str in dr_by_image:
                    dr_val = dr_by_image[image_id_str]
                    all_dr_gradings.append(
                        await process_disease_grade(
                            grade_value=dr_val,
                            disease_type="DR",
                            scale_name=DR_SCALE,
                            image_id=image_id,
                            raw_data_id=raw_data_id,
                            provenance_chain_id=provenance_chain_id,
                            annotation_method="manual",
                        )
                    )
                q = quality_by_image.get(image_id_str, {})
                if q.get("Overall quality") is not None or q.get("Clarity") is not None or q.get("Field definition") is not None or q.get("Artifact") is not None:
                    all_quality_annotations.extend(
                        await process_deepdrid_quality_custom(
                            image_id=image_id,
                            overall_quality=q.get("Overall quality"),
                            clarity=q.get("Clarity"),
                            field_definition=q.get("Field definition"),
                            artifact=q.get("Artifact"),
                            raw_data_id=raw_data_id,
                        )
                    )
                tracker.update(success=True)
        logger.info(f"Regular Eval: {len(dr_by_image)} DR labels, {len(quality_by_image)} quality labels")
    
    # Step 6c: Process ultra-widefield Online-Challenge3-Evaluation (labels from xlsx)
    eval_uwf_root = ultra_data_root / "Online-Challenge3-Evaluation"
    eval_uwf_images_dir = eval_uwf_root / "Images"
    challenge3_xlsx = eval_uwf_root / "Challenge3_labels.xlsx"
    if eval_uwf_images_dir.exists() and challenge3_xlsx.exists():
        logger.info("Processing ultra-widefield Online-Challenge3-Evaluation...")
        c3_rows = await asyncio.to_thread(read_excel_sheet, challenge3_xlsx, sheet=0)
        uwf_dr_by_image: Dict[str, int] = {}
        for r in c3_rows:
            iid = str(r.get("image_id", "")).strip()
            val = r.get("UWF_DR_Levels")
            if iid and val is not None and str(val).strip() != "":
                uwf_dr_by_image[iid] = int(float(val))
        for patient_dir in sorted(eval_uwf_images_dir.iterdir()):
            if not patient_dir.is_dir():
                continue
            original_patient_id = patient_dir.name
            if original_patient_id not in patient_lookup:
                patient_id = generate_patient_uuid(dataset_id, original_patient_id)
                all_patients.append(Patient(patient_id=patient_id, dataset_id=dataset_id, original_patient_id=original_patient_id))
                patient_lookup[original_patient_id] = patient_id
            patient_id = patient_lookup[original_patient_id]
            for jpg_path in sorted(patient_dir.glob("*.jpg")):
                image_id_str = jpg_path.stem
                if image_id_str not in uwf_dr_by_image:
                    continue
                rel_path = jpg_path.relative_to(data_root)
                original_image_id = str(rel_path).replace("\\", "/")
                image_id = generate_image_uuid(dataset_id, original_image_id)
                laterality = extract_laterality(image_id_str)
                image = Image(
                    image_id=image_id,
                    dataset_id=dataset_id,
                    original_image_id=original_image_id,
                    **get_image_metadata_dict(jpg_path),
                    modality="uwf",
                    eye_laterality=laterality,
                )
                all_images.append(image)
                image_to_split[image_id] = "test"
                all_patient_image_pairs.append((patient_id, image_id))
                raw_data_id, provenance_chain_id = get_current_provenance()
                dr_val = uwf_dr_by_image[image_id_str]
                all_dr_gradings.append(
                    await process_disease_grade(
                        grade_value=dr_val,
                        disease_type="DR",
                        scale_name=DR_SCALE,
                        image_id=image_id,
                        raw_data_id=raw_data_id,
                        provenance_chain_id=provenance_chain_id,
                        annotation_method="manual",
                    )
                )
                tracker.update(success=True)
        logger.info(f"UWF Eval: {len(uwf_dr_by_image)} DR labels")
    
    # Step 7: Bulk upsert all items
    logger.info("Bulk upserting items...")
    
    # Upsert patients
    if all_patients:
        await bulk_upsert_patients(all_patients, batch_size=1000)
        logger.info(f"Upserted {len(all_patients)} patients")
    
    # Upsert images
    if all_images:
        await bulk_upsert_images(all_images, batch_size=1000)
        logger.info(f"Upserted {len(all_images)} images")
    
    # Upsert patient-image links
    if all_patient_image_pairs:
        from datetime import datetime
        patient_image_models = []
        for patient_id, image_id in all_patient_image_pairs:
            relationship_id = generate_patient_image_uuid(patient_id, image_id)
            patient_image = PatientImage(
                relationship_id=relationship_id,
                patient_id=patient_id,
                image_id=image_id,
                exam_date=None,
                created_at=datetime.now(),
            )
            patient_image_models.append(patient_image)
        
        await bulk_upsert_patient_images(patient_image_models, batch_size=1000)
        logger.info(f"Linked {len(all_patient_image_pairs)} patient-image pairs")
    
    # Upsert DR gradings
    if all_dr_gradings:
        await bulk_upsert_disease_gradings(all_dr_gradings, batch_size=1000)
        logger.info(f"Upserted {len(all_dr_gradings)} DR gradings")
    
    # Upsert quality annotations
    if all_quality_annotations:
        await bulk_upsert_quality_annotations(all_quality_annotations, batch_size=1000)
        logger.info(f"Upserted {len(all_quality_annotations)} quality annotations")
    
    # Step 8: Register splits and assign images
    logger.info("Registering dataset splits...")
    
    # Count images per split
    train_image_ids = [
        img_id for img_id, split in image_to_split.items() if split == "train"
    ]
    val_image_ids = [
        img_id for img_id, split in image_to_split.items() if split == "val"
    ]
    test_image_ids = [
        img_id for img_id, split in image_to_split.items() if split == "test"
    ]
    
    splits = await register_standard_splits(
        dataset_id=dataset_id,
        split_type="explicit",
        train_count=len(train_image_ids),
        test_count=len(test_image_ids),
        val_count=len(val_image_ids),
    )
    
    # Assign images to splits
    if train_image_ids:
        await bulk_assign_images_to_split(train_image_ids, splits["train"])
        logger.info(f"Assigned {len(train_image_ids)} images to train split")
    
    if val_image_ids:
        await bulk_assign_images_to_split(val_image_ids, splits["val"])
        logger.info(f"Assigned {len(val_image_ids)} images to val split")
    
    if test_image_ids:
        await bulk_assign_images_to_split(test_image_ids, splits["test"])
        logger.info(f"Assigned {len(test_image_ids)} images to test split")
    
    tracker.finish()
    final_stats = tracker.get_statistics()
    
    # Final summary
    logger.info("=" * 80)
    logger.info("Ingestion Summary:")
    logger.info(f"  Total items: {final_stats.total_items}")
    logger.info(f"  Successful: {final_stats.successful_items}")
    logger.info(f"  Failed: {final_stats.failed_items}")
    logger.info(f"  Skipped: {final_stats.skipped_items}")
    logger.info(f"  Patients: {len(all_patients)}")
    logger.info(f"  Images: {len(all_images)}")
    logger.info(f"  Patient-image links: {len(all_patient_image_pairs)}")
    logger.info(f"  DR gradings: {len(all_dr_gradings)}")
    logger.info(f"  Quality annotations: {len(all_quality_annotations)}")
    logger.info(f"  Train images: {len(train_image_ids)}")
    logger.info(f"  Val images: {len(val_image_ids)}")
    logger.info(f"  Test images: {len(test_image_ids)}")
    if final_stats.errors:
        logger.warning(f"  Total errors: {len(final_stats.errors)}")
        for error_type, count in final_stats.error_counts.items():
            logger.warning(f"    {error_type}: {count}")
    logger.info("=" * 80)
    
    return final_stats


async def main():
    """Entry point for script execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    try:
        stats = await ingest_deepdrid()
        
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
