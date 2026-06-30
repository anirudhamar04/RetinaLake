"""
Ingestion script for BRSET dataset.

Dataset: Brazilian Retinal Fundus Image Dataset
Structure: labels_brset.csv with image_id, patient_id, and multiple annotation columns
Annotations: 
  - Multi-label classifications (diabetic_retinopathy, macular_edema, scar, nevus, amd, etc.)
  - DR grading (SDRG and ICDR scales)
  - Quality annotations
Tasks: Patient registration, Multi-label classification, DR grading (SDRG and ICDR), Quality annotation
"""

import asyncio
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional
from uuid import UUID

from chaksudb.common.progress import ProgressTracker, OperationStatistics
from chaksudb.config.config import get_data_root
from chaksudb.db.models import (
    ClassificationAnnotation,
    Dataset,
    DiseaseGrading,
    GradingScaleMapping,
    Image,
    Patient,
    PatientImage,
    QualityAnnotation,
)
from chaksudb.db.queries import (
    bulk_upsert_classification_annotations as bulk_upsert_classifications,
    bulk_upsert_images,
    bulk_upsert_disease_gradings,
    bulk_upsert_patient_images,
    bulk_upsert_patients,
    bulk_upsert_quality_annotations,
    upsert_dataset,
)
from chaksudb.db.queries.grading import (
    find_grading_scale_by_id,
    find_grading_scale_mapping_to_standard,
    upsert_grading_scale,
    upsert_grading_scale_mapping,
)
from chaksudb.ingest.framework import (
    process_csv,
    read_csv_auto,
    get_image_metadata_dict,
)
from chaksudb.ingest.framework.gen_uuid import (
    generate_dataset_uuid,
    generate_patient_image_uuid,
    generate_grading_scale_mapping_uuid,
    generate_grading_scale_uuid,
    generate_image_uuid,
    generate_patient_uuid,
)
from chaksudb.ingest.framework.patient_register import register_patient, link_patient_to_image
from chaksudb.ingest.framework.provenance_context import get_current_provenance
from chaksudb.ingest.framework.scale_bootstrap.bootstrap_scale_mappings import validate_mappings
from chaksudb.ingest.framework.split_assigner import auto_stratified_splits
from chaksudb.ingest.framework.task_processors.classification_processor import process_classification
from chaksudb.ingest.framework.task_processors.grading_processor import (
    get_or_create_scale,
    process_disease_grade,
)
from chaksudb.ingest.framework.task_processors.quality_processor import process_quality_annotation

logger = logging.getLogger(__name__)

# Dataset metadata
DATASET_NAME = "BRSET"
DATASET_URL = "https://physionet.org/content/brazilian-ophthalmological/1.0.1"
DATASET_LICENSE = "CC-BY-4.0"

# Disease columns for multi-label classification
DISEASE_COLUMNS = [
    "diabetic_retinopathy",
    "macular_edema",
    "scar",
    "nevus",
    "amd",
    "vascular_occlusion",
    "hypertensive_retinopathy",
    "drusens",
    "hemorrhage",
    "retinal_detachment",
    "myopic_fundus",
    "increased_cup_disc",
    "other",
]

# Overall quality column (Adequate/Inadequate)
QUALITY_COLUMN = "quality"

# Per-parameter image quality columns (1=normal, 2=abnormal) -> canonical quality_type.
# Maps BRSET's acquisition-quality parameters onto the quality_annotations vocabulary.
QUALITY_PARAM_COLUMNS = {
    "focus": "clarity",
    "Illuminaton": "illumination",   # note: BRSET's column is misspelled
    "image_field": "field_definition",
    "artifacts": "artifact",
}

# Anatomical normal/abnormal parameters (1=normal, 2=abnormal). These are structural
# findings, not image quality and not diseases -> stored as binary classification tasks.
ANATOMICAL_PARAM_COLUMNS = ["optic_disc", "vessels", "macula"]


async def bootstrap_scottish_scale_mappings(
    csv_path: Path,
    dataset_id: UUID,
) -> Optional[UUID]:
    """
    Bootstrap Scottish scale mappings from BRSET CSV if mappings don't exist.
    
    Analyzes DR_SDRG (which uses Scottish scale) and DR_ICDR columns to learn
    mappings, then stores mappings to ICDR_0_4. The Scottish scale itself should
    already be registered by bootstrap_grading_scales.py.
    
    Args:
        csv_path: Path to labels_brset.csv
        dataset_id: Dataset UUID
        
    Returns:
        Scottish scale_id if mappings were bootstrapped, None otherwise
    """
    # Check if Scottish scale exists (should be registered by bootstrap script)
    scottish_scale_id = generate_grading_scale_uuid("Scottish", "DR")
    existing_scale = await find_grading_scale_by_id(scottish_scale_id)
    
    if not existing_scale:
        logger.warning("Scottish scale not found. Run bootstrap_grading_scales.py first.")
        return None
    
    logger.info("Scottish scale found, checking for existing mappings...")
    
    # Get ICDR_0_4 scale ID
    icdr_scale_id = generate_grading_scale_uuid("ICDR_0_4", "DR")
    icdr_scale = await find_grading_scale_by_id(icdr_scale_id)
    
    if not icdr_scale:
        logger.warning("ICDR_0_4 scale not found, cannot bootstrap mappings")
        return None
    
    # Check if mappings already exist
    test_mapping = await find_grading_scale_mapping_to_standard(
        scottish_scale_id,
        "0",
        "ICDR_0_4"
    )
    if test_mapping:
        logger.info("Scottish->ICDR_0_4 mappings already exist, skipping bootstrap")
        return scottish_scale_id
    
    logger.info("No existing mappings found, bootstrapping from BRSET data...")
    
    # Analyze CSV to learn mappings
    # Format: source_scale -> target_scale -> source_value -> [target_values]
    mapping_observations: Dict[str, Dict[str, Dict[str, List[int]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    
    rows = await asyncio.to_thread(read_csv_auto, csv_path)
    
    for row in rows:
        sdrg_grade = row.get("DR_SDRG", "").strip()  # SDRG uses Scottish scale
        icdr_grade = row.get("DR_ICDR", "").strip()
        
        if not sdrg_grade or not icdr_grade:
            continue
        
        try:
            scottish_val = str(int(float(sdrg_grade)))
            icdr_val = int(float(icdr_grade))
            mapping_observations["Scottish"]["ICDR_0_4"][scottish_val].append(icdr_val)
        except (ValueError, TypeError):
            continue
    
    if not mapping_observations or "Scottish" not in mapping_observations:
        logger.warning("No valid Scottish->ICDR mappings found in CSV")
        return None
    
    # Validate mappings using bootstrap logic
    validated = validate_mappings(mapping_observations)
    
    # Store mappings
    mapping_count = 0
    if "Scottish" in validated and "ICDR_0_4" in validated["Scottish"]:
        mappings = validated["Scottish"]["ICDR_0_4"]
        
        for source_value, (target_value, confidence) in mappings.items():
            mapping_id = generate_grading_scale_mapping_uuid(
                scottish_scale_id,
                icdr_scale_id,
                source_value,
            )
            
            mapping = GradingScaleMapping(
                mapping_id=mapping_id,
                source_scale_id=scottish_scale_id,
                target_scale_id=icdr_scale_id,
                source_value=source_value,
                target_value=target_value,
                mapping_confidence=confidence,
            )
            
            await upsert_grading_scale_mapping(mapping)
            mapping_count += 1
            logger.debug(
                f"Stored mapping: Scottish:{source_value} -> ICDR_0_4:{target_value} "
                f"(confidence: {confidence})"
            )
        
        logger.info(f"Stored {mapping_count} Scottish->ICDR_0_4 mappings")
    
    return scottish_scale_id


async def ingest_brset() -> OperationStatistics:
    """
    Main ingestion function for BRSET dataset.
    
    Returns:
        OperationStatistics with success/error counts
    """
    data_root = get_data_root() / "27_BRSET"
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
    
    # Step 2: Bootstrap Scottish scale mappings if needed
    csv_path = data_root / "labels_brset.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    logger.info("Checking Scottish scale mappings...")
    await bootstrap_scottish_scale_mappings(csv_path, dataset_id)
    
    # Step 3: Count total rows for progress tracking
    logger.info("Counting CSV rows...")
    rows = await asyncio.to_thread(read_csv_auto, csv_path)
    total_count = len(rows)
    logger.info(f"Found {total_count} images")
    
    # Step 4: Setup progress tracker
    tracker = ProgressTracker(
        total=total_count,
        description=f"Ingesting {DATASET_NAME}"
    )
    
    # Collect items for bulk upsert
    all_images: List[Image] = []
    all_patients: List[Patient] = []
    all_patient_image_pairs: List[tuple[UUID, UUID]] = []
    all_classifications: List[ClassificationAnnotation] = []
    all_dr_gradings: List[DiseaseGrading] = []
    all_quality_annotations: List[QualityAnnotation] = []
    image_to_split: Dict[UUID, str] = {}  # For split assignment
    image_labels: dict = {}  # image_id → DR ICDR grade for stratified splitting
    
    # Track patients to avoid duplicates
    patient_lookup: Dict[str, UUID] = {}  # original_patient_id -> patient_id
    
    async def process_row(row, idx):
        """Process a single CSV row."""
        try:
            image_id_str = row["image_id"]
            image_id = generate_image_uuid(dataset_id, image_id_str)
            
            # Find image file - try different extensions
            image_dir = data_root / "fundus_photos"
            image_path = None
            
            for ext in [".jpg", ".jpeg", ".png"]:
                candidate = image_dir / f"{image_id_str}{ext}"
                if await asyncio.to_thread(candidate.exists):
                    image_path = candidate
                    break
            
            if not image_path:
                tracker.record_error(
                    error_type="file_not_found",
                    error_message=f"Image not found: {image_id_str}",
                    item_id=image_id_str,
                )
                tracker.update(success=False)
                return
            
            # Extract laterality from exam_eye (1=right, 2=left, or other encoding)
            laterality = None
            exam_eye = row.get("exam_eye", "").strip()
            if exam_eye == "1":
                laterality = "right"
            elif exam_eye == "2":
                laterality = "left"
            
            # Create image with automatic metadata extraction
            image = Image(
                image_id=image_id,
                dataset_id=dataset_id,
                original_image_id=image_id_str,
                **get_image_metadata_dict(image_path),
                modality="fundus",
                eye_laterality=laterality,
            )
            all_images.append(image)
            image_to_split[image_id] = "train"  # BRSET doesn't have explicit splits
            
            # Register patient if we have patient_id
            patient_id = None
            original_patient_id = row.get("patient_id", "").strip()
            if original_patient_id:
                if original_patient_id not in patient_lookup:
                    # Extract patient metadata
                    age = None
                    try:
                        age_val = row.get("patient_age", "").strip()
                        if age_val:
                            age = int(float(age_val))
                    except (ValueError, TypeError):
                        pass
                    
                    sex = None
                    sex_val = row.get("patient_sex", "").strip()
                    if sex_val == "1":
                        sex = "male"
                    elif sex_val == "2":
                        sex = "female"
                    
                    nationality = row.get("nationality", "").strip() or None
                    
                    # Build comorbidities dict
                    comorbidities = {}
                    if row.get("diabetes", "").strip().lower() == "yes":
                        comorbidities["diabetes"] = True
                    if row.get("comorbidities", "").strip():
                        comorbidities["comorbidities"] = row.get("comorbidities", "").strip()
                    if row.get("diabetes_time_y", "").strip():
                        try:
                            comorbidities["diabetes_duration_years"] = int(
                                float(row.get("diabetes_time_y", "").strip())
                            )
                        except (ValueError, TypeError):
                            pass
                    if row.get("insuline", "").strip().lower() == "yes":
                        comorbidities["insulin"] = True
                    
                    patient_id = generate_patient_uuid(dataset_id, original_patient_id)
                    patient = Patient(
                        patient_id=patient_id,
                        dataset_id=dataset_id,
                        original_patient_id=original_patient_id,
                        age=age,
                        sex=sex,
                        nationality=nationality,
                        comorbidities=comorbidities if comorbidities else None,
                    )
                    all_patients.append(patient)
                    patient_lookup[original_patient_id] = patient_id
                else:
                    patient_id = patient_lookup[original_patient_id]
                
                # Link patient to image
                all_patient_image_pairs.append((patient_id, image_id))
            
            # Get provenance from context
            raw_data_id, provenance_chain_id = get_current_provenance()
            
            # Process multi-label classification for diseases
            disease_labels = {}
            for disease_col in DISEASE_COLUMNS:
                value = row.get(disease_col, "").strip()
                if value:
                    try:
                        # Convert to int (0/1) or keep as is
                        int_val = int(float(value))
                        disease_labels[disease_col] = int_val
                    except (ValueError, TypeError):
                        # Skip invalid values
                        continue
            
            if disease_labels:
                classifications = await process_classification(
                    class_value=disease_labels,
                    task_type="multi_label",
                    task_name="disease_panel",
                    class_name="disease_panel",
                    image_id=image_id,
                    raw_data_id=raw_data_id,
                    provenance_chain_id=provenance_chain_id,
                    annotation_method="manual",
                )
                all_classifications.extend(classifications)
            
            # Store ONLY the ICDR grade. DR_SDRG (Scottish) is not stored per-image — the
            # Scottish<->ICDR relationship is captured once in grading_scale_mappings (used
            # for bootstrap/normalization), so storing both scales per image is redundant.
            # ICDR is the canonical scale, so the trigger sets scaled_grade = original_grade.
            icdr_grade = row.get("DR_ICDR", "").strip()
            if icdr_grade:
                try:
                    icdr_value = int(float(icdr_grade))
                    icdr_grading = await process_disease_grade(
                        grade_value=icdr_value,
                        disease_type="DR",
                        scale_name="ICDR_0_4",
                        image_id=image_id,
                        raw_data_id=raw_data_id,
                        provenance_chain_id=provenance_chain_id,
                        annotation_method="manual",
                    )
                    all_dr_gradings.append(icdr_grading)
                    image_labels[image_id] = icdr_value
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid ICDR grade '{icdr_grade}' for {image_id_str}: {e}")
            
            # Process quality annotation
            quality_value = row.get(QUALITY_COLUMN, "").strip()
            if quality_value:
                # BRSET quality is categorical: "Adequate", etc.
                quality_annotation = await process_quality_annotation(
                    quality_type="overall",
                    image_id=image_id,
                    quality_label=quality_value,
                    scale_description="BRSET quality assessment",
                    raw_data_id=raw_data_id,
                    provenance_chain_id=provenance_chain_id,
                )
                all_quality_annotations.append(quality_annotation)

            # Per-parameter acquisition quality (focus/illumination/field/artifacts).
            for col, qtype in QUALITY_PARAM_COLUMNS.items():
                raw = row.get(col, "").strip()
                if raw in ("1", "2"):
                    all_quality_annotations.append(await process_quality_annotation(
                        quality_type=qtype,
                        image_id=image_id,
                        quality_label="normal" if raw == "1" else "abnormal",
                        scale_description="BRSET per-parameter quality (1=normal, 2=abnormal)",
                        raw_data_id=raw_data_id,
                        provenance_chain_id=provenance_chain_id,
                    ))

            # Anatomical normal/abnormal findings -> binary classification per structure.
            for col in ANATOMICAL_PARAM_COLUMNS:
                raw = row.get(col, "").strip()
                if raw in ("1", "2"):
                    all_classifications.extend(await process_classification(
                        class_value=(raw == "2"),  # abnormal == positive
                        task_type="binary",
                        task_name=f"{col}_abnormality",
                        class_name=f"{col}_abnormality",
                        image_id=image_id,
                        class_labels={1: "abnormal", 0: "normal"},
                        raw_data_id=raw_data_id,
                        provenance_chain_id=provenance_chain_id,
                        annotation_method="manual",
                    ))

            tracker.update(count=1, success=True)
            
        except Exception as e:
            tracker.update(count=1, success=False)
            tracker.record_error(
                error_type="processing",
                error_message=str(e),
                item_id=row.get("image_id", ""),
            )
            logger.error(f"Failed to process row {idx}: {e}")
    
    # Step 5: Process CSV with provenance tracking
    logger.info("Processing annotations...")
    # BRSET primary annotation type is "grading" (DR grading is the main task)
    stats, raw_file_id, chain_id = await process_csv(
        csv_path,
        dataset_id,
        "grading",  # Primary annotation type
        process_row
    )
    
    # Log provenance information
    logger.info(f"CSV registered: raw_file_id={raw_file_id}, chain_id={chain_id}")
    
    # Step 6: Bulk upsert - images first, then patients, then annotations
    logger.info(f"Upserting {len(all_images)} images...")
    await bulk_upsert_images(all_images, batch_size=1000)
    
    if all_patients:
        logger.info(f"Upserting {len(all_patients)} patients...")
        await bulk_upsert_patients(all_patients, batch_size=1000)
    
    if all_patient_image_pairs:
        logger.info(f"Linking {len(all_patient_image_pairs)} patient-image pairs...")
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
        
        await bulk_upsert_patient_images(patient_image_models)
    
    logger.info(
        f"Upserting {len(all_classifications)} classifications, "
        f"{len(all_dr_gradings)} DR gradings, and "
        f"{len(all_quality_annotations)} quality annotations..."
    )
    await asyncio.gather(
        bulk_upsert_classifications(all_classifications, batch_size=1000),
        bulk_upsert_disease_gradings(all_dr_gradings, batch_size=1000),
        bulk_upsert_quality_annotations(all_quality_annotations, batch_size=1000),
    )
    
    # Step 7: Register splits — stratified 90/10 train+test, then 90/10 train+val
    logger.info("Registering dataset splits...")
    train_image_ids = list(image_to_split.keys())
    if train_image_ids:
        await auto_stratified_splits(
            dataset_id=dataset_id,
            split_assignments={"train": train_image_ids},
            labels=image_labels,
            split_type="undefined",
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
        stats = await ingest_brset()
        
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
