"""
Ingestion script for RFMiD dataset.

Dataset: Retinal Fundus Multi-disease Image Dataset (RFMiD)
Structure: CSV files with binary labels for 46 retinal conditions
Annotations: Classification (DR, ARMD), Patient comorbidities (44 other conditions)
Tasks: Binary classification (DR, ARMD), Patient metadata (comorbidities)
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from uuid import UUID

from chaksudb.common.progress import ProgressTracker, OperationStatistics
from chaksudb.config.config import get_data_root
from chaksudb.db.models import Dataset, Image, ClassificationAnnotation, Patient, PatientImage
from chaksudb.db.queries import (
    upsert_dataset,
    bulk_upsert_images,
    bulk_upsert_classification_annotations,
    bulk_upsert_patients,
    bulk_upsert_patient_images,
)
from chaksudb.ingest.framework import (
    find_images,
    process_csv,
    read_csv_auto,
    get_image_metadata_dict,
)
from chaksudb.ingest.framework.gen_uuid import (
    generate_dataset_uuid,
    generate_image_uuid,
    generate_patient_uuid,
    generate_patient_image_uuid,
)
from chaksudb.ingest.framework.task_processors.classification_processor import process_classification
from chaksudb.ingest.framework.split_assigner import (
    register_standard_splits,
    bulk_assign_images_to_split,
)

logger = logging.getLogger(__name__)

# Dataset metadata
DATASET_NAME = "RFMiD"
DATASET_URL = "https://ieee-dataport.org/open-access/retinal-fundus-multi-disease-image-dataset-rfmid"
DATASET_LICENSE = "CC-BY-4.0"

# Full condition names mapping (from documentation)
CONDITION_NAMES = {
    "Disease_Risk": "Disease Risk",
    "DR": "DR",  
    "ARMD": "AMD",  
    "MH": "Macular Hole",
    "DN": "Drusen",
    "MYA": "Myopia-related retinal changes",
    "BRVO": "Branch Retinal Vein Occlusion",
    "TSLN": "Tessellation",
    "ERM": "Epiretinal Membrane",
    "LS": "Laser Scars",
    "MS": "Macular Scar",
    "CSR": "Central Serous Retinopathy",
    "ODC": "Optic Disc Cupping",
    "CRVO": "Central Retinal Vein Occlusion",
    "TV": "Tortuous Vessels",
    "AH": "Asteroid Hyalosis",
    "ODP": "Optic Disc Pallor",
    "ODE": "Optic Disc Edema",
    "ST": "Subretinal Tissue or Scar",
    "AION": "Anterior Ischemic Optic Neuropathy",
    "PT": "Papillitis",
    "RT": "Retinal Tear",
    "RS": "Retinal Scar",
    "CRS": "Chorioretinal Scar",
    "EDN": "Exudation",
    "RPEC": "Retinal Pigment Epithelial Changes",
    "MHL": "Lamellar Macular Hole",
    "RP": "Retinitis Pigmentosa",
    "CWS": "Cotton Wool Spots",
    "CB": "Coloboma",
    "ODPM": "Optic Disc Pit Maculopathy",
    "PRH": "Preretinal Hemorrhage",
    "MNF": "Myelinated Nerve Fibers",
    "HR": "Hemorrhage",
    "CRAO": "Central Retinal Artery Occlusion",
    "TD": "Tractional Detachment",
    "CME": "Cystoid Macular Edema",
    "PTCR": "Post-Traumatic Chorioretinopathy",
    "CF": "Chorioretinal Fibrosis",
    "VH": "Vitreous Hemorrhage",
    "MCA": "Microaneurysm",
    "VS": "Vessel Sheathing",
    "BRAO": "Branch Retinal Artery Occlusion",
    "PLQ": "Plaque",
    "HPED": "Hemorrhagic Pigment Epithelial Detachment",
    "CL": "Chorioretinal Lesion",
}

# Conditions to store as classification annotations
CLASSIFICATION_CONDITIONS = ["DR", "ARMD"]

# All other conditions (excluding ID, Disease_Risk, DR, ARMD) go to comorbidities
COMORBIDITY_CONDITIONS = [
    key for key in CONDITION_NAMES.keys()
    if key not in CLASSIFICATION_CONDITIONS + ["ID"]
]


async def ingest_rfmid() -> OperationStatistics:
    """
    Main ingestion function for RFMiD dataset.
    
    Returns:
        OperationStatistics with success/error counts
    """
    data_root = get_data_root() / "04_RFMid"
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
    
    # Step 2: Count total rows for progress tracking
    logger.info("Counting CSV rows...")
    train_csv_path = data_root / "RFMiD_Training_Labels.csv"
    test_csv_path = data_root / "RFMiD_Testing_Labels.csv"
    val_csv_path = data_root / "RFMiD_Validation_Labels.csv"
    
    train_rows = await asyncio.to_thread(read_csv_auto, train_csv_path)
    test_rows = await asyncio.to_thread(read_csv_auto, test_csv_path)
    val_rows = await asyncio.to_thread(read_csv_auto, val_csv_path)
    total_count = len(train_rows) + len(test_rows) + len(val_rows)
    
    logger.info(f"Found {len(train_rows)} training images, {len(test_rows)} test images, {len(val_rows)} validation images")
    
    # Step 3: Setup progress tracker
    tracker = ProgressTracker(
        total=total_count,
        description=f"Ingesting {DATASET_NAME}"
    )
    
    # Collect items for bulk upsert
    all_images: List[Image] = []
    all_classifications: List[ClassificationAnnotation] = []
    all_patients: List[Patient] = []
    all_patient_images: List[PatientImage] = []
    image_to_split: Dict[UUID, str] = {}  # For split assignment
    
    async def process_row(row, idx, split_name):
        """Process a single CSV row."""
        try:
            image_id_str = str(row["ID"])
            image_id = generate_image_uuid(dataset_id, image_id_str)
            
            # Find image file (images are named <ID>.png)
            image_dir = data_root / split_name
            image_path = image_dir / f"{image_id_str}.png"
            
            if not await asyncio.to_thread(image_path.exists):
                tracker.record_error(
                    error_type="file_not_found",
                    error_message=f"Image not found: {image_path}",
                    item_id=image_id_str,
                )
                tracker.update(success=False)
                return
            
            # Create image with automatic metadata extraction
            image = Image(
                image_id=image_id,
                dataset_id=dataset_id,
                original_image_id=image_id_str,
                **get_image_metadata_dict(image_path),
                modality="fundus",
            )
            all_images.append(image)
            image_to_split[image_id] = split_name
            
            # Standard: a multi-disease dataset is ONE multi_label panel (sub_key per
            # disease), not N independent binary rows. RFMiD assesses every finding per
            # image, so the whole panel is stored together; concepts (DR/AMD/...) are
            # derived per sub_key for cross-dataset concept queries.
            disease_labels = {
                code: bool(int(row[code]))
                for code in CONDITION_NAMES
                if code != "Disease_Risk" and code in row
            }
            if disease_labels:
                all_classifications.extend(await process_classification(
                    class_value=disease_labels,
                    task_type="multi_label",
                    task_name="disease_panel",
                    class_name="disease_panel",
                    image_id=image_id,
                    annotation_method="manual",  # Adjudicated consensus of two senior experts
                ))

            # "Disease_Risk" is a meta "any abnormality" flag, not a specific disease.
            if "Disease_Risk" in row:
                all_classifications.extend(await process_classification(
                    class_value=bool(int(row["Disease_Risk"])),
                    task_type="binary",
                    task_name="disease_risk",
                    class_name="disease_risk",
                    image_id=image_id,
                    annotation_method="manual",
                ))
            
            # Create patient record with comorbidities
            # Use image ID as patient ID since RFMiD is one image per patient
            patient_id = generate_patient_uuid(dataset_id, image_id_str)
            
            # Build comorbidities dictionary with full condition names
            comorbidities = {}
            for condition_code in COMORBIDITY_CONDITIONS:
                if condition_code in row:
                    full_name = CONDITION_NAMES[condition_code]
                    comorbidities[full_name] = bool(int(row[condition_code]))
            
            # Add Disease_Risk to comorbidities
            if "Disease_Risk" in row:
                comorbidities[CONDITION_NAMES["Disease_Risk"]] = bool(int(row["Disease_Risk"]))
            
            # Create patient model
            patient = Patient(
                patient_id=patient_id,
                dataset_id=dataset_id,
                original_patient_id=image_id_str,
                comorbidities=comorbidities,
                created_at=datetime.now(),
            )
            all_patients.append(patient)
            
            # Link patient to image
            relationship_id = generate_patient_image_uuid(patient_id, image_id)
            patient_image = PatientImage(
                relationship_id=relationship_id,
                patient_id=patient_id,
                image_id=image_id,
                created_at=datetime.now(),
            )
            all_patient_images.append(patient_image)
            
            tracker.update(count=1, success=True)
            
        except Exception as e:
            tracker.update(count=1, success=False)
            tracker.record_error(
                error_type="processing",
                error_message=str(e),
                item_id=row.get("ID"),
            )
            logger.error(f"Failed to process row {idx}: {e}")
    
    # Step 4: Process all three CSVs in parallel
    logger.info("Processing annotations...")
    
    # Create separate handlers for each split
    async def train_handler(row, idx):
        await process_row(row, idx, "Training")
    
    async def test_handler(row, idx):
        await process_row(row, idx, "Testing")
    
    async def val_handler(row, idx):
        await process_row(row, idx, "Validation")
    
    # Process all CSVs in parallel with provenance tracking
    # RFMiD primary annotation type is "classification" (multi-disease binary classification)
    train_results, test_results, val_results = await asyncio.gather(
        process_csv(
            train_csv_path,
            dataset_id,
            "classification",  # Primary annotation type
            train_handler,
            progress_tracker=tracker,
        ),
        process_csv(
            test_csv_path,
            dataset_id,
            "classification",
            test_handler,
            progress_tracker=tracker,
        ),
        process_csv(
            val_csv_path,
            dataset_id,
            "classification",
            val_handler,
            progress_tracker=tracker,
        ),
    )
    
    # Log provenance information
    train_stats, train_raw_id, train_chain_id = train_results
    test_stats, test_raw_id, test_chain_id = test_results
    val_stats, val_raw_id, val_chain_id = val_results
    logger.info(f"Training CSV registered: raw_file_id={train_raw_id}, chain_id={train_chain_id}")
    logger.info(f"Testing CSV registered: raw_file_id={test_raw_id}, chain_id={test_chain_id}")
    logger.info(f"Validation CSV registered: raw_file_id={val_raw_id}, chain_id={val_chain_id}")
    
    # Step 5: Bulk upsert in proper order (respecting foreign key constraints)
    logger.info(f"Upserting {len(all_images)} images...")
    await bulk_upsert_images(all_images, batch_size=1000)
    
    logger.info(f"Upserting {len(all_patients)} patients...")
    await bulk_upsert_patients(all_patients, batch_size=1000)
    
    logger.info(f"Upserting {len(all_classifications)} classifications...")
    await bulk_upsert_classification_annotations(all_classifications, batch_size=1000)
    
    logger.info(f"Upserting {len(all_patient_images)} patient-image links...")
    await bulk_upsert_patient_images(all_patient_images, batch_size=1000)
    
    # Step 6: Register splits and assign images
    logger.info("Registering dataset splits...")
    train_image_ids = [img_id for img_id, split in image_to_split.items() if split == "Training"]
    test_image_ids = [img_id for img_id, split in image_to_split.items() if split == "Testing"]
    val_image_ids = [img_id for img_id, split in image_to_split.items() if split == "Validation"]
    
    splits = await register_standard_splits(
        dataset_id=dataset_id,
        split_type="explicit",
        train_count=len(train_image_ids),
        val_count=len(val_image_ids),
        test_count=len(test_image_ids),
    )
    
    # Assign images to splits
    await asyncio.gather(
        bulk_assign_images_to_split(train_image_ids, splits["train"]),
        bulk_assign_images_to_split(val_image_ids, splits["val"]),
        bulk_assign_images_to_split(test_image_ids, splits["test"]),
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
    logger.info(f"  Images: {len(all_images)}")
    logger.info(f"  Patients: {len(all_patients)}")
    logger.info(f"  Classifications (DR + AMD): {len(all_classifications)}")
    logger.info(f"  Patient-Image Links: {len(all_patient_images)}")
    logger.info(f"  Comorbidity conditions per patient: {len(COMORBIDITY_CONDITIONS) + 1}")  # +1 for Disease_Risk
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
        stats = await ingest_rfmid()
        
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
