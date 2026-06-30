"""
Ingestion script for RFMiD 2.0 dataset.

Dataset: Retinal Fundus Multi-disease Image Dataset v2 (RFMiD 2.0)
Structure: CSV files with binary labels for 51 retinal conditions per split
Annotations: Classification (DR, ARMD), Patient comorbidities (49 other conditions)
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
DATASET_NAME = "RFMID2"
DATASET_URL = "https://ieee-dataport.org/documents/retinal-fundus-multi-disease-image-dataset-rfmid-20"
DATASET_LICENSE = "CC-BY-4.0"

# Full condition names mapping
CONDITION_NAMES = {
    "WNL": "Within Normal Limits",
    "AH": "Asteroid Hyalosis",
    "AION": "Anterior Ischemic Optic Neuropathy",
    "ARMD": "AMD",
    "BRVO": "Branch Retinal Vein Occlusion",
    "CB": "Coloboma",
    "CF": "Chorioretinal Fibrosis",
    "CL": "Chorioretinal Lesion",
    "CME": "Cystoid Macular Edema",
    "CNV": "Choroidal Neovascularization",
    "CRAO": "Central Retinal Artery Occlusion",
    "CRS": "Chorioretinal Scar",
    "CRVO": "Central Retinal Vein Occlusion",
    "CSR": "Central Serous Retinopathy",
    "CWS": "Cotton Wool Spots",
    "CSC": "Central Serous Chorioretinopathy",
    "DN": "Drusen",
    "DR": "DR",
    "EDN": "Exudation",
    "ERM": "Epiretinal Membrane",
    "GRT": "Giant Retinal Tear",
    "HPED": "Hemorrhagic Pigment Epithelial Detachment",
    "HR": "Hemorrhage",
    "LS": "Laser Scars",
    "MCA": "Microaneurysm",
    "ME": "Macular Edema",
    "MH": "Macular Hole",
    "MHL": "Lamellar Macular Hole",
    "MS": "Macular Scar",
    "MYA": "Myopia-related retinal changes",
    "ODC": "Optic Disc Cupping",
    "ODE": "Optic Disc Edema",
    "ODP": "Optic Disc Pallor",
    "ON": "Optic Neuritis",
    "OPDM": "Optic Disc Pit Maculopathy",
    "PRH": "Preretinal Hemorrhage",
    "RD": "Retinal Detachment",
    "RHL": "Retinal Hemorrhage",
    "RTR": "Retinal Traction",
    "RP": "Retinitis Pigmentosa",
    "RPEC": "Retinal Pigment Epithelial Changes",
    "RS": "Retinal Scar",
    "RT": "Retinal Tear",
    "SOFE": "Subfoveal Fibrosis",
    "ST": "Subretinal Tissue or Scar",
    "TD": "Tractional Detachment",
    "TSLN": "Tessellation",
    "TV": "Tortuous Vessels",
    "VS": "Vessel Sheathing",
    "HTN": "Hypertensive Retinopathy",
    "IIH": "Idiopathic Intracranial Hypertension",
}

# Conditions to store as classification annotations (matching RFMiD v1 pattern)
CLASSIFICATION_CONDITIONS = ["DR", "ARMD"]

# All other conditions go to patient comorbidities
COMORBIDITY_CONDITIONS = [
    key for key in CONDITION_NAMES.keys()
    if key not in CLASSIFICATION_CONDITIONS and key != "ID"
]

# CSV filename -> image folder -> split label
SPLIT_CONFIG = [
    ("RFMiD_2_Training_labels.csv", "Training", "train"),
    ("RFMiD_2_Testing_labels.csv", "Test", "test"),
    ("RFMiD_2_Validation_labels.csv", "Validation", "val"),
]


def _clean_column_name(name: str) -> str:
    """Strip non-breaking spaces and whitespace from column names."""
    return name.replace("\xa0", "").strip()


async def ingest_rfmid2() -> OperationStatistics:
    """
    Main ingestion function for RFMiD 2.0 dataset.

    Returns:
        OperationStatistics with success/error counts
    """
    data_root = get_data_root() / "40_RFMID2"
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
        description=(
            "RFMiD 2.0 (Retinal Fundus Multi-disease Image Dataset v2) with 860 images "
            "and binary labels for 51 retinal conditions. Successor to RFMiD with expanded "
            "condition set including CNV, CSC, HTN, IIH, and others."
        ),
    )
    await upsert_dataset(dataset)

    # Step 2: Count total rows for progress tracking
    logger.info("Counting CSV rows...")
    total_count = 0
    split_row_counts = {}
    for csv_name, _, split_label in SPLIT_CONFIG:
        csv_path = data_root / csv_name
        rows = await asyncio.to_thread(read_csv_auto, csv_path)
        split_row_counts[split_label] = len(rows)
        total_count += len(rows)

    logger.info(
        f"Found {split_row_counts.get('train', 0)} train, "
        f"{split_row_counts.get('test', 0)} test, "
        f"{split_row_counts.get('val', 0)} val images (total: {total_count})"
    )

    # Step 3: Setup progress tracker
    tracker = ProgressTracker(
        total=total_count,
        description=f"Ingesting {DATASET_NAME}",
    )

    # Collect items for bulk upsert
    all_images: List[Image] = []
    all_classifications: List[ClassificationAnnotation] = []
    all_patients: List[Patient] = []
    all_patient_images: List[PatientImage] = []
    image_to_split: Dict[UUID, str] = {}

    async def process_row(row, idx, split_folder, split_label):
        """Process a single CSV row."""
        try:
            # Clean column names (handles ON\xa0 -> ON, trailing empty column)
            cleaned = {_clean_column_name(k): v for k, v in row.items() if _clean_column_name(k)}

            image_id_str = str(cleaned["ID"])
            image_id = generate_image_uuid(dataset_id, image_id_str)

            # Find image file
            image_dir = data_root / split_folder
            image_path = image_dir / f"{image_id_str}.jpg"

            if not await asyncio.to_thread(image_path.exists):
                # Try alternate extensions
                for ext in [".JPG", ".jpeg", ".png"]:
                    candidate = image_dir / f"{image_id_str}{ext}"
                    if await asyncio.to_thread(candidate.exists):
                        image_path = candidate
                        break
                else:
                    tracker.record_error(
                        error_type="file_not_found",
                        error_message=f"Image not found: {image_path}",
                        item_id=image_id_str,
                    )
                    tracker.update(success=False)
                    return

            # Create image
            image = Image(
                image_id=image_id,
                dataset_id=dataset_id,
                original_image_id=image_id_str,
                **get_image_metadata_dict(image_path),
                modality="fundus",
            )
            all_images.append(image)
            image_to_split[image_id] = split_label

            # Standard: store the full multi-disease assessment as ONE multi_label panel
            # (sub_key per disease), consistent with RFMiD/MuReD/ODIR.
            disease_labels = {
                code: bool(int(cleaned[code]))
                for code in CONDITION_NAMES
                if code != "Disease_Risk" and code in cleaned
            }
            if disease_labels:
                all_classifications.extend(await process_classification(
                    class_value=disease_labels,
                    task_type="multi_label",
                    task_name="disease_panel",
                    class_name="disease_panel",
                    image_id=image_id,
                    annotation_method="manual",
                ))

            if "Disease_Risk" in cleaned:
                all_classifications.extend(await process_classification(
                    class_value=bool(int(cleaned["Disease_Risk"])),
                    task_type="binary",
                    task_name="disease_risk",
                    class_name="disease_risk",
                    image_id=image_id,
                    annotation_method="manual",
                ))

            # Create patient record with comorbidities
            patient_id = generate_patient_uuid(dataset_id, image_id_str)

            comorbidities = {}
            for condition_code in COMORBIDITY_CONDITIONS:
                if condition_code in cleaned:
                    full_name = CONDITION_NAMES[condition_code]
                    comorbidities[full_name] = bool(int(cleaned[condition_code]))

            patient = Patient(
                patient_id=patient_id,
                dataset_id=dataset_id,
                original_patient_id=image_id_str,
                comorbidities=comorbidities,
                created_at=datetime.now(),
            )
            all_patients.append(patient)

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

    async def make_handler(split_folder, split_label):
        async def handler(row, idx):
            await process_row(row, idx, split_folder, split_label)
        return handler

    tasks = []
    for csv_name, split_folder, split_label in SPLIT_CONFIG:
        handler = await make_handler(split_folder, split_label)
        tasks.append(process_csv(
            data_root / csv_name,
            dataset_id,
            "classification",
            handler,
            progress_tracker=tracker,
        ))
    results = await asyncio.gather(*tasks)

    for (csv_name, _, _), (_, raw_id, chain_id) in zip(SPLIT_CONFIG, results):
        logger.info(f"{csv_name} registered: raw_file_id={raw_id}, chain_id={chain_id}")

    # Step 5: Bulk upsert in FK order
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
    train_ids = [uid for uid, s in image_to_split.items() if s == "train"]
    test_ids = [uid for uid, s in image_to_split.items() if s == "test"]
    val_ids = [uid for uid, s in image_to_split.items() if s == "val"]

    splits = await register_standard_splits(
        dataset_id=dataset_id,
        split_type="explicit",
        train_count=len(train_ids),
        val_count=len(val_ids),
        test_count=len(test_ids),
    )

    await asyncio.gather(
        bulk_assign_images_to_split(train_ids, splits["train"]),
        bulk_assign_images_to_split(val_ids, splits["val"]),
        bulk_assign_images_to_split(test_ids, splits["test"]),
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
    logger.info(f"  Comorbidity conditions per patient: {len(COMORBIDITY_CONDITIONS)}")
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
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        stats = await ingest_rfmid2()

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
