"""
Ingestion script for MBRSET dataset.

Dataset: MBRSET - Macula-Based Retinal fundus diabetic retinopathy SET
Structure: Single labels_mbrset.csv; images in flat images/ directory named {patient}.{view}.jpg
Annotations: DR grading (ICDR 0-4), DME classification (yes/no), quality (artifacts + gradability)
Tasks: DR grading (ICDR 0-4), Binary DME classification, Quality annotation, Patient registration
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from uuid import UUID

from chaksudb.common.progress import OperationStatistics, ProgressTracker
from chaksudb.config.config import get_data_root
from chaksudb.db.models import (
    ClassificationAnnotation,
    Dataset,
    DiseaseGrading,
    Image,
    Patient,
    PatientImage,
    QualityAnnotation,
)
from chaksudb.db.queries import (
    bulk_upsert_classification_annotations,
    bulk_upsert_disease_gradings,
    bulk_upsert_images,
    bulk_upsert_patient_images,
    bulk_upsert_patients,
    bulk_upsert_quality_annotations,
    upsert_dataset,
)
from chaksudb.ingest.framework import (
    get_image_metadata_dict,
    process_csv,
    read_csv_auto,
)
from chaksudb.ingest.framework.gen_uuid import (
    generate_dataset_uuid,
    generate_image_uuid,
    generate_patient_image_uuid,
    generate_patient_uuid,
)
from chaksudb.ingest.framework.provenance_context import get_current_provenance
from chaksudb.ingest.framework.split_assigner import auto_stratified_splits
from chaksudb.ingest.framework.task_processors.classification_processor import (
    process_classification,
)
from chaksudb.ingest.framework.task_processors.grading_processor import process_disease_grade
from chaksudb.ingest.framework.task_processors.quality_processor import process_quality_annotation

logger = logging.getLogger(__name__)

DATASET_NAME = "MBRSET"
DATASET_URL = "https://physionet.org/content/mbrset/1.0/"
DATASET_LICENSE = "PhysioNet Credentialed Health Data License 1.5.0"


async def ingest_mbrset() -> OperationStatistics:
    """Main ingestion function for MBRSET dataset."""
    data_root = get_data_root() / "48_mbrset" / "physionet.org" / "files" / "mbrset" / "1.0"
    dataset_id = generate_dataset_uuid(DATASET_NAME)

    logger.info("=" * 80)
    logger.info(f"Starting ingestion: {DATASET_NAME}")
    logger.info(f"Data root: {data_root}")
    logger.info("=" * 80)

    # Step 1: Register dataset
    dataset = Dataset(
        dataset_id=dataset_id,
        dataset_name=DATASET_NAME,
        source_url=DATASET_URL,
        license=DATASET_LICENSE,
        modality_types=["fundus"],
    )
    await upsert_dataset(dataset)

    csv_path = data_root / "labels_mbrset.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Labels CSV not found: {csv_path}")

    # Step 2: Count rows for progress tracking
    rows = await asyncio.to_thread(read_csv_auto, csv_path)
    total_count = len(rows)
    logger.info(f"Found {total_count} rows in CSV")

    tracker = ProgressTracker(total=total_count, description=f"Ingesting {DATASET_NAME}")

    # Collect items for bulk upsert
    all_images: List[Image] = []
    all_patients: List[Patient] = []
    all_patient_image_pairs: List[tuple[UUID, UUID]] = []
    all_dr_gradings: List[DiseaseGrading] = []
    all_classifications: List[ClassificationAnnotation] = []
    all_quality: List[QualityAnnotation] = []

    # Map image_id → ICDR grade for stratified splits
    image_labels: Dict[UUID, int] = {}
    # Track patients to avoid duplicates within this run
    patient_lookup: Dict[str, UUID] = {}

    image_dir = data_root / "images"

    async def process_row(row, idx):
        try:
            file_name = row.get("file", "").strip()
            if not file_name:
                tracker.update(count=1, success=False)
                tracker.record_error("missing_filename", "Empty file field", item_id=str(idx))
                return

            image_path = image_dir / file_name
            if not await asyncio.to_thread(image_path.exists):
                tracker.record_error("file_not_found", f"Image not found: {file_name}", item_id=file_name)
                tracker.update(count=1, success=False)
                return

            image_id = generate_image_uuid(dataset_id, file_name)

            # Laterality
            laterality_raw = row.get("laterality", "").strip().lower()
            laterality = laterality_raw if laterality_raw in ("left", "right") else None

            image = Image(
                image_id=image_id,
                dataset_id=dataset_id,
                original_image_id=file_name,
                **get_image_metadata_dict(image_path),
                modality="fundus",
                eye_laterality=laterality,
            )
            all_images.append(image)

            # --- Patient registration ---
            original_patient_id = row.get("patient", "").strip()
            if original_patient_id:
                if original_patient_id not in patient_lookup:
                    age = None
                    try:
                        age_raw = row.get("age", "").strip()
                        if age_raw:
                            age = int(float(age_raw))
                    except (ValueError, TypeError):
                        pass

                    # sex: 0=female, 1=male (common convention for this Brazilian dataset)
                    sex = None
                    sex_raw = row.get("sex", "").strip()
                    if sex_raw == "1":
                        sex = "male"
                    elif sex_raw == "0":
                        sex = "female"

                    comorbidities: dict = {}

                    def _int_flag(col: str) -> bool | None:
                        v = row.get(col, "").strip()
                        if v == "1":
                            return True
                        if v == "0":
                            return False
                        return None

                    def _add_flag(key: str, col: str) -> None:
                        v = _int_flag(col)
                        if v is True:
                            comorbidities[key] = True

                    _add_flag("systemic_hypertension", "systemic_hypertension")
                    _add_flag("insulin", "insulin")
                    _add_flag("oral_treatment_dm", "oraltreatment_dm")
                    _add_flag("obesity", "obesity")
                    _add_flag("vascular_disease", "vascular_disease")
                    _add_flag("acute_myocardial_infarction", "acute_myocardial_infarction")
                    _add_flag("nephropathy", "nephropathy")
                    _add_flag("neuropathy", "neuropathy")
                    _add_flag("diabetic_foot", "diabetic_foot")
                    _add_flag("alcohol_consumption", "alcohol_consumption")
                    _add_flag("smoking", "smoking")

                    try:
                        dm_time_raw = row.get("dm_time", "").strip()
                        if dm_time_raw:
                            comorbidities["diabetes_duration_years"] = int(float(dm_time_raw))
                    except (ValueError, TypeError):
                        pass

                    patient_id = generate_patient_uuid(dataset_id, original_patient_id)
                    patient = Patient(
                        patient_id=patient_id,
                        dataset_id=dataset_id,
                        original_patient_id=original_patient_id,
                        age=age,
                        sex=sex,
                        comorbidities=comorbidities if comorbidities else None,
                    )
                    all_patients.append(patient)
                    patient_lookup[original_patient_id] = patient_id
                else:
                    patient_id = patient_lookup[original_patient_id]

                all_patient_image_pairs.append((patient_id, image_id))

            raw_data_id, provenance_chain_id = get_current_provenance()

            # --- DR grading (ICDR 0-4) ---
            icdr_raw = row.get("final_icdr", "").strip()
            if icdr_raw:
                try:
                    icdr_value = int(float(icdr_raw))
                    grading = await process_disease_grade(
                        grade_value=icdr_value,
                        disease_type="DR",
                        scale_name="ICDR_0_4",
                        image_id=image_id,
                        raw_data_id=raw_data_id,
                        provenance_chain_id=provenance_chain_id,
                        scale_description="5-level ICDR diabetic retinopathy grading",
                        min_value=0,
                        max_value=4,
                        value_labels={
                            "0": "No DR",
                            "1": "Mild NPDR",
                            "2": "Moderate NPDR",
                            "3": "Severe NPDR",
                            "4": "PDR",
                        },
                        annotation_method="manual",
                    )
                    all_dr_gradings.append(grading)
                    image_labels[image_id] = icdr_value
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid ICDR grade '{icdr_raw}' for {file_name}: {e}")

            # --- DME binary classification ---
            edema_raw = row.get("final_edema", "").strip().lower()
            if edema_raw in ("yes", "no"):
                edema_value = 1 if edema_raw == "yes" else 0
                classifications = await process_classification(
                    class_value=edema_value,
                    task_type="binary",
                    class_name="DME",
                    image_id=image_id,
                    raw_data_id=raw_data_id,
                    provenance_chain_id=provenance_chain_id,
                    annotation_method="manual",
                )
                all_classifications.extend(classifications)

            # --- Quality: overall gradability ---
            quality_raw = row.get("final_quality", "").strip().lower()
            if quality_raw in ("yes", "no"):
                quality_label = "good" if quality_raw == "yes" else "bad"
                quality_ann = await process_quality_annotation(
                    quality_type="gradability",
                    image_id=image_id,
                    quality_label=quality_label,
                    scale_description="MBRSET overall image gradability (yes=gradable)",
                    raw_data_id=raw_data_id,
                    provenance_chain_id=provenance_chain_id,
                )
                all_quality.append(quality_ann)

            # --- Quality: artifact presence ---
            artifact_raw = row.get("final_artifacts", "").strip().lower()
            if artifact_raw in ("yes", "no"):
                # yes=artifacts present → bad; no=clean → good
                artifact_label = "bad" if artifact_raw == "yes" else "good"
                artifact_ann = await process_quality_annotation(
                    quality_type="artifact",
                    image_id=image_id,
                    quality_label=artifact_label,
                    scale_description="MBRSET artifact presence (yes=artifacts present)",
                    raw_data_id=raw_data_id,
                    provenance_chain_id=provenance_chain_id,
                )
                all_quality.append(artifact_ann)

            tracker.update(count=1, success=True)

        except Exception as e:
            tracker.update(count=1, success=False)
            tracker.record_error(
                error_type="processing",
                error_message=str(e),
                item_id=row.get("file", str(idx)),
            )
            logger.error(f"Failed to process row {idx}: {e}")

    # Step 3: Process CSV with provenance tracking
    logger.info("Processing annotations...")
    await process_csv(csv_path, dataset_id, "grading", process_row)

    # Step 4: Bulk upsert — images first (FK dependency order)
    logger.info(f"Upserting {len(all_images)} images...")
    await bulk_upsert_images(all_images, batch_size=1000)

    if all_patients:
        logger.info(f"Upserting {len(all_patients)} patients...")
        await bulk_upsert_patients(all_patients, batch_size=1000)

    if all_patient_image_pairs:
        logger.info(f"Linking {len(all_patient_image_pairs)} patient-image pairs...")
        patient_image_models = [
            PatientImage(
                relationship_id=generate_patient_image_uuid(pid, iid),
                patient_id=pid,
                image_id=iid,
                exam_date=None,
                created_at=datetime.now(),
            )
            for pid, iid in all_patient_image_pairs
        ]
        await bulk_upsert_patient_images(patient_image_models)

    logger.info(
        f"Upserting {len(all_dr_gradings)} DR gradings, "
        f"{len(all_classifications)} classifications, "
        f"{len(all_quality)} quality annotations..."
    )
    await asyncio.gather(
        bulk_upsert_disease_gradings(all_dr_gradings, batch_size=1000),
        bulk_upsert_classification_annotations(all_classifications, batch_size=1000),
        bulk_upsert_quality_annotations(all_quality, batch_size=1000),
    )

    # Step 5: Auto-stratified splits (no explicit split info in dataset)
    logger.info("Registering dataset splits...")
    all_image_ids = [img.image_id for img in all_images]
    if all_image_ids:
        await auto_stratified_splits(
            dataset_id=dataset_id,
            split_assignments={"train": all_image_ids},
            labels=image_labels,
            split_type="undefined",
        )

    tracker.finish()
    stats = tracker.get_statistics()

    logger.info("=" * 80)
    logger.info("Ingestion Summary:")
    logger.info(f"  Total items:  {stats.total_items}")
    logger.info(f"  Successful:   {stats.successful_items}")
    logger.info(f"  Failed:       {stats.failed_items}")
    logger.info(f"  Skipped:      {stats.skipped_items}")
    if stats.errors:
        logger.warning(f"  Total errors: {len(stats.errors)}")
        for error_type, count in stats.error_counts.items():
            logger.warning(f"    {error_type}: {count}")
    logger.info("=" * 80)

    return stats


async def main():
    """Entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    try:
        stats = await ingest_mbrset()
        if stats.failed_items > 0:
            logger.error(f"Ingestion completed with {stats.failed_items} errors")
            return 1
        logger.info("Ingestion completed successfully!")
        return 0
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        return 1


if __name__ == "__main__":
    exit(asyncio.run(main()))
