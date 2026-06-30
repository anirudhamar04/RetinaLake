"""
Ingestion script for MMAC dataset.

Dataset: MMAC 2023 - Myopic Maculopathy Analysis Challenge
Structure:
  Task 1 — Classification of Myopic Maculopathy
    1. Classification of Myopic Maculopathy/1. Images/{1. Training Set, 2. Validation Set}/
    1. Classification of Myopic Maculopathy/2. Groundtruths/{train_csv, val_csv}
    CSV columns: image, myopic_maculopathy_grade, age, sex, height, weight, data_center

  Task 2 — Segmentation of Myopic Maculopathy Plus Lesions
    Three lesion types, each in its own subfolder:
      1. Lacquer Cracks  (LC)
      2. Choroidal Neovascularization (CNV)
      3. Fuchs Spot  (FS)
    Each type has:
      2. Groundtruths/1. Training Set/    — binary PNG masks (same name as image)
      2. Groundtruths/2. Validation Set/  — binary PNG masks
      2. Groundtruths/{train_csv, val_csv} — CSV with patient details

Annotations:
  - Disease grading (META-PM scale, C0-run aC4) from Task 1 CSV
  - Segmentation (binary PNG masks) for LC, CNV, FS
  - Patient metadata (age, sex, height, weight, data_center) — skipped gracefully
    if values are missing

Grading scale:
  META_PM_0_4 (disease_type="myopic_maculopathy"):
    0 = No myopic retinal degenerative lesions
    1 = Tessellated fundus
    2 = Diffuse chorioretinal atrophy
    3 = Patchy chorioretinal atrophy
    4 = Macular atrophy
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from uuid import UUID

from chaksudb.common.progress import ProgressTracker, OperationStatistics
from chaksudb.config.config import get_data_root
from chaksudb.db.models import (
    Dataset,
    DiseaseGrading,
    Image,
    Patient,
    PatientImage,
    SegmentationAnnotation,
)
from chaksudb.db.queries import (
    upsert_dataset,
    bulk_upsert_images,
    bulk_upsert_disease_gradings,
    upsert_segmentation_annotation,
)
from chaksudb.db.queries.patients import bulk_upsert_patients
from chaksudb.db.queries.images import bulk_upsert_patient_images
from chaksudb.ingest.framework import (
    read_csv_auto,
    get_image_metadata_dict,
)
from chaksudb.ingest.framework.gen_uuid import (
    generate_dataset_uuid,
    generate_image_uuid,
    generate_patient_image_uuid,
    generate_patient_uuid,
)
from chaksudb.ingest.framework.raw_file_helpers import register_individual_file
from chaksudb.ingest.framework.split_assigner import (
    register_standard_splits,
    bulk_assign_images_to_split,
)
from chaksudb.ingest.framework.task_processors.grading_processor import (
    process_disease_grade,
)
from chaksudb.ingest.framework.task_processors.segmentation_processor import (
    process_segmentation_from_binary_mask,
)

logger = logging.getLogger(__name__)

# Dataset metadata
DATASET_NAME = "MMAC"
DATASET_URL = "https://codalab.lisn.upsaclay.fr/competitions/12441"
DATASET_LICENSE = "Research/Academic Use"

# META-PM grading scale
META_PM_SCALE = "META_PM_0_4"
META_PM_LABELS = {
    "0": "No myopic retinal degenerative lesions",
    "1": "Tessellated fundus",
    "2": "Diffuse chorioretinal atrophy",
    "3": "Patchy chorioretinal atrophy",
    "4": "Macular atrophy",
}

# Segmentation lesion types
LESION_TYPES = [
    {
        "folder": "1. Lacquer Cracks",
        "annotation_type": "lacquer_cracks",
        "description": "Lacquer crack lesion segmentation (Myopic Maculopathy)",
        "train_csv_name": "1. MMAC2023_Myopic_Maculopathy_Plus_Lesions_Segmentation_Training_Images_Lacquer_Cracks.csv",
        "val_csv_name": "2. MMAC2023_Myopic_Maculopathy_Plus_Lesions_Segmentation_Validation_Images_Lacquer_Cracks.csv",
    },
    {
        "folder": "2. Choroidal Neovascularization",
        "annotation_type": "choroidal_neovascularization",
        "description": "Choroidal neovascularization segmentation (Myopic Maculopathy)",
        "train_csv_name": "1. MMAC2023_Myopic_Maculopathy_Plus_Lesions_Segmentation_Training_Images_Choroidal_Neovascularization.csv",
        "val_csv_name": "2. MMAC2023_Myopic_Maculopathy_Plus_Lesions_Segmentation_Validation_Images_Choroidal_Neovascularization.csv",
    },
    {
        "folder": "3. Fuchs Spot",
        "annotation_type": "fuchs_spot",
        "description": "Fuchs spot lesion segmentation (Myopic Maculopathy)",
        "train_csv_name": "1. MMAC2023_Myopic_Maculopathy_Plus_Lesions_Segmentation_Training_Images_Fuchs_Spot.csv",
        "val_csv_name": "2. MMAC2023_Myopic_Maculopathy_Plus_Lesions_Segmentation_Validation_Images_Fuchs_Spot.csv",
    },
]

# Classification CSV paths (relative to dataset root)
CLASSIFICATION_ROOT = "1. Classification of Myopic Maculopathy"
SEGMENTATION_ROOT = "2. Segmentation of Myopic Maculopathy Plus Lesions"


def _extract_patient_data(
    row: dict,
) -> Optional[dict]:
    """Extract patient demographic data from a CSV row.

    Returns a dict with demographic fields, or None if all optional fields
    are missing/empty (in which case no patient record should be created).
    Row contains: age, sex, height, weight, data_center
    """
    age_raw = str(row.get("age", "")).strip()
    sex_raw = str(row.get("sex", "")).strip().lower()
    height_raw = str(row.get("height", "")).strip()
    weight_raw = str(row.get("weight", "")).strip()
    data_center_raw = str(row.get("data_center", "")).strip()

    # Normalise age
    age = None
    try:
        if age_raw:
            age = int(float(age_raw))
    except (ValueError, TypeError):
        pass

    # Normalise sex
    sex = None
    if sex_raw in ("male", "m"):
        sex = "male"
    elif sex_raw in ("female", "f"):
        sex = "female"

    # At minimum we need age or sex to justify a patient record
    if age is None and sex is None:
        return None

    # Build comorbidities dict for extra fields
    comorbidities: dict = {}
    try:
        if height_raw:
            comorbidities["height_cm"] = float(height_raw)
    except ValueError:
        pass
    try:
        if weight_raw:
            comorbidities["weight_kg"] = float(weight_raw)
    except ValueError:
        pass
    if data_center_raw:
        comorbidities["data_center"] = data_center_raw

    return {
        "age": age,
        "sex": sex,
        "comorbidities": comorbidities if comorbidities else None,
    }


async def _process_classification_split(
    split_name: str,
    csv_path: Path,
    images_dir: Path,
    dataset_id: UUID,
    tracker: ProgressTracker,
) -> Tuple[List[Image], List[DiseaseGrading], List[UUID], List[Tuple[dict, UUID]]]:
    """Process one classification split (train or val).

    Returns:
        (images, gradings, image_ids_for_split, patient_rows)
        patient_rows: list of (patient_info_dict, image_id) for later patient registration.
    """
    all_images: List[Image] = []
    all_gradings: List[DiseaseGrading] = []
    image_ids: List[UUID] = []
    # Each row becomes its own "patient" keyed by image name (no shared patient IDs)
    patient_data_rows: List[Tuple[dict, UUID]] = []

    if not csv_path.exists():
        logger.warning("Classification CSV not found: %s", csv_path)
        return all_images, all_gradings, image_ids, patient_data_rows

    # Register CSV for provenance
    raw_file_id, chain_id = await register_individual_file(
        file_path=csv_path,
        dataset_id=dataset_id,
        unified_annotation_type="grading",
    )

    rows = await asyncio.to_thread(read_csv_auto, csv_path)
    logger.info("  %s: %d rows in %s", split_name, len(rows), csv_path.name)

    for row in rows:
        image_name = str(row.get("image", "")).strip()
        if not image_name:
            tracker.update(success=False)
            tracker.record_error(
                error_type="missing_image_name",
                error_message="Empty image name in CSV row",
                item_id="unknown",
            )
            continue

        image_path = images_dir / image_name
        if not image_path.exists():
            tracker.update(success=False)
            tracker.record_error(
                error_type="file_not_found",
                error_message=f"Image not found: {image_name}",
                item_id=image_name,
                item_path=str(image_path),
            )
            continue

        grade_raw = str(row.get("myopic_maculopathy_grade", "")).strip()
        try:
            grade_int = int(float(grade_raw))
        except (ValueError, TypeError):
            tracker.update(success=False)
            tracker.record_error(
                error_type="invalid_grade",
                error_message=f"Cannot parse grade: {grade_raw!r}",
                item_id=image_name,
            )
            continue

        try:
            image_stem = Path(image_name).stem
            image_id = generate_image_uuid(dataset_id, image_stem)

            image = Image(
                image_id=image_id,
                dataset_id=dataset_id,
                original_image_id=image_stem,
                **get_image_metadata_dict(image_path),
                modality="fundus",
            )
            all_images.append(image)
            image_ids.append(image_id)

            grading = await process_disease_grade(
                grade_value=grade_int,
                disease_type="myopic_maculopathy",
                scale_name=META_PM_SCALE,
                image_id=image_id,
                scale_description="META-PM myopic maculopathy grading scale (C0–C4)",
                min_value=0,
                max_value=4,
                value_labels=META_PM_LABELS,
                grade_label=META_PM_LABELS.get(str(grade_int)),
                raw_data_id=raw_file_id,
                provenance_chain_id=chain_id,
                annotation_method="manual",
            )
            all_gradings.append(grading)

            # Collect patient data (graceful: skip if incomplete)
            patient_info = _extract_patient_data(row)
            if patient_info is not None:
                patient_data_rows.append((patient_info, image_id))

            tracker.update(success=True)
            tracker.record_success("image")

        except Exception as exc:
            logger.error("Failed to process %s: %s", image_name, exc, exc_info=True)
            tracker.update(success=False)
            tracker.record_error(
                error_type="processing",
                error_message=str(exc),
                item_id=image_name,
            )

    return all_images, all_gradings, image_ids, patient_data_rows


async def _process_segmentation_lesion(
    lesion_info: dict,
    split_name: str,
    csv_path: Path,
    images_dir: Path,
    masks_dir: Path,
    dataset_id: UUID,
    tracker: ProgressTracker,
) -> Tuple[List[Image], List[SegmentationAnnotation], List[UUID], List[Tuple[dict, UUID]]]:
    """Process one lesion type for one split (train or val).

    Images may already be registered from another lesion type or from the
    classification task; duplicates are handled by the idempotent upsert.
    """
    all_images: List[Image] = []
    all_segmentations: List[SegmentationAnnotation] = []
    image_ids: List[UUID] = []
    patient_data_rows: List[Tuple[dict, UUID]] = []

    annotation_type = lesion_info["annotation_type"]
    description = lesion_info["description"]

    if not csv_path.exists():
        logger.warning("Segmentation CSV not found: %s", csv_path)
        return all_images, all_segmentations, image_ids, patient_data_rows

    rows = await asyncio.to_thread(read_csv_auto, csv_path)
    logger.info("  %s/%s: %d rows", split_name, annotation_type, len(rows))

    for row in rows:
        image_name = str(row.get("image", "")).strip()
        if not image_name:
            continue

        image_path = images_dir / image_name
        mask_path = masks_dir / image_name

        if not image_path.exists():
            tracker.update(success=False)
            tracker.record_error(
                error_type="file_not_found",
                error_message=f"Image not found: {image_name}",
                item_id=image_name,
                item_path=str(image_path),
            )
            continue

        try:
            image_stem = Path(image_name).stem
            image_id = generate_image_uuid(dataset_id, image_stem)

            image = Image(
                image_id=image_id,
                dataset_id=dataset_id,
                original_image_id=image_stem,
                **get_image_metadata_dict(image_path),
                modality="fundus",
            )
            all_images.append(image)
            image_ids.append(image_id)

            # Segmentation mask (may not exist for validation set of some lesion types)
            if mask_path.exists():
                mask_raw_id, mask_chain_id = await register_individual_file(
                    file_path=mask_path,
                    dataset_id=dataset_id,
                    unified_annotation_type="segmentation",
                    file_type=None,
                    auto_detect_type=False,
                )
                seg = await process_segmentation_from_binary_mask(
                    mask_path=mask_path,
                    annotation_type=annotation_type,
                    image_id=image_id,
                    annotation_description=description,
                    merge_nonzero=True,
                    fill_holes=False,
                    raw_data_id=mask_raw_id,
                    annotation_method="manual",
                    provenance_chain_id=mask_chain_id,
                    dataset_name=DATASET_NAME,
                    dataset_id=dataset_id,
                )
                all_segmentations.append(seg)
            else:
                logger.debug("Mask not found for %s (lesion: %s)", image_name, annotation_type)

            # Patient data
            patient_info = _extract_patient_data(row)
            if patient_info is not None:
                patient_data_rows.append((patient_info, image_id))

            tracker.update(success=True)
            tracker.record_success("image")

        except Exception as exc:
            logger.error(
                "Failed to process %s [%s]: %s", image_name, annotation_type, exc, exc_info=True
            )
            tracker.update(success=False)
            tracker.record_error(
                error_type="processing",
                error_message=str(exc),
                item_id=image_name,
            )

    return all_images, all_segmentations, image_ids, patient_data_rows


async def ingest_mmac() -> OperationStatistics:
    """Main ingestion function for MMAC dataset.

    Strategy:
    - Task 1: Process training + validation CSVs → disease gradings + images
    - Task 2: For each of 3 lesion types, process training + validation → segmentations
    - Patient metadata extracted from CSVs; rows with missing age/sex skipped silently
    - Splits registered: train / validation

    Returns:
        OperationStatistics with success/error counts
    """
    data_root = get_data_root() / "38_MMAC"
    dataset_id = generate_dataset_uuid(DATASET_NAME)

    logger.info("=" * 80)
    logger.info("Starting ingestion: %s", DATASET_NAME)
    logger.info("Data root: %s", data_root)
    logger.info("=" * 80)

    # Step 1: Register dataset
    dataset = Dataset(
        dataset_id=dataset_id,
        dataset_name=DATASET_NAME,
        source_url=DATASET_URL,
        license=DATASET_LICENSE,
        modality_types=["fundus"],
        description=(
            "MMAC 2023 (Myopic Maculopathy Analysis Challenge) dataset. "
            "Task 1: Graded classification of myopic maculopathy severity "
            "(META-PM scale C0–C4). Task 2: Binary segmentation of three "
            "myopic maculopathy plus lesions — Lacquer Cracks, Choroidal "
            "Neovascularization, and Fuchs Spot."
        ),
    )
    await upsert_dataset(dataset)

    # Step 2: Count total items for progress tracking
    cls_root = data_root / CLASSIFICATION_ROOT
    seg_root = data_root / SEGMENTATION_ROOT

    train_cls_csv = cls_root / "2. Groundtruths" / "1. MMAC2023_Myopic_Maculopathy_Classification_Training_Labels.csv"
    val_cls_csv = cls_root / "2. Groundtruths" / "2. MMAC2023_Myopic_Maculopathy_Classification_Validation_Labels.csv"

    def _count_csv(path: Path) -> int:
        if not path.exists():
            return 0
        rows = read_csv_auto(path)
        return len(rows)

    count_tasks = [
        asyncio.to_thread(_count_csv, train_cls_csv),
        asyncio.to_thread(_count_csv, val_cls_csv),
    ]
    for lesion in LESION_TYPES:
        for csv_name_key in ("train_csv_name", "val_csv_name"):
            gt_dir = seg_root / lesion["folder"] / "2. Groundtruths"
            count_tasks.append(asyncio.to_thread(_count_csv, gt_dir / lesion[csv_name_key]))
    counts = await asyncio.gather(*count_tasks)
    cls_train_count, cls_val_count = counts[0], counts[1]
    seg_counts = sum(counts[2:])

    total_items = cls_train_count + cls_val_count + seg_counts
    logger.info(
        "Total items: %d (cls_train=%d, cls_val=%d, seg=%d)",
        total_items,
        cls_train_count,
        cls_val_count,
        seg_counts,
    )

    tracker = ProgressTracker(total=total_items, description=f"Ingesting {DATASET_NAME}")

    # Collect everything for bulk upsert
    all_images: List[Image] = []
    all_gradings: List[DiseaseGrading] = []
    all_segmentations: List[SegmentationAnnotation] = []
    train_image_ids: List[UUID] = []
    val_image_ids: List[UUID] = []
    all_patient_rows: List[Tuple[dict, UUID]] = []

    # --- Step 3: Task 1 — Classification ---
    logger.info("=" * 60)
    logger.info("Task 1: Classification")

    for split_name, csv_path, images_subdir, image_ids_list in [
        (
            "train",
            train_cls_csv,
            cls_root / "1. Images" / "1. Training Set",
            train_image_ids,
        ),
        (
            "val",
            val_cls_csv,
            cls_root / "1. Images" / "2. Validation Set",
            val_image_ids,
        ),
    ]:
        images, gradings, ids, patient_rows = await _process_classification_split(
            split_name=split_name,
            csv_path=csv_path,
            images_dir=images_subdir,
            dataset_id=dataset_id,
            tracker=tracker,
        )
        all_images.extend(images)
        all_gradings.extend(gradings)
        image_ids_list.extend(ids)
        all_patient_rows.extend(patient_rows)

    # --- Step 4: Task 2 — Segmentation ---
    logger.info("=" * 60)
    logger.info("Task 2: Segmentation")

    for lesion in LESION_TYPES:
        lesion_dir = seg_root / lesion["folder"]
        gt_dir = lesion_dir / "2. Groundtruths"

        for split_name, csv_name_key, images_subdir, masks_subdir, image_ids_list in [
            (
                "train",
                "train_csv_name",
                lesion_dir / "1. Images" / "1. Training Set",
                gt_dir / "1. Training Set",
                train_image_ids,
            ),
            (
                "val",
                "val_csv_name",
                lesion_dir / "1. Images" / "2. Validation Set",
                gt_dir / "2. Validation Set",
                val_image_ids,
            ),
        ]:
            csv_path = gt_dir / lesion[csv_name_key]
            images, segs, ids, patient_rows = await _process_segmentation_lesion(
                lesion_info=lesion,
                split_name=split_name,
                csv_path=csv_path,
                images_dir=images_subdir,
                masks_dir=masks_subdir,
                dataset_id=dataset_id,
                tracker=tracker,
            )
            all_images.extend(images)
            all_segmentations.extend(segs)
            image_ids_list.extend(ids)
            all_patient_rows.extend(patient_rows)

    # --- Step 5: Bulk upsert ---
    # Deduplicate images — same image referenced from classification + multiple lesion CSVs
    all_images = list({img.image_id: img for img in all_images}.values())
    logger.info("Upserting %d images...", len(all_images))
    if all_images:
        await bulk_upsert_images(all_images, batch_size=500)

    logger.info("Upserting %d disease gradings...", len(all_gradings))
    if all_gradings:
        await bulk_upsert_disease_gradings(all_gradings, batch_size=500)

    logger.info("Upserting %d segmentation annotations...", len(all_segmentations))
    for seg in all_segmentations:
        try:
            await upsert_segmentation_annotation(seg)
        except Exception as exc:
            logger.error("Failed to upsert segmentation %s: %s", seg.segmentation_id, exc)
            tracker.record_error(
                error_type="segmentation_upsert",
                error_message=str(exc),
                item_id=str(seg.segmentation_id),
            )

    # --- Step 6: Patient registration ---
    # Deduplicate by image_id — same image appears in classification + segmentation CSVs
    unique_patient_rows = list({img_id: info for info, img_id in all_patient_rows}.items())
    logger.info("Registering %d patient records...", len(unique_patient_rows))
    if unique_patient_rows:
        patient_models: List[Patient] = []
        patient_image_models: List[PatientImage] = []

        for image_id, patient_info in unique_patient_rows:
            # Use image_id as patient key (one image = one row in MMAC)
            original_patient_id = str(image_id)
            patient_id = generate_patient_uuid(
                dataset_id=dataset_id,
                original_patient_id=original_patient_id,
            )
            patient_models.append(
                Patient(
                    patient_id=patient_id,
                    dataset_id=dataset_id,
                    original_patient_id=original_patient_id,
                    age=patient_info.get("age"),
                    sex=patient_info.get("sex"),
                    comorbidities=patient_info.get("comorbidities"),
                    created_at=datetime.now(),
                )
            )
            rel_id = generate_patient_image_uuid(patient_id=patient_id, image_id=image_id)
            patient_image_models.append(
                PatientImage(
                    relationship_id=rel_id,
                    patient_id=patient_id,
                    image_id=image_id,
                    created_at=datetime.now(),
                )
            )

        await bulk_upsert_patients(patient_models)
        await bulk_upsert_patient_images(patient_image_models)

    # --- Step 7: Splits ---
    logger.info("Registering dataset splits...")
    # Deduplicate image IDs (same image may appear under multiple lesion types)
    train_unique = list(dict.fromkeys(train_image_ids))
    val_unique = list(dict.fromkeys(val_image_ids))

    splits = await register_standard_splits(
        dataset_id=dataset_id,
        split_type="explicit",
        train_count=len(train_unique),
        val_count=len(val_unique),
    )
    await asyncio.gather(
        bulk_assign_images_to_split(train_unique, splits["train"]) if train_unique else asyncio.sleep(0),
        bulk_assign_images_to_split(val_unique, splits["val"]) if val_unique else asyncio.sleep(0),
    )

    tracker.finish()
    final_stats = tracker.get_statistics()

    logger.info("=" * 80)
    logger.info("Ingestion Summary: %s", DATASET_NAME)
    logger.info("  Total items:   %d", final_stats.total_items)
    logger.info("  Successful:    %d", final_stats.successful_items)
    logger.info("  Failed:        %d", final_stats.failed_items)
    logger.info("  Images:        %d", len(all_images))
    logger.info("  Gradings:      %d", len(all_gradings))
    logger.info("  Segmentations: %d", len(all_segmentations))
    logger.info("  Patients:      %d", len(unique_patient_rows))
    if final_stats.errors:
        logger.warning("  Errors (%d):", len(final_stats.errors))
        for err_type, count in final_stats.error_counts.items():
            logger.warning("    %s: %d", err_type, count)
    logger.info("=" * 80)

    return final_stats


async def main():
    """Entry point for script execution."""
    import sys

    log_file = Path("./logs/ingest_38_mmac.log")
    log_file.parent.mkdir(exist_ok=True)
    log_file.touch(exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    stats = await ingest_mmac()
    return 0 if stats.failed_items == 0 else 1


if __name__ == "__main__":
    exit(asyncio.run(main()))
