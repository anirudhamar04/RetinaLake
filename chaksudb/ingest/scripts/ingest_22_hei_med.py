"""
Ingestion script for HEI-MED dataset.

Dataset: HEI-MED — Hamburg Eye Image Database for Early Detection of Exudates
         in Diabetic Macular Edema
Structure: DMED/ folder — per image, companion files share a stem:
    (NNNNNNN).jpg       — colour fundus image
    (NNNNNNN)_vess.png  — pre-computed binary vessel segmentation mask
    (NNNNNNN).meta      — tilde-delimited patient metadata

Annotations:
  - Segmentation: vessel mask (from _vess.png, pseudo/pre-computed)
  - Quality: QualityValue from .meta
  - Patient: PatientGender, PatientRace, DiabetesType from .meta (if present)
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple
from uuid import UUID

from chaksudb.common.progress import ProgressTracker, OperationStatistics
from chaksudb.config.config import get_data_root
from chaksudb.db.models import (
    Dataset,
    Image,
    Patient,
    PatientImage,
    QualityAnnotation,
    SegmentationAnnotation,
)
from chaksudb.db.queries import (
    upsert_dataset,
    bulk_upsert_images,
    bulk_upsert_quality_annotations,
    upsert_segmentation_annotation,
)
from chaksudb.db.queries.patients import bulk_upsert_patients
from chaksudb.db.queries.images import bulk_upsert_patient_images
from chaksudb.ingest.framework import (
    find_images,
    get_image_metadata_dict,
)
from chaksudb.ingest.framework.gen_uuid import (
    generate_dataset_uuid,
    generate_image_uuid,
    generate_patient_uuid,
    generate_patient_image_uuid,
)
from chaksudb.ingest.framework.mask_converter.gnd_maps import parse_meta_file
from chaksudb.ingest.framework.raw_file_helpers import register_individual_file
from chaksudb.ingest.framework.task_processors.quality_processor import (
    process_quality_annotation,
)
from chaksudb.ingest.framework.task_processors.segmentation_processor import (
    process_segmentation_from_binary_mask,
)

logger = logging.getLogger(__name__)

# Dataset metadata
DATASET_NAME = "HEI-MED"
DATASET_URL = "https://github.com/lgiancaUTH/HEI-MED"
DATASET_LICENSE = "Research/Academic Use"


def _normalise_sex(gender_str: str) -> Optional[str]:
    """Map HEI-MED gender codes to canonical values."""
    g = gender_str.strip().upper()
    if g in ("M", "MALE"):
        return "male"
    if g in ("F", "FEMALE"):
        return "female"
    return None


async def _process_image(
    jpg_path: Path,
    dataset_id: UUID,
    tracker: ProgressTracker,
) -> Tuple[
    Optional[Image],
    List[SegmentationAnnotation],
    Optional[QualityAnnotation],
    Optional[Tuple[dict, UUID]],
]:
    """Process one HEI-MED image and all its companion files.

    Returns:
        (image, segmentations, quality_annotation, patient_tuple)
        patient_tuple: (patient_data_dict, image_id) or None
    """
    stem = jpg_path.stem  # e.g. "(00000003)"
    parent = jpg_path.parent

    vess_path = parent / f"{stem}_vess.png"
    meta_path = parent / f"{stem}.meta"

    try:
        image_id = generate_image_uuid(dataset_id, stem)

        image = Image(
            image_id=image_id,
            dataset_id=dataset_id,
            original_image_id=stem,
            **get_image_metadata_dict(jpg_path),
            modality="fundus",
        )

        segmentations: List[SegmentationAnnotation] = []

        # --- Vessel segmentation from _vess.png ---
        if vess_path.exists():
            try:
                vess_raw_id, vess_chain_id = await register_individual_file(
                    file_path=vess_path,
                    dataset_id=dataset_id,
                    unified_annotation_type="segmentation",
                    file_type=None,
                    auto_detect_type=False,
                )
                vess_seg = await process_segmentation_from_binary_mask(
                    mask_path=vess_path,
                    annotation_type="vessels",
                    image_id=image_id,
                    annotation_description="Retinal vessel segmentation (HEI-MED pre-computed)",
                    merge_nonzero=True,
                    fill_holes=False,  # Never fill holes for vessels
                    raw_data_id=vess_raw_id,
                    annotation_method="pseudo",
                    provenance_chain_id=vess_chain_id,
                    dataset_name=DATASET_NAME,
                    dataset_id=dataset_id,
                )
                segmentations.append(vess_seg)
            except Exception as exc:
                logger.warning("Failed to process vessel mask for %s: %s", stem, exc)
                tracker.record_error(
                    error_type="vessel_mask_error",
                    error_message=str(exc),
                    item_id=stem,
                )

        # --- Quality annotation from .meta ---
        quality_annotation: Optional[QualityAnnotation] = None
        patient_tuple: Optional[Tuple[dict, UUID]] = None

        if meta_path.exists():
            meta = await asyncio.to_thread(parse_meta_file, meta_path)

            # Quality
            quality_raw = meta.get("QualityValue", "").strip()
            if quality_raw:
                try:
                    quality_score = float(quality_raw)
                    quality_annotation = await process_quality_annotation(
                        quality_type="image_quality",
                        image_id=image_id,
                        quality_score=quality_score,
                        quality_label=None,
                    )
                except (ValueError, TypeError) as exc:
                    logger.debug("Could not parse QualityValue for %s: %s", stem, exc)

            # Patient
            gender_raw = meta.get("PatientGender", "").strip()
            sex = _normalise_sex(gender_raw) if gender_raw else None
            race = meta.get("PatientRace", "").strip() or None
            diabetes_type = meta.get("DiabetesType", "").strip() or None

            if sex or race or diabetes_type:
                comorbidities: dict = {}
                if diabetes_type:
                    comorbidities["diabetes_type"] = diabetes_type
                patient_tuple = (
                    {
                        "sex": sex,
                        "ethnicity": race,
                        "comorbidities": comorbidities if comorbidities else None,
                    },
                    image_id,
                )

        tracker.update(success=True)
        tracker.record_success("image")
        return image, segmentations, quality_annotation, patient_tuple

    except Exception as exc:
        logger.error("Failed to process image %s: %s", jpg_path.name, exc, exc_info=True)
        tracker.update(success=False)
        tracker.record_error(
            error_type="processing",
            error_message=str(exc),
            item_id=stem,
            item_path=str(jpg_path),
        )
        return None, [], None, None


async def ingest_hei_med() -> OperationStatistics:
    """Main ingestion function for HEI-MED dataset.

    Strategy:
    - Find all .jpg images in DMED/
    - For each image: ingest exudate mask (from .map.gz), vessel mask (_vess.png),
      quality score and patient demographics (.meta)
    - No train/val/test split information available → all assigned to train

    Returns:
        OperationStatistics with success/error counts
    """
    data_root = get_data_root() / "22_HEI-MED"
    dmed_dir = data_root / "DMED"
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
            "HEI-MED (Hamburg Eye Image Database) — colour fundus images with "
            "pixel-level exudate annotations for diabetic macular edema detection. "
            "Each image is accompanied by an exudate probability map (.map.gz), "
            "a vessel segmentation mask (_vess.png), and patient metadata (.meta)."
        ),
    )
    await upsert_dataset(dataset)

    # Step 2: Find all .jpg images
    jpg_images = await asyncio.to_thread(
        find_images, dmed_dir, recursive=False
    )
    # Exclude _vess.png companion files (they end with _vess.png, not .jpg)
    jpg_images = [p for p in jpg_images if not p.stem.endswith("_vess")]
    total_images = len(jpg_images)
    logger.info("Found %d images in %s", total_images, dmed_dir)

    tracker = ProgressTracker(total=total_images, description=f"Ingesting {DATASET_NAME}")

    all_images: List[Image] = []
    all_segmentations: List[SegmentationAnnotation] = []
    all_quality: List[QualityAnnotation] = []
    all_patient_models: List[Patient] = []
    all_patient_image_models: List[PatientImage] = []
    all_image_ids: List[UUID] = []

    # Step 3: Process each image
    for jpg_path in jpg_images:
        image, segs, quality, patient_tuple = await _process_image(
            jpg_path=jpg_path,
            dataset_id=dataset_id,
            tracker=tracker,
        )
        if image is not None:
            all_images.append(image)
            all_image_ids.append(image.image_id)
        all_segmentations.extend(segs)
        if quality is not None:
            all_quality.append(quality)

        if patient_tuple is not None:
            patient_data, img_id = patient_tuple
            patient_id = generate_patient_uuid(
                dataset_id=dataset_id,
                original_patient_id=str(img_id),
            )
            all_patient_models.append(
                Patient(
                    patient_id=patient_id,
                    dataset_id=dataset_id,
                    original_patient_id=str(img_id),
                    age=patient_data.get("age"),
                    sex=patient_data.get("sex"),
                    ethnicity=patient_data.get("ethnicity"),
                    comorbidities=patient_data.get("comorbidities"),
                    created_at=datetime.now(),
                )
            )
            rel_id = generate_patient_image_uuid(patient_id=patient_id, image_id=img_id)
            all_patient_image_models.append(
                PatientImage(
                    relationship_id=rel_id,
                    patient_id=patient_id,
                    image_id=img_id,
                    created_at=datetime.now(),
                )
            )

    # Step 4: Bulk upserts
    logger.info("Upserting %d images...", len(all_images))
    if all_images:
        await bulk_upsert_images(all_images, batch_size=500)

    logger.info("Upserting %d quality annotations...", len(all_quality))
    if all_quality:
        await bulk_upsert_quality_annotations(all_quality, batch_size=500)

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

    logger.info("Registering %d patients...", len(all_patient_models))
    if all_patient_models:
        await bulk_upsert_patients(all_patient_models)
        await bulk_upsert_patient_images(all_patient_image_models)

    # Step 5: Register splits — random 90/10 train+test, then 90/10 train+val
    from chaksudb.ingest.framework.split_assigner import auto_stratified_splits
    logger.info("Registering dataset splits...")
    if all_image_ids:
        await auto_stratified_splits(
            dataset_id=dataset_id,
            split_assignments={"train": all_image_ids},
            split_type="explicit",
        )

    tracker.finish()
    final_stats = tracker.get_statistics()

    logger.info("=" * 80)
    logger.info("Ingestion Summary: %s", DATASET_NAME)
    logger.info("  Total items:      %d", final_stats.total_items)
    logger.info("  Successful:       %d", final_stats.successful_items)
    logger.info("  Failed:           %d", final_stats.failed_items)
    logger.info("  Images:           %d", len(all_images))
    logger.info("  Segmentations:    %d", len(all_segmentations))
    logger.info("  Quality annots:   %d", len(all_quality))
    logger.info("  Patients:         %d", len(all_patient_models))
    if final_stats.errors:
        logger.warning("  Errors (%d):", len(final_stats.errors))
        for err_type, count in final_stats.error_counts.items():
            logger.warning("    %s: %d", err_type, count)
    logger.info("=" * 80)

    return final_stats


async def main():
    """Entry point for script execution."""
    import sys

    log_file = Path("./logs/ingest_22_hei_med.log")
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
    stats = await ingest_hei_med()
    return 0 if stats.failed_items == 0 else 1


if __name__ == "__main__":
    exit(asyncio.run(main()))
