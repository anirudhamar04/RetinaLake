"""
Ingestion script for justRAIGS dataset.

Dataset: justRAIGS — Justified Referral in AI Glaucoma Screening
Source:  https://justraigs.grand-challenge.org/
Structure:
  JustRAIGS_Train_labels.csv   — semicolon-delimited labels for 101 423 eyes
  train/{0..N}/                 — images named {Eye ID}.JPG, split into
                                  sequential chunk subdirectories (NOT by last digit)

CSV columns:
  Eye ID          — image identifier (e.g. TRAIN000000)
  Final Label     — RG (Referable Glaucoma) or NRG (Non-Referable Glaucoma)
  Fellow Eye ID   — ID of the contralateral eye
  Age             — patient age
  Label G1        — G1's individual decision (RG / NRG)
  Label G2        — G2's individual decision (RG / NRG)
  Label G3        — G3's tiebreaker decision (only present when G1 and G2 disagree)
  G1 ANRS … G1 LC — 10 binary features for grader 1
  G2 ANRS … G2 LC — 10 binary features for grader 2
  G3 ANRS … G3 LC — 10 binary features for grader 3 (empty when G3 not called)

  Feature abbreviations (all binary 0/1):
    ANRS   Appearance of Neuroretinal Rim Superiorly
    ANRI   Appearance of Neuroretinal Rim Inferiorly
    RNFLDS Retinal Nerve Fiber Layer Defect Superiorly
    RNFLDI Retinal Nerve Fiber Layer Defect Inferiorly
    BCLVS  Baring of Circumlinear Vessel Superiorly
    BCLVI  Baring of Circumlinear Vessel Inferiorly
    NVT    Nasalisation of Vessel Trunk
    DH     Disc Hemorrhages
    LD     Laminar Dots
    LC     Large Cup

Grading protocol:
  G1 and G2 are the two primary graders (independent, manual).
  G3 is the tiebreaker: only called when G1 and G2 disagree; their label is
  adjudicated (resolves the conflict). Final Label is the consensus result.

Annotations stored:
  - Binary glaucoma classification (Final Label) as consensus annotation
  - G1/G2 per-image binary classification + 10-feature multi-label (annotation_method="manual")
  - G3 per-image binary classification + 10-feature multi-label (annotation_method="adjudicated")
  All per-grader annotations are linked to their respective ExpertAnnotation record.
"""

import asyncio
import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from chaksudb.common.progress import ProgressTracker, OperationStatistics
from chaksudb.config.config import get_data_root
from datetime import datetime

from chaksudb.db.models import (
    ClassificationAnnotation,
    Dataset,
    Expert,
    ExpertAnnotation,
    Image,
    Patient,
    PatientImage,
)
from chaksudb.db.queries import (
    upsert_dataset,
    bulk_upsert_images,
    bulk_upsert_classification_annotations,
    bulk_upsert_expert_annotations,
    upsert_expert,
)
from chaksudb.db.queries.images import bulk_upsert_patient_images
from chaksudb.db.queries.patients import bulk_upsert_patients
from chaksudb.ingest.framework import (
    get_image_metadata_dict,
)
from chaksudb.ingest.framework.gen_uuid import (
    generate_dataset_uuid,
    generate_image_uuid,
    generate_expert_uuid,
    generate_expert_annotation_uuid,
    generate_patient_uuid,
    generate_patient_image_uuid,
)
from chaksudb.ingest.framework.provenance_context import (
    set_provenance_context,
    reset_provenance_context,
)
from chaksudb.ingest.framework.raw_file_helpers import register_csv_file
from chaksudb.ingest.framework.split_assigner import (
    auto_stratified_splits,
)
from chaksudb.ingest.framework.task_processors.classification_processor import (
    process_classification,
)

logger = logging.getLogger(__name__)

# Dataset metadata
DATASET_NAME = "justRAIGS"
DATASET_URL = "https://zenodo.org/records/10035093"
DATASET_LICENSE = "Research/Academic Use"

# Mapping from CSV column abbreviation → full descriptive name (stored as JSON key)
GRADER_FEATURE_NAMES = {
    "ANRS":   "appearance_neuroretinal_rim_superiorly",
    "ANRI":   "appearance_neuroretinal_rim_inferiorly",
    "RNFLDS": "retinal_nerve_fiber_layer_defect_superiorly",
    "RNFLDI": "retinal_nerve_fiber_layer_defect_inferiorly",
    "BCLVS":  "baring_circumlinear_vessel_superiorly",
    "BCLVI":  "baring_circumlinear_vessel_inferiorly",
    "NVT":    "nasalisation_of_vessel_trunk",
    "DH":     "disc_hemorrhages",
    "LD":     "laminar_dots",
    "LC":     "large_cup",
}
GRADER_FEATURES = list(GRADER_FEATURE_NAMES.keys())

# Primary graders (independent, manual) and tiebreaker (adjudicated)
PRIMARY_GRADERS = ["G1", "G2"]
TIEBREAKER_GRADER = "G3"
GRADERS = PRIMARY_GRADERS + [TIEBREAKER_GRADER]


def _build_image_index(data_root: Path) -> Dict[str, Path]:
    """Scan all train subdirectories once and return a dict of eye_id → path.

    Images are stored in sequential chunks under train/0/, train/1/, etc.
    The subdirectory is NOT based on the last digit of the Eye ID.
    """
    index: Dict[str, Path] = {}
    train_dir = data_root / "train"
    if not train_dir.exists():
        return index
    for subdir in sorted(train_dir.iterdir()):
        if not subdir.is_dir():
            continue
        for f in subdir.iterdir():
            if f.suffix.lower() in (".jpg", ".jpeg"):
                index[f.stem] = f
    return index


def _find_image(index: Dict[str, Path], eye_id: str) -> Optional[Path]:
    """Locate the image file for a given Eye ID using the pre-built index."""
    return index.get(eye_id)


def _parse_binary_feature(value: str) -> Optional[bool]:
    """Parse a binary feature column value (0/1/empty string)."""
    v = str(value).strip()
    if v == "1":
        return True
    if v == "0":
        return False
    return None  # Missing / not graded


async def register_experts(dataset_id: UUID) -> Dict[str, UUID]:
    """Register graders for justRAIGS.

    G1 and G2 are the primary independent graders.
    G3 is the tiebreaker called only when G1 and G2 disagree.
    """
    expert_ids: Dict[str, UUID] = {}
    grader_names = {
        "G1": "justRAIGS G1 (primary grader)",
        "G2": "justRAIGS G2 (primary grader)",
        "G3": "justRAIGS G3 (tiebreaker grader)",
    }
    for grader_key in GRADERS:
        expert_name = grader_names[grader_key]
        expert_id = generate_expert_uuid(
            dataset_id=dataset_id,
            model_id=None,
            expert_name=expert_name,
        )
        expert = Expert(
            expert_id=expert_id,
            expert_name=expert_name,
            dataset_id=dataset_id,
            model_id=None,
        )
        await upsert_expert(expert)
        expert_ids[grader_key] = expert_id
        logger.info("Registered expert: %s", expert_name)
    return expert_ids


async def ingest_justraigs() -> OperationStatistics:
    """Main ingestion function for justRAIGS dataset.

    Strategy:
    - Read semicolon-delimited CSV
    - Register 3 experts (G1, G2, G3)
    - For each row:
        * Binary consensus classification (Final Label)
        * Per-grader classification + 10-feature multi-label (if grader present)
    - All images are in train split (no official test split with labels)

    Returns:
        OperationStatistics with success/error counts
    """
    data_root = get_data_root() / "39_justRAIGS"
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
            "justRAIGS (Justified Referral in AI Glaucoma Screening) — ~101K colour "
            "fundus images annotated by up to 3 graders for referable glaucoma. "
            "Each image has a consensus Final Label (RG/NRG) plus per-grader labels "
            "and 10 binary structural features (e.g. ANRS, RNFLDS, DH, LC)."
        ),
    )
    await upsert_dataset(dataset)

    # Step 2: Register experts
    logger.info("Registering experts...")
    expert_ids = await register_experts(dataset_id)

    # Step 3: Read semicolon-delimited CSV and count rows
    csv_path = data_root / "JustRAIGS_Train_labels.csv"

    def _read_semicolon_csv(path: Path) -> List[Dict[str, Any]]:
        with open(path, "r", encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh, delimiter=";")
            return list(reader)

    rows = await asyncio.to_thread(_read_semicolon_csv, csv_path)
    total_count = len(rows)
    logger.info("Found %d rows in %s", total_count, csv_path.name)

    tracker = ProgressTracker(total=total_count, description=f"Ingesting {DATASET_NAME}")

    # Build image index once (folders are sequential chunks, not last-digit based)
    logger.info("Building image index...")
    image_index = await asyncio.to_thread(_build_image_index, data_root)
    logger.info("Image index built: %d images found", len(image_index))

    # Collections for bulk upsert
    all_images: List[Image] = []
    all_classifications: List[ClassificationAnnotation] = []
    all_expert_annotations: List[ExpertAnnotation] = []
    all_image_ids: List[UUID] = []
    image_labels: dict = {}  # image_id → final label (RG/NRG) for stratified splitting
    all_patient_models: List[Patient] = []
    all_patient_image_models: List[PatientImage] = []

    async def process_row(row: dict, idx: int) -> None:
        """Process a single CSV row."""
        try:
            eye_id = str(row.get("Eye ID", "")).strip()
            if not eye_id:
                tracker.update(success=False)
                tracker.record_error(
                    error_type="missing_eye_id",
                    error_message=f"Empty Eye ID at row {idx}",
                    item_id=f"row_{idx}",
                )
                return

            final_label = str(row.get("Final Label", "")).strip().upper()
            if final_label not in ("RG", "NRG"):
                tracker.update(success=False)
                tracker.record_error(
                    error_type="invalid_final_label",
                    error_message=f"Final Label must be RG or NRG, got: {final_label!r}",
                    item_id=eye_id,
                )
                return

            # Find image file
            image_path = _find_image(image_index, eye_id)
            if image_path is None:
                tracker.update(success=False)
                tracker.record_error(
                    error_type="file_not_found",
                    error_message=f"Image not found for Eye ID: {eye_id}",
                    item_id=eye_id,
                )
                return

            image_id = generate_image_uuid(dataset_id, eye_id)

            image = Image(
                image_id=image_id,
                dataset_id=dataset_id,
                original_image_id=eye_id,
                **get_image_metadata_dict(image_path),
                modality="fundus",
            )
            all_images.append(image)
            all_image_ids.append(image_id)
            image_labels[image_id] = final_label

            # Consensus binary classification (Final Label)
            is_rg = final_label == "RG"
            consensus_cls_list = await process_classification(
                class_value=is_rg,
                task_type="binary",
                class_name="glaucoma",
                image_id=image_id,
                class_labels={True: "RG", False: "NRG"},
                annotation_method="consensus",
            )
            all_classifications.extend(consensus_cls_list)

            # Per-grader annotations
            # G1/G2 are independent primary graders → "manual"
            # G3 is the tiebreaker called only when G1/G2 disagree → "adjudicated"
            for grader_key in GRADERS:
                grader_label_raw = str(row.get(f"Label {grader_key}", "")).strip().upper()
                if not grader_label_raw:
                    continue

                if grader_label_raw not in ("RG", "NRG"):
                    logger.debug(
                        "Skipping grader %s for %s: invalid label %r",
                        grader_key,
                        eye_id,
                        grader_label_raw,
                    )
                    continue

                expert_ann_id = expert_ann_ids[grader_key]
                grader_method = "consensus" if grader_key == TIEBREAKER_GRADER else "manual"

                # Per-grader binary classification
                is_rg_grader = grader_label_raw == "RG"
                grader_cls_list = await process_classification(
                    class_value=is_rg_grader,
                    task_type="binary",
                    class_name="glaucoma",
                    image_id=image_id,
                    class_labels={True: "RG", False: "NRG"},
                    expert_annotation_id=expert_ann_id,
                    annotation_method=grader_method,
                )
                all_classifications.extend(grader_cls_list)

                # Per-grader 10-feature multi-label classification
                feature_values: Dict[str, bool] = {}
                for feat in GRADER_FEATURES:
                    col = f"{grader_key} {feat}"
                    parsed = _parse_binary_feature(row.get(col, ""))
                    if parsed is not None:
                        feature_values[GRADER_FEATURE_NAMES[feat]] = parsed

                if feature_values:
                    features_cls_list = await process_classification(
                        class_value=feature_values,
                        task_type="multi_label",
                        class_name=f"glaucoma_features_{grader_key.lower()}",
                        image_id=image_id,
                        expert_annotation_id=expert_ann_id,
                        annotation_method=grader_method,
                    )
                    all_classifications.extend(features_cls_list)

            # Patient — only if age is present
            age_raw = str(row.get("Age", "")).strip()
            age = None
            try:
                if age_raw:
                    age = int(float(age_raw))
            except (ValueError, TypeError):
                pass
            if age is not None:
                patient_id = generate_patient_uuid(
                    dataset_id=dataset_id,
                    original_patient_id=eye_id,
                )
                all_patient_models.append(
                    Patient(
                        patient_id=patient_id,
                        dataset_id=dataset_id,
                        original_patient_id=eye_id,
                        age=age,
                        created_at=datetime.now(),
                    )
                )
                rel_id = generate_patient_image_uuid(patient_id=patient_id, image_id=image_id)
                all_patient_image_models.append(
                    PatientImage(
                        relationship_id=rel_id,
                        patient_id=patient_id,
                        image_id=image_id,
                        created_at=datetime.now(),
                    )
                )

            tracker.update(success=True)
            tracker.record_success("image")

        except Exception as exc:
            logger.error("Failed to process row %d (%s): %s", idx, row.get("Eye ID", "?"), exc, exc_info=True)
            tracker.update(success=False)
            tracker.record_error(
                error_type="processing",
                error_message=str(exc),
                item_id=str(row.get("Eye ID", f"row_{idx}")),
            )

    # Step 4: Register CSV for provenance
    logger.info("Processing annotations...")
    raw_file_id, chain_id = await register_csv_file(
        csv_path, dataset_id, "classification"
    )
    logger.info("CSV registered: raw_file_id=%s, chain_id=%s", raw_file_id, chain_id)

    # Pre-create one ExpertAnnotation per grader (session-level, keyed by expert + source file).
    # All per-image classifications for a grader share the same expert_annotation_id.
    expert_ann_ids: Dict[str, UUID] = {}
    for grader_key in GRADERS:
        expert_id = expert_ids[grader_key]
        ann_id = generate_expert_annotation_uuid(
            expert_id=expert_id,
            annotation_task="classification",
            raw_data_id=raw_file_id,
        )
        all_expert_annotations.append(
            ExpertAnnotation(
                expert_annotation_id=ann_id,
                expert_id=expert_id,
                annotation_task="classification",
                raw_data_id=raw_file_id,
            )
        )
        expert_ann_ids[grader_key] = ann_id

    token_raw, token_chain = set_provenance_context(raw_file_id, chain_id)
    try:
        for idx, row in enumerate(rows):
            await process_row(row, idx)
    finally:
        reset_provenance_context(token_raw, token_chain)

    # Step 5: Bulk upserts
    logger.info("Upserting %d images...", len(all_images))
    if all_images:
        await bulk_upsert_images(all_images, batch_size=1000)

    logger.info("Upserting %d expert annotations...", len(all_expert_annotations))
    if all_expert_annotations:
        await bulk_upsert_expert_annotations(all_expert_annotations, batch_size=1000)

    logger.info("Upserting %d classification annotations...", len(all_classifications))
    if all_classifications:
        await bulk_upsert_classification_annotations(all_classifications, batch_size=1000)

    logger.info("Upserting %d patients...", len(all_patient_models))
    if all_patient_models:
        await bulk_upsert_patients(all_patient_models, batch_size=1000)
        await bulk_upsert_patient_images(all_patient_image_models, batch_size=1000)

    # Step 6: Splits — stratified 90/10 train+test, then 90/10 train+val
    logger.info("Registering dataset splits...")
    if all_image_ids:
        await auto_stratified_splits(
            dataset_id=dataset_id,
            split_assignments={"train": all_image_ids},
            labels=image_labels,
            split_type="explicit",
        )

    tracker.finish()
    final_stats = tracker.get_statistics()

    rg_count = sum(
        1 for c in all_classifications
        if c.class_name == "glaucoma"
        and c.expert_annotation_id is None
        and c.class_value.get("glaucoma") is True
    )
    nrg_count = sum(
        1 for c in all_classifications
        if c.class_name == "glaucoma"
        and c.expert_annotation_id is None
        and c.class_value.get("glaucoma") is False
    )

    logger.info("=" * 80)
    logger.info("Ingestion Summary: %s", DATASET_NAME)
    logger.info("  Total items:        %d", final_stats.total_items)
    logger.info("  Successful:         %d", final_stats.successful_items)
    logger.info("  Failed:             %d", final_stats.failed_items)
    logger.info("  Images:             %d", len(all_images))
    logger.info("  Classifications:    %d", len(all_classifications))
    logger.info("    RG (consensus):   %d", rg_count)
    logger.info("    NRG (consensus):  %d", nrg_count)
    logger.info("  Expert annotations: %d", len(all_expert_annotations))
    logger.info("  Patients (w/ age):  %d", len(all_patient_models))
    if final_stats.errors:
        logger.warning("  Errors (%d):", len(final_stats.errors))
        for err_type, count in final_stats.error_counts.items():
            logger.warning("    %s: %d", err_type, count)
    logger.info("=" * 80)

    return final_stats


async def main():
    """Entry point for script execution."""
    import sys

    log_file = Path("./logs/ingest_39_justraigs.log")
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
    stats = await ingest_justraigs()
    return 0 if stats.failed_items == 0 else 1


if __name__ == "__main__":
    exit(asyncio.run(main()))
