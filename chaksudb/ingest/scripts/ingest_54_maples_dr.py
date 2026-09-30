"""
Ingestion script for MAPLES-DR dataset.

Dataset: MAPLES-DR - MESSIDOR Anatomical and Pathological Labels for
         Explainable Screening of Diabetic Retinopathy
Structure:
  train/{BiomarkerName}/<image_name>.png  — 138 training consensus masks
  test/{BiomarkerName}/<image_name>.png   — 60 test consensus masks
Annotations:
  12 binary consensus segmentation masks per image (annotated by 7 retinologists):
  OpticDisc, OpticCup, Macula, Vessels, Microaneurysms, Hemorrhages,
  Neovascularization, Exudates, CottonWoolSpots, Drusens,
  BrightUncertains, RedUncertains
Tasks: Segmentation (12 retinal structures)

Note: Images are from MESSIDOR-1. Segmentation annotations are attached to
existing MESSIDOR images in the DB (matched by filename). DR and ME grading
are already ingested via MESSIDOR (ICDR_0_4 scale) and are not re-ingested here.
Images absent from the MESSIDOR dataset (36 of 198) are skipped.
"""

import asyncio
import csv
import logging
from pathlib import Path
from typing import Dict, Set
from uuid import UUID

from chaksudb.common.progress import OperationStatistics, ProgressTracker
from chaksudb.config.config import get_data_root
from chaksudb.db.models import Dataset
from chaksudb.db.queries import upsert_dataset, upsert_segmentation_annotation
from chaksudb.db.queries.images import add_image_dataset_membership
from chaksudb.ingest.framework.gen_uuid import generate_dataset_uuid, generate_image_uuid
from chaksudb.ingest.framework.split_assigner import (
    bulk_assign_images_to_split,
    register_standard_splits,
)
from chaksudb.ingest.framework.task_processors.segmentation_processor import (
    get_or_create_annotation_type,
    process_segmentation_from_binary_mask,
)

logger = logging.getLogger(__name__)

DATASET_NAME = "MAPLES-DR"
DATASET_URL = "https://doi.org/10.6084/m9.figshare.24328660"
DATASET_LICENSE = "CC BY 4.0"

# folder name → (annotation_type, lesion_subtype or None, description)
# Structural structures get their own annotation_type (consistent with ORIGA, G1020, STARE, etc.)
# Lesions use annotation_type="lesions" + lesion_subtype (consistent with IDRID, DDR, OIA-DDR)
BIOMARKERS = [
    ("OpticDisc",          "optic_disc", None,               "Optic disc segmentation"),
    ("OpticCup",           "optic_cup",  None,               "Optic cup segmentation"),
    ("Macula",             "macula",     None,               "Macula segmentation"),
    ("Vessels",            "vessels",    None,               "Blood vessel segmentation"),
    ("Microaneurysms",     "lesions",    "MA",               "Microaneurysm segmentation"),
    ("Hemorrhages",        "lesions",    "HE",               "Hemorrhage segmentation"),
    ("Neovascularization", "lesions",    "NV",               "Neovascularization segmentation"),
    ("Exudates",           "lesions",    "EX",               "Hard exudate segmentation"),
    ("CottonWoolSpots",    "lesions",    "SE",               "Soft exudate / cotton wool spot segmentation"),
    ("Drusens",            "lesions",    "DR",               "Drusen segmentation"),
    ("BrightUncertains",   "lesions",    "bright_uncertain", "Uncertain bright lesion segmentation"),
    ("RedUncertains",      "lesions",    "red_uncertain",    "Uncertain red lesion segmentation"),
]


def _load_messidor_image_names(data_root: Path) -> Set[str]:
    """Return the set of MESSIDOR image stems (no extension) from its CSV."""
    csv_path = data_root / "02_MESSIDOR" / "messidor_data.csv"
    names: Set[str] = set()
    if not csv_path.exists():
        logger.warning("MESSIDOR CSV not found at %s — all MAPLES images may be skipped", csv_path)
        return names
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            names.add(row["image_id"].replace(".png", ""))
    return names


def _read_maples_images(data_root: Path) -> Dict[str, str]:
    """Return {image_name_stem: split} for all 198 MAPLES images."""
    image_splits: Dict[str, str] = {}
    for split in ("train", "test"):
        csv_path = data_root / "54_MAPLES" / split / "diagnosis.csv"
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_splits[row["name"]] = split
    return image_splits


async def ingest_maples_dr() -> OperationStatistics:
    """Attach MAPLES-DR segmentation masks to existing MESSIDOR images."""
    data_root = get_data_root()
    maples_root = data_root / "54_MAPLES"
    messidor_dataset_id = generate_dataset_uuid("MESSIDOR")
    maples_dataset_id = generate_dataset_uuid(DATASET_NAME)

    logger.info("=" * 80)
    logger.info("Starting ingestion: %s", DATASET_NAME)
    logger.info("Data root: %s", maples_root)
    logger.info("=" * 80)

    # Step 1: Register MAPLES-DR as a dataset
    await upsert_dataset(Dataset(
        dataset_id=maples_dataset_id,
        dataset_name=DATASET_NAME,
        source_url=DATASET_URL,
        license=DATASET_LICENSE,
        modality_types=["fundus"],
    ))

    # Step 2: Pre-register structural annotation types (lesions type is shared/already registered)
    seen_types: set = set()
    for _, ann_type, _, ann_desc in BIOMARKERS:
        if ann_type not in seen_types:
            await get_or_create_annotation_type(ann_type, ann_desc)
            seen_types.add(ann_type)

    # Step 3: Build image list; filter to those present in MESSIDOR
    messidor_names = _load_messidor_image_names(data_root)
    image_splits = _read_maples_images(data_root)

    valid = {name: split for name, split in image_splits.items() if name in messidor_names}
    skipped_names = set(image_splits) - set(valid)

    logger.info("MAPLES total: %d | linked to MESSIDOR: %d | skipped: %d",
                len(image_splits), len(valid), len(skipped_names))
    if skipped_names:
        logger.warning("Skipped images (not in MESSIDOR CSV): %s", sorted(skipped_names))

    total_masks = len(valid) * len(BIOMARKERS)
    tracker = ProgressTracker(total=total_masks, description=f"Ingesting {DATASET_NAME}")

    # Step 4: Process masks image-by-image, biomarker-by-biomarker
    # MAPLES-DR reuses MESSIDOR's canonical images; record cross-dataset membership so
    # exports filtered to "MAPLES-DR" still resolve these annotations (see
    # image_dataset_memberships and the membership-aware export filter).
    split_to_image_ids: Dict[str, list] = {"train": [], "test": []}
    for image_name, split in valid.items():
        # MESSIDOR ingestion uses filename WITH .png as original_image_id
        messidor_image_id: UUID = generate_image_uuid(messidor_dataset_id, image_name + ".png")
        split_dir = maples_root / split

        await add_image_dataset_membership(
            messidor_image_id, maples_dataset_id, original_image_id=image_name
        )
        split_to_image_ids[split].append(messidor_image_id)

        for folder, ann_type, lesion_subtype, _ in BIOMARKERS:
            mask_path = split_dir / folder / f"{image_name}.png"
            item_id = f"{image_name}/{folder}"

            if not mask_path.exists():
                tracker.update(count=1, success=False)
                tracker.record_error(
                    error_type="mask_not_found",
                    error_message=f"Mask missing: {mask_path}",
                    item_id=item_id,
                )
                continue

            try:
                seg = await process_segmentation_from_binary_mask(
                    mask_path=mask_path,
                    annotation_type=ann_type,
                    image_id=messidor_image_id,
                    lesion_subtype=lesion_subtype,
                    annotation_method="manual",
                    dataset_name=DATASET_NAME,
                )
                await upsert_segmentation_annotation(seg)
                tracker.update(count=1, success=True)
            except Exception as e:
                tracker.update(count=1, success=False)
                tracker.record_error(
                    error_type="segmentation_error",
                    error_message=str(e),
                    item_id=item_id,
                )
                logger.error("Failed %s/%s: %s", image_name, folder, e)

    # Step 5: Register and assign MAPLES-DR's own train/test splits (paper-defined).
    # These reference the canonical MESSIDOR images via their own split_ids, so they
    # coexist with any MESSIDOR splits on the same images.
    splits = await register_standard_splits(
        maples_dataset_id,
        split_type="explicit",
        train_count=len(split_to_image_ids["train"]),
        test_count=len(split_to_image_ids["test"]),
    )
    if split_to_image_ids["train"]:
        await bulk_assign_images_to_split(split_to_image_ids["train"], splits["train"])
    if split_to_image_ids["test"]:
        await bulk_assign_images_to_split(split_to_image_ids["test"], splits["test"])
    logger.info(
        "Assigned MAPLES splits — train: %d, test: %d",
        len(split_to_image_ids["train"]), len(split_to_image_ids["test"]),
    )

    tracker.finish()
    stats = tracker.get_statistics()

    logger.info("=" * 80)
    logger.info("Ingestion Summary:")
    logger.info("  Total masks: %d", stats.total_items)
    logger.info("  Successful:  %d", stats.successful_items)
    logger.info("  Failed:      %d", stats.failed_items)
    logger.info("  Images skipped (not in MESSIDOR): %d", len(skipped_names))
    logger.info("=" * 80)

    return stats


async def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    try:
        stats = await ingest_maples_dr()
        if stats.failed_items > 0:
            logger.error("Ingestion completed with %d errors", stats.failed_items)
            return 1
        logger.info("Ingestion completed successfully!")
        return 0
    except Exception as e:
        logger.exception("Fatal error: %s", e)
        return 1


if __name__ == "__main__":
    exit(asyncio.run(main()))
