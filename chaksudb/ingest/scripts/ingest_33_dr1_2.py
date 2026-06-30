"""
Ingestion script for DR1-2 dataset.

Dataset: DR1-2 - Diabetic Retinopathy dataset with lesion type classifications
Structure: Folder-based classification with disease category folders
Annotations:
  - Multi-class classification (lesion types) from root label folders
  - Multi-class classification (referral decision) from DR2-images-by-referral/
  - Images only (no classification) from DR1-additional-marked-images/
  - Bounding box localizations from markings/*.txt (tab-separated, Spanish names)
"""

import asyncio
import logging
from pathlib import Path
from typing import List

from chaksudb.common.progress import ProgressTracker, OperationStatistics
from chaksudb.config.config import get_data_root
from chaksudb.db.models import Dataset, Image, ClassificationAnnotation, LocalizationAnnotation
from chaksudb.db.queries import (
    upsert_dataset,
    bulk_upsert_images,
    bulk_upsert_classification_annotations,
    bulk_upsert_localization_annotations,
)
from chaksudb.ingest.framework import (
    find_images,
    process_folder_tree,
    get_image_metadata_dict,
)
from chaksudb.ingest.framework.task_processors.classification_processor import (
    process_classification,
)
from chaksudb.ingest.framework.localization.text_parsers import parse_tsv_bounding_boxes
from chaksudb.ingest.framework.gen_uuid import (
    generate_dataset_uuid,
    generate_image_uuid,
    generate_localization_uuid,
)
from chaksudb.ingest.framework.task_processors.localization_processor import (
    compute_coordinates_hash,
)
from chaksudb.ingest.framework.split_assigner import auto_stratified_splits

logger = logging.getLogger(__name__)

# Dataset metadata
DATASET_NAME = "DR1-2"
DATASET_URL = "https://doi.org/10.6084/m9.figshare.953671.v3"
DATASET_LICENSE = "Unknown"

# Mapping from folder names to normalized class names (lesion_type task)
FOLDER_TO_CLASS = {
    "Cotton-wool Spots": "cotton_wool_spots",
    "Deep Hemorrhages": "deep_hemorrhages",
    "Drusen": "drusen",
    "Hard Exudates": "hard_exudates",
    "Normal Images": "normal",
    "Red Lesions": "red_lesions",
    "Superficial Hemorrhages": "superficial_hemorrhages",
}

# DR2 referral task: folder name → class value
REFERRAL_FOLDER_TO_CLASS = {
    "Referable": "referable",
    "Non-Referable": "non_referable",
}

# Deterministic index->label maps so multi_class class_index is always populated.
LESION_CLASS_LABELS = {i: v for i, v in enumerate(sorted(set(FOLDER_TO_CLASS.values())))}
REFERRAL_CLASS_LABELS = {i: v for i, v in enumerate(sorted(set(REFERRAL_FOLDER_TO_CLASS.values())))}

# DR1 additional images: ingest images only, no classification
ADDITIONAL_IMAGES_FOLDER = "DR1-additional-marked-images"

# Mapping from Portuguese/Spanish lesion names in markings/*.txt to
# (target_structure, lesion_subtype). Names not in this map are stored
# with the raw name as target_structure and lesion_subtype=None.
MARKING_CLASS_MAP: dict[str, tuple[str, str | None]] = {
    "exsudato-duro":                   ("lesions", "EX"),
    "hemorragia-profunda":             ("lesions", "HE"),
    "hemorragia-superficial":          ("lesions", "HE"),
    "mancha-algodonosa":               ("lesions", "SE"),
    "drusas-maculares":                ("lesions", "drusen"),
    "escavacao-aumentada":             ("optic_disc", None),
    "borramento-de-papila":            ("optic_disc", None),
    "atrofia-foveal":                  ("fovea", None),
    "aumento-de-tortuosidade-vascular": ("vessels", None),
    "cicatriz-de-coriorretinite":      ("lesions", "scar"),
    "dr-seroso-macular":               ("lesions", "serous_dr"),
    "rarefacao-epr":                   ("lesions", "rpe_rarefaction"),
    "imagem-normal":                   ("normal", None),
}


async def ingest_dr1_2() -> OperationStatistics:
    """
    Main ingestion function for DR1-2 dataset.

    Strategy:
    - Use process_folder_tree() to walk disease category folders
    - Extract disease category from folder name
    - Create multi-class classifications for lesion types
    - Bulk upsert all annotations

    Returns:
        OperationStatistics with success/error counts
    """
    data_root = get_data_root() / "33_DR1-2"
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

    # Step 2: Count total images for progress tracking
    logger.info("Counting images across all folders...")
    total_images = 0

    # Lesion type folders
    for folder_name in FOLDER_TO_CLASS.keys():
        folder_path = data_root / folder_name
        if folder_path.exists():
            images_in_folder = await asyncio.to_thread(
                find_images, folder_path, recursive=False
            )
            total_images += len(images_in_folder)
            logger.info(f"  {folder_name}: {len(images_in_folder)} images")
        else:
            logger.warning(f"Folder not found: {folder_path}")

    # Referral folders
    for folder_name in REFERRAL_FOLDER_TO_CLASS.keys():
        folder_path = data_root / folder_name
        if folder_path.exists():
            images_in_folder = await asyncio.to_thread(
                find_images, folder_path, recursive=False
            )
            total_images += len(images_in_folder)
            logger.info(f"  {folder_name}: {len(images_in_folder)} images")
        else:
            logger.warning(f"Folder not found: {folder_path}")

    # Additional images (no label)
    additional_path = data_root / ADDITIONAL_IMAGES_FOLDER
    if additional_path.exists():
        additional_images = await asyncio.to_thread(
            find_images, additional_path, recursive=False
        )
        total_images += len(additional_images)
        logger.info(f"  {ADDITIONAL_IMAGES_FOLDER}: {len(additional_images)} images (no label)")
    else:
        logger.warning(f"Folder not found: {additional_path}")

    logger.info(f"Total images found: {total_images}")

    # Count markings files up front so the tracker total includes the bounding-box step
    # (otherwise localization work runs silently after the counted work is "done").
    markings_dir = data_root / "markings"
    n_markings = len(list(markings_dir.glob("*.txt"))) if markings_dir.exists() else 0
    logger.info(f"Total marking files found: {n_markings}")

    # Step 3: Setup progress tracker
    tracker = ProgressTracker(
        total=total_images + n_markings, description=f"Ingesting {DATASET_NAME}"
    )

    # Collect for bulk upsert
    all_images: List[Image] = []
    all_classifications: List[ClassificationAnnotation] = []
    image_labels: dict = {}  # image_id -> folder class label (for stratified splits)

    async def handle_image(file_path: Path, rel_path: Path, depth: int):
        """
        Process each image. Three cases:
          1. Root label folder (depth=1) → lesion_type classification
          2. DR2-images-by-referral/<negative|positive>/ → referral classification
          3. DR1-additional-marked-images/ → image only, no classification
        """
        if depth == 0:
            logger.warning(f"Skipping file in root: {file_path.name}")
            tracker.update(success=False)
            tracker.record_error(
                error_type="unknown_location",
                error_message="File not in a recognized folder",
                item_id=file_path.stem,
                item_path=str(file_path),
            )
            return

        folder_name = rel_path.parts[0]

        try:
            image_id = generate_image_uuid(dataset_id, file_path.stem)
            image = Image(
                image_id=image_id,
                dataset_id=dataset_id,
                original_image_id=file_path.stem,
                **get_image_metadata_dict(file_path),
                modality="fundus",
            )
            all_images.append(image)

            if folder_name in FOLDER_TO_CLASS:
                # Case 1: lesion type classification
                classifications = await process_classification(
                    class_value=FOLDER_TO_CLASS[folder_name],
                    task_type="multi_class",
                    task_name="lesion_type",
                    class_name="lesion_type",
                    image_id=image_id,
                    class_labels=LESION_CLASS_LABELS,
                    raw_data_id=None,
                    expert_annotation_id=None,
                    annotation_method="manual",
                    provenance_chain_id=None,
                )
                all_classifications.extend(classifications)
                image_labels[image_id] = FOLDER_TO_CLASS[folder_name]

            elif folder_name in REFERRAL_FOLDER_TO_CLASS:
                # Case 2: referral classification
                classifications = await process_classification(
                    class_value=REFERRAL_FOLDER_TO_CLASS[folder_name],
                    task_type="multi_class",
                    task_name="referral",
                    class_name="referral",
                    image_id=image_id,
                    class_labels=REFERRAL_CLASS_LABELS,
                    raw_data_id=None,
                    expert_annotation_id=None,
                    annotation_method="manual",
                    provenance_chain_id=None,
                )
                all_classifications.extend(classifications)
                image_labels[image_id] = REFERRAL_FOLDER_TO_CLASS[folder_name]

            elif folder_name == ADDITIONAL_IMAGES_FOLDER:
                # Case 3: image only, no classification
                pass

            else:
                logger.warning(f"Unknown folder: {folder_name} (file: {file_path.name})")
                tracker.update(success=False)
                tracker.record_error(
                    error_type="unknown_folder",
                    error_message=f"Folder not recognized: {folder_name}",
                    item_id=file_path.stem,
                    item_path=str(file_path),
                )
                return

            tracker.update(success=True)
            tracker.record_success("image")

        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}", exc_info=True)
            tracker.update(success=False)
            tracker.record_error(
                error_type="processing",
                error_message=str(e),
                item_id=file_path.stem,
                item_path=str(file_path),
            )

    # Step 4: Process folder tree with automatic per-file provenance
    logger.info("Processing images from folder structure...")
    stats = await process_folder_tree(
        root_dir=data_root,
        dataset_id=dataset_id,
        unified_annotation_type="classification",  # Primary annotation type
        process_file_fn=handle_image,
        file_extensions={".jpg", ".JPG", ".jpeg", ".JPEG", ".tif", ".TIF"},
        recursive=True,
        include_dirs=False,
        progress_tracker=tracker,
        skip_errors=True,
    )

    # Step 5: Bulk upsert - images first, then classifications (due to foreign key constraint)
    logger.info(f"Upserting {len(all_images)} images...")
    if all_images:
        await bulk_upsert_images(all_images, batch_size=1000)
    
    logger.info(f"Upserting {len(all_classifications)} classifications...")
    if all_classifications:
        await bulk_upsert_classification_annotations(all_classifications, batch_size=1000)

    # Step 6: Ingest bounding box localizations from markings/*.txt
    # (markings_dir / n_markings were computed earlier for the tracker total)
    all_localizations: list[LocalizationAnnotation] = []
    # Index ingested images by stem so we skip markings whose image is absent from disk
    ingested_by_stem = {img.original_image_id: img.image_id for img in all_images}
    if markings_dir.exists():
        txt_files = sorted(markings_dir.glob("*.txt"))
        logger.info(f"Processing {len(txt_files)} marking files from {markings_dir}")
        skipped_no_image = 0
        for txt_path in txt_files:
            # One tracker tick per markings file so localization progress is reported.
            tracker.update(success=True)
            image_id = ingested_by_stem.get(txt_path.stem)
            if image_id is None:
                skipped_no_image += 1
                logger.debug(f"No ingested image for marking {txt_path.name}, skipping")
                continue
            try:
                boxes_by_class = parse_tsv_bounding_boxes(txt_path)
            except Exception as e:
                logger.warning(f"Could not parse {txt_path.name}: {e}")
                continue
            for raw_class, boxes in boxes_by_class.items():
                target_structure, lesion_subtype = MARKING_CLASS_MAP.get(
                    raw_class, (raw_class, None)
                )
                if target_structure == "normal":
                    continue  # skip "imagem-normal" lines — no spatial annotation
                for box in boxes:
                    coords = box["coordinates"]
                    loc_id = generate_localization_uuid(
                        image_id=image_id,
                        localization_type="bounding_box",
                        target_structure=target_structure,
                        coordinates_hash=compute_coordinates_hash(coords),
                    )
                    all_localizations.append(LocalizationAnnotation(
                        localization_id=loc_id,
                        image_id=image_id,
                        localization_type="bounding_box",
                        target_structure=target_structure,
                        lesion_subtype=lesion_subtype,
                        coordinates=coords,
                        annotation_method="manual",
                    ))
        if skipped_no_image:
            logger.warning(f"Skipped {skipped_no_image} marking files with no matching image on disk")
        logger.info(f"Upserting {len(all_localizations)} localization bounding boxes...")
        if all_localizations:
            await bulk_upsert_localization_annotations(all_localizations, batch_size=1000)
    else:
        logger.warning(f"markings/ directory not found: {markings_dir}")

    # Step 7: Register splits
    all_image_ids = [img.image_id for img in all_images]
    if all_image_ids:
        logger.info("Registering dataset splits...")
        await auto_stratified_splits(
            dataset_id=dataset_id,
            split_assignments={"train": all_image_ids},
            labels=image_labels,
            split_type="explicit",
        )

    # Finish progress tracking
    tracker.finish()
    final_stats = tracker.get_statistics()

    # Log final summary
    logger.info("=" * 80)
    logger.info("Ingestion Summary:")
    logger.info(f"  Total items: {final_stats.total_items}")
    logger.info(f"  Successful: {final_stats.successful_items}")
    logger.info(f"  Failed: {final_stats.failed_items}")
    logger.info(f"  Skipped: {final_stats.skipped_items}")
    logger.info(f"  Images: {len(all_images)}")
    logger.info(f"  Classifications: {len(all_classifications)}")
    logger.info(f"    lesion_type: {sum(1 for c in all_classifications if c.class_name == 'lesion_type')}")
    logger.info(f"    referral: {sum(1 for c in all_classifications if c.class_name == 'referral')}")
    logger.info(f"  Localizations (bounding boxes): {len(all_localizations)}")
    logger.info(f"  Images without classification: {len(all_images) - len(all_classifications)}")
    if final_stats.errors:
        logger.warning(f"  Total errors: {len(final_stats.errors)}")
        for error_type, count in final_stats.error_counts.items():
            logger.warning(f"    {error_type}: {count}")
    logger.info("=" * 80)

    return final_stats


async def main():
    """Entry point for script execution."""
    import sys
    from pathlib import Path
    log_file = Path("./logs/ingest_33_dr1_2.log")
    log_file.touch(exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
        logging.FileHandler(log_file, mode='w'), 
        logging.StreamHandler(sys.stdout),          
        ],
    )
    stats = await ingest_dr1_2()

    logger.info("=" * 80)
    logger.info(f"Ingestion complete!")
    logger.info(f"Total: {stats.total_items}")
    logger.info(f"Successful: {stats.successful_items}")
    logger.info(f"Failed: {stats.failed_items}")
    logger.info(f"Errors: {len(stats.errors)}")
    logger.info("=" * 80)

    return 0 if stats.failed_items == 0 else 1


if __name__ == "__main__":
    exit(asyncio.run(main()))
