"""
Ingestion script for OIA-DDR dataset.

Dataset: OIA-DDR (Ophthalmology Image Analysis - Diabetic Retinopathy Detection)
Structure: Multi-task dataset with three annotation types:
  - DR_grading/: train/test/val splits with .txt files (format: filename.jpg grade)
  - lesion_detection/: train/test/val splits with XML files (Pascal VOC format)
  - lesion_segmentation/: train/test/val splits with image/ and label/ folders
    - label/ contains subfolders: EX, HE, MA, SE (4 lesion types)
Annotations: 
  - DR grading (ICDR scale)
  - Lesion detection (localization from XML)
  - Lesion segmentation (4 types: EX, HE, MA, SE)
Tasks: DR grading, Localization (lesion detection), Segmentation (4 lesion types)
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional
from uuid import UUID

from chaksudb.common.progress import ProgressTracker, OperationStatistics
from chaksudb.config.config import get_data_root
from chaksudb.db.models import (
    Dataset,
    DiseaseGrading,
    Image,
    LocalizationAnnotation,
    SegmentationAnnotation,
)
from chaksudb.db.queries import (
    bulk_upsert_disease_gradings,
    bulk_upsert_images,
    bulk_upsert_localization_annotations,
    upsert_dataset,
    upsert_segmentation_annotation,
)
from chaksudb.ingest.framework import (
    find_files_by_extension,
    find_images,
    get_image_metadata_dict,
    process_folder_tree,
    process_text_file,
)
from chaksudb.ingest.framework.gen_uuid import (
    generate_dataset_uuid,
    generate_image_uuid,
)
from chaksudb.ingest.framework.provenance_context import (
    get_current_provenance,
    set_provenance_context,
    reset_provenance_context,
)
from chaksudb.ingest.framework.raw_file_helpers import register_individual_file
from chaksudb.ingest.framework.split_assigner import (
    register_standard_splits,
    bulk_assign_images_to_split,
)
from chaksudb.ingest.framework.task_processors.grading_processor import process_disease_grade
from chaksudb.ingest.framework.task_processors.localization_processor import (
    process_localization_from_xml,
)
from chaksudb.ingest.framework.task_processors.segmentation_processor import (
    process_segmentation_from_binary_mask,
)

logger = logging.getLogger(__name__)

DATASET_NAME = "OIA-DDR"
DATASET_URL = "https://github.com/nkicsl/DDR-dataset"
DATASET_LICENSE = "CC-BY-4.0"

LESION_TYPES = ["EX", "HE", "MA", "SE"]
SPLITS = ["train", "test", "valid"]


def _count_grading_lines(file_path: Path, skip_empty: bool = True, skip_comments: bool = True, comment_char: str = "#") -> int:
    """Count non-empty, non-comment lines in a text file."""
    if not file_path.exists():
        return 0
    count = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if skip_empty and not line:
                continue
            if skip_comments and line.startswith(comment_char):
                continue
            count += 1
    return count


async def ingest_oia_ddr() -> OperationStatistics:
    data_root = get_data_root() / "28_OIA-DDR"
    dataset_id = generate_dataset_uuid(DATASET_NAME)

    logger.info("=" * 80)
    logger.info(f"Starting ingestion: {DATASET_NAME}")
    logger.info(f"Data root: {data_root}")
    logger.info("=" * 80)

    logger.info(f"Registering dataset: {DATASET_NAME}")
    await upsert_dataset(
        Dataset(
            dataset_id=dataset_id,
            dataset_name=DATASET_NAME,
            source_url=DATASET_URL,
            license=DATASET_LICENSE,
            modality_types=["fundus"],
        )
    )

    # Count total items for progress tracking
    logger.info("Counting items across DR grading, localization, and segmentation...")
    grading_count = 0
    for split in SPLITS:
        f = data_root / "DR_grading" / f"{split}.txt"
        n = await asyncio.to_thread(_count_grading_lines, f)
        grading_count += n
        if n:
            logger.info(f"  DR_grading {split}: {n} lines")
    loc_count = 0
    for split in SPLITS:
        dir_xml = data_root / "lesion_detection" / split
        if dir_xml.exists():
            xml_files = await asyncio.to_thread(find_files_by_extension, dir_xml, ".xml", recursive=True)
            loc_count += len(xml_files)
            if xml_files:
                logger.info(f"  lesion_detection {split}: {len(xml_files)} XML files")
    seg_count = 0
    for split in SPLITS:
        img_dir = data_root / "lesion_segmentation" / split / "image"
        if img_dir.exists():
            img_files = await asyncio.to_thread(
                find_images, img_dir, recursive=False
            )
            seg_count += len(img_files)
            if img_files:
                logger.info(f"  lesion_segmentation {split}/image: {len(img_files)} images")
    total_count = grading_count + loc_count + seg_count
    logger.info(f"Total items to process: {total_count} (grading: {grading_count}, localization: {loc_count}, segmentation: {seg_count})")

    tracker = ProgressTracker(total=total_count, description="Ingesting OIA-DDR")

    # ------------------------------------------------------------------
    # GLOBAL REGISTRY – filename = SAME image across all tasks
    # ------------------------------------------------------------------
    existing_image_index: Dict[str, UUID] = {}
    image_to_splits: Dict[UUID, str] = {}  # first-seen split wins

    all_images: List[Image] = []
    all_gradings: List[DiseaseGrading] = []
    all_localizations: List[LocalizationAnnotation] = []
    all_segmentations: List[SegmentationAnnotation] = []

    async def get_or_create_image(filename: str, image_path: Path, split_name: str) -> UUID:
        """dataset + filename = SAME image"""
        key = filename

        if key in existing_image_index:
            img_id = existing_image_index[key]
            image_to_splits.setdefault(img_id, set()).add(split_name)
            return img_id

        img_id = generate_image_uuid(dataset_id, filename)

        image = Image(
            image_id=img_id,
            dataset_id=dataset_id,
            original_image_id=filename,
            **get_image_metadata_dict(image_path),
            modality="fundus",
        )

        all_images.append(image)
        existing_image_index[key] = img_id
        image_to_splits.setdefault(img_id, set()).add(split_name)

        return img_id

    # ================================================================
    # DR GRADING
    # ================================================================
    logger.info("Processing DR grading files...")
    async def process_grading_line(line: str, _: int):
        try:
            parts = line.strip().split()
            if len(parts) != 2:
                tracker.update(success=False)
                tracker.record_error(
                    error_type="grading_parse",
                    error_message="Expected 'filename grade'",
                    item_id=line[:50] if line else None,
                )
                return
            filename, grade_str = parts[0], parts[1]
            split_name = process_grading_line._current_split

            image_dir = data_root / "DR_grading" / split_name
            image_path = image_dir / filename

            if not image_path.exists():
                tracker.update(success=False)
                tracker.record_error(
                    error_type="missing_image",
                    error_message=f"Image not found: {image_path}",
                    item_id=filename,
                    item_path=str(image_path),
                )
                return

            image_id = await get_or_create_image(filename, image_path, split_name)

            try:
                grade_value = int(grade_str)
            except ValueError as e:
                tracker.update(success=False)
                tracker.record_error(
                    error_type="grading_value",
                    error_message=str(e),
                    item_id=filename,
                    item_path=str(image_path),
                )
                return

            raw_data_id, prov = get_current_provenance()

            grading = await process_disease_grade(
                grade_value=grade_value,
                disease_type="DR",
                scale_name="ICDR_0_5",
                image_id=image_id,
                raw_data_id=raw_data_id,
                provenance_chain_id=prov,
            )

            all_gradings.append(grading)
            tracker.update(success=True)
            tracker.record_success("grading")
        except Exception as e:
            logger.exception(f"Grading line failed: {line[:50]}")
            tracker.update(success=False)
            tracker.record_error(
                error_type="grading",
                error_message=str(e),
                item_id=line[:80] if line else None,
            )

    for split in SPLITS:
        file = data_root / "DR_grading" / f"{split}.txt"
        if not file.exists():
            continue

        process_grading_line._current_split = split

        raw_id, chain = await register_individual_file(
            file_path=file,
            dataset_id=dataset_id,
            unified_annotation_type="grading",
            file_type="txt",
        )

        token_raw, token_chain = set_provenance_context(raw_id, chain)

        await process_text_file(file, process_grading_line, tracker)

        reset_provenance_context(token_raw, token_chain)

    # ================================================================
    # LOCALIZATION XML
    # ================================================================
    logger.info("Processing localization (lesion detection) XML files...")
    async def process_xml_file(xml_path: Path, *_):
        import xml.etree.ElementTree as ET

        try:
            split_name = xml_path.parent.name
            root = ET.parse(xml_path).getroot()
            fn_elem = root.find("filename")
            if fn_elem is None or fn_elem.text is None:
                tracker.update(success=False)
                tracker.record_error(
                    error_type="localization_parse",
                    error_message="Missing or empty filename in XML",
                    item_path=str(xml_path),
                )
                return
            filename = fn_elem.text.strip()

            for base in [
                data_root / "DR_grading" / split_name,
                data_root / "lesion_segmentation" / split_name / "image",
            ]:
                candidate = base / filename
                if candidate.exists():
                    image_path = candidate
                    break
            else:
                tracker.update(success=False)
                tracker.record_error(
                    error_type="missing_image",
                    error_message=f"Image not found for XML: {filename}",
                    item_id=filename,
                    item_path=str(xml_path),
                )
                return

            image_id = await get_or_create_image(filename, image_path, split_name)

            raw, prov = get_current_provenance()

            locs = await process_localization_from_xml(
                xml_path=xml_path,
                image_id=image_id,
                raw_data_id=raw,
                provenance_chain_id=prov,
                annotation_method="manual",
            )

            all_localizations.extend(locs)
            tracker.update(success=True)
            tracker.record_success("localization")
        except Exception as e:
            logger.exception(f"Localization failed for {xml_path}: {e}")
            tracker.update(success=False)
            tracker.record_error(
                error_type="localization",
                error_message=str(e),
                item_path=str(xml_path),
            )

    for split in SPLITS:
        dir_xml = data_root / "lesion_detection" / split
        if dir_xml.exists():
            await process_folder_tree(
                root_dir=dir_xml,
                dataset_id=dataset_id,
                unified_annotation_type="localization",
                process_file_fn=process_xml_file,
                file_extensions={".xml"},
                progress_tracker=tracker,
            )

    # ================================================================
    # SEGMENTATION
    # ================================================================
    logger.info("Processing lesion segmentation images...")
    async def process_image_file(img_path: Path, *_):
        try:
            split_name = img_path.parent.parent.name
            filename = img_path.name

            image_id = await get_or_create_image(filename, img_path, split_name)

            label_dir = img_path.parent.parent / "label"

            for lesion in LESION_TYPES:
                mask = label_dir / lesion / f"{img_path.stem}.tif"

                if not mask.exists():
                    continue

                raw_id, chain = await register_individual_file(
                    file_path=mask,
                    dataset_id=dataset_id,
                    unified_annotation_type="segmentation",
                    auto_detect_type=False,
                )

                seg = await process_segmentation_from_binary_mask(
                    mask_path=mask,
                    annotation_type="lesions",
                    image_id=image_id,
                    lesion_subtype=lesion,
                    raw_data_id=raw_id,
                    provenance_chain_id=chain,
                    annotation_method="manual",
                    dataset_name=DATASET_NAME,
                )

                all_segmentations.append(seg)

            tracker.update(success=True)
            tracker.record_success("segmentation")
        except Exception as e:
            logger.exception(f"Segmentation failed for {img_path}: {e}")
            tracker.update(success=False)
            tracker.record_error(
                error_type="segmentation",
                error_message=str(e),
                item_id=img_path.stem,
                item_path=str(img_path),
            )

    for split in SPLITS:
        img_dir = data_root / "lesion_segmentation" / split / "image"
        if img_dir.exists():
            await process_folder_tree(
                root_dir=img_dir,
                dataset_id=dataset_id,
                unified_annotation_type="segmentation",
                process_file_fn=process_image_file,
                file_extensions={".jpg", ".jpeg", ".JPG", ".JPEG"},
                progress_tracker=tracker,
            )

    # ================================================================
    # UPSERT
    # ================================================================
    await bulk_upsert_images(all_images, batch_size=1000)

    await asyncio.gather(
        bulk_upsert_disease_gradings(all_gradings, batch_size=1000),
        bulk_upsert_localization_annotations(all_localizations, batch_size=1000),
    )

    for seg in all_segmentations:
        await upsert_segmentation_annotation(seg)

    # ================================================================
    # SPLIT ASSIGNMENT – MULTIPLE ALLOWED
    # ================================================================
    train_ids = [i for i, s in image_to_splits.items() if "train" in s]
    test_ids  = [i for i, s in image_to_splits.items() if "test" in s]
    val_ids   = [i for i, s in image_to_splits.items() if "valid" in s]

    splits = await register_standard_splits(
        dataset_id=dataset_id,
        split_type="explicit",
        train_count=len(train_ids),
        test_count=len(test_ids),
        val_count=len(val_ids),
    )

    if train_ids:
        await bulk_assign_images_to_split(train_ids, splits["train"])
    if test_ids:
        await bulk_assign_images_to_split(test_ids, splits["test"])
    if val_ids:
        await bulk_assign_images_to_split(val_ids, splits["val"])

    tracker.finish()
    final_stats = tracker.get_statistics()

    logger.info("=" * 80)
    logger.info("Ingestion Summary:")
    logger.info(f"  Total count (expected): {total_count} (grading: {grading_count}, localization: {loc_count}, segmentation: {seg_count})")
    logger.info(f"  Total items: {final_stats.total_items}")
    logger.info(f"  Successful: {final_stats.successful_items}")
    logger.info(f"  Failed: {final_stats.failed_items}")
    logger.info(f"  Skipped: {final_stats.skipped_items}")
    logger.info(f"  Images: {len(all_images)}")
    logger.info(f"  DR gradings: {len(all_gradings)}")
    logger.info(f"  Localizations: {len(all_localizations)}")
    logger.info(f"  Segmentations: {len(all_segmentations)}")
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
    log_file = Path("./logs/ingest_28_oia_ddr.log")
    log_file.touch(exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
        logging.FileHandler(log_file, mode='w'), 
        logging.StreamHandler(sys.stdout),          
        ],
    )
    
    try:
        stats = await ingest_oia_ddr()

        logger.info("=" * 80)
        logger.info("Ingestion complete!")
        logger.info(f"Total: {stats.total_items} | Successful: {stats.successful_items} | Failed: {stats.failed_items} | Errors: {len(stats.errors)}")
        logger.info("=" * 80)

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
    exit_code=asyncio.run(main())
    exit(exit_code)
