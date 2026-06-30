"""
Ingestion script for ScarDat dataset.

Dataset: ScarDat - Retinal Scar Detection Dataset
Structure: Folder-based binary classification with CSV labels
  - train/, test/, val/ splits
  - Each split has positive/ and negative/ folders
  - CSV files: kaggle_label.csv, non-kaggle_label.csv (format: image_id,label)
Annotations: Binary classification (scar presence)
Tasks: Binary classification (scar presence)
"""

import asyncio
import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set
from uuid import UUID

from chaksudb.common.progress import ProgressTracker, OperationStatistics
from chaksudb.config.config import get_data_root
from chaksudb.db.models import Dataset, Image, ClassificationAnnotation
from chaksudb.db.queries import (
    bulk_upsert_images,
    bulk_upsert_classification_annotations,
    upsert_dataset,
)
from chaksudb.ingest.framework import (
    get_image_metadata_dict,
    process_folder_tree,
)
from chaksudb.ingest.framework.gen_uuid import (
    generate_dataset_uuid,
    generate_image_uuid,
)
from chaksudb.ingest.framework.provenance_context import get_current_provenance
from chaksudb.ingest.framework.split_assigner import (
    register_standard_splits,
    bulk_assign_images_to_split,
)
from chaksudb.ingest.framework.task_processors.classification_processor import (
    process_classification,
)

logger = logging.getLogger(__name__)

# Dataset metadata
DATASET_NAME = "ScarDat"
DATASET_URL = "https://github.com/li-xirong/fundus10k"
DATASET_LICENSE = "CC0: Public Domain"

# Standard splits
SPLITS = ["train", "test", "val"]


def load_csv_labels(csv_path: Path) -> Dict[str, int]:
    """
    Load labels from CSV file.
    
    CSV format: image_id,label
    where label is 0 (negative) or 1 (positive)
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        Dictionary mapping image_id (without extension) to label (0 or 1)
        Also includes variations (with/without _left/_right suffix) for matching
    """
    labels = {}
    if not csv_path.exists():
        return labels
    
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    image_id = row[0].strip()
                    label = int(row[1].strip())
                    # Remove file extension if present
                    image_id_stem = Path(image_id).stem
                    labels[image_id_stem] = label
                    
                    # Also add variations for matching (remove _left/_right suffix)
                    if "_left" in image_id_stem or "_right" in image_id_stem:
                        base_id = image_id_stem.replace("_left", "").replace("_right", "")
                        if base_id not in labels:
                            labels[base_id] = label
    except Exception as e:
        logger.warning(f"Failed to load CSV labels from {csv_path}: {e}")
    
    return labels


def validate_csv_vs_folder_structure(
    split_dir: Path,
    csv_labels: Dict[str, int],
    split_name: str,
) -> Dict[str, List[str]]:
    """
    Validate CSV labels against folder structure and report mismatches.
    
    Args:
        split_dir: Directory for the split (e.g., train/, test/, val/)
        csv_labels: Dictionary of CSV labels (image_id -> label)
        split_name: Name of the split
        
    Returns:
        Dictionary with validation results:
        - "matches": List of image IDs that match
        - "mismatches": List of tuples (image_id, folder_label, csv_label)
        - "csv_only": List of image IDs in CSV but not in folders
        - "folder_only": List of image IDs in folders but not in CSV
    """
    results = {
        "matches": [],
        "mismatches": [],
        "csv_only": [],
        "folder_only": [],
    }
    
    # Get images from folders
    positive_dir = split_dir / "positive"
    negative_dir = split_dir / "negative"
    
    folder_images: Dict[str, int] = {}  # image_stem -> label
    
    if positive_dir.exists():
        for img_file in positive_dir.glob("*.jpg"):
            folder_images[img_file.stem] = 1
    
    if negative_dir.exists():
        for img_file in negative_dir.glob("*.jpg"):
            folder_images[img_file.stem] = 0
    
    # Compare CSV vs folders
    csv_image_ids = set(csv_labels.keys())
    folder_image_ids = set(folder_images.keys())
    
    # Find matches and mismatches
    for image_id in csv_image_ids & folder_image_ids:
        csv_label = csv_labels[image_id]
        folder_label = folder_images[image_id]
        if csv_label == folder_label:
            results["matches"].append(image_id)
        else:
            results["mismatches"].append((image_id, folder_label, csv_label))
    
    # Images only in CSV
    results["csv_only"] = list(csv_image_ids - folder_image_ids)
    
    # Images only in folders
    results["folder_only"] = list(folder_image_ids - csv_image_ids)
    
    # Log validation summary
    logger.info(f"  Validation for {split_name}:")
    logger.info(f"    Matches: {len(results['matches'])}")
    logger.info(f"    Mismatches: {len(results['mismatches'])}")
    if results["mismatches"]:
        logger.warning(f"    Mismatch examples (first 5):")
        for img_id, folder_lbl, csv_lbl in results["mismatches"][:5]:
            logger.warning(
                f"      {img_id}: folder={folder_lbl} (folder: {'positive' if folder_lbl == 1 else 'negative'}), "
                f"CSV={csv_lbl}"
            )
    logger.info(f"    CSV only: {len(results['csv_only'])}")
    logger.info(f"    Folder only: {len(results['folder_only'])}")
    
    return results


async def ingest_scardat() -> OperationStatistics:
    """
    Main ingestion function for ScarDat dataset.
    
    The dataset contains fundus images organized into train/test/val splits.
    Each split has:
    - positive/ folder: Images with retinal scars
    - negative/ folder: Images without retinal scars
    - kaggle_label.csv: CSV file with labels (optional validation)
    - non-kaggle_label.csv: CSV file with labels (optional validation)
    
    Returns:
        OperationStatistics with success/error counts
    """
    data_root = get_data_root() / "35_ScarDat"
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
    logger.info("Counting images...")
    total_images = 0
    split_counts = {}
    
    for split_name in SPLITS:
        split_dir = data_root / split_name
        if not split_dir.exists():
            logger.warning(f"Split directory {split_name} does not exist, skipping")
            split_counts[split_name] = 0
            continue
        
        positive_dir = split_dir / "positive"
        negative_dir = split_dir / "negative"
        test80_dir = split_dir / "test80" if split_name == "test" else None
        
        positive_count = len(list(positive_dir.glob("*.jpg"))) if positive_dir.exists() else 0
        negative_count = len(list(negative_dir.glob("*.jpg"))) if negative_dir.exists() else 0
        test80_count = len(list(test80_dir.glob("*.jpg"))) if test80_dir and test80_dir.exists() else 0
        
        split_total = positive_count + negative_count + test80_count
        split_counts[split_name] = split_total
        total_images += split_total
        
        logger.info(
            f"  {split_name}: {positive_count} positive, {negative_count} negative"
            + (f", {test80_count} test80 (no labels)" if test80_count > 0 else "")
            + f" (total: {split_total})"
        )
    
    logger.info(f"Total images: {total_images}")
    
    # Step 3: Load CSV labels and validate against folder structure
    logger.info("Loading CSV labels and validating against folder structure...")
    csv_labels: Dict[str, Dict[str, int]] = {}  # split -> image_id -> label
    validation_results: Dict[str, Dict[str, List]] = {}  # split -> validation results
    
    for split_name in SPLITS:
        split_dir = data_root / split_name
        if not split_dir.exists():
            continue
        
        split_labels: Dict[str, int] = {}
        
        # Load kaggle_label.csv
        kaggle_csv = split_dir / "kaggle_label.csv"
        kaggle_labels = load_csv_labels(kaggle_csv)
        split_labels.update(kaggle_labels)
        logger.info(f"  {split_name}: Loaded {len(kaggle_labels)} labels from kaggle_label.csv")
        
        # Load non-kaggle_label.csv
        non_kaggle_csv = split_dir / "non-kaggle_label.csv"
        non_kaggle_labels = load_csv_labels(non_kaggle_csv)
        split_labels.update(non_kaggle_labels)
        logger.info(
            f"  {split_name}: Loaded {len(non_kaggle_labels)} labels from non-kaggle_label.csv "
            f"(total: {len(split_labels)})"
        )
        
        csv_labels[split_name] = split_labels
        
        # Validate CSV vs folder structure
        if split_labels:
            validation_results[split_name] = validate_csv_vs_folder_structure(
                split_dir, split_labels, split_name
            )
    
    # Step 4: Setup progress tracker
    tracker = ProgressTracker(
        total=total_images,
        description=f"Ingesting {DATASET_NAME}"
    )
    
    # Step 5: Collect items for bulk upsert
    all_images: List[Image] = []
    all_classifications: List[ClassificationAnnotation] = []
    image_to_split: Dict[UUID, str] = {}
    
    # Step 6: Process images from each split
    for split_name in SPLITS:
        split_dir = data_root / split_name
        if not split_dir.exists():
            continue
        
        logger.info(f"Processing {split_name} split...")
        
        # Get CSV labels for this split (if available)
        split_csv_labels = csv_labels.get(split_name, {})
        
        async def process_image_file(
            file_path: Path,
            rel_path: Path,
            depth: int,
        ) -> None:
            """
            Process a single image file.
            
            Args:
                file_path: Absolute path to image file
                rel_path: Path relative to split directory
                depth: Directory depth
            """
            try:
                # Extract label from folder name (positive or negative)
                # Path structure: {split}/{positive|negative}/filename.jpg
                # Since we're processing from positive_dir/negative_dir, get parent from absolute path
                parent_folder = file_path.parent.name
                
                if parent_folder == "positive":
                    folder_label = 1
                elif parent_folder == "negative":
                    folder_label = 0
                else:
                    logger.warning(
                        f"Unexpected folder name '{parent_folder}' for {file_path}, "
                        f"skipping"
                    )
                    tracker.update(success=False)
                    tracker.record_error(
                        error_type="invalid_structure",
                        error_message=f"Unexpected folder: {parent_folder}",
                        item_id=file_path.stem,
                        item_path=str(file_path),
                    )
                    return
                
                # Get provenance from context (set by process_folder_tree)
                raw_data_id, provenance_chain_id = get_current_provenance()
                
                # Generate image ID
                image_stem = file_path.stem
                image_id = generate_image_uuid(dataset_id, f"{split_name}_{image_stem}")
                
                # Determine label: prefer CSV if available, otherwise use folder structure
                # Try exact match first, then try without _left/_right suffix
                csv_label = None
                if split_csv_labels:
                    csv_label = split_csv_labels.get(image_stem)
                    if csv_label is None:
                        # Try variations (remove _left/_right if present)
                        base_stem = image_stem.replace("_left", "").replace("_right", "")
                        if base_stem != image_stem:
                            csv_label = split_csv_labels.get(base_stem)
                
                # Use CSV label if available, otherwise use folder label
                if csv_label is not None:
                    final_label = csv_label
                    label_source = "CSV"
                    if csv_label != folder_label:
                        logger.warning(
                            f"Label mismatch for {image_stem} in {split_name}: "
                            f"folder={parent_folder} (label={folder_label}), "
                            f"CSV={csv_label} - using CSV label"
                        )
                else:
                    final_label = folder_label
                    label_source = "folder"
                
                is_positive = (final_label == 1)
                
                # Create image with automatic metadata extraction
                image = Image(
                    image_id=image_id,
                    dataset_id=dataset_id,
                    original_image_id=image_stem,
                    **get_image_metadata_dict(file_path),
                    modality="fundus",
                )
                all_images.append(image)
                image_to_split[image_id] = split_name
                
                # Process scar classification
                classifications = await process_classification(
                    class_value=is_positive,
                    task_type="binary",
                    class_name="retinal_scar",
                    image_id=image_id,
                    class_labels={0: "negative", 1: "positive"},
                    raw_data_id=raw_data_id,
                    provenance_chain_id=provenance_chain_id,
                    annotation_method="manual",
                )
                all_classifications.extend(classifications)
                
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
        
        # Process images from positive and negative folders
        positive_dir = split_dir / "positive"
        negative_dir = split_dir / "negative"
        
        # Process positive images
        if positive_dir.exists():
            logger.info(f"  Processing positive images from {split_name}...")
            await process_folder_tree(
                root_dir=positive_dir,
                dataset_id=dataset_id,
                unified_annotation_type="classification",
                process_file_fn=process_image_file,
                file_extensions={".jpg", ".JPG", ".jpeg", ".JPEG"},
                recursive=False,  # Only process files directly in positive/negative folders
                progress_tracker=tracker,
                skip_errors=True,
            )
        
        # Process negative images
        if negative_dir.exists():
            logger.info(f"  Processing negative images from {split_name}...")
            await process_folder_tree(
                root_dir=negative_dir,
                dataset_id=dataset_id,
                unified_annotation_type="classification",
                process_file_fn=process_image_file,
                file_extensions={".jpg", ".JPG", ".jpeg", ".JPEG"},
                recursive=False,  # Only process files directly in positive/negative folders
                progress_tracker=tracker,
                skip_errors=True,
            )
        
        # Process test80 images (test split only, no classifications)
        # These images have no annotations, so we process them directly without using
        # process_folder_tree() which is designed for annotation files
        if split_name == "test":
            test80_dir = split_dir / "test80"
            if test80_dir.exists():
                logger.info(f"  Processing test80 images (no classifications)...")
                
                # Get all image files
                test80_images = list(test80_dir.glob("*.jpg")) + list(test80_dir.glob("*.JPG")) + \
                                list(test80_dir.glob("*.jpeg")) + list(test80_dir.glob("*.JPEG"))
                
                for image_path in test80_images:
                    try:
                        # Generate image ID
                        image_stem = image_path.stem
                        image_id = generate_image_uuid(dataset_id, f"{split_name}_test80_{image_stem}")
                        
                        # Create image with automatic metadata extraction (no classification)
                        # No provenance tracking needed since these are unannotated images
                        image = Image(
                            image_id=image_id,
                            dataset_id=dataset_id,
                            original_image_id=f"test80_{image_stem}",
                            **get_image_metadata_dict(image_path),
                            modality="fundus",
                        )
                        all_images.append(image)
                        image_to_split[image_id] = split_name
                        
                        tracker.update(success=True)
                        tracker.record_success("image")
                        
                    except Exception as e:
                        logger.error(f"Failed to process test80 image {image_path}: {e}", exc_info=True)
                        tracker.update(success=False)
                        tracker.record_error(
                            error_type="processing",
                            error_message=str(e),
                            item_id=image_path.stem,
                            item_path=str(image_path),
                        )
    
    # Step 7: Bulk upsert images and classifications
    logger.info(f"Upserting {len(all_images)} images...")
    if all_images:
        await bulk_upsert_images(all_images, batch_size=1000)
        logger.info(f"Successfully upserted {len(all_images)} images")
    
    logger.info(f"Upserting {len(all_classifications)} classifications...")
    if all_classifications:
        await bulk_upsert_classification_annotations(
            all_classifications, batch_size=1000
        )
        logger.info(
            f"Successfully upserted {len(all_classifications)} classifications"
        )
    
    # Step 8: Register splits and assign images
    logger.info("Registering dataset splits...")
    
    # Count images per split
    train_image_ids = [
        img_id for img_id, split in image_to_split.items() if split == "train"
    ]
    test_image_ids = [
        img_id for img_id, split in image_to_split.items() if split == "test"
    ]
    val_image_ids = [
        img_id for img_id, split in image_to_split.items() if split == "val"
    ]
    
    splits = await register_standard_splits(
        dataset_id=dataset_id,
        split_type="explicit",
        train_count=len(train_image_ids),
        test_count=len(test_image_ids),
        val_count=len(val_image_ids),
    )
    
    # Assign images to splits
    if train_image_ids:
        await bulk_assign_images_to_split(train_image_ids, splits["train"])
        logger.info(f"Assigned {len(train_image_ids)} images to train split")
    
    if test_image_ids:
        await bulk_assign_images_to_split(test_image_ids, splits["test"])
        logger.info(f"Assigned {len(test_image_ids)} images to test split")
    
    if val_image_ids:
        await bulk_assign_images_to_split(val_image_ids, splits["val"])
        logger.info(f"Assigned {len(val_image_ids)} images to val split")
    
    tracker.finish()
    final_stats = tracker.get_statistics()
    
    # Final summary
    logger.info("=" * 80)
    logger.info("Ingestion Summary:")
    logger.info(f"  Total items: {final_stats.total_items}")
    logger.info(f"  Successful: {final_stats.successful_items}")
    logger.info(f"  Failed: {final_stats.failed_items}")
    logger.info(f"  Skipped: {final_stats.skipped_items}")
    logger.info(f"  Images: {len(all_images)}")
    logger.info(f"  Classifications: {len(all_classifications)}")
    logger.info(f"  Train images: {len(train_image_ids)}")
    logger.info(f"  Test images: {len(test_image_ids)}")
    logger.info(f"  Val images: {len(val_image_ids)}")
    if final_stats.errors:
        logger.warning(f"  Total errors: {len(final_stats.errors)}")
        for error_type, count in final_stats.error_counts.items():
            logger.warning(f"    {error_type}: {count}")
    logger.info("=" * 80)
    
    return final_stats


async def main():
    """Entry point for script execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    try:
        stats = await ingest_scardat()
        
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
