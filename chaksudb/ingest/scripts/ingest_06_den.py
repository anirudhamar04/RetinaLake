"""
Ingestion script for DeepEyeNet (DEN) dataset.

Dataset: Large-scale multi-modal retinal imaging dataset with clinical descriptions
Structure: JSON files with keywords and clinical captions
Annotations: Diagnostic keywords and free-form clinical descriptions
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
from uuid import UUID

from chaksudb.common.progress import ProgressTracker, OperationStatistics
from chaksudb.config.config import get_data_root
from chaksudb.db.models import (
    Dataset,
    Image,
    KeywordAnnotation,
    ClinicalDescription,
)
from chaksudb.db.queries import (
    upsert_dataset,
    bulk_upsert_images,
    upsert_keyword_annotation,
    upsert_clinical_description,
)
from chaksudb.ingest.framework import (
    process_json,
    get_image_metadata_dict,
)
from chaksudb.ingest.framework.gen_uuid import (
    generate_dataset_uuid,
    generate_image_uuid,
    generate_description_uuid,
)
from chaksudb.ingest.framework.task_processors.keyword_processor import (
    process_keywords_batch,
)
from chaksudb.ingest.framework.provenance_context import get_current_provenance
from chaksudb.ingest.framework.split_assigner import (
    register_standard_splits,
    bulk_assign_images_to_split,
)

logger = logging.getLogger(__name__)


def detect_modality(keywords: str, description: str) -> str:
    """Detect image modality from keywords and clinical description text.

    Returns 'fa', 'oct', or 'fundus' (default).
    """
    text = f"{keywords} {description}".lower()
    fa_terms = [
        "fluorescein angiogra",
        " fa ",
        "angiogram",
        "angiography",
    ]
    oct_terms = [
        "optical coherence tomography",
        " oct ",
        "oct scan",
        "oct image",
    ]
    for term in fa_terms:
        if term in text:
            return "fa"
    for term in oct_terms:
        if term in text:
            return "oct"
    return "fundus"


# Dataset metadata
DATASET_NAME = "DeepEyeNet"
DATASET_URL = "https://github.com/Jhhuangkay/DeepOpht-Medical-Report-Generation-for-Retinal-Images-via-Deep-Models-and-Visual-Explanation"
DATASET_LICENSE = "Unknown"

# Internal split keys ("train"/"valid"/"test") map to the canonical DB split names
# ("train"/"val"/"test") via register_standard_splits(); no separate mapping needed here.


def count_json_entries(json_path: Path) -> int:
    """Count the number of entries in a DeepEyeNet JSON file."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # DeepEyeNet JSON is a list of single-entry dicts
            if isinstance(data, list):
                return len(data)
            elif isinstance(data, dict):
                return len(data)
            else:
                return 0
    except Exception as e:
        logger.warning(f"Failed to count entries in {json_path}: {e}")
        return 0


async def ingest_deepeyenet() -> OperationStatistics:
    """
    Main ingestion function for DeepEyeNet dataset.
    
    Returns:
        OperationStatistics with success/error counts
    """
    data_root = get_data_root() / "06_DEN"
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
        modality_types=["fundus", "fa", "oct"],
        description="Large-scale multi-modal retinal imaging dataset with expert-written clinical descriptions and diagnostic keywords",
    )
    await upsert_dataset(dataset)
    
    # Step 2: Count total entries across all splits
    logger.info("Counting entries in JSON files...")
    json_files = [
        ("train", data_root / "DeepEyeNet_train.json"),
        ("valid", data_root / "DeepEyeNet_valid.json"),
        ("test", data_root / "DeepEyeNet_test.json"),
    ]
    
    total_entries = sum(
        count_json_entries(json_path) for _, json_path in json_files
    )
    logger.info(f"Total entries to process: {total_entries}")
    
    # Step 3: Setup progress tracker
    tracker = ProgressTracker(
        total=total_entries,
        description=f"Ingesting {DATASET_NAME}"
    )
    
    # Collections for bulk upsert
    all_images: List[Image] = []
    all_keywords: List[KeywordAnnotation] = []
    all_descriptions: List[ClinicalDescription] = []
    image_to_split: Dict[UUID, str] = {}
    
    # Step 4: Process each JSON file
    for split_name, json_path in json_files:
        if not json_path.exists():
            logger.warning(f"JSON file not found: {json_path}")
            continue
        
        logger.info(f"Processing {split_name} split: {json_path}")
        
        async def process_entry(entry: Any, idx: int) -> None:
            """Process a single JSON entry."""
            try:
                # DeepEyeNet JSON structure: list of single-entry dicts
                # Each entry is like: {"eyenet0420/train_set/image.jpg": {"keywords": "...", "clinical-description": "..."}}
                if isinstance(entry, dict):
                    # Get the single key-value pair
                    if len(entry) != 1:
                        tracker.update(success=False)
                        tracker.record_error(
                            error_type="invalid_format",
                            error_message=f"Expected single entry dict, got {len(entry)} entries",
                            item_id=str(idx),
                        )
                        return
                    
                    image_path_str, metadata = next(iter(entry.items()))
                else:
                    # Fallback for tuple format
                    image_path_str, metadata = entry
                
                # Extract image filename
                image_rel_path = Path(image_path_str)
                image_filename = image_rel_path.name
                
                # Generate image UUID
                image_id = generate_image_uuid(dataset_id, image_filename)
                
                # Find actual image file
                # Images are in eyenet0420/{train_set,val_set,test_set}/
                split_folder_map = {
                    "train": "train_set",
                    "valid": "val_set",
                    "test": "test_set",
                }
                split_folder = split_folder_map.get(split_name, "train_set")
                image_path = data_root / "eyenet0420" / split_folder / image_filename
                
                if not image_path.exists():
                    tracker.update(success=False)
                    tracker.record_error(
                        error_type="file_not_found",
                        error_message=f"Image file not found",
                        item_id=image_filename,
                        item_path=str(image_path),
                    )
                    return
                
                # Detect modality from keywords and clinical description
                keywords_str = metadata.get("keywords", "")
                description_text = metadata.get("clinical-description", "")
                modality = detect_modality(keywords_str, description_text)

                # Create image with automatic metadata extraction
                image = Image(
                    image_id=image_id,
                    dataset_id=dataset_id,
                    original_image_id=image_filename,
                    **get_image_metadata_dict(image_path),
                    modality=modality,
                )
                all_images.append(image)
                image_to_split[image_id] = split_name
                
                # Process keywords - provenance automatically from context
                if keywords_str and keywords_str.strip():
                    keyword_annotations = await process_keywords_batch(
                        keywords=keywords_str,
                        keyword_source="diagnostic_keywords",
                        image_id=image_id,
                        dataset_id=dataset_id,
                        delimiter=",",
                        annotation_method="manual",
                    )
                    all_keywords.extend(keyword_annotations)
                
                # Process clinical description - provenance from context
                if description_text and description_text.strip():
                    raw_file_id, chain_id = get_current_provenance()
                    
                    # Calculate word count
                    word_count = len(description_text.split())
                    
                    description = ClinicalDescription(
                        description_id=generate_description_uuid(
                            image_id=image_id,
                            description_type="clinical_caption",
                            raw_data_id=raw_file_id,
                        ),
                        image_id=image_id,
                        description_text=description_text,
                        description_type="clinical_caption",
                        raw_data_id=raw_file_id,
                        word_count=word_count,
                    )
                    all_descriptions.append(description)
                
                tracker.update(success=True)
                tracker.record_success("image")
                
            except Exception as e:
                tracker.update(success=False)
                tracker.record_error(
                    error_type="processing",
                    error_message=str(e),
                    item_id=str(idx),
                )
                logger.error(f"Failed to process entry {idx}: {e}", exc_info=True)
        
        # Process JSON file with automatic provenance tracking
        try:
            stats, raw_file_id, chain_id = await process_json(
                json_path=json_path,
                dataset_id=dataset_id,
                unified_annotation_type="keyword",  # Primary annotation type
                process_entry_fn=process_entry,
                progress_tracker=tracker,
                skip_errors=True,
            )
            logger.info(f"Processed {split_name}: {stats.successful_items} successful, {stats.failed_items} failed")
        except Exception as e:
            logger.error(f"Failed to process {split_name} JSON: {e}", exc_info=True)
    
    # Step 5: Bulk upsert images
    logger.info(f"Upserting {len(all_images)} images...")
    if all_images:
        await bulk_upsert_images(all_images, batch_size=1000)
    
    # Step 6: Upsert keywords (no bulk operation available yet)
    logger.info(f"Upserting {len(all_keywords)} keyword annotations...")
    for keyword_ann in all_keywords:
        try:
            await upsert_keyword_annotation(keyword_ann)
        except Exception as e:
            logger.error(f"Failed to upsert keyword annotation: {e}")
            tracker.record_error(
                error_type="upsert_failed",
                error_message=str(e),
                item_id=str(keyword_ann.annotation_id),
            )
    
    # Step 7: Upsert clinical descriptions (no bulk operation available yet)
    logger.info(f"Upserting {len(all_descriptions)} clinical descriptions...")
    for desc in all_descriptions:
        try:
            await upsert_clinical_description(desc)
        except Exception as e:
            logger.error(f"Failed to upsert clinical description: {e}")
            tracker.record_error(
                error_type="upsert_failed",
                error_message=str(e),
                item_id=str(desc.description_id),
            )
    
    # Step 8: Register splits and assign images
    logger.info("Registering dataset splits...")
    
    # Count images per split
    train_count = sum(1 for s in image_to_split.values() if s == "train")
    val_count = sum(1 for s in image_to_split.values() if s == "valid")
    test_count = sum(1 for s in image_to_split.values() if s == "test")
    
    splits = await register_standard_splits(
        dataset_id=dataset_id,
        split_type="explicit",
        train_count=train_count,
        val_count=val_count,
        test_count=test_count,
    )
    
    # Assign images to splits
    train_images = [img_id for img_id, s in image_to_split.items() if s == "train"]
    val_images = [img_id for img_id, s in image_to_split.items() if s == "valid"]
    test_images = [img_id for img_id, s in image_to_split.items() if s == "test"]
    
    await asyncio.gather(
        bulk_assign_images_to_split(train_images, splits["train"]) if train_images else asyncio.sleep(0),
        bulk_assign_images_to_split(val_images, splits["val"]) if val_images else asyncio.sleep(0),
        bulk_assign_images_to_split(test_images, splits["test"]) if test_images else asyncio.sleep(0),
    )
    
    logger.info(f"Assigned {len(train_images)} images to train split")
    logger.info(f"Assigned {len(val_images)} images to validation split")
    logger.info(f"Assigned {len(test_images)} images to test split")
    
    tracker.finish()
    return tracker.get_statistics()


async def main():
    """Entry point for script execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    try:
        stats = await ingest_deepeyenet()
        
        logger.info("=" * 80)
        logger.info("Ingestion Summary:")
        logger.info(f"  Total items: {stats.total_items}")
        logger.info(f"  Successful: {stats.successful_items}")
        logger.info(f"  Failed: {stats.failed_items}")
        logger.info(f"  Skipped: {stats.skipped_items}")
        if stats.errors:
            logger.warning(f"  Total errors: {len(stats.errors)}")
            for error_type, count in stats.error_counts.items():
                logger.warning(f"    {error_type}: {count}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.exception(f"Ingestion failed with error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
