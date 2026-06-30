"""
Ingestion script for DiaRetDB1 dataset.

Dataset: DiaRetDB1 - Diabetic Retinopathy Database
Structure: Split files (train/test) with image paths and XML annotation paths
Annotations: Multi-expert lesion localization (Hard exudates, Soft exudates, Haemorrhages, Red small dots, IRMA, Disc)
Tasks: Localization (lesion detection with circle regions and polygons)
Experts: DiaRetDB01, DiaRetDB02 (experts 01 and 02 only)
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from uuid import UUID

from chaksudb.common.progress import ProgressTracker, OperationStatistics
from chaksudb.config.config import get_data_root
from chaksudb.db.models import (
    Dataset,
    Expert,
    ExpertAnnotation,
    Image,
    LocalizationAnnotation,
)
from chaksudb.db.queries import (
    bulk_upsert_images,
    bulk_upsert_localization_annotations,
    upsert_dataset,
    upsert_expert,
    upsert_expert_annotation,
)
from chaksudb.ingest.framework import get_image_metadata_dict
from chaksudb.ingest.framework.gen_uuid import (
    generate_dataset_uuid,
    generate_expert_annotation_uuid,
    generate_expert_uuid,
    generate_image_uuid,
)
from chaksudb.ingest.framework.provenance_context import (
    set_provenance_context,
    reset_provenance_context,
)
from chaksudb.ingest.framework.raw_file_helpers import register_individual_file
from chaksudb.ingest.framework.split_assigner import (
    bulk_assign_images_to_split,
    register_standard_splits,
)
from chaksudb.ingest.framework.task_processors.localization_processor import (
    process_localization_from_xml,
)

logger = logging.getLogger(__name__)

# Dataset metadata
DATASET_NAME = "DiaRetDB1"
DATASET_URL = "https://www.kaggle.com/datasets/nguyenhung1903/diaretdb1-v21"
DATASET_LICENSE = "Restricted academic license"

# Expert information (only experts 01 and 02 as specified)
EXPERTS = {
    "01": {
        "name": "DiaRetDB01",
        "expertise": "diabetic_retinopathy",
    },
    "02": {
        "name": "DiaRetDB02",
        "expertise": "diabetic_retinopathy",
    },
}


async def register_experts(dataset_id: UUID) -> Dict[str, UUID]:
    """Register the two lesion annotation experts."""
    expert_ids = {}
    
    for expert_key, expert_info in EXPERTS.items():
        expert_id = generate_expert_uuid(
            dataset_id=dataset_id,
            model_id=None,
            expert_name=expert_info["name"],
        )
        
        expert = Expert(
            expert_id=expert_id,
            expert_name=expert_info["name"],
            dataset_id=dataset_id,
            model_id=None,
            expertise_area=expert_info["expertise"],
        )
        
        await upsert_expert(expert)
        expert_ids[expert_key] = expert_id
        logger.info(f"Registered expert: {expert_info['name']} ({expert_key})")
    
    return expert_ids


def parse_split_file(split_file_path: Path) -> List[Tuple[str, List[str]]]:
    """
    Parse split file to extract image paths and XML annotation paths.
    
    Format: Each line contains:
    images/diaretdb1_image010.png groundtruth/diaretdb1_image010_01_plain.xml groundtruth/diaretdb1_image010_02_plain.xml ...
    
    Returns:
        List of tuples: (image_path, [xml_path_01, xml_path_02, ...])
    """
    entries = []
    
    with open(split_file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) < 2:
                logger.warning(f"Skipping malformed line in {split_file_path}: {line}")
                continue
            
            image_path = parts[0]
            xml_paths = parts[1:]
            
            entries.append((image_path, xml_paths))
    
    return entries


async def ingest_diaretdb1() -> OperationStatistics:
    """
    Main ingestion function for DiaRetDB1 dataset.
    
    Returns:
        OperationStatistics with success/error counts
    """
    data_root = get_data_root() / "17_DiaRetDB1"
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
    
    # Step 2: Register experts
    logger.info("Registering experts...")
    expert_ids = await register_experts(dataset_id)
    
    # Step 3: Parse split files
    logger.info("Parsing split files...")
    train_file = data_root / "ddb1_v02_01_train_plain.txt"
    test_file = data_root / "ddb1_v02_01_test_plain.txt"
    
    train_entries = await asyncio.to_thread(parse_split_file, train_file)
    test_entries = await asyncio.to_thread(parse_split_file, test_file)
    
    total_entries = len(train_entries) + len(test_entries)
    logger.info(f"Found {len(train_entries)} training images and {len(test_entries)} test images")
    
    # Step 4: Setup progress tracker
    tracker = ProgressTracker(
        total=total_entries,
        description=f"Ingesting {DATASET_NAME}"
    )
    
    # Collect items for bulk upsert
    all_images: List[Image] = []
    all_localizations: List[LocalizationAnnotation] = []
    image_to_split: Dict[UUID, str] = {}
    
    # Step 5: Process images and annotations
    async def process_entry(entry: Tuple[str, List[str]], split_name: str):
        """Process a single entry from split file."""
        image_path_str, xml_paths = entry
        
        try:
            # Resolve image path (may be in images/ or documents/ directory)
            image_path = data_root / image_path_str
            if not await asyncio.to_thread(image_path.exists):
                # Try documents directory
                image_path = data_root / "documents" / Path(image_path_str).name
                if not await asyncio.to_thread(image_path.exists):
                    tracker.record_error(
                        error_type="image_not_found",
                        error_message=f"Image not found: {image_path_str}",
                        item_id=image_path_str,
                    )
                    tracker.update(success=False)
                    return
            
            # Extract image identifier from filename
            image_identifier = Path(image_path_str).stem
            image_id = generate_image_uuid(dataset_id, image_identifier)
            
            # Create image with automatic metadata extraction
            image = Image(
                image_id=image_id,
                dataset_id=dataset_id,
                original_image_id=image_identifier,
                **get_image_metadata_dict(image_path),
                modality="fundus",
            )
            all_images.append(image)
            image_to_split[image_id] = split_name
            
            # Process XML annotations from experts 01 and 02 only
            for xml_path_str in xml_paths:
                # Extract expert number from filename
                # Format: diaretdb1_image010_01_plain.xml or diaretdb1_image010_01.xml
                xml_filename = Path(xml_path_str).name
                if "_01_plain.xml" in xml_filename or xml_filename.endswith("_01.xml"):
                    expert_key = "01"
                elif "_02_plain.xml" in xml_filename or xml_filename.endswith("_02.xml"):
                    expert_key = "02"
                else:
                    # Skip experts 03 and 04 (only process 01 and 02)
                    continue
                
                expert_id = expert_ids[expert_key]
                
                # Resolve XML path
                xml_path = data_root / xml_path_str
                if not await asyncio.to_thread(xml_path.exists):
                    tracker.record_error(
                        error_type="xml_not_found",
                        error_message=f"XML annotation not found: {xml_path_str}",
                        item_id=xml_path_str,
                    )
                    continue
                
                # Register XML file for provenance
                raw_file_id, chain_id = await register_individual_file(
                    file_path=xml_path,
                    dataset_id=dataset_id,
                    unified_annotation_type="localization",
                    file_type="xml",
                    auto_detect_type=False,
                )
                
                # Set provenance context for this XML file
                token_raw, token_chain = set_provenance_context(raw_file_id, chain_id)
                
                try:
                    # Generate expert annotation ID
                    expert_annotation_id = generate_expert_annotation_uuid(
                        expert_id=expert_id,
                        annotation_task="localization",
                        raw_data_id=raw_file_id,
                        annotation_value_hash=None,
                    )
                    
                    # Create expert annotation record
                    expert_annotation = ExpertAnnotation(
                        expert_annotation_id=expert_annotation_id,
                        expert_id=expert_id,
                        annotation_task="localization",
                        raw_data_id=raw_file_id,
                        annotation_value=None,
                        confidence_level=None,
                        annotation_timestamp=None,
                    )
                    await upsert_expert_annotation(expert_annotation)
                    
                    # Process localization annotations from XML
                    # The XML parser automatically detects ImageRet circle format and extracts coordinates
                    localizations = await process_localization_from_xml(
                        xml_path=xml_path,
                        image_id=image_id,
                        raw_data_id=raw_file_id,
                        expert_annotation_id=expert_annotation_id,
                        annotation_method="manual",
                        provenance_chain_id=chain_id,
                    )
                    
                    all_localizations.extend(localizations)
                finally:
                    # Always reset provenance context
                    reset_provenance_context(token_raw, token_chain)
            
            tracker.update(success=True)
            
        except Exception as e:
            tracker.update(success=False)
            tracker.record_error(
                error_type="processing",
                error_message=str(e),
                item_id=image_path_str if isinstance(image_path_str, str) else "unknown",
            )
            logger.error(f"Failed to process entry: {e}")
    
    # Process train and test entries sequentially — this dataset is small (89 images)
    # and the previous concurrent approach caused connection pool exhaustion that
    # silently dropped all 61 test images (asyncio.gather return_exceptions=True
    # swallowed the failures without logging them).
    logger.info("Processing annotations...")
    for entry in train_entries:
        await process_entry(entry, "train")
    for entry in test_entries:
        await process_entry(entry, "test")
    
    # Step 6: Bulk upsert - images first, then localizations
    logger.info(f"Upserting {len(all_images)} images...")
    await bulk_upsert_images(all_images, batch_size=1000)
    
    logger.info(f"Upserting {len(all_localizations)} localization annotations...")
    await bulk_upsert_localization_annotations(all_localizations, batch_size=1000)
    
    # Step 7: Register splits and assign images
    logger.info("Registering dataset splits...")
    train_image_ids = [img_id for img_id, split in image_to_split.items() if split == "train"]
    test_image_ids = [img_id for img_id, split in image_to_split.items() if split == "test"]
    
    splits = await register_standard_splits(
        dataset_id=dataset_id,
        split_type="explicit",
        train_count=len(train_image_ids),
        test_count=len(test_image_ids),
    )
    
    # Assign images to splits
    await asyncio.gather(
        bulk_assign_images_to_split(train_image_ids, splits["train"]),
        bulk_assign_images_to_split(test_image_ids, splits["test"]),
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
    logger.info(f"  Localization annotations: {len(all_localizations)}")
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
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    try:
        stats = await ingest_diaretdb1()
        
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
