"""
Ingestion script for ARIA (Automated Retinal Image Analysis) dataset.

Dataset: ARIA - Blood vessel and optic disc/fovea segmentation
Structure: 3 subsets (a, c, d) with paired images and segmentation masks
Annotations:
  - Blood vessel segmentation (2 experts: BDP and BSS)
  - Optic disc and fovea segmentation (subsets c and d only)

Key Features:
  - 3 subsets representing different populations (healthy, diabetic, age-related macular degeneration)
  - Multi-expert vessel annotations (BDP and BSS)
  - Anatomical structure segmentation (optic disc and fovea combined masks)
  - TIFF format images and masks
"""

import asyncio
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from uuid import UUID

from chaksudb.common.progress import ProgressTracker, OperationStatistics
from chaksudb.config.config import get_data_root
from chaksudb.db.models import (
    Dataset,
    Image,
    SegmentationAnnotation,
    Expert,
    ExpertAnnotation,
)
from chaksudb.db.queries import (
    upsert_dataset,
    bulk_upsert_images,
    upsert_segmentation_annotation,
    upsert_expert,
    upsert_expert_annotation,
)
from chaksudb.ingest.framework import (
    get_image_metadata_dict,
    find_images,
)
from chaksudb.ingest.framework.gen_uuid import (
    generate_dataset_uuid,
    generate_image_uuid,
    generate_expert_uuid,
    generate_expert_annotation_uuid,
)
from chaksudb.ingest.framework.task_processors.segmentation_processor import (
    process_segmentation_from_binary_mask,
    process_segmentation_from_soft_map
)
from chaksudb.ingest.framework.raw_file_helpers import (
    register_individual_file,
)
from chaksudb.ingest.framework.split_assigner import auto_stratified_splits

logger = logging.getLogger(__name__)

# Dataset metadata
DATASET_NAME = "ARIA"
DATASET_URL = "https://www.damianjjfarnell.com/pages/ARIA"
DATASET_LICENSE = "Research/Academic Use"

# Expert metadata for vessel segmentation
VESSEL_EXPERTS = {
    "BDP": {
        "name": "Expert BDP",
        "affiliation": "ARIA Project",
        "expertise": "Blood Vessel Segmentation",
    },
    "BSS": {
        "name": "Expert BSS",
        "affiliation": "ARIA Project",
        "expertise": "Blood Vessel Segmentation",
    },
}

# ARIA subsets
# Note: Subset descriptions are not documented in the dataset
SUBSETS = ["a", "c", "d"]


def extract_image_identifier(filepath: Path) -> str:
    """
    Extract the base identifier from an ARIA file.
    
    Examples:
        aria_a_12_15.tif -> aria_a_12_15
        aria_a_12_15_BDP.tif -> aria_a_12_15
        aria_c_23_4_dfs.tif -> aria_c_23_4
        (0001)aria_d_26.tif -> aria_d_26
    
    Args:
        filepath: Path to ARIA image or mask file
    
    Returns:
        Base image identifier (without prefix or suffix)
    """
    filename = filepath.stem
    
    # Remove prefix like "(0001)" if present
    filename = re.sub(r'^\(\d+\)', '', filename)
    
    # Remove expert suffixes (_BDP, _BSS)
    filename = re.sub(r'_(BDP|BSS)$', '', filename)
    
    # Remove disc/fovea suffixes (_dfs, _dfd)
    filename = re.sub(r'_(dfs|dfd)$', '', filename)
    
    return filename


async def register_experts(dataset_id: UUID) -> Dict[str, UUID]:
    """Register vessel segmentation experts."""
    expert_ids = {}
    
    for expert_key, expert_info in VESSEL_EXPERTS.items():
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


async def process_images(
    data_root: Path,
    dataset_id: UUID,
    tracker: ProgressTracker,
) -> Dict[str, UUID]:
    """
    Discover and process all images from ARIA subsets.
    
    Returns:
        Dictionary mapping image identifier -> image_id
    """
    all_images: List[Image] = []
    image_id_map: Dict[str, UUID] = {}
    
    # Process each subset
    for subset_key in SUBSETS:
        markups_dir = data_root / f"aria_{subset_key}_markups"
        
        if not markups_dir.exists():
            logger.warning(f"Markups directory not found: {markups_dir}")
            continue
        
        # Find all images in this subset
        image_paths = await asyncio.to_thread(find_images, markups_dir)
        logger.info(f"Found {len(image_paths)} images in subset {subset_key}")
        
        for image_path in image_paths:
            try:
                # Extract identifier
                image_identifier = extract_image_identifier(image_path)
                
                # Generate image UUID
                image_id = generate_image_uuid(dataset_id, image_identifier)
                
                # Create image model
                image = Image(
                    image_id=image_id,
                    dataset_id=dataset_id,
                    original_image_id=image_identifier,
                    **get_image_metadata_dict(image_path),
                    modality="fundus",
                    acquisition_date=None,
                    image_quality=None,
                    # Store subset as metadata in comorbidities
                    comorbidities={"subset": subset_key},
                )
                
                all_images.append(image)
                image_id_map[image_identifier] = image_id
                
                tracker.update(success=True)
                tracker.record_success("image")
                
            except Exception as e:
                tracker.update(success=False)
                tracker.record_error(
                    error_type="image_processing",
                    error_message=str(e),
                    item_id=image_path.stem,
                    item_path=str(image_path),
                )
                logger.error(f"Failed to process image {image_path}: {e}")
    
    # Bulk upsert all images
    if all_images:
        logger.info(f"Upserting {len(all_images)} images...")
        try:
            await bulk_upsert_images(all_images, batch_size=500)
            logger.info(f"Successfully upserted {len(all_images)} images")
        except Exception as e:
            logger.error(f"Failed to bulk upsert images: {e}")
            raise
    
    return image_id_map


async def process_vessel_segmentation(
    data_root: Path,
    dataset_id: UUID,
    image_id_map: Dict[str, UUID],
    expert_ids: Dict[str, UUID],
    tracker: ProgressTracker,
) -> List[SegmentationAnnotation]:
    """
    Process vessel segmentation masks from both experts.
    
    Returns:
        List of SegmentationAnnotation models
    """
    all_segmentations: List[SegmentationAnnotation] = []
    
    # Process each subset
    for subset_key in SUBSETS:
        vessel_dir = data_root / f"aria_{subset_key}_markup_vessel"
        
        if not vessel_dir.exists():
            logger.warning(f"Vessel directory not found: {vessel_dir}")
            continue
        
        # Find all vessel masks
        vessel_masks = list(vessel_dir.glob("*.tif")) + list(vessel_dir.glob("*.tiff"))
        logger.info(f"Found {len(vessel_masks)} vessel masks in subset {subset_key}")
        
        for mask_path in vessel_masks:
            try:
                # Extract image identifier and expert
                image_identifier = extract_image_identifier(mask_path)
                
                # Determine expert from filename suffix
                expert_key = None
                if mask_path.stem.endswith("_BDP"):
                    expert_key = "BDP"
                elif mask_path.stem.endswith("_BSS"):
                    expert_key = "BSS"
                else:
                    logger.warning(f"Cannot determine expert from filename: {mask_path.name}")
                    continue
                
                # Check if image exists
                if image_identifier not in image_id_map:
                    tracker.record_error(
                        error_type="image_not_found",
                        error_message=f"Image not found for mask: {image_identifier}",
                        item_id=image_identifier,
                        item_path=str(mask_path),
                    )
                    continue
                
                image_id = image_id_map[image_identifier]
                expert_id = expert_ids[expert_key]
                
                # Register mask file for provenance
                raw_file_id, chain_id = await register_individual_file(
                    file_path=mask_path,
                    dataset_id=dataset_id,
                    unified_annotation_type="segmentation",
                    file_type=None,
                    auto_detect_type=False,
                )
                
                # Generate expert annotation ID
                expert_annotation_id = generate_expert_annotation_uuid(
                    expert_id=expert_id,
                    annotation_task="segmentation",
                    raw_data_id=raw_file_id,
                    annotation_value_hash=None,
                )
                
                # Create expert annotation record
                expert_annotation = ExpertAnnotation(
                    expert_annotation_id=expert_annotation_id,
                    expert_id=expert_id,
                    annotation_task="segmentation",
                    raw_data_id=raw_file_id,
                    annotation_value=None,
                    confidence_level=None,
                    annotation_timestamp=None,
                )
                await upsert_expert_annotation(expert_annotation)
                
                # Process vessel segmentation
                segmentation = await process_segmentation_from_soft_map(
                    soft_map_path=mask_path,
                    annotation_type="vessels",
                    image_id=image_id,
                    annotation_description=f"Blood vessel segmentation (grayscale intensity map) by {VESSEL_EXPERTS[expert_key]['name']}",
                    raw_data_id=raw_file_id,
                    expert_annotation_id=expert_annotation_id,
                    annotation_method="manual",
                    provenance_chain_id=chain_id,
                )
                
                all_segmentations.append(segmentation)
                tracker.update(success=True)
                tracker.record_success("vessel_segmentation")
                
            except Exception as e:
                tracker.update(success=False)
                tracker.record_error(
                    error_type="vessel_segmentation_processing",
                    error_message=str(e),
                    item_id=mask_path.stem,
                    item_path=str(mask_path),
                )
                logger.error(f"Failed to process vessel mask {mask_path}: {e}")
    
    return all_segmentations


async def process_disc_fovea_segmentation(
    data_root: Path,
    dataset_id: UUID,
    image_id_map: Dict[str, UUID],
    tracker: ProgressTracker,
) -> List[SegmentationAnnotation]:
    """
    Process optic disc and fovea segmentation masks.
    
    Note: These are combined masks containing both disc and fovea.
    Only available for subsets c and d.
    
    Returns:
        List of SegmentationAnnotation models
    """
    all_segmentations: List[SegmentationAnnotation] = []
    
    # Process subsets c and d (only these have disc/fovea annotations)
    for subset_key in ["c", "d"]:
        discfovea_dir = data_root / f"aria_{subset_key}_markupdiscfovea"
        
        if not discfovea_dir.exists():
            logger.warning(f"Disc/fovea directory not found: {discfovea_dir}")
            continue
        
        # Find all disc/fovea masks
        discfovea_masks = list(discfovea_dir.glob("*.tif")) + list(discfovea_dir.glob("*.tiff"))
        logger.info(f"Found {len(discfovea_masks)} disc/fovea masks in subset {subset_key}")
        
        for mask_path in discfovea_masks:
            try:
                # Extract image identifier
                image_identifier = extract_image_identifier(mask_path)
                
                # Check if image exists
                if image_identifier not in image_id_map:
                    tracker.record_error(
                        error_type="image_not_found",
                        error_message=f"Image not found for mask: {image_identifier}",
                        item_id=image_identifier,
                        item_path=str(mask_path),
                    )
                    continue
                
                image_id = image_id_map[image_identifier]
                
                # Register mask file for provenance
                raw_file_id, chain_id = await register_individual_file(
                    file_path=mask_path,
                    dataset_id=dataset_id,
                    unified_annotation_type="segmentation",
                    file_type=None,
                    auto_detect_type=False,
                )
                
                # Process disc/fovea segmentation
                # Note: These masks contain both disc and fovea, we'll store as "optic_disc_and_fovea"
                segmentation = await process_segmentation_from_binary_mask(
                    mask_path=mask_path,
                    annotation_type="optic_disc_and_fovea",
                    image_id=image_id,
                    annotation_description="Combined optic disc and fovea segmentation",
                    fill_holes=False,
                    raw_data_id=raw_file_id,
                    expert_annotation_id=None,  # No specific expert attribution
                    annotation_method="manual",
                    provenance_chain_id=chain_id,
                    dataset_name=DATASET_NAME,
                )
                
                all_segmentations.append(segmentation)
                tracker.update(success=True)
                tracker.record_success("disc_fovea_segmentation")
                
            except Exception as e:
                tracker.update(success=False)
                tracker.record_error(
                    error_type="disc_fovea_segmentation_processing",
                    error_message=str(e),
                    item_id=mask_path.stem,
                    item_path=str(mask_path),
                )
                logger.error(f"Failed to process disc/fovea mask {mask_path}: {e}")
    
    return all_segmentations


async def ingest_aria() -> OperationStatistics:
    """
    Main ingestion function for ARIA dataset.
    
    Returns:
        OperationStatistics with success/error counts
    """
    data_root = get_data_root() / "12_ARIA"
    dataset_id = generate_dataset_uuid(DATASET_NAME)
    
    logger.info("=" * 80)
    logger.info(f"Starting ARIA dataset ingestion")
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
        task_types=["segmentation"],
        description=(
            "ARIA (Automated Retinal Image Analysis) dataset contains retinal fundus images "
            "from three subsets representing different populations: healthy controls, "
            "diabetic retinopathy patients, and age-related macular degeneration patients. "
            "Annotations include blood vessel segmentation from two experts (BDP and BSS), "
            "and optic disc/fovea segmentation for subsets c and d."
        ),
    )
    await upsert_dataset(dataset)
    
    # Step 2: Register experts
    logger.info("Registering vessel segmentation experts...")
    expert_ids = await register_experts(dataset_id)
    
    # Step 3: Setup progress tracker
    # Estimate: ~23 images + ~46 vessel masks (2 per image) + ~11 disc/fovea masks
    tracker = ProgressTracker(
        total=100,  # Rough estimate, will update as we go
        description=f"Ingesting {DATASET_NAME}"
    )
    
    # Step 4: Process all images
    logger.info("Processing images from all subsets...")
    image_id_map = await process_images(
        data_root=data_root,
        dataset_id=dataset_id,
        tracker=tracker,
    )
    logger.info(f"Processed {len(image_id_map)} images")
    
    # Step 5: Process vessel segmentation
    logger.info("Processing vessel segmentation masks...")
    vessel_segmentations = await process_vessel_segmentation(
        data_root=data_root,
        dataset_id=dataset_id,
        image_id_map=image_id_map,
        expert_ids=expert_ids,
        tracker=tracker,
    )
    
    # Step 6: Upsert vessel segmentations
    logger.info(f"Upserting {len(vessel_segmentations)} vessel segmentation annotations...")
    for segmentation in vessel_segmentations:
        try:
            await upsert_segmentation_annotation(segmentation)
        except Exception as e:
            tracker.record_error(
                error_type="vessel_segmentation_upsert",
                error_message=str(e),
                item_id=str(segmentation.segmentation_id),
            )
            logger.error(f"Failed to upsert vessel segmentation: {e}")
    
    # Step 7: Process disc/fovea segmentation
    logger.info("Processing optic disc and fovea segmentation masks...")
    discfovea_segmentations = await process_disc_fovea_segmentation(
        data_root=data_root,
        dataset_id=dataset_id,
        image_id_map=image_id_map,
        tracker=tracker,
    )
    
    # Step 8: Upsert disc/fovea segmentations
    logger.info(f"Upserting {len(discfovea_segmentations)} disc/fovea segmentation annotations...")
    for segmentation in discfovea_segmentations:
        try:
            await upsert_segmentation_annotation(segmentation)
        except Exception as e:
            tracker.record_error(
                error_type="disc_fovea_segmentation_upsert",
                error_message=str(e),
                item_id=str(segmentation.segmentation_id),
            )
            logger.error(f"Failed to upsert disc/fovea segmentation: {e}")
    
    # Register splits — random 90/10 train+test, then 90/10 train+val (no class labels)
    all_image_ids_for_split = list(image_id_map.values())
    if all_image_ids_for_split:
        logger.info("Registering dataset splits...")
        await auto_stratified_splits(
            dataset_id=dataset_id,
            split_assignments={"train": all_image_ids_for_split},
            split_type="explicit",
        )

    tracker.finish()
    return tracker.get_statistics()


async def main():
    """Entry point for script execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    try:
        stats = await ingest_aria()
        
        logger.info("=" * 80)
        logger.info("Ingestion Summary:")
        logger.info(f"  Total items: {stats.total_items}")
        logger.info(f"  Successful: {stats.successful_items}")
        logger.info(f"  Failed: {stats.failed_items}")
        logger.info(f"  Skipped: {stats.skipped_items}")
        
        # Show breakdown by type
        if hasattr(stats, 'success_counts') and stats.success_counts:
            logger.info("")
            logger.info("  Breakdown by type:")
            for item_type, count in sorted(stats.success_counts.items()):
                logger.info(f"    {item_type}: {count}")
        
        if stats.errors:
            logger.warning(f"  Total errors: {len(stats.errors)}")
            # Group errors by type
            error_types = {}
            for error in stats.errors:
                error_type = error.get("error_type", "unknown")
                error_types[error_type] = error_types.get(error_type, 0) + 1
            
            for error_type, count in sorted(error_types.items()):
                logger.warning(f"    {error_type}: {count}")
        else:
            logger.info("  No errors encountered")
        
        logger.info("=" * 80)
        
        # Exit with error code if there were failures
        if stats.failed_items > 0:
            logger.error("Ingestion completed with errors")
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
