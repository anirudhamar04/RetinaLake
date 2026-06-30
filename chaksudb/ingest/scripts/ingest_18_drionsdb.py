"""
Ingestion script for DRIONS-DB dataset.

Dataset: DRIONS-DB - Digital Retinal Images for Optic Nerve Segmentation Database
Structure: JSON metadata file with image paths and contour paths
Annotations:
  - Optic disc segmentation (contour files, averaged expert annotations)

Key Features:
  - 110 color fundus images
  - Optic disc segmentation via contour coordinates
  - Averaged expert annotations (consensus)
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from uuid import UUID

from chaksudb.common.progress import ProgressTracker, OperationStatistics
from chaksudb.config.config import get_data_root
from chaksudb.db.models import (
    Dataset,
    Image,
    SegmentationAnnotation,
)
from chaksudb.db.queries import (
    upsert_dataset,
    bulk_upsert_images,
    upsert_segmentation_annotation,
)
from chaksudb.ingest.framework import (
    get_image_metadata_dict,
)
from chaksudb.ingest.framework.gen_uuid import (
    generate_dataset_uuid,
    generate_image_uuid,
)
from chaksudb.ingest.framework.ingestion_helpers import process_json
from chaksudb.ingest.framework.provenance_context import get_current_provenance
from chaksudb.ingest.framework.raw_file_helpers import register_individual_file
from chaksudb.ingest.framework.task_processors.segmentation_processor import (
    process_segmentation_from_contour,
)
from chaksudb.ingest.framework.image_metadata import extract_image_metadata
from chaksudb.ingest.framework.split_assigner import auto_stratified_splits

logger = logging.getLogger(__name__)

# Dataset metadata
DATASET_NAME = "DRIONS-DB"
DATASET_URL = "https://www.ia.uned.es/~ejcarmona/DRIONS-DB.html"
DATASET_LICENSE = "Research/Academic Use"


async def process_metadata_entry(
    entry: dict,
    idx: int,
    data_root: Path,
    dataset_id: UUID,
    tracker: ProgressTracker,
) -> Tuple[Optional[Image], List[SegmentationAnnotation]]:
    """
    Process a single entry from metadata.json.
    
    Args:
        entry: JSON entry with image_file_name, image_file_path, segmentation
        idx: Entry index
        data_root: Dataset root directory
        dataset_id: Dataset UUID
        tracker: Progress tracker
    
    Returns:
        Tuple of (Image, List[SegmentationAnnotation]) or (None, []) on error
    """
    try:
        # Get provenance from context (set by process_json)
        json_raw_file_id, json_chain_id = get_current_provenance()
        
        # Extract image information
        image_file_name = entry.get("image_file_name")
        image_file_path_rel = entry.get("image_file_path", "")
        
        if not image_file_name:
            logger.warning(f"Entry {idx}: Missing image_file_name")
            tracker.update(success=False)
            tracker.record_error(
                error_type="missing_image_name",
                error_message="Missing image_file_name in entry",
                item_id=f"entry_{idx}",
            )
            return None, None
        
        # Resolve image path from JSON metadata (relative to metadata.json location)
        # JSON has paths like "../FundusImages/..." which should resolve to actual file locations
        # Files have the same name, so we can match by filename if resolved path doesn't exist
        image_path = None
        if image_file_path_rel:
            # Resolve relative path from metadata.json location
            metadata_json_path = data_root / "metadata.json"
            resolved_path = (metadata_json_path.parent / image_file_path_rel).resolve()
            if resolved_path.exists():
                image_path = resolved_path
        
        # If resolved path doesn't exist, try finding by filename in documents/ folder
        if image_path is None or not image_path.exists():
            image_path = data_root / "documents" / image_file_name
        
        if not image_path.exists():
            logger.warning(f"Entry {idx}: Image not found: {image_path}")
            tracker.update(success=False)
            tracker.record_error(
                error_type="image_not_found",
                error_message=f"Image file not found: {image_path}",
                item_id=image_file_name,
                item_path=str(image_path),
            )
            return None, None
        
        # Extract image dimensions (required for contour processing)
        try:
            image_metadata = extract_image_metadata(image_path)
            if image_metadata.resolution_width is None or image_metadata.resolution_height is None:
                logger.warning(f"Entry {idx}: Could not extract image dimensions from {image_path}")
                tracker.update(success=False)
                tracker.record_error(
                    error_type="missing_dimensions",
                    error_message=f"Could not extract image dimensions: {image_path}",
                    item_id=image_file_name,
                    item_path=str(image_path),
                )
                return None, None
            image_size = (image_metadata.resolution_width, image_metadata.resolution_height)
        except Exception as e:
            logger.error(f"Entry {idx}: Failed to extract image dimensions: {e}")
            tracker.update(success=False)
            tracker.record_error(
                error_type="dimension_extraction_error",
                error_message=str(e),
                item_id=image_file_name,
                item_path=str(image_path),
            )
            return None, None
        
        # Generate image UUID
        original_image_id = Path(image_file_name).stem
        image_id = generate_image_uuid(dataset_id, original_image_id)
        
        # Create image model
        image = Image(
            image_id=image_id,
            dataset_id=dataset_id,
            original_image_id=original_image_id,
            **get_image_metadata_dict(image_path),
            modality="fundus",
            acquisition_date=None,
            image_quality=None,
        )
        
        # Process segmentation (contour files)
        segmentations: List[SegmentationAnnotation] = []
        segmentation_list = entry.get("segmentation", [])
        
        if not segmentation_list:
            logger.warning(f"Entry {idx}: No segmentation data found for {image_file_name}")
            tracker.update(success=False)
            tracker.record_error(
                error_type="missing_segmentation",
                error_message="No segmentation data in entry",
                item_id=image_file_name,
            )
            return image, []
        
        # Process each segmentation entry (typically one OD contour)
        for seg_entry in segmentation_list:
            seg_type = seg_entry.get("type", "").strip()
            contour_path_rel = seg_entry.get("contour_path", "").strip()
            
            if not contour_path_rel:
                logger.warning(f"Entry {idx}: Missing contour_path in segmentation entry")
                continue
            
            # Resolve contour path from JSON metadata (relative to metadata.json location)
            # JSON has paths like "../Contours/OD/..." which should resolve to actual file locations
            # Files have the same name, so we can match by filename if resolved path doesn't exist
            contour_path = None
            metadata_json_path = data_root / "metadata.json"
            resolved_contour = (metadata_json_path.parent / contour_path_rel).resolve()
            if resolved_contour.exists():
                contour_path = resolved_contour
            
            # If resolved path doesn't exist, try finding by filename
            if contour_path is None or not contour_path.exists():
                contour_filename = Path(contour_path_rel).name
                # Try to determine subfolder from path or type
                if "OD" in seg_type.upper() or "OD" in contour_path_rel:
                    contour_path = data_root / "Contours" / "OD" / contour_filename
                else:
                    # Fallback: try to find in Contours folder
                    contour_path = data_root / "Contours" / contour_filename
            
            if not contour_path.exists():
                logger.warning(f"Entry {idx}: Contour file not found: {contour_path}")
                tracker.record_error(
                    error_type="contour_not_found",
                    error_message=f"Contour file not found: {contour_path}",
                    item_id=image_file_name,
                    item_path=str(contour_path),
                )
                continue
            
            # Register contour file for provenance
            contour_raw_file_id, contour_chain_id = await register_individual_file(
                file_path=contour_path,
                dataset_id=dataset_id,
                unified_annotation_type="segmentation",
                file_type="txt",
            )
            
            # Map segmentation type to annotation type
            annotation_type_map = {
                "OD": "optic_disc",
                "OC": "optic_cup",
            }
            annotation_type = annotation_type_map.get(seg_type, seg_type.lower())
            
            # Process contour to segmentation annotation
            segmentation = await process_segmentation_from_contour(
                contour_path=contour_path,
                annotation_type=annotation_type,
                image_id=image_id,
                image_size=image_size,
                annotation_description=f"{seg_type} segmentation from averaged expert annotations",
                raw_data_id=contour_raw_file_id,
                expert_annotation_id=None,  # Averaged annotations, no single expert
                annotation_method="manual",
                provenance_chain_id=contour_chain_id,
                dataset_name=DATASET_NAME,
                coordinate_format="line_separated",  # Format: "x y" per line
            )
            
            segmentations.append(segmentation)
        
        if not segmentations:
            logger.warning(f"Entry {idx}: Failed to process any segmentation for {image_file_name}")
            tracker.update(success=False)
            tracker.record_error(
                error_type="segmentation_processing_failed",
                error_message="Failed to process segmentation",
                item_id=image_file_name,
            )
            return image, []
        
        tracker.update(success=True)
        tracker.record_success("image")
        for _ in segmentations:
            tracker.record_success("segmentation")
        
        return image, segmentations
        
    except Exception as e:
        logger.error(f"Entry {idx}: Error processing entry: {e}", exc_info=True)
        tracker.update(success=False)
        tracker.record_error(
            error_type="entry_processing_error",
            error_message=str(e),
            item_id=f"entry_{idx}",
        )
        return None, None


async def ingest_drionsdb() -> OperationStatistics:
    """
    Main ingestion function for DRIONS-DB dataset.
    
    Returns:
        OperationStatistics with success/error counts
    """
    data_root = get_data_root() / "18_DRIONS-DB"
    dataset_id = generate_dataset_uuid(DATASET_NAME)
    
    logger.info("=" * 80)
    logger.info(f"Starting {DATASET_NAME} dataset ingestion")
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
            "DRIONS-DB (Digital Retinal Images for Optic Nerve Segmentation Database) "
            "contains 110 color fundus images with optic disc segmentation annotations. "
            "The segmentations are provided as contour coordinates representing averaged "
            "expert annotations (consensus)."
        ),
    )
    await upsert_dataset(dataset)
    
    # Step 2: Setup progress tracker
    # Estimate: ~110 images + ~110 segmentations
    tracker = ProgressTracker(
        total=220,  # Rough estimate
        description=f"Ingesting {DATASET_NAME}"
    )
    
    # Step 3: Process metadata.json
    logger.info("Processing metadata.json...")
    metadata_path = data_root / "metadata.json"
    
    if not metadata_path.exists():
        logger.error(f"Metadata file not found: {metadata_path}")
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    # Collect images and segmentations
    all_images: List[Image] = []
    all_segmentations: List[SegmentationAnnotation] = []
    
    async def process_entry(entry: dict, idx: int) -> None:
        """Wrapper to process JSON entry and collect results."""
        image, segmentations = await process_metadata_entry(
            entry=entry,
            idx=idx,
            data_root=data_root,
            dataset_id=dataset_id,
            tracker=tracker,
        )
        if image:
            all_images.append(image)
        all_segmentations.extend(segmentations)
    
    # Process JSON with automatic provenance
    stats, json_raw_file_id, json_chain_id = await process_json(
        json_path=metadata_path,
        dataset_id=dataset_id,
        unified_annotation_type="segmentation",
        process_entry_fn=process_entry,
        progress_tracker=tracker,
        skip_errors=True,
    )
    
    logger.info(
        f"Processed metadata.json: {stats.successful_items} successful, "
        f"{stats.failed_items} failed"
    )
    logger.info(f"Collected {len(all_images)} images and {len(all_segmentations)} segmentations")
    
    # Step 4: Bulk upsert images
    logger.info(f"Upserting {len(all_images)} images...")
    await bulk_upsert_images(all_images, batch_size=1000)
    
    # Step 5: Upsert segmentations (individual, no bulk operation)
    logger.info(f"Upserting {len(all_segmentations)} segmentation annotations...")
    for segmentation in all_segmentations:
        try:
            await upsert_segmentation_annotation(segmentation)
        except Exception as e:
            tracker.record_error(
                error_type="segmentation_upsert",
                error_message=str(e),
                item_id=str(segmentation.segmentation_id),
            )
            logger.error(f"Failed to upsert segmentation: {e}")
    
    # Register splits — random 90/10 train+test, then 90/10 train+val (no class labels)
    all_image_ids_for_split = [img.image_id for img in all_images]
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
        stats = await ingest_drionsdb()
        
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
