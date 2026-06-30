"""
Ingestion script for G1020 dataset.

Dataset: G1020 - Glaucoma classification and segmentation dataset
Structure: CSV file with binary labels, images in Images/ folder, contours in Contours/
Annotations:
  - Classification: healthy/glaucoma (binary) from CSV binaryLabels column
  - Segmentation: Optic disc (OD) and optic cup (OC) contours matched by filename

Key Features:
  - CSV with imageID (image_N.jpg format) and binaryLabels (0=healthy, 1=glaucoma)
  - Images in Images/ folder named as 21_G1020_NNN.jpg
  - Contour files in Contours/OD/ and Contours/OC/ named as 21_G1020_NNN.txt
  - Filename matching: image_9.jpg -> 21_G1020_009.jpg
"""

import asyncio
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from uuid import UUID

from chaksudb.common.progress import ProgressTracker, OperationStatistics
from chaksudb.config.config import get_data_root
from chaksudb.db.models import (
    Dataset,
    Image,
    SegmentationAnnotation,
    ClassificationAnnotation,
)
from chaksudb.db.queries import (
    upsert_dataset,
    bulk_upsert_images,
    bulk_upsert_classification_annotations,
    upsert_segmentation_annotation,
)
from chaksudb.ingest.framework import (
    get_image_metadata_dict,
)
from chaksudb.ingest.framework.gen_uuid import (
    generate_dataset_uuid,
    generate_image_uuid,
)
from chaksudb.ingest.framework.ingestion_helpers import process_csv
from chaksudb.ingest.framework.provenance_context import get_current_provenance
from chaksudb.ingest.framework.raw_file_helpers import register_individual_file
from chaksudb.ingest.framework.task_processors.segmentation_processor import (
    process_segmentation_from_contour,
)
from chaksudb.ingest.framework.task_processors.classification_processor import (
    process_classification,
)
from chaksudb.ingest.framework.image_metadata import extract_image_metadata
from chaksudb.ingest.framework.split_assigner import auto_stratified_splits

logger = logging.getLogger(__name__)

# Dataset metadata
DATASET_NAME = "G1020"
DATASET_URL = "https://arxiv.org/abs/2006.09158"
DATASET_LICENSE = "Research/Academic Use"


def convert_csv_image_id_to_filename(csv_image_id: str) -> Optional[str]:
    """
    Convert CSV imageID format to actual image filename.
    
    Args:
        csv_image_id: Image ID from CSV (e.g., "image_9.jpg")
    
    Returns:
        Actual image filename (e.g., "21_G1020_009.jpg") or None if parsing fails
    """
    # Pattern: image_N.jpg -> 21_G1020_NNN.jpg
    match = re.search(r"image_(\d+)\.jpg", csv_image_id)
    if not match:
        return None
    
    number = match.group(1)
    # Pad with leading zeros to 3 digits
    padded_number = number.zfill(3)
    return f"21_G1020_{padded_number}.jpg"


async def process_csv_entry(
    row: dict,
    idx: int,
    data_root: Path,
    dataset_id: UUID,
    tracker: ProgressTracker,
) -> Tuple[Optional[Image], List[ClassificationAnnotation], List[SegmentationAnnotation], Optional[int]]:
    """
    Process a single row from G1020.csv.
    
    Args:
        row: CSV row with imageID and binaryLabels
        idx: Row index
        data_root: Dataset root directory
        dataset_id: Dataset UUID
        tracker: Progress tracker
    
    Returns:
        Tuple of (Image, List[ClassificationAnnotation], List[SegmentationAnnotation])
        or (None, [], []) on error
    """
    try:
        # Get provenance from context (set by process_csv)
        csv_raw_file_id, csv_chain_id = get_current_provenance()
        
        # Extract image information from CSV
        csv_image_id = row.get("imageID", "").strip()
        if not csv_image_id:
            logger.warning(f"Row {idx}: Missing imageID")
            tracker.update(success=False)
            tracker.record_error(
                error_type="missing_image_id",
                error_message="Missing imageID in CSV row",
                item_id=f"row_{idx}",
            )
            return None, [], [], None
        
        # Convert CSV imageID to actual filename
        image_file_name = convert_csv_image_id_to_filename(csv_image_id)
        if not image_file_name:
            logger.warning(f"Row {idx}: Could not convert imageID to filename: {csv_image_id}")
            tracker.update(success=False)
            tracker.record_error(
                error_type="invalid_image_id",
                error_message=f"Could not parse imageID: {csv_image_id}",
                item_id=csv_image_id,
            )
            return None, [], [], None
        
        # Find image file
        image_path = data_root / "Images" / image_file_name
        if not image_path.exists():
            logger.warning(f"Row {idx}: Image not found: {image_path}")
            tracker.update(success=False)
            tracker.record_error(
                error_type="image_not_found",
                error_message=f"Image file not found: {image_path}",
                item_id=csv_image_id,
                item_path=str(image_path),
            )
            return None, [], [], None
        
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
                return None, [], [], None
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
            return None, [], [], None
        
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
        
        # Process classification from binaryLabels column
        classifications: List[ClassificationAnnotation] = []
        binary_label_str = row.get("binaryLabels", "").strip()
        binary_label: Optional[int] = None  # Track for stratified splitting
        
        if binary_label_str == "":
            logger.warning(f"Row {idx}: Missing binaryLabels for {csv_image_id}")
            tracker.record_error(
                error_type="missing_binary_label",
                error_message="Missing binaryLabels in CSV row",
                item_id=csv_image_id,
            )
        else:
            try:
                binary_label = int(binary_label_str)
                if binary_label not in [0, 1]:
                    logger.warning(f"Row {idx}: Invalid binaryLabel value: {binary_label}")
                    tracker.record_error(
                        error_type="invalid_binary_label",
                        error_message=f"Invalid binaryLabel value: {binary_label}",
                        item_id=csv_image_id,
                    )
                else:
                    # 0 = healthy, 1 = glaucoma
                    classifications_list = await process_classification(
                        class_value=binary_label,
                        task_type="binary",
                        class_name="glaucoma",
                        image_id=image_id,
                        class_labels={
                            0: "healthy",
                            1: "glaucoma",
                        },
                        raw_data_id=csv_raw_file_id,
                        provenance_chain_id=csv_chain_id,
                        annotation_method="manual",
                    )
                    classifications.extend(classifications_list)
            except ValueError:
                logger.warning(f"Row {idx}: Could not parse binaryLabels as integer: {binary_label_str}")
                tracker.record_error(
                    error_type="invalid_binary_label_format",
                    error_message=f"Could not parse binaryLabels: {binary_label_str}",
                    item_id=csv_image_id,
                )
        
        # Process segmentation (contour files) - match by image filename
        segmentations: List[SegmentationAnnotation] = []
        
        # Extract base name without extension for matching contour files
        base_name = Path(image_file_name).stem  # e.g., "21_G1020_009"
        
        # Try to find OD and OC contour files
        contour_types = [
            ("OD", "optic_disc"),
            ("OC", "optic_cup"),
        ]
        
        for seg_type, annotation_type in contour_types:
            contour_path = data_root / "Contours" / seg_type / f"{base_name}.txt"
            
            if contour_path.exists():
                # Register contour file for provenance
                contour_raw_file_id, contour_chain_id = await register_individual_file(
                    file_path=contour_path,
                    dataset_id=dataset_id,
                    unified_annotation_type="segmentation",
                    file_type="txt",
                )
                
                # Process contour to segmentation annotation
                segmentation = await process_segmentation_from_contour(
                    contour_path=contour_path,
                    annotation_type=annotation_type,
                    image_id=image_id,
                    image_size=image_size,
                    annotation_description=f"{seg_type} segmentation from G1020 contours",
                    raw_data_id=contour_raw_file_id,
                    expert_annotation_id=None,
                    annotation_method="manual",
                    provenance_chain_id=contour_chain_id,
                    dataset_name=DATASET_NAME,
                    coordinate_format="line_separated",  # Format: "x y" per line
                )
                
                segmentations.append(segmentation)
            else:
                # Log missing contour but don't fail - not all images may have both OD and OC
                logger.debug(f"Row {idx}: {seg_type} contour not found: {contour_path}")
        
        # Consider row successful if we have at least classification
        # Segmentation is optional (not all images may have contours)
        if classifications:
            tracker.update(success=True)
            tracker.record_success("classification")
            for _ in segmentations:
                tracker.record_success("segmentation")
        else:
            logger.warning(f"Row {idx}: No valid annotations processed for {csv_image_id}")
            tracker.update(success=False)
            tracker.record_error(
                error_type="no_annotations",
                error_message="No valid annotations processed",
                item_id=csv_image_id,
            )
        
        return image, classifications, segmentations, binary_label

    except Exception as e:
        logger.error(f"Row {idx}: Error processing row: {e}", exc_info=True)
        tracker.update(success=False)
        tracker.record_error(
            error_type="row_processing_error",
            error_message=str(e),
            item_id=f"row_{idx}",
        )
        return None, [], [], None


async def ingest_g1020() -> OperationStatistics:
    """
    Main ingestion function for G1020 dataset.
    
    Returns:
        OperationStatistics with success/error counts
    """
    data_root = get_data_root() / "21_G1020"
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
        task_types=["classification", "segmentation"],
        description=(
            "G1020 dataset contains fundus images with glaucoma classification "
            "and optic disc/cup segmentation annotations. The dataset includes "
            "binary classification (healthy vs glaucoma) and contour-based "
            "segmentation for optic disc (OD) and optic cup (OC)."
        ),
    )
    await upsert_dataset(dataset)
    
    # Step 2: Setup progress tracker
    # Estimate: ~102 images + ~102 classifications + ~200 segmentations (OD+OC)
    tracker = ProgressTracker(
        total=404,  # Rough estimate
        description=f"Ingesting {DATASET_NAME}"
    )
    
    # Step 3: Process G1020.csv
    logger.info("Processing G1020.csv...")
    csv_path = data_root / "G1020.csv"
    
    if not csv_path.exists():
        logger.error(f"CSV file not found: {csv_path}")
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Collect images, classifications, and segmentations
    all_images: List[Image] = []
    all_classifications: List[ClassificationAnnotation] = []
    all_segmentations: List[SegmentationAnnotation] = []
    image_labels: dict = {}

    async def process_row(row: dict, idx: int) -> None:
        """Wrapper to process CSV row and collect results."""
        image, classifications, segmentations, label = await process_csv_entry(
            row=row,
            idx=idx,
            data_root=data_root,
            dataset_id=dataset_id,
            tracker=tracker,
        )
        if image:
            all_images.append(image)
            if label is not None:
                image_labels[image.image_id] = label
        all_classifications.extend(classifications)
        all_segmentations.extend(segmentations)
    
    # Process CSV with automatic provenance
    stats, csv_raw_file_id, csv_chain_id = await process_csv(
        csv_path=csv_path,
        dataset_id=dataset_id,
        unified_annotation_type="classification",  # Primary task
        process_row_fn=process_row,
        progress_tracker=tracker,
        skip_errors=True,
    )
    
    logger.info(
        f"Processed G1020.csv: {stats.successful_items} successful, "
        f"{stats.failed_items} failed"
    )
    logger.info(
        f"Collected {len(all_images)} images, {len(all_classifications)} classifications, "
        f"and {len(all_segmentations)} segmentations"
    )
    
    # Step 4: Bulk upsert images
    logger.info(f"Upserting {len(all_images)} images...")
    await bulk_upsert_images(all_images, batch_size=1000)
    
    # Step 5: Bulk upsert classifications
    logger.info(f"Upserting {len(all_classifications)} classification annotations...")
    await bulk_upsert_classification_annotations(all_classifications, batch_size=1000)
    
    # Step 6: Upsert segmentations (individual, no bulk operation available)
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

    tracker.finish()
    return tracker.get_statistics()


async def main():
    """Entry point for script execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    try:
        stats = await ingest_g1020()
        
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
