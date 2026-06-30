"""
Ingestion script for Drishti-GS1 dataset.

Dataset: Drishti-GS1 - Glaucoma Screening dataset
Structure: Excel files + Training/Test images with multi-expert annotations
Annotations:
  - Glaucoma classification (from Excel consensus column)
  - Expert glaucoma markings (Marking 1-4: -1 for normal, 1 for glaucomatous)
  - Multi-expert OD/cup segmentation:
    * Soft maps (OD and cup, consensus from 4 experts)
    * Average boundaries (OD and cup, converted from contours, consensus from 4 experts)
  - CDR values (Cup-to-Disc Ratio from 4 experts, registered as raw files only)
  - Disc center localization

Key Features:
  - Training and Test splits
  - 4 expert annotations per image
  - CDR (Cup-to-Disc Ratio): Key glaucoma metric - ratio of cup area to disc area
    Higher CDR values (>0.5-0.6) indicate potential glaucoma
  - Both soft maps AND average boundaries are saved (different UUIDs via raw_data_id)
  - Expert markings: -1 = normal, 1 = glaucomatous (from Excel, used for classification)
  - CDR files registered as raw files for provenance (CDR can be derived from segmentations)
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
    ClassificationAnnotation,
    LocalizationAnnotation,
    Expert,
    ExpertAnnotation,
)
from chaksudb.db.queries import (
    upsert_dataset,
    bulk_upsert_images,
    upsert_segmentation_annotation,
    bulk_upsert_classification_annotations,
    bulk_upsert_localization_annotations,
    bulk_upsert_expert_annotations,
    upsert_expert
)
from chaksudb.ingest.framework import (
    get_image_metadata_dict,
    find_images,
    process_excel,
)
from chaksudb.ingest.framework.gen_uuid import (
    generate_dataset_uuid,
    generate_image_uuid,
    generate_expert_uuid,
    generate_expert_annotation_uuid,
)
from chaksudb.ingest.framework.image_metadata import extract_image_metadata
from chaksudb.ingest.framework.provenance_context import get_current_provenance
from chaksudb.ingest.framework.raw_file_helpers import register_individual_file
from chaksudb.ingest.framework.split_assigner import (
    register_standard_splits,
    bulk_assign_images_to_split,
)
from chaksudb.ingest.framework.task_processors.segmentation_processor import (
    process_segmentation_from_soft_map,
    process_segmentation_from_contour,
)
from chaksudb.ingest.framework.task_processors.classification_processor import (
    process_classification,
)
from chaksudb.ingest.framework.task_processors.localization_processor import (
    process_localization_from_text_keypoint,
)

logger = logging.getLogger(__name__)

# Dataset metadata
DATASET_NAME = "Drishti-GS1"
DATASET_URL = "https://cvit.iiit.ac.in/projects/mip/drishti-gs/mip-dataset2/Home.php"
DATASET_LICENSE = "Research/Academic Use"

# Expert information
NUM_EXPERTS = 4
EXPERTS = {
    f"Expert_{i+1}": {
        "name": f"Drishti-GS1 Expert {i+1}",
        "expertise": "glaucoma_screening",
    }
    for i in range(NUM_EXPERTS)
}


async def register_experts(dataset_id: UUID) -> Dict[str, UUID]:
    """Register the 4 experts for Drishti-GS1."""
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
        )
        
        await upsert_expert(expert)
        expert_ids[expert_key] = expert_id
        logger.info(f"Registered expert: {expert_info['name']} ({expert_key})")
    
    return expert_ids


def parse_image_id_from_folder(folder_name: str) -> Optional[str]:
    """
    Parse image ID from folder name.
    
    Args:
        folder_name: Folder name like "drishtiGS_031"
    
    Returns:
        Image ID string like "drishtiGS_031" or None if invalid
    """
    if folder_name.startswith("drishtiGS_"):
        return folder_name
    return None


def parse_cdr_values(cdr_file: Path) -> List[float]:
    """
    Parse CDR values from text file.
    
    Format: "0.74 0.81 0.60 0.68" (4 space-separated values, one per expert)
    
    Args:
        cdr_file: Path to CDR values file
    
    Returns:
        List of 4 CDR values (one per expert)
    """
    if not cdr_file.exists():
        raise FileNotFoundError(f"CDR file not found: {cdr_file}")
    
    with open(cdr_file, "r") as f:
        line = f.readline().strip()
    
    if not line:
        raise ValueError(f"Empty CDR file: {cdr_file}")
    
    parts = line.split()
    if len(parts) != NUM_EXPERTS:
        raise ValueError(
            f"Expected {NUM_EXPERTS} CDR values, got {len(parts)} in {cdr_file}"
        )
    
    try:
        cdr_values = [float(x) for x in parts]
        return cdr_values
    except ValueError as e:
        raise ValueError(f"Invalid CDR values in {cdr_file}: {line} ({e})")


async def process_diagnosis_excel(
    excel_path: Path,
    dataset_id: UUID,
    tracker: ProgressTracker,
) -> Dict[str, Dict]:
    """
    Process diagnosis Excel file and build lookup.
    
    Expected columns:
    - Image ID / ImageID / image_id
    - Marking 1, Marking 2, Marking 3, Marking 4 (values: -1 for normal, 1 for glaucomatous)
    - Consensus (consensus diagnosis)
    - Other diagnosis columns
    
    Returns:
        Dictionary mapping image_id -> {
            glaucoma_label (from consensus),
            expert_markings: [marking1, marking2, marking3, marking4],
            consensus_value,
            raw_file_id,
            chain_id
        }
    """
    diagnosis_lookup: Dict[str, Dict] = {}
    
    async def process_diagnosis_row(row: Dict, idx: int):
        try:
            # Get provenance from context
            raw_file_id, chain_id = get_current_provenance()
            
            # Extract image ID (column is "Drishti-GS File" in the Excel)
            image_id_col = None
            for col in ["Drishti-GS File", "Drishti-GS File", "drishti-gs file", "Image ID", "ImageID", "image_id", "Image", "image"]:
                if col in row:
                    image_id_col = col
                    break
            
            if not image_id_col:
                logger.warning(f"Row {idx}: Could not find image ID column. Available: {list(row.keys())}")
                tracker.update(success=False)
                tracker.record_error(
                    error_type="missing_image_id_column",
                    error_message=f"Could not find image ID column in row {idx}",
                    item_id=f"row_{idx}",
                )
                return
            
            image_id_str = str(row[image_id_col]).strip()
            # Remove trailing quote if present (e.g., "drishtiGS_003'" -> "drishtiGS_003")
            if image_id_str.endswith("'"):
                image_id_str = image_id_str[:-1]
            if not image_id_str:
                logger.warning(f"Row {idx}: Empty image ID")
                tracker.update(success=False)
                tracker.record_error(
                    error_type="empty_image_id",
                    error_message=f"Empty image ID in row {idx}",
                    item_id=f"row_{idx}",
                )
                return
            
            # Extract expert markings (Marking 1-4: -1 for normal, 1 for glaucomatous)
            expert_markings = []
            for i in range(1, NUM_EXPERTS + 1):
                marking_col = None
                for col in [f"Marking {i}", f"marking {i}", f"Marking{i}", f"marking{i}", 
                           f"Expert {i}", f"expert {i}", f"Expert{i}", f"expert{i}"]:
                    if col in row:
                        marking_col = col
                        break
                
                marking_value = None
                if marking_col:
                    marking_raw = row[marking_col]
                    if marking_raw is not None:
                        try:
                            marking_value = int(float(marking_raw))  # Handle both int and float
                        except (ValueError, TypeError):
                            # Try string parsing
                            marking_str = str(marking_raw).strip().lower()
                            if marking_str in ["1", "glaucoma", "g", "glaucomatous"]:
                                marking_value = 1
                            elif marking_str in ["-1", "normal", "n", "0"]:
                                marking_value = -1
                
                expert_markings.append(marking_value)
            
            # Extract consensus column (named "Total" in the Excel file)
            consensus_col = None
            for col in ["Total", "total", "Consensus", "consensus", "Consensus Diagnosis", "consensus_diagnosis"]:
                if col in row:
                    consensus_col = col
                    break
            
            consensus_value = None
            glaucoma_label = None
            if consensus_col:
                consensus_raw = row[consensus_col]
                if consensus_raw is not None:
                    consensus_str = str(consensus_raw).strip().lower()
                    # Map consensus to glaucoma label
                    # Values can be "Glaucomatous" or "Normal" (strings) or numeric
                    if consensus_str in ["1", "yes", "true", "glaucoma", "g", "glaucomatous"]:
                        glaucoma_label = "glaucoma"
                        consensus_value = 1
                    elif consensus_str in ["-1", "0", "no", "false", "normal", "n"]:
                        glaucoma_label = "normal"
                        consensus_value = -1
                    else:
                        # Store original value
                        consensus_value = str(consensus_raw)
                        # Try to infer label from value
                        if "glaucomatous" in consensus_str:
                            glaucoma_label = "glaucoma"
                        elif "normal" in consensus_str:
                            glaucoma_label = "normal"
            
            # Also try to extract from other common diagnosis columns
            if not glaucoma_label:
                for col in ["Glaucoma", "glaucoma", "Diagnosis", "diagnosis", "Label", "label"]:
                    if col in row and col not in [consensus_col, image_id_col]:
                        glaucoma_value = row[col]
                        if glaucoma_value is not None:
                            glaucoma_str = str(glaucoma_value).strip().lower()
                            if glaucoma_str in ["1", "yes", "true", "glaucoma", "g"]:
                                glaucoma_label = "glaucoma"
                            elif glaucoma_str in ["0", "no", "false", "normal", "n", "-1"]:
                                glaucoma_label = "normal"
                            break
            
            # Store in lookup
            diagnosis_lookup[image_id_str] = {
                "glaucoma_label": glaucoma_label,
                "expert_markings": expert_markings,  # List of 4 values: -1 or 1
                "consensus_value": consensus_value,
                "raw_file_id": raw_file_id,
                "chain_id": chain_id,
            }
            
            tracker.update(success=True)
            tracker.record_success("diagnosis_row")
            
        except Exception as e:
            tracker.update(success=False)
            tracker.record_error(
                error_type="diagnosis_row_processing",
                error_message=str(e),
                item_id=f"row_{idx}",
            )
            logger.error(f"Failed to process diagnosis row {idx}: {e}")
    
    # Process Excel with automatic provenance
    stats, raw_file_id, chain_id = await process_excel(
        excel_path=excel_path,
        dataset_id=dataset_id,
        unified_annotation_type="classification",
        process_row_fn=process_diagnosis_row,
        sheet_name=0,  # First sheet
        progress_tracker=tracker,
        skip_errors=True,
    )
    
    logger.info(
        f"Processed diagnosis Excel: {stats.successful_items} successful, "
        f"{stats.failed_items} failed. Built lookup for {len(diagnosis_lookup)} images."
    )
    
    return diagnosis_lookup


async def process_notching_excel(
    excel_path: Path,
    dataset_id: UUID,
    tracker: ProgressTracker,
) -> Dict[str, Dict]:
    """
    Process notching decisions Excel file and build lookup.
    
    Returns:
        Dictionary mapping image_id -> {notching_label, ...}
    """
    notching_lookup: Dict[str, Dict] = {}
    
    async def process_notching_row(row: Dict, idx: int):
        try:
            # Get provenance from context
            raw_file_id, chain_id = get_current_provenance()
            
            # Extract image ID (column is "Drishti-GS File" in the Excel)
            image_id_col = None
            for col in ["Drishti-GS File", "Drishti-GS File", "drishti-gs file", "Image ID", "ImageID", "image_id", "Image", "image"]:
                if col in row:
                    image_id_col = col
                    break
            
            if not image_id_col:
                logger.warning(f"Row {idx}: Could not find image ID column")
                tracker.update(success=False)
                tracker.record_error(
                    error_type="missing_image_id_column",
                    error_message=f"Could not find image ID column in row {idx}",
                    item_id=f"row_{idx}",
                )
                return
            
            image_id_str = str(row[image_id_col]).strip()
            # Remove trailing quote if present (e.g., "drishtiGS_003'" -> "drishtiGS_003")
            if image_id_str.endswith("'"):
                image_id_str = image_id_str[:-1]
            if not image_id_str:
                return
            
            # Extract notching decision
            notching_col = None
            for col in ["Notching", "notching", "Decision", "decision", "Label", "label"]:
                if col in row:
                    notching_col = col
                    break
            
            notching_label = None
            if notching_col:
                notching_value = row[notching_col]
                if notching_value is not None:
                    notching_str = str(notching_value).strip().lower()
                    if notching_str in ["1", "yes", "true", "notching", "n"]:
                        notching_label = "notching"
                    elif notching_str in ["0", "no", "false", "no_notching", "nn"]:
                        notching_label = "no_notching"
            
            # Store in lookup
            notching_lookup[image_id_str] = {
                "notching_label": notching_label,
                "raw_file_id": raw_file_id,
                "chain_id": chain_id,
            }
            
            tracker.update(success=True)
            tracker.record_success("notching_row")
            
        except Exception as e:
            tracker.update(success=False)
            tracker.record_error(
                error_type="notching_row_processing",
                error_message=str(e),
                item_id=f"row_{idx}",
            )
            logger.error(f"Failed to process notching row {idx}: {e}")
    
    # Process Excel with automatic provenance
    stats, raw_file_id, chain_id = await process_excel(
        excel_path=excel_path,
        dataset_id=dataset_id,
        unified_annotation_type="classification",
        process_row_fn=process_notching_row,
        sheet_name=0,  # First sheet
        progress_tracker=tracker,
        skip_errors=True,
    )
    
    logger.info(
        f"Processed notching Excel: {stats.successful_items} successful, "
        f"{stats.failed_items} failed. Built lookup for {len(notching_lookup)} images."
    )
    
    return notching_lookup


async def process_image_with_annotations(
    image_path: Path,
    gt_folder: Path,
    dataset_id: UUID,
    expert_ids: Dict[str, UUID],
    diagnosis_lookup: Dict[str, Dict],
    notching_lookup: Dict[str, Dict],
    tracker: ProgressTracker,
) -> Tuple[
    Optional[Image],
    List[SegmentationAnnotation],
    List[ClassificationAnnotation],
    List[LocalizationAnnotation],
    UUID,
    List[ExpertAnnotation],
]:
    """
    Process a single image with all its annotations.
    
    Returns:
        Tuple of (Image, segmentations, classifications, localizations, expert_annotations, image_id)
    """
    try:
        image_stem = image_path.stem
        image_id_str = parse_image_id_from_folder(image_stem)
        
        if not image_id_str:
            logger.warning(f"Could not parse image ID from: {image_stem}")
            tracker.update(success=False)
            tracker.record_error(
                error_type="invalid_image_id",
                error_message=f"Could not parse image ID from: {image_stem}",
                item_id=image_stem,
                item_path=str(image_path),
            )
            return None, [], [], [], [], None
        
        # Generate image UUID
        image_id = generate_image_uuid(dataset_id, image_id_str)
        
        # Extract image dimensions (required for contour processing)
        try:
            image_metadata = extract_image_metadata(image_path)
            if image_metadata.resolution_width is None or image_metadata.resolution_height is None:
                logger.warning(f"Could not extract image dimensions from {image_path}")
                tracker.update(success=False)
                tracker.record_error(
                    error_type="missing_dimensions",
                    error_message=f"Could not extract image dimensions: {image_path}",
                    item_id=image_id_str,
                    item_path=str(image_path),
                )
                return None, [], [], [], [], None
            image_size = (image_metadata.resolution_width, image_metadata.resolution_height)
        except Exception as e:
            logger.error(f"Failed to extract image dimensions: {e}")
            tracker.update(success=False)
            tracker.record_error(
                error_type="dimension_extraction_error",
                error_message=str(e),
                item_id=image_id_str,
                item_path=str(image_path),
            )
            return None, [], [], [], [], None
        
        # Create image model
        image = Image(
            image_id=image_id,
            dataset_id=dataset_id,
            original_image_id=image_id_str,
            **get_image_metadata_dict(image_path),
            modality="fundus",
            acquisition_date=None,
            image_quality=None,
        )
        
        segmentations: List[SegmentationAnnotation] = []
        classifications: List[ClassificationAnnotation] = []
        localizations: List[LocalizationAnnotation] = []
        expert_annotations: List[ExpertAnnotation] = []
        
        # Register CDR file as raw file for provenance
        # Note: CDR (Cup-to-Disc Ratio) can be derived from cup/disc segmentations,
        # so we don't create ExpertAnnotation records - just register the file for provenance
        cdr_file = gt_folder / f"{image_id_str}_cdrValues.txt"
        if cdr_file.exists():
            try:
                # Register CDR file for provenance (CDR is derived from segmentation)
                await register_individual_file(
                    file_path=cdr_file,
                    dataset_id=dataset_id,
                    unified_annotation_type="segmentation",
                    file_type="txt",
                )
            except Exception as e:
                logger.warning(f"Failed to register CDR file for {image_id_str}: {e}")
                tracker.record_error(
                    error_type="cdr_file_registration_error",
                    error_message=str(e),
                    item_id=image_id_str,
                    item_path=str(cdr_file),
                )
        
        # Process soft maps (multi-expert segmentation)
        softmap_dir = gt_folder / "SoftMap"
        if softmap_dir.exists():
            # Process OD soft maps (one per expert, but we have combined soft maps)
            od_softmap = softmap_dir / f"{image_id_str}_ODsegSoftmap.png"
            cup_softmap = softmap_dir / f"{image_id_str}_cupsegSoftmap.png"
            
            # Register soft map files for provenance
            # PNG files: set file_type=None (not in allowed list, will be NULL in DB)
            if od_softmap.exists():
                od_raw_file_id, od_chain_id = await register_individual_file(
                    file_path=od_softmap,
                    dataset_id=dataset_id,
                    unified_annotation_type="segmentation",
                    file_type=None,  # PNG not in allowed file types, will be NULL
                    auto_detect_type=False,  # Don't auto-detect PNG extension
                )
                
                # Process OD soft map (consensus from 4 experts)
                od_seg = await process_segmentation_from_soft_map(
                    soft_map_path=od_softmap,
                    annotation_type="optic_disc",
                    image_id=image_id,
                    annotation_description="Optic disc segmentation from soft map (4 experts consensus)",
                    raw_data_id=od_raw_file_id,
                    expert_annotation_id=None,  # Consensus, not single expert
                    consensus_id=None,  # Could create consensus annotation here
                    annotation_method="manual",
                    provenance_chain_id=od_chain_id,
                )
                segmentations.append(od_seg)
            
            if cup_softmap.exists():
                cup_raw_file_id, cup_chain_id = await register_individual_file(
                    file_path=cup_softmap,
                    dataset_id=dataset_id,
                    unified_annotation_type="segmentation",
                    file_type=None,  # PNG not in allowed file types, will be NULL
                    auto_detect_type=False,  # Don't auto-detect PNG extension
                )
                
                # Process cup soft map (consensus from 4 experts)
                cup_seg = await process_segmentation_from_soft_map(
                    soft_map_path=cup_softmap,
                    annotation_type="optic_cup",
                    image_id=image_id,
                    annotation_description="Optic cup segmentation from soft map (4 experts consensus)",
                    raw_data_id=cup_raw_file_id,
                    expert_annotation_id=None,  # Consensus, not single expert
                    consensus_id=None,
                    annotation_method="manual",
                    provenance_chain_id=cup_chain_id,
                )
                segmentations.append(cup_seg)
        
        # Process average boundaries (consensus segmentation from contours)
        avgboundary_dir = gt_folder / "AvgBoundary"
        if avgboundary_dir.exists():
            # Process OD average boundary
            od_boundary = avgboundary_dir / f"{image_id_str}_ODAvgBoundary.txt"
            if od_boundary.exists():
                od_boundary_raw_file_id, od_boundary_chain_id = await register_individual_file(
                    file_path=od_boundary,
                    dataset_id=dataset_id,
                    unified_annotation_type="segmentation",
                    file_type="txt",
                )
                
                od_boundary_seg = await process_segmentation_from_contour(
                    contour_path=od_boundary,
                    annotation_type="optic_disc",
                    image_id=image_id,
                    image_size=image_size,
                    annotation_description="Optic disc segmentation from average boundary (4 experts consensus)",
                    raw_data_id=od_boundary_raw_file_id,
                    expert_annotation_id=None,
                    annotation_method="manual",
                    provenance_chain_id=od_boundary_chain_id,
                    dataset_name=DATASET_NAME,
                    coordinate_format="line_separated",
                )
                segmentations.append(od_boundary_seg)
            
            # Process cup average boundary
            cup_boundary = avgboundary_dir / f"{image_id_str}_CupAvgBoundary.txt"
            if cup_boundary.exists():
                cup_boundary_raw_file_id, cup_boundary_chain_id = await register_individual_file(
                    file_path=cup_boundary,
                    dataset_id=dataset_id,
                    unified_annotation_type="segmentation",
                    file_type="txt",
                )
                
                cup_boundary_seg = await process_segmentation_from_contour(
                    contour_path=cup_boundary,
                    annotation_type="optic_cup",
                    image_id=image_id,
                    image_size=image_size,
                    annotation_description="Optic cup segmentation from average boundary (4 experts consensus)",
                    raw_data_id=cup_boundary_raw_file_id,
                    expert_annotation_id=None,
                    annotation_method="manual",
                    provenance_chain_id=cup_boundary_chain_id,
                    dataset_name=DATASET_NAME,
                    coordinate_format="line_separated",
                )
                segmentations.append(cup_boundary_seg)
            
            # Process disc center localization
            disc_center = avgboundary_dir / f"{image_id_str}_diskCenter.txt"
            if disc_center.exists():
                disc_center_raw_file_id, disc_center_chain_id = await register_individual_file(
                    file_path=disc_center,
                    dataset_id=dataset_id,
                    unified_annotation_type="localization",
                    file_type="txt",
                )
                
                disc_center_loc = await process_localization_from_text_keypoint(
                    txt_path=disc_center,
                    image_id=image_id,
                    structure_name="optic_disc_center",
                    raw_data_id=disc_center_raw_file_id,
                    annotation_method="manual",
                    provenance_chain_id=disc_center_chain_id,
                )
                localizations.append(disc_center_loc)
        
        # Process classification annotations (from Excel lookups)
        diagnosis_data = diagnosis_lookup.get(image_id_str, {})
        expert_markings = diagnosis_data.get("expert_markings", [])
        excel_raw_file_id = diagnosis_data.get("raw_file_id")
        excel_chain_id = diagnosis_data.get("chain_id")
        
        # Create ExpertAnnotation and ClassificationAnnotation for each expert marking
        # This connects experts to their glaucoma classifications
        for expert_idx, (expert_key, expert_id) in enumerate(expert_ids.items()):
            if expert_idx < len(expert_markings) and expert_markings[expert_idx] is not None:
                marking = expert_markings[expert_idx]
                marking_label = "glaucoma" if marking == 1 else "normal"
                # Convert to boolean for binary classification (glaucoma=True, normal=False)
                marking_bool = marking == 1
                
                # Create ExpertAnnotation to link expert to their classification
                expert_annotation_id = generate_expert_annotation_uuid(
                    expert_id=expert_id,
                    annotation_task="classification",
                    raw_data_id=excel_raw_file_id,
                    annotation_value_hash=None,
                )
                
                expert_annotation = ExpertAnnotation(
                    expert_annotation_id=expert_annotation_id,
                    expert_id=expert_id,
                    annotation_task="classification",
                    raw_data_id=excel_raw_file_id,
                    annotation_value={
                        "glaucoma_marking": marking,  # -1 for normal, 1 for glaucomatous
                        "glaucoma_label": marking_label,
                        "image_id": str(image_id),
                        "image_stem": image_id_str,
                    },
                    confidence_level=None,
                    annotation_timestamp=None,
                )
                expert_annotations.append(expert_annotation)
                
                # Create ClassificationAnnotation linked to ExpertAnnotation
                # Use boolean value for binary classification (True=glaucoma, False=normal)
                expert_classes = await process_classification(
                    class_value=marking_bool,  # Boolean: True for glaucoma, False for normal
                    task_type="binary",
                    class_name="glaucoma",
                    image_id=image_id,
                    raw_data_id=excel_raw_file_id,
                    expert_annotation_id=expert_annotation_id,  # Link to expert annotation
                    provenance_chain_id=excel_chain_id,
                    annotation_method="manual",
                )
                classifications.extend(expert_classes)
        
        # Glaucoma classification (from consensus "Total" column in Excel)
        # This is the consensus diagnosis, not linked to a specific expert
        if diagnosis_data.get("glaucoma_label"):
            # Convert label to boolean for binary classification (glaucoma=True, normal=False)
            consensus_label = diagnosis_data["glaucoma_label"]
            consensus_bool = consensus_label == "glaucoma"
            
            consensus_classes = await process_classification(
                class_value=consensus_bool,  # Boolean: True for glaucoma, False for normal
                task_type="binary",
                class_name="glaucoma",
                image_id=image_id,
                raw_data_id=excel_raw_file_id,
                expert_annotation_id=None,  # Consensus, not from single expert
                provenance_chain_id=excel_chain_id,
                annotation_method="manual",
            )
            classifications.extend(consensus_classes)
        
        # Notching classification disabled - Excel file not processed
        
        tracker.update(success=True)
        tracker.record_success("image")
        
        return image, segmentations, classifications, localizations, expert_annotations, image_id
        
    except Exception as e:
        logger.error(f"Failed to process image {image_path}: {e}", exc_info=True)
        tracker.update(success=False)
        tracker.record_error(
            error_type="image_processing_error",
            error_message=str(e),
            item_id=image_path.stem,
            item_path=str(image_path),
        )
        return None, [], [], [], [], None


async def ingest_drishti_gs1() -> OperationStatistics:
    """
    Main ingestion function for Drishti-GS1 dataset.
    
    Returns:
        OperationStatistics with success/error counts
    """
    data_root = get_data_root() / "19_Drishti-GS1"
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
        description=(
            "Drishti-GS1 dataset for glaucoma screening with multi-expert annotations. "
            "Contains 101 color fundus images with 4 expert annotations per image, "
            "including soft maps for OD/cup segmentation, average boundaries, CDR values, "
            "and disc center localizations."
        ),
    )
    await upsert_dataset(dataset)
    
    # Step 2: Register experts
    logger.info("Registering experts...")
    expert_ids = await register_experts(dataset_id)
    
    # Step 3: Setup progress tracker
    # Estimate: ~101 images + Excel processing + annotations
    tracker = ProgressTracker(
        total=200,  # Rough estimate
        description=f"Ingesting {DATASET_NAME}"
    )
    
    # Step 4: Process Excel files
    logger.info("Processing diagnosis Excel file...")
    diagnosis_excel = data_root / "Drishti-GS1_diagnosis.xlsx"
    diagnosis_lookup = {}
    if diagnosis_excel.exists():
        diagnosis_lookup = await process_diagnosis_excel(
            diagnosis_excel, dataset_id, tracker
        )
    else:
        logger.warning(f"Diagnosis Excel file not found: {diagnosis_excel}")
    
    # Notching Excel file processing disabled - file format issues
    notching_lookup = {}
    
    # Step 5: Find all images (both Training and Test splits)
    logger.info("Finding images...")
    train_images_dir = data_root / "Drishti-GS1_files" / "Training" / "Images"
    test_images_dir = data_root / "Drishti-GS1_files" / "Test" / "Images"
    
    train_image_paths = []
    test_image_paths = []
    
    if train_images_dir.exists():
        train_image_paths = await asyncio.to_thread(find_images, train_images_dir, recursive=False)
        logger.info(f"Found {len(train_image_paths)} training images")
    else:
        logger.warning(f"Training images directory not found: {train_images_dir}")
    
    if test_images_dir.exists():
        test_image_paths = await asyncio.to_thread(find_images, test_images_dir, recursive=False)
        logger.info(f"Found {len(test_image_paths)} test images")
    else:
        logger.warning(f"Test images directory not found: {test_images_dir}")
    
    total_images = len(train_image_paths) + len(test_image_paths)
    logger.info(f"Total images: {total_images} (Train: {len(train_image_paths)}, Test: {len(test_image_paths)})")
    
    if total_images == 0:
        raise FileNotFoundError("No images found in Training or Test directories")
    
    # Update tracker total
    tracker.total = tracker.total + total_images
    
    # Step 6: Process all images
    logger.info("Processing images and annotations...")
    all_images: List[Image] = []
    all_segmentations: List[SegmentationAnnotation] = []
    all_classifications: List[ClassificationAnnotation] = []
    all_localizations: List[LocalizationAnnotation] = []
    all_expert_annotations: List[ExpertAnnotation] = []
    train_image_ids: List[UUID] = []
    test_image_ids: List[UUID] = []
    
    # Process training images
    train_gt_base_dir = data_root / "Drishti-GS1_files" / "Training" / "GT"
    logger.info(f"Processing {len(train_image_paths)} training images...")
    for idx, image_path in enumerate(train_image_paths):
        logger.info(f"  Processing training image {idx+1}/{len(train_image_paths)}: {image_path.name}")
        image_stem = image_path.stem
        gt_folder = train_gt_base_dir / image_stem
        
        if not gt_folder.exists():
            logger.warning(f"GT folder not found for image: {gt_folder}")
            tracker.record_error(
                error_type="gt_folder_not_found",
                error_message=f"GT folder not found: {gt_folder}",
                item_id=image_stem,
                item_path=str(image_path),
            )
            continue
        
        image, segmentations, classifications, localizations, expert_annotations, image_id = (
            await process_image_with_annotations(
                image_path=image_path,
                gt_folder=gt_folder,
                dataset_id=dataset_id,
                expert_ids=expert_ids,
                diagnosis_lookup=diagnosis_lookup,
                notching_lookup=notching_lookup,
                tracker=tracker,
            )
        )
        
        if image is not None:
            all_images.append(image)
            train_image_ids.append(image_id)
            all_segmentations.extend(segmentations)
            all_classifications.extend(classifications)
            all_localizations.extend(localizations)
            all_expert_annotations.extend(expert_annotations)
    
    # Process test images
    test_gt_base_dir = data_root / "Drishti-GS1_files" / "Test" / "Test_GT"
    logger.info(f"Processing {len(test_image_paths)} test images...")
    for idx, image_path in enumerate(test_image_paths):
        logger.info(f"  Processing test image {idx+1}/{len(test_image_paths)}: {image_path.name}")
        image_stem = image_path.stem
        gt_folder = test_gt_base_dir / image_stem
        
        if not gt_folder.exists():
            logger.warning(f"GT folder not found for image: {gt_folder}")
            tracker.record_error(
                error_type="gt_folder_not_found",
                error_message=f"GT folder not found: {gt_folder}",
                item_id=image_stem,
                item_path=str(image_path),
            )
            continue
        
        image, segmentations, classifications, localizations, expert_annotations, image_id = (
            await process_image_with_annotations(
                image_path=image_path,
                gt_folder=gt_folder,
                dataset_id=dataset_id,
                expert_ids=expert_ids,
                diagnosis_lookup=diagnosis_lookup,
                notching_lookup=notching_lookup,
                tracker=tracker,
            )
        )
        
        if image is not None:
            all_images.append(image)
            test_image_ids.append(image_id)
            all_segmentations.extend(segmentations)
            all_classifications.extend(classifications)
            all_localizations.extend(localizations)
            all_expert_annotations.extend(expert_annotations)
    
    logger.info(f"Finished processing all images. Summary:")
    logger.info(f"  Images: {len(all_images)}")
    logger.info(f"  Segmentations: {len(all_segmentations)}")
    logger.info(f"  Classifications: {len(all_classifications)}")
    logger.info(f"  Localizations: {len(all_localizations)}")
    logger.info(f"  Expert annotations: {len(all_expert_annotations)}")
    
    # Step 7: Bulk upsert images
    logger.info(f"Step 7: Upserting {len(all_images)} images...")
    if all_images:
        try:
            logger.info("  Starting bulk_upsert_images operation...")
            await bulk_upsert_images(all_images, batch_size=1000)
            logger.info(f"Successfully upserted {len(all_images)} images")
        except Exception as e:
            logger.error(f"Failed to bulk upsert images: {e}")
            raise
    
    # Step 8: Upsert segmentations (individual, no bulk operation available)
    logger.info(f"Step 8: Upserting {len(all_segmentations)} segmentation annotations...")
    for idx, segmentation in enumerate(all_segmentations):
        if idx % 10 == 0:
            logger.info(f"  Progress: {idx}/{len(all_segmentations)} segmentations upserted...")
        try:
            await upsert_segmentation_annotation(segmentation)
        except Exception as e:
            logger.error(f"Failed to upsert segmentation {segmentation.segmentation_id}: {e}")
            tracker.record_error(
                error_type="segmentation_upsert_error",
                error_message=str(e),
                item_id=str(segmentation.segmentation_id),
            )
    logger.info(f"Successfully upserted {len(all_segmentations)} segmentation annotations")
    
    # Step 9: Bulk upsert expert annotations (must be before classifications due to FK constraint)
    logger.info(f"Upserting {len(all_expert_annotations)} expert annotations...")
    if all_expert_annotations:
        try:
            await bulk_upsert_expert_annotations(all_expert_annotations, batch_size=1000)
            logger.info(f"Successfully upserted {len(all_expert_annotations)} expert annotations")
        except Exception as e:
            logger.error(f"Failed to bulk upsert expert annotations: {e}")
            raise
    
    # Step 10: Bulk upsert classifications
    logger.info(f"Upserting {len(all_classifications)} classification annotations...")
    if all_classifications:
        try:
            await bulk_upsert_classification_annotations(all_classifications, batch_size=1000)
            logger.info(f"Successfully upserted {len(all_classifications)} classification annotations")
        except Exception as e:
            logger.error(f"Failed to bulk upsert classifications: {e}")
            raise
    
    # Step 11: Bulk upsert localizations
    logger.info(f"Upserting {len(all_localizations)} localization annotations...")
    if all_localizations:
        try:
            await bulk_upsert_localization_annotations(all_localizations, batch_size=1000)
            logger.info(f"Successfully upserted {len(all_localizations)} localization annotations")
        except Exception as e:
            logger.error(f"Failed to bulk upsert localizations: {e}")
            raise
    
    # Step 12: Register splits and assign images
    # Note: Expert annotations for classifications are stored above (Step 9).
    # CDR values are registered as raw files for provenance but NOT stored as ExpertAnnotation records
    # since CDR can be derived from cup/disc segmentations (user requirement: CDRs not stored).
    logger.info("Registering dataset splits...")
    splits = await register_standard_splits(
        dataset_id=dataset_id,
        split_type="explicit",
        train_count=len(train_image_ids),
        test_count=len(test_image_ids) if test_image_ids else 0,
    )
    
    # Assign images to appropriate splits
    if train_image_ids:
        logger.info(f"Assigning {len(train_image_ids)} images to train split...")
        await bulk_assign_images_to_split(train_image_ids, splits["train"])
    
    if test_image_ids:
        logger.info(f"Assigning {len(test_image_ids)} images to test split...")
        await bulk_assign_images_to_split(test_image_ids, splits["test"])
    
    # Finish tracking
    tracker.finish()
    final_stats = tracker.get_statistics()
    
    # Final summary
    logger.info("=" * 80)
    logger.info("Ingestion Summary:")
    logger.info(f"  Total items: {final_stats.total_items}")
    logger.info(f"  Successful: {final_stats.successful_items}")
    logger.info(f"  Failed: {final_stats.failed_items}")
    logger.info(f"  Skipped: {final_stats.skipped_items}")
    logger.info(f"  Images registered: {len(all_images)}")
    logger.info(f"  Segmentation annotations: {len(all_segmentations)}")
    logger.info(f"  Classification annotations: {len(all_classifications)}")
    logger.info(f"  Localization annotations: {len(all_localizations)}")
    logger.info(f"  Expert annotations: {len(all_expert_annotations)}")
    if final_stats.errors:
        logger.warning(f"  Errors encountered: {len(final_stats.errors)}")
        for error in final_stats.errors[:10]:  # Show first 10 errors
            logger.warning(f"    - {error.error_type}: {error.error_message}")
    logger.info("=" * 80)
    
    return final_stats


async def main():
    """Entry point for script execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    try:
        stats = await ingest_drishti_gs1()
        
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
