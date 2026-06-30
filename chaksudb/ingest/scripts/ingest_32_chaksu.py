"""
Ingestion script for CHAKSU dataset.

Dataset: CHAKSU - Multi-expert Glaucoma Dataset
Structure: 
  - Train/ and Test/ splits
  - 1.0_Original_Fundus_Images/: Images organized by camera (Bosch, Forus, Remidio)
  - 3.0_Doctors_Annotations_Binary_OD_OC/: Binary masks per expert (Expert 1-5) organized by camera and Cup/Disc
  - 5.0_OD_OC_Mean_Median_Majority_STAPLE/: Consensus masks (Majority, Mean, Median, STAPLE) organized by camera and Cup/Disc
  - 6.0_Glaucoma_Decision/: CSV files with glaucoma classifications per expert and consensus methods
  - Note: 2.0_Doctors_Annotations is NOT processed (use 3.0_Doctors_Annotations_Binary_OD_OC instead)
Annotations: 
  - Multi-expert segmentation (OD/OC) with expert IDs from 3.0_Doctors_Annotations_Binary_OD_OC
  - Consensus segmentation (OD/OC) with different methods from 5.0_OD_OC_Mean_Median_Majority_STAPLE
  - Glaucoma classification (binary) per expert and consensus from 6.0_Glaucoma_Decision
Tasks: Multi-expert segmentation (OD/OC), Glaucoma classification, Expert registration
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from uuid import UUID

from chaksudb.common.progress import ProgressTracker, OperationStatistics
from chaksudb.config.config import get_data_root
from chaksudb.db.models import (
    ClassificationAnnotation,
    ConsensusAnnotation,
    Dataset,
    Expert,
    ExpertAnnotation,
    Image,
    SegmentationAnnotation,
)
from chaksudb.db.queries import (
    bulk_upsert_classification_annotations,
    bulk_upsert_expert_annotations,
    bulk_upsert_images,
    upsert_consensus_annotation,
    upsert_dataset,
    upsert_expert,
    upsert_segmentation_annotation,
)
from chaksudb.ingest.framework import (
    get_image_metadata_dict,
    process_csv,
    process_folder_tree,
    process_paired_files,
)
from chaksudb.ingest.framework.gen_uuid import (
    generate_dataset_uuid,
    generate_expert_annotation_uuid,
    generate_expert_uuid,
    generate_image_uuid,
)
from chaksudb.ingest.framework.provenance_context import get_current_provenance
from chaksudb.ingest.framework.split_assigner import (
    bulk_assign_images_to_split,
    register_standard_splits,
)
from chaksudb.ingest.framework.task_processors.classification_processor import (
    process_classification,
)
from chaksudb.ingest.framework.task_processors.segmentation_processor import (
    process_segmentation_from_binary_mask,
)

logger = logging.getLogger(__name__)

# Dataset metadata
DATASET_NAME = "CHAKSU"
DATASET_URL = "https://doi.org/10.6084/m9.figshare.20123135"  
DATASET_LICENSE = "CC-BY-4.0"  # Update with actual license if available

# Expert information
EXPERTS = {
    "1": {"name": "Expert 1", "expertise": "Glaucoma diagnosis and OD/OC segmentation"},
    "2": {"name": "Expert 2", "expertise": "Glaucoma diagnosis and OD/OC segmentation"},
    "3": {"name": "Expert 3", "expertise": "Glaucoma diagnosis and OD/OC segmentation"},
    "4": {"name": "Expert 4", "expertise": "Glaucoma diagnosis and OD/OC segmentation"},
    "5": {"name": "Expert 5", "expertise": "Glaucoma diagnosis and OD/OC segmentation"},
}

# Cameras
CAMERAS = ["Bosch", "Forus", "Remidio"]

# Consensus methods
CONSENSUS_METHODS = ["Majority", "Mean", "Median", "STAPLE"]

# Map consensus method names to schema values
CONSENSUS_METHOD_MAP = {
    "Majority": "majority_vote",
    "Mean": "mean",
    "Median": "median",
    "STAPLE": "staple",
}

# Standard splits
SPLITS = ["Train", "Test"]


async def register_experts(dataset_id: UUID) -> Dict[str, UUID]:
    """Register all 5 experts for CHAKSU dataset."""
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


def extract_image_id_from_path(image_path: Path, split_name: str) -> str:
    """
    Extract image identifier from path.
    
    Format: {split}/1.0_Original_Fundus_Images/{camera}/{filename}
    Returns: {camera}/{filename} as identifier
    """
    # Get relative path from split directory
    parts = image_path.parts
    try:
        # Find camera name (one of Bosch, Forus, Remidio)
        camera_idx = None
        for i, part in enumerate(parts):
            if part in CAMERAS:
                camera_idx = i
                break
        
        if camera_idx is None:
            # Fallback: use filename
            return image_path.name
        
        camera = parts[camera_idx]
        filename = image_path.name
        return f"{camera}/{filename}"
    except Exception:
        return image_path.name


async def process_images(
    data_root: Path,
    dataset_id: UUID,
    split_name: str,
    tracker: ProgressTracker,
) -> Tuple[List[Image], Dict[str, UUID], Dict[UUID, str]]:
    """
    Process images from a split.
    
    Returns:
        Tuple of (images list, image_id_map, image_to_split)
    """
    all_images: List[Image] = []
    image_id_map: Dict[str, UUID] = {}
    image_to_split: Dict[UUID, str] = {}
    
    images_dir = data_root / split_name / "1.0_Original_Fundus_Images"
    
    if not images_dir.exists():
        logger.warning(f"Images directory not found: {images_dir}")
        return all_images, image_id_map, image_to_split
    
    async def process_image_file(image_path: Path, rel_path: Path, depth: int) -> None:
        """Process a single image file."""
        try:
            # Extract image identifier
            image_id_str = extract_image_id_from_path(image_path, split_name)
            
            # Generate image ID
            if image_id_str not in image_id_map:
                image_id = generate_image_uuid(dataset_id, image_id_str)
                image_id_map[image_id_str] = image_id
            else:
                image_id = image_id_map[image_id_str]
            
            # Create image if not already created
            if image_id not in [img.image_id for img in all_images]:
                image = Image(
                    image_id=image_id,
                    dataset_id=dataset_id,
                    original_image_id=image_id_str,
                    **get_image_metadata_dict(image_path),
                    modality="fundus",
                )
                all_images.append(image)
                image_to_split[image_id] = split_name.lower()
            
            tracker.update(success=True)
            tracker.record_success("image")
            
        except Exception as e:
            tracker.update(success=False)
            tracker.record_error(
                error_type="processing",
                error_message=str(e),
                item_id=str(image_path),
            )
            logger.error(f"Failed to process image {image_path}: {e}")
    
    # Process all image files
    stats = await process_folder_tree(
        root_dir=images_dir,
        dataset_id=dataset_id,
        unified_annotation_type="segmentation",  # Images are used for segmentation
        process_file_fn=process_image_file,
        file_extensions={".jpg", ".JPG", ".png", ".PNG"},
        progress_tracker=tracker,
    )
    
    logger.info(f"Processed {stats.successful_items} images from {split_name}")
    
    return all_images, image_id_map, image_to_split


async def process_expert_segmentation(
    data_root: Path,
    dataset_id: UUID,
    split_name: str,
    expert_key: str,
    expert_id: UUID,
    image_id_map: Dict[str, UUID],
    tracker: ProgressTracker,
) -> Tuple[List[SegmentationAnnotation], List[ExpertAnnotation]]:
    """
    Process expert binary masks for OD and OC segmentation.
    
    Structure: {split}/3.0_Doctors_Annotations_Binary_OD_OC/Expert {N}/{camera}/{Cup|Disc}/{mask_file}
    """
    all_segmentations: List[SegmentationAnnotation] = []
    all_expert_annotations: List[ExpertAnnotation] = []
    
    expert_dir = data_root / split_name / "3.0_Doctors_Annotations_Binary_OD_OC" / f"Expert {expert_key}"
    
    if not expert_dir.exists():
        logger.warning(f"Expert directory not found: {expert_dir}")
        return all_segmentations, all_expert_annotations
    
    # Process each camera
    for camera in CAMERAS:
        camera_dir = expert_dir / camera
        if not camera_dir.exists():
            continue
        
        # Process Cup and Disc separately
        for annotation_type_name, folder_name in [("optic_cup", "Cup"), ("optic_disc", "Disc")]:
            mask_dir = camera_dir / folder_name
            if not mask_dir.exists():
                continue
            
            # Find corresponding images directory
            images_dir = data_root / split_name / "1.0_Original_Fundus_Images" / camera
            
            async def process_mask_pair(image_path: Optional[Path], mask_path: Optional[Path]) -> None:
                """Process an image-mask pair."""
                if mask_path is None:
                    return
                
                try:
                    # Extract image identifier from mask path
                    # Mask filename should match image filename (different extension)
                    mask_stem = mask_path.stem
                    
                    # Find matching image
                    image_id = None
                    for img_id_str, img_id in image_id_map.items():
                        # Check if this mask matches an image
                        if img_id_str.endswith(f"/{mask_stem}") or img_id_str.endswith(f"/{mask_stem}.jpg") or img_id_str.endswith(f"/{mask_stem}.JPG"):
                            image_id = img_id
                            break
                    
                    # Alternative: try to find image by matching stem in images directory
                    if image_id is None and images_dir.exists():
                        for ext in [".jpg", ".JPG", ".png", ".PNG"]:
                            candidate_image = images_dir / f"{mask_stem}{ext}"
                            if await asyncio.to_thread(candidate_image.exists):
                                image_id_str = f"{camera}/{candidate_image.name}"
                                if image_id_str in image_id_map:
                                    image_id = image_id_map[image_id_str]
                                    break
                    
                    if image_id is None:
                        tracker.record_error(
                            error_type="image_not_found",
                            error_message=f"Image not found for mask: {mask_path}",
                            item_id=str(mask_path),
                        )
                        tracker.update(success=False)
                        return
                    
                    # Get provenance
                    raw_data_id, provenance_chain_id = get_current_provenance()
                    
                    # Create ExpertAnnotation first
                    expert_annotation_id = generate_expert_annotation_uuid(
                        expert_id=expert_id,
                        annotation_task="segmentation",
                        raw_data_id=raw_data_id,
                        annotation_value_hash=None,
                    )
                    
                    expert_annotation = ExpertAnnotation(
                        expert_annotation_id=expert_annotation_id,
                        expert_id=expert_id,
                        annotation_task="segmentation",
                        raw_data_id=raw_data_id,
                        annotation_value={
                            "image_id": str(image_id),
                            "annotation_type": annotation_type_name,
                            "mask_file": str(mask_path),
                            "camera": camera,
                            "split": split_name,
                        },
                        confidence_level=None,
                        annotation_timestamp=None,
                    )
                    all_expert_annotations.append(expert_annotation)
                    
                    # Process segmentation with expert_annotation_id
                    segmentation = await process_segmentation_from_binary_mask(
                        mask_path=mask_path,
                        annotation_type=annotation_type_name,
                        image_id=image_id,
                        expert_annotation_id=expert_annotation_id,
                        raw_data_id=raw_data_id,
                        provenance_chain_id=provenance_chain_id,
                        annotation_method="manual",
                        dataset_name=DATASET_NAME,
                    )
                    all_segmentations.append(segmentation)
                    
                    tracker.update(success=True)
                    tracker.record_success("segmentation")
                    
                except Exception as e:
                    tracker.update(success=False)
                    tracker.record_error(
                        error_type="processing",
                        error_message=str(e),
                        item_id=str(mask_path) if mask_path else "unknown",
                    )
                    logger.error(f"Failed to process mask {mask_path}: {e}")
            
            # Use process_paired_files to match images with masks
            if images_dir.exists():
                stats = await process_paired_files(
                    primary_dir=images_dir,
                    secondary_dir=mask_dir,
                    process_pair_fn=process_mask_pair,
                    primary_extensions={".jpg", ".JPG", ".png", ".PNG"},
                    secondary_extensions={".tif", ".TIF", ".png", ".PNG"},
                    match_by="stem",
                    require_secondary=False,
                    progress_tracker=tracker,
                )
                logger.info(
                    f"Processed {stats.successful_items} {annotation_type_name} masks "
                    f"for Expert {expert_key} ({camera}) in {split_name}"
                )
            else:
                # Process masks directly if images directory not found
                async def process_mask_file(mask_path: Path, rel_path: Path, depth: int) -> None:
                    await process_mask_pair(None, mask_path)
                
                stats = await process_folder_tree(
                    root_dir=mask_dir,
                    dataset_id=dataset_id,
                    unified_annotation_type="segmentation",
                    process_file_fn=process_mask_file,
                    file_extensions={".tif", ".TIF", ".png", ".PNG"},
                    progress_tracker=tracker,
                )
    
    return all_segmentations, all_expert_annotations


async def process_consensus_segmentation(
    data_root: Path,
    dataset_id: UUID,
    split_name: str,
    consensus_method: str,
    expert_ids: Dict[str, UUID],
    image_id_map: Dict[str, UUID],
    tracker: ProgressTracker,
) -> Tuple[List[SegmentationAnnotation], List[ConsensusAnnotation]]:
    """
    Process consensus masks for OD and OC segmentation.
    
    Structure: {split}/5.0_OD_OC_Mean_Median_Majority_STAPLE/{camera}/{Cup|Disc}/{method}/{mask_file}
    """
    all_segmentations: List[SegmentationAnnotation] = []
    all_consensus_annotations: List[ConsensusAnnotation] = []
    # Track consensus IDs to avoid duplicates (same image + method = same consensus_id for Cup and Disc)
    seen_consensus_ids: Dict[UUID, ConsensusAnnotation] = {}
    
    consensus_dir = data_root / split_name / "5.0_OD_OC_Mean_Median_Majority_STAPLE"
    
    if not consensus_dir.exists():
        logger.warning(f"Consensus directory not found: {consensus_dir}")
        return all_segmentations, all_consensus_annotations
    
    # Process each camera
    for camera in CAMERAS:
        camera_dir = consensus_dir / camera
        if not camera_dir.exists():
            continue
        
        # Process Cup and Disc separately
        for annotation_type_name, folder_name in [("optic_cup", "Cup"), ("optic_disc", "Disc")]:
            type_dir = camera_dir / folder_name
            if not type_dir.exists():
                continue
            
            method_dir = type_dir / consensus_method
            if not method_dir.exists():
                continue
            
            # Find corresponding images directory
            images_dir = data_root / split_name / "1.0_Original_Fundus_Images" / camera
            
            async def process_mask_pair(image_path: Optional[Path], mask_path: Optional[Path]) -> None:
                """Process an image-mask pair."""
                if mask_path is None:
                    return
                
                try:
                    # Extract image identifier from mask path
                    mask_stem = mask_path.stem
                    
                    # Find matching image
                    image_id = None
                    for img_id_str, img_id in image_id_map.items():
                        if img_id_str.endswith(f"/{mask_stem}") or img_id_str.endswith(f"/{mask_stem}.jpg") or img_id_str.endswith(f"/{mask_stem}.JPG"):
                            image_id = img_id
                            break
                    
                    if image_id is None and images_dir.exists():
                        for ext in [".jpg", ".JPG", ".png", ".PNG"]:
                            candidate_image = images_dir / f"{mask_stem}{ext}"
                            if await asyncio.to_thread(candidate_image.exists):
                                image_id_str = f"{camera}/{candidate_image.name}"
                                if image_id_str in image_id_map:
                                    image_id = image_id_map[image_id_str]
                                    break
                    
                    if image_id is None:
                        tracker.record_error(
                            error_type="image_not_found",
                            error_message=f"Image not found for consensus mask: {mask_path}",
                            item_id=str(mask_path),
                        )
                        tracker.update(success=False)
                        return
                    
                    # Get provenance
                    raw_data_id, provenance_chain_id = get_current_provenance()
                    
                    # Map consensus method name to schema value
                    schema_method = CONSENSUS_METHOD_MAP.get(consensus_method, consensus_method.lower())
                    if schema_method not in ["majority_vote", "mean", "median", "staple"]:
                        # Fallback to majority_vote if unknown
                        schema_method = "majority_vote"
                    
                    # Generate consensus ID
                    # Note: We don't have the exact expert annotation IDs that contributed to the consensus
                    # since the masks are pre-computed. We'll use an empty list for expert_annotation_ids.
                    from chaksudb.ingest.framework.gen_uuid import generate_consensus_uuid
                    # Use empty list since we don't know which specific expert annotations were used
                    consensus_id = generate_consensus_uuid(
                        image_id=image_id,
                        annotation_task="segmentation",
                        consensus_method=schema_method,
                        expert_annotation_ids=[],  # Empty since masks are pre-computed
                    )
                    
                    # Create ConsensusAnnotation record only if we haven't seen this consensus_id before
                    # (Cup and Disc masks for same image+method will have same consensus_id)
                    if consensus_id not in seen_consensus_ids:
                        consensus_annotation = ConsensusAnnotation(
                            consensus_id=consensus_id,
                            image_id=image_id,
                            annotation_task="segmentation",
                            consensus_method=schema_method,
                            expert_annotation_ids=[],  # Empty since we don't have the mapping
                            consensus_value={
                                "annotation_types": [annotation_type_name],  # List to support multiple types
                                "method": consensus_method,
                                "mask_files": [str(mask_path)],
                            },
                            agreement_score=None,
                            disagreement_details=None,
                            adjudicator_id=None,
                        )
                        seen_consensus_ids[consensus_id] = consensus_annotation
                        all_consensus_annotations.append(consensus_annotation)
                    else:
                        # Update existing consensus to include this annotation type
                        existing = seen_consensus_ids[consensus_id]
                        if existing.consensus_value:
                            # Add annotation type if not already present
                            ann_types = existing.consensus_value.get("annotation_types", [])
                            if annotation_type_name not in ann_types:
                                ann_types.append(annotation_type_name)
                                existing.consensus_value["annotation_types"] = ann_types
                            # Add mask file if not already present
                            mask_files = existing.consensus_value.get("mask_files", [])
                            mask_file_str = str(mask_path)
                            if mask_file_str not in mask_files:
                                mask_files.append(mask_file_str)
                                existing.consensus_value["mask_files"] = mask_files
                    
                    # Process segmentation with consensus ID
                    # Use "manual" since consensus is computed from multiple manual annotations
                    # Pass dataset_id explicitly so raw_annotation_files insert uses same FK as upserted dataset
                    segmentation = await process_segmentation_from_binary_mask(
                        mask_path=mask_path,
                        annotation_type=annotation_type_name,
                        image_id=image_id,
                        consensus_id=consensus_id,
                        raw_data_id=raw_data_id,
                        provenance_chain_id=provenance_chain_id,
                        annotation_method="manual",  # Manual annotation linked to consensus
                        dataset_id=dataset_id,
                        dataset_name=DATASET_NAME,
                        merge_nonzero=True
                    )
                    all_segmentations.append(segmentation)
                    
                    tracker.update(success=True)
                    tracker.record_success("segmentation")
                    
                except Exception as e:
                    tracker.update(success=False)
                    tracker.record_error(
                        error_type="processing",
                        error_message=str(e),
                        item_id=str(mask_path) if mask_path else "unknown",
                    )
                    logger.error(f"Failed to process consensus mask {mask_path}: {e}")
            
            # Use process_paired_files to match images with masks
            if images_dir.exists():
                stats = await process_paired_files(
                    primary_dir=images_dir,
                    secondary_dir=method_dir,
                    process_pair_fn=process_mask_pair,
                    primary_extensions={".jpg", ".JPG", ".png", ".PNG"},
                    secondary_extensions={".tif", ".TIF", ".png", ".PNG"},
                    match_by="stem",
                    require_secondary=False,
                    progress_tracker=tracker,
                )
                logger.info(
                    f"Processed {stats.successful_items} {annotation_type_name} consensus masks "
                    f"({consensus_method}) for {camera} in {split_name}"
                )
            else:
                # Process masks directly
                async def process_mask_file(mask_path: Path, rel_path: Path, depth: int) -> None:
                    await process_mask_pair(None, mask_path)
                
                stats = await process_folder_tree(
                    root_dir=method_dir,
                    dataset_id=dataset_id,
                    unified_annotation_type="segmentation",
                    process_file_fn=process_mask_file,
                    file_extensions={".tif", ".TIF", ".png", ".PNG"},
                    progress_tracker=tracker,
                )
    
    return all_segmentations, all_consensus_annotations


async def process_glaucoma_classifications(
    data_root: Path,
    dataset_id: UUID,
    split_name: str,
    expert_ids: Dict[str, UUID],
    image_id_map: Dict[str, UUID],
    tracker: ProgressTracker,
) -> Tuple[List[ClassificationAnnotation], List[ExpertAnnotation], List[ConsensusAnnotation]]:
    """
    Process glaucoma classifications from CSV files.
    
    Structure: 
    - Per expert: {split}/6.0_Glaucoma_Decision/Expert {N}/{camera}.csv
    - Consensus: {split}/6.0_Glaucoma_Decision/{method}/{camera}.csv
    """
    all_classifications: List[ClassificationAnnotation] = []
    all_expert_annotations: List[ExpertAnnotation] = []
    all_consensus_annotations: List[ConsensusAnnotation] = []
    
    decision_dir = data_root / split_name / "6.0_Glaucoma_Decision"
    
    if not decision_dir.exists():
        logger.warning(f"Glaucoma decision directory not found: {decision_dir}")
        return all_classifications, all_expert_annotations, all_consensus_annotations
    
    # Store expert_ids for use in nested functions
    _expert_ids = expert_ids
    
    # Process expert classifications
    for expert_key, expert_id in expert_ids.items():
        expert_decision_dir = decision_dir / f"Expert {expert_key}"
        if not expert_decision_dir.exists():
            continue
        
        for camera in CAMERAS:
            csv_path = expert_decision_dir / f"{camera}.csv"
            if not csv_path.exists():
                continue
            
            async def process_expert_row(row: Dict, idx: int) -> None:
                """Process a row from expert CSV file."""
                try:
                    # CSV format: Images,Expert 1,Expert 2,...
                    # Image format: "Image107.jpg-Image107-1.jpg" or similar
                    image_key = row.get("Images", "").strip()
                    if not image_key:
                        return
                    
                    # Extract image filename (before first dash if present)
                    image_filename = image_key.split("-")[0] if "-" in image_key else image_key
                    # Handle different extensions
                    for ext in [".jpg", ".JPG", ".tif", ".TIF", ".png", ".PNG"]:
                        if image_filename.endswith(ext):
                            image_filename = image_filename[:-len(ext)]
                            break
                    
                    # Find matching image ID
                    image_id = None
                    for img_id_str, img_id in image_id_map.items():
                        # Try matching with various extensions
                        for ext in ["", ".jpg", ".JPG", ".tif", ".TIF", ".png", ".PNG"]:
                            if img_id_str.endswith(f"/{image_filename}{ext}"):
                                image_id = img_id
                                break
                        if image_id:
                            break
                    
                    if image_id is None:
                        # Try to find by camera and filename
                        for ext in [".jpg", ".JPG", ".tif", ".TIF", ".png", ".PNG"]:
                            image_id_str = f"{camera}/{image_filename}{ext}"
                            if image_id_str in image_id_map:
                                image_id = image_id_map[image_id_str]
                                break
                    
                    if image_id is None:
                        tracker.record_error(
                            error_type="image_not_found",
                            error_message=f"Image not found for classification: {image_key}",
                            item_id=image_key,
                        )
                        tracker.update(success=False)
                        return
                    
                    # Get expert decision from "Glaucoma Decision" column
                    decision_str = row.get("Glaucoma Decision", "").strip().upper()
                    if not decision_str:
                        return
                    
                    # Map decision to boolean (GLAUCOMA SUSPECT or GLAUCOMA = True, NORMAL = False)
                    is_glaucoma = "GLAUCOMA" in decision_str
                    
                    # Get provenance
                    raw_data_id, provenance_chain_id = get_current_provenance()
                    
                    # Create ExpertAnnotation first
                    expert_annotation_id = generate_expert_annotation_uuid(
                        expert_id=expert_id,
                        annotation_task="classification",
                        raw_data_id=raw_data_id,
                        annotation_value_hash=None,
                    )
                    
                    expert_annotation = ExpertAnnotation(
                        expert_annotation_id=expert_annotation_id,
                        expert_id=expert_id,
                        annotation_task="classification",
                        raw_data_id=raw_data_id,
                        annotation_value={
                            "image_id": str(image_id),
                            "image_key": image_key,
                            "glaucoma_decision": decision_str,
                            "is_glaucoma": is_glaucoma,
                            "camera": camera,
                            "split": split_name,
                        },
                        confidence_level=None,
                        annotation_timestamp=None,
                    )
                    all_expert_annotations.append(expert_annotation)
                    
                    # Process classification with expert_annotation_id
                    classifications = await process_classification(
                        class_value=is_glaucoma,
                        task_type="binary",
                        class_name="glaucoma",
                        image_id=image_id,
                        expert_annotation_id=expert_annotation_id,
                        raw_data_id=raw_data_id,
                        provenance_chain_id=provenance_chain_id,
                        annotation_method="manual",
                    )
                    all_classifications.extend(classifications)
                    
                    tracker.update(success=True)
                    tracker.record_success("classification")
                    
                except Exception as e:
                    tracker.update(success=False)
                    tracker.record_error(
                        error_type="processing",
                        error_message=str(e),
                        item_id=row.get("Images", "unknown"),
                    )
                    logger.error(f"Failed to process expert classification row: {e}")
            
            stats, raw_file_id, chain_id = await process_csv(
                csv_path,
                dataset_id,
                "classification",
                process_expert_row,
                progress_tracker=tracker,
            )
            logger.info(
                f"Processed {stats.successful_items} expert classifications "
                f"from Expert {expert_key} ({camera}) in {split_name}"
            )
    
    # Process consensus classifications from comparison CSVs
    # The CSVs in Majority/Mean/Median folders only have measurements, not decisions
    # The actual consensus decisions are in comparison CSVs in the root directory
    
    # Process comparison CSVs for Majority consensus
    # Format: Glaucoma_Decision_Comparison_{camera}_majority.csv or Glaucoma_Decision_Majority_{camera}.csv
    for camera in CAMERAS:
        # Try both naming patterns
        comparison_csvs = [
            decision_dir / f"Glaucoma_Decision_Comparison_{camera}_majority.csv",
            decision_dir / f"Glaucoma_Decision_Majority_{camera}.csv",
        ]
        
        for csv_path in comparison_csvs:
            if not csv_path.exists():
                continue
            
            async def process_majority_consensus_row(row: Dict, idx: int) -> None:
                """Process a row from majority consensus comparison CSV file."""
                try:
                    image_key = row.get("Images", "").strip()
                    if not image_key:
                        return
                    
                    image_filename = image_key.split("-")[0] if "-" in image_key else image_key
                    # Handle different extensions
                    for ext in [".jpg", ".JPG", ".tif", ".TIF", ".png", ".PNG"]:
                        if image_filename.endswith(ext):
                            image_filename = image_filename[:-len(ext)]
                            break
                    
                    # Find matching image ID
                    image_id = None
                    for img_id_str, img_id in image_id_map.items():
                        # Try matching with various extensions
                        for ext in ["", ".jpg", ".JPG", ".tif", ".TIF", ".png", ".PNG"]:
                            if img_id_str.endswith(f"/{image_filename}{ext}"):
                                image_id = img_id
                                break
                        if image_id:
                            break
                    
                    if image_id is None:
                        # Try with camera prefix
                        image_id_str = f"{camera}/{image_filename}"
                        for ext in [".jpg", ".JPG", ".tif", ".TIF", ".png", ".PNG"]:
                            candidate = f"{image_id_str}{ext}"
                            if candidate in image_id_map:
                                image_id = image_id_map[candidate]
                                break
                    
                    if image_id is None:
                        tracker.record_error(
                            error_type="image_not_found",
                            error_message=f"Image not found for consensus classification: {image_key}",
                            item_id=image_key,
                        )
                        tracker.update(success=False)
                        return
                    
                    # Get consensus decision from "Glaucoma Decision" column
                    decision_str = row.get("Glaucoma Decision", "").strip().upper()
                    if not decision_str:
                        # Skip if no decision column
                        return
                    
                    is_glaucoma = "GLAUCOMA" in decision_str
                    
                    # Get provenance
                    raw_data_id, provenance_chain_id = get_current_provenance()
                    
                    # Map to majority_vote
                    schema_method = "majority_vote"
                    
                    # Generate consensus ID for classification
                    from chaksudb.ingest.framework.gen_uuid import generate_consensus_uuid
                    consensus_id = generate_consensus_uuid(
                        image_id=image_id,
                        annotation_task="classification",
                        consensus_method=schema_method,
                        expert_annotation_ids=[],  # Empty since we don't have the mapping
                    )
                    
                    # Create ConsensusAnnotation record
                    consensus_annotation = ConsensusAnnotation(
                        consensus_id=consensus_id,
                        image_id=image_id,
                        annotation_task="classification",
                        consensus_method=schema_method,
                        expert_annotation_ids=[],  # Empty since we don't have the mapping
                        consensus_value={
                            "class_name": "glaucoma",
                            "class_value": is_glaucoma,
                            "method": "Majority",
                        },
                        agreement_score=None,
                        disagreement_details=None,
                        adjudicator_id=None,
                    )
                    all_consensus_annotations.append(consensus_annotation)
                    
                    # Process classification
                    classifications = await process_classification(
                        class_value=is_glaucoma,
                        task_type="binary",
                        class_name="glaucoma",
                        image_id=image_id,
                        consensus_id=consensus_id,
                        raw_data_id=raw_data_id,
                        provenance_chain_id=provenance_chain_id,
                        annotation_method="manual",
                    )
                    all_classifications.extend(classifications)
                    
                    tracker.update(success=True)
                    tracker.record_success("classification")
                    
                except Exception as e:
                    tracker.update(success=False)
                    tracker.record_error(
                        error_type="processing",
                        error_message=str(e),
                        item_id=row.get("Images", "unknown"),
                    )
                    logger.error(f"Failed to process majority consensus classification row: {e}")
            
            stats, raw_file_id, chain_id = await process_csv(
                csv_path,
                dataset_id,
                "classification",
                process_majority_consensus_row,
                progress_tracker=tracker,
            )
            logger.info(
                f"Processed {stats.successful_items} majority consensus classifications "
                f"from {csv_path.name} in {split_name}"
            )
    
    # Note: Mean, Median, and STAPLE consensus decisions may not be available in CSV format
    # They are only available as segmentation masks in 5.0_OD_OC_Mean_Median_Majority_STAPLE
    
    return all_classifications, all_expert_annotations, all_consensus_annotations


async def ingest_chaksu() -> OperationStatistics:
    """
    Main ingestion function for CHAKSU dataset.
    
    Returns:
        OperationStatistics with success/error counts
    """
    data_root = get_data_root() / "32_CHAKSU"
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
    
    # Step 3: Count total items for progress tracking
    # Count actual items that will be processed (each tracker.update() call = 1 item)
    # Note: We count all files, but some masks may not match images and won't be processed.
    # The actual processed count may be lower than the total.
    logger.info("Counting items to process...")
    total_count = 0
    
    for split_name in SPLITS:
        split_dir = data_root / split_name
        if not split_dir.exists():
            continue
        
        # Count images (1 item per image - all will be processed)
        images_dir = split_dir / "1.0_Original_Fundus_Images"
        image_count = 0
        if images_dir.exists():
            for ext in [".jpg", ".JPG", ".png", ".PNG"]:
                image_count += len(list(images_dir.rglob(f"*{ext}")))
        total_count += image_count
        
        # Count expert segmentation masks (1 item per mask file that matches an image)
        # Structure: {split}/3.0_Doctors_Annotations_Binary_OD_OC/Expert {N}/{camera}/{Cup|Disc}/{mask_file}
        # Note: Only masks that match images will be processed, but we count all for estimate
        masks_dir = split_dir / "3.0_Doctors_Annotations_Binary_OD_OC"
        if masks_dir.exists():
            for ext in [".tif", ".TIF", ".png", ".PNG"]:
                total_count += len(list(masks_dir.rglob(f"*{ext}")))
        
        # Count consensus segmentation masks (1 item per mask file that matches an image)
        # Structure: {split}/5.0_OD_OC_Mean_Median_Majority_STAPLE/{camera}/{Cup|Disc}/{method}/{mask_file}
        # Note: Only masks that match images will be processed, but we count all for estimate
        consensus_dir = split_dir / "5.0_OD_OC_Mean_Median_Majority_STAPLE"
        if consensus_dir.exists():
            for ext in [".tif", ".TIF", ".png", ".PNG"]:
                total_count += len(list(consensus_dir.rglob(f"*{ext}")))
        
        # Count classification annotations (1 item per CSV row - all will be processed)
        # Structure: {split}/6.0_Glaucoma_Decision/{expert_or_consensus}/{csv_file}
        decision_dir = split_dir / "6.0_Glaucoma_Decision"
        if decision_dir.exists():
            for csv_file in decision_dir.rglob("*.csv"):
                import csv as csv_module
                try:
                    with open(csv_file, "r", encoding="utf-8") as f:
                        reader = csv_module.reader(f)
                        next(reader)  # Skip header
                        total_count += sum(1 for _ in reader)
                except Exception:
                    pass
    
    logger.info(f"Estimated total items to process: {total_count} (actual may be lower due to unmatched masks)")
    
    # Step 4: Setup progress tracker
    tracker = ProgressTracker(
        total=total_count,
        description=f"Ingesting {DATASET_NAME}"
    )
    
    # Step 5: Collect items for bulk upsert
    all_images: List[Image] = []
    all_segmentations: List[SegmentationAnnotation] = []
    all_expert_annotations: List[ExpertAnnotation] = []
    all_classifications: List[ClassificationAnnotation] = []
    all_consensus_annotations: List[ConsensusAnnotation] = []
    all_image_to_split: Dict[UUID, str] = {}
    
    # Step 6: Process each split
    for split_name in SPLITS:
        split_dir = data_root / split_name
        if not split_dir.exists():
            logger.warning(f"Split directory not found: {split_dir}")
            continue
        
        logger.info(f"Processing {split_name} split...")
        
        # Process images
        images, image_id_map, image_to_split = await process_images(
            data_root, dataset_id, split_name, tracker
        )
        all_images.extend(images)
        all_image_to_split.update(image_to_split)
        
        # Process expert segmentations
        for expert_key, expert_id in expert_ids.items():
            segmentations, expert_anns = await process_expert_segmentation(
                data_root, dataset_id, split_name, expert_key, expert_id,
                image_id_map, tracker
            )
            all_segmentations.extend(segmentations)
            all_expert_annotations.extend(expert_anns)
        
        # Process consensus segmentations
        for method in CONSENSUS_METHODS:
            segmentations, consensus_seg_anns = await process_consensus_segmentation(
                data_root, dataset_id, split_name, method, expert_ids, image_id_map, tracker
            )
            all_segmentations.extend(segmentations)
            all_consensus_annotations.extend(consensus_seg_anns)
        
        # Process glaucoma classifications
        classifications, classification_expert_anns, consensus_cls_anns = await process_glaucoma_classifications(
            data_root, dataset_id, split_name, expert_ids, image_id_map, tracker
        )
        all_classifications.extend(classifications)
        all_expert_annotations.extend(classification_expert_anns)
        all_consensus_annotations.extend(consensus_cls_anns)
    
    # Step 7: Bulk upsert
    logger.info(f"Upserting {len(all_images)} images...")
    await bulk_upsert_images(all_images, batch_size=1000)
    
    logger.info(f"Upserting {len(all_expert_annotations)} expert annotations...")
    await bulk_upsert_expert_annotations(all_expert_annotations, batch_size=1000)
    
    logger.info(f"Upserting {len(all_consensus_annotations)} consensus annotations...")
    # Deduplicate consensus annotations by consensus_id (keep last one if duplicates)
    # This can happen when same image has multiple annotation types (e.g., Cup and Disc)
    consensus_dict: Dict[UUID, ConsensusAnnotation] = {}
    for consensus in all_consensus_annotations:
        consensus_dict[consensus.consensus_id] = consensus
    
    deduplicated_consensus = list(consensus_dict.values())
    if len(deduplicated_consensus) < len(all_consensus_annotations):
        logger.info(
            f"Deduplicated {len(all_consensus_annotations)} consensus annotations "
            f"to {len(deduplicated_consensus)} unique records"
        )
    
    for consensus in deduplicated_consensus:
        try:
            await upsert_consensus_annotation(consensus)
        except Exception as e:
            logger.error(f"Failed to upsert consensus {consensus.consensus_id}: {e}")
    
    logger.info(f"Upserting {len(all_segmentations)} segmentations...")
    # Upsert segmentations individually (no bulk operation available)
    for segmentation in all_segmentations:
        try:
            await upsert_segmentation_annotation(segmentation)
        except Exception as e:
            logger.error(f"Failed to upsert segmentation {segmentation.segmentation_id}: {e}")
    
    logger.info(f"Upserting {len(all_classifications)} classifications...")
    await bulk_upsert_classification_annotations(all_classifications, batch_size=1000)
    
    # Step 8: Register splits and assign images
    logger.info("Registering splits and assigning images...")
    splits = await register_standard_splits(
        dataset_id=dataset_id,
        split_type="explicit",
        train_count=sum(1 for split in all_image_to_split.values() if split == "train"),
        test_count=sum(1 for split in all_image_to_split.values() if split == "test"),
        val_count=0,
    )
    
    # Assign images to splits
    # splits is a dict: {"train": train_split_id, "val": val_split_id, "test": test_split_id}
    for image_id, split_name in all_image_to_split.items():
        split_id = splits.get(split_name)
        if split_id:
            await bulk_assign_images_to_split(
                image_ids=[image_id],
                split_id=split_id,
                task_type=None,
            )
    
    # Update total to match actual processed count for accurate statistics
    # This fixes the issue where the pre-calculated total (based on file counts)
    # doesn't match the actual items processed (some masks may not match images)
    if tracker.current > 0:
        tracker.total = tracker.current
        tracker.stats.total_items = tracker.current
    
    tracker.finish()
    return tracker.get_statistics()


async def main():
    import sys
    from pathlib import Path
    log_file = Path("./logs/ingest_32_chaksu.log")
    log_file.touch(exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
        logging.FileHandler(log_file, mode='w'), 
        logging.StreamHandler(sys.stdout),          
        ],
    )
    stats = await ingest_chaksu()
    
    logger.info("=" * 80)
    logger.info("Ingestion Summary:")
    logger.info(f"  Successful: {stats.successful_items}")
    logger.info(f"  Failed: {stats.failed_items}")
    logger.info(f"  Skipped: {stats.skipped_items}")
    logger.info("=" * 80)
    
    return 0 if stats.failed_items == 0 else 1


if __name__ == "__main__":
    exit(asyncio.run(main()))
