"""
Ingestion script for STARE (STructured Analysis of the Retina).

Dataset: STARE - Multi-expert vessel segmentation with clinical diagnosis
Structure: 
- Images: 
  * stare-images/ (20 images with vessel segmentation)
  * documents/ (~400 raw images without annotations)
- Vessel masks: labels-ah/ (Expert Adam Hoover), labels-vk/ (Expert Valentina Kouznetsova)
- Diagnosis: diagnosis.txt (diagnosis codes and text descriptions)
- Manifestations: annotations/ (39 feature annotations encoded as digit strings)

Annotations:
- Vessel segmentation (2 experts: AH and VK) - 40 annotations for 20 images in stare-images/
- Clinical descriptions (diagnosis text) - ~403 diagnosis texts from diagnosis.txt
- Manifestation annotations (39 features) - processed as keyword annotations from .fea.mg.txt files


"""

import asyncio
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from uuid import UUID

from chaksudb.common.progress import ProgressTracker, OperationStatistics
from chaksudb.config.config import get_data_root
from chaksudb.db.models import (
    Dataset,
    Image,
    SegmentationAnnotation,
    Expert,
    ExpertAnnotation,
    ClinicalDescription,
    KeywordAnnotation,
)
from chaksudb.db.queries import (
    upsert_dataset,
    bulk_upsert_images,
    upsert_segmentation_annotation,
    upsert_expert,
    upsert_expert_annotation,
    upsert_clinical_description,
    upsert_keyword_annotation,
)
from chaksudb.ingest.framework import (
    find_images,
    find_matching_file,
    get_image_metadata_dict,
)
from chaksudb.ingest.framework.gen_uuid import (
    generate_dataset_uuid,
    generate_image_uuid,
    generate_expert_uuid,
    generate_expert_annotation_uuid,
    generate_description_uuid,
)
from chaksudb.ingest.framework.task_processors.segmentation_processor import (
    process_segmentation_from_binary_mask,
)
from chaksudb.ingest.framework.task_processors.keyword_processor import (
    process_keyword_annotation,
)
from chaksudb.ingest.framework.raw_file_helpers import (
    register_individual_file,
    register_individual_file as register_text_file,
)
from chaksudb.ingest.framework.provenance_context import get_current_provenance
from chaksudb.ingest.framework.split_assigner import auto_stratified_splits

logger = logging.getLogger(__name__)

# Dataset metadata
DATASET_NAME = "STARE"
DATASET_URL = "https://cecas.clemson.edu/~ahoover/stare/"
DATASET_LICENSE = "Public Domain"  # Research/academic use

# Expert metadata
EXPERTS = {
    "ah": {
        "name": "Adam Hoover",
        "affiliation": "Clemson University",
        "expertise": "Vessel Segmentation",
    },
    "vk": {
        "name": "Valentina Kouznetsova",
        "affiliation": "UC San Diego",
        "expertise": "Vessel Segmentation",
    },
}

# Manifestation mappings from official STARE documentation
# Structure: {manifestation_number: {"name": str, "states": {state_value: description}}}
# Position in .fea.mg.txt string (0-based) = Manifestation Number - 1
# Only pathological states are included (excludes Absent, Unknown, Normal)
MANIFESTATION_STATES = {
    1: {"name": "RPED Manifestation", "states": {2: "present"}},
    2: {"name": "CME", "states": {2: "Visible"}},
    3: {"name": "ERM", "states": {2: "Present"}},
    4: {"name": "Subretinal Fibrosis", "states": {2: "Present"}},
    5: {"name": "CNV Manifestation", "states": {2: "Observable"}},
    6: {"name": "Drusen", "states": {
        2: "Fine, few",
        3: "Fine, many",
        4: "Large, soft, few",
        5: "Large, soft, many"
    }},
    7: {"name": "Preretinal Hemmorhage", "states": {2: "Present anywhere"}},
    8: {"name": "Subretinal Hemmorhage", "states": {2: "Present anywhere"}},
    9: {"name": "Microaneurism or Dot Hemmorhage", "states": {
        2: "Few anywhere",
        3: "Many anywhere"
    }},
    10: {"name": "VH", "states": {2: "Present anywhere"}},
    11: {"name": "Small or Medium blot Hemmorhage", "states": {
        2: "Low density, not regional",
        3: "High density, not regional"
    }},
    12: {"name": "Retinal Hemmorhage", "states": {
        2: "Low density, not regional",
        3: "High density, not regional",
        4: "Low density, regional not crossing horiz. meridian"
    }},
    13: {"name": "Retinal or Subretinal Exudate", "states": {
        2: "Low severity, no circinate",
        3: "High severity, no circinate"
    }},
    14: {"name": "Circinate Pattern", "states": {
        2: "Low severity, no circinate",
        3: "High severity, no circinate"
    }},
    15: {"name": "Macula Data", "states": {
        2: "Present incomplete",
        3: "Present three hundred and sixty degrees"
    }},
    18: {"name": "ON Collateral", "states": {2: "Present"}},
    19: {"name": "ON Swelling", "states": {
        2: "Low severity",
        3: "High severity"
    }},
    20: {"name": "ON Hemmorhage", "states": {
        2: "Splinter",
        3: "Blob"
    }},
    21: {"name": "ON Color", "states": {
        2: "Sector palor",
        4: "sector erythema",
        5: "rosy or red, whole nerve"
        # Note: State 3 "Normal" excluded, State 1 "pale/white" is pathological but listed as 1
    }},
    22: {"name": "NVD", "states": {
        2: "Less than one disk area",
        3: "Greater than one disk area"
    }},
    23: {"name": "Artery Color", "states": {
        3: "copper wire",
        4: "silver wire"
        # Note: State 2 "normal" excluded, State 1 "dark (deoxygenated)" is pathological but listed as 1
    }},
    24: {"name": "Artery Sheath", "states": {2: "Present"}},
    25: {"name": "Vein Color", "states": {
        2: "Gray or white (ghost vessel)",
        3: "Yellow (sheathed)"
    }},
    26: {"name": "Artery Diameter", "states": {
        1: "extreme, global",
        2: "moderate, global",
        3: "Moderate, branch or single",
        4: "Focal, one or more segments",
        6: "Tortuosity, branch or single",
        7: "Tortuosity, global"
        # Note: State 5 "Normal" excluded
    }},
    27: {"name": "Vein Diameter", "states": {
        1: "narrowing, Entire venous tree",
        2: "narrowing, Tributary vein or single",
        4: "Tortuosity, tributary vein or single",
        5: "Tortuosity, global",
        6: "Tortuosity"
        # Note: State 3 "normal" excluded
    }},
    28: {"name": "Teleanglectasis", "states": {
        2: "Present, not crossing horizontal median",
        3: "Present, crossing horizantal meridian"
    }},
    29: {"name": "BV Specular Reflex", "states": {2: "Wide and bright"}},
    30: {"name": "Macroaneurism", "states": {
        2: "Single",
        3: "Multiple"
    }},
    31: {"name": "A-V Change", "states": {2: "One or more examples"}},
    32: {"name": "TXRD Schisis", "states": {2: "Present"}},
    33: {"name": "Cotton-Wool Spot", "states": {
        2: "Few",
        3: "Many"
    }},
    34: {"name": "Inner Retinal Infarct", "states": {
        2: "Not involving the macula",
        3: "Involving the macula (cherry red spot)"
    }},
    35: {"name": "Cherry Red Spot", "states": {2: "Present"}},
    36: {"name": "Ghost BV", "states": {2: "Present"}},
    37: {"name": "NVE", "states": {
        2: "Few or small",
        3: "Many or large"
    }},
    39: {"name": "Photocoagular Scar", "states": {
        2: "Fine (grid)",
        3: "Large, in or near macula",
        4: "Around five hundred μm, round, arcade or peripheral, many"
    }},
    41: {"name": "Emboli Manifestation", "states": {
        2: "One",
        3: "More than one"
    }},
}


def parse_diagnosis_file(diagnosis_path: Path) -> Dict[str, Tuple[str, str]]:
    """
    Parse diagnosis.txt file.
    
    Returns:
        Dictionary mapping image_id -> (diagnosis_codes, diagnosis_text)
    """
    diagnosis_map = {}
    
    with open(diagnosis_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Format: im0001\t7\t\tBackground Diabetic Retinopathy\t\t
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            
            image_id = parts[0].strip()
            diagnosis_codes = parts[1].strip() if len(parts) > 1 else ""
            # Diagnosis text is in the remaining fields (may have multiple tabs)
            diagnosis_text = " ".join([p.strip() for p in parts[2:] if p.strip()])
            
            diagnosis_map[image_id] = (diagnosis_codes, diagnosis_text)
    
    logger.info(f"Parsed {len(diagnosis_map)} diagnosis entries")
    return diagnosis_map


def parse_manifestation_file(manifestation_path: Path) -> Dict[int, int]:
    """
    Parse manifestation annotation file (.fea.mg.txt).
    
    Format: Single line with 42+ characters, each digit represents state of a manifestation.
    - Position in string (0-41) maps to Manifestation Number (1-42) via: manifestation_num = position + 1
    - Digit value represents state (varies by manifestation - see MANIFESTATION_STATES)
    
    Returns:
        Dictionary mapping manifestation_number (1-41) -> state_value
        Only includes manifestations where the state is defined in MANIFESTATION_STATES
    """
    with open(manifestation_path, "r", encoding="utf-8") as f:
        line = f.readline().strip()
    
    # Each character is a digit representing the state of that manifestation
    manifestations = {}
    for position, char in enumerate(line):
        if char.isdigit():
            state = int(char)
            # Convert 0-based position to 1-based manifestation number
            manifestation_num = position + 1
            
            # Check if this manifestation and state are defined (pathological)
            if manifestation_num in MANIFESTATION_STATES:
                if state in MANIFESTATION_STATES[manifestation_num]["states"]:
                    manifestations[manifestation_num] = state
    
    return manifestations


async def register_experts(dataset_id: UUID) -> Dict[str, UUID]:
    """Register the two vessel segmentation experts."""
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
        List of SegmentationAnnotation models ready for upsert
    """
    all_segmentations: List[SegmentationAnnotation] = []
    
    # Process both experts
    for expert_key, expert_id in expert_ids.items():
        labels_dir = data_root / f"labels-{expert_key}"
        
        if not labels_dir.exists():
            logger.warning(f"Labels directory not found: {labels_dir}")
            continue
        
        mask_files = list(labels_dir.glob(f"*.{expert_key}.ppm"))
        logger.info(f"Processing {len(mask_files)} vessel masks from expert {expert_key}")
        
        for mask_path in mask_files:
            try:
                # Extract image ID from mask filename (e.g., im0001.ah.ppm -> im0001)
                image_name = mask_path.stem.split(".")[0]  # im0001.ah -> im0001
                
                if image_name not in image_id_map:
                    tracker.record_error(
                        error_type="image_not_found",
                        error_message=f"Image not found for mask: {image_name}",
                        item_id=image_name,
                        item_path=str(mask_path),
                    )
                    continue
                
                image_id = image_id_map[image_name]
                
                # Register mask file for provenance
                # Note: PPM files don't have a specific type in the schema, use None
                raw_file_id, chain_id = await register_individual_file(
                    file_path=mask_path,
                    dataset_id=dataset_id,
                    unified_annotation_type="segmentation",
                    file_type=None,  # PPM mask files
                    auto_detect_type=False,  # Don't auto-detect .ppm extension
                )
                
                # Generate expert annotation ID for this segmentation
                expert_annotation_id = generate_expert_annotation_uuid(
                    expert_id=expert_id,
                    annotation_task="segmentation",
                    raw_data_id=raw_file_id,
                    annotation_value_hash=None,
                )
                
                # Create and register the expert annotation record
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
                segmentation = await process_segmentation_from_binary_mask(
                    mask_path=mask_path,
                    annotation_type="vessels",
                    image_id=image_id,
                    annotation_description=f"Blood vessel segmentation by {EXPERTS[expert_key]['name']}",
                    fill_holes=False,  # CRITICAL: Don't fill holes for vessels!
                    raw_data_id=raw_file_id,
                    expert_annotation_id=expert_annotation_id,
                    annotation_method="manual",
                    provenance_chain_id=chain_id,
                    dataset_name=DATASET_NAME,
                )
                
                all_segmentations.append(segmentation)
                tracker.update(success=True)
                tracker.record_success("vessel_segmentation")
                
            except Exception as e:
                tracker.update(success=False)
                tracker.record_error(
                    error_type="segmentation_processing",
                    error_message=str(e),
                    item_id=mask_path.stem,
                    item_path=str(mask_path),
                )
                logger.error(f"Failed to process mask {mask_path}: {e}")
    
    return all_segmentations


async def process_diagnosis_descriptions(
    data_root: Path,
    dataset_id: UUID,
    image_id_map: Dict[str, UUID],
    diagnosis_map: Dict[str, Tuple[str, str]],
    tracker: ProgressTracker,
) -> List[ClinicalDescription]:
    """
    Process diagnosis text as clinical descriptions.
    
    Returns:
        List of ClinicalDescription models ready for upsert
    """
    all_descriptions: List[ClinicalDescription] = []
    
    # Register diagnosis.txt file for provenance
    diagnosis_path = data_root / "diagnosis.txt"
    if not diagnosis_path.exists():
        logger.warning(f"Diagnosis file not found: {diagnosis_path}")
        return all_descriptions
    
    raw_file_id, chain_id = await register_text_file(
        file_path=diagnosis_path,
        dataset_id=dataset_id,
        unified_annotation_type="description",  # Valid type from schema
        file_type="txt",  # Valid file type from schema
    )
    
    for image_name, (diagnosis_codes, diagnosis_text) in diagnosis_map.items():
        try:
            # Only process if we have diagnosis text and the image exists
            if not diagnosis_text or not diagnosis_text.strip():
                continue
            
            if image_name not in image_id_map:
                tracker.record_error(
                    error_type="image_not_found_for_diagnosis",
                    error_message=f"Image not found for diagnosis: {image_name}",
                    item_id=image_name,
                )
                continue
            
            image_id = image_id_map[image_name]
            
            # Generate description ID
            description_id = generate_description_uuid(
                image_id=image_id,
                description_type="diagnosis_text",
                expert_id=None,
                raw_data_id=raw_file_id,
            )
            
            # Calculate word count
            word_count = len(diagnosis_text.split())
            
            # Create ClinicalDescription model
            description = ClinicalDescription(
                description_id=description_id,
                image_id=image_id,
                description_text=diagnosis_text,
                description_type="diagnosis_text",
                raw_data_id=raw_file_id,
                expert_id=None,
                word_count=word_count,
            )
            
            all_descriptions.append(description)
            tracker.update(success=True)
            tracker.record_success("clinical_description")
            
        except Exception as e:
            tracker.update(success=False)
            tracker.record_error(
                error_type="diagnosis_processing",
                error_message=str(e),
                item_id=image_name,
            )
            logger.error(f"Failed to process diagnosis for {image_name}: {e}")
    
    logger.info(f"Processed {len(all_descriptions)} clinical descriptions from diagnosis text")
    return all_descriptions


async def process_manifestation_annotations(
    data_root: Path,
    dataset_id: UUID,
    image_id_map: Dict[str, UUID],
    tracker: ProgressTracker,
) -> List[KeywordAnnotation]:
    """
    Process manifestation annotations as keyword annotations.
    
    Manifestations are encoded as 42-character strings where:
    - Position in string = manifestation ID
    - Digit value = state (0=Unknown, 1=Absent, 2+=Present)
    
    We only process manifestations with state >= 2 (present).
    
    Returns:
        List of KeywordAnnotation models ready for upsert
    """
    all_keywords: List[KeywordAnnotation] = []
    
    annotations_dir = data_root / "annotations"
    if not annotations_dir.exists():
        logger.warning(f"Annotations directory not found: {annotations_dir}")
        return all_keywords
    
    # Find all manifestation files
    manifestation_files = list(annotations_dir.glob("*.fea.mg.txt"))
    logger.info(f"Found {len(manifestation_files)} manifestation files")
    
    for manifest_path in manifestation_files:
        try:
            # Extract image name (e.g., im0001.fea.mg.txt -> im0001)
            image_name = manifest_path.stem.split(".")[0]
            
            if image_name not in image_id_map:
                tracker.record_error(
                    error_type="image_not_found_for_manifestation",
                    error_message=f"Image not found for manifestation: {image_name}",
                    item_id=image_name,
                    item_path=str(manifest_path),
                )
                continue
            
            image_id = image_id_map[image_name]
            
            # Register manifestation file for provenance
            raw_file_id, chain_id = await register_text_file(
                file_path=manifest_path,
                dataset_id=dataset_id,
                unified_annotation_type="keyword",
                file_type="txt",  # Valid file type from schema
            )
            
            # Parse manifestation file
            manifestations = parse_manifestation_file(manifest_path)
            
            # Process each manifestation as a keyword
            for manifestation_num, state_value in manifestations.items():
                # Get manifestation name and state description
                if manifestation_num not in MANIFESTATION_STATES:
                    logger.warning(
                        f"Unknown manifestation number {manifestation_num} in {manifest_path.name}"
                    )
                    continue
                
                manifestation_info = MANIFESTATION_STATES[manifestation_num]
                manifestation_name = manifestation_info["name"]
                state_description = manifestation_info["states"].get(state_value)
                
                if not state_description:
                    logger.warning(
                        f"Unknown state {state_value} for manifestation {manifestation_num} ({manifestation_name}) in {manifest_path.name}"
                    )
                    continue
                
                # Format as "Manifestation Name: State Description"
                keyword_term = f"{manifestation_name}: {state_description}"
                
                # Create keyword annotation for this manifestation
                keyword_annotation = await process_keyword_annotation(
                    keyword_term=keyword_term,
                    keyword_source="clinical_description",  # Manifestations are clinical findings
                    image_id=image_id,
                    dataset_id=dataset_id,
                    category="manifestation",
                    raw_data_id=raw_file_id,
                    expert_id=None,
                    annotation_method="manual",
                    provenance_chain_id=chain_id,
                )
                
                all_keywords.append(keyword_annotation)
            
            if manifestations:
                tracker.update(success=True)
                tracker.record_success("manifestation_annotation")
            
        except Exception as e:
            tracker.update(success=False)
            tracker.record_error(
                error_type="manifestation_processing",
                error_message=str(e),
                item_id=manifest_path.stem,
                item_path=str(manifest_path),
            )
            logger.error(f"Failed to process manifestation file {manifest_path}: {e}")
    
    logger.info(f"Processed {len(all_keywords)} manifestation keywords")
    return all_keywords


async def ingest_stare() -> OperationStatistics:
    """
    Main ingestion function for STARE dataset.
    
    Returns:
        OperationStatistics with success/error counts
    """
    data_root = get_data_root() / "11_STARE"
    dataset_id = generate_dataset_uuid(DATASET_NAME)
    
    logger.info("=" * 80)
    logger.info(f"Starting STARE dataset ingestion")
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
            "STructured Analysis of the Retina (STARE) Project. "
            "Contains retinal fundus images with blood vessel segmentation "
            "from two experts, clinical diagnosis, and 39 manifestation features."
        ),
    )
    await upsert_dataset(dataset)
    
    # Step 2: Register experts
    logger.info("Registering experts...")
    expert_ids = await register_experts(dataset_id)
    
    # Step 3: Discover images from both directories
    logger.info("Discovering images...")
    
    # Images with vessel segmentation annotations
    stare_images_dir = data_root / "stare-images"
    stare_image_paths = await asyncio.to_thread(find_images, stare_images_dir)
    logger.info(f"Found {len(stare_image_paths)} images with vessel annotations in {stare_images_dir}")
    
    # Raw images without annotations
    documents_dir = data_root / "documents"
    documents_image_paths = await asyncio.to_thread(find_images, documents_dir)
    logger.info(f"Found {len(documents_image_paths)} raw images in {documents_dir}")
    
    # Combine all images
    all_image_paths = stare_image_paths + documents_image_paths
    logger.info(f"Total images to process: {len(all_image_paths)}")
    
    # Create a set of image names that have vessel annotations
    images_with_vessel_annotations = {p.stem for p in stare_image_paths}
    
    # Step 4: Parse diagnosis file
    logger.info("Parsing diagnosis file...")
    diagnosis_path = data_root / "diagnosis.txt"
    diagnosis_map = {}
    if diagnosis_path.exists():
        diagnosis_map = parse_diagnosis_file(diagnosis_path)
    else:
        logger.warning(f"Diagnosis file not found: {diagnosis_path}")
    
    # Step 5: Setup progress tracker
    # Total: all images + vessel masks (2 experts × 20 images) + clinical descriptions + manifestations
    num_images_with_annotations = len(stare_image_paths)
    # Estimate: ~422 images + 40 vessel masks + ~403 diagnoses + ~397 manifestation files
    # We'll update as we go
    total_items = len(all_image_paths) + (num_images_with_annotations * 2) + len(diagnosis_map) + len(all_image_paths)
    tracker = ProgressTracker(
        total=total_items,
        description=f"Ingesting {DATASET_NAME}"
    )
    
    # Step 6: Process all images (both with and without annotations)
    logger.info(f"Processing {len(all_image_paths)} images...")
    all_images: List[Image] = []
    image_id_map: Dict[str, UUID] = {}  # Map image_name -> image_id
    
    for image_path in all_image_paths:
        try:
            image_name = image_path.stem  # im0001
            image_id = generate_image_uuid(dataset_id, image_name)
            
            # Create image with automatic metadata extraction
            image = Image(
                image_id=image_id,
                dataset_id=dataset_id,
                original_image_id=image_name,
                **get_image_metadata_dict(image_path),
                modality="fundus",
                acquisition_date=None,  # Not provided
                image_quality=None,  # Not provided
            )
            
            all_images.append(image)
            image_id_map[image_name] = image_id
            
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
    
    # Step 7: Bulk upsert images
    logger.info(f"Upserting {len(all_images)} images...")
    if all_images:
        try:
            await bulk_upsert_images(all_images, batch_size=500)
            logger.info(f"Successfully upserted {len(all_images)} images")
        except Exception as e:
            logger.error(f"Failed to bulk upsert images: {e}")
            raise
    
    # Step 8: Process vessel segmentation only for images that have annotations
    logger.info(f"Processing vessel segmentation for {len(images_with_vessel_annotations)} images...")
    
    # Filter image_id_map to only include images with vessel annotations
    annotated_image_id_map = {
        name: image_id 
        for name, image_id in image_id_map.items() 
        if name in images_with_vessel_annotations
    }
    
    all_segmentations = await process_vessel_segmentation(
        data_root=data_root,
        dataset_id=dataset_id,
        image_id_map=annotated_image_id_map,
        expert_ids=expert_ids,
        tracker=tracker,
    )
    
    # Step 9: Upsert segmentations (no bulk operation available yet)
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
    
    # Step 10: Process diagnosis text as clinical descriptions
    logger.info("Processing diagnosis text as clinical descriptions...")
    all_clinical_descriptions = await process_diagnosis_descriptions(
        data_root=data_root,
        dataset_id=dataset_id,
        image_id_map=image_id_map,
        diagnosis_map=diagnosis_map,
        tracker=tracker,
    )
    
    # Step 11: Upsert clinical descriptions
    logger.info(f"Upserting {len(all_clinical_descriptions)} clinical descriptions...")
    for description in all_clinical_descriptions:
        try:
            await upsert_clinical_description(description)
        except Exception as e:
            tracker.record_error(
                error_type="clinical_description_upsert",
                error_message=str(e),
                item_id=str(description.description_id),
            )
            logger.error(f"Failed to upsert clinical description: {e}")
    
    # Step 12: Process manifestation annotations as keywords
    logger.info("Processing manifestation annotations...")
    all_manifestation_keywords = await process_manifestation_annotations(
        data_root=data_root,
        dataset_id=dataset_id,
        image_id_map=image_id_map,
        tracker=tracker,
    )
    
    # Step 13: Upsert manifestation keyword annotations
    logger.info(f"Upserting {len(all_manifestation_keywords)} manifestation keywords...")
    for keyword_annotation in all_manifestation_keywords:
        try:
            await upsert_keyword_annotation(keyword_annotation)
        except Exception as e:
            tracker.record_error(
                error_type="manifestation_keyword_upsert",
                error_message=str(e),
                item_id=str(keyword_annotation.keyword_annotation_id),
            )
            logger.error(f"Failed to upsert manifestation keyword: {e}")
    
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
        stats = await ingest_stare()
        
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
