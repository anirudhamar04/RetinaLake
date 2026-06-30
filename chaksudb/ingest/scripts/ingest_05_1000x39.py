"""
Ingestion script for 1000x39 dataset.

Dataset: 1000 fundus images across 39 disease classes from JSIEC
Structure: Hierarchical folders with disease categories
Annotations: DR grading (0-3), binary classification, and keywords
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Dict
from uuid import UUID

from chaksudb.common.progress import ProgressTracker, OperationStatistics
from chaksudb.config.config import get_data_root
from chaksudb.db.models import (
    Dataset,
    Image,
    DiseaseGrading,
    ClassificationAnnotation,
    KeywordAnnotation,
)
from chaksudb.db.queries import (
    upsert_dataset,
    bulk_upsert_images,
    bulk_upsert_disease_gradings,
    bulk_upsert_classification_annotations,
    upsert_keyword_annotation,
)
from chaksudb.ingest.framework import (
    find_images,
    process_folder_tree,
    get_image_metadata_dict,
)
from chaksudb.ingest.framework.gen_uuid import (
    generate_dataset_uuid,
    generate_image_uuid,
)
from chaksudb.ingest.framework.task_processors.grading_processor import (
    process_disease_grade,
)
from chaksudb.ingest.framework.task_processors.classification_processor import (
    process_classification,
)
from chaksudb.ingest.framework.task_processors.keyword_processor import (
    process_keywords_batch,
)
from chaksudb.ingest.framework.split_assigner import auto_stratified_splits

logger = logging.getLogger(__name__)

# Dataset metadata
DATASET_NAME = "1000x39"
DATASET_URL = "https://doi.org/10.1038/s41598-019-47181-w"
DATASET_LICENSE = "CC-BY-4.0"

# Complete mapping for all 39 classes - ONLY explicit keywords from dataset
FOLDER_ANNOTATIONS = {
    # DR classes (4 classes) → disease_grading + classification + keywords
    "0.0.Normal": {
        "dr_grade": 0,
        "classification": "normal",
        "keywords": ["normal"],
    },
    "0.3.DR1": {
        "dr_grade": 1,
        "classification": "dr1",
        "keywords": ["DR grade 1", "diabetic retinopathy"],
    },
    "1.0.DR2": {
        "dr_grade": 2,
        "classification": "dr2",
        "keywords": ["DR grade 2", "diabetic retinopathy"],
    },
    "1.1.DR3": {
        "dr_grade": 3,
        "classification": "dr3",
        "keywords": ["DR grade 3", "diabetic retinopathy"],
    },
    # Non-DR classes (35 classes) → classification + keywords only
    "0.1.Tessellated fundus": {
        "classification": "tessellated_fundus",
        "keywords": ["tessellated fundus"],
    },
    "0.2.Large optic cup": {
        "classification": "large_optic_cup",
        "keywords": ["large optic cup"],
    },
    "2.0.BRVO": {
        "classification": "brvo",
        "keywords": ["BRVO", "branch retinal vein occlusion"],
    },
    "2.1.CRVO": {
        "classification": "crvo",
        "keywords": ["CRVO", "central retinal vein occlusion"],
    },
    "3.RAO": {
        "classification": "rao",
        "keywords": ["RAO", "retinal artery occlusion"],
    },
    "4.Rhegmatogenous RD": {
        "classification": "rhegmatogenous_rd",
        "keywords": ["rhegmatogenous RD", "retinal detachment"],
    },
    "5.0.CSCR": {
        "classification": "cscr",
        "keywords": ["CSCR", "central serous chorioretinopathy"],
    },
    "5.1.VKH disease": {
        "classification": "vkh_disease",
        "keywords": ["VKH disease"],
    },
    "6.Maculopathy": {
        "classification": "maculopathy",
        "keywords": ["maculopathy"],
    },
    "7.ERM": {
        "classification": "erm",
        "keywords": ["ERM", "epiretinal membrane"],
    },
    "8.MH": {
        "classification": "mh",
        "keywords": ["MH", "macular hole"],
    },
    "9.Pathological myopia": {
        "classification": "pathological_myopia",
        "keywords": ["pathological myopia"],
    },
    "10.0.Possible glaucoma": {
        "classification": "possible_glaucoma",
        "keywords": ["possible glaucoma", "glaucoma"],
    },
    "10.1.Optic atrophy": {
        "classification": "optic_atrophy",
        "keywords": ["optic atrophy"],
    },
    "11.Severe hypertensive retinopathy": {
        "classification": "severe_hypertensive_retinopathy",
        "keywords": ["hypertensive retinopathy"],
    },
    "12.Disc swelling and elevation": {
        "classification": "disc_swelling_and_elevation",
        "keywords": ["disc swelling", "disc elevation"],
    },
    "13.Dragged Disc": {
        "classification": "dragged_disc",
        "keywords": ["dragged disc"],
    },
    "14.Congenital disc abnormality": {
        "classification": "congenital_disc_abnormality",
        "keywords": ["congenital disc abnormality"],
    },
    "15.0.Retinitis pigmentosa": {
        "classification": "retinitis_pigmentosa",
        "keywords": ["retinitis pigmentosa"],
    },
    "15.1.Bietti crystalline dystrophy": {
        "classification": "bietti_crystalline_dystrophy",
        "keywords": ["Bietti crystalline dystrophy"],
    },
    "16.Peripheral retinal degeneration and break": {
        "classification": "peripheral_retinal_degeneration_and_break",
        "keywords": ["peripheral retinal degeneration", "retinal break"],
    },
    "17.Myelinated nerve fiber": {
        "classification": "myelinated_nerve_fiber",
        "keywords": ["myelinated nerve fiber"],
    },
    "18.Vitreous particles": {
        "classification": "vitreous_particles",
        "keywords": ["vitreous particles"],
    },
    "19.Fundus neoplasm": {
        "classification": "fundus_neoplasm",
        "keywords": ["fundus neoplasm"],
    },
    "20.Massive hard exudates": {
        "classification": "massive_hard_exudates",
        "keywords": ["massive hard exudates", "hard exudates"],
    },
    "21.Yellow-white spots-flecks": {
        "classification": "yellow_white_spots_flecks",
        "keywords": ["yellow-white spots", "flecks"],
    },
    "22.Cotton-wool spots": {
        "classification": "cotton_wool_spots",
        "keywords": ["cotton-wool spots"],
    },
    "23.Vessel tortuosity": {
        "classification": "vessel_tortuosity",
        "keywords": ["vessel tortuosity"],
    },
    "24.Chorioretinal atrophy-coloboma": {
        "classification": "chorioretinal_atrophy_coloboma",
        "keywords": ["chorioretinal atrophy", "coloboma"],
    },
    "25.Preretinal hemorrhage": {
        "classification": "preretinal_hemorrhage",
        "keywords": ["preretinal hemorrhage"],
    },
    "26.Fibrosis": {
        "classification": "fibrosis",
        "keywords": ["fibrosis"],
    },
    "27.Laser Spots": {
        "classification": "laser_spots",
        "keywords": ["laser spots"],
    },
    "28.Silicon oil in eye": {
        "classification": "silicon_oil_in_eye",
        "keywords": ["silicon oil in eye", "silicon oil"],
    },
    "29.0.Blur fundus without PDR": {
        "classification": "blur_fundus_without_pdr",
        "keywords": ["blur fundus", "without PDR"],
    },
    "29.1.Blur fundus with suspected PDR": {
        "classification": "blur_fundus_with_suspected_pdr",
        "keywords": ["blur fundus", "suspected PDR"],
    },
}


async def ingest_1000x39() -> OperationStatistics:
    """
    Main ingestion function for 1000x39 dataset.

    Strategy:
    - Use process_folder_tree() to walk 39 disease folders
    - Store DR grades (0-3) for 4 DR folders in disease_grading
    - Store all 39 classes as binary classification
    - Extract keywords from folder names
    - Bulk upsert all annotations

    Returns:
        OperationStatistics with success/error counts
    """
    data_root = get_data_root() / "05_1000x39"
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
    logger.info("Counting images across all folders...")
    total_images = 0
    for folder_name in FOLDER_ANNOTATIONS.keys():
        folder_path = data_root / folder_name
        if folder_path.exists():
            images_in_folder = await asyncio.to_thread(
                find_images, folder_path, recursive=False
            )
            total_images += len(images_in_folder)
            logger.info(f"  {folder_name}: {len(images_in_folder)} images")

    logger.info(f"Total images found: {total_images}")

    # Step 3: Setup progress tracker
    tracker = ProgressTracker(total=total_images, description=f"Ingesting {DATASET_NAME}")

    # Collect for bulk upsert
    all_images: List[Image] = []
    all_gradings: List[DiseaseGrading] = []
    all_classifications: List[ClassificationAnnotation] = []
    all_keywords: List[KeywordAnnotation] = []
    image_labels: dict = {}  # image_id → folder name (disease class) for stratified splitting

    async def handle_image(file_path: Path, rel_path: Path, depth: int):
        """
        Process each image with multi-table annotations:
        1. DR grading (for DR classes including normal)
        2. Classification (all 39 classes)
        3. Keywords (all 39 classes)
        """
        # Get folder name from parent directory
        folder_name = rel_path.parent.name

        # Skip if folder not in mapping
        if folder_name not in FOLDER_ANNOTATIONS:
            logger.warning(f"Unknown folder: {folder_name} (file: {file_path.name})")
            tracker.update(success=False)
            tracker.record_error(
                error_type="unknown_folder",
                error_message=f"Folder not in FOLDER_ANNOTATIONS mapping",
                item_id=file_path.stem,
                item_path=str(file_path),
            )
            return

        try:
            image_id = generate_image_uuid(dataset_id, file_path.stem)

            # Create image with automatic metadata extraction
            image = Image(
                image_id=image_id,
                dataset_id=dataset_id,
                original_image_id=file_path.stem,
                **get_image_metadata_dict(file_path),
                modality="fundus",
            )
            all_images.append(image)
            image_labels[image_id] = folder_name

            # Get annotation config
            config = FOLDER_ANNOTATIONS[folder_name]

            # 1. DR GRADING (for 4 DR classes: Normal + DR1 + DR2 + DR3)
            # Store ONLY in disease_grading table (not classification)
            if "dr_grade" in config:
                grading = await process_disease_grade(
                    grade_value=config["dr_grade"],  # 0, 1, 2, or 3
                    disease_type="DR",
                    scale_name="1000x39_DR_0_3",
                    image_id=image_id,
                    annotation_method="manual",  # Folder structure manually curated
                    scale_description="1000x39 DR grading scale (0=No DR, 1=DR1, 2=DR2, 3=DR3)",
                    min_value=0,
                    max_value=3,
                    value_labels={
                        "0": "No DR (Normal)",
                        "1": "DR1",
                        "2": "DR2",
                        "3": "DR3",
                    },
                )
                all_gradings.append(grading)
            else:
                # 2. CLASSIFICATION (only for non-DR classes)
                classifications = await process_classification(
                    class_value=True,  # Present
                    task_type="binary",
                    class_name=config["classification"],
                    image_id=image_id,
                    annotation_method="manual",  # Folder structure manually curated
                )
                all_classifications.extend(classifications)

            # 3. KEYWORDS (all 39 classes)
            keyword_annotations = await process_keywords_batch(
                keywords=", ".join(config["keywords"]),
                keyword_source="diagnostic_keywords",
                image_id=image_id,
                dataset_id=dataset_id,
                delimiter=",",
                annotation_method="manual",  # Folder structure manually curated
            )
            all_keywords.extend(keyword_annotations)

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

    # Step 4: Process folder tree with automatic per-file provenance
    logger.info("Processing images from folder structure...")
    stats = await process_folder_tree(
        root_dir=data_root,
        dataset_id=dataset_id,
        unified_annotation_type="classification",  # Primary annotation type
        process_file_fn=handle_image,
        file_extensions={".jpg", ".JPG", ".jpeg", ".JPEG"},
        recursive=True,
        include_dirs=False,
        progress_tracker=tracker,
        skip_errors=True,
    )

    # Step 5: Bulk upsert in parallel
    logger.info(
        f"Upserting {len(all_images)} images, {len(all_gradings)} gradings, "
        f"{len(all_classifications)} classifications, {len(all_keywords)} keywords..."
    )

    await bulk_upsert_images(all_images, batch_size=1000)
    await asyncio.gather(
        bulk_upsert_disease_gradings(all_gradings, batch_size=1000),
        bulk_upsert_classification_annotations(all_classifications, batch_size=1000),
    )

    # Keywords don't have bulk operation yet - insert individually
    logger.info(f"Upserting {len(all_keywords)} keyword annotations...")
    for keyword_ann in all_keywords:
        await upsert_keyword_annotation(keyword_ann)

    # Register splits — stratified 90/10 train+test, then 90/10 train+val
    all_image_ids = [img.image_id for img in all_images]
    if all_image_ids:
        logger.info("Registering dataset splits...")
        await auto_stratified_splits(
            dataset_id=dataset_id,
            split_assignments={"train": all_image_ids},
            labels=image_labels,
            split_type="explicit",
        )

    # Finish progress tracking
    tracker.finish()
    final_stats = tracker.get_statistics()

    # Log final summary
    logger.info("=" * 80)
    logger.info("Ingestion Summary:")
    logger.info(f"  Total items: {final_stats.total_items}")
    logger.info(f"  Successful: {final_stats.successful_items}")
    logger.info(f"  Failed: {final_stats.failed_items}")
    logger.info(f"  Skipped: {final_stats.skipped_items}")
    logger.info(f"  Images: {len(all_images)}")
    logger.info(f"  DR gradings: {len(all_gradings)}")
    logger.info(f"  Classifications: {len(all_classifications)}")
    logger.info(f"  Keywords: {len(all_keywords)}")
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
    log_file = Path("./logs/ingest_05_1000x39.log")
    log_file.touch(exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
        logging.FileHandler(log_file, mode='w'), 
        logging.StreamHandler(sys.stdout),          
        ],
    )

    stats = await ingest_1000x39()

    logger.info("=" * 80)
    logger.info(f"Ingestion complete!")
    logger.info(f"Total: {stats.total_items}")
    logger.info(f"Successful: {stats.successful_items}")
    logger.info(f"Failed: {stats.failed_items}")
    logger.info(f"Errors: {len(stats.errors)}")
    logger.info("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
