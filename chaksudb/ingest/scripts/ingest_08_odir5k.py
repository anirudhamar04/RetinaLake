"""
Ingestion script for ODIR-5K dataset.

Dataset: Ocular Disease Intelligent Recognition Database 5K
Structure: Excel file with patient data, paired left/right fundus images
Annotations: 
  - Patient-level: Age, sex, comorbidities (cataract, hypertension, myopia)
  - Image-level: Multi-label disease classification (N, D, G, A, O)
  - Image-level: Diagnostic keywords (comma-separated)

Disease Indicators:
  - Patient comorbidities: C (Cataract), H (Hypertension), M (Myopia)
  - Image classifications: N (Normal), D (Diabetes), G (Glaucoma), A (AMD), O (Other)
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Optional, Dict
from uuid import UUID

from chaksudb.common.progress import ProgressTracker, OperationStatistics
from chaksudb.config.config import get_data_root
from chaksudb.db.models import (
    Dataset,
    Image,
    Patient,
    PatientImage,
    ClassificationAnnotation,
    KeywordAnnotation,
)
from chaksudb.db.queries import (
    upsert_dataset,
    bulk_upsert_images,
    bulk_upsert_patients,
    bulk_upsert_patient_images,
    bulk_upsert_classification_annotations,
    upsert_keyword_annotation,
)
from chaksudb.ingest.framework import (
    get_image_metadata_dict,
    process_excel,
    find_matching_file,
)
from chaksudb.ingest.framework.gen_uuid import (
    generate_dataset_uuid,
    generate_image_uuid,
    generate_patient_uuid,
    generate_patient_image_uuid,
)
from chaksudb.ingest.framework.task_processors.classification_processor import process_classification
from chaksudb.ingest.framework.task_processors.keyword_processor import process_keywords_batch
from chaksudb.ingest.framework.split_assigner import register_standard_splits, bulk_assign_images_to_split

logger = logging.getLogger(__name__)

# Dataset metadata
DATASET_NAME = "ODIR-5K"
DATASET_URL = "https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k"
DATASET_LICENSE = "CC-BY-4.0"


async def ingest_odir5k() -> OperationStatistics:
    """
    Main ingestion function for ODIR-5K dataset.
    
    The ODIR-5K dataset contains:
    - Excel file with patient demographics and disease indicators
    - Paired left/right fundus images per patient
    - Diagnostic keywords for each image
    - Binary disease indicators (N, D, G, C, A, H, M, O)
    
    Processing:
    - C, H, M → Patient-level comorbidities
    - N, D, G, A, O → Image-level multi-label classifications
    - Diagnostic keywords → Image-level keyword annotations
    
    Returns:
        OperationStatistics with success/error counts
    """
    data_root = get_data_root() / "08_ODIR-5K" / "ODIR-5K" / "ODIR-5K"
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
    
    # Step 2: Check paths
    excel_path = data_root / "data.xlsx"
    train_dir = data_root / "Training Images"
    test_dir = data_root / "Testing Images"
    
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")
    
    logger.info(f"Found Excel file: {excel_path}")
    logger.info(f"Training images: {train_dir}")
    logger.info(f"Testing images: {test_dir}")
    
    # Step 3: Count rows for progress tracking
    logger.info("Counting patient records...")
    import openpyxl
    wb = openpyxl.load_workbook(excel_path, read_only=True)
    ws = wb.active
    total_rows = ws.max_row - 1  # Exclude header
    wb.close()
    logger.info(f"Found {total_rows} patient records")
    
    # Step 4: Setup progress tracker
    # Each patient has 2 images (left + right)
    tracker = ProgressTracker(
        total=total_rows * 2,  # 2 images per patient
        description=f"Ingesting {DATASET_NAME}"
    )
    
    # Collect items for bulk upsert
    all_patients: List[Patient] = []
    all_images: List[Image] = []
    all_patient_images: List[PatientImage] = []
    all_classifications: List[ClassificationAnnotation] = []
    all_keywords: List[KeywordAnnotation] = []
    
    # Track splits
    image_to_split: Dict[UUID, str] = {}  # image_id -> "train" or "test"
    
    # Step 5: Process Excel rows
    async def process_patient_row(row, idx):
        """Process a single patient row from the Excel file."""
        try:
            patient_id_str = str(row["ID"])
            
            # Extract patient demographics
            age = row.get("Patient Age")
            if age is not None and isinstance(age, (int, float)):
                age = int(age)
            else:
                age = None
            
            sex_raw = row.get("Patient Sex")
            sex = None
            if sex_raw:
                sex_str = str(sex_raw).strip().lower()
                if sex_str in ["male", "m"]:
                    sex = "male"
                elif sex_str in ["female", "f"]:
                    sex = "female"
                else:
                    sex = "unknown"
            
            # Extract comorbidities (C, H, M columns)
            comorbidities = {}
            if "C" in row:
                comorbidities["cataract"] = bool(int(row["C"]) if row["C"] is not None else 0)
            if "H" in row:
                comorbidities["hypertension"] = bool(int(row["H"]) if row["H"] is not None else 0)
            if "M" in row:
                comorbidities["myopia"] = bool(int(row["M"]) if row["M"] is not None else 0)
            
            # Register patient
            patient_id = generate_patient_uuid(dataset_id, patient_id_str)
            patient = Patient(
                patient_id=patient_id,
                dataset_id=dataset_id,
                original_patient_id=patient_id_str,
                age=age,
                sex=sex,
                comorbidities=comorbidities if comorbidities else None,
            )
            all_patients.append(patient)
            
            # Extract disease indicators for classifications (N, D, G, A, O)
            # These are image-level annotations (can differ per eye)
            disease_indicators = {}
            for disease_col, disease_name in [
                ("N", "normal"),
                ("D", "diabetes"),
                ("G", "glaucoma"),
                ("A", "amd"),
                ("O", "other"),
            ]:
                if disease_col in row:
                    disease_indicators[disease_name] = bool(int(row[disease_col]) if row[disease_col] is not None else 0)
            
            # Process left and right images
            for laterality, fundus_col, keywords_col in [
                ("left", "Left-Fundus", "Left-Diagnostic Keywords"),
                ("right", "Right-Fundus", "Right-Diagnostic Keywords"),
            ]:
                image_filename = row.get(fundus_col)
                if not image_filename:
                    logger.warning(f"No {laterality} image filename for patient {patient_id_str}")
                    tracker.update(success=False)
                    tracker.record_error(
                        error_type="missing_filename",
                        error_message=f"Missing {laterality} image filename",
                        item_id=patient_id_str,
                    )
                    continue
                
                # Find image file in train or test directory
                # Check both directories for the image file
                image_path = None
                for search_dir in [train_dir, test_dir]:
                    candidate = search_dir / image_filename
                    if candidate.exists():
                        image_path = candidate
                        break
                
                if not image_path:
                    logger.warning(f"Image not found: {image_filename}")
                    tracker.update(success=False)
                    tracker.record_error(
                        error_type="file_not_found",
                        error_message=f"Image file not found: {image_filename}",
                        item_id=patient_id_str,
                        item_path=image_filename,
                    )
                    continue
                
                # Determine split based on directory
                split_name = "train" if "Training" in str(image_path) else "test"
                
                # Generate image ID
                image_id = generate_image_uuid(dataset_id, image_filename)
                
                # Create image with metadata
                image = Image(
                    image_id=image_id,
                    dataset_id=dataset_id,
                    original_image_id=image_filename,
                    eye_laterality=laterality,
                    **get_image_metadata_dict(image_path),
                    modality="fundus",
                )
                all_images.append(image)
                image_to_split[image_id] = split_name
                
                # Link patient to image
                relationship_id = generate_patient_image_uuid(patient_id, image_id)
                patient_image = PatientImage(
                    relationship_id=relationship_id,
                    patient_id=patient_id,
                    image_id=image_id,
                )
                all_patient_images.append(patient_image)
                
                # Standard multi-disease panel: always store the FULL vector (including the
                # all-negative case) so panel completeness holds and negatives are real.
                classifications = await process_classification(
                    class_value=disease_indicators,
                    task_type="multi_label",
                    task_name="disease_panel",
                    class_name="disease_panel",
                    image_id=image_id,
                    annotation_method="manual",
                )
                all_classifications.extend(classifications)
                
                # Process diagnostic keywords
                keywords_str = row.get(keywords_col)
                if keywords_str and str(keywords_str).strip():
                    # ODIR-5K uses Chinese comma (，) as delimiter
                    # Normalize by replacing Chinese comma with regular comma
                    keywords_normalized = str(keywords_str).replace("，", ",")
                    
                    # Use the keyword processor to parse and register keywords
                    keyword_annotations = await process_keywords_batch(
                        keywords=keywords_normalized,
                        keyword_source="diagnostic_keywords",
                        image_id=image_id,
                        dataset_id=dataset_id,
                        delimiter=",",
                        annotation_method="manual",
                    )
                    all_keywords.extend(keyword_annotations)
                
                # Record success for this image
                tracker.update(count=1, success=True)
                
        except Exception as e:
            logger.error(f"Failed to process patient row {idx}: {e}", exc_info=True)
            # Update for both images (left and right) that failed
            tracker.update(count=2, success=False)
            tracker.record_error(
                error_type="processing",
                error_message=str(e),
                item_id=row.get("ID"),
            )
    
    # Process Excel file with automatic provenance tracking
    logger.info("Processing patient records...")
    stats, raw_file_id, chain_id = await process_excel(
        excel_path=excel_path,
        dataset_id=dataset_id,
        unified_annotation_type="classification",  # Primary annotation type
        process_row_fn=process_patient_row,
        sheet_name=0,  # Use first sheet (default)
        progress_tracker=tracker,
        skip_errors=True,
    )
    
    # Step 6: Bulk upsert in parallel
    logger.info(f"Upserting {len(all_patients)} patients...")
    logger.info(f"Upserting {len(all_images)} images...")
    logger.info(f"Upserting {len(all_patient_images)} patient-image links...")
    logger.info(f"Upserting {len(all_classifications)} classifications...")
    
    await asyncio.gather(
        bulk_upsert_patients(all_patients, batch_size=1000),
        bulk_upsert_images(all_images, batch_size=1000),
    )
    
    # Patient-images must be after patients and images
    await bulk_upsert_patient_images(all_patient_images, batch_size=1000)
    
    # Classifications
    if all_classifications:
        await bulk_upsert_classification_annotations(all_classifications, batch_size=1000)
    
    # Keywords (no bulk operation yet - upsert individually)
    logger.info(f"Upserting {len(all_keywords)} keyword annotations...")
    for keyword_ann in all_keywords:
        await upsert_keyword_annotation(keyword_ann)
    
    # Step 7: Register splits and assign images
    logger.info("Registering dataset splits...")
    
    # Separate images by split first
    train_image_ids = [img_id for img_id, split in image_to_split.items() if split == "train"]
    test_image_ids = [img_id for img_id, split in image_to_split.items() if split == "test"]
    
    splits = await register_standard_splits(
        dataset_id=dataset_id,
        split_type="explicit",
        train_count=len(train_image_ids),
        test_count=len(test_image_ids),
    )
    
    logger.info(f"Assigning {len(train_image_ids)} images to train split...")
    logger.info(f"Assigning {len(test_image_ids)} images to test split...")
    
    await asyncio.gather(
        bulk_assign_images_to_split(train_image_ids, splits["train"]) if train_image_ids else asyncio.sleep(0),
        bulk_assign_images_to_split(test_image_ids, splits["test"]) if test_image_ids else asyncio.sleep(0),
    )
    
    # Finish tracking
    tracker.finish()
    return tracker.get_statistics()


async def main():
    """Entry point for script execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    try:
        stats = await ingest_odir5k()
        
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
