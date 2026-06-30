"""
Ingestion script for SUSTech-SYSU dataset.

Dataset: SUSTech-SYSU Fundus Image Dataset
Structure: 
  - drLabels.csv: DR grading with multiple scales (ICDR, AAO, Scottish) + laterality
  - c5_DR_reclassified.csv: Reclassified grade 5 images with multiple scales
  - exudatesLabels/: XML files for exudate localization (Pascal VOC format)
  - odFoveaLabels/: XML files for OD and Fovea localization (Pascal VOC format)
  - originalImages/: Fundus images (.jpg)
Annotations: 
  - DR grading (3 scales: ICDR, AAO, Scottish)
  - Localization (exudates, OD, Fovea)
Tasks: DR grading (multiple scales), Localization (exudates, OD, Fovea)
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional
from uuid import UUID

from chaksudb.common.progress import ProgressTracker, OperationStatistics
from chaksudb.config.config import get_data_root
from chaksudb.db.models import (
    Dataset,
    DiseaseGrading,
    Image,
    LocalizationAnnotation,
)
from chaksudb.db.queries import (
    bulk_upsert_disease_gradings,
    bulk_upsert_images,
    bulk_upsert_localization_annotations,
    upsert_dataset,
)
from chaksudb.ingest.framework import (
    get_image_metadata_dict,
    process_csv,
    process_folder_tree,
)
from chaksudb.ingest.framework.gen_uuid import (
    generate_dataset_uuid,
    generate_image_uuid,
)
from chaksudb.ingest.framework.provenance_context import get_current_provenance
from chaksudb.ingest.framework.task_processors.grading_processor import process_disease_grade
from chaksudb.ingest.framework.task_processors.localization_processor import (
    process_localization_from_xml,
)
from chaksudb.ingest.framework.split_assigner import auto_stratified_splits

logger = logging.getLogger(__name__)

# Dataset metadata
DATASET_NAME = "SUSTech-SYSU"
DATASET_URL = "https://doi.org/10.6084/m9.figshare.12570770.v1"
DATASET_LICENSE = "CC-BY-4.0"

# DR grading scale names (must match scales registered in bootstrap_scale_mappings.py)
ICDR_SCALE = "ICDR_0_4"
AAO_SCALE = "AAO"  # Note: registered as "AAO", not "AAO_DR_0_4"
SCOTTISH_SCALE = "Scottish"


async def ingest_sustech_sysu() -> OperationStatistics:
    """
    Main ingestion function for SUSTech-SYSU dataset.
    
    Returns:
        OperationStatistics with success/error counts
    """
    data_root = get_data_root() / "30_SUSTech-SYSU"
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
    
    # Step 2: Count total items for progress tracking
    logger.info("Counting items to process...")
    total_count = 0
    
    # Count CSV rows
    dr_labels_csv = data_root / "drLabels.csv"
    c5_reclassified_csv = data_root / "c5_DR_reclassified.csv"
    
    if dr_labels_csv.exists():
        import csv
        with open(dr_labels_csv, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            total_count += sum(1 for _ in reader)
            logger.info(f"  drLabels.csv: {total_count} rows")
    
    if c5_reclassified_csv.exists():
        with open(c5_reclassified_csv, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            c5_count = sum(1 for _ in reader)
            total_count += c5_count
            logger.info(f"  c5_DR_reclassified.csv: {c5_count} rows")
    
    # Count XML files
    exudates_dir = data_root / "exudatesLabels"
    od_fovea_dir = data_root / "odFoveaLabels"
    
    if exudates_dir.exists():
        xml_files = list(exudates_dir.glob("*.xml"))
        total_count += len(xml_files)
        logger.info(f"  exudatesLabels/: {len(xml_files)} XML files")
    
    if od_fovea_dir.exists():
        xml_files = list(od_fovea_dir.glob("*.xml"))
        total_count += len(xml_files)
        logger.info(f"  odFoveaLabels/: {len(xml_files)} XML files")
    
    logger.info(f"Total items to process: {total_count}")
    
    # Step 3: Setup progress tracker
    tracker = ProgressTracker(
        total=total_count,
        description=f"Ingesting {DATASET_NAME}"
    )
    
    # Step 4: Collect items for bulk upsert
    all_images: List[Image] = []
    all_gradings: List[DiseaseGrading] = []
    all_localizations: List[LocalizationAnnotation] = []
    image_id_map: Dict[str, UUID] = {}  # image_filename -> image_id
    image_labels: dict = {}  # image_id -> ICDR grade (for stratified splits)
    
    # Step 5: Process DR grading from drLabels.csv
    logger.info("Processing DR grading from drLabels.csv...")
    image_dir = data_root / "originalImages"
    
    async def process_dr_labels_row(row, idx):
        """Process a single row from drLabels.csv."""
        try:
            image_filename = row["Fundus_images"].strip()
            image_id_str = image_filename
            
            # Generate or get image ID
            if image_id_str not in image_id_map:
                image_id = generate_image_uuid(dataset_id, image_id_str)
                image_id_map[image_id_str] = image_id
            else:
                image_id = image_id_map[image_id_str]
            
            # Find image file
            image_path = image_dir / image_filename
            if not await asyncio.to_thread(image_path.exists):
                # Try case variations
                stem = image_path.stem
                for ext in [".jpg", ".JPG", ".jpeg", ".JPEG"]:
                    candidate = image_dir / f"{stem}{ext}"
                    if await asyncio.to_thread(candidate.exists):
                        image_path = candidate
                        break
                else:
                    tracker.record_error(
                        error_type="file_not_found",
                        error_message=f"Image not found: {image_filename}",
                        item_id=image_filename,
                    )
                    tracker.update(success=False)
                    return
            
            # Create image if not already created
            if image_id not in [img.image_id for img in all_images]:
                image = Image(
                    image_id=image_id,
                    dataset_id=dataset_id,
                    original_image_id=image_filename,
                    **get_image_metadata_dict(image_path),
                    modality="fundus",
                )
                all_images.append(image)
            
            # Get provenance
            raw_data_id, provenance_chain_id = get_current_provenance()
            
            # Process ICDR grading
            icdr_grade_str = row.get("DR_grade(International_Clinical_DR_Severity_Scale)", "").strip()
            if icdr_grade_str:
                try:
                    icdr_grade = int(icdr_grade_str)
                    grading = await process_disease_grade(
                        grade_value=icdr_grade,
                        disease_type="DR",
                        scale_name=ICDR_SCALE,
                        image_id=image_id,
                        raw_data_id=raw_data_id,
                        provenance_chain_id=provenance_chain_id,
                        annotation_method="manual",
                    )
                    all_gradings.append(grading)
                    image_labels[image_id] = icdr_grade  # Track for stratified splits
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid ICDR grade '{icdr_grade_str}' for {image_filename}: {e}")

            # Process AAO grading
            aao_grade_str = row.get("DR_grade(American_Academy_of_Ophthalmology)", "").strip()
            if aao_grade_str:
                try:
                    aao_grade = int(aao_grade_str)
                    grading = await process_disease_grade(
                        grade_value=aao_grade,
                        disease_type="DR",
                        scale_name=AAO_SCALE,
                        image_id=image_id,
                        raw_data_id=raw_data_id,
                        provenance_chain_id=provenance_chain_id,
                        annotation_method="manual",
                    )
                    all_gradings.append(grading)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid AAO grade '{aao_grade_str}' for {image_filename}: {e}")
            
            # Process Scottish grading
            scottish_grade_str = row.get("DR_grade(Scottish_DR_grading_protocol)", "").strip()
            if scottish_grade_str:
                try:
                    scottish_grade = int(scottish_grade_str)
                    grading = await process_disease_grade(
                        grade_value=scottish_grade,
                        disease_type="DR",
                        scale_name=SCOTTISH_SCALE,
                        image_id=image_id,
                        raw_data_id=raw_data_id,
                        provenance_chain_id=provenance_chain_id,
                        annotation_method="manual",
                    )
                    all_gradings.append(grading)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid Scottish grade '{scottish_grade_str}' for {image_filename}: {e}")
            
            tracker.update(success=True)
            
        except Exception as e:
            tracker.update(success=False)
            tracker.record_error(
                error_type="processing",
                error_message=str(e),
                item_id=row.get("Fundus_images", "unknown"),
            )
            logger.error(f"Failed to process drLabels row {idx}: {e}")
    
    if dr_labels_csv.exists():
        stats, raw_file_id, chain_id = await process_csv(
            dr_labels_csv,
            dataset_id,
            "grading",
            process_dr_labels_row,
            progress_tracker=tracker,
        )
        logger.info(f"Processed {stats.successful_items} rows from drLabels.csv")
    
    # Step 6: Process reclassified DR grading from c5_DR_reclassified.csv
    logger.info("Processing reclassified DR grading from c5_DR_reclassified.csv...")
    
    async def process_c5_reclassified_row(row, idx):
        """Process a single row from c5_DR_reclassified.csv."""
        try:
            image_filename = row["Fundus_images"].strip()
            image_id_str = image_filename
            
            # Generate or get image ID
            if image_id_str not in image_id_map:
                image_id = generate_image_uuid(dataset_id, image_id_str)
                image_id_map[image_id_str] = image_id
            else:
                image_id = image_id_map[image_id_str]
            
            # Find image file
            image_path = image_dir / image_filename
            if not await asyncio.to_thread(image_path.exists):
                # Try case variations
                stem = image_path.stem
                for ext in [".jpg", ".JPG", ".jpeg", ".JPEG"]:
                    candidate = image_dir / f"{stem}{ext}"
                    if await asyncio.to_thread(candidate.exists):
                        image_path = candidate
                        break
                else:
                    tracker.record_error(
                        error_type="file_not_found",
                        error_message=f"Image not found: {image_filename}",
                        item_id=image_filename,
                    )
                    tracker.update(success=False)
                    return
            
            # Create image if not already created
            if image_id not in [img.image_id for img in all_images]:
                image = Image(
                    image_id=image_id,
                    dataset_id=dataset_id,
                    original_image_id=image_filename,
                    **get_image_metadata_dict(image_path),
                    modality="fundus",
                )
                all_images.append(image)
            
            # Get provenance
            raw_data_id, provenance_chain_id = get_current_provenance()
            
            # Process ICDR grading (reclassified)
            icdr_grade_str = row.get("DR_grade(International_Clinical_DR_Severity_Scale)", "").strip()
            if icdr_grade_str:
                try:
                    icdr_grade = int(icdr_grade_str)
                    grading = await process_disease_grade(
                        grade_value=icdr_grade,
                        disease_type="DR",
                        scale_name=ICDR_SCALE,
                        image_id=image_id,
                        raw_data_id=raw_data_id,
                        provenance_chain_id=provenance_chain_id,
                        annotation_method="manual",
                    )
                    all_gradings.append(grading)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid ICDR grade '{icdr_grade_str}' for {image_filename}: {e}")
            
            # Process AAO grading (reclassified)
            aao_grade_str = row.get("DR_grade(American_Academy_of_Ophthalmology)", "").strip()
            if aao_grade_str:
                try:
                    aao_grade = int(aao_grade_str)
                    grading = await process_disease_grade(
                        grade_value=aao_grade,
                        disease_type="DR",
                        scale_name=AAO_SCALE,
                        image_id=image_id,
                        raw_data_id=raw_data_id,
                        provenance_chain_id=provenance_chain_id,
                        annotation_method="manual",
                    )
                    all_gradings.append(grading)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid AAO grade '{aao_grade_str}' for {image_filename}: {e}")
            
            # Process Scottish grading (reclassified)
            scottish_grade_str = row.get("DR_grade(Scottish_DR_grading_protocol)", "").strip()
            if scottish_grade_str:
                try:
                    scottish_grade = int(scottish_grade_str)
                    grading = await process_disease_grade(
                        grade_value=scottish_grade,
                        disease_type="DR",
                        scale_name=SCOTTISH_SCALE,
                        image_id=image_id,
                        raw_data_id=raw_data_id,
                        provenance_chain_id=provenance_chain_id,
                        annotation_method="manual",
                    )
                    all_gradings.append(grading)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid Scottish grade '{scottish_grade_str}' for {image_filename}: {e}")
            
            tracker.update(success=True)
            
        except Exception as e:
            tracker.update(success=False)
            tracker.record_error(
                error_type="processing",
                error_message=str(e),
                item_id=row.get("Fundus_images", "unknown"),
            )
            logger.error(f"Failed to process c5_DR_reclassified row {idx}: {e}")
    
    if c5_reclassified_csv.exists():
        stats, raw_file_id, chain_id = await process_csv(
            c5_reclassified_csv,
            dataset_id,
            "grading",
            process_c5_reclassified_row,
            progress_tracker=tracker,
        )
        logger.info(f"Processed {stats.successful_items} rows from c5_DR_reclassified.csv")
    
    # Step 7: Process exudates localization from XML files
    logger.info("Processing exudates localization from XML files...")
    
    async def process_exudates_xml(xml_path: Path, rel_path: Path, depth: int) -> None:
        """Process a single exudates XML file."""
        try:
            # Match XML filename stem to image filename
            xml_stem = xml_path.stem  # e.g., "0680" from "0680.xml"
            image_filename = f"{xml_stem}.jpg"
            image_id_str = image_filename
            
            # Generate or get image ID
            if image_id_str not in image_id_map:
                image_id = generate_image_uuid(dataset_id, image_id_str)
                image_id_map[image_id_str] = image_id
            else:
                image_id = image_id_map[image_id_str]
            
            # Find image file
            image_path = image_dir / image_filename
            if not await asyncio.to_thread(image_path.exists):
                # Try case variations
                for ext in [".jpg", ".JPG", ".jpeg", ".JPEG"]:
                    candidate = image_dir / f"{xml_stem}{ext}"
                    if await asyncio.to_thread(candidate.exists):
                        image_path = candidate
                        break
                else:
                    tracker.record_error(
                        error_type="file_not_found",
                        error_message=f"Image not found for XML: {xml_path.name}",
                        item_id=xml_path.name,
                    )
                    tracker.update(success=False)
                    return
            
            # Create image if not already created
            if image_id not in [img.image_id for img in all_images]:
                image = Image(
                    image_id=image_id,
                    dataset_id=dataset_id,
                    original_image_id=image_filename,
                    **get_image_metadata_dict(image_path),
                    modality="fundus",
                )
                all_images.append(image)
            
            # Process localization from XML (filter for exudates only)
            raw_data_id, provenance_chain_id = get_current_provenance()
            
            localizations = await process_localization_from_xml(
                xml_path=xml_path,
                image_id=image_id,
                class_filter=["ex", "exudates", "hard_exudates"],  # Filter for exudates
                raw_data_id=raw_data_id,
                provenance_chain_id=provenance_chain_id,
                annotation_method="manual",
            )
            all_localizations.extend(localizations)
            tracker.update(success=True)
            
        except Exception as e:
            tracker.update(success=False)
            tracker.record_error(
                error_type="processing",
                error_message=str(e),
                item_id=str(xml_path),
            )
            logger.error(f"Failed to process exudates XML {xml_path}: {e}")
    
    if exudates_dir.exists():
        stats = await process_folder_tree(
            root_dir=exudates_dir,
            dataset_id=dataset_id,
            unified_annotation_type="localization",
            process_file_fn=process_exudates_xml,
            file_extensions={".xml"},
            progress_tracker=tracker,
        )
        logger.info(f"Processed {stats.successful_items} exudates XML files")
    
    # Step 8: Process OD and Fovea localization from XML files
    logger.info("Processing OD and Fovea localization from XML files...")
    
    async def process_od_fovea_xml(xml_path: Path, rel_path: Path, depth: int) -> None:
        """Process a single OD/Fovea XML file."""
        try:
            # Match XML filename stem to image filename
            xml_stem = xml_path.stem  # e.g., "0680" from "0680.xml"
            image_filename = f"{xml_stem}.jpg"
            image_id_str = image_filename
            
            # Generate or get image ID
            if image_id_str not in image_id_map:
                image_id = generate_image_uuid(dataset_id, image_id_str)
                image_id_map[image_id_str] = image_id
            else:
                image_id = image_id_map[image_id_str]
            
            # Find image file
            image_path = image_dir / image_filename
            if not await asyncio.to_thread(image_path.exists):
                # Try case variations
                for ext in [".jpg", ".JPG", ".jpeg", ".JPEG"]:
                    candidate = image_dir / f"{xml_stem}{ext}"
                    if await asyncio.to_thread(candidate.exists):
                        image_path = candidate
                        break
                else:
                    tracker.record_error(
                        error_type="file_not_found",
                        error_message=f"Image not found for XML: {xml_path.name}",
                        item_id=xml_path.name,
                    )
                    tracker.update(success=False)
                    return
            
            # Create image if not already created
            if image_id not in [img.image_id for img in all_images]:
                image = Image(
                    image_id=image_id,
                    dataset_id=dataset_id,
                    original_image_id=image_filename,
                    **get_image_metadata_dict(image_path),
                    modality="fundus",
                )
                all_images.append(image)
            
            # Process localization from XML (filter for OD and Fovea)
            raw_data_id, provenance_chain_id = get_current_provenance()
            
            localizations = await process_localization_from_xml(
                xml_path=xml_path,
                image_id=image_id,
                class_filter=["OD", "fovea", "optic_disc", "macula"],  # Filter for OD and Fovea
                raw_data_id=raw_data_id,
                provenance_chain_id=provenance_chain_id,
                annotation_method="manual",
            )
            all_localizations.extend(localizations)
            tracker.update(success=True)
            
        except Exception as e:
            tracker.update(success=False)
            tracker.record_error(
                error_type="processing",
                error_message=str(e),
                item_id=str(xml_path),
            )
            logger.error(f"Failed to process OD/Fovea XML {xml_path}: {e}")
    
    if od_fovea_dir.exists():
        stats = await process_folder_tree(
            root_dir=od_fovea_dir,
            dataset_id=dataset_id,
            unified_annotation_type="localization",
            process_file_fn=process_od_fovea_xml,
            file_extensions={".xml"},
            progress_tracker=tracker,
        )
        logger.info(f"Processed {stats.successful_items} OD/Fovea XML files")
    
    # Step 9: Bulk upsert - images first, then annotations
    logger.info(f"Upserting {len(all_images)} images...")
    await bulk_upsert_images(all_images, batch_size=1000)
    
    logger.info(f"Upserting {len(all_gradings)} DR gradings...")
    await bulk_upsert_disease_gradings(all_gradings, batch_size=1000)
    
    logger.info(f"Upserting {len(all_localizations)} localizations...")
    await bulk_upsert_localization_annotations(all_localizations, batch_size=1000)

    # Step 10: Register splits
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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    stats = await ingest_sustech_sysu()
    
    logger.info("=" * 80)
    logger.info("Ingestion Summary:")
    logger.info(f"  Successful: {stats.successful_items}")
    logger.info(f"  Failed: {stats.failed_items}")
    logger.info(f"  Skipped: {stats.skipped_items}")
    logger.info("=" * 80)
    
    return 0 if stats.failed_items == 0 else 1


if __name__ == "__main__":
    exit(asyncio.run(main()))
