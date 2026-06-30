"""
Helper functions for raw annotation file registration with provenance tracking.

This module provides utilities for registering raw annotation files (CSVs, Excel,
JSON, masks, XMLs, etc.) and creating their initial provenance chains. These
functions are used by framework ingestion helpers to automatically track the
source of all annotations.

All registration functions:
1. Compute file hash for idempotency
2. Generate deterministic UUID
3. Create RawAnnotationFile model
4. Store in database with provenance chain
5. Return (raw_file_id, chain_id) tuple
"""

import hashlib
import logging
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

from chaksudb.db.models import RawAnnotationFile
from chaksudb.db.queries import upsert_raw_annotation_file
from chaksudb.ingest.framework.gen_uuid import generate_raw_file_uuid
from chaksudb.ingest.framework.hashing import compute_file_hash
from chaksudb.ingest.framework.provenance import (
    ingest_raw_annotation_file_with_provenance,
)

logger = logging.getLogger(__name__)


async def register_csv_file(
    csv_path: Path,
    dataset_id: uuid.UUID,
    unified_annotation_type: str,
) -> Tuple[uuid.UUID, uuid.UUID]:
    """
    Register a CSV file as a raw annotation file with provenance tracking.
    
    Args:
        csv_path: Path to CSV file
        dataset_id: Dataset UUID
        unified_annotation_type: Primary annotation type in this CSV
            (grading, classification, quality, localization, keyword, description)
    
    Returns:
        Tuple of (raw_file_id, chain_id)
        
    Raises:
        FileNotFoundError: If CSV file does not exist
        
    Example:
        ```python
        raw_file_id, chain_id = await register_csv_file(
            csv_path=data_root / "labels.csv",
            dataset_id=dataset_id,
            unified_annotation_type="grading"
        )
        ```
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    file_hash = compute_file_hash(csv_path)
    raw_file_id = generate_raw_file_uuid(dataset_id, file_hash)
    
    raw_file = RawAnnotationFile(
        raw_file_id=raw_file_id,
        dataset_id=dataset_id,
        storage_provider="local",
        file_path=str(csv_path),
        file_type="csv",
        file_name=csv_path.name,
        file_hash=file_hash,
        file_size=csv_path.stat().st_size,
        parsed_status="not_parsed",
    )
    
    raw_file_id, chain_id = await ingest_raw_annotation_file_with_provenance(
        raw_file=raw_file,
        unified_annotation_type=unified_annotation_type,
    )
    
    logger.debug(
        f"Registered CSV file: {csv_path.name} "
        f"(raw_file_id={raw_file_id}, annotation_type={unified_annotation_type})"
    )
    
    return raw_file_id, chain_id


async def register_excel_file(
    excel_path: Path,
    dataset_id: uuid.UUID,
    unified_annotation_type: str,
    sheet_name: Optional[str] = None,
) -> Tuple[uuid.UUID, uuid.UUID]:
    """
    Register an Excel file as a raw annotation file with provenance tracking.
    
    Args:
        excel_path: Path to Excel file (.xlsx, .xls)
        dataset_id: Dataset UUID
        unified_annotation_type: Primary annotation type in this Excel file
        sheet_name: Optional sheet name being processed (for logging)
    
    Returns:
        Tuple of (raw_file_id, chain_id)
        
    Raises:
        FileNotFoundError: If Excel file does not exist
        
    Example:
        ```python
        raw_file_id, chain_id = await register_excel_file(
            excel_path=data_root / "annotations.xlsx",
            dataset_id=dataset_id,
            unified_annotation_type="classification",
            sheet_name="Labels"
        )
        ```
    """
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")
    
    file_hash = compute_file_hash(excel_path)
    raw_file_id = generate_raw_file_uuid(dataset_id, file_hash)
    
    # Use "excel" as file_type (sheet_name is for logging only)
    # Don't append sheet_name to file_type as it creates invalid types like "excel_0"
    file_type = "excel"
    
    raw_file = RawAnnotationFile(
        raw_file_id=raw_file_id,
        dataset_id=dataset_id,
        storage_provider="local",
        file_path=str(excel_path),
        file_type=file_type,
        file_name=excel_path.name,
        file_hash=file_hash,
        file_size=excel_path.stat().st_size,
        parsed_status="not_parsed",
    )
    
    raw_file_id, chain_id = await ingest_raw_annotation_file_with_provenance(
        raw_file=raw_file,
        unified_annotation_type=unified_annotation_type,
    )
    
    logger.debug(
        f"Registered Excel file: {excel_path.name} "
        f"(sheet={sheet_name}, raw_file_id={raw_file_id})"
    )
    
    return raw_file_id, chain_id


async def register_json_file(
    json_path: Path,
    dataset_id: uuid.UUID,
    unified_annotation_type: str,
) -> Tuple[uuid.UUID, uuid.UUID]:
    """
    Register a JSON file as a raw annotation file with provenance tracking.
    
    Args:
        json_path: Path to JSON file (.json, .jsonl)
        dataset_id: Dataset UUID
        unified_annotation_type: Primary annotation type in this JSON file
    
    Returns:
        Tuple of (raw_file_id, chain_id)
        
    Raises:
        FileNotFoundError: If JSON file does not exist
        
    Example:
        ```python
        raw_file_id, chain_id = await register_json_file(
            json_path=data_root / "metadata.json",
            dataset_id=dataset_id,
            unified_annotation_type="keyword"
        )
        ```
    """
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    
    file_hash = compute_file_hash(json_path)
    raw_file_id = generate_raw_file_uuid(dataset_id, file_hash)
    
    raw_file = RawAnnotationFile(
        raw_file_id=raw_file_id,
        dataset_id=dataset_id,
        storage_provider="local",
        file_path=str(json_path),
        file_type="json",
        file_name=json_path.name,
        file_hash=file_hash,
        file_size=json_path.stat().st_size,
        parsed_status="not_parsed",
    )
    
    raw_file_id, chain_id = await ingest_raw_annotation_file_with_provenance(
        raw_file=raw_file,
        unified_annotation_type=unified_annotation_type,
    )
    
    logger.debug(
        f"Registered JSON file: {json_path.name} (raw_file_id={raw_file_id})"
    )
    
    return raw_file_id, chain_id


async def register_individual_file(
    file_path: Path,
    dataset_id: uuid.UUID,
    unified_annotation_type: str,
    file_type: Optional[str] = None,
    auto_detect_type: bool = True,
) -> Tuple[uuid.UUID, uuid.UUID]:
    """
    Register any individual file (mask, XML, text, etc.) as raw annotation file.
    
    This is the generic registration function used by process_folder_tree()
    to register individual mask files, XML annotations, or any other file type.
    
    Args:
        file_path: Path to the file
        dataset_id: Dataset UUID
        unified_annotation_type: Annotation type (segmentation, localization, etc.)
        file_type: Optional file type override. If None and auto_detect_type=True,
                  will be detected from extension. If None and auto_detect_type=False, will be NULL.
        auto_detect_type: If True and file_type is None, auto-detect from extension.
                         If False, file_type will remain NULL in database (for binary files like masks).
    
    Returns:
        Tuple of (raw_file_id, chain_id)
        
    Raises:
        FileNotFoundError: If file does not exist
        
    Example:
        ```python
        # Register a mask file
        raw_file_id, chain_id = await register_individual_file(
            file_path=masks_dir / "image_001_mask.png",
            dataset_id=dataset_id,
            unified_annotation_type="segmentation"
        )
        
        # Register an XML annotation
        raw_file_id, chain_id = await register_individual_file(
            file_path=xml_dir / "image_001.xml",
            dataset_id=dataset_id,
            unified_annotation_type="localization",
            file_type="xml_voc"
        )
        ```
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    file_hash = compute_file_hash(file_path)
    raw_file_id = generate_raw_file_uuid(dataset_id, file_hash)
    
    # Auto-detect file type from extension if not provided AND auto_detect_type is True
    # Otherwise file_type remains None (will be NULL in database)
    if file_type is None and auto_detect_type:
        file_type = file_path.suffix.lstrip(".") or None
    
    raw_file = RawAnnotationFile(
        raw_file_id=raw_file_id,
        dataset_id=dataset_id,
        storage_provider="local",
        file_path=str(file_path),
        file_type=file_type,
        file_name=file_path.name,
        file_hash=file_hash,
        file_size=file_path.stat().st_size,
        parsed_status="not_parsed",
    )
    
    raw_file_id, chain_id = await ingest_raw_annotation_file_with_provenance(
        raw_file=raw_file,
        unified_annotation_type=unified_annotation_type,
    )
    
    logger.debug(
        f"Registered file: {file_path.name} "
        f"(type={file_type}, raw_file_id={raw_file_id})"
    )
    
    return raw_file_id, chain_id


async def register_mask_directory(
    directory_path: Path,
    dataset_id: uuid.UUID,
    unified_annotation_type: str = "segmentation",
) -> Tuple[uuid.UUID, uuid.UUID]:
    """
    Register entire mask directory as a single raw annotation file.
    
    NOTE: This function is provided for backwards compatibility and special cases
    where you want to track a directory as a single unit. For most use cases,
    use register_individual_file() to track each mask separately.
    
    Computes a directory hash based on the sorted list of files and their sizes.
    If any file in the directory changes, the hash changes.
    
    Args:
        directory_path: Path to mask directory
        dataset_id: Dataset UUID
        unified_annotation_type: Annotation type (usually "segmentation")
    
    Returns:
        Tuple of (raw_file_id, chain_id)
        
    Raises:
        FileNotFoundError: If directory does not exist
        
    Example:
        ```python
        # Register entire masks folder
        raw_file_id, chain_id = await register_mask_directory(
            directory_path=data_root / "Masks",
            dataset_id=dataset_id,
        )
        ```
    """
    if not directory_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory_path}")
    
    if not directory_path.is_dir():
        raise ValueError(f"Path is not a directory: {directory_path}")
    
    # Compute directory hash
    dir_hash = compute_directory_hash(directory_path)
    raw_file_id = generate_raw_file_uuid(dataset_id, dir_hash)
    
    raw_file = RawAnnotationFile(
        raw_file_id=raw_file_id,
        dataset_id=dataset_id,
        storage_provider="local",
        file_path=str(directory_path),
        file_type=None,
        file_name=directory_path.name,
        file_hash=dir_hash,
        file_size=None,  # Directory doesn't have a size
        parsed_status="not_parsed",
    )
    
    raw_file_id, chain_id = await ingest_raw_annotation_file_with_provenance(
        raw_file=raw_file,
        unified_annotation_type=unified_annotation_type,
    )
    
    logger.debug(
        f"Registered mask directory: {directory_path.name} (raw_file_id={raw_file_id})"
    )
    
    return raw_file_id, chain_id


async def update_raw_file_status(
    raw_file_id: uuid.UUID,
    parsed_status: str,
    errors: Optional[List[str]] = None,
) -> None:
    """
    Update parsing status of a raw annotation file after processing.
    
    This function is called by framework functions after processing to mark
    files as successfully parsed or with errors.
    
    Args:
        raw_file_id: UUID of the raw annotation file
        parsed_status: Status to set ("parsed", "error", "not_parsed")
        errors: Optional list of error messages (first 10 will be stored)
        
    Example:
        ```python
        # After successful processing
        await update_raw_file_status(raw_file_id, "parsed", None)
        
        # After errors
        await update_raw_file_status(
            raw_file_id,
            "error",
            ["Row 5: Invalid grade", "Row 10: Missing field"]
        )
        ```
    """
    # Prepare error text (limit to first 10 errors)
    parse_errors = None
    if errors:
        error_lines = errors[:10]
        if len(errors) > 10:
            error_lines.append(f"... and {len(errors) - 10} more errors")
        parse_errors = "\n".join(error_lines)
    
    # Update the raw file record
    from chaksudb.db.queries import get_raw_annotation_file
    
    # Get existing record
    existing_raw_file = await get_raw_annotation_file(raw_file_id)
    
    if existing_raw_file is None:
        logger.warning(f"Raw file {raw_file_id} not found for status update")
        return
    
    # Update fields
    existing_raw_file.parsed_status = parsed_status
    existing_raw_file.parse_errors = parse_errors
    
    # Upsert with updated values
    await upsert_raw_annotation_file(existing_raw_file)
    
    logger.debug(
        f"Updated raw file {raw_file_id} status to '{parsed_status}'"
        + (f" with {len(errors)} errors" if errors else "")
    )


def compute_directory_hash(directory: Path) -> str:
    """
    Compute hash of directory contents (file list + sizes).
    
    Creates a deterministic hash based on:
    - Sorted list of all file paths (relative to directory)
    - File size for each file
    
    If any file is added, removed, or modified, the hash changes.
    
    Args:
        directory: Directory to hash
        
    Returns:
        Hexadecimal SHA256 hash string (64 characters)
        
    Raises:
        FileNotFoundError: If directory does not exist
        ValueError: If path is not a directory
        
    Example:
        ```python
        # Hash a masks directory
        dir_hash = compute_directory_hash(Path("data/IDRID/Masks"))
        # Returns: "a3f5b2c8..."
        
        # If any mask file changes, hash will be different
        ```
    """
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    if not directory.is_dir():
        raise ValueError(f"Path is not a directory: {directory}")
    
    # Collect all files with their relative paths and sizes
    file_info = []
    for file_path in sorted(directory.rglob("*")):
        if file_path.is_file():
            rel_path = file_path.relative_to(directory)
            size = file_path.stat().st_size
            # Format: "relative/path:size"
            file_info.append(f"{rel_path}:{size}")
    
    # Hash the concatenated file info
    content = "\n".join(file_info).encode("utf-8")
    sha256_hash = hashlib.sha256()
    sha256_hash.update(content)
    
    return sha256_hash.hexdigest()
