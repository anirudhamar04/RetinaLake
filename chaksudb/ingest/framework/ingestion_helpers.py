"""
Generic ingestion helper utilities.

These are lightweight wrappers that handle common boilerplate:
- File reading (CSV, JSON, Excel, text files)
- Iteration and progress tracking
- Error handling and logging
- File matching and pairing
- Automatic provenance tracking (raw file registration + chain creation)

The actual processing logic is provided by the user via callback functions.
These are NOT rigid adapters - they're flexible tools.
"""

import logging
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union, Awaitable

from chaksudb.common.progress import OperationStatistics, ProgressTracker
from chaksudb.ingest.framework.annotation_io import read_csv_auto, read_excel_sheet, read_json_file
from chaksudb.ingest.framework.provenance_context import (
    reset_provenance_context,
    set_provenance_context,
)
from chaksudb.ingest.framework.raw_file_helpers import (
    register_csv_file,
    register_excel_file,
    register_individual_file,
    register_json_file,
    update_raw_file_status,
)

logger = logging.getLogger(__name__)


# ============================================
# Generic CSV Processing
# ============================================


async def process_csv(
    csv_path: Path,
    dataset_id: uuid.UUID,
    unified_annotation_type: str,
    process_row_fn: Callable[[Dict[str, Any], int], Awaitable[None]],
    progress_tracker: Optional[ProgressTracker] = None,
    skip_errors: bool = True,
) -> Tuple[OperationStatistics, uuid.UUID, uuid.UUID]:
    """
    Generic CSV processor with automatic provenance tracking.
    
    Reads CSV file, registers it as a raw annotation file, and calls your
    function for each row. Provenance information (raw_file_id, chain_id) is
    automatically available to callbacks via get_current_provenance().
    
    Args:
        csv_path: Path to CSV file
        dataset_id: Dataset UUID (REQUIRED for provenance)
        unified_annotation_type: Primary annotation type in this CSV (REQUIRED)
            Must be one of: grading, segmentation, classification, localization,
            quality, keyword, description
        process_row_fn: async function(row: Dict[str, Any], row_index: int) -> None
            - row: Dictionary with CSV columns as keys
            - row_index: 0-based row number
            - Can call get_current_provenance() to get raw_file_id and chain_id
        progress_tracker: Optional progress tracker
        skip_errors: If True, log errors and continue; if False, raise
    
    Returns:
        Tuple of (OperationStatistics, raw_file_id, chain_id)
    
    Example:
        ```python
        from chaksudb.ingest.framework.provenance_context import get_current_provenance
        
        async def handle_row(row, idx):
            image_id = generate_image_uuid(dataset_id, row["filename"])
            
            # Provenance IDs automatically available
            raw_data_id, provenance_chain_id = get_current_provenance()
            
            await process_disease_grade(
                image_id=image_id,
                grade_value=int(row["level"]),
                raw_data_id=raw_data_id,
                provenance_chain_id=provenance_chain_id,
            )
        
        stats, raw_file_id, chain_id = await process_csv(
            csv_path=data_root / "train.csv",
            dataset_id=dataset_id,
            unified_annotation_type="grading",
            process_row_fn=handle_row
        )
        ```
    """
    # 1. Register raw file and create provenance chain
    raw_file_id, chain_id = await register_csv_file(
        csv_path, dataset_id, unified_annotation_type
    )
    
    # 2. Set provenance context for callbacks
    token_raw, token_chain = set_provenance_context(raw_file_id, chain_id)
    
    stats = OperationStatistics()
    
    try:
        # 3. Read and process CSV
        rows = read_csv_auto(csv_path)
        total_rows = len(rows)
        
        for idx, row in enumerate(rows):
            try:
                await process_row_fn(row, idx)
                stats.successful_items += 1
                    
            except Exception as e:
                stats.failed_items += 1
                stats.errors.append(f"Row {idx}: {str(e)}")
                logger.error(f"Error processing row {idx}: {e}")
                
                if not skip_errors:
                    raise
        
        # 4. Update raw file status
        final_status = "parsed" if stats.failed_items == 0 else "error"
        await update_raw_file_status(raw_file_id, final_status, stats.errors)
        
        logger.debug(
            f"CSV processing complete: {stats.successful_items} success, "
            f"{stats.failed_items} errors"
        )
        
    except Exception as e:
        # Mark as error on exception
        await update_raw_file_status(raw_file_id, "error", [str(e)])
        logger.error(f"Failed to process CSV {csv_path}: {e}")
        if not skip_errors:
            raise
    finally:
        # 5. Always cleanup provenance context
        reset_provenance_context(token_raw, token_chain)
    
    return stats, raw_file_id, chain_id


# ============================================
# Generic Excel Processing
# ============================================


async def process_excel(
    excel_path: Path,
    dataset_id: uuid.UUID,
    unified_annotation_type: str,
    process_row_fn: Callable[[Dict[str, Any], int], Awaitable[None]],
    sheet_name: Union[str, int] = 0,
    progress_tracker: Optional[ProgressTracker] = None,
    skip_errors: bool = True,
) -> Tuple[OperationStatistics, uuid.UUID, uuid.UUID]:
    """
    Generic Excel processor with automatic provenance tracking.
    
    Reads Excel file, registers it as a raw annotation file, and calls your
    function for each row. Provenance information is automatically available
    to callbacks via get_current_provenance().
    
    Args:
        excel_path: Path to Excel file (.xlsx, .xls)
        dataset_id: Dataset UUID (REQUIRED for provenance)
        unified_annotation_type: Primary annotation type in this Excel file (REQUIRED)
        process_row_fn: async function(row: Dict[str, Any], row_index: int) -> None
            - row: Dictionary with Excel columns as keys
            - row_index: 0-based row number
            - Can call get_current_provenance() to get raw_file_id and chain_id
        sheet_name: Sheet name or index (default: 0)
        progress_tracker: Optional progress tracker
        skip_errors: If True, log errors and continue; if False, raise
    
    Returns:
        Tuple of (OperationStatistics, raw_file_id, chain_id)
    
    Example:
        ```python
        from chaksudb.ingest.framework.provenance_context import get_current_provenance
        
        async def handle_patient_row(row, idx):
            patient_id = await register_patient(
                dataset_id=dataset_id,
                original_patient_id=row["PatientID"],
                age=row["Age"],
                sex=row["Sex"]
            )
        
        stats, raw_file_id, chain_id = await process_excel(
            excel_path=data_root / "patient_data.xlsx",
            dataset_id=dataset_id,
            unified_annotation_type="classification",
            process_row_fn=handle_patient_row,
            sheet_name="Patients"
        )
        ```
    """
    # 1. Register raw file and create provenance chain
    sheet_name_str = str(sheet_name) if isinstance(sheet_name, int) else sheet_name
    raw_file_id, chain_id = await register_excel_file(
        excel_path, dataset_id, unified_annotation_type, sheet_name_str
    )
    
    # 2. Set provenance context for callbacks
    token_raw, token_chain = set_provenance_context(raw_file_id, chain_id)
    
    stats = OperationStatistics()
    
    try:
        # 3. Read and process Excel
        rows = read_excel_sheet(excel_path, sheet=sheet_name)
        total_rows = len(rows)
        
        for idx, row in enumerate(rows):
            try:
                await process_row_fn(row, idx)
                stats.successful_items += 1
                    
            except Exception as e:
                stats.failed_items += 1
                stats.errors.append(f"Row {idx}: {str(e)}")
                logger.error(f"Error processing row {idx}: {e}")
                
                if not skip_errors:
                    raise
        
        # 4. Update raw file status
        final_status = "parsed" if stats.failed_items == 0 else "error"
        await update_raw_file_status(raw_file_id, final_status, stats.errors)
        
        logger.debug(
            f"Excel processing complete: {stats.successful_items} success, "
            f"{stats.failed_items} errors"
        )
        
    except Exception as e:
        # Mark as error on exception
        await update_raw_file_status(raw_file_id, "error", [str(e)])
        logger.error(f"Failed to process Excel {excel_path}: {e}")
        if not skip_errors:
            raise
    finally:
        # 5. Always cleanup provenance context
        reset_provenance_context(token_raw, token_chain)
    
    return stats, raw_file_id, chain_id


# ============================================
# Generic JSON Processing
# ============================================


async def process_json(
    json_path: Path,
    dataset_id: uuid.UUID,
    unified_annotation_type: str,
    process_entry_fn: Callable[[Any, int], Awaitable[None]],
    progress_tracker: Optional[ProgressTracker] = None,
    skip_errors: bool = True,
) -> Tuple[OperationStatistics, uuid.UUID, uuid.UUID]:
    """
    Generic JSON processor with automatic provenance tracking.
    
    Reads JSON file, registers it as a raw annotation file, and calls your
    function for each entry. Provenance information is automatically available
    to callbacks via get_current_provenance().
    
    Works with:
    - JSON arrays: [{"key": "value"}, ...]
    - JSON objects: {"entry1": {...}, "entry2": {...}}
    - JSONL: One JSON object per line
    
    Args:
        json_path: Path to JSON file
        dataset_id: Dataset UUID (REQUIRED for provenance)
        unified_annotation_type: Primary annotation type in this JSON file (REQUIRED)
        process_entry_fn: async function(entry: Any, index: int) -> None
            - entry: Each item from array, or (key, value) tuple from object
            - index: 0-based entry number
            - Can call get_current_provenance() to get raw_file_id and chain_id
        progress_tracker: Optional progress tracker
        skip_errors: If True, log errors and continue; if False, raise
    
    Returns:
        Tuple of (OperationStatistics, raw_file_id, chain_id)
    
    Example:
        ```python
        from chaksudb.ingest.framework.provenance_context import get_current_provenance
        
        async def handle_json_entry(entry, idx):
            # For DeepEyeNet: entry = (image_path, metadata_dict)
            image_path, metadata = entry
            
            raw_data_id, provenance_chain_id = get_current_provenance()
            
            await process_keywords(
                metadata["keywords"],
                raw_data_id=raw_data_id,
                provenance_chain_id=provenance_chain_id
            )
        
        stats, raw_file_id, chain_id = await process_json(
            json_path=data_root / "metadata.json",
            dataset_id=dataset_id,
            unified_annotation_type="keyword",
            process_entry_fn=handle_json_entry
        )
        ```
    """
    # 1. Register raw file and create provenance chain
    raw_file_id, chain_id = await register_json_file(
        json_path, dataset_id, unified_annotation_type
    )
    
    # 2. Set provenance context for callbacks
    token_raw, token_chain = set_provenance_context(raw_file_id, chain_id)
    
    stats = OperationStatistics()
    
    try:
        # 3. Read and process JSON
        data = read_json_file(json_path)
        
        # Convert to iterable
        if isinstance(data, list):
            entries = list(enumerate(data))
            total = len(data)
        elif isinstance(data, dict):
            entries = list(enumerate(data.items()))
            total = len(data)
        else:
            raise ValueError(f"Unsupported JSON structure: {type(data)}")
        
        for idx, entry in entries:
            try:
                await process_entry_fn(entry, idx)
                stats.successful_items += 1
                    
            except Exception as e:
                stats.failed_items += 1
                stats.errors.append(f"Entry {idx}: {str(e)}")
                logger.error(f"Error processing entry {idx}: {e}")
                
                if not skip_errors:
                    raise
        
        # 4. Update raw file status
        final_status = "parsed" if stats.failed_items == 0 else "error"
        await update_raw_file_status(raw_file_id, final_status, stats.errors)
        
        logger.debug(
            f"JSON processing complete: {stats.successful_items} success, "
            f"{stats.failed_items} errors"
        )
        
    except Exception as e:
        # Mark as error on exception
        await update_raw_file_status(raw_file_id, "error", [str(e)])
        logger.error(f"Failed to process JSON {json_path}: {e}")
        if not skip_errors:
            raise
    finally:
        # 5. Always cleanup provenance context
        reset_provenance_context(token_raw, token_chain)
    
    return stats, raw_file_id, chain_id


# ============================================
# Generic Text File Processing
# ============================================


async def process_text_file(
    text_path: Path,
    process_line_fn: Callable[[str, int], Awaitable[None]],
    skip_empty: bool = True,
    skip_comments: bool = True,
    comment_char: str = "#",
    progress_tracker: Optional[ProgressTracker] = None,
    skip_errors: bool = True,
) -> OperationStatistics:
    """
    Generic text file processor - reads text file and calls your function per line.
    
    Args:
        text_path: Path to text file
        process_line_fn: async function(line: str, line_number: int) -> None
            - line: Text line (stripped of whitespace)
            - line_number: 1-based line number
        skip_empty: Skip empty lines
        skip_comments: Skip comment lines
        comment_char: Comment character (default: "#")
        progress_tracker: Optional progress tracker
        skip_errors: If True, log errors and continue; if False, raise
    
    Returns:
        OperationStatistics with success/error counts
    
    Example:
        ```python
        async def handle_line(line, line_num):
            # For OIA-DDR train.txt: "filename.jpg grade"
            parts = line.split()
            filename, grade = parts[0], int(parts[1])
            await process_image_with_grade(filename, grade)
        
        stats = await process_text_file(
            text_path=data_root / "train.txt",
            process_line_fn=handle_line
        )
        ```
    """
    stats = OperationStatistics()
    
    try:
        with open(text_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        for line_num, line in enumerate(lines, start=1):
            line = line.strip()
            
            # Skip based on criteria
            if skip_empty and not line:
                continue
            if skip_comments and line.startswith(comment_char):
                continue
            
            try:
                await process_line_fn(line, line_num)
                stats.successful_items += 1
                    
            except Exception as e:
                stats.failed_items += 1
                stats.errors.append(f"Line {line_num}: {str(e)}")
                logger.error(f"Error processing line {line_num}: {e}")
                
                if not skip_errors:
                    raise
        
        logger.debug(
            f"Text file processing complete: {stats.successful_items} success, "
            f"{stats.failed_items} errors"
        )
        
    except Exception as e:
        logger.error(f"Failed to process text file {text_path}: {e}")
        if not skip_errors:
            raise
    
    return stats


# ============================================
# Generic Folder Walking
# ============================================


async def process_folder_tree(
    root_dir: Path,
    dataset_id: uuid.UUID,
    unified_annotation_type: str,
    process_file_fn: Callable[[Path, Path, int], Awaitable[None]],
    file_extensions: Optional[Set[str]] = None,
    recursive: bool = True,
    include_dirs: bool = False,
    progress_tracker: Optional[ProgressTracker] = None,
    skip_errors: bool = True,
) -> OperationStatistics:
    """
    Generic folder walker with per-file provenance tracking.
    
    Walks directory tree, registers EACH file as a raw annotation file, and
    calls your function per file. Each file gets its own provenance chain,
    which is automatically available to callbacks via get_current_provenance().
    
    Args:
        root_dir: Root directory to walk
        dataset_id: Dataset UUID (REQUIRED for provenance)
        unified_annotation_type: Annotation type (REQUIRED, usually "segmentation")
        process_file_fn: async function(file_path: Path, relative_path: Path, depth: int) -> None
            - file_path: Absolute path to file
            - relative_path: Path relative to root_dir
            - depth: Directory depth (0 = root)
            - Can call get_current_provenance() to get raw_file_id and chain_id
        file_extensions: Only process files with these extensions (e.g., {".jpg", ".png"})
        recursive: Walk subdirectories recursively
        include_dirs: Also call function for directories
        progress_tracker: Optional progress tracker
        skip_errors: If True, log errors and continue; if False, raise
    
    Returns:
        OperationStatistics with success/error counts
    
    Example:
        ```python
        from chaksudb.ingest.framework.provenance_context import get_current_provenance
        
        async def handle_mask(file_path, rel_path, depth):
            # Extract lesion type from filename or path
            lesion_type = extract_lesion_type(file_path)
            
            # Provenance automatically available
            raw_data_id, provenance_chain_id = get_current_provenance()
            
            await process_segmentation_from_binary_mask(
                mask_path=file_path,
                annotation_type=lesion_type,
                image_id=image_id,
                raw_data_id=raw_data_id,
                provenance_chain_id=provenance_chain_id,
            )
        
        stats = await process_folder_tree(
            root_dir=data_root / "Masks",
            dataset_id=dataset_id,
            unified_annotation_type="segmentation",
            process_file_fn=handle_mask,
            file_extensions={".png", ".tif"},
            recursive=True
        )
        ```
    """
    stats = OperationStatistics()
    
    # Collect all files first for progress tracking
    all_files = []
    for item in root_dir.rglob("*") if recursive else root_dir.glob("*"):
        if item.is_file():
            if file_extensions is None or item.suffix.lower() in file_extensions:
                all_files.append(item)
        elif include_dirs and item.is_dir():
            all_files.append(item)
    
    for file_path in all_files:
        raw_file_id = None
        try:
            # 1. Register THIS specific file
            # For image files, set auto_detect_type=False to leave file_type as NULL
            # (image files are not annotation files and shouldn't violate ck_raw_files_type)
            from chaksudb.ingest.framework.file_types import is_image_file
            is_image = is_image_file(file_path)
            
            raw_file_id, chain_id = await register_individual_file(
                file_path=file_path,
                dataset_id=dataset_id,
                unified_annotation_type=unified_annotation_type,
                auto_detect_type=not is_image,  # Don't auto-detect type for images
            )
            
            # 2. Set provenance context for this file
            token_raw, token_chain = set_provenance_context(raw_file_id, chain_id)
            
            try:
                # 3. Process the file
                relative_path = file_path.relative_to(root_dir)
                depth = len(relative_path.parts) - 1
                
                await process_file_fn(file_path, relative_path, depth)
                stats.successful_items += 1
                
                # 4. Mark this file as successfully parsed
                await update_raw_file_status(raw_file_id, "parsed", None)
                
            except Exception as e:
                stats.failed_items += 1
                error_msg = f"{file_path}: {str(e)}"
                stats.errors.append(error_msg)
                
                # Mark this file as error
                await update_raw_file_status(raw_file_id, "error", [str(e)])
                
                logger.error(f"Error processing {file_path}: {e}")
                
                if not skip_errors:
                    raise
            finally:
                # 5. Reset context after each file
                reset_provenance_context(token_raw, token_chain)
                
        except Exception as e:
            stats.failed_items += 1
            stats.errors.append(f"{file_path}: {str(e)}")
            logger.error(f"Error processing {file_path}: {e}")
            
            # Try to mark as error if we got a raw_file_id
            if raw_file_id:
                try:
                    await update_raw_file_status(raw_file_id, "error", [str(e)])
                except Exception:
                    pass  # Already in error handling, don't propagate
            
            if not skip_errors:
                raise
    
    logger.debug(
        f"Folder tree processing complete: {stats.successful_items} success, "
        f"{stats.failed_items} errors"
    )
    
    return stats


# ============================================
# Generic File Matching/Pairing
# ============================================


async def process_paired_files(
    primary_dir: Path,
    secondary_dir: Path,
    process_pair_fn: Callable[[Path, Optional[Path]], Awaitable[None]],
    primary_extensions: Optional[Set[str]] = None,
    secondary_extensions: Optional[Set[str]] = None,
    match_by: str = "stem",  # "stem", "name", or custom function
    require_secondary: bool = False,
    progress_tracker: Optional[ProgressTracker] = None,
    skip_errors: bool = True,
) -> OperationStatistics:
    """
    Generic file pairing - matches files between two directories and processes pairs.
    
    Useful for:
    - Image + mask pairs
    - Image + XML annotation pairs
    - Image + segmentation pairs
    - Left + right eye pairs
    
    Args:
        primary_dir: Directory with primary files (e.g., images)
        secondary_dir: Directory with secondary files (e.g., masks, XMLs)
        process_pair_fn: async function(primary_path: Path, secondary_path: Optional[Path]) -> None
        primary_extensions: Filter primary files by extensions
        secondary_extensions: Filter secondary files by extensions
        match_by: How to match files:
            - "stem": Match by filename stem (without extension)
            - "name": Match by full filename
            - Custom function: fn(primary_path) -> secondary_filename
        require_secondary: If True, skip primary files without matches
        progress_tracker: Optional progress tracker
        skip_errors: If True, log errors and continue; if False, raise
    
    Returns:
        OperationStatistics with success/error counts
    
    Example:
        ```python
        async def handle_image_mask_pair(image_path, mask_path):
            if mask_path:
                # Process segmentation
                await process_segmentation(image_path, mask_path)
            else:
                # Image only, no mask
                await process_image_only(image_path)
        
        stats = await process_paired_files(
            primary_dir=data_root / "images",
            secondary_dir=data_root / "masks",
            process_pair_fn=handle_image_mask_pair,
            primary_extensions={".jpg"},
            secondary_extensions={".png"},
            match_by="stem"
        )
        ```
    """
    stats = OperationStatistics()
    
    # Build index of secondary files
    secondary_index: Dict[str, Path] = {}
    if secondary_dir.exists():
        for sec_file in secondary_dir.rglob("*") if secondary_dir.is_dir() else [secondary_dir]:
            if sec_file.is_file():
                if secondary_extensions is None or sec_file.suffix.lower() in secondary_extensions:
                    if match_by == "stem":
                        key = sec_file.stem
                    elif match_by == "name":
                        key = sec_file.name
                    else:
                        key = sec_file.stem  # Default
                    secondary_index[key] = sec_file
    
    # Process primary files
    primary_files = []
    for prim_file in primary_dir.rglob("*") if primary_dir.is_dir() else [primary_dir]:
        if prim_file.is_file():
            if primary_extensions is None or prim_file.suffix.lower() in primary_extensions:
                primary_files.append(prim_file)
    
    for prim_path in primary_files:
        try:
            # Find matching secondary file
            if match_by == "stem":
                search_key = prim_path.stem
            elif match_by == "name":
                search_key = prim_path.name
            elif callable(match_by):
                search_key = match_by(prim_path)
            else:
                search_key = prim_path.stem
            
            sec_path = secondary_index.get(search_key)
            
            # Skip if secondary required but not found
            if require_secondary and sec_path is None:
                stats.skipped_items += 1
                logger.debug(f"Skipping {prim_path.name}: no matching secondary file")
                continue
            
            await process_pair_fn(prim_path, sec_path)
            stats.successful_items += 1
                
        except Exception as e:
            stats.failed_items += 1
            stats.errors.append(f"{prim_path}: {str(e)}")
            logger.error(f"Error processing {prim_path}: {e}")
            
            if not skip_errors:
                raise
    
    logger.debug(
        f"Paired files processing complete: {stats.successful_items} success, "
        f"{stats.failed_items} errors, {stats.skipped_items} skipped"
    )
    
    return stats


# ============================================
# Utility: Find File by Pattern
# ============================================


def find_file_for_stem(
    file_stem: str,
    search_dir: Path,
    extensions: Optional[Set[str]] = None,
) -> Optional[Path]:
    """
    Find a file in directory matching the given stem.
    
    Args:
        file_stem: Filename stem (without extension)
        search_dir: Directory to search in
        extensions: Allowed extensions (e.g., {".jpg", ".png"})
    
    Returns:
        Path to matched file, or None if not found
    
    Example:
        ```python
        # Find image for "train_001" - could be .jpg, .png, etc.
        image_path = find_file_for_stem(
            file_stem="train_001",
            search_dir=image_dir,
            extensions={".jpg", ".jpeg", ".png"}
        )
        ```
    """
    if not search_dir.exists():
        return None
    
    for file in search_dir.iterdir():
        if file.is_file() and file.stem == file_stem:
            if extensions is None or file.suffix.lower() in extensions:
                return file
    
    return None
