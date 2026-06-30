"""
Simple file finding utilities for locating images and matching files.

Provides basic file discovery and matching operations.
"""

import logging
from pathlib import Path
from typing import List, Optional

from chaksudb.ingest.framework.file_types import IMAGE_EXTENSIONS

logger = logging.getLogger(__name__)


def find_images(directory: Path, recursive: bool = True) -> List[Path]:
    """
    Find all image files in directory.
    
    Uses IMAGE_EXTENSIONS from file_types module to determine what counts
    as an image file. Supports both lowercase and uppercase extensions.
    
    Args:
        directory: Directory to search
        recursive: If True, search subdirectories recursively
        
    Returns:
        List of image file paths, sorted by name
        
    Raises:
        FileNotFoundError: If directory doesn't exist
        
    Example:
        >>> images = find_images(Path("data/train"))
        >>> print(f"Found {len(images)} images")
    """
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    if not directory.is_dir():
        raise ValueError(f"Path is not a directory: {directory}")
    
    images = []
    
    # Search for each image extension (case-insensitive)
    for ext in IMAGE_EXTENSIONS:
        if recursive:
            # Recursive search with ** glob
            images.extend(directory.glob(f"**/*{ext}"))
            # Also search uppercase variants
            images.extend(directory.glob(f"**/*{ext.upper()}"))
        else:
            # Non-recursive search
            images.extend(directory.glob(f"*{ext}"))
            images.extend(directory.glob(f"*{ext.upper()}"))
    
    # Remove duplicates and sort
    images = sorted(set(images))
    
    logger.debug(f"Found {len(images)} images in {directory} (recursive={recursive})")
    return images


def find_files_by_extension(
    directory: Path, 
    ext: str, 
    recursive: bool = True
) -> List[Path]:
    """
    Find all files with given extension in directory.
    
    Searches for both lowercase and uppercase variants of the extension.
    
    Args:
        directory: Directory to search
        ext: File extension (with or without leading dot, e.g., '.xml' or 'xml')
        recursive: If True, search subdirectories recursively
        
    Returns:
        List of file paths, sorted by name
        
    Raises:
        FileNotFoundError: If directory doesn't exist
        
    Example:
        >>> xml_files = find_files_by_extension(Path("annotations"), ".xml")
        >>> csv_files = find_files_by_extension(Path("labels"), "csv", recursive=False)
    """
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    if not directory.is_dir():
        raise ValueError(f"Path is not a directory: {directory}")
    
    # Ensure extension starts with a dot
    if not ext.startswith('.'):
        ext = f'.{ext}'
    
    files = []
    
    if recursive:
        # Recursive search
        files.extend(directory.glob(f"**/*{ext}"))
        files.extend(directory.glob(f"**/*{ext.upper()}"))
    else:
        # Non-recursive search
        files.extend(directory.glob(f"*{ext}"))
        files.extend(directory.glob(f"*{ext.upper()}"))
    
    # Remove duplicates and sort
    files = sorted(set(files))
    
    logger.debug(f"Found {len(files)} {ext} files in {directory} (recursive={recursive})")
    return files


def find_matching_file(
    source_file: Path,
    target_dir: Path,
    new_ext: Optional[str] = None,
    suffix: str = ""
) -> Optional[Path]:
    """
    Find file with matching stem in target directory.
    
    Useful for finding corresponding annotation/mask files for an image.
    Matches by filename stem (name without extension).
    
    Args:
        source_file: Source file to match from
        target_dir: Directory to search for matching file
        new_ext: New extension for target file (e.g., '.xml', '.png')
                 If None, keeps same extension as source
        suffix: Additional suffix to add before extension (e.g., '_mask', '_EX')
        
    Returns:
        Path to matching file if found, None otherwise
        
    Example:
        >>> # Find mask for image: image.jpg -> mask_dir/image_mask.png
        >>> mask = find_matching_file(
        ...     Path("images/0001.jpg"),
        ...     Path("masks"),
        ...     new_ext=".png",
        ...     suffix="_mask"
        ... )
        >>> 
        >>> # Find XML for image: image.jpg -> xml_dir/image.xml
        >>> xml = find_matching_file(
        ...     Path("images/0001.jpg"),
        ...     Path("annotations"),
        ...     new_ext=".xml"
        ... )
    """
    if not target_dir.exists():
        logger.warning(f"Target directory not found: {target_dir}")
        return None
    
    if not target_dir.is_dir():
        logger.warning(f"Target path is not a directory: {target_dir}")
        return None
    
    # Get the stem (filename without extension)
    stem = source_file.stem
    
    # Determine extension
    if new_ext is None:
        ext = source_file.suffix
    else:
        # Ensure extension starts with a dot
        ext = new_ext if new_ext.startswith('.') else f'.{new_ext}'
    
    # Build target filename
    target_filename = f"{stem}{suffix}{ext}"
    
    # Try exact match first
    target_path = target_dir / target_filename
    if target_path.exists():
        logger.debug(f"Found matching file: {target_path}")
        return target_path
    
    # Try case-insensitive search (useful for cross-platform datasets)
    # Search for files with matching stem (case-insensitive)
    for file in target_dir.iterdir():
        if file.is_file():
            # Check if stem and suffix match (case-insensitive)
            expected_stem_suffix = f"{stem}{suffix}".lower()
            actual_stem = file.stem.lower()
            
            if actual_stem == expected_stem_suffix:
                # Check if extension matches (case-insensitive)
                if file.suffix.lower() == ext.lower():
                    logger.debug(f"Found matching file (case-insensitive): {file}")
                    return file
    
    # Also search recursively if target_dir has subdirectories
    # This handles cases where annotations might be in nested folders
    pattern = f"**/{stem}{suffix}{ext}"
    matches = list(target_dir.glob(pattern))
    
    if matches:
        # Return first match
        logger.debug(f"Found matching file (recursive): {matches[0]}")
        return matches[0]
    
    # Try case-insensitive recursive search
    pattern_lower = f"**/{stem.lower()}{suffix.lower()}{ext.lower()}"
    for file in target_dir.rglob("*"):
        if file.is_file():
            file_parts = f"{file.stem}{file.suffix}".lower()
            expected_parts = f"{stem}{suffix}{ext}".lower()
            if file_parts == expected_parts:
                logger.debug(f"Found matching file (case-insensitive recursive): {file}")
                return file
    
    logger.debug(f"No matching file found for {source_file.name} in {target_dir}")
    return None
