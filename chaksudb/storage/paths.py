"""
Path management utilities for storage operations.

This module provides utilities for path computation, normalization, and storage
directory management. It handles cross-platform path operations and integrates
with the storage configuration.
"""

import logging
import os
from pathlib import Path
from typing import Optional

from chaksudb.config.config import storage_config

logger = logging.getLogger(__name__)


def normalize_path(path: str) -> str:
    """
    Normalize a path string for cross-platform compatibility.
    
    This function converts Windows-style backslashes to forward slashes
    and normalizes the path string. This ensures consistent path representation
    across different operating systems.
    
    Args:
        path: Path string to normalize
    
    Returns:
        Normalized path string with forward slashes
    """
    if not path:
        return path
    
    # Convert backslashes to forward slashes (cross-platform)
    normalized = path.replace("\\", "/")
    
    # Remove duplicate slashes (except for protocol prefixes like http://)
    if "://" in normalized:
        parts = normalized.split("://", 1)
        normalized = parts[0] + "://" + parts[1].replace("//", "/")
    else:
        normalized = normalized.replace("//", "/")
    
    return normalized


def compute_relative_path(absolute_path: Path, dataset_root: Path) -> str:
    """
    Compute relative path from absolute path relative to dataset root.
    
    This function converts an absolute file path to a relative path
    based on the dataset root directory. The resulting path is normalized
    for cross-platform compatibility.
    
    Args:
        absolute_path: Absolute path to the file
        dataset_root: Root directory of the dataset
    
    Returns:
        Relative path string (normalized with forward slashes)
    
    Raises:
        ValueError: If absolute_path is not within dataset_root
    """
    try:
        # Resolve both paths to absolute paths
        abs_path = absolute_path.resolve()
        root_path = dataset_root.resolve()
        
        # Compute relative path
        try:
            relative = abs_path.relative_to(root_path)
        except ValueError:
            raise ValueError(
                f"Path {absolute_path} is not within dataset root {dataset_root}"
            )
        
        # Normalize to use forward slashes
        return normalize_path(str(relative))
    
    except Exception as e:
        logger.error(
            f"Failed to compute relative path from {absolute_path} "
            f"relative to {dataset_root}: {e}"
        )
        raise


def get_storage_root() -> Path:
    """
    Get the root directory for local file storage.
    
    This is a convenience function that wraps the config function.
    
    Returns:
        Path to storage root directory
    """
    return storage_config.local_root


def ensure_storage_directory(path: Path) -> None:
    """
    Ensure a storage directory exists, creating it if necessary.
    
    This function creates the directory (and any parent directories) if it
    doesn't exist. If the path already exists and is a directory, no action
    is taken. If the path exists but is a file, an error is raised.
    
    Args:
        path: Directory path to ensure exists
    
    Raises:
        ValueError: If path exists but is a file
        OSError: If directory creation fails
    """
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"Path exists but is not a directory: {path}")
        return
    
    try:
        path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created storage directory: {path}")
    except OSError as e:
        logger.error(f"Failed to create storage directory {path}: {e}")
        raise


def generate_storage_path(
    dataset_name: str,
    subdirectory: Optional[str] = None,
    filename: Optional[str] = None,
) -> Path:
    """
    Generate a storage path for processed/converted files.
    
    This function generates a path within the storage root directory
    organized by dataset name and optional subdirectory. Useful for
    storing converted files (e.g., normalized masks, processed images).
    
    Args:
        dataset_name: Name of the dataset
        subdirectory: Optional subdirectory (e.g., 'masks', 'processed')
        filename: Optional filename (if None, returns directory path)
    
    Returns:
        Path object for the generated storage location
    
    Example:
        >>> path = generate_storage_path("EYEPACS", "masks", "mask_123.png")
        >>> # Returns: storage/EYEPACS/masks/mask_123.png
    """
    storage_root = get_storage_root()
    
    # Build path components
    components = [storage_root, dataset_name]
    if subdirectory:
        components.append(subdirectory)
    if filename:
        components.append(filename)
    
    # Create path
    storage_path = Path(*components)
    
    # Ensure parent directory exists
    if storage_path.parent != storage_root:
        ensure_storage_directory(storage_path.parent)
    
    return storage_path


def resolve_storage_path(relative_path: str, dataset_root: Optional[Path] = None) -> Path:
    """
    Resolve a relative path to an absolute path.
    
    This function converts a relative path (stored in database) back to
    an absolute path. If dataset_root is provided, the path is resolved
    relative to it. Otherwise, it's resolved relative to the current
    working directory.
    
    Args:
        relative_path: Relative path string (from database)
        dataset_root: Optional dataset root directory
    
    Returns:
        Absolute Path object
    """
    if dataset_root:
        return (dataset_root / relative_path).resolve()
    else:
        return Path(relative_path).resolve()


def get_file_size(file_path: Path) -> int:
    """
    Get file size in bytes.
    
    Args:
        file_path: Path to the file
    
    Returns:
        File size in bytes
    
    Raises:
        FileNotFoundError: If file does not exist
        OSError: If file cannot be accessed
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if not file_path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")
    
    return file_path.stat().st_size


