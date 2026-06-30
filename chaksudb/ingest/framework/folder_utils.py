"""
Simple folder structure utilities for navigating dataset directories.

Provides basic operations for working with dataset folder structures.
"""

import logging
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)


def get_split_folders(root_dir: Path) -> Dict[str, Path]:
    """
    Find train/test/val split folders if they exist.
    
    Looks for common split folder names (case-insensitive):
    - train, training, Train, TRAIN
    - test, testing, Test, TEST
    - val, validation, valid, Val, Validation
    
    Args:
        root_dir: Root directory to search for split folders
        
    Returns:
        Dictionary mapping split names to paths.
        Keys are normalized: 'train', 'test', 'val'
        Returns empty dict if no split folders found.
        
    Example:
        >>> splits = get_split_folders(Path("data/dataset"))
        >>> if 'train' in splits:
        ...     train_images = find_images(splits['train'])
        >>> # Returns: {'train': Path('data/dataset/train'), 
        >>> #           'test': Path('data/dataset/test')}
    """
    if not root_dir.exists():
        logger.warning(f"Root directory not found: {root_dir}")
        return {}
    
    if not root_dir.is_dir():
        logger.warning(f"Path is not a directory: {root_dir}")
        return {}
    
    splits = {}
    
    # Define mappings from folder names to normalized split names
    train_names = ['train', 'training', 'Train', 'TRAIN', 'Training']
    test_names = ['test', 'testing', 'Test', 'TEST', 'Testing']
    val_names = ['val', 'validation', 'valid', 'Val', 'Validation', 'Valid']
    
    # Check immediate subdirectories
    for item in root_dir.iterdir():
        if item.is_dir():
            dir_name = item.name
            
            # Check for train
            if dir_name in train_names and 'train' not in splits:
                splits['train'] = item
                logger.debug(f"Found train split: {item}")
            
            # Check for test
            elif dir_name in test_names and 'test' not in splits:
                splits['test'] = item
                logger.debug(f"Found test split: {item}")
            
            # Check for val
            elif dir_name in val_names and 'val' not in splits:
                splits['val'] = item
                logger.debug(f"Found val split: {item}")
    
    if splits:
        logger.debug(f"Found {len(splits)} split folders in {root_dir}: {list(splits.keys())}")
    else:
        logger.debug(f"No split folders found in {root_dir}")
    
    return splits


def get_immediate_subdirs(directory: Path) -> List[Path]:
    """
    Get list of immediate subdirectories (not recursive).
    
    Returns only directories, not files, in alphabetical order.
    
    Args:
        directory: Directory to list
        
    Returns:
        List of immediate subdirectory paths, sorted alphabetically
        
    Example:
        >>> subdirs = get_immediate_subdirs(Path("data"))
        >>> for subdir in subdirs:
        ...     print(subdir.name)
        # Output: Patient_1, Patient_2, Patient_3
    """
    if not directory.exists():
        logger.warning(f"Directory not found: {directory}")
        return []
    
    if not directory.is_dir():
        logger.warning(f"Path is not a directory: {directory}")
        return []
    
    subdirs = [item for item in directory.iterdir() if item.is_dir()]
    subdirs.sort()
    
    logger.debug(f"Found {len(subdirs)} immediate subdirectories in {directory}")
    return subdirs
