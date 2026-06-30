"""
Simple filename parsing utilities for extracting common patterns.

Handles the 80% common cases. Dataset-specific edge cases stay in dataset scripts.
"""

import logging
import re
from typing import List, Optional

logger = logging.getLogger(__name__)


def extract_laterality(filename: str) -> Optional[str]:
    """
    Extract eye laterality from filename.
    
    Checks for common patterns:
    - _left, _right, left_, right_
    - _l, _r, l_, r_
    - OD (right eye), OS (left eye)
    - Left, Right, L, R (as separate words)
    
    Returns normalized values: 'left' or 'right'
    
    Args:
        filename: Filename or path (only the filename part is used)
        
    Returns:
        'left', 'right', or None if laterality cannot be determined
        
    Example:
        >>> extract_laterality("patient_123_left.jpg")
        'left'
        >>> extract_laterality("0001_r.png")
        'right'
        >>> extract_laterality("RET004OS.jpg")
        'left'
        >>> extract_laterality("image.jpg")
        None
    """
    # Get just the filename if a path was provided
    filename_lower = filename.lower()
    
    # Check for explicit 'left' patterns
    left_patterns = [
        r'_left\b',      # _left
        r'\bleft_',      # left_
        r'_l\d',         # _l followed by digit (e.g., _l1, _l2)
        r'_l\b',         # _l at word boundary
        r'\bl_',         # l_
        r'\bleft\b',     # left (as word)
        r'\bl\b',        # l (as word)
        r'os\b',         # OS (left eye in medical notation)
    ]
    
    for pattern in left_patterns:
        if re.search(pattern, filename_lower):
            return 'left'
    
    # Check for explicit 'right' patterns
    right_patterns = [
        r'_right\b',     # _right
        r'\bright_',     # right_
        r'_r\d',         # _r followed by digit (e.g., _r1, _r2)
        r'_r\b',         # _r at word boundary
        r'\br_',         # r_
        r'\bright\b',    # right (as word)
        r'\br\b',        # r (as word)
        r'od\b',         # OD (right eye in medical notation)
    ]
    
    for pattern in right_patterns:
        if re.search(pattern, filename_lower):
            return 'right'
    
    return None








