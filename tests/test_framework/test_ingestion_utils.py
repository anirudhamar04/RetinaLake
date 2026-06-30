"""
Tests for ingestion utility functions.

Tests the simple utility functions with real dataset examples.
"""

import pytest
from pathlib import Path

from chaksudb.ingest.framework import (
    # Annotation I/O
    read_csv_auto,
    read_excel_sheet,
    read_json_file,
    # File finder
    find_images,
    find_files_by_extension,
    find_matching_file,
    # Filename utils
    extract_laterality,
    # Folder utils
    get_split_folders,
    get_immediate_subdirs,
)


# ============================================
# Filename Utils Tests
# ============================================


def test_extract_laterality():
    """Test laterality extraction from various filename patterns."""
    # Common patterns
    assert extract_laterality("patient_123_left.jpg") == "left"
    assert extract_laterality("patient_123_right.jpg") == "right"
    assert extract_laterality("0001_l.png") == "left"
    assert extract_laterality("0001_r.png") == "right"
    
    # Medical notation
    assert extract_laterality("RET004OS.jpg") == "left"  # OS = left eye
    assert extract_laterality("RET004OD.jpg") == "right"  # OD = right eye
    
    # ODIR style
    assert extract_laterality("1063_left.jpg") == "left"
    assert extract_laterality("1063_right.jpg") == "right"
    
    # DeepDRiD style
    assert extract_laterality("133_l1.jpg") == "left"
    assert extract_laterality("133_r2.jpg") == "right"
    
    # No laterality
    assert extract_laterality("image.jpg") is None
    assert extract_laterality("patient_123.jpg") is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
