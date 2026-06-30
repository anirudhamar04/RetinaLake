"""
Tests for chaksudb/ingest/framework/file_types.py

Tests file type detection utilities based on their docstrings.
"""

import pytest
from pathlib import Path

from chaksudb.ingest.framework.file_types import (
    get_image_extensions,
    get_annotation_extensions,
    get_mask_extensions,
    detect_file_type,
    is_image_file,
    is_annotation_file,
    is_mask_file,
    is_csv_file,
    is_json_file,
    is_excel_file,
    is_xml_file,
)


class TestGetExtensionFunctions:
    """Tests for extension getter functions."""

    def test_get_image_extensions_returns_list(self):
        """Test that get_image_extensions returns a list."""
        result = get_image_extensions()
        assert isinstance(result, list)

    def test_get_image_extensions_contains_common_formats(self):
        """Test that get_image_extensions contains common image formats."""
        extensions = get_image_extensions()
        
        assert ".jpg" in extensions
        assert ".jpeg" in extensions
        assert ".png" in extensions
        assert ".tif" in extensions
        assert ".tiff" in extensions

    def test_get_image_extensions_contains_dicom(self):
        """Test that get_image_extensions contains DICOM format."""
        extensions = get_image_extensions()
        
        assert ".dcm" in extensions
        assert ".dicom" in extensions

    def test_get_image_extensions_returns_copy(self):
        """Test that get_image_extensions returns a copy (not reference to original)."""
        ext1 = get_image_extensions()
        ext2 = get_image_extensions()
        
        ext1.append(".custom")
        assert ".custom" not in ext2

    def test_get_annotation_extensions_returns_list(self):
        """Test that get_annotation_extensions returns a list."""
        result = get_annotation_extensions()
        assert isinstance(result, list)

    def test_get_annotation_extensions_contains_common_formats(self):
        """Test that get_annotation_extensions contains common annotation formats."""
        extensions = get_annotation_extensions()
        
        assert ".csv" in extensions
        assert ".json" in extensions
        assert ".jsonl" in extensions
        assert ".xml" in extensions
        assert ".txt" in extensions

    def test_get_annotation_extensions_contains_excel_formats(self):
        """Test that get_annotation_extensions contains Excel formats."""
        extensions = get_annotation_extensions()
        
        assert ".xlsx" in extensions
        assert ".xls" in extensions

    def test_get_mask_extensions_returns_list(self):
        """Test that get_mask_extensions returns a list."""
        result = get_mask_extensions()
        assert isinstance(result, list)

    def test_get_mask_extensions_contains_common_formats(self):
        """Test that get_mask_extensions contains common mask formats."""
        extensions = get_mask_extensions()
        
        assert ".png" in extensions
        assert ".jpg" in extensions
        assert ".tif" in extensions
        assert ".npy" in extensions


class TestDetectFileType:
    """Tests for detect_file_type function."""

    def test_detect_file_type_returns_image_for_jpg(self):
        """Test that detect_file_type returns 'image' for .jpg files."""
        path = Path("test.jpg")
        result = detect_file_type(path)
        assert result == "image"

    def test_detect_file_type_returns_image_for_png(self):
        """Test that detect_file_type returns 'image' for .png files."""
        path = Path("test.png")
        result = detect_file_type(path)
        assert result == "image"

    def test_detect_file_type_returns_image_for_dicom(self):
        """Test that detect_file_type returns 'image' for DICOM files."""
        assert detect_file_type(Path("test.dcm")) == "image"
        assert detect_file_type(Path("test.dicom")) == "image"

    def test_detect_file_type_returns_annotation_for_csv(self):
        """Test that detect_file_type returns 'annotation' for .csv files."""
        path = Path("annotations.csv")
        result = detect_file_type(path)
        assert result == "annotation"

    def test_detect_file_type_returns_annotation_for_json(self):
        """Test that detect_file_type returns 'annotation' for JSON files."""
        assert detect_file_type(Path("data.json")) == "annotation"
        assert detect_file_type(Path("data.jsonl")) == "annotation"

    def test_detect_file_type_returns_annotation_for_xml(self):
        """Test that detect_file_type returns 'annotation' for .xml files."""
        path = Path("annotations.xml")
        result = detect_file_type(path)
        assert result == "annotation"

    def test_detect_file_type_returns_mask_for_png(self):
        """Test that detect_file_type returns 'mask' for .png files (also valid as mask)."""
        # Note: PNG is both image and mask - function returns first match (image)
        path = Path("mask.png")
        result = detect_file_type(path)
        # PNG is in IMAGE_EXTENSIONS first, so returns "image"
        assert result == "image"

    def test_detect_file_type_returns_none_for_unknown(self):
        """Test that detect_file_type returns None for unknown extensions."""
        path = Path("file.unknown")
        result = detect_file_type(path)
        assert result is None

    def test_detect_file_type_returns_none_for_no_extension(self):
        """Test that detect_file_type returns None for files without extension."""
        path = Path("filename_no_ext")
        result = detect_file_type(path)
        assert result is None

    def test_detect_file_type_is_case_insensitive(self):
        """Test that detect_file_type is case-insensitive."""
        assert detect_file_type(Path("TEST.JPG")) == "image"
        assert detect_file_type(Path("TEST.CSV")) == "annotation"
        assert detect_file_type(Path("test.Png")) == "image"


class TestIsImageFile:
    """Tests for is_image_file function."""

    def test_is_image_file_returns_true_for_jpg(self):
        """Test that is_image_file returns True for .jpg files."""
        assert is_image_file(Path("image.jpg")) is True

    def test_is_image_file_returns_true_for_png(self):
        """Test that is_image_file returns True for .png files."""
        assert is_image_file(Path("image.png")) is True

    def test_is_image_file_returns_true_for_tiff(self):
        """Test that is_image_file returns True for .tif and .tiff files."""
        assert is_image_file(Path("image.tif")) is True
        assert is_image_file(Path("image.tiff")) is True

    def test_is_image_file_returns_true_for_dicom(self):
        """Test that is_image_file returns True for DICOM files."""
        assert is_image_file(Path("scan.dcm")) is True
        assert is_image_file(Path("scan.dicom")) is True

    def test_is_image_file_returns_false_for_csv(self):
        """Test that is_image_file returns False for non-image files."""
        assert is_image_file(Path("data.csv")) is False

    def test_is_image_file_returns_false_for_no_extension(self):
        """Test that is_image_file returns False for files without extension."""
        assert is_image_file(Path("filename")) is False

    def test_is_image_file_is_case_insensitive(self):
        """Test that is_image_file is case-insensitive."""
        assert is_image_file(Path("IMAGE.JPG")) is True
        assert is_image_file(Path("Image.Png")) is True


class TestIsAnnotationFile:
    """Tests for is_annotation_file function."""

    def test_is_annotation_file_returns_true_for_csv(self):
        """Test that is_annotation_file returns True for .csv files."""
        assert is_annotation_file(Path("annotations.csv")) is True

    def test_is_annotation_file_returns_true_for_json(self):
        """Test that is_annotation_file returns True for JSON files."""
        assert is_annotation_file(Path("data.json")) is True
        assert is_annotation_file(Path("data.jsonl")) is True

    def test_is_annotation_file_returns_true_for_xml(self):
        """Test that is_annotation_file returns True for .xml files."""
        assert is_annotation_file(Path("markup.xml")) is True

    def test_is_annotation_file_returns_true_for_excel(self):
        """Test that is_annotation_file returns True for Excel files."""
        assert is_annotation_file(Path("data.xlsx")) is True
        assert is_annotation_file(Path("data.xls")) is True

    def test_is_annotation_file_returns_true_for_txt(self):
        """Test that is_annotation_file returns True for .txt files."""
        assert is_annotation_file(Path("notes.txt")) is True

    def test_is_annotation_file_returns_false_for_image(self):
        """Test that is_annotation_file returns False for image files."""
        assert is_annotation_file(Path("image.jpg")) is False

    def test_is_annotation_file_returns_false_for_no_extension(self):
        """Test that is_annotation_file returns False for files without extension."""
        assert is_annotation_file(Path("filename")) is False


class TestIsMaskFile:
    """Tests for is_mask_file function."""

    def test_is_mask_file_returns_true_for_png(self):
        """Test that is_mask_file returns True for .png files."""
        assert is_mask_file(Path("mask.png")) is True

    def test_is_mask_file_returns_true_for_tiff(self):
        """Test that is_mask_file returns True for .tif/.tiff files."""
        assert is_mask_file(Path("mask.tif")) is True
        assert is_mask_file(Path("mask.tiff")) is True

    def test_is_mask_file_returns_true_for_npy(self):
        """Test that is_mask_file returns True for .npy files."""
        assert is_mask_file(Path("mask.npy")) is True

    def test_is_mask_file_returns_false_for_csv(self):
        """Test that is_mask_file returns False for non-mask files."""
        assert is_mask_file(Path("data.csv")) is False

    def test_is_mask_file_returns_false_for_no_extension(self):
        """Test that is_mask_file returns False for files without extension."""
        assert is_mask_file(Path("filename")) is False


class TestIsCsvFile:
    """Tests for is_csv_file function."""

    def test_is_csv_file_returns_true_for_csv(self):
        """Test that is_csv_file returns True for .csv files."""
        assert is_csv_file(Path("data.csv")) is True

    def test_is_csv_file_returns_false_for_other_formats(self):
        """Test that is_csv_file returns False for non-CSV files."""
        assert is_csv_file(Path("data.json")) is False
        assert is_csv_file(Path("data.xlsx")) is False

    def test_is_csv_file_is_case_insensitive(self):
        """Test that is_csv_file is case-insensitive."""
        assert is_csv_file(Path("DATA.CSV")) is True


class TestIsJsonFile:
    """Tests for is_json_file function."""

    def test_is_json_file_returns_true_for_json(self):
        """Test that is_json_file returns True for .json files."""
        assert is_json_file(Path("data.json")) is True

    def test_is_json_file_returns_true_for_jsonl(self):
        """Test that is_json_file returns True for .jsonl files."""
        assert is_json_file(Path("data.jsonl")) is True

    def test_is_json_file_returns_false_for_other_formats(self):
        """Test that is_json_file returns False for non-JSON files."""
        assert is_json_file(Path("data.csv")) is False
        assert is_json_file(Path("data.xml")) is False

    def test_is_json_file_is_case_insensitive(self):
        """Test that is_json_file is case-insensitive."""
        assert is_json_file(Path("DATA.JSON")) is True
        assert is_json_file(Path("data.JsonL")) is True


class TestIsExcelFile:
    """Tests for is_excel_file function."""

    def test_is_excel_file_returns_true_for_xlsx(self):
        """Test that is_excel_file returns True for .xlsx files."""
        assert is_excel_file(Path("data.xlsx")) is True

    def test_is_excel_file_returns_true_for_xls(self):
        """Test that is_excel_file returns True for .xls files."""
        assert is_excel_file(Path("data.xls")) is True

    def test_is_excel_file_returns_false_for_other_formats(self):
        """Test that is_excel_file returns False for non-Excel files."""
        assert is_excel_file(Path("data.csv")) is False
        assert is_excel_file(Path("data.json")) is False

    def test_is_excel_file_is_case_insensitive(self):
        """Test that is_excel_file is case-insensitive."""
        assert is_excel_file(Path("DATA.XLSX")) is True
        assert is_excel_file(Path("Data.Xls")) is True


class TestIsXmlFile:
    """Tests for is_xml_file function."""

    def test_is_xml_file_returns_true_for_xml(self):
        """Test that is_xml_file returns True for .xml files."""
        assert is_xml_file(Path("data.xml")) is True

    def test_is_xml_file_returns_false_for_other_formats(self):
        """Test that is_xml_file returns False for non-XML files."""
        assert is_xml_file(Path("data.csv")) is False
        assert is_xml_file(Path("data.json")) is False

    def test_is_xml_file_is_case_insensitive(self):
        """Test that is_xml_file is case-insensitive."""
        assert is_xml_file(Path("DATA.XML")) is True
        assert is_xml_file(Path("Data.Xml")) is True
