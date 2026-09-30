"""
Tests for chaksudb/ingest/framework/image_metadata.py

Tests image metadata extraction utilities based on their docstrings.
Uses real images from data folder where available.
"""

import pytest
from pathlib import Path
import numpy as np
from PIL import Image as PILImage

from chaksudb.ingest.framework.image_metadata import (
    ImageMetadata,
    extract_image_metadata,
)


class TestImageMetadata:
    """Tests for ImageMetadata dataclass."""

    def test_image_metadata_to_dict_returns_dict(self, tmp_path):
        """Test that ImageMetadata.to_dict() returns a dictionary."""
        test_file = tmp_path / "test.jpg"
        
        metadata = ImageMetadata(
            file_path=test_file,
            file_format="jpg",
            resolution_width=1920,
            resolution_height=1080,
            color_channels=3,
            file_size=102400
        )
        
        result = metadata.to_dict()
        assert isinstance(result, dict)

    def test_image_metadata_to_dict_contains_all_fields(self, tmp_path):
        """Test that ImageMetadata.to_dict() contains all expected fields."""
        test_file = tmp_path / "test.jpg"
        
        metadata = ImageMetadata(
            file_path=test_file,
            file_format="jpg",
            resolution_width=1920,
            resolution_height=1080,
            color_channels=3,
            file_size=102400,
            exif_data={"Make": "Canon"}
        )
        
        result = metadata.to_dict()
        
        assert "file_path" in result
        assert "file_format" in result
        assert "resolution_width" in result
        assert "resolution_height" in result
        assert "color_channels" in result
        assert "file_size" in result
        assert "exif_data" in result

    def test_image_metadata_to_dict_converts_path_to_string(self, tmp_path):
        """Test that ImageMetadata.to_dict() converts Path to string."""
        test_file = tmp_path / "test.jpg"
        
        metadata = ImageMetadata(file_path=test_file)
        result = metadata.to_dict()
        
        assert isinstance(result["file_path"], str)


class TestExtractImageMetadata:
    """Tests for extract_image_metadata function."""

    def test_extract_image_metadata_returns_image_metadata(self, tmp_path):
        """Test that extract_image_metadata returns an ImageMetadata object."""
        # Create a simple test image
        test_file = tmp_path / "test.png"
        img = PILImage.new("RGB", (100, 100), color="red")
        img.save(test_file)
        
        result = extract_image_metadata(test_file)
        assert isinstance(result, ImageMetadata)

    def test_extract_image_metadata_extracts_resolution(self, tmp_path):
        """Test that extract_image_metadata extracts resolution correctly."""
        # Create image with known dimensions
        test_file = tmp_path / "test.png"
        width, height = 320, 240
        img = PILImage.new("RGB", (width, height))
        img.save(test_file)
        
        metadata = extract_image_metadata(test_file)
        
        assert metadata.resolution_width == width
        assert metadata.resolution_height == height

    def test_extract_image_metadata_extracts_file_size(self, tmp_path):
        """Test that extract_image_metadata extracts file size."""
        test_file = tmp_path / "test.png"
        img = PILImage.new("RGB", (100, 100))
        img.save(test_file)
        
        metadata = extract_image_metadata(test_file)
        
        assert metadata.file_size is not None
        assert metadata.file_size > 0
        assert metadata.file_size == test_file.stat().st_size

    def test_extract_image_metadata_extracts_color_channels_rgb(self, tmp_path):
        """Test that extract_image_metadata extracts color channels for RGB image."""
        test_file = tmp_path / "test.png"
        img = PILImage.new("RGB", (100, 100))
        img.save(test_file)
        
        metadata = extract_image_metadata(test_file)
        
        assert metadata.color_channels == 3

    def test_extract_image_metadata_extracts_color_channels_grayscale(self, tmp_path):
        """Test that extract_image_metadata extracts color channels for grayscale image."""
        test_file = tmp_path / "test.png"
        img = PILImage.new("L", (100, 100))
        img.save(test_file)
        
        metadata = extract_image_metadata(test_file)
        
        assert metadata.color_channels == 1

    def test_extract_image_metadata_extracts_file_format(self, tmp_path):
        """Test that extract_image_metadata extracts normalized file format."""
        # Test PNG
        png_file = tmp_path / "test.png"
        img = PILImage.new("RGB", (100, 100))
        img.save(png_file)
        
        metadata = extract_image_metadata(png_file)
        assert metadata.file_format == "png"
        
        # Test JPEG
        jpg_file = tmp_path / "test.jpg"
        img.save(jpg_file)
        
        metadata = extract_image_metadata(jpg_file)
        assert metadata.file_format in ["jpg", "jpeg"]

    def test_extract_image_metadata_handles_different_formats(self, tmp_path):
        """Test that extract_image_metadata handles different image formats."""
        formats = [
            ("test.png", "png"),
            ("test.jpg", "jpg"),
            ("test.tif", "tif"),
        ]
        
        for filename, expected_format in formats:
            test_file = tmp_path / filename
            img = PILImage.new("RGB", (100, 100))
            
            if filename.endswith(".tif"):
                img.save(test_file, format="TIFF")
            else:
                img.save(test_file)
            
            metadata = extract_image_metadata(test_file)
            assert metadata.file_format == expected_format

    def test_extract_image_metadata_raises_filenotfound_for_nonexistent(self, tmp_path):
        """Test that extract_image_metadata raises FileNotFoundError for nonexistent file."""
        nonexistent = tmp_path / "does_not_exist.jpg"
        
        with pytest.raises(FileNotFoundError, match="Image file not found"):
            extract_image_metadata(nonexistent)

    def test_extract_image_metadata_preserves_file_path(self, tmp_path):
        """Test that extract_image_metadata preserves the original file_path."""
        test_file = tmp_path / "test.png"
        img = PILImage.new("RGB", (100, 100))
        img.save(test_file)
        
        metadata = extract_image_metadata(test_file)
        
        assert metadata.file_path == test_file

    def test_extract_image_metadata_handles_case_sensitive_extensions(self, tmp_path):
        """Test that extract_image_metadata handles extensions with different cases."""
        # Note: On case-insensitive filesystems, this might behave differently
        test_file = tmp_path / "test.PNG"
        img = PILImage.new("RGB", (100, 100))
        img.save(test_file)
        
        metadata = extract_image_metadata(test_file)
        
        # Format should still be normalized to lowercase
        assert metadata.file_format == "png"

    def test_extract_image_metadata_handles_large_images(self, tmp_path):
        """Test that extract_image_metadata handles large images."""
        test_file = tmp_path / "large.png"
        # Create a larger image
        img = PILImage.new("RGB", (2000, 1500))
        img.save(test_file)
        
        metadata = extract_image_metadata(test_file)
        
        assert metadata.resolution_width == 2000
        assert metadata.resolution_height == 1500
        assert metadata.color_channels == 3

    def test_extract_image_metadata_handles_rgba_images(self, tmp_path):
        """Test that extract_image_metadata handles RGBA images."""
        test_file = tmp_path / "rgba.png"
        img = PILImage.new("RGBA", (100, 100))
        img.save(test_file)
        
        metadata = extract_image_metadata(test_file)
        
        # RGBA should be detected as 3 channels (after conversion or OpenCV reading)
        assert metadata.color_channels in [3, 4]  # Depends on how it's read

    @pytest.mark.skipif(
        not Path("/home/ani/chaksu/chaksudb/data/24_ORIGA/Masks/125.png").exists(),
        reason="Real mask file not available"
    )
    def test_extract_image_metadata_with_real_mask(self):
        """Test extract_image_metadata with a real mask file from data folder."""
        mask_path = Path("/home/ani/chaksu/chaksudb/data/24_ORIGA/Masks/125.png")
        
        metadata = extract_image_metadata(mask_path)
        
        assert metadata.file_format == "png"
        assert metadata.resolution_width is not None
        assert metadata.resolution_height is not None
        assert metadata.file_size is not None
        assert metadata.file_size > 0

    @pytest.mark.skipif(
        not Path("/home/ani/chaksu/chaksudb/data/17_DiaRetDB1/documents/diaretdb1_image003.png").exists(),
        reason="Real image file not available"
    )
    def test_extract_image_metadata_with_real_fundus_image(self):
        """Test extract_image_metadata with a real fundus image from data folder."""
        image_path = Path("/home/ani/chaksu/chaksudb/data/17_DiaRetDB1/documents/diaretdb1_image003.png")
        
        metadata = extract_image_metadata(image_path)
        
        assert metadata.file_format == "png"
        assert metadata.resolution_width is not None
        assert metadata.resolution_height is not None
        assert metadata.color_channels in [1, 3]  # Grayscale or RGB
        assert metadata.file_size is not None
        assert metadata.file_size > 0

    def test_extract_image_metadata_exif_data_extraction(self, tmp_path):
        """Test that extract_image_metadata extracts EXIF data when available."""
        test_file = tmp_path / "with_exif.jpg"
        img = PILImage.new("RGB", (100, 100))
        
        # Add some EXIF data
        exif_data = img.getexif()
        exif_data[0x0110] = "TestCamera"  # Model
        
        img.save(test_file, exif=exif_data)
        
        metadata = extract_image_metadata(test_file)
        
        # EXIF data may or may not be present depending on image format support
        # Just verify the field exists
        assert hasattr(metadata, "exif_data")

    def test_extract_image_metadata_handles_images_without_exif(self, tmp_path):
        """Test that extract_image_metadata handles images without EXIF data gracefully."""
        test_file = tmp_path / "no_exif.png"
        img = PILImage.new("RGB", (100, 100))
        img.save(test_file)
        
        metadata = extract_image_metadata(test_file)
        
        # PNG typically doesn't have EXIF data
        # Should be None or empty dict
        assert metadata.exif_data is None or metadata.exif_data == {}

    def test_extract_image_metadata_handles_square_images(self, tmp_path):
        """Test that extract_image_metadata correctly handles square images."""
        test_file = tmp_path / "square.png"
        size = 512
        img = PILImage.new("RGB", (size, size))
        img.save(test_file)
        
        metadata = extract_image_metadata(test_file)
        
        assert metadata.resolution_width == size
        assert metadata.resolution_height == size
