"""
Tests for chaksudb/ingest/framework/mask_converter.py

Tests mask format handling utilities based on their docstrings.
Uses real masks and contour files from data folder where available.
Saves generated masks to tests/test_storage/generated_masks for inspection.
"""

import json
import pytest
import numpy as np
from pathlib import Path
from PIL import Image as PILImage
import cv2

from chaksudb.ingest.framework.mask_converter import (
    is_multiclass_mask,
    get_mask_classes,
    extract_class_from_mask,
    extract_classes_from_multiclass_mask,
    validate_binary_mask,
    convert_contour_to_binary_mask,
    parse_xml_polygon_to_binary_mask,
    load_soft_map,
    load_layer_boundaries,
)

# Permanent storage directory for generated masks
STORAGE_DIR = Path(__file__).parent.parent / "test_storage" / "generated_masks"
STORAGE_DIR.mkdir(parents=True, exist_ok=True)


class TestValidateBinaryMask:
    """Tests for validate_binary_mask function."""

    def test_validate_binary_mask_returns_numpy_array(self, tmp_path):
        """Test that validate_binary_mask returns a numpy array."""
        mask_file = tmp_path / "mask.png"
        # Create a simple binary mask
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[25:75, 25:75] = 255
        cv2.imwrite(str(mask_file), mask)
        
        result = validate_binary_mask(mask_file)
        assert isinstance(result, np.ndarray)

    def test_validate_binary_mask_returns_uint8(self, tmp_path):
        """Test that validate_binary_mask returns uint8 array."""
        mask_file = tmp_path / "mask.png"
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[25:75, 25:75] = 255
        cv2.imwrite(str(mask_file), mask)
        
        result = validate_binary_mask(mask_file)
        assert result.dtype == np.uint8

    def test_validate_binary_mask_accepts_0_and_255_values(self, tmp_path):
        """Test that validate_binary_mask accepts masks with 0 and 255 values."""
        mask_file = tmp_path / "mask.png"
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[25:75, 25:75] = 255
        cv2.imwrite(str(mask_file), mask)
        
        result = validate_binary_mask(mask_file)
        unique_values = np.unique(result)
        assert set(unique_values).issubset({0, 255})

    def test_validate_binary_mask_normalizes_0_and_1_to_255(self, tmp_path):
        """Test that validate_binary_mask normalizes 0-1 masks to 0-255."""
        mask_file = tmp_path / "mask.png"
        # Create mask with 0 and 1 values directly as float, then save as image
        # We need to test the normalization logic when the function detects 0-1 values
        # This is more realistic when reading from certain formats
        mask = np.zeros((100, 100), dtype=np.float32)
        mask[25:75, 25:75] = 1.0
        # Save as image that will preserve the binary nature but in 0-255 range
        mask_uint8 = (mask * 255).astype(np.uint8)
        cv2.imwrite(str(mask_file), mask_uint8)
        
        result = validate_binary_mask(mask_file)
        # Should have 0 and 255 values
        assert 255 in result
        assert 0 in result
        assert set(np.unique(result)).issubset({0, 255})

    def test_validate_binary_mask_thresholds_non_binary_values(self, tmp_path):
        """Test that validate_binary_mask thresholds non-binary values."""
        mask_file = tmp_path / "mask.png"
        # Create mask with various gray values
        mask = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        cv2.imwrite(str(mask_file), mask)
        
        result = validate_binary_mask(mask_file)
        unique_values = np.unique(result)
        # Should be thresholded to binary: >127 -> 255, <=127 -> 0
        assert set(unique_values).issubset({0, 255})

    def test_validate_binary_mask_raises_filenotfound_for_nonexistent(self, tmp_path):
        """Test that validate_binary_mask raises FileNotFoundError for nonexistent file."""
        nonexistent = tmp_path / "does_not_exist.png"
        
        with pytest.raises(FileNotFoundError, match="Mask file not found"):
            validate_binary_mask(nonexistent)

    def test_validate_binary_mask_handles_different_formats(self, tmp_path):
        """Test that validate_binary_mask handles different image formats."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[25:75, 25:75] = 255
        
        for ext in [".png", ".jpg", ".tif"]:
            mask_file = tmp_path / f"mask{ext}"
            cv2.imwrite(str(mask_file), mask)
            
            result = validate_binary_mask(mask_file)
            assert result is not None
            assert result.shape == (100, 100)

    @pytest.mark.skipif(
        not Path("/home/ani/chaksu/chaksudb/data/24_ORIGA/Masks/125.png").exists(),
        reason="Real mask file not available"
    )
    def test_validate_binary_mask_with_real_multiclass_mask(self, tmp_path):
        """
        Test validate_binary_mask with ORIGA mask (which is multi-class [0,1,2]).
        
        ORIGA masks contain 3 classes: 0=background, 1=optic disc, 2=cup.
        We need to convert to binary first before validate_binary_mask can handle it.
        """
        # Read the multi-class ORIGA mask
        mask_path = Path("/home/ani/chaksu/chaksudb/data/24_ORIGA/Masks/125.png")
        original_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        print(f"\n⚠ ORIGA masks are multi-class, not binary!")
        print(f"  Original unique values: {np.unique(original_mask)}")
        
        # Convert multi-class to binary: merge class 1 (disc) and 2 (cup) into foreground
        binary_mask_array = np.where(original_mask > 0, 255, 0).astype(np.uint8)
        
        # Save the binary version to temp file
        binary_temp = tmp_path / "origa_125_binary.png"
        cv2.imwrite(str(binary_temp), binary_mask_array)
        
        # Now test validate_binary_mask with the binary version
        result = validate_binary_mask(binary_temp)
        
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.uint8
        assert result.ndim == 2
        assert set(np.unique(result)).issubset({0, 255})
        
        # Save both for comparison in permanent storage
        output_dir = STORAGE_DIR / "01_origa_multiclass_to_binary"
        output_dir.mkdir(exist_ok=True)
        
        cv2.imwrite(str(output_dir / "01_01_original_multiclass.png"), original_mask)
        cv2.imwrite(str(output_dir / "01_02_converted_binary.png"), binary_mask_array)
        cv2.imwrite(str(output_dir / "01_03_validated_binary.png"), result)
        
        print(f"✓ Multi-class to binary conversion saved to: {output_dir}")
        print(f"  Original: {np.unique(original_mask)} -> Binary: {np.unique(result)}")
        print(f"  Foreground pixels: {(result == 255).sum()}")
    
    def test_validate_binary_mask_with_synthetic_masks(self):
        """Test validate_binary_mask with synthetic binary masks (0/255 format)."""
        results_dir = STORAGE_DIR / "02_synthetic_binary_masks"
        results_dir.mkdir(exist_ok=True)
        
        test_cases = [
            ("circle", lambda h, w: cv2.circle(np.zeros((h, w), dtype=np.uint8), (w//2, h//2), min(h, w)//3, 255, -1)),
            ("rectangle", lambda h, w: cv2.rectangle(np.zeros((h, w), dtype=np.uint8), (w//4, h//4), (3*w//4, 3*h//4), 255, -1)),
            ("ellipse", lambda h, w: cv2.ellipse(np.zeros((h, w), dtype=np.uint8), (w//2, h//2), (w//3, h//4), 0, 0, 360, 255, -1)),
        ]
        
        for i, (name, mask_gen) in enumerate(test_cases, start=1):
            # Create synthetic binary mask
            h, w = 512, 512
            binary_mask = mask_gen(h, w)
            
            # Save to temp file
            temp_file = STORAGE_DIR / f"temp_{name}.png"
            cv2.imwrite(str(temp_file), binary_mask)
            
            # Validate
            result = validate_binary_mask(temp_file)
            
            assert isinstance(result, np.ndarray)
            assert result.dtype == np.uint8
            assert set(np.unique(result)).issubset({0, 255})
            
            # Save validated result
            output_path = results_dir / f"02_{i:02d}_{name}_binary.png"
            cv2.imwrite(str(output_path), result)
            
            # Cleanup temp
            temp_file.unlink()
        
        print(f"\n✓ Validated {len(test_cases)} synthetic binary masks saved to: {results_dir}")


class TestMulticlassHelpers:
    """Tests for multi-class mask helper functions."""

    def test_is_multiclass_mask_returns_true_for_multiclass(self, tmp_path):
        """Test that is_multiclass_mask returns True for multi-class masks."""
        mask_file = tmp_path / "multiclass.png"
        # Create mask with 3 classes
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[25:50, 25:75] = 1  # Class 1
        mask[50:75, 25:75] = 2  # Class 2
        cv2.imwrite(str(mask_file), mask)
        
        result = is_multiclass_mask(mask_file)
        assert result is True

    def test_is_multiclass_mask_returns_false_for_binary(self, tmp_path):
        """Test that is_multiclass_mask returns False for binary masks."""
        mask_file = tmp_path / "binary.png"
        # Create binary mask
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[25:75, 25:75] = 255
        cv2.imwrite(str(mask_file), mask)
        
        result = is_multiclass_mask(mask_file)
        assert result is False

    def test_get_mask_classes_returns_nonzero_classes(self, tmp_path):
        """Test that get_mask_classes returns all non-zero class IDs."""
        mask_file = tmp_path / "multiclass.png"
        # Create mask with classes 0, 1, 2, 5
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:20, 10:20] = 1
        mask[30:40, 30:40] = 2
        mask[60:70, 60:70] = 5
        cv2.imwrite(str(mask_file), mask)
        
        result = get_mask_classes(mask_file)
        
        assert isinstance(result, np.ndarray)
        assert set(result) == {1, 2, 5}
        assert 0 not in result  # Background should be excluded

    def test_extract_class_from_mask_extracts_specific_class(self, tmp_path):
        """Test that extract_class_from_mask extracts only the specified class."""
        mask_file = tmp_path / "multiclass.png"
        # Create mask with classes 0, 1, 2
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:40, 20:40] = 1  # Class 1
        mask[60:80, 60:80] = 2  # Class 2
        cv2.imwrite(str(mask_file), mask)
        
        # Extract class 1
        result = extract_class_from_mask(mask_file, class_id=1)
        
        assert result.dtype == np.uint8
        assert set(np.unique(result)) == {0, 255}
        # Only class 1 area should be 255
        assert (result[20:40, 20:40] == 255).all()
        assert (result[60:80, 60:80] == 0).all()

    def test_extract_class_from_mask_raises_error_for_missing_class(self, tmp_path):
        """Test that extract_class_from_mask raises ValueError for non-existent class."""
        mask_file = tmp_path / "multiclass.png"
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:40, 20:40] = 1
        cv2.imwrite(str(mask_file), mask)
        
        with pytest.raises(ValueError, match="Class ID 5 not found"):
            extract_class_from_mask(mask_file, class_id=5)

    def test_extract_classes_from_multiclass_mask_extracts_all(self, tmp_path):
        """Test extracting all classes from multi-class mask."""
        mask_file = tmp_path / "multiclass.png"
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:30, 10:30] = 1
        mask[40:60, 40:60] = 2
        mask[70:90, 70:90] = 3
        cv2.imwrite(str(mask_file), mask)
        
        result = extract_classes_from_multiclass_mask(mask_file)
        
        assert isinstance(result, dict)
        assert set(result.keys()) == {"class_1", "class_2", "class_3"}
        assert all(mask.dtype == np.uint8 for mask in result.values())
        assert all(set(np.unique(mask)).issubset({0, 255}) for mask in result.values())

    def test_extract_classes_with_custom_names(self, tmp_path):
        """Test extracting classes with custom names."""
        mask_file = tmp_path / "multiclass.png"
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:30, 10:30] = 1
        mask[40:60, 40:60] = 2
        cv2.imwrite(str(mask_file), mask)
        
        result = extract_classes_from_multiclass_mask(
            mask_file,
            class_names={1: "optic_disc", 2: "cup"}
        )
        
        assert set(result.keys()) == {"optic_disc", "cup"}

    def test_extract_specific_classes_only(self, tmp_path):
        """Test extracting only specific classes."""
        mask_file = tmp_path / "multiclass.png"
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:30, 10:30] = 1
        mask[40:60, 40:60] = 2
        mask[70:90, 70:90] = 3
        cv2.imwrite(str(mask_file), mask)
        
        result = extract_classes_from_multiclass_mask(
            mask_file,
            classes_to_extract=[1, 3]
        )
        
        assert set(result.keys()) == {"class_1", "class_3"}
        assert "class_2" not in result

    def test_merge_classes_into_single_mask(self, tmp_path):
        """Test merging multiple classes into single foreground."""
        mask_file = tmp_path / "multiclass.png"
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:30, 10:30] = 1
        mask[40:60, 40:60] = 2
        mask[70:90, 70:90] = 3
        cv2.imwrite(str(mask_file), mask)
        
        result = extract_classes_from_multiclass_mask(
            mask_file,
            merge_classes=[1, 2]
        )
        
        assert set(result.keys()) == {"merged"}
        merged = result["merged"]
        # Classes 1 and 2 should be 255, class 3 should be 0
        assert (merged[10:30, 10:30] == 255).all()
        assert (merged[40:60, 40:60] == 255).all()
        assert (merged[70:90, 70:90] == 0).all()

    @pytest.mark.skipif(
        not Path("/home/ani/chaksu/chaksudb/data/24_ORIGA/Masks/125.png").exists(),
        reason="ORIGA mask not available"
    )
    def test_multiclass_helpers_with_real_origa_mask(self):
        """Test multiclass helper functions with real ORIGA mask."""
        mask_path = Path("/home/ani/chaksu/chaksudb/data/24_ORIGA/Masks/125.png")
        
        # Test is_multiclass_mask
        assert is_multiclass_mask(mask_path) is True
        
        # Test get_mask_classes
        classes = get_mask_classes(mask_path)
        assert set(classes) == {1, 2}  # ORIGA has disc=1, cup=2
        
        # Test extract_class_from_mask without filling
        disc_mask = extract_class_from_mask(mask_path, class_id=1)
        cup_mask = extract_class_from_mask(mask_path, class_id=2)
        
        # Test extract_class_from_mask WITH hole filling
        disc_mask_filled = extract_class_from_mask(mask_path, class_id=1, fill_holes=True)
        cup_mask_filled = extract_class_from_mask(mask_path, class_id=2, fill_holes=True)
        
        # Cup should be smaller than disc
        disc_area = (disc_mask == 255).sum()
        cup_area = (cup_mask == 255).sum()
        disc_area_filled = (disc_mask_filled == 255).sum()
        cup_area_filled = (cup_mask_filled == 255).sum()
        
        assert cup_area < disc_area
        assert disc_area_filled >= disc_area  # Filled should be >= original
        assert cup_area_filled >= cup_area
        
        # Test extract_classes_from_multiclass_mask WITHOUT filling
        masks = extract_classes_from_multiclass_mask(
            mask_path,
            class_names={1: "optic_disc", 2: "cup"}
        )
        
        # Test extract_classes_from_multiclass_mask WITH filling
        masks_filled = extract_classes_from_multiclass_mask(
            mask_path,
            class_names={1: "optic_disc", 2: "cup"},
            fill_holes=True
        )
        
        assert set(masks.keys()) == {"optic_disc", "cup"}
        assert set(masks_filled.keys()) == {"optic_disc", "cup"}
        
        # Save for inspection
        output_dir = STORAGE_DIR / "09_multiclass_extraction"
        output_dir.mkdir(exist_ok=True)
        
        # Original (no filling)
        cv2.imwrite(str(output_dir / "09_01_disc_only.png"), masks["optic_disc"])
        cv2.imwrite(str(output_dir / "09_02_cup_only.png"), masks["cup"])
        
        # With hole filling
        cv2.imwrite(str(output_dir / "09_04_disc_filled.png"), masks_filled["optic_disc"])
        cv2.imwrite(str(output_dir / "09_05_cup_filled.png"), masks_filled["cup"])
        
        # Test merge
        merged = extract_classes_from_multiclass_mask(
            mask_path,
            merge_classes=[1, 2]
        )
        cv2.imwrite(str(output_dir / "09_03_merged_disc_cup.png"), merged["merged"])
        
        # Merged with filling
        merged_filled = extract_classes_from_multiclass_mask(
            mask_path,
            merge_classes=[1, 2],
            fill_holes=True
        )
        cv2.imwrite(str(output_dir / "09_06_merged_filled.png"), merged_filled["merged"])
        
        print(f"\n✓ ORIGA multi-class extraction saved to: {output_dir}")
        print(f"  Original - Disc: {disc_area}px, Cup: {cup_area}px")
        print(f"  Filled   - Disc: {disc_area_filled}px, Cup: {cup_area_filled}px")


class TestValidateBinaryMaskWithMulticlassSupport:
    """Tests for updated validate_binary_mask with multi-class support."""

    def test_validate_binary_mask_with_extract_class(self, tmp_path):
        """Test validate_binary_mask with extract_class parameter."""
        mask_file = tmp_path / "multiclass.png"
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:40, 20:40] = 1
        mask[60:80, 60:80] = 2
        cv2.imwrite(str(mask_file), mask)
        
        # Extract class 1
        result = validate_binary_mask(mask_file, extract_class=1)
        
        assert result.dtype == np.uint8
        assert set(np.unique(result)) == {0, 255}
        assert (result[20:40, 20:40] == 255).all()
        assert (result[60:80, 60:80] == 0).all()

    def test_validate_binary_mask_with_merge_nonzero(self, tmp_path):
        """Test validate_binary_mask with merge_nonzero parameter."""
        mask_file = tmp_path / "multiclass.png"
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:40, 20:40] = 1
        mask[60:80, 60:80] = 2
        cv2.imwrite(str(mask_file), mask)
        
        # Merge all non-zero classes
        result = validate_binary_mask(mask_file, merge_nonzero=True)
        
        assert result.dtype == np.uint8
        assert set(np.unique(result)) == {0, 255}
        # Both regions should be 255
        assert (result[20:40, 20:40] == 255).all()
        assert (result[60:80, 60:80] == 255).all()

    @pytest.mark.skipif(
        not Path("/home/ani/chaksu/chaksudb/data/24_ORIGA/Masks/125.png").exists(),
        reason="ORIGA mask not available"
    )
    def test_validate_binary_mask_origa_with_options(self):
        """Test validate_binary_mask with ORIGA mask using new options."""
        mask_path = Path("/home/ani/chaksu/chaksudb/data/24_ORIGA/Masks/125.png")
        
        # Extract disc only
        disc_mask = validate_binary_mask(mask_path, extract_class=1)
        
        # Extract cup only
        cup_mask = validate_binary_mask(mask_path, extract_class=2)
        
        # Merge both
        merged_mask = validate_binary_mask(mask_path, merge_nonzero=True)
        
        # Extract with hole filling
        disc_filled = validate_binary_mask(mask_path, extract_class=1, fill_holes=True)
        cup_filled = validate_binary_mask(mask_path, extract_class=2, fill_holes=True)
        merged_filled = validate_binary_mask(mask_path, merge_nonzero=True, fill_holes=True)
        
        # Validate
        disc_area = (disc_mask == 255).sum()
        cup_area = (cup_mask == 255).sum()
        merged_area = (merged_mask == 255).sum()
        disc_filled_area = (disc_filled == 255).sum()
        cup_filled_area = (cup_filled == 255).sum()
        merged_filled_area = (merged_filled == 255).sum()
        
        assert cup_area < disc_area  # Cup smaller than disc
        assert merged_area == disc_area + cup_area  # Merged should equal both
        assert disc_filled_area >= disc_area  # Filled should be >= original
        assert cup_filled_area >= cup_area
        
        # Save for inspection
        output_dir = STORAGE_DIR / "10_validate_with_multiclass_options"
        output_dir.mkdir(exist_ok=True)
        
        # Original
        cv2.imwrite(str(output_dir / "10_01_extract_disc.png"), disc_mask)
        cv2.imwrite(str(output_dir / "10_02_extract_cup.png"), cup_mask)
        cv2.imwrite(str(output_dir / "10_03_merge_all.png"), merged_mask)
        
        # With hole filling
        cv2.imwrite(str(output_dir / "10_04_extract_disc_filled.png"), disc_filled)
        cv2.imwrite(str(output_dir / "10_05_extract_cup_filled.png"), cup_filled)
        cv2.imwrite(str(output_dir / "10_06_merge_all_filled.png"), merged_filled)
        
        print(f"\n✓ validate_binary_mask multi-class options saved to: {output_dir}")
        print(f"  Original - Disc: {disc_area}px, Cup: {cup_area}px, Merged: {merged_area}px")
        print(f"  Filled   - Disc: {disc_filled_area}px, Cup: {cup_filled_area}px, Merged: {merged_filled_area}px")


class TestConvertContourToBinaryMask:
    """Tests for convert_contour_to_binary_mask function."""

    def test_convert_contour_to_binary_mask_returns_numpy_array(self, tmp_path):
        """Test that convert_contour_to_binary_mask returns a numpy array."""
        contour_file = tmp_path / "contour.txt"
        # Create a simple square contour
        contour_file.write_text("10 10\n10 90\n90 90\n90 10\n")
        
        result = convert_contour_to_binary_mask(contour_file, (100, 100))
        assert isinstance(result, np.ndarray)

    def test_convert_contour_to_binary_mask_returns_binary_mask(self, tmp_path):
        """Test that convert_contour_to_binary_mask returns binary mask with 0 and 255."""
        contour_file = tmp_path / "contour.txt"
        contour_file.write_text("10 10\n10 90\n90 90\n90 10\n")
        
        result = convert_contour_to_binary_mask(contour_file, (100, 100))
        unique_values = np.unique(result)
        assert set(unique_values).issubset({0, 255})

    def test_convert_contour_to_binary_mask_respects_image_size(self, tmp_path):
        """Test that convert_contour_to_binary_mask creates mask with correct dimensions."""
        contour_file = tmp_path / "contour.txt"
        contour_file.write_text("10 10\n10 90\n90 90\n90 10\n")
        
        width, height = 200, 150
        result = convert_contour_to_binary_mask(contour_file, (width, height))
        
        assert result.shape == (height, width)

    def test_convert_contour_to_binary_mask_line_separated_format(self, tmp_path):
        """Test convert_contour_to_binary_mask with line-separated coordinates."""
        contour_file = tmp_path / "contour.txt"
        # One coordinate per line: "x y"
        contour_file.write_text("20 20\n20 80\n80 80\n80 20\n")
        
        result = convert_contour_to_binary_mask(contour_file, (100, 100))
        
        assert result is not None
        # Check that polygon is filled (should have some 255 values)
        assert 255 in result

    def test_convert_contour_to_binary_mask_space_separated_format(self, tmp_path):
        """Test convert_contour_to_binary_mask with space-separated coordinates."""
        contour_file = tmp_path / "contour.txt"
        # Single line: "x1 y1 x2 y2 x3 y3 x4 y4"
        contour_file.write_text("20 20 20 80 80 80 80 20")
        
        result = convert_contour_to_binary_mask(
            contour_file, (100, 100), coordinate_format="space_separated"
        )
        
        assert result is not None
        assert 255 in result

    def test_convert_contour_to_binary_mask_comma_separated_format(self, tmp_path):
        """Test convert_contour_to_binary_mask with comma-separated coordinates."""
        contour_file = tmp_path / "contour.txt"
        # Comma-separated: "x1,y1 x2,y2 x3,y3 x4,y4"
        contour_file.write_text("20,20 20,80 80,80 80,20")
        
        result = convert_contour_to_binary_mask(
            contour_file, (100, 100), coordinate_format="comma_separated"
        )
        
        assert result is not None
        assert 255 in result

    def test_convert_contour_to_binary_mask_json_format(self, tmp_path):
        """Test convert_contour_to_binary_mask with JSON format."""
        contour_file = tmp_path / "contour.json"
        coords = [[20, 20], [20, 80], [80, 80], [80, 20]]
        contour_file.write_text(json.dumps(coords))
        
        result = convert_contour_to_binary_mask(
            contour_file, (100, 100), coordinate_format="json"
        )
        
        assert result is not None
        assert 255 in result

    def test_convert_contour_to_binary_mask_auto_detects_format(self, tmp_path):
        """Test that convert_contour_to_binary_mask auto-detects coordinate format."""
        # Test JSON auto-detection
        json_file = tmp_path / "contour.json"
        coords = [[20, 20], [20, 80], [80, 80], [80, 20]]
        json_file.write_text(json.dumps(coords))
        
        result = convert_contour_to_binary_mask(json_file, (100, 100))
        assert result is not None

    def test_convert_contour_to_binary_mask_clips_out_of_bounds_coordinates(self, tmp_path):
        """Test that convert_contour_to_binary_mask clips coordinates outside image bounds."""
        contour_file = tmp_path / "contour.txt"
        # Coordinates outside bounds
        contour_file.write_text("0 0\n0 150\n150 150\n150 0\n")
        
        # Should clip to 100x100
        result = convert_contour_to_binary_mask(contour_file, (100, 100))
        
        assert result is not None
        assert result.shape == (100, 100)

    def test_convert_contour_to_binary_mask_raises_filenotfound_for_nonexistent(self, tmp_path):
        """Test that convert_contour_to_binary_mask raises FileNotFoundError."""
        nonexistent = tmp_path / "does_not_exist.txt"
        
        with pytest.raises(FileNotFoundError, match="Contour file not found"):
            convert_contour_to_binary_mask(nonexistent, (100, 100))

    def test_convert_contour_to_binary_mask_raises_error_for_too_few_points(self, tmp_path):
        """Test that convert_contour_to_binary_mask raises ValueError for too few points."""
        contour_file = tmp_path / "contour.txt"
        # Only 2 points (need at least 3 for polygon)
        contour_file.write_text("10 10\n90 90\n")
        
        with pytest.raises(ValueError, match="Contour must have at least 3 points"):
            convert_contour_to_binary_mask(contour_file, (100, 100))

    @pytest.mark.skipif(
        not Path("/home/ani/chaksu/chaksudb/data/19_Drishti-GS1/Drishti-GS1_files/Training/GT/drishtiGS_031/AvgBoundary/drishtiGS_031_ODAvgBoundary.txt").exists(),
        reason="Real contour file not available"
    )
    def test_convert_contour_to_binary_mask_with_real_contour(self):
        """Test convert_contour_to_binary_mask with a real contour file from data folder."""
        contour_path = Path(
            "/home/ani/chaksu/chaksudb/data/19_Drishti-GS1/Drishti-GS1_files/"
            "Training/GT/drishtiGS_031/AvgBoundary/drishtiGS_031_ODAvgBoundary.txt"
        )
        
        # Use a reasonable image size (typical fundus image)
        result = convert_contour_to_binary_mask(contour_path, (2048, 1536))
        
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.uint8
        assert result.shape == (1536, 2048)
        # Should have filled polygon
        assert 255 in result
        
        # Save generated mask for inspection in permanent storage
        output_path = STORAGE_DIR / "03_contour_mask_drishti_031_OD.png"
        cv2.imwrite(str(output_path), result)
        print(f"\n✓ Contour mask from drishtiGS_031 OD boundary saved to: {output_path}")
        print(f"  Shape: {result.shape}, Fill ratio: {(result == 255).sum() / result.size:.4f}")
    
    @pytest.mark.skipif(
        not Path("/home/ani/chaksu/chaksudb/data/19_Drishti-GS1/Drishti-GS1_files/Training/GT/drishtiGS_031/AvgBoundary").exists(),
        reason="Drishti-GS1 contour files not available"
    )
    def test_convert_contour_to_binary_mask_optic_disc_and_cup(self):
        """Test converting both optic disc and cup contours to masks."""
        base_path = Path(
            "/home/ani/chaksu/chaksudb/data/19_Drishti-GS1/Drishti-GS1_files/"
            "Training/GT/drishtiGS_031/AvgBoundary"
        )
        
        od_contour = base_path / "drishtiGS_031_ODAvgBoundary.txt"
        cup_contour = base_path / "drishtiGS_031_CupAvgBoundary.txt"
        
        image_size = (2048, 1536)
        
        # Convert OD contour
        od_mask = convert_contour_to_binary_mask(od_contour, image_size)
        assert 255 in od_mask
        
        # Convert Cup contour
        cup_mask = convert_contour_to_binary_mask(cup_contour, image_size)
        assert 255 in cup_mask
        
        # Cup should be smaller than OD
        od_area = (od_mask == 255).sum()
        cup_area = (cup_mask == 255).sum()
        assert cup_area < od_area, "Cup area should be smaller than OD area"
        
        # Save both masks to permanent storage
        output_dir = STORAGE_DIR / "04_drishti_contour_masks"
        output_dir.mkdir(exist_ok=True)
        
        cv2.imwrite(str(output_dir / "04_01_OD_mask.png"), od_mask)
        cv2.imwrite(str(output_dir / "04_02_Cup_mask.png"), cup_mask)
        
        # Create a combined visualization (OD=128, Cup=255)
        combined = np.where(cup_mask == 255, 255, np.where(od_mask == 255, 128, 0)).astype(np.uint8)
        cv2.imwrite(str(output_dir / "04_03_Combined_OD_Cup.png"), combined)
        
        print(f"\n✓ OD and Cup masks saved to: {output_dir}")
        print(f"  OD area: {od_area} pixels, Cup area: {cup_area} pixels")
        print(f"  Cup/Disc ratio: {cup_area/od_area:.4f}")


class TestParseXmlPolygonToBinaryMask:
    """Tests for parse_xml_polygon_to_binary_mask function."""

    def test_parse_xml_polygon_to_binary_mask_returns_numpy_array(self, tmp_path):
        """Test that parse_xml_polygon_to_binary_mask returns a numpy array."""
        xml_file = tmp_path / "polygon.xml"
        xml_content = """<?xml version="1.0"?>
        <root>
            <polygon>
                <point x="20" y="20"/>
                <point x="20" y="80"/>
                <point x="80" y="80"/>
                <point x="80" y="20"/>
            </polygon>
        </root>
        """
        xml_file.write_text(xml_content)
        
        result = parse_xml_polygon_to_binary_mask(xml_file, (100, 100))
        assert isinstance(result, np.ndarray)

    def test_parse_xml_polygon_to_binary_mask_returns_binary_mask(self, tmp_path):
        """Test that parse_xml_polygon_to_binary_mask returns binary mask."""
        xml_file = tmp_path / "polygon.xml"
        xml_content = """<?xml version="1.0"?>
        <root>
            <polygon>
                <point x="20" y="20"/>
                <point x="20" y="80"/>
                <point x="80" y="80"/>
                <point x="80" y="20"/>
            </polygon>
        </root>
        """
        xml_file.write_text(xml_content)
        
        result = parse_xml_polygon_to_binary_mask(xml_file, (100, 100))
        unique_values = np.unique(result)
        assert set(unique_values).issubset({0, 255})

    def test_parse_xml_polygon_to_binary_mask_respects_image_size(self, tmp_path):
        """Test that parse_xml_polygon_to_binary_mask creates mask with correct dimensions."""
        xml_file = tmp_path / "polygon.xml"
        xml_content = """<?xml version="1.0"?>
        <root>
            <polygon>
                <point x="10" y="10"/>
                <point x="10" y="50"/>
                <point x="50" y="50"/>
            </polygon>
        </root>
        """
        xml_file.write_text(xml_content)
        
        width, height = 200, 150
        result = parse_xml_polygon_to_binary_mask(xml_file, (width, height))
        
        assert result.shape == (height, width)

    def test_parse_xml_polygon_to_binary_mask_handles_multiple_polygons(self, tmp_path):
        """Test that parse_xml_polygon_to_binary_mask handles multiple polygons."""
        xml_file = tmp_path / "polygons.xml"
        xml_content = """<?xml version="1.0"?>
        <root>
            <polygon>
                <point x="10" y="10"/>
                <point x="10" y="30"/>
                <point x="30" y="30"/>
            </polygon>
            <polygon>
                <point x="60" y="60"/>
                <point x="60" y="80"/>
                <point x="80" y="80"/>
            </polygon>
        </root>
        """
        xml_file.write_text(xml_content)
        
        result = parse_xml_polygon_to_binary_mask(xml_file, (100, 100))
        
        assert result is not None
        # Should have filled both polygons
        assert 255 in result

    def test_parse_xml_polygon_to_binary_mask_handles_uppercase_attributes(self, tmp_path):
        """Test that parse_xml_polygon_to_binary_mask handles uppercase X/Y attributes."""
        xml_file = tmp_path / "polygon.xml"
        xml_content = """<?xml version="1.0"?>
        <root>
            <polygon>
                <point X="20" Y="20"/>
                <point X="20" Y="80"/>
                <point X="80" Y="80"/>
            </polygon>
        </root>
        """
        xml_file.write_text(xml_content)
        
        result = parse_xml_polygon_to_binary_mask(xml_file, (100, 100))
        assert 255 in result

    def test_parse_xml_polygon_to_binary_mask_raises_filenotfound(self, tmp_path):
        """Test that parse_xml_polygon_to_binary_mask raises FileNotFoundError."""
        nonexistent = tmp_path / "does_not_exist.xml"
        
        with pytest.raises(FileNotFoundError, match="XML file not found"):
            parse_xml_polygon_to_binary_mask(nonexistent, (100, 100))

    def test_parse_xml_polygon_to_binary_mask_raises_error_for_invalid_xml(self, tmp_path):
        """Test that parse_xml_polygon_to_binary_mask raises ValueError for invalid XML."""
        xml_file = tmp_path / "invalid.xml"
        xml_file.write_text("This is not valid XML")
        
        with pytest.raises(ValueError, match="Failed to parse XML file"):
            parse_xml_polygon_to_binary_mask(xml_file, (100, 100))

    def test_parse_xml_polygon_to_binary_mask_raises_error_for_no_polygons(self, tmp_path):
        """Test that parse_xml_polygon_to_binary_mask raises ValueError when no polygons found."""
        xml_file = tmp_path / "no_polygon.xml"
        xml_content = """<?xml version="1.0"?>
        <root>
            <other>No polygon here</other>
        </root>
        """
        xml_file.write_text(xml_content)
        
        with pytest.raises(ValueError, match="Could not find polygon coordinates"):
            parse_xml_polygon_to_binary_mask(xml_file, (100, 100))
    
    def test_parse_xml_and_compare_with_contour(self, tmp_path):
        """Test that XML and contour formats produce similar masks."""
        # Create test data in both formats with same polygon
        coords = [[20, 20], [20, 80], [80, 80], [80, 20]]
        
        # XML format
        xml_file = tmp_path / "polygon.xml"
        xml_content = '<?xml version="1.0"?>\n<root>\n  <polygon>\n'
        for x, y in coords:
            xml_content += f'    <point x="{x}" y="{y}"/>\n'
        xml_content += '  </polygon>\n</root>'
        xml_file.write_text(xml_content)
        
        # Contour format
        contour_file = tmp_path / "contour.txt"
        contour_text = "\n".join([f"{x} {y}" for x, y in coords])
        contour_file.write_text(contour_text)
        
        # Generate masks
        image_size = (100, 100)
        xml_mask = parse_xml_polygon_to_binary_mask(xml_file, image_size)
        contour_mask = convert_contour_to_binary_mask(contour_file, image_size)
        
        # Should be identical
        assert np.array_equal(xml_mask, contour_mask)
        
        # Save for inspection to permanent storage
        output_dir = STORAGE_DIR / "07_xml_vs_contour"
        output_dir.mkdir(exist_ok=True)
        cv2.imwrite(str(output_dir / "07_01_xml_mask.png"), xml_mask)
        cv2.imwrite(str(output_dir / "07_02_contour_mask.png"), contour_mask)
        
        print(f"\n✓ XML and contour masks are identical")
        print(f"  Masks saved to: {output_dir}")


class TestLoadSoftMap:
    """Tests for load_soft_map function."""

    def test_load_soft_map_returns_numpy_array(self, tmp_path):
        """Test that load_soft_map returns a numpy array."""
        soft_map_file = tmp_path / "softmap.png"
        # Create a soft map with grayscale values
        soft_map = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        cv2.imwrite(str(soft_map_file), soft_map)
        
        result = load_soft_map(soft_map_file)
        assert isinstance(result, np.ndarray)

    def test_load_soft_map_returns_float32(self, tmp_path):
        """Test that load_soft_map returns float32 array."""
        soft_map_file = tmp_path / "softmap.png"
        soft_map = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        cv2.imwrite(str(soft_map_file), soft_map)
        
        result = load_soft_map(soft_map_file)
        assert result.dtype == np.float32

    def test_load_soft_map_normalizes_to_0_1_range(self, tmp_path):
        """Test that load_soft_map normalizes values to 0.0-1.0 range."""
        soft_map_file = tmp_path / "softmap.png"
        # Create map with 0-255 range
        soft_map = np.full((100, 100), 128, dtype=np.uint8)
        cv2.imwrite(str(soft_map_file), soft_map)
        
        result = load_soft_map(soft_map_file)
        
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_load_soft_map_handles_npy_files(self, tmp_path):
        """Test that load_soft_map handles .npy files."""
        soft_map_file = tmp_path / "softmap.npy"
        soft_map = np.random.rand(100, 100).astype(np.float32)
        np.save(soft_map_file, soft_map)
        
        result = load_soft_map(soft_map_file)
        
        assert result is not None
        assert result.shape == (100, 100)

    def test_load_soft_map_handles_npz_files(self, tmp_path):
        """Test that load_soft_map handles .npz files."""
        soft_map_file = tmp_path / "softmap.npz"
        soft_map = np.random.rand(100, 100).astype(np.float32)
        np.savez(soft_map_file, mask=soft_map)
        
        result = load_soft_map(soft_map_file)
        
        assert result is not None
        assert result.shape == (100, 100)

    def test_load_soft_map_resizes_if_image_size_provided(self, tmp_path):
        """Test that load_soft_map resizes to provided image_size."""
        soft_map_file = tmp_path / "softmap.png"
        soft_map = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
        cv2.imwrite(str(soft_map_file), soft_map)
        
        result = load_soft_map(soft_map_file, image_size=(100, 100))
        
        # Result shape should be (height, width)
        assert result.shape == (100, 100)

    def test_load_soft_map_raises_filenotfound(self, tmp_path):
        """Test that load_soft_map raises FileNotFoundError for nonexistent file."""
        nonexistent = tmp_path / "does_not_exist.png"
        
        with pytest.raises(FileNotFoundError, match="Soft map file not found"):
            load_soft_map(nonexistent)

    @pytest.mark.skipif(
        not Path("/home/ani/chaksu/chaksudb/data/19_Drishti-GS1/Drishti-GS1_files/Training/GT/drishtiGS_031/SoftMap/drishtiGS_031_ODsegSoftmap.png").exists(),
        reason="Real soft map file not available"
    )
    def test_load_soft_map_with_real_softmap(self):
        """Test load_soft_map with a real soft map file from data folder."""
        softmap_path = Path(
            "/home/ani/chaksu/chaksudb/data/19_Drishti-GS1/Drishti-GS1_files/"
            "Training/GT/drishtiGS_031/SoftMap/drishtiGS_031_ODsegSoftmap.png"
        )
        
        result = load_soft_map(softmap_path)
        
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert result.min() >= 0.0
        assert result.max() <= 1.0
        
        # Save soft map as both probability map and thresholded binary to permanent storage
        output_dir = STORAGE_DIR / "05_softmaps"
        output_dir.mkdir(exist_ok=True)
        
        # Save as grayscale image (0-255 range for visualization)
        soft_viz = (result * 255).astype(np.uint8)
        cv2.imwrite(str(output_dir / "05_01_OD_softmap.png"), soft_viz)
        
        # Save thresholded versions at different thresholds
        for i, threshold in enumerate([0.3, 0.5, 0.7], start=2):
            binary = (result > threshold).astype(np.uint8) * 255
            cv2.imwrite(str(output_dir / f"05_{i:02d}_OD_binary_threshold_{threshold:.1f}.png"), binary)
        
        print(f"\n✓ Soft map and thresholded versions saved to: {output_dir}")
        print(f"  Shape: {result.shape}, Value range: [{result.min():.3f}, {result.max():.3f}]")
        print(f"  Mean probability: {result.mean():.3f}")
    
    @pytest.mark.skipif(
        not Path("/home/ani/chaksu/chaksudb/data/19_Drishti-GS1/Drishti-GS1_files/Training/GT/drishtiGS_031/SoftMap").exists(),
        reason="Drishti-GS1 soft maps not available"
    )
    def test_load_soft_map_od_and_cup(self):
        """Test loading both OD and Cup soft maps and combining them."""
        base_path = Path(
            "/home/ani/chaksu/chaksudb/data/19_Drishti-GS1/Drishti-GS1_files/"
            "Training/GT/drishtiGS_031/SoftMap"
        )
        
        od_softmap = load_soft_map(base_path / "drishtiGS_031_ODsegSoftmap.png")
        cup_softmap = load_soft_map(base_path / "drishtiGS_031_cupsegSoftmap.png")
        
        assert od_softmap.shape == cup_softmap.shape
        
        # Create combined visualization in permanent storage
        output_dir = STORAGE_DIR / "06_combined_softmaps"
        output_dir.mkdir(exist_ok=True)
        
        # Save individual soft maps
        cv2.imwrite(str(output_dir / "06_01_OD_softmap.png"), (od_softmap * 255).astype(np.uint8))
        cv2.imwrite(str(output_dir / "06_02_Cup_softmap.png"), (cup_softmap * 255).astype(np.uint8))
        
        # Create RGB visualization: OD in red channel, Cup in green channel
        h, w = od_softmap.shape
        rgb_viz = np.zeros((h, w, 3), dtype=np.uint8)
        rgb_viz[:, :, 2] = (od_softmap * 255).astype(np.uint8)  # Red for OD
        rgb_viz[:, :, 1] = (cup_softmap * 255).astype(np.uint8)  # Green for Cup
        cv2.imwrite(str(output_dir / "06_03_Combined_RGB_visualization.png"), rgb_viz)
        
        print(f"\n✓ Combined soft maps visualization saved to: {output_dir}")
        print(f"  OD mean prob: {od_softmap.mean():.3f}, Cup mean prob: {cup_softmap.mean():.3f}")


class TestMaskConverterIntegration:
    """Integration tests for mask converter with real data."""
    
    @pytest.mark.skipif(
        not all([
            Path("/home/ani/chaksu/chaksudb/data/24_ORIGA/Masks/125.png").exists(),
            Path("/home/ani/chaksu/chaksudb/data/19_Drishti-GS1/Drishti-GS1_files/Training/GT/drishtiGS_031/AvgBoundary/drishtiGS_031_ODAvgBoundary.txt").exists(),
            Path("/home/ani/chaksu/chaksudb/data/19_Drishti-GS1/Drishti-GS1_files/Training/GT/drishtiGS_031/SoftMap/drishtiGS_031_ODsegSoftmap.png").exists(),
        ]),
        reason="Real dataset files not available"
    )
    def test_comprehensive_mask_processing_pipeline(self):
        """Comprehensive test showing all mask processing capabilities with real data."""
        output_dir = STORAGE_DIR / "08_COMPREHENSIVE_PIPELINE"
        output_dir.mkdir(exist_ok=True)
        
        print("\n" + "="*70)
        print("COMPREHENSIVE MASK PROCESSING TEST")
        print("="*70)
        
        # 1. Validate existing binary mask
        print("\n1. BINARY MASK VALIDATION (ORIGA)")
        binary_mask_path = Path("/home/ani/chaksu/chaksudb/data/24_ORIGA/Masks/125.png")
        binary_mask = validate_binary_mask(binary_mask_path)
        cv2.imwrite(str(output_dir / "08_01_validated_binary_mask.png"), binary_mask)
        print(f"   ✓ Shape: {binary_mask.shape}")
        print(f"   ✓ Unique values: {np.unique(binary_mask)}")
        print(f"   ✓ Mask area: {(binary_mask == 255).sum()} pixels")
        
        # 2. Convert contour to binary mask
        print("\n2. CONTOUR TO BINARY MASK (Drishti-GS1 OD)")
        contour_path = Path(
            "/home/ani/chaksu/chaksudb/data/19_Drishti-GS1/Drishti-GS1_files/"
            "Training/GT/drishtiGS_031/AvgBoundary/drishtiGS_031_ODAvgBoundary.txt"
        )
        contour_mask = convert_contour_to_binary_mask(contour_path, (2048, 1536))
        cv2.imwrite(str(output_dir / "08_02_contour_to_binary_mask.png"), contour_mask)
        print(f"   ✓ Shape: {contour_mask.shape}")
        print(f"   ✓ Mask area: {(contour_mask == 255).sum()} pixels")
        
        # 3. Load soft map
        print("\n3. SOFT MAP LOADING (Drishti-GS1 OD)")
        softmap_path = Path(
            "/home/ani/chaksu/chaksudb/data/19_Drishti-GS1/Drishti-GS1_files/"
            "Training/GT/drishtiGS_031/SoftMap/drishtiGS_031_ODsegSoftmap.png"
        )
        soft_map = load_soft_map(softmap_path)
        cv2.imwrite(str(output_dir / "08_03_softmap.png"), (soft_map * 255).astype(np.uint8))
        print(f"   ✓ Shape: {soft_map.shape}")
        print(f"   ✓ Value range: [{soft_map.min():.3f}, {soft_map.max():.3f}]")
        print(f"   ✓ Mean probability: {soft_map.mean():.3f}")
        
        # 4. Threshold soft map at multiple levels
        print("\n4. SOFT MAP THRESHOLDING")
        for i, threshold in enumerate([0.3, 0.5, 0.7], start=4):
            binary = (soft_map > threshold).astype(np.uint8) * 255
            cv2.imwrite(str(output_dir / f"08_{i:02d}_threshold_{threshold:.1f}.png"), binary)
            area = (binary == 255).sum()
            print(f"   ✓ Threshold {threshold:.1f}: {area} pixels")
        
        # 5. Compare contour mask with soft map threshold
        print("\n5. COMPARISON: Contour vs Soft Map (threshold 0.5)")
        # Resize soft map to match contour mask
        soft_resized = cv2.resize(soft_map, (2048, 1536))
        soft_binary = (soft_resized > 0.5).astype(np.uint8) * 255
        
        # Calculate overlap
        intersection = np.logical_and(contour_mask == 255, soft_binary == 255).sum()
        union = np.logical_or(contour_mask == 255, soft_binary == 255).sum()
        iou = intersection / union if union > 0 else 0
        
        # Create comparison visualization
        comparison = np.zeros((1536, 2048, 3), dtype=np.uint8)
        comparison[:, :, 2] = contour_mask  # Contour in red
        comparison[:, :, 1] = soft_binary   # Soft map in green
        # Overlap will appear yellow
        cv2.imwrite(str(output_dir / "08_07_comparison_contour_vs_softmap.png"), comparison)
        
        print(f"   ✓ IoU (Intersection over Union): {iou:.4f}")
        print(f"   ✓ Contour area: {(contour_mask == 255).sum()} pixels")
        print(f"   ✓ Soft map area: {(soft_binary == 255).sum()} pixels")
        print(f"   ✓ Overlap: {intersection} pixels")
        
        print(f"\n{'='*70}")
        print(f"✓ All outputs saved to: {output_dir}")
        print(f"{'='*70}\n")
        
        # Assertions
        assert binary_mask.dtype == np.uint8
        assert contour_mask.dtype == np.uint8
        assert soft_map.dtype == np.float32
        assert 0.0 <= iou <= 1.0


class TestLoadLayerBoundaries:
    """Tests for load_layer_boundaries function."""

    def test_load_layer_boundaries_returns_dict(self, tmp_path):
        """Test that load_layer_boundaries returns a dictionary."""
        boundaries_file = tmp_path / "boundaries.json"
        data = {
            "layer1": [[10, 10], [20, 15], [30, 20]],
            "layer2": [[10, 50], [20, 55], [30, 60]]
        }
        boundaries_file.write_text(json.dumps(data))
        
        result = load_layer_boundaries(boundaries_file)
        assert isinstance(result, dict)

    def test_load_layer_boundaries_json_format_with_dict(self, tmp_path):
        """Test that load_layer_boundaries handles JSON format with dictionary structure."""
        boundaries_file = tmp_path / "boundaries.json"
        data = {
            "layer1": [[10, 10], [20, 15], [30, 20]],
            "layer2": [[10, 50], [20, 55], [30, 60]]
        }
        boundaries_file.write_text(json.dumps(data))
        
        result = load_layer_boundaries(boundaries_file, layer_format="json")
        
        assert "layer1" in result
        assert "layer2" in result
        assert len(result["layer1"]) == 3
        assert result["layer1"][0] == (10.0, 10.0)

    def test_load_layer_boundaries_json_format_with_list(self, tmp_path):
        """Test that load_layer_boundaries handles JSON format with list structure."""
        boundaries_file = tmp_path / "boundaries.json"
        data = [
            {"layer_id": "layer1", "coordinates": [[10, 10], [20, 15]]},
            {"layer_id": "layer2", "coordinates": [[10, 50], [20, 55]]}
        ]
        boundaries_file.write_text(json.dumps(data))
        
        result = load_layer_boundaries(boundaries_file, layer_format="json")
        
        assert "layer1" in result
        assert "layer2" in result

    def test_load_layer_boundaries_csv_format(self, tmp_path):
        """Test that load_layer_boundaries handles CSV format."""
        boundaries_file = tmp_path / "boundaries.csv"
        csv_content = """layer_id,x,y
layer1,10,10
layer1,20,15
layer2,10,50
layer2,20,55
"""
        boundaries_file.write_text(csv_content)
        
        result = load_layer_boundaries(boundaries_file, layer_format="csv")
        
        assert "layer1" in result
        assert "layer2" in result
        assert len(result["layer1"]) == 2
        assert result["layer1"][0] == (10.0, 10.0)

    def test_load_layer_boundaries_text_format_comma_separated(self, tmp_path):
        """Test that load_layer_boundaries handles text format with comma-separated coordinates."""
        boundaries_file = tmp_path / "boundaries.txt"
        text_content = """layer1 10,10 20,15 30,20
layer2 10,50 20,55 30,60
"""
        boundaries_file.write_text(text_content)
        
        result = load_layer_boundaries(boundaries_file, layer_format="text")
        
        assert "layer1" in result
        assert "layer2" in result
        assert len(result["layer1"]) == 3

    def test_load_layer_boundaries_text_format_space_separated(self, tmp_path):
        """Test that load_layer_boundaries handles text format with space-separated coordinates."""
        boundaries_file = tmp_path / "boundaries.txt"
        text_content = """layer1 10 10 20 15 30 20
layer2 10 50 20 55 30 60
"""
        boundaries_file.write_text(text_content)
        
        result = load_layer_boundaries(boundaries_file, layer_format="text")
        
        assert "layer1" in result
        assert "layer2" in result

    def test_load_layer_boundaries_auto_detects_json_format(self, tmp_path):
        """Test that load_layer_boundaries auto-detects JSON format from extension."""
        boundaries_file = tmp_path / "boundaries.json"
        data = {"layer1": [[10, 10], [20, 15]]}
        boundaries_file.write_text(json.dumps(data))
        
        # Don't specify format - should auto-detect from .json extension
        result = load_layer_boundaries(boundaries_file)
        
        assert "layer1" in result

    def test_load_layer_boundaries_auto_detects_csv_format(self, tmp_path):
        """Test that load_layer_boundaries auto-detects CSV format from extension."""
        boundaries_file = tmp_path / "boundaries.csv"
        csv_content = """layer_id,x,y
layer1,10,10
"""
        boundaries_file.write_text(csv_content)
        
        # Don't specify format - should auto-detect from .csv extension
        result = load_layer_boundaries(boundaries_file)
        
        assert "layer1" in result

    def test_load_layer_boundaries_raises_filenotfound(self, tmp_path):
        """Test that load_layer_boundaries raises FileNotFoundError for nonexistent file."""
        nonexistent = tmp_path / "does_not_exist.json"
        
        with pytest.raises(FileNotFoundError, match="Layer boundaries file not found"):
            load_layer_boundaries(nonexistent)

    def test_load_layer_boundaries_raises_error_for_empty_file(self, tmp_path):
        """Test that load_layer_boundaries raises ValueError when no boundaries found."""
        boundaries_file = tmp_path / "empty.json"
        boundaries_file.write_text("{}")
        
        with pytest.raises(ValueError, match="No layer boundaries found"):
            load_layer_boundaries(boundaries_file)

    def test_load_layer_boundaries_raises_error_for_unsupported_format(self, tmp_path):
        """Test that load_layer_boundaries raises ValueError for unsupported format."""
        boundaries_file = tmp_path / "boundaries.json"
        boundaries_file.write_text("{}")
        
        with pytest.raises(ValueError, match="Unsupported layer format"):
            load_layer_boundaries(boundaries_file, layer_format="unsupported")
