# Generated Test Masks

This directory contains masks generated during testing of the mask converter utilities. All masks are created from real medical imaging data in the `/data` folder.

## Directory Structure

### 01. Single ORIGA Mask Validation
- **`01_validated_origa_125.png`** - Binary mask validated from ORIGA dataset (mask #125)
  - Source: `/data/24_ORIGA/Masks/125.png`
  - Type: Binary mask (0 and 255 values)

### 02. Multiple ORIGA Masks
- **`02_validated_origa_masks/`** - Collection of 5 validated ORIGA masks
  - `02_01_validated_334.png`
  - `02_02_validated_534.png`
  - `02_03_validated_424.png`
  - `02_04_validated_352.png`
  - `02_05_validated_620.png`
  - All validated to ensure proper binary format (0 and 255)

### 03. Contour to Binary Mask
- **`03_contour_mask_drishti_031_OD.png`** - Optic disc mask from contour file
  - Source: Drishti-GS1 dataset (drishtiGS_031 OD average boundary)
  - Converted from text coordinates to binary mask
  - Size: 2048×1536 pixels
  - Area: ~110,971 pixels

### 04. Optic Disc & Cup Contours
- **`04_drishti_contour_masks/`** - OD and Cup masks with combined visualization
  - `04_01_OD_mask.png` - Optic disc mask (area: ~110,971 pixels)
  - `04_02_Cup_mask.png` - Cup mask (area: ~54,999 pixels)
  - `04_03_Combined_OD_Cup.png` - Composite view (OD=gray 128, Cup=white 255)
  - Cup/Disc Ratio: ~0.496 (clinically relevant metric for glaucoma)

### 05. Soft Map Thresholding
- **`05_softmaps/`** - Probability maps and thresholded versions
  - `05_01_OD_softmap.png` - Raw probability map (0-1 range converted to 0-255)
  - `05_02_OD_binary_threshold_0.3.png` - Thresholded at 30% probability
  - `05_03_OD_binary_threshold_0.5.png` - Thresholded at 50% probability
  - `05_04_OD_binary_threshold_0.7.png` - Thresholded at 70% probability
  - Shows how different thresholds affect segmentation

### 06. Combined OD & Cup Soft Maps
- **`06_combined_softmaps/`** - Dual soft map visualization
  - `06_01_OD_softmap.png` - Optic disc probability map
  - `06_02_Cup_softmap.png` - Cup probability map
  - `06_03_Combined_RGB_visualization.png` - RGB overlay (Red=OD, Green=Cup)
  - Mean probabilities: OD ~0.025, Cup ~0.013

### 07. XML vs Contour Format Comparison
- **`07_xml_vs_contour/`** - Validation that different formats produce identical masks
  - `07_01_xml_mask.png` - Mask from XML polygon format
  - `07_02_contour_mask.png` - Mask from text contour format
  - Both masks are pixel-perfect identical (validated)

### 08. Comprehensive Pipeline
- **`08_COMPREHENSIVE_PIPELINE/`** - Full end-to-end processing demonstration
  1. `08_01_validated_binary_mask.png` - Binary mask validation
  2. `08_02_contour_to_binary_mask.png` - Contour→binary conversion
  3. `08_03_softmap.png` - Probability map loading
  4. `08_04_threshold_0.3.png` - Soft map thresholded at 0.3
  5. `08_05_threshold_0.5.png` - Soft map thresholded at 0.5
  6. `08_06_threshold_0.7.png` - Soft map thresholded at 0.7
  7. `08_07_comparison_contour_vs_softmap.png` - RGB comparison (Red=contour, Green=soft map, Yellow=overlap)
     - IoU (Intersection over Union) can be computed from overlap

### 09. Multi-Class Extraction
- **`09_multiclass_extraction/`** - Extracting individual classes from multi-class masks
  - `09_01_disc_only.png` - Optic disc only (class 1 from ORIGA, original)
  - `09_02_cup_only.png` - Optic cup only (class 2 from ORIGA, original)
  - `09_03_merged_disc_cup.png` - Merged disc + cup into single foreground
  - `09_04_disc_filled.png` - **Optic disc with holes filled (103,211 pixels)**
  - `09_05_cup_filled.png` - Optic cup with holes filled
  - `09_06_merged_filled.png` - Merged mask with holes filled
  - **Source**: ORIGA mask #125 (multi-class: 0=background, 1=disc, 2=cup)
  - **Original disc area**: ~69,992 pixels (outline only)
  - **Filled disc area**: ~103,211 pixels (completely filled)
  - **Cup area**: ~33,219 pixels (already filled, no change)
  - **Cup/Disc ratio**: 0.322 (clinically significant for glaucoma assessment)

### 10. Validate Binary Mask with Multi-Class Options
- **`10_validate_with_multiclass_options/`** - Using `validate_binary_mask()` with multi-class support
  - `10_01_extract_disc.png` - Disc extracted using `extract_class=1` (original, 69,992px)
  - `10_02_extract_cup.png` - Cup extracted using `extract_class=2` (33,219px)
  - `10_03_merge_all.png` - All classes merged using `merge_nonzero=True` (103,211px)
  - `10_04_extract_disc_filled.png` - **Disc with holes filled (103,211px)**
  - `10_05_extract_cup_filled.png` - Cup with holes filled (33,219px)
  - `10_06_merge_all_filled.png` - Merged with holes filled (103,211px)
  - Demonstrates new kwargs: `extract_class`, `merge_nonzero`, and `fill_holes`

## Data Sources

- **ORIGA**: Optic disc and cup segmentation masks
- **Drishti-GS1**: Optic disc contours and soft probability maps
  - Training set: drishtiGS_031 used for demonstrations
  - Contains both average boundaries (contours) and soft maps (probabilities)

## Multi-Class Mask Functions

The mask converter now includes powerful multi-class mask handling:

### Helper Functions

```python
from chaksudb.ingest.framework.mask_converter import (
    is_multiclass_mask,
    get_mask_classes,
    extract_class_from_mask,
    extract_classes_from_multiclass_mask,
)

# Check if mask is multi-class
is_multi = is_multiclass_mask(mask_path)  # Returns True if > 2 unique values

# Get all class IDs (excluding background)
classes = get_mask_classes(mask_path)  # Returns array like [1, 2, 5]

# Extract single class as binary mask
disc_mask = extract_class_from_mask(mask_path, class_id=1)

# Extract with hole filling (for masks with only boundaries)
disc_filled = extract_class_from_mask(mask_path, class_id=1, fill_holes=True)

# Extract all classes with custom names
masks = extract_classes_from_multiclass_mask(
    mask_path,
    class_names={1: "optic_disc", 2: "cup"},
    classes_to_extract=[1, 2]
)
# Returns: {"optic_disc": binary_mask1, "cup": binary_mask2}

# Extract with hole filling
masks_filled = extract_classes_from_multiclass_mask(
    mask_path,
    class_names={1: "optic_disc", 2: "cup"},
    fill_holes=True
)

# Merge multiple classes into single foreground
merged = extract_classes_from_multiclass_mask(
    mask_path,
    merge_classes=[1, 2]
)
# Returns: {"merged": binary_mask}
```

### Updated `validate_binary_mask()`

The main validation function now supports multi-class masks:

```python
from chaksudb.ingest.framework.mask_converter import validate_binary_mask

# Original behavior - binary masks work as before
binary_mask = validate_binary_mask(path)

# Extract specific class from multi-class mask
disc_only = validate_binary_mask(path, extract_class=1)
cup_only = validate_binary_mask(path, extract_class=2)

# Extract with hole filling (for boundary-only masks)
disc_filled = validate_binary_mask(path, extract_class=1, fill_holes=True)

# Merge all non-zero classes into foreground
combined = validate_binary_mask(path, merge_nonzero=True)

# Merge with hole filling
combined_filled = validate_binary_mask(path, merge_nonzero=True, fill_holes=True)
```

## Usage

These masks are generated during pytest runs and can be used to:
1. Verify mask converter functions work correctly
2. Inspect visual quality of generated masks
3. Debug issues with coordinate conversion
4. Understand different mask formats (binary, probability, contours, multi-class)
5. Validate multi-class mask extraction and class separation

## Regeneration

To regenerate all masks, run:
```bash
uv run pytest tests/test_framework/test_mask_converter.py -v
```

Or run specific tests:
```bash
# Comprehensive pipeline (generates all visualization types)
uv run pytest tests/test_framework/test_mask_converter.py::TestMaskConverterIntegration::test_comprehensive_mask_processing_pipeline -v -s

# OD and Cup contours
uv run pytest tests/test_framework/test_mask_converter.py::TestConvertContourToBinaryMask::test_convert_contour_to_binary_mask_optic_disc_and_cup -v -s

# Soft maps
uv run pytest tests/test_framework/test_mask_converter.py::TestLoadSoftMap::test_load_soft_map_od_and_cup -v -s
```

## Notes

- All masks are PNG format for lossless storage
- Binary masks use 0 (background) and 255 (foreground)
- Soft maps are stored as grayscale 0-255 (representing probabilities 0.0-1.0)
- RGB visualizations use color channels to show multiple masks simultaneously
- Numbered prefixes (01_, 02_, etc.) organize files by test category
