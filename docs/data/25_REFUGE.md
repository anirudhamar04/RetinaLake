# REFUGE (Retinal Fundus Glaucoma Challenge)

## Overview

The dataset was designed for two primary tasks: optic disc and optic cup segmentation, and glaucoma classification. It consists of 1200 color fundus images with associated ground truth segmentations and clinical glaucoma labels, making it the largest publicly available glaucoma dataset at the time of release. 
An evaluation framework was provided as part of the challenge to enable fair comparison of different methods. A total of 12 teams participated in the online challenge, and results showed that some top-ranked methods outperformed human experts in glaucoma classification.

## Images
- Images are color fundus photographs (CFP).
- The dataset contains a total of 1200 images, divided equally into training, validation, and test sets.
- Training images have a resolution of 2124 × 2056 pixels.
- Validation and test images have a resolution of 1634 × 1634 pixels (square images).
- Multiple image variants are provided:
- Original images.
- Cropped images.
- Square-format images.
- Image naming conventions encode dataset split and, for training images, class labels:
- `gXXXX.jpg` -- glaucoma images in the training set.
- `nXXXX.jpg` -- normal images in the training set.
- `VXXXX.jpg` -- validation images.
- `TXXXX.jpg` -- test images.
- Ground truth segmentation masks are provided in **BMP** format.
- Additional segmentation masks are provided in **PNG** format.
## Metadata
Metadata for the dataset is provided in JSON format, with one `index.json` file per dataset split (train, validation, and test).

#### train/index.json
Contains metadata for 400 training images.
- **ImgName** -- Image filename (e.g., `g0001.jpg`, `n0001.jpg`).
- **Fovea_X** -- X coordinate of the fovea center (decimal value).
- **Fovea_Y** -- Y coordinate of the fovea center (decimal value).
- **Size_X** -- Image width in pixels (2124).
- **Size_Y** -- Image height in pixels (2056).
- **Label** -- Binary glaucoma label: 0 (normal), 1 (glaucoma).
#### val/index.json
Contains metadata for 400 validation images.
- **ImgName** -- Image filename (e.g., `V0001.jpg`).
- **Fovea_X** -- X coordinate of the fovea center (decimal value).
- **Fovea_Y** -- Y coordinate of the fovea center (decimal value).
- **Size_X** -- Image width in pixels (1634).
- **Size_Y** -- Image height in pixels (1634).
- **Label** -- Binary glaucoma label: 0 (normal), 1 (glaucoma).
#### test/index.json
Contains metadata for 400 test images.
- **ImgName** -- Image filename (e.g., `T0001.jpg`).
- **Size_X** -- Image width in pixels (1634).
- **Size_Y** -- Image height in pixels (1634).
Fovea coordinates and labels are not provided for the test set.

## Splits
Explicit training, validation, and test splits are provided:
- **Train**: 400 images (40 glaucoma, 360 normal).
- **Validation**: 400 images.
- **Test**: 400 images (labels not provided).
## File Schema
```
25_REFUGE/
train/
Images/
Images_Cropped/
Masks/
Masks_Cropped/
gts/
illustrations/
index.json
val/
Images/
Images_Cropped/
Masks/
Masks_Cropped/
gts/
index.json
test/
Images/
Images_Cropped/
Masks/
Masks_Cropped/
gts/
index.json
Images_Square/
Masks_Square/
```

**Note on Images_Square and Masks_Square Directories**:
The Images_Square/ and Masks_Square/ directories at the root level contain square-format versions of all 1,200 images and masks from the train, validation, and test splits, consolidated into a single directory. These are the same images found in train/Images/, val/Images/, and test/Images/, but preprocessed to square dimensions. The naming convention remains consistent: files with g and n prefixes correspond to training images (40 glaucoma + 360 normal), V prefix to validation images (400), and T prefix to test images (400). Each image in Images_Square/ has a corresponding mask in Masks_Square/ with the same filename (.jpg for images, .png for masks).

## Storage in database

### Tables populated

**`datasets`**
One record is inserted for REFUGE with `modality_types=["fundus"]` and `task_types=["classification", "segmentation", "localization"]`.

**`images`**
One row per entry in each split's `index.json`. Image metadata is extracted from the physical file in `<split>/Images/`. Fields stored:
- `original_image_id`: stem of the `ImgName` field (e.g., `g0001`).
- `modality="fundus"`, `acquisition_date=None`, `image_quality=None`.
- No laterality is set (not provided).
- Image UUID is generated from `dataset_id` + `"{split}_{original_image_id}"` to avoid collisions across splits with overlapping stems.

**`classification_annotations`**
One binary classification annotation per image in `train` and `val` splits (the `test` split has no `Label` field):
- `task_type="binary"`, `class_name="glaucoma"`, `annotation_method="manual"`.
- Labels: `0 → "normal"`, `1 → "glaucoma"`.
- Provenance linked to the respective split's `index.json` raw file.

**`localization_annotations`**
One keypoint localization per image in `train` and `val` splits (fovea coordinates are absent from the `test` split `index.json`):
- `localization_type="keypoint"`, `target_structure="fovea"`.
- Coordinates are the `Fovea_X` and `Fovea_Y` values from `index.json`, normalized via `normalize_keypoint_coordinates`.
- `annotation_method="manual"`.
- Provenance linked to the respective split's `index.json` raw file.
- Localization UUID is derived from `image_id`, type, target structure, raw_data_id, and a SHA-256 hash of the coordinate payload.

**`segmentation_annotations`**
Up to two segmentation annotations (optic disc and optic cup) per image for which a mask file exists:
- The script looks for a mask first in `<split>/Masks/<stem>.png`, then falls back to `<split>/gts/<stem>.bmp`.
- Mask is processed as a multi-class mask with class mapping `{1: "optic_disc", 2: "optic_cup"}`, using `fill_holes=False`.
- If multiclass processing fails, a fallback attempts binary mask extraction of class 1 (`optic_disc`) only.
- `annotation_method="manual"`.
- Each mask file is registered individually in `raw_annotation_files` with `file_type=None`, `auto_detect_type=False`.

### Annotation types
- `classification` — binary glaucoma label (normal / glaucoma), train and val splits only.
- `localization` — fovea keypoint coordinates, train and val splits only.
- `segmentation` — optic disc and optic cup from PNG or BMP mask files, all splits that have masks.

### Splits
Three splits are registered with `split_type="explicit"`:
- `"train"` — 400 images.
- `"val"` — 400 images.
- `"test"` — 400 images (no classification labels or fovea coordinates).

### Provenance / raw annotation files
- Each split's `index.json` (`train/index.json`, `val/index.json`, `test/index.json`) is registered in `raw_annotation_files` with `unified_annotation_type="classification"`. Classification and localization annotations for that split are linked to their respective JSON file.
- Each mask file (PNG or BMP) that is found is registered individually with `unified_annotation_type="segmentation"`, `file_type=None`. Segmentation annotations are linked to their respective mask file's provenance chain.

### Special processing
- All three splits (`train`, `val`, `test`) are processed sequentially; test images are ingested (image rows created) even though they carry no annotations.
- The image UUID namespace includes the split name to prevent ID collision between splits that might share `ImgName` stems.
- Mask lookup order: `Masks/<stem>.png` first, then `gts/<stem>.bmp`. Missing masks are logged at DEBUG level and do not fail the entry.