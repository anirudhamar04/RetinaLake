# ROC

## Overview
The ROC dataset consists of retinal fundus images. The dataset contains 100 images total, with 50 unique image identifiers, where each identifier appears in both test and training splits.

## Images
- All images are in **JPEG** format.
- Images are retinal fundus photographs showing the posterior pole of the retina, including the optic disc, macula, and vascular structures.
- Images are labeled with a numerical identifier and split designation (e.g., `image0_test.jpg` is image ID 0 in the test split, `image0_training.jpg` is image ID 0 in the training split).
- Image identifiers range from 0 to 49.
- Image resolution, field of view, and camera specifications are not provided.
- Eye laterality (left/right) information is not explicitly encoded in the filename or provided metadata.
## Metadata
No metadata files (CSV, JSON, or other structured formats) are provided with this dataset. All information is encoded in the image filenames.

## Splits
The dataset provides explicit test and training splits. Each of the 50 unique image identifiers (0-49) has a corresponding file in both the test split (suffix `_test.jpg`) and the training split (suffix `_training.jpg`), resulting in 50 test images and 50 training images.

## File Schema
```
26_ROC/
image0_test.jpg
image0_training.jpg
image1_test.jpg
image1_training.jpg
...{
``` (images 2-49 in both test and training).
.2 image49_test.jpg.
.2 image49_training.jpg.
}

## Storage in database

### Tables populated

**`datasets`**
One record is inserted for ROC with `modality_types=["fundus"]`. No `task_types` or description fields are set beyond the programmatic description string.

**`images`**
One row per image file found (non-recursively) in the `26_ROC/` data root. Image metadata is extracted from the physical file. Fields stored:
- `original_image_id`: file stem (e.g., `image0_test`, `image0_training`).
- `modality="fundus"`.
- No laterality, acquisition date, or quality metadata is set (none available).

Images whose filenames do not match the expected pattern `image{number}_{test|training}.{jpg|jpeg|png}` (case-insensitive) are skipped with an error recorded.

### Annotation types
None. The dataset provides images only; no classification, segmentation, localization, or grading annotations are stored.

### Splits
Two splits are registered with `split_type="explicit"`:
- `"train"` ← images whose filename ends with `_training`.
- `"test"` ← images whose filename ends with `_test`.

Split membership is determined entirely by parsing the filename suffix at ingest time.

### Provenance / raw annotation files
No annotation files are registered in `raw_annotation_files`. All information (including split assignment) is derived from the image filenames; there are no separate annotation or metadata files.

### Special processing
- Images are discovered non-recursively from the dataset root directory.
- The filename parser maps `training` → `"train"` for consistency with the standard split naming scheme.
- No annotations of any kind are produced; the dataset is images-only in the database.