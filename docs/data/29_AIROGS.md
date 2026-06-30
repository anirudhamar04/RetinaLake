# AIROGS

## Overview
The Rotterdam EyePACS AIROGS dataset contains 113,893 color fundus images from 60,357 subjects and approximately 500 different sites with a heterogeneous ethnicity. All images were assigned by human experts with the labels referable glaucoma, no referable glaucoma, or ungradable. The training set contains approximately 101,000 gradable images where only gradable images are considered and ungradable images excluded. The test set contains about 11,000 gradable and ungradable images (both gradable and ungradable), simulating a real-world scenario. The test set is closed and cannot be downloaded.

## Images
- All images are in **JPEG** format (stored as both `.JPG` and `.jpg` extensions).
- Images are labeled with challenge IDs following the pattern `TRAIN` followed by a 6-digit zero-padded number (e.g., `TRAIN000000.JPG`, `TRAIN101441.JPG`).
- The training set contains 101,442 gradable fundus images.
- Some images in the test directory have variant suffixes (e.g., `TRAIN000100_ANRI.JPG`, `TRAIN000100_ANRS.JPG`, `TRAIN000100_LC.JPG`, `TRAIN000100_LD.JPG`, `TRAIN000100_NVT.JPG`).
## Metadata
The metadata for the training set is stored in a single CSV file named **train_labels.csv**, which contains binary classification labels for each gradable image.

### train_labels.csv
The CSV file contains the following columns:
- **challenge_id** -- Image identifier following the pattern `TRAIN` followed by a 6-digit zero-padded number (e.g., `TRAIN000000`, `TRAIN101441`).
- **class** -- Binary classification label for referable glaucoma:
- `RG` -- Referable Glaucoma (eye should be referred to specialist)
- `NRG` -- No Referable Glaucoma (eye does not need referral)
 The dataset contains 101,442 labeled training samples with approximately 3,270 RG (Referable Glaucoma) and 98,172 NRG (Non-Referable Glaucoma) images, representing a highly imbalanced class distribution.

## Splits
The dataset provides explicit splits between training and test sets. The training set contains approximately 101,000 gradable images and is available for download. The test set contains about 11,000 images (both gradable and ungradable) and is closed, meaning it cannot be downloaded. Test set evaluation is performed through a Docker container submission to the Grand Challenge evaluation platform.
## File Schema
```
29_AIROGS/
documents/
*.JPG
documents_org/
*.jpg
documents_test/
*.JPG
train_labels.csv
```

## Storage in database

### Tables populated

**`datasets`**
One record is inserted for AIROGS with `modality_types=["fundus"]`.

**`images`**
One row per row in `train_labels.csv` for which an image file can be found. Image files are looked up by `challenge_id` in two directories, tried in order: `documents/` then `documents_org/`. Within each directory both `.JPG` and `.jpg` extensions are tried. Image metadata is extracted from the physical file. Fields stored:
- `original_image_id`: the `challenge_id` string (e.g., `TRAIN000000`).
- `modality="fundus"`.
- No laterality, acquisition date, or quality metadata is set.

**`classification_annotations`**
One binary classification annotation per image:
- `task_type="binary"`, `class_name="glaucoma"`, `annotation_method="manual"`.
- The `class` column is read from `train_labels.csv` and must be `"RG"` or `"NRG"` (case-insensitive comparison after upper-casing).
- Mapping: `"RG"` (Referable Glaucoma) → `True`, `"NRG"` (Non-Referable Glaucoma) → `False`.
- Label dictionary stored in annotation: `{True: "RG", False: "NRG"}`.
- Provenance linked to the registered `train_labels.csv` raw file via the context set by `process_csv`.

### Annotation types
- `classification` — binary referable glaucoma label (RG / NRG) from `train_labels.csv`.

### Splits
A single `"train"` split is registered with `split_type="explicit"`. All successfully ingested images are assigned to it. No test split is created (the test set is closed and not downloadable).

### Provenance / raw annotation files
`train_labels.csv` is registered in `raw_annotation_files` with `unified_annotation_type="classification"`. All classification annotations are linked to it via a provenance chain provided by `process_csv`.

### Special processing
- Image file discovery tries two directories (`documents/`, `documents_org/`) and two case variants (`.JPG`, `.jpg`) per directory, stopping at the first match.
- Rows with invalid class labels (anything other than `RG` or `NRG`) are skipped with an error recorded.
- The test set (stored in `documents_test/`) is not ingested (no labels available).