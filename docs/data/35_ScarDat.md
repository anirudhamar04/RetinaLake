# ScarDat

## Overview
The ScarDat dataset is a binary classification dataset for scar detection in retinal fundus images. The dataset contains images organized into training, validation, and test splits, with each split containing positive (scar present) and negative (no scar) samples.

## Images
- All images are in **JPG** format.
- Images are organized into `positive/` and `negative/` subdirectories within each split.
- Image filenames follow multiple naming conventions:
- Numeric IDs with eye laterality indicator (e.g., `26054_left.jpg`, `31108_left.jpg`, `11629_right.jpg`)
- Date-based format (e.g., `20140610-14081-42806.jpg`, `20140529-13767-41774.jpg`)
- Simple numeric format (e.g., `136-95517.jpg`, `141-96220.jpg`, `2-71850.jpg`)
- UUID format (in `test80/` subdirectory, e.g., `001da5ac-f79c-5ff2-992a-d8e7e33a9147.jpg`)
- Alphanumeric codes (in `test80/` subdirectory, e.g., `001-SKNQ.jpg`, `008-TUOZ.jpg`)
- Images span dates from 2014 to 2016 based on date-based filenames.
- Many images include left/right eye indicators in their filenames or metadata.
## Metadata
The metadata for each split is stored in two CSV files: `kaggle_label.csv` and `non-kaggle_label.csv`. Each file contains image labels in a two-column format.

### kaggle_label.csv
The `kaggle_label.csv` file is present in train, val, and test directories. It contains the following columns:
- **image_id** - Image identifier matching the filename (without extension). Examples include `26054_left`, `20140610-14081-42806`, `136-95517`.
- **label** - Binary classification label:
- 0 - Negative (no scar)
- 1 - Positive (scar present)
 The train split contains 6,903 entries, the val split contains 987 entries (33 positive, 954 negative), and the test split contains 1,974 entries.

### non-kaggle_label.csv
The `non-kaggle_label.csv` file is present in train, val, and test directories. It contains the same column structure as `kaggle_label.csv`:
- **image_id** - Image identifier matching the filename (without extension). Examples include `20-72557`, `20141223-19746-57735`, `20141016-18047-52958`.
- **label** - Binary classification label:
- 0 - Negative (no scar)
- 1 - Positive (scar present)
 The train split contains 700 entries, the val split contains 100 entries, and the test split contains 200 entries.

## Splits
The dataset provides explicit train, validation, and test splits. Each split contains:
- `positive/` subdirectory with positive samples
- `negative/` subdirectory with negative samples
- `kaggle_label.csv` with labels
- `non-kaggle_label.csv` with alternative labels
The test split additionally contains a `test80/` subdirectory with 41 images using different naming conventions (UUIDs and alphanumeric codes).

## File Schema
```
35_ScarDat/
train/
positive/
negative/
kaggle_label.csv
non-kaggle_label.csv
val/
positive/
negative/
kaggle_label.csv
non-kaggle_label.csv
test/
positive/
negative/
test80/
kaggle_label.csv
non-kaggle_label.csv
```

## Storage in database

### Tables populated

- **`datasets`**: One record for ScarDat (name=`"ScarDat"`, source_url=`https://www.kaggle.com/datasets/andrewmvd/retinal-scar-detection`, license=`CC0: Public Domain`, `modality_types=['fundus']`).
- **`images`**: One row per image found in `positive/`, `negative/`, and (test split only) `test80/` subfolders across all three splits. `original_image_id` is the file stem for images from `positive/`and `negative/` folders; for `test80/` images the `original_image_id` is prefixed with `"test80_"`. Image UUIDs are generated from `"{split_name}_{image_stem}"` (or `"{split_name}_test80_{image_stem}"` for test80 images) to ensure uniqueness across splits. `modality='fundus'`. Image metadata (dimensions, format, etc.) is extracted via `get_image_metadata_dict`.
- **`classification_annotations`**: One `ClassificationAnnotation` per image from `positive/` or `negative/` folders (`task_type="binary"`, `class_name="retinal_scar"`, `class_labels={0: "negative", 1: "positive"}`). `class_value` is a boolean. `annotation_method="manual"`. Images from `test80/` receive **no** classification annotation.
  - **Label resolution**: CSV labels are loaded from `kaggle_label.csv` and `non-kaggle_label.csv` (both are read and merged; later file overwrites earlier on conflict). For each image, the script first looks up the image stem in the merged CSV labels. If found, that label is used; if not found, the folder (`positive/` or `negative/`) determines the label. In case of a CSV/folder mismatch, the CSV label takes precedence and a warning is logged.
  - **Laterality suffix stripping**: When looking up CSV labels, if the image stem contains `_left` or `_right`, a base stem (without the suffix) is also tried as a fallback key.
- **`raw_annotation_files`**: Registered automatically per `process_folder_tree` call (one raw file / provenance chain per call to `process_folder_tree` for the `positive/` and `negative/` directories — two calls per split). The CSV files (`kaggle_label.csv`, `non-kaggle_label.csv`) are read by helper functions but are **not** explicitly registered as raw annotation files via `process_csv`; they are used only for label lookup.
- **`provenance_chains`**: One chain per `process_folder_tree` call.
- **`dataset_splits`**: Three splits — `"train"`, `"test"`, `"val"` — registered as `split_type="explicit"`.
- **`image_splits`**: Each image is assigned to its corresponding split (`train`, `val`, or `test`).

### Annotation types
- **`classification_annotations`**: `class_name="retinal_scar"`, `task_type="binary"` (True=scar present, False=no scar). Labels sourced primarily from CSV files, with folder structure as fallback.

### Splits created
- `train` (explicit): images from `train/positive/` and `train/negative/`.
- `val` (explicit): images from `val/positive/` and `val/negative/`.
- `test` (explicit): images from `test/positive/`, `test/negative/`, and `test/test80/` (unannotated).

### Provenance / raw annotation files registered
- One raw file registration per `process_folder_tree` call (i.e., per `positive/` and `negative/` directory per split). The CSV label files are read for label resolution but not registered as raw annotation files.

### Special processing
- Images in `test80/` (test split only) are processed directly (not via `process_folder_tree`) and stored as images only — no classification is created for them, and no provenance chain is assigned.
- The CSV files are validated against the folder structure via `validate_csv_vs_folder_structure`, which logs matches, mismatches, and images found in one source but not the other. This validation is informational only and does not affect ingestion.
- Only `.jpg`, `.JPG`, `.jpeg`, `.JPEG` files are processed.
- The ingestion process is idempotent. Re-running it will not create duplicate images or annotations due to deterministic UUID generation and upsert logic.