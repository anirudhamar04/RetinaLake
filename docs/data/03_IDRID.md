# IDRID
## Overview
The Indian Diabetic Retinopathy Image Dataset (IDRiD) was created from real clinical examinations acquired at an eye clinic in India. The images were captured with a 50-degree field of view and have a resolution of 4288 × 2848 pixels. 
The dataset consists of 516 retinal fundus images annotated according to international clinical standards. It provides expert annotations for typical Diabetic Retinopathy (DR) lesions, normal retinal structures, and disease severity levels for both Diabetic Retinopathy (DR) and Diabetic Macular Edema (DME). In this work, only the severity levels of DR and DME are considered.
The dataset is divided into three parts A. Segmentation B. Disease Grading and C. Localisation 

## Images
The IDRiD dataset is divided into three major components: Segmentation, Disease Grading, and Localization.
- All original fundus images are color retinal images stored in **JPG** format.
- Images have a fixed resolution of 4288 × 2848 pixels with a 50-degree field of view.
- The dataset contains a total of 516 images, with predefined training and testing splits depending on the task.
- Groundtruth images for segmentation, for the Lesions (Microaneurysms, Haemorrhages, Hard Exudates and Soft Exudates divided into train and test set - TIF Files) and Optic Disc (divided into train and test set - TIF Files)
## Metadata
The dataset provides task-specific metadata in the form of CSV files corresponding to disease grading and anatomical localization annotations.

### Disease Grading Metadata
The disease grading annotations are provided through two CSV files:
- `a. IDRiD_Disease Grading_Training Labels.csv`
- `b. IDRiD_Disease Grading_Testing Labels.csv`
Each CSV file contains the following columns:
- **Image name** - Image identifier without file extension.
- **Retinopathy grade** - Diabetic Retinopathy severity on a scale of 0–4:
- 0 - No DR
- 1 - Mild DR
- 2 - Moderate DR
- 3 - Severe DR
- 4 - Proliferative DR (PDR)
- **Risk of macular edema** - Diabetic Macular Edema (DME) severity on a scale of 0–2.
Note that the training CSV file contains additional trailing columns that are empty and can be ignored.

### Localization Metadata
Localization annotations are provided via separate CSV files for optic disc and fovea center locations:
- `a. IDRiD_OD_Center_Training Set_Markups.csv`
- `b. IDRiD_OD_Center_Testing Set_Markups.csv`
- `IDRiD_Fovea_Center_Training Set_Markups.csv`
- `IDRiD_Fovea_Center_Testing Set_Markups.csv`
Each CSV file contains the following columns:
- **Image No** - Image identifier.
- **X-Coordinate** - Horizontal pixel coordinate of the anatomical center.
- **Y-Coordinate** - Vertical pixel coordinate of the anatomical center.
Additional trailing columns present in these CSV files are empty.

## Splits
The dataset provides explicit train–test splits for each task:
- **Segmentation**: 81 images divided into training and testing sets.
- **Disease Grading**: 516 images divided into 413 training images and 103 testing images.
- **Localization**: 516 images divided into 413 training images and 103 testing images.
## File Schema
```
03_IDRID/
A. Segmentation/
1. Original Images/
a. Training Set/
b. Testing Set/
2. All Segmentation Groundtruths/
a. Training Set/
1. Microaneurysms/
2. Haemorrhages/
3. Hard Exudates/
4. Soft Exudates/
5. Optic Disc/
b. Testing Set/
1. Microaneurysms/
2. Haemorrhages/
3. Hard Exudates/
4. Soft Exudates/
5. Optic Disc/
B. Disease Grading/
1. Original Images/
a. Training Set/
b. Testing Set/
2. Groundtruths/
C. Localization/
1. Original Images/
a. Training Set/
b. Testing Set/
2. Groundtruths/
1. Optic Disc Center Location/
2. Fovea Center Location/
```

## Storage in database

### Ingestion phases

Ingestion is divided into two phases processed independently before a combined bulk upsert.

**Phase 1 — Disease Grading + Localization (Task B and Task C, 51 shared images)**

- **`images`** — One row per image from `B. Disease Grading/1. Original Images/` (training and testing sets). Fields populated via `get_image_metadata_dict` plus `modality="fundus"`. No laterality is stored. The `original_image_id` is the image filename stem (e.g., `IDRiD_01`).
- **`disease_grading`** — Two rows per image:
  - DR grade from the `Retinopathy grade` column, stored under scale `ICDR_0_4` (0=No DR, 1=Mild, 2=Moderate, 3=Severe, 4=Proliferative DR). `annotation_method="manual"`.
  - DME grade from the `Risk of macular edema` column, stored under a custom scale `IDRID_DME_0_2` (0=No DME, 1=Mild/Moderate DME, 2=Severe DME). `annotation_method="manual"`.
- **`localization_annotations`** — One row per image per structure (optic disc center and fovea center), sourced from four localization CSVs (OD train, OD test, fovea train, fovea test). Each annotation has `localization_type="center_point"`, `target_structure` set to `"optic_disc"` or `"fovea"`, and `coordinates` as `{"x": float, "y": float}`. Empty or malformed rows are silently skipped.

**Phase 2 — Segmentation (Task A, 7 separate images: 5 train, 2 test)**

- **`images`** — One row per image from `A. Segmentation/1. Original Images/`. These are distinct images from Phase 1.
- **`segmentation_annotations`** — Up to 5 rows per image, one per lesion type present:
  - Microaneurysms (`annotation_type="lesions"`, `lesion_subtype="MA"`, suffix `_MA`)
  - Haemorrhages (`annotation_type="lesions"`, `lesion_subtype="HE"`, suffix `_HE`)
  - Hard Exudates (`annotation_type="lesions"`, `lesion_subtype="EX"`, suffix `_EX`)
  - Soft Exudates (`annotation_type="lesions"`, `lesion_subtype="SE"`, suffix `_SE`)
  - Optic Disc (`annotation_type="optic_disc"`, no subtype, suffix `_OD`)
  - Mask files are `.tif` binary masks located under `A. Segmentation/2. All Segmentation Groundtruths/`. Missing masks are skipped without error. Masks are standardised to PNG via `process_segmentation_from_binary_mask`. `annotation_method="manual"`.

### Provenance / raw annotation files

- The four grading/localization CSVs are registered in `raw_annotation_files` via `process_csv` (with `annotation_type="grading"` or `"localization"`) and each gets a `provenance_chain` entry.
- Each `.tif` segmentation mask file is individually registered in `raw_annotation_files` via `register_individual_file` with `file_type=None` (auto-detect disabled) and `annotation_type="segmentation"`.

### Splits

Two explicit splits are created: `train` and `test`. Phase 1 images are assigned based on which training/testing subdirectory they come from. Phase 2 segmentation images are similarly split. All images from both phases are combined and assigned in bulk.

### Special processing

- Tasks B and C share the same 51 images; images are registered only once (a `processed_images` set prevents duplication when both CSVs reference the same image).
- The four localization CSVs are processed concurrently via `asyncio.gather`.
- Column names in grading and localization CSVs may contain trailing spaces or variations; the script uses fuzzy key matching to locate the correct column.
- Localizations and segmentations are individually upserted (no bulk operation); errors per item are tracked but do not abort the overall ingestion.
- The script is idempotent: re-running does not create duplicates due to deterministic UUID generation and upsert logic.