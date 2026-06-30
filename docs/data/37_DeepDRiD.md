# DeepDRiD

## Overview
The Deep Diabetic Retinopathy Image Dataset (DeepDRiD) is a dataset designed for diabetic retinopathy grading and image quality estimation. The dataset was used in the ISBI-2020 Challenge 5: Diabetic Retinopathy Assessment Grading and Diagnosis (AM Session). The challenge is subdivided into three tasks: (A) Dual-View Disease Grading: Classification of fundus images according to the severity level of diabetic retinopathy using dual view retinal fundus images; (B) Image Quality Estimation: Fundus quality assessment for overall image quality, artifacts, clarity, and field definition; (C) Transfer Learning: Explore the generalizability of a Diabetic Retinopathy (DR) grading system. The dataset contains both regular fundus images and ultra-widefield fundus images, organized into training, validation, and online challenge evaluation sets.

## Images
- All images are in **JPG** format.
- The dataset contains two types of fundus images:
- **Regular fundus images**: Standard fundus photographs organized by patient ID in subdirectories.
- **Ultra-widefield images**: Ultra-widefield fundus images organized by patient ID in subdirectories.
- Image naming convention for regular fundus images: `{patient_id`_{eye}_{image_number}.jpg}, where `{eye`} is `l` (left) or `r` (right), and `{image_number`} is a sequential number (e.g., `1_l1.jpg`, `1_l2.jpg`, `1_r1.jpg`, `1_r2.jpg`).
- Image naming convention for ultra-widefield images: Same as regular fundus images (e.g., `1_l2.jpg`, `1_r1.jpg`).
- Images are organized in patient-based subdirectories within `Images/` folders (e.g., `1/`, `2/`, `265/`).
- Regular fundus images typically have 4 images per patient (2 per eye), while ultra-widefield images have a variable number of images per patient.
- Regular fundus images come from three studies: Niching Diabetes Screening Project (labeled as ``Nicheng''), Shanghai Diabetic Complication Screening Project (labeled as ``Shanghai''), and Nationwide Screening for Complications of Diabetes (labeled as ``Nation'').
## Metadata
The metadata for the DeepDRiD dataset is stored in CSV files. Different CSV files are provided for regular fundus images and ultra-widefield images, with separate files for training, validation, and challenge evaluation sets.

### regular-fundus-training.csv / regular-fundus-validation.csv
The CSV files for regular fundus training and validation contain the following columns:
- **patient_id** - Patient identifier (numeric).
- **image_id** - Image identifier matching the filename without extension (e.g., `1_l1`, `1_r2`).
- **image_path** - Relative path to the image file (e.g., ` regular-fundus-training 1 1_l1.jpg`).
- **Overall quality** - Binary image quality indicator:
- 0 - Quality is not good enough for the diagnosis of retinal diseases
- 1 - Quality is good enough for the diagnosis of retinal diseases
- **left_eye_DR_Level** - Diabetic retinopathy grade for the left eye (0-4, empty if the image is from the right eye):
- 0 - No apparent retinopathy (no abnormalities)
- 1 - Mild NPDR (microaneurysms only)
- 2 - Moderate NPDR (between just microaneurysms and severe NPDR)
- 3 - Severe NPDR (any of: more than 20 intraretinal hemorrhages in each of 4 quadrants; definite venous beading in more than 2 quadrants; prominent intraretinal microvascular abnormalities in more than 1 quadrant; no signs of PDR)
- 4 - PDR (one or more of: neovascularization; vitreous/preretinal hemorrhage)
- **right_eye_DR_Level** - Diabetic retinopathy grade for the right eye (0-4, empty if the image is from the left eye). Uses the same grading scale as `left_eye_DR_Level`.
- **patient_DR_Level** - Patient-level diabetic retinopathy grade (0-4). Uses the same grading scale as eye-level DR levels.
- **Clarity** - Clarity score (1-10):
- 1 - Only level I vascular arch is visible
- 4 - Level II vascular arch and a small number of lesions are visible
- 6 - Level III vascular arch and some lesions are visible
- 8 - Level III vascular arch and most lesions are visible
- 10 - Level III vascular arch and all lesions are visible
- **Field definition** - Field definition score (1-10):
- 1 - Do not include the optic disc and macula
- 4 - Only contain either optic disc or macula
- 6 - Contain optic disc and macula
- 8 - The optic disc or macula is outside the 1 papillary diameter and within the 2 papillary diameter range of the center
- 10 - The optic disc and macula are within 1 papillary diameter of the center
- **Artifact** - Artifact score (0-10):
- 0 - No artifacts
- 1 - Artifacts are outside the aortic arch with scope less than 1/4 of the image
- 4 - Artifacts do not affect the macular area with range less than 1/4
- 6 - Artifacts cover more than 1/4 but less than 1/2 of the image
- 8 - Artifacts cover more than 1/2 without fully covering the posterior pole
- 10 - Cover the entire posterior pole
 ### regular-fundus-source-training.csv / regular-fundus-source-validation.csv
The CSV files containing source information for regular fundus images contain the following columns:
- **patient_id** - Patient identifier (numeric).
- **image_id** - Image identifier matching the filename without extension.
- **image_path** - Relative path to the image file.
- **Source** - Study source of the image:
- `Nicheng` - Niching Diabetes Screening Project
- `Shanghai` - Shanghai Diabetic Complication Screening Project
- `Nation` - Nationwide Screening for Complications of Diabetes
 ### ultra-widefield-training.csv / ultra-widefield-validation.csv
The CSV files for ultra-widefield training and validation contain the following columns:
- **patient_id** - Patient identifier (numeric).
- **image_id** - Image identifier matching the filename without extension (e.g., `1_l2`, `1_r1`).
- **image_path** - Relative path to the image file (e.g., `ultra-widefield-training 1 1_r1.jpg`).
- **DR_level** - Diabetic retinopathy grade (0-5):
- 0 - No apparent retinopathy
- 1 - Mild NPDR
- 2 - Moderate NPDR
- 3 - Severe NPDR
- 4 - PDR
- 5 - Image quality is low and cannot be diagnosed and graded
- **position** - Eye position label:
- `right_eye` - Right eye image
- `left_eye` - Left eye image
 ### Challenge1_upload.csv
The CSV file for Challenge 1 (Dual-View Disease Grading) evaluation contains the following columns:
- **image_id** - Image identifier.
- **DR_Level** - Diabetic retinopathy grade (empty for unlabeled evaluation set, provided in `Challenge1_labels.csv` after competition).
### Challenge2_upload.csv
The CSV file for Challenge 2 (Image Quality Estimation) evaluation contains the following columns:
- **Overall quality** - Binary image quality indicator (empty for unlabeled evaluation set).
- **Artifact** - Artifact score (empty for unlabeled evaluation set).
- **Clarity** - Clarity score (empty for unlabeled evaluation set).
- **Field definition** - Field definition score (empty for unlabeled evaluation set).
- **image_id** - Image identifier.
### Challenge3_upload.csv
The CSV file for Challenge 3 (Transfer Learning) evaluation contains the following columns:
- **image_id** - Image identifier.
- **DR_level** - Diabetic retinopathy grade (empty for unlabeled evaluation set, provided in `Challenge3_labels.xlsx` after competition).
## Splits
The dataset provides explicit splits for both regular fundus images and ultra-widefield images:
- **Training sets**: Labeled data for model development (`regular-fundus-training`, `ultra-widefield-training`).
- **Validation sets**: Labeled data for validation (`regular-fundus-validation`, `ultra-widefield-validation`).
- **Online Challenge Evaluation sets**: Initially unlabeled evaluation data for challenges 1, 2, and 3 (`Online-Challenge1&2-Evaluation`, `Online-Challenge3-Evaluation`). Labels were provided after the competition in separate label files (`Challenge1_labels.csv`, `Challenge2_labels.xlsx`, `Challenge3_labels.xlsx`).
## File Schema
```
37_DeepDRiD/
LICENSE
README.md
regular_fundus_images/
regular-fundus-training/
Images/
1/
1_l1.jpg
1_l2.jpg
1_r1.jpg
1_r2.jpg
...
regular-fundus-training.csv
regular-fundus-source-training.csv
Readme.docx
regular-fundus-validation/
Images/
265/
265_l1.jpg
265_l2.jpg
265_r1.jpg
265_r2.jpg
...
regular-fundus-validation.csv
regular-fundus-source-validation.csv
Readme.docx
Online-Challenge1&2-Evaluation/
Images/
...
Challenge1_upload.csv
Challenge2_upload.csv
Challenge1_labels.csv
Challenge1_labels.xlsx
Challenge2_labels.xlsx
Readme.docx
ultra-widefield_images/
ultra-widefield-training/
Images/
1/
1_l2.jpg
1_r1.jpg
...
ultra-widefield-training.csv
Readme.txt
ultra-widefield-validation/
Images/
34/
34_l2.jpg
34_r1.jpg
...
ultra-widefield-validation.csv
Readme.txt
Online-Challenge3-Evaluation/
Images/
...
Challenge3_upload.csv
Challenge3_labels.xlsx
Readme.docx
```

## Storage in database

### Tables populated

- **`datasets`**: One record for DeepDRiD (name=`"DeepDRiD"`, source_url=`https://www.kaggle.com/datasets/linchundan/fundusimage1000`, license=`CC0: Public Domain`, `modality_types=['fundus', 'uwf']`).
- **`patients`**: One record per unique `patient_id` from the CSV files across all splits and modalities. Only `original_patient_id` is stored; no age, sex, or other clinical metadata is available in the CSVs.
- **`images`**: One row per image found via the CSVs (and, for the evaluation sets, via filesystem scan). Fields:
  - `original_image_id`: the relative path from the dataset root (forward slashes), e.g. `"regular_fundus_images/regular-fundus-training/Images/1/1_l1.jpg"`.
  - `modality`: `"fundus"` for regular fundus images; `"uwf"` for ultra-widefield images.
  - `eye_laterality`: extracted from the `image_id` column using `extract_laterality` (parses `_l` / `_r` suffix patterns). For ultra-widefield images, the `position` column (`"left_eye"` or `"right_eye"`) is used first, with `extract_laterality` as fallback.
  - Image metadata (dimensions, format, etc.) extracted via `get_image_metadata_dict`.
  - Image UUIDs generated from dataset UUID and `original_image_id`.
  - Image file path is resolved by trying three strategies: (1) path from CSV relative to the modality data root, (2) path with an `Images/{patient_id}/` subdirectory injected, (3) direct path using `image_id_str`.
- **`patient_images`**: One record per (patient, image) pair linking each image to its patient. `exam_date=None`.
- **`disease_grading`**: DR grading using the `"ICDR_0_4"` scale. Stored separately by image type and source:
  - **Regular fundus training/validation** (`left_eye_DR_Level` / `right_eye_DR_Level`): per-image grading is only stored when the image's laterality matches the column — left-eye images use `left_eye_DR_Level`, right-eye images use `right_eye_DR_Level`. `patient_DR_Level` is **not** stored as a separate grading.
  - **Ultra-widefield training/validation** (`DR_level`): one grading per image. Ultra-widefield `DR_level` uses a 0–5 scale (5 = ungradable), but the scale name stored is still `"ICDR_0_4"`.
  - **Challenge evaluation sets** (regular: `Challenge1_labels.xlsx`, column `DR_Levels`; ultra-widefield: `Challenge3_labels.xlsx`, column `UWF_DR_Levels`): images with a matching label entry in the Excel file are ingested with their DR grading; images without a label entry in the Excel file are skipped entirely.
  - `annotation_method="manual"`.
- **`quality_annotations`**: Four quality dimensions stored per image, only for regular fundus images (training, validation, and evaluation sets). Quality values are parsed as integers from the CSV / Excel. The scales used:
  - `quality_type="overall"`: 0–2 (0=poor, 1=good, 2=excellent). Note: the doc describes the CSV scale as binary (0/1), but the script code defines the actual stored scale as 0–2.
  - `quality_type="clarity"`: 0–10 (higher = better).
  - `quality_type="field_definition"`: 0–10 (higher = better).
  - `quality_type="artifact"`: 0–10 (lower = better; 0 = no artifact, 10 = severe artifact).
  - Only dimensions with a non-null value are stored for a given image. Ultra-widefield images receive no quality annotations.
- **`raw_annotation_files`**: Registered automatically per `process_csv` call — one raw file record per CSV file (`regular-fundus-training.csv`, `regular-fundus-validation.csv`, `ultra-widefield-training.csv`, `ultra-widefield-validation.csv`). The challenge Excel files (`Challenge1_labels.xlsx`, `Challenge2_labels.xlsx`, `Challenge3_labels.xlsx`) are read via `read_excel_sheet` but not passed through `process_csv`, so they are **not** registered as raw annotation files.
- **`provenance_chains`**: One chain per `process_csv` call (one per CSV file), linking all images and annotations from that CSV to the source file.
- **`dataset_splits`**: Three splits — `"train"`, `"val"`, `"test"` — registered as `split_type="explicit"`.
- **`image_splits`**: Each image is assigned to its corresponding split:
  - `train`: images from training CSVs and (for regular) those with labels from `Challenge1_labels.xlsx` / `Challenge2_labels.xlsx` (actually assigned to `test` in the evaluation processing — see below).
  - `val`: images from validation CSVs.
  - `test`: images from `Online-Challenge1&2-Evaluation/` (regular fundus, if they appear in the challenge label files) and `Online-Challenge3-Evaluation/` (ultra-widefield, if they appear in `Challenge3_labels.xlsx`).

### Annotation types
- **`disease_grading`**: DR grading on the `ICDR_0_4` scale (0–4 for regular; 0–5 for ultra-widefield, but stored under the same scale name). `annotation_method="manual"`. No expert or consensus linkage.
- **`quality_annotations`**: `overall` (0–2), `clarity` (0–10), `field_definition` (0–10), `artifact` (0–10). Regular fundus images only.

### Splits created
- `train` (explicit): regular fundus and ultra-widefield training set images.
- `val` (explicit): regular fundus and ultra-widefield validation set images.
- `test` (explicit): evaluation-set images that appear in the challenge label files.

### Provenance / raw annotation files registered
- `regular-fundus-training.csv`
- `regular-fundus-validation.csv`
- `ultra-widefield-training.csv`
- `ultra-widefield-validation.csv`

The challenge Excel files are **not** registered as raw annotation files.

### Special processing
- **Laterality detection**: For regular fundus images, `extract_laterality` parses `_l1`, `_l2` → `"left"`; `_r1`, `_r2` → `"right"` from the `image_id` column. For ultra-widefield images, the `position` column (`"left_eye"` / `"right_eye"`) is the primary source of laterality, with `extract_laterality` as fallback.
- **Challenge/evaluation set handling**: The `Online-Challenge1&2-Evaluation` directory (regular) and `Online-Challenge3-Evaluation` directory (ultra-widefield) are processed by scanning the filesystem by patient directory, not by CSV. Only images whose `image_id` stem appears in the corresponding label Excel file are ingested; all others are skipped.
- **Patient deduplication**: Patients are tracked in a `patient_lookup` dictionary across all CSVs so that the same `patient_id` appearing in multiple splits results in a single `patients` record.
- Ingestion is idempotent; re-running the script will not create duplicate patients, images, or annotations due to deterministic UUID generation and upsert logic.