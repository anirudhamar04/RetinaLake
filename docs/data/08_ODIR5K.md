# ODIR5K

## Overview
The Ocular Disease Intelligent Recognition (ODIR5K) dataset is a structured ophthalmic database consisting of 5,000 patients. It includes patient age, color fundus photographs of both left and right eyes, and diagnostic keywords provided by doctors. 
This dataset is a modified version of the original ODIR dataset, where features have been extracted and associated with their corresponding individual images. The dataset supports multiple disease categories, including Normal, Diabetes, Glaucoma, Cataract, Age related Macular Degeneration, Hypertension, Pathological Myopia, and Other diseases or abnormalities.

## Images
- Images are color fundus photographs.
- All images are stored in **jpg** format.
- Each patient has a pair of images corresponding to the left and right eye.
- Image filenames follow the pattern `<ID>_<left/right>.jpg`.
## Metadata
Metadata for the dataset is provided in a single CSV file named **full_df.csv**.

### Metadata Files
**full_df.csv** contains patient information, image references, diagnostic keywords, and per-image labels.
- **ID** -- Unique identifier for the patient.
- **Patient Age** -- Age of the patient.
- **Patient Sex** -- Sex of the patient.
- **Left-Fundus** -- Filename or reference to the left eye fundus image.
- **Right-Fundus** -- Filename or reference to the right eye fundus image.
- **Left-Diagnostic Keywords** -- Diagnostic keywords assigned by doctors for the left eye.
- **Right-Diagnostic Keywords** -- Diagnostic keywords assigned by doctors for the right eye.
- **N** -- Indicator for Normal.
- **D** -- Indicator for Diabetes.
- **G** -- Indicator for Glaucoma.
- **C** -- Indicator for Cataract.
- **A** -- Indicator for Age related Macular Degeneration.
- **H** -- Indicator for Hypertension.
- **M** -- Indicator for Pathological Myopia.
- **O** -- Indicator for Other diseases or abnormalities.
- **filepath** -- File path pointing to a single fundus image for the patient.
- **labels** -- Holds the label information for the single fundus image. (ignore this)
- **target** -- Holds the target label for the single fundus image. (ignore this)
- **filename** -- Filename corresponding to the single fundus image.
## Splits
A train--test split exists in the non-preprocessed data. Additionally, there is one preprocessed dataset that contains the full data without explicit splits.

## File Schema
```
08_ODIR-5K/
ODIR-5K/
ODIR-5K/
Testing Images/
Training Images/
preprocessed_images/
full_df.csv
```

## Storage in database

### Tables populated

- **`datasets`** — One row registered with `dataset_name="ODIR-5K"`, `source_url`, `license="CC-BY-4.0"`, and `modality_types=["fundus"]`.
- **`patients`** — One row per Excel row (one per patient). Fields stored:
  - `age`: integer from `Patient Age` column (or None if missing/non-numeric).
  - `sex`: mapped from `Patient Sex` (`"male"` / `"female"` / `"unknown"` / None).
  - `comorbidities` (JSONB): three boolean fields from columns `C` (cataract), `H` (hypertension), `M` (myopia).
- **`images`** — Two rows per patient (one per eye). Fields populated via `get_image_metadata_dict` plus `modality="fundus"`, `eye_laterality` set to `"left"` or `"right"`, and `original_image_id` set to the image filename. The image file is searched first in `Training Images/`, then in `Testing Images/`.
- **`patient_images`** — Two rows per patient linking the patient to the left and right eye images.
- **`classification_annotations`** — One multi-label classification row per image, if at least one of the five disease indicators is True. `class_name="disease_indicators"`, `task_type="multi_label"`, `class_value` is a dict of boolean flags for: `normal` (N), `diabetes` (D), `glaucoma` (G), `amd` (A), `other` (O). `annotation_method="manual"`. Note: the three comorbidity indicators C (cataract), H (hypertension), M (myopia) are stored in patient comorbidities only and are NOT stored in classification_annotations.
- **`keyword_annotations`** — One row per keyword per eye-image. Keywords come from `Left-Diagnostic Keywords` and `Right-Diagnostic Keywords` columns. The script normalises Chinese commas (`，`) to ASCII commas before splitting on `,`. Stored with `keyword_source="diagnostic_keywords"` and `annotation_method="manual"`. Keywords are individually upserted.
- **`dataset_splits`** / **`image_split`** — Two explicit splits: `train` (images found in `Training Images/`) and `test` (images found in `Testing Images/`). Split membership is determined by which directory the image file was located in.

### Provenance / raw annotation files

`data.xlsx` is processed via `process_excel` (first sheet), which registers the file in `raw_annotation_files` with `annotation_type="classification"` and creates a `provenance_chain` entry. All classification and keyword annotations reference this file's `raw_file_id` and `provenance_chain_id`.

### Image metadata extraction

Each image file is located by searching `Training Images/` then `Testing Images/` for the exact filename from the `Left-Fundus` or `Right-Fundus` column. If not found in either directory, the image is skipped with a `file_not_found` error. The split is determined by the directory where the file is found.

### Special processing

Upsert order respects foreign key constraints: patients and images are upserted in parallel first, then patient-image links, then classifications, then keywords individually. The script is idempotent: re-running does not create duplicates due to deterministic UUID generation and upsert logic.