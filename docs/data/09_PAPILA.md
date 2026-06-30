# PAPILA

## Overview

The PAPILA dataset contains records of 244 patients. Each record provides structured clinical data, retinal fundus images of both eyes, optic disc and optic cup segmentations for both eyes, and diagnostic labels derived from clinical data.
All records are anonymized and assigned a unique patient identifier. The dataset is designed to support research on glaucoma, with joint information from both eyes of each patient.

## Images

- Images are retinal fundus images (RFI) centered at the papilla.
- Images are provided for both the left eye (OS) and right eye (OD) of each patient.
- All images are stored in **JPEG** format.
- Images have a resolution of **2576 × 1934** pixels.
- Images are acquired with an aperture of **30$^\circ$**.
- Images are captured using a **Topcon TRC-NW400** non-mydriatic retinal camera.
- Images are acquired by ophthalmologists or technicians at HGURS (Murcia, Spain).
- Images with superimposed segmentation contours are also provided in JPEG format.

## Metadata

Metadata is provided in spreadsheet files contained in the **ClinicalData** folder. Clinical data and diagnoses are organized separately for the right eye (OD) and left eye (OS).

### Metadata Files

Two spreadsheet files are provided, one corresponding to the right eye (OD) and one to the left eye (OS). Filenames are not provided.

- **Patient ID** -- Unique identifier assigned to each patient.
- **Age** -- Age of the patient.
- **Gender** -- Gender of the patient, encoded as 0 for male and 1 for female.
- **Diagnosis** -- Clinical diagnosis with values:
  - 0 -- Healthy
  - 1 -- Glaucoma
  - 2 -- Suspicious
- **Refractive Error** -- Refractive error measurement.
- **Phakic/Pseudophakic** -- Lens status encoded as 1 if the crystalline lens has been removed and 0 if the eye keeps the lens.
- **Intraocular Pressure** -- Measured intraocular pressure.
- **Pachymetry** -- Corneal thickness measurement.
- **Axial Length** -- Axial length of the eye.
- **Mean Defect** -- Mean defect value from visual field testing.

## Splits

Explicit train, test, or validation splits are not provided at the dataset root level.
Cross-validation dataset splits are indicated within the **HelpCode** folder.

## File Schema

```
09_PAPILA/
ClinicalData/
ExpertsSegmentations/
Contours/
ImagesWithContours/
FundusImages/
HelpCode/
kfold/
Test 1/
Test/
Train/
Test 2/
Test/
Train/
README.txt
desktop.ini
```

## Storage in database

### Tables populated

- **`datasets`** — One row registered with `dataset_name="PAPILA"`, `source_url`, `license="CC-BY-4.0"`, and `modality_types=["fundus"]`.
- **`patients`** — One row per unique patient ID (deduplicated across the two Excel files). Fields stored:
  - `age` from the `Age` column.
  - `sex`: mapped from `Gender` (0→`"male"`, 1→`"female"`).
  - `comorbidities` (JSONB): nested clinical measurements including:
    - `refractive_defect`: `{dioptre_1, dioptre_2, astigmatism}` from `Refractive_Defect` and unnamed columns 5–6.
    - `phakic_pseudophakic`: from `Phakic/Pseudophakic`.
    - `iop`: `{pneumatic, perkins}` from `IOP` and unnamed column 9.
    - `pachymetry`: from `Pachymetry`.
    - `axial_length`: from `Axial_Length`.
    - `vf_md`: from `VF_MD`.
- **`images`** — One row per fundus image (up to 488 total; 2 per patient). Fields populated via `get_image_metadata_dict` plus `modality="fundus"`, `eye_laterality` set to `"right"` (OD) or `"left"` (OS), and `original_image_id` set to the image stem (e.g., `RET002OD`). The image filename is constructed as `RET{patient_number}{OD|OS}` where `patient_number` is the numeric part of the patient ID (e.g., `"#002"` → `"002"`).
- **`patient_images`** — One row per image linking the patient to their eye image.
- **`classification_annotations`** — One multi-label classification row per image. `class_name="glaucoma"`, `task_type="multi_label"`. The `class_value` dict has three boolean keys derived from the `Diagnosis` code: `normal` (code 0), `glaucoma` (code 1), `glaucoma_suspicious` (code 2). Only one flag is True per image. `annotation_method="manual"`.
- **`experts`** — Two records created: `Expert_1` and `Expert_2`, both with `expertise_area="ophthalmology"` and linked to this dataset.
- **`expert_annotations`** — One row per contour file. Each expert annotation records: `expert_id`, `annotation_task="segmentation"`, `raw_data_id` (from the contour file), and an `annotation_value` JSON containing `image_id`, `segmentation_id`, `annotation_type`, `structure` (`"disc"` or `"cup"`), `image_stem`, and `contour_file`.
- **`segmentation_annotations`** — One row per contour file. Contour files are named `{image_stem}_{structure}_{expert}.txt` (e.g., `RET004OD_disc_exp1.txt`). Each is processed via `process_segmentation_from_contour` into a pixel mask using actual image dimensions. `annotation_type` is set to `"optic_disc"` or `"optic_cup"` from the structure part. The `expert_annotation_id` links the segmentation to its expert annotation record. `annotation_method="manual"`.
- **`dataset_splits`** / **`image_split`** — No splits are created. The dataset provides no train/test partition and the script does not call `register_standard_splits`.

### Provenance / raw annotation files

- The two Excel files (`patient_data_od.xlsx`, `patient_data_os.xlsx`) are processed via `process_excel` with `annotation_type="classification"`, registering each in `raw_annotation_files` with a `provenance_chain` entry.
- Each `.txt` contour file in `ExpertsSegmentations/Contours/` is individually registered via `register_individual_file` with `file_type="txt"` and `annotation_type="segmentation"`. The resulting `raw_file_id` and `provenance_chain_id` are stored on both the expert annotation and the segmentation annotation.

### Contour filename convention

Files follow the pattern `{image_stem}_{structure}_{expert}.txt`:
- `image_stem`: `RET{patient_number}{OD|OS}` — identifies the image.
- `structure`: `disc` → `annotation_type="optic_disc"`, `cup` → `annotation_type="optic_cup"`.
- `expert`: `exp1` → Expert_1, `exp2` → Expert_2.
- Files not matching exactly 3 underscore-separated parts are skipped with a warning.

### Special processing

- Both Excel files are processed concurrently via `asyncio.gather`. Patients are deduplicated across OD and OS Excel files using an in-memory dict keyed on `patient_id_str`.
- Contours are processed concurrently (up to 10 simultaneous) using a semaphore.
- Image dimensions are extracted during the Excel phase and cached for use during segmentation processing.
- Upsert order: patients → images → (patient links + classifications in parallel) → experts → contour segmentations.
- The script is idempotent: re-running does not create duplicates due to deterministic UUID generation and upsert logic.
