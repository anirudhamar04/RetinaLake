# BRSET

## Overview
This dataset consists of 16,266 images from 8,524 Brazilian patients evaluated from 2010 to 2020. The dataset was collected from three Brazilian ophthalmological centers in São Paulo and includes only one macula-centered paired exam from each patient. This dataset enables computer vision models to predict demographic characteristics and multi-label disease classification using retinal fundus photos.

## Images
- All images are in **JPEG** format.
- Images were captured using a Nikon NF505 (Tokyo, Japan) and a Canon CR-2 (Canon Inc, Melville, NY, USA) retinal camera.
- Retinographies were taken by previously trained non-medical professionals in pharmacological mydriasis.
- Images are fovea-centered with both temporal retinal vascular arcades and at least one disc diameter of retina nasally to optic disc visible, with 45 degrees angle and optic disc centered images.
- The viewpoint is macula centered in all images.
- Images were exported directly from the cameras in JPEG format, and no preprocessing techniques were performed.
- Image filenames follow the pattern `img#####`.jpg (e.g., `img00001.jpg`, `img16265.jpg`).
- Excluded from the dataset: fluorescein angiogram photos, non-retinal images, and duplicated images.
## Metadata
The metadata for each image in the dataset is stored in a single CSV file named **labels_brset.csv**, which contains the identifier for each image, demographic information, structural labels, diagnosis, and quality parameters labels.

### labels_brset.csv
The CSV file contains the following columns:
- **image_id** -- Image identifier (e.g., img00001, img16265).
- **patient_id** -- Patient identifier.
- **camera** -- Retinal camera model: ``Canon CR'' or ``NIKON NF5050''.
- **patient_age** -- Age of patient in years.
- **comorbidities** -- Free text of self-referred clinical antecedents (e.g., ``diabetes1'', ``diabetes1, hypertension'').
- **diabetes_time_y** -- Self-referred time of diabetes diagnosis in years (may contain ``NA'' for missing values).
- **insuline** -- Self-referred use of insulin: ``yes'', ``no'', or ``NA''.
- **patient_sex** -- Enumerated values: 1 for male and 2 for female.
- **exam_eye** -- Enumerated values: 1 for the right eye and 2 for the left eye.
- **diabetes** -- Diabetes diagnosis: ``yes''.
- **nationality** -- The patient's nationality (all patients are Brazilian).
- **optic_disc** -- Anatomical parameter: Enumerated values: 1 for normal and 2 for abnormal.
- **vessels** -- Anatomical parameter: Enumerated values: 1 for normal and 2 for abnormal.
- **macula** -- Anatomical parameter: Enumerated values: 1 for normal and 2 for abnormal.
- **DR_ICDR** -- International Clinic Diabetic Retinopathy classification with enumerated values from 0 to 4:
- 0 -- No retinopathy
- 1 -- Mild non-proliferative diabetic retinopathy
- 2 -- Moderate non-proliferative diabetic retinopathy
- 3 -- Severe non-proliferative diabetic retinopathy
- 4 -- Proliferative diabetic retinopathy and post-laser status
- **DR_SDRG** -- Scottish Diabetic Retinopathy Grading Scheme classification with enumerated values from 0 to 4:
- 0 -- No retinopathy
- 1 -- Mild Background
- 2 -- Moderate Background
- 3 -- Severe non-proliferative or pre-proliferative diabetic retinopathy
- 4 -- Proliferative diabetic retinopathy and post-laser status
- **focus** -- Quality parameter: Enumerated values: 1 for normal and 2 for abnormal.
- **Illuminaton** -- Quality parameter: Enumerated values: 1 for normal and 2 for abnormal (note: column name uses ``Illuminaton'' spelling).
- **image_field** -- Quality parameter: Enumerated values: 1 for normal and 2 for abnormal.
- **artifacts** -- Quality parameter: Enumerated values: 1 for normal and 2 for abnormal.
- **diabetic_retinopathy** -- Classification parameter: 1 present and 0 absent.
- **macular_edema** -- Classification parameter: 1 present and 0 absent.
- **scar** -- Classification parameter: 1 present and 0 absent (toxoplasmosis).
- **nevus** -- Classification parameter: 1 present and 0 absent.
- **amd** -- Classification parameter: 1 present and 0 absent (age-related macular degeneration).
- **vascular_occlusion** -- Classification parameter: 1 present and 0 absent.
- **hypertensive_retinopathy** -- Classification parameter: 1 present and 0 absent.
- **drusens** -- Classification parameter: 1 present and 0 absent.
- **hemorrhage** -- Classification parameter: 1 present and 0 absent (nondiabetic retinal hemorrhage).
- **retinal_detachment** -- Classification parameter: 1 present and 0 absent.
- **myopic_fundus** -- Classification parameter: 1 present and 0 absent.
- **increased_cup_disc** -- Classification parameter: 1 present and 0 absent.
- **other** -- Classification parameter: 1 present and 0 absent.
- **quality** -- Overall image quality: ``Adequate'' or ``Inadequate''.
## Splits
The dataset does not provide explicit train--test splits. All images are provided in a single directory, and any data partitioning must be performed externally.

## File Schema
```
27_BRSET/
fundus_photos/
img00001.jpg
img00002.jpg
...
labels_brset.csv
LICENSE.txt
SHA256SUMS.txt
index.html
```

## Storage in database

### Tables populated

**`datasets`**
One record is inserted for BRSET with `modality_types=["fundus"]` and `license="CC-BY-4.0"`.

**`grading_scale_mappings`** (bootstrapped on first run)
Before processing images, the script checks whether Scottish → ICDR_0_4 mappings already exist. If not, it reads the entire `labels_brset.csv`, pairs each row's `DR_SDRG` (Scottish) value with the corresponding `DR_ICDR` value, validates the mapping observations, and upserts the resulting mappings into `grading_scale_mappings`. Requires the Scottish and ICDR_0_4 scales to already exist (registered by `bootstrap_grading_scales.py`).

**`images`**
One row per row in `labels_brset.csv`. Images are looked up in `fundus_photos/` by `image_id` with `.jpg`, `.jpeg`, or `.png` extension tried in order. Fields stored:
- `original_image_id`: the `image_id` string from the CSV (e.g., `img00001`).
- `eye_laterality`: derived from `exam_eye` column (`"1"` → `"right"`, `"2"` → `"left"`).
- `modality="fundus"`.

**`patients`**
One record per unique `patient_id` in the CSV. Fields stored:
- `age`: from `patient_age` (parsed as int).
- `sex`: from `patient_sex` (`"1"` → `"male"`, `"2"` → `"female"`).
- `nationality`: from `nationality` column.
- `comorbidities` (JSONB): populated from `diabetes` (`"yes"` → `True`), `comorbidities` (free text), `diabetes_time_y` (int), and `insuline` (`"yes"` → `True`).

A patient is created once on first encounter; subsequent images for the same `patient_id` reuse the existing patient UUID.

**`patient_images`**
One record per patient-image pair, linking `patient_id` to `image_id` (with `exam_date=None`).

**`classification_annotations`**
One multi-label classification annotation per image, collecting all disease columns into a single JSONB payload:
- `task_type="multi_label"`, `class_name="ocular_diseases"`, `annotation_method="manual"`.
- Columns included (13 total, each 0/1): `diabetic_retinopathy`, `macular_edema`, `scar`, `nevus`, `amd`, `vascular_occlusion`, `hypertensive_retinopathy`, `drusens`, `hemorrhage`, `retinal_detachment`, `myopic_fundus`, `increased_cup_disc`, `other`.
- Only columns with non-empty values that parse as integers are included.
- Provenance linked to the registered `labels_brset.csv` raw file.

**`disease_grading`**
Up to two DR grading records per image:
1. Scottish scale — from `DR_SDRG` column, `scale_name="Scottish"`, `disease_type="DR"`, `annotation_method="manual"`.
2. ICDR_0_4 scale — from `DR_ICDR` column, `scale_name="ICDR_0_4"`, `disease_type="DR"`, `annotation_method="manual"`.

**`quality_annotations`**
One quality annotation per image that has a non-empty `quality` column value (e.g., `"Adequate"`, `"Inadequate"`):
- `quality_type="overall"`, `scale_description="BRSET quality assessment"`.
- Provenance linked to the registered `labels_brset.csv` raw file.

### Annotation types
- `classification` — multi-label binary disease flags (13 conditions) from `labels_brset.csv`.
- `disease_grading` — DR severity on two scales (Scottish and ICDR_0_4).
- `quality` — overall image quality categorical label.

### Splits
A single `"train"` split is registered with `split_type="undefined"` (BRSET has no official train/test split). All images are assigned to this split.

### Provenance / raw annotation files
`labels_brset.csv` is registered in `raw_annotation_files` with `unified_annotation_type="grading"` (the primary task). All classification, grading, and quality annotations are linked to it via a provenance chain.

### Special processing
- The Scottish → ICDR_0_4 scale mapping bootstrap runs before any rows are processed, analyzing the full CSV to learn and store the mappings if absent.
- Duplicate patient detection is done via an in-memory dict (`original_patient_id → patient_uuid`) within the single ingestion run; across runs, deduplication is handled by deterministic UUID generation and upsert logic.
- Images, patients, and patient-image pairs are bulk-upserted first; classifications, gradings, and quality annotations are bulk-upserted concurrently afterwards (via `asyncio.gather`).