# MESSIDOR
## Overview
The Messidor-2 dataset is a collection of Diabetic Retinopathy (DR) examinations, where each examination consists of two macula-centered retinal fundus images (one per eye). Only macula-centered images are included in the dataset. 
The dataset contains a total of 874 examinations corresponding to 1748 images and is provided along with metadata describing image pairing and clinical annotations.

## Images
The Messidor-2 dataset is composed of images originating from two subsets: Messidor-Original and Messidor-Extension.
- Messidor-Original consists of 529 examinations (1058 images) collected from the original Messidor dataset.
- Messidor-Extension consists of 345 examinations (690 images) collected at the Ophthalmology Department of Brest University Hospital (France) between October 2009 and September 2010.
- Images were acquired without pharmacological dilation using a Topcon TRC NW6 non-mydriatic fundus camera with a 45-degree field of view.
- Messidor-Original images are stored in **PNG** format (1058 imgaes), while Messidor-Extension images are stored in **JPG** format (690 images).
- Image filenames have an unknown naming system with a different one for each of Messidor-Original and Messidor-Extension. No explicit left/right eye information in either ID or in the metadata.
Further information can be found at: [Messidor-2 Dataset Page](https://www.adcis.net/en/third-party/messidor2/)

## Metadata
The metadata for the Messidor-2 dataset is stored in a single CSV file named **messidor_data.csv**, which contains adjudicated clinical labels for each image.

### messidor_data.csv
The CSV file contains the following columns:
- **image_id** - Filename as provided in the Messidor-2 dataset. (eg 20051020_43808_0100_PP.png, IM000595.jpg)
- **adjudicated_dr_grade** - Clinically adjudicated 5-point ICDR grade:
- 0 - No DR
- 1 - Mild DR
- 2 - Moderate DR
- 3 - Severe DR
- 4 - Proliferative DR (PDR)
- **adjudicated_dme** - Referable Diabetic Macular Edema (DME), defined by hard exudates within 1 disc diameter:
- 0 - No Referable DME
- 1 - Referable DME
- **adjudicated_gradable** - Image quality indicator:
- 0 - Ungradable (no DR or DME grade provided)
- 1 - Gradable (both DR and DME graded)
 For images where `adjudicated_gradable = 0`, the fields `adjudicated_dr_grade` and `adjudicated_dme` are empty.

## Splits
The dataset does not provide explicit train–test splits. All images are provided in a single directory, and any data partitioning must be performed externally.

## File Schema
```
02_MESSIDOR/
train_org/
messidor_data.csv
```

Filtering CSV file to match sampled images...
 Sampled 87 images
 Matched 44 CSV entries
 WARNING: Mismatch between sampled images (87) and CSV entries (44)
 This may indicate some images don't have CSV entries
Happened as in csv they are .jpg and in the folder they are .JPG.

## Storage in database

### Tables populated

- **`datasets`** — One row registered with `dataset_name="MESSIDOR"`, `source_url`, `license="Custom - Educational and research use"`, and `modality_types=["fundus"]`.
- **`images`** — One row per image. Fields populated via `get_image_metadata_dict` plus `modality="fundus"` and `original_image_id` set to the `image_id` value from `messidor_data.csv`. No laterality is stored (the dataset provides no left/right information). Images are located under `train_org/`; the script handles case-insensitive extension matching (`.jpg` vs `.JPG`, `.png` vs `.PNG`).
- **`disease_grading`** — One row per image. The `adjudicated_dr_grade` column is stored under scale `ICDR_0_4` (0=No DR, 1=Mild, 2=Moderate, 3=Severe, 4=Proliferative DR). `annotation_method` is `"manual"`.
- **`classification_annotations`** — One row per image. The `adjudicated_dme` column is stored as a binary classification with `task_type="binary"` and `class_name="DME"`. Value is cast to bool (0→False, 1→True). `annotation_method` is `"manual"`.
- **`quality_annotations`** — One row per image. The `adjudicated_gradable` column is stored as a `gradability` quality annotation with `quality_score` equal to the raw integer value (0 or 1) and `quality_label` set to `"gradable"` or `"not_gradable"`.
- **`dataset_splits`** / **`image_split`** — A single explicit `train` split is created and all images are assigned to it (MESSIDOR provides no explicit train/test partition).

### Provenance / raw annotation files

`messidor_data.csv` is processed via `process_csv`, which registers it in `raw_annotation_files` (with file path, hash, and `annotation_type="grading"`) and creates a `provenance_chain` entry. Every annotation row (grading, classification, quality) references the same `raw_file_id` and `provenance_chain_id`.

### Image metadata extraction

Image metadata is extracted automatically from each file found in `train_org/` using `get_image_metadata_dict`. The script first tries the exact filename from the CSV, then falls back to an alternative capitalisation (e.g., `.JPG` if `.jpg` is not found). If neither path exists, the image is skipped and an error is recorded.

### Special processing

The three annotation types (DR grading, DME classification, gradability) are bulk-upserted in parallel via `asyncio.gather` after images are inserted. All three annotations share the same provenance entry derived from the single `messidor_data.csv` file. The script is idempotent: re-running does not create duplicates due to deterministic UUID generation and upsert logic.