# Paraguay

## Overview
The Paraguay dataset is a collection of color fundus images acquired at the Department of Ophthalmology of the Hospital de Clínicas, Facultad de Ciencias Médicas, Universidad Nacional de Asunción, Paraguay. 

The dataset contains a total of 757 retinal images collected following a clinical acquisition procedure and classified by expert ophthalmologists. 
The images are annotated for Diabetic Retinopathy (DR), covering both Non-Proliferative Diabetic Retinopathy (NPDR) and Proliferative Diabetic Retinopathy (PDR) across multiple disease stages.

## Images
- The dataset consists of **color fundus (retinal)** images.
- All images are stored in **JPG** format.
- Images were acquired using a **Visucam 500** fundus camera manufactured by **Zeiss**.
- A total of **757** images are included in the dataset.
- Images are anonymized and renamed using numeric identifiers, as referenced in the metadata.
- The dataset includes images corresponding to seven DR categories: No DR signs, Mild (or early) NPDR, Moderate NPDR, Severe NPDR, Very Severe NPDR, PDR, and Advanced PDR.
## Metadata
Metadata for the dataset is provided in a single spreadsheet file containing image identifiers, image format information, and clinical DR classification labels. 
In addition to the metadata file, the images are also organized into folders corresponding to their diagnostic labels.

### Metadata Files
The dataset provides one metadata file:
- **Annotations of the classifications.xlsx**
This XLSX file contains the labels and associated information for each fundus image.
- **Image** -- Numeric identifier of the anonymized and renamed patient image.
- **Format** -- File format of the image; all entries are `.jpg`.
- **Status** -- Clinical classification of Diabetic Retinopathy corresponding to the image. The possible values correspond to the seven DR categories defined in the dataset.
The class that the fundus images are classsifed into are
- No DR signs (187 images)
- Mild (or early) NPDR (4 images)
- Moderate NPDR (80 images)
- Severe NPDR (176 images)
- Very Severe NPDR (108 images)
- PDR (88 images)
- Advanced PDR (114 images)
## Splits
Explicit train, test, or validation splits are **not provided**. 
All images are distributed together, and any data partitioning must be performed externally by the user.

## File Schema
```
10_PARAGUAY/
1. No DR signs/
2. Mild (or early) NPDR/
3. Moderate NPDR/
4. Severe NPDR/
5. Very Severe NPDR/
6. PDR/
7. Advanced PDR/
Annotations of the classifications.xlsx
```

## Storage in database

### Tables populated

- **`datasets`** — One row registered with `dataset_name="PARAGUAY"`, `source_url="https://zenodo.org/record/4647952"`, `license="CC-BY-4.0"`, and `modality_types=["fundus"]`.
- **`images`** — One row per `.jpg` image found in the seven grade folders. Fields populated via `get_image_metadata_dict` plus `modality="fundus"` and `original_image_id` set to the image filename stem. No laterality is stored.
- **`disease_grading`** — One row per image. Grade value and label are determined by the parent folder name using the following mapping:

  | Folder | Numeric grade | Label |
  |---|---|---|
  | `1. No DR signs` | 0 | No DR signs |
  | `2. Mild (or early) NPDR` | 1 | Mild NPDR |
  | `3. Moderate NPDR` | 2 | Moderate NPDR |
  | `4. Severe NPDR` | 3 | Severe NPDR |
  | `5. Very Severe NPDR` | 4 | Very Severe NPDR |
  | `6. PDR` | 4 | PDR |
  | `7. Advanced PDR` | 4 | Advanced PDR |

  Note: folders 5, 6, and 7 all map to numeric grade 4 — the original folder label is preserved in the `grade_label` field to distinguish them. Scale name: `PARAGUAY_DR_7_level` (min=0, max=4, with value labels 0–4). `annotation_method="manual"`.

- **`dataset_splits`** / **`image_split`** — A single explicit `train` split is created and all images are assigned to it (no train/test partition is provided by the dataset).

### Provenance / raw annotation files

The data root directory is registered as a raw annotation source via `register_mask_directory` with `annotation_type="grading"`, producing a single `raw_file_id` and `provenance_chain_id`. A provenance context is set for the entire folder traversal so that all `disease_grading` rows share the same provenance entry referencing the folder structure. The provenance context is reset (via a `finally` block) after all images are processed.

### Image metadata extraction

Images are discovered by globbing `*.jpg` in each of the seven named grade folders. There is no separate metadata CSV used during ingestion; the `Annotations of the classifications.xlsx` file present in the dataset root is not read by the script.

### Special processing

Folders are traversed sequentially in the order defined by the `GRADE_MAP`. Images and gradings are collected and then bulk-upserted (images first, then gradings, in batches of 1000). The script is idempotent: re-running does not create duplicates due to deterministic UUID generation and upsert logic.