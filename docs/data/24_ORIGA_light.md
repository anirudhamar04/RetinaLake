# ORIGA(-light)

## Overview

The dataset contains 650 retinal fundus images collected and annotated by trained professionals from the Singapore Eye Research Institute. The primary focus of the dataset is glaucoma diagnosis, with detailed annotations for optic disc and optic cup segmentation and the calculation of the Cup-to-Disc Ratio (CDR). Binary glaucoma classification labels are provided for each image. The dataset is intended to support research in automated glaucoma screening and morphological analysis.

## Images
- The dataset contains color retinal fundus images.
- A total of 650 images are included.
- Images are provided in three variants:
- Original images.
- Cropped images.
- Square images.
- All image variants use the **.jpg** format.
- Image filenames follow a numeric naming convention in the form `XXX.jpg`, where `XXX` ranges from 001 to 650.
- Eye laterality information is provided in the metadata:
- `OD` -- right eye (oculus dexter).
- `OS` -- left eye (oculus sinister).
- Corresponding segmentation masks are provided for all image variants.
- Segmentation masks are stored in **.png** format.
- Semi-automatic annotation files are provided in MATLAB **.mat** format.
## Metadata
Metadata for the ORIGA(-light) dataset is provided through two CSV files: **OrigaList.csv** and **origa_info.csv**.

#### OrigaList.csv
This file contains 651 rows (including header) and provides clinical and dataset split information.
- **Eye** -- Eye laterality label: `OD` (right eye) or `OS` (left eye).
- **Filename** -- Image filename in the format `XXX.jpg`.
- **ExpCDR** -- Expert Cup-to-Disc Ratio, a decimal value between 0.0 and 1.0.
- **Set** -- Dataset subset indicator: `A` or `B`.
- **Glaucoma** -- Binary glaucoma label: 0 (no glaucoma) or 1 (glaucoma present).
#### origa_info.csv
This file contains 651 rows (including header) and provides derived and structural information.
- **Image** -- Full file path to the corresponding image.
- **Source** -- Dataset source identifier, always `Origa`.
- **Cropped** -- Indicates whether the image is cropped; always `True`.
- **CDR** -- Cup-to-Disc Ratio, decimal value corresponding to `ExpCDR`.
- **Ecc-Cup** -- Eccentricity of the optic cup, decimal value.
- **Ecc-Disc** -- Eccentricity of the optic disc, decimal value.
- **Label** -- Binary glaucoma label: 0 (no glaucoma) or 1 (glaucoma present).
## Splits
The dataset is divided into two subsets, Set A and Set B, as indicated in the `Set` column of **OrigaList.csv**. No explicit training, testing, or validation splits are predefined, and users are required to define their own splits based on these subsets or other criteria.

## File Schema
```
24_ORIGA/
Images/
001.jpg
\dots
650.jpg
Images_Cropped/
001.jpg
\dots
650.jpg
Images_Square/
001.jpg
\dots
650.jpg
Masks/
001.png
\dots
650.png
Masks_Cropped/
001.png
\dots
650.png
Masks_Square/
001.png
\dots
650.png
Semi-automatic-annotations/
001.mat
\dots
650.mat
013.tif
110.tif
OrigaList.csv
origa_info.csv
```

## Storage in database

### Tables populated

**`datasets`**
One record is inserted for ORIGA with `modality_types=["fundus"]` and `task_types=["classification", "segmentation"]`.

**`images`**
One row per entry in `OrigaList.csv`, corresponding to images in `Images/` (original images only; cropped and square variants are not ingested). Image metadata is extracted from the physical file. Fields stored:
- `original_image_id`: numeric stem of the filename (e.g., `"007"`).
- `eye_laterality`: parsed from the `Eye` column (`"OD"` → `"right"`, `"OS"` → `"left"`).
- `modality="fundus"`, `acquisition_date=None`.

**`classification_annotations`**
One binary classification annotation per image, sourced from `OrigaList.csv` (`Glaucoma` column) joined with `origa_info.csv`:
- `task_type="binary"`, `class_name="glaucoma"`, `annotation_method="manual"`.
- The `class_value` JSONB contains `{"glaucoma": bool}` and additionally, if present in `origa_info.csv`: `"cdr"` (float), `"ecc_cup"` (float), `"ecc_disc"` (float).
- Provenance linked to the registered `OrigaList.csv` raw file.

**`segmentation_annotations`**
Up to four segmentation annotations per image:

1. **Manual optic disc** — extracted from the multi-class PNG in `Masks/` (class value `1 = optic_disc`), `annotation_method="manual"`.
2. **Manual optic cup** — extracted from the same PNG (class value `2 = optic_cup`), `annotation_method="manual"`.
3. **Pseudo optic disc** — extracted from the `.mat` file in `Semi-automatic-annotations/` (key `mask`, same class encoding), `annotation_method="pseudo"`.
4. **Pseudo optic cup** — extracted from the same `.mat` file (class value `2`), `annotation_method="pseudo"`.

The `.mat` mask array (shape 2048×3072, dtype uint8, values 0/1/2) is written to a temporary PNG and processed via the multiclass mask processor; the temp file is deleted afterwards. Missing `.mat` files are silently skipped.

### Annotation types
- `classification` — binary glaucoma label with optional CDR/eccentricity metadata, from `OrigaList.csv` + `origa_info.csv`.
- `segmentation` — optic disc and optic cup; two sets: manual (PNG masks) and pseudo (MAT files).

### Splits
Two splits are registered with `split_type="explicit"`:
- `"train"` ← images where `Set = "A"` in `OrigaList.csv`.
- `"test"` ← images where `Set = "B"` in `OrigaList.csv`.

### Provenance / raw annotation files
- `OrigaList.csv` is registered in `raw_annotation_files` with `unified_annotation_type="classification"`. All classification annotations are linked to it.
- Each manual mask PNG (`Masks/NNN.png`) is registered individually with `unified_annotation_type="segmentation"`, `file_type=None`, `auto_detect_type=False`. Manual segmentation annotations are linked to their respective mask file.
- Each `.mat` file (`Semi-automatic-annotations/NNN.mat`) is registered individually with `unified_annotation_type="segmentation"`, `file_type="mat"`. Pseudo segmentation annotations are linked to the `.mat` file (not the temporary PNG).
- `origa_info.csv` is loaded as a lookup dictionary but is **not** separately registered in `raw_annotation_files`.

### Special processing
- `origa_info.csv` is pre-loaded into memory as a dict keyed by filename and joined to each `OrigaList.csv` row to enrich the classification `class_value`.
- Semi-automatic `.mat` annotations use a temporary file pattern: the `mask` numpy array is converted to uint8 and saved as a temporary PNG for processing; the provenance points to the original `.mat` file, not the temp file.
- Only original images from `Images/` are ingested; `Images_Cropped/` and `Images_Square/` are ignored, as are `Masks_Cropped/` and `Masks_Square/`.