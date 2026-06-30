# FUND-OCT

## Overview
The FUND-OCT dataset is a collection of ophthalmic imaging data containing both OCT (Optical Coherence Tomography) and fundus images. The dataset is organized by anatomical region into **Macula** and **OD (Optic Disc)** folders. 
The Macula subset contains images from patients with multiple retinal conditions, while the OD subset contains optic disc images from glaucoma and healthy cases. Images are organized at the patient level, with further separation by eye (Left Eye and/or Right Eye) and imaging modality.

## Images
- The dataset contains **OCT images** and **fundus images**.
- Images are organized by anatomical region: **Macula** and **OD (Optic Disc)**.
- Macula images include cases of Acute CSR, Chronic CSR, ci-DME, Geographic AMD, Neovascular AMD, and Healthy controls.
- OD images include cases of Glaucoma and Healthy controls.
- Images are organized by **patient ID** (e.g., P_1, P_2) and further divided into **Left Eye** and/or **Right Eye**.
- File naming conventions vary by condition:
- CSR cases use filenames containing patient ID, date, time, and modality (e.g., `*_B-scan_L_001.jpg`).
- AMD, Healthy, and ci-DME cases use condition-specific prefixes (e.g., `OCT_Left_AMD_ILM.jpg`, `OCT_Left_Normal_ILM.jpg`).
- OD cases include disc and cup annotations (e.g., `Fundus_Left_Glaucoma_Cup.jpg`, `OCT_Left_Normal_Disc.jpg`).
- Volume frames for CSR cases are stored as sequentially numbered JPG files.
- Fluid masks are present only in acute CSR cases.
- Image resolution, field of view, camera model, and acquisition parameters are not provided.
## Metadata
No centralized CSV metadata file is provided. Metadata and disease labels are encoded in the folder structure. Retinal layer annotations are provided for specific Macula conditions through auxiliary files.

### Metadata Files
The following metadata files are present only for the **Macula/ci-DME**, **Macula/geographic_AMD**, **Macula/Healthy**, and **Macula/neovascular_AMD** subsets.

#### layerAnnotations.csv
This CSV file contains pixel-wise retinal layer boundary annotations for OCT B-scans.
- **Row index** -- Each row corresponds to a retinal layer boundary (ILM, RNFL-GCL, IPL-INL, INL-OPL, OPL-ONL, ONL-IS, IS-OS, OS-RPE, BrM).
- **Column index** -- Each column corresponds to a horizontal A-scan position across the OCT B-scan.
- **Cell values** -- Pixel row (y-coordinate) of the corresponding layer boundary.
- **NaN values** -- Indicate missing or undefined layer boundaries.
#### layerAnnotations.mat
A MATLAB format file containing the same retinal layer boundary data as `layerAnnotations.csv`. Pixel coordinates for retinal layer boundaries are stored in MATLAB data structures.

#### layerNames.txt
A text file listing the names of retinal layers in order.
- **Layer N** -- Layer index number.
- **LayerName** -- Name of the retinal layer corresponding to the row order in `layerAnnotations.csv`.
## Splits
No explicit train, test, or validation splits are provided. Dataset partitioning must be performed externally by the user.
## File Schema
```
16_FUND-OCT/
Macula/
acute CSR/
P_1/
Left Eye/
VolumeFrames/
*_B-scan_L_001.jpg
*_Color_L_001.jpg
*_Red-free_L_001.jpg
2d lt.wmv
fluid.jpg
Right Eye/
P_2/
P_3/
chronic CSR/
P_1/
Left Eye/
VolumeFrames/
*_B-scan_L_001.jpg
*_Color_L_001.jpg
*_Red-free_L_001.jpg
2D LT.wmv
Right Eye/
P_2/
ci-DME/
P_1/
Left Eye/
layerAnnotations.csv
layerAnnotations.mat
layerNames.txt
OCT_Left_Normal_*.jpg
original.jpg
2D LT.wmv
Right Eye/
P_2/
P_12/
geographic_AMD/
P_1/
Left Eye/
layerAnnotations.csv
layerAnnotations.mat
layerNames.txt
OCT_Left_AMD_*.jpg
*_Color_L_001.jpg
*_Red-free_L_001.jpg
original.jpg
Right Eye/
P_6/
Healthy/
P_1/
Left Eye/
layerAnnotations.csv
layerAnnotations.mat
layerNames.txt
OCT_Left_Normal_*.jpg
*_Color_L_001.jpg
*_Red-free_L_001.jpg
original.jpg
Right Eye/
P_32/
neovascular_AMD/
P_1/
Left Eye/
layerAnnotations.csv
layerAnnotations.mat
layerNames.txt
OCT_Left_AMD_*.jpg
*_Color_L_001.jpg
*_Red-free_L_001.jpg
original.jpg
Right Eye/
P_6/
OD/
Glaucoma/
P_1/
Left Eye/
*_B-scan_L_001.jpg
*_Color_L_001.jpg
*_Red-free_L_001.jpg
Fundus_Left_Glaucoma_Cup.jpg
Fundus_Left_Glaucoma_Disc.jpg
OCT_Left_Glaucoma_Cup.jpg
OCT_Left_Glaucoma_Disc.jpg
P_26/
Healthy/
P_1/
Left Eye/
*_B-scan_L_001.jpg
*_Color_L_001.jpg
*_Red-free_L_001.jpg
Fundus_Left_Normal_Cup.jpg
Fundus_Left_Normal_Disc.jpg
OCT_Left_Normal_Cup.jpg
OCT_Left_Normal_Disc.jpg
Right Eye/
P_18/
```

## Storage in database

### Tables populated
- **`datasets`**: One record for FUND-OCT with `modality_types=["oct", "fundus"]`.
- **`patients`**: One record per unique patient folder name (e.g., `P_1`, `P_2`). Only `original_patient_id` is populated; `age`, `sex`, `ethnicity`, `nationality`, and `comorbidities` are all `None`.
- **`image_groups`**: One record per OCT volume, `group_type="oct_volume"`. A volume is identified by a key of `{patient_id}:{laterality}:{eye_folder_path}`. Groups are shared by the B-scan and all VolumeFrames for the same eye.
- **`images`**: One row per image file. `modality` is determined by filename:
  - Files with "color", "red-free", or "redfree" in the name → `modality="fundus"`, `group_id=None`, `frame_index=None`.
  - Files with "B-scan" or "b-scan" in the name → `modality="oct"`, `frame_index=0` (key frame), linked to `image_group`.
  - Files in a `VolumeFrames/` subdirectory → `modality="oct"`, `frame_index` parsed from the numeric filename stem (e.g., `1.jpg` → `frame_index=1`), linked to `image_group`.
  - All other image files → `modality="oct"`, `group_id=None`, `frame_index=None`.
  - `eye_laterality` is set to `"left"` or `"right"` based on the "Left Eye" / "Right Eye" folder name. `original_image_id` is the relative path from the `Macula/` or `OD/` root.
- **`patient_images`**: One record linking each patient to each of their images.
- **`classification_annotations`**: One binary classification per OCT image (not for fundus images). `task_type="binary"`, `class_value=True`, `class_name` is the normalized disease category derived from the folder name. Macula categories: `"acute_csr"`, `"chronic_csr"`, `"ci_dme"`, `"geographic_amd"`, `"healthy"`, `"neovascular_amd"`. OD categories: `"glaucoma"`, `"healthy"`. `annotation_method="manual"`.

### Image metadata extraction
All image files are processed via `get_image_metadata_dict`. The patient ID is extracted from any path component matching `P_\d+`. Laterality is extracted from the "Left Eye" / "Right Eye" folder name. The disease category is extracted from the first folder level under `Macula/` or `OD/`.

### Annotation types
- **Classification** (binary): One annotation per OCT image with `class_name` set to the disease category string and `class_value=True`. Fundus images do not receive classification annotations.

### Splits
No train/test splits are created. The dataset has no split definitions.

### Provenance
No annotation source files are registered in `raw_annotation_files`. All disease categories are derived implicitly from the folder hierarchy. Provenance for classifications is set via `process_folder_tree` with `unified_annotation_type="classification"`.

### Special processing
- The Macula and OD folder trees are traversed separately using `process_folder_tree` with handlers that capture the root type (`"macula"` or `"od"`).
- Upsert order: image groups → patients → images → patient links and classifications (in parallel).
- Non-image files (`.wmv`, `.csv`, `.mat`, `.txt`) are ignored; only files with extensions `.jpg`, `.JPG`, `.jpeg`, `.JPEG`, `.png`, `.PNG` are processed.
- Ingestion is idempotent due to deterministic UUID generation and upsert logic.