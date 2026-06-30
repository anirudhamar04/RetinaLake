# RFMid
## Overview
Retinal Fundus Multi-disease Image Dataset (RFMiD) consists of 3200 fundus images captured using three different fundus cameras with 46 conditions annotated through adjudicated consensus of two senior retinal experts.

## Images
The RFMID dataset contains Standard color fundus images captured with three different cameras.
- All images are in **png** format
- Images are named 1.png to x.png 
## Metadata
The metadata for each image in the dataset is concisely stored for train, test, and validation data in `RFMiD_Training_Labels.csv`, `RFMiD_Testing_Labels.csv`, and `RFMiD_Validation_Labels.csv` respectively. They each contain the following metadata. With all fields having 0 to indicate absence and 1 to indicate presence.
- **ID** -- Unique identifier for the fundus image. Corresponds to the image filename (without extension) in the RFMID dataset.
- **Disease_Risk** -- Indicator of whether the image contains any retinal abnormality
- **DR** -- Diabetic Retinopathy
- **ARMD** -- Age-Related Macular Degeneration
- **MH** -- Macular Hole
- **DN** -- Drusen
- **MYA** -- Myopia-related retinal changes
- **BRVO** -- Branch Retinal Vein Occlusion
- **TSLN** -- Tessellation
- **ERM** -- Epiretinal Membrane
- **LS** -- Laser Scars
- **MS** -- Macular Scar
- **CSR** -- Central Serous Retinopathy
- **ODC** -- Optic Disc Cupping
- **CRVO** -- Central Retinal Vein Occlusion
- **TV** -- Tortuous Vessels
- **AH** -- Asteroid Hyalosis
- **ODP** -- Optic Disc Pallor
- **ODE** -- Optic Disc Edema
- **ST** -- Subretinal Tissue or Scar
- **AION** -- Anterior Ischemic Optic Neuropathy
- **PT** -- Papillitis
- **RT** -- Retinal Tear
- **RS** -- Retinal Scar
- **CRS** -- Chorioretinal Scar
- **EDN** -- Exudation
- **RPEC** -- Retinal Pigment Epithelial Changes
- **MHL** -- Lamellar Macular Hole
- **RP** -- Retinitis Pigmentosa
- **CWS** -- Cotton Wool Spots
- **CB** -- Coloboma
- **ODPM** -- Optic Disc Pit Maculopathy
- **PRH** -- Preretinal Hemorrhage
- **MNF** -- Myelinated Nerve Fibers
- **HR** -- Hemorrhage
- **CRAO** -- Central Retinal Artery Occlusion
- **TD** -- Tractional Detachment
- **CME** -- Cystoid Macular Edema
- **PTCR** -- Post-Traumatic Chorioretinopathy
- **CF** -- Chorioretinal Fibrosis
- **VH** -- Vitreous Hemorrhage
- **MCA** -- Microaneurysm
- **VS** -- Vessel Sheathing
- **BRAO** -- Branch Retinal Artery Occlusion
- **PLQ** -- Plaque
- **HPED** -- Hemorrhagic Pigment Epithelial Detachment
- **CL** -- Chorioretinal Lesion
## Splits
The dataset explicitly splits the data into Train (1920 images), Test (640 images) and Validation (640 images) sets 
## File Schema
```
04_RFMid/
Testing/
Training/
Validation/
RFMiD_Testing_Labels.csv
RFMiD_Training_Labels.csv
RFMiD_Validation_Labels.csv
```

## Storage in database

### Tables populated

- **`datasets`** — One row registered with `dataset_name="RFMiD"`, `source_url`, `license="CC-BY-4.0"`, and `modality_types=["fundus"]`.
- **`images`** — One row per image. Fields populated via `get_image_metadata_dict` plus `modality="fundus"` and `original_image_id` set to the string form of the `ID` column. Images are located at `{split_folder}/{ID}.png` (split folder is `Training/`, `Testing/`, or `Validation/`). No laterality is stored.
- **`patients`** — One patient record per image (RFMiD provides one image per patient). The `original_patient_id` is the same as the image ID string. The `comorbidities` field (JSONB) stores a dictionary of all 44 remaining conditions (excluding `DR` and `ARMD`) with their full names as keys and boolean values, plus `Disease_Risk`. For example: `{"Disease Risk": true, "Macular Hole": false, ...}`.
- **`patient_images`** — One row linking each patient to their single image.
- **`classification_annotations`** — Two rows per image: one binary classification for `DR` (class name `"DR"`) and one for `ARMD` (class name `"AMD"`). Values are cast to bool from the CSV integer (0→False, 1→True). `task_type="binary"`, `annotation_method="manual"`.
- **`dataset_splits`** / **`image_split`** — Three explicit splits are created: `train` (from `RFMiD_Training_Labels.csv` / `Training/`), `test` (from `RFMiD_Testing_Labels.csv` / `Testing/`), and `val` (from `RFMiD_Validation_Labels.csv` / `Validation/`). Each image is assigned to its corresponding split.

### Provenance / raw annotation files

The three CSV files (`RFMiD_Training_Labels.csv`, `RFMiD_Testing_Labels.csv`, `RFMiD_Validation_Labels.csv`) are processed via `process_csv`, which registers each in `raw_annotation_files` (with file path, hash, and `annotation_type="classification"`) and creates a `provenance_chain` entry per file. Classification annotations reference the corresponding `raw_file_id` and `provenance_chain_id`.

### Annotation taxonomy

- **Classified as `classification_annotations`**: `DR` (Diabetic Retinopathy) and `ARMD` (AMD — Age-Related Macular Degeneration).
- **Stored in patient `comorbidities` JSONB** (44 conditions + Disease_Risk): Disease Risk, Macular Hole, Drusen, Myopia-related retinal changes, Branch Retinal Vein Occlusion, Tessellation, Epiretinal Membrane, Laser Scars, Macular Scar, Central Serous Retinopathy, Optic Disc Cupping, Central Retinal Vein Occlusion, Tortuous Vessels, Asteroid Hyalosis, Optic Disc Pallor, Optic Disc Edema, Subretinal Tissue or Scar, Anterior Ischemic Optic Neuropathy, Papillitis, Retinal Tear, Retinal Scar, Chorioretinal Scar, Exudation, Retinal Pigment Epithelial Changes, Lamellar Macular Hole, Retinitis Pigmentosa, Cotton Wool Spots, Coloboma, Optic Disc Pit Maculopathy, Preretinal Hemorrhage, Myelinated Nerve Fibers, Hemorrhage, Central Retinal Artery Occlusion, Tractional Detachment, Cystoid Macular Edema, Post-Traumatic Chorioretinopathy, Chorioretinal Fibrosis, Vitreous Hemorrhage, Microaneurysm, Vessel Sheathing, Branch Retinal Artery Occlusion, Plaque, Hemorrhagic Pigment Epithelial Detachment, Chorioretinal Lesion.

### Special processing

The three CSVs are processed concurrently via `asyncio.gather`. Upsert order respects foreign key constraints: images → patients → classifications → patient-image links. The script is idempotent: re-running does not create duplicates due to deterministic UUID generation and upsert logic.