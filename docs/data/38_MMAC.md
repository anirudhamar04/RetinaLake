# MMAC

## Overview
MMAC 2023 (Myopic Maculopathy Analysis Challenge) is a public benchmark dataset for automated analysis of myopic maculopathy in colour fundus images. The challenge covers two ingested tasks: (1) graded classification of myopic maculopathy severity using the META-PM scale, and (2) binary segmentation of three types of myopic maculopathy plus lesions. Patient demographics (age, sex, height, weight, acquisition site) are provided in the annotation CSVs.

Annotations were independently reviewed by two ophthalmologists; a third senior ophthalmologist resolved discrepancies. Segmentation lesions were annotated by one ophthalmologist, refined by a second, and finalised by a senior clinician.

**Note:** Task 3 (Spherical Equivalent prediction) is not ingested — it is a regression target with no corresponding image annotation.

## Images
- All images are **PNG** format.
- Task 1 images: `mmac_task_1_train_NNNN.png` / `mmac_task_1_val_NNNN.png`.
- Task 2 images include a lesion suffix: `mmac_task_2_train_LC_NNNN.png` (Lacquer Cracks), `mmac_task_2_train_CNV_NNNN.png` (Choroidal Neovascularization), `mmac_task_2_train_FS_NNNN.png` (Fuchs Spot).
- Training and validation sets are provided; no labelled public test set.

## Annotations

### Task 1 — Myopic Maculopathy Classification (META-PM Scale)
Stored as `disease_gradings` with `disease_type = "myopic_maculopathy"` and `scale_name = "META_PM_0_4"`.

| Grade | Label | Description |
|-------|-------|-------------|
| 0 | C0 | No myopic retinal degenerative lesions |
| 1 | C1 | Tessellated fundus (distinct choroidal vessels visible) |
| 2 | C2 | Diffuse chorioretinal atrophy (yellowish-white posterior pole) |
| 3 | C3 | Patchy chorioretinal atrophy (well-defined white lesions) |
| 4 | C4 | Macular atrophy (white atrophic lesion in the fovea) |

### Task 2 — Segmentation of Plus Lesions
Three binary PNG segmentation masks per image:

| Lesion | `annotation_type` |
|--------|------------------|
| Lacquer Cracks | `lacquer_cracks` |
| Choroidal Neovascularization | `choroidal_neovascularization` |
| Fuchs Spot | `fuchs_spot` |

Masks use the same filename as the corresponding image. Some validation images may not have an associated mask (withheld for challenge evaluation).

### Patient Metadata
Each annotation CSV includes `age`, `sex`, `height` (cm), `weight` (kg), `data_center`. Rows where both `age` and `sex` are missing/empty are ingested without a patient record (no error raised).

## Splits
Explicit training (`train`) and validation (`val`) splits. Images appearing in multiple lesion-type CSVs share a single image record (idempotent upsert) and are assigned to the same split.

## File Schema
```
38_MMAC/
├── 1. Classification of Myopic Maculopathy/
│   ├── 1. Images/
│   │   ├── 1. Training Set/    mmac_task_1_train_NNNN.png
│   │   └── 2. Validation Set/  mmac_task_1_val_NNNN.png
│   └── 2. Groundtruths/
│       ├── 1. MMAC2023_Myopic_Maculopathy_Classification_Training_Labels.csv
│       └── 2. MMAC2023_Myopic_Maculopathy_Classification_Validation_Labels.csv
└── 2. Segmentation of Myopic Maculopathy Plus Lesions/
    ├── 1. Lacquer Cracks/
    │   ├── 1. Images/{1. Training Set, 2. Validation Set}/
    │   └── 2. Groundtruths/
    │       ├── 1. Training Set/      <binary PNG masks>
    │       ├── 2. Validation Set/    <binary PNG masks>
    │       ├── 1. MMAC2023_..._Lacquer_Cracks.csv
    │       └── 2. MMAC2023_..._Lacquer_Cracks.csv
    ├── 2. Choroidal Neovascularization/  (same layout)
    └── 3. Fuchs Spot/                    (same layout)
```

## Storage in Database

### Tables Populated

**`datasets`** — one record, `dataset_name = "MMAC"`.

**`images`** — one record per unique image file; duplicates across lesion CSVs resolved by idempotent upsert.

**`disease_gradings`** — Task 1 grades: `disease_type = "myopic_maculopathy"`, `scale_name = "META_PM_0_4"`, `annotation_method = "manual"`.

**`segmentation_annotations`** — Task 2 binary masks for `lacquer_cracks`, `choroidal_neovascularization`, `fuchs_spot`. Processed PNGs saved under `<storage_root>/MMAC/masks/<annotation_type>/`.

**`patients`** / **`patient_images`** — one patient per image row where `age` or `sex` is present.

**`grading_scales`** — `META_PM_0_4` auto-registered on first run.

**`dataset_splits`** / **`image_split_assignments`** — `train` and `val` (explicit split type).
