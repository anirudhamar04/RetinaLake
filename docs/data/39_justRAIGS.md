# justRAIGS

## Overview
justRAIGS (Justified Referral in AI Glaucoma Screening) is a large-scale colour fundus dataset of ~101,000 eyes from the Rotterdam EyePACS screening programme. Each eye is labelled for referable glaucoma by up to three independent graders. In addition to a consensus final label (RG/NRG), individual grader decisions and 10 binary structural features (e.g. disc haemorrhage, rim absence, RNFL defect) are provided per grader, enabling analysis of grader agreement and feature-driven referral justification.

## Images
- All images are **JPEG** format (`.JPG` extension).
- Images are stored under `train/{last_digit_of_Eye_ID}/Eye_ID.JPG`.
  E.g. `TRAIN000000.JPG` → `train/0/TRAIN000000.JPG`
- ~101,000 training images; no labelled public test set.

## Metadata
Labels are stored in `JustRAIGS_Train_labels.csv` (semicolon-delimited, ~101,424 rows).

### CSV Columns
| Column | Description |
|--------|-------------|
| `Eye ID` | Image identifier (e.g. `TRAIN000000`) |
| `Final Label` | Consensus label: `RG` (Referable Glaucoma) or `NRG` (Non-Referable Glaucoma) |
| `Fellow Eye ID` | Contralateral eye ID |
| `Age` | Patient age |
| `Label G1` | Grader 1 label (`RG`/`NRG`/empty) |
| `Label G2` | Grader 2 label |
| `Label G3` | Grader 3 label (often empty — third grader not always present) |
| `G1 ANRS` … `G1 LC` | 10 binary features for grader 1 |
| `G2 ANRS` … `G2 LC` | 10 binary features for grader 2 |
| `G3 ANRS` … `G3 LC` | 10 binary features for grader 3 (all empty when G3 absent) |

### Binary Feature Abbreviations
| Code | Feature |
|------|---------|
| ANRS | Absent Neuroretinal Rim Superiorly |
| ANRI | Absent Neuroretinal Rim Inferiorly |
| RNFLDS | Retinal Nerve Fiber Layer Defect Superiorly |
| RNFLDI | Retinal Nerve Fiber Layer Defect Inferiorly |
| BCLVS | Baring of Circumlinear Vessel Superiorly |
| BCLVI | Baring of Circumlinear Vessel Inferiorly |
| NVT | Nasalisation of Vessels at the Disc |
| DH | Disc Hemorrhage |
| LD | Large Disc |
| LC | Large Cup |

## Splits
All labelled images (training set) are assigned to the `train` split. The challenge test set is closed and has no public labels.

## File Schema
```
39_justRAIGS/
├── JustRAIGS_Train_labels.csv
└── train/
    ├── 0/    TRAIN000000.JPG, TRAIN000010.JPG, ...
    ├── 1/    TRAIN000001.JPG, TRAIN000011.JPG, ...
    ...
    └── 9/    TRAIN000009.JPG, TRAIN000019.JPG, ...
```

## Storage in Database

### Tables Populated

**`datasets`** — one record, `dataset_name = "justRAIGS"`.

**`images`** — one record per Eye ID (~101,000 images).

**`experts`** — three records: `justRAIGS G1`, `justRAIGS G2`, `justRAIGS G3`.

**`expert_annotations`** — one record per grader per image where that grader was present, `annotation_type = "classification"`.

**`classification_annotations`**
- Consensus Final Label: `class_name = "glaucoma"`, `task_type = "binary"`, `annotation_method = "consensus"`, `expert_annotation_id = NULL`.
- Per-grader label: same `class_name = "glaucoma"`, `annotation_method = "manual"`, linked to `expert_annotations` via `expert_annotation_id`.
- Per-grader features: `class_name = "glaucoma_features_g1"` / `"glaucoma_features_g2"` / `"glaucoma_features_g3"`, `task_type = "multi_label"`, keys are lowercase feature codes (`anrs`, `anri`, `rnflds`, …).

**`dataset_splits`** / **`image_split_assignments`** — all images assigned to `train`.

### Notes
- The CSV uses semicolon (`;`) as delimiter — not supported by `read_csv_auto`; the ingestion script reads it with `csv.DictReader(delimiter=";")` and registers provenance via `register_csv_file` directly.
- G3 grader columns are empty for many rows; the script skips any grader with an empty `Label Gn` column without raising an error.
- Feature columns with empty values (not graded) are omitted from the multi-label dict.
