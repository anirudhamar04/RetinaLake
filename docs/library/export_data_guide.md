# Exporting Data from ChaksuDB

A complete guide to using ChaksuDB as a Python library for exporting ophthalmic image data and annotations to power downstream machine learning tasks.

---

## Table of Contents

1. [Installation &amp; Setup](#1-installation--setup)
2. [Core Concepts](#2-core-concepts)
3. [The `export()` Function](#3-the-export-function)
4. [Building an ExportSpec](#4-building-an-exportspec)
5. [Output Formats](#5-output-formats)
6. [Annotation Task Reference](#6-annotation-task-reference)
7. [Transforms (Spatial &amp; Photometric)](#7-transforms-spatial--photometric)
8. [End-to-End Examples](#8-end-to-end-examples)
9. [Output Schema Reference](#9-output-schema-reference)
10. [Performance &amp; Best Practices](#10-performance--best-practices)
11. [Debugging &amp; Troubleshooting](#11-debugging--troubleshooting)

---

## 1. Installation & Setup

### Install the package

```bash
# With uv (recommended)
uv sync

# With pip (editable install)
pip install -e .
```

**Requirements:** Python >= 3.11

### Configure the database connection

Create a `.env` file in the project root (or set environment variables directly):

```bash
DB_HOST=127.0.0.1
DB_PORT=5432
DB_DATABASE=chaksudb
DB_USER=chaksuai
DB_PASSWORD=<your-password>

STORAGE_LOCAL_ROOT=./data/processed
STORAGE_DATA_ROOT=./data/raw
```

The PostgreSQL database must be running and populated before exporting. See
[`database_setup.md`](database_setup.md) for creating the database and applying the schema.

### Verify connectivity

```python
from chaksudb.db import init_pool, close_pool

import asyncio

async def check():
    await init_pool()
    print("Connected to ChaksuDB")
    await close_pool()

asyncio.run(check())
```

---

## 2. Core Concepts

### Architecture

```
ExportSpec  →  QueryBuilder  →  SQL  →  Streaming Rows  →  Parquet / PyTorch
```

| Concept                       | Description                                                                                                       |
| ----------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| **`ExportSpec`**      | A Pydantic model that declares*what* to export — datasets, annotation tasks, filters, and options.             |
| **`export()`**        | Single entry-point function. Give it a spec and choose Parquet, PyTorch Dataset, or DataLoader.                   |
| **Modules**             | Composable query modules (grading, segmentation, classification, etc.) automatically activated based on the spec. |
| **Image-centric rows**  | Every output row corresponds to one image. Annotations are pivoted or aggregated onto that row.                   |
| **No pixels in the DB** | The database stores file paths and metadata only. Images are loaded at read-time by the PyTorch datasets.         |

### Imports

```python
from chaksudb.export import ExportSpec, export
```

These two imports are all you need for most workflows.

---

## 3. The `export()` Function

```python
def export(
    spec: ExportSpec,
    *,
    parquet_path: Optional[Path] = None,        # Write a Parquet file
    torch: Optional[Literal["dataset",
                             "dataloader"]] = None,  # Return PyTorch object
    spatial: Optional[list] = None,             # List of BaseSpatialTransform (image + masks + coords)
    transform: Any = None,                       # Image-only transform(s): list or callable PIL -> PIL
    collate_fn: Any = "default",                 # "default" | "padded" | "packed" | callable
    batch_size: int = 32,                        # DataLoader batch size
    shuffle: bool = False,                       # DataLoader shuffle
    num_workers: int = 0,                        # DataLoader workers
    cache_rows: bool = False,                    # Cache all DB rows in RAM
    parquet_batch_size: int = 5000,              # Streaming batch size
    **dataloader_kwargs,                         # Extra DataLoader args
) -> Path | Dataset | DataLoader | None
```

- **`spatial`** — List of spatial transform instances. Applied to the image, all segmentation masks, and all localization coordinates (bboxes, keypoints, circles) *together*, so geometry stays aligned. See [§7 Transforms (Spatial & Photometric)](#7-transforms-spatial--photometric).
- **`transform`** — Image-only pipeline: a single callable `PIL.Image -> PIL.Image` (or Tensor), or a list of photometric transforms. Applied *after* the spatial pipeline. The old `(image, annotations) -> (image, annotations)` signature is no longer supported.
- **`collate_fn`** — Batching strategy: `"default"` (stack images, lists for annotations), `"padded"` (pad masks/coords to max count), or `"packed"` (concatenate with sample indices). See [§7.4 Collate strategies](#74-collate-strategies).

### Behaviour matrix

| `parquet_path` | `torch`        | Returns                                          |
| ---------------- | ---------------- | ------------------------------------------------ |
| set              | `None`         | `Path` — the written Parquet file             |
| `None`         | `"dataset"`    | `QueryDataset` (query-backed PyTorch Dataset)  |
| `None`         | `"dataloader"` | `DataLoader` wrapping a `QueryDataset`       |
| set              | `"dataset"`    | `ParquetDataset` (reads from the Parquet file) |
| set              | `"dataloader"` | `DataLoader` wrapping a `ParquetDataset`     |
| `None`         | `None`         | `None`                                         |

When both `parquet_path` and `torch` are provided, the Parquet file is written first, then the PyTorch object is backed by the file (not the DB).

---

## 4. Building an ExportSpec

`ExportSpec` is a Pydantic model. Every field is optional, so a minimal spec is valid:

```python
spec = ExportSpec()  # All images, no annotations
```

### 4.1 Dataset filters

```python
# By name
spec = ExportSpec(dataset_names=["EYEPACS", "MESSIDOR"])

# By UUID
from uuid import UUID
spec = ExportSpec(dataset_ids=[UUID("...")])
```

### 4.2 Split filters

```python
spec = ExportSpec(
    split_names=["train", "val"],       # Filter by split name
    split_task_type="classification",   # Filter by the task type of the split
)
```

### 4.3 Image scope filters

```python
spec = ExportSpec(
    modalities=["fundus", "oct"],     # Valid: fundus, oct, fa, uwf
    storage_provider="local",         # Valid: local, s3, gcs, azure, http
)
```

### 4.4 Annotation tasks

Choose which annotation types to include. Each one activates its query module:

```python
spec = ExportSpec(
    annotation_tasks=[
        "grading",          # Disease severity grades (DR, DME, Glaucoma, AMD)
        "segmentation",     # Segmentation masks (vessels, lesions, optic disc)
        "classification",   # Binary / multi-class / multi-label labels
        "localization",     # Bounding boxes, keypoints, center points
        "quality",          # Image quality scores and labels
        "keyword",          # Diagnostic keywords
        "description",      # Clinical text descriptions
    ]
)
```

### 4.5 Annotation source preference

Controls whether to use expert annotations, consensus annotations, or both:

```python
spec = ExportSpec(
    annotation_source="prefer_consensus",  # Default — consensus if available, else expert
    # Options:
    #   "prefer_consensus"  — prefer consensus, fall back to expert
    #   "expert_only"       — only expert annotations
    #   "consensus_only"    — only consensus annotations
    #   "both"              — include both when available
)
```

### 4.6 Annotation requirement mode

Controls how images without annotations are handled:

```python
spec = ExportSpec(
    annotation_tasks=["grading"],
    require_annotations_mode="all",
    # Options:
    #   "none"  — include all images (LEFT JOIN); missing annotations are null
    #   "all"   — only images with ALL requested annotation tasks (INNER JOIN)
    #   "any"   — only images with at least ONE requested annotation task
)
```

### 4.7 Grading options

Requires `"grading"` in `annotation_tasks`:

```python
spec = ExportSpec(
    annotation_tasks=["grading"],
    disease_types=["DR", "DME"],        # Filter by disease: DR, DME, Glaucoma, AMD
    grading_scale_name="ICDR",          # Use a specific grading scale
    include_original_grade=True,        # Include the original text grade
    include_scaled_grade=True,          # Include the normalised integer grade
    grade_filter={
        "DR": {"min": 1, "max": 3},    # Range filter (inclusive)
        # OR: "DR": {"values": [0, 1, 2]}
        # OR: "DR": [0, 1, 2]           # Shorthand for values
    },
)
```

### 4.8 Segmentation filters

Requires `"segmentation"` in `annotation_tasks`:

```python
spec = ExportSpec(
    annotation_tasks=["segmentation"],
    segmentation_types=["vessel", "optic_disc", "lesion"],
    lesion_subtypes=["microaneurysm", "hemorrhage"],
)
```

### 4.9 Localization filters

Requires `"localization"` in `annotation_tasks`:

```python
spec = ExportSpec(
    annotation_tasks=["localization"],
    localization_types=["bounding_box", "keypoint", "center_point"],
)
```

### 4.10 Classification options

Requires `"classification"` in `annotation_tasks`. There are **two interfaces**:

#### Concept-centric (recommended)

The simplest, cross-dataset path. Ask for a clinical *concept* and get a unified
`{concept}_present` 0/1 column, no matter how each dataset stored it (binary / multi_class /
multi_label):

```python
spec = ExportSpec(
    annotation_tasks=["classification"],
    classification_concepts=["DR", "AMD", "Glaucoma"],   # → DR_present, AMD_present, Glaucoma_present
    # Optional: keep only images positive for any of these concepts (any storage shape)
    classification_positive_for=["Glaucoma"],
)
```

Concepts are defined once in `chaksudb/ingest/framework/concepts.py` and understood by both
ingest and export. A `{concept}_present=0` is only trustworthy where the dataset actually
assessed that concept — scope `dataset_names` for clean negatives.

#### Task pivots (exact, per-task columns)

For exact task pivots, set `classification_class_names` (now **optional** — use it only when
you want specific task columns; otherwise prefer concepts or `build_dataset_spec` discovery):

```python
spec = ExportSpec(
    annotation_tasks=["classification"],

    # Optional — which task_names to pivot into flat columns
    classification_class_names=["glaucoma", "disease_category"],

    # Optional — explicitly declare task type per class
    classification_task_types={
        "glaucoma": "binary",
        "disease_category": "multi_class",
    },

    # Label type: "int" (default, for CrossEntropyLoss) or "float" (for BCELoss / soft labels)
    classification_label_type="int",
)
```

For **multi-label** classification with explicit sub-keys:

```python
spec = ExportSpec(
    annotation_tasks=["classification"],
    classification_class_names=["disease_indicators"],
    classification_task_types={"disease_indicators": "multi_label"},
    multi_label_keys={
        "disease_indicators": ["normal", "diabetes", "glaucoma", "cataract", "amd"]
    },
)
# Produces columns: disease_indicators_normal, disease_indicators_diabetes, ...
```

Without `multi_label_keys`, multi-label results are stored as a JSON string column.

### 4.11 OR filter groups

Combine conditions with OR logic within a single annotation task:

```python
from chaksudb.export.spec import AnnotationOrFilter

spec = ExportSpec(
    annotation_tasks=["segmentation"],
    annotation_or_filters=[
        AnnotationOrFilter(
            task="segmentation",
            conditions=[
                {"segmentation_types": ["lesion"], "lesion_subtypes": ["microaneurysm"]},
                {"segmentation_types": ["vessel"]},
            ]
        )
    ],
)
# Matches: (lesion + microaneurysm) OR (vessel)
```

### 4.12 Path handling

```python
spec = ExportSpec(
    base_path_for_paths="/mnt/data/images",
    # file_path values are prepended with this base path in the output
)
```

### 4.13 Patient data

Include patient demographics alongside image annotations. Requires `patients` rows to exist for the relevant images (populated by ingestion scripts that call `upsert_patient`).

```python
spec = ExportSpec(
    dataset_names=["PAPILA", "ODIR-5K"],
    annotation_tasks=["grading"],
    disease_types=["Glaucoma"],
    include_patient_data=True,
)
# Added columns: patient_id, original_patient_id, age, sex, ethnicity, comorbidities
```

This is a `LEFT JOIN` — images without an associated patient still appear in the output; the patient columns are `NULL` for those rows.

### 4.14 Caption generation for VLM tasks

Generate text captions from existing annotations for vision-language model (VLM) training.

```python
spec = ExportSpec(
    dataset_names=["EYEPACS", "IDRID"],
    caption_mode="all",
    # Options: "clinical" | "keyword" | "grading" | "classification" | "all"
)
# Added columns:
#   caption_clinical_text    — clinical description text (if present)
#   caption_keywords         — comma-joined keyword terms
#   caption_grade_data       — JSONB array of {disease_type, original_grade, grade_label}
#   caption_class_data       — JSONB array of {class_name, class_value}
#   caption_loc_structures   — JSONB array of localisation target structures
#   caption_seg_structures   — JSONB array of segmentation annotation types
```

**Two-phase design**: `CaptionModule` adds the raw structured caption columns to the SQL query. After the query, `CaptionEngine.synthesize(row)` combines them into a single human-readable sentence using a definitions dictionary.

```python
from chaksudb.export.caption_engine import CaptionEngine

# Instantiate with your definitions and abbreviation dictionaries
engine = CaptionEngine(definitions=my_defs, abbreviations=my_abbrevs)

# Synthesize a caption from a row (dict)
caption = engine.synthesize(row)
# e.g. "A fundus photograph showing moderate non-proliferative diabetic retinopathy
#        with microaneurysms and hard exudates."
```

See [§8.9 VLM captioning end-to-end](#89-vlm-captioning-end-to-end) for a complete example.

### 4.15 COCO detection format

Export localization annotations as a sidecar COCO JSON file, compatible with detectron2, mmdet, and pycocotools.

```python
from pathlib import Path
from chaksudb.export import ExportSpec, export

spec = ExportSpec(
    annotation_tasks=["localization"],
    localization_types=["bounding_box"],
    require_annotations_mode="all",
    detection_format="coco",
    detection_category_map={"lesion": 1, "optic_disc": 2, "fovea": 3},
)

export(
    spec,
    parquet_path=Path("out.parquet"),
    coco_path=Path("out_coco.json"),
)
# Writes two files:
#   out.parquet    — standard tabular export (always written when parquet_path is set)
#   out_coco.json  — COCO-format JSON with images, annotations, categories
```

If `detection_category_map` is omitted, categories are auto-assigned from the data.

### 4.16 Health status (normal / abnormal)

A single cross-dataset field derived from grading + disease classification. Useful for
assembling clean "normal" or "abnormal" cohorts across many datasets at once.

```python
spec = ExportSpec(
    dataset_names=["FIVES", "AIROGS"],
    include_health_status=True,        # adds a `health_status` column: 'normal'/'abnormal'/None
)

# Or keep only one class (implies include_health_status):
spec = ExportSpec(dataset_names=["FIVES", "AIROGS"], health_status_filter="normal")
```

`health_status` is `None` (and excluded by the filter) where a dataset never assessed the
relevant concept — **absence is not a negative**. Scope `dataset_names` for trustworthy labels.

### 4.17 IQA quality filtering & fundus ROI

`scripts/run_roi_iqa.py` pre-computes an AutoMorph image-quality score and a fundus ROI circle
for every image (see [`iqa_roi_detection.md`](iqa_roi_detection.md)). At export time:

```python
# Keep only good-quality images (p_good >= threshold, 0–1)
spec = ExportSpec(iqa_min_quality_score=0.7)

# Or filter by label
spec = ExportSpec(iqa_quality_labels=["good", "usable"])

# Add flat ROI columns for custom DataLoaders / masking
spec = ExportSpec(include_fundus_roi=True)
# → columns: fundus_roi_cx, fundus_roi_cy, fundus_roi_radius, fundus_roi_method
```

The ROI circle can also be applied as a transform — see `FundusROIMask` in §7.

### 4.18 Discover everything a dataset has (`build_dataset_spec`)

Don't know a dataset's exact tasks? `build_dataset_spec` introspects the DB and returns a fully
populated `ExportSpec` that flattens everything it finds (grading + full classification panel +
per-type quality + segmentation + localization + patient):

```python
from chaksudb.export.discovery import build_dataset_spec   # async

spec = await build_dataset_spec(["BRSET"])
export(spec, parquet_path="brset_flat.parquet")
```

To inspect what's available without building a spec:

```python
from chaksudb.db.queries.annotation_types import (
    list_classification_tasks, list_quality_types,
)
```

---

## 5. Output Formats

### 5.1 Parquet

Best for: data exploration, Pandas/Polars analysis, archival, and fast repeated PyTorch loading.

```python
from pathlib import Path
from chaksudb.export import ExportSpec, export

spec = ExportSpec(
    dataset_names=["EYEPACS"],
    annotation_tasks=["grading"],
    disease_types=["DR"],
)

path = export(spec, parquet_path=Path("eyepacs_dr.parquet"))
```

Read the file back:

```python
import pyarrow.parquet as pq
table = pq.read_table("eyepacs_dr.parquet")
df = table.to_pandas()
print(df.head())
print(df.columns.tolist())
```

### 5.2 PyTorch Dataset

Best for: custom iteration, inspection, or building your own DataLoader.

```python
dataset = export(spec, torch="dataset")

print(len(dataset))        # Total row count
image, annotations = dataset[0]
# image: PIL.Image.Image (RGB)
# annotations: dict with all annotation fields
```

Use `cache_rows=True` for small datasets to avoid per-index DB queries:

```python
dataset = export(spec, torch="dataset", cache_rows=True)
```

### 5.3 PyTorch DataLoader

Best for: training loops. Images are auto-loaded, batched, and optionally padded.

```python
dataloader = export(
    spec,
    torch="dataloader",
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True,       # Passed through to DataLoader
)

for images, annotations in dataloader:
    # images: Tensor [B, 3, H, W]  (padded to max size in batch)
    # annotations: dict of lists, e.g. annotations["dr_grade"] → [2, 0, 3, ...]
    pass
```

### 5.4 Parquet then PyTorch (combined)

Write once, load fast on every training run:

```python
dataloader = export(
    spec,
    parquet_path=Path("train.parquet"),
    torch="dataloader",
    batch_size=32,
    shuffle=True,
)
# Parquet is written first; DataLoader reads from the file, not the DB.
```

### 5.5 Presets

One-line factory functions for the most common ML configurations. Import from `internal.export.presets`.

```python
from chaksudb.export import presets

spec = presets.dr_classification(datasets=["EYEPACS"], split="train")
```

All presets accept optional `datasets` (list of dataset names) and `split` (split name) arguments. Task-specific presets may accept additional arguments as noted below.

| Preset | Function | Description |
|---|---|---|
| DR grading | `dr_classification(datasets, split)` | 5-class DR severity; requires grading annotations |
| Glaucoma detection | `glaucoma_detection(datasets, split)` | Binary glaucoma; uses classification annotations |
| Lesion segmentation | `lesion_segmentation(lesion_types, datasets, split)` | MA, HE, EX, SE lesion masks |
| Optic disc segmentation | `optic_disc_segmentation(datasets, split)` | OD + cup masks for CDR estimation |
| COCO lesion detection | `lesion_detection_coco(datasets, split, category_map)` | Bounding boxes in COCO format |
| Fundus captioning | `fundus_captioning(datasets, split)` | All caption columns; for VLM training |
| Quality assessment | `quality_assessment(datasets, split)` | Image quality scores and labels |
| Multi-label disease | `multi_label_disease(class_names, datasets, split)` | ODIR-style multi-label classification |
| Landmark detection | `landmark_detection(datasets, split)` | Fovea + OD keypoint detection |
| Multi-task | `multi_task(tasks, datasets, split)` | Grading + segmentation + localization combined |

### 5.6 ContrastiveDataset

Yields `(anchor, positive, negative)` triplets from a Parquet file for contrastive or retrieval-based training. Images sharing the same value in `label_column` are treated as positives; images with different values are negatives.

```python
from chaksudb.export.contrastive_dataset import ContrastiveDataset
from pathlib import Path

# 1. Export a Parquet file first
spec = ExportSpec(
    annotation_tasks=["grading"],
    disease_types=["DR"],
    require_annotations_mode="all",
)
export(spec, parquet_path=Path("dr_train.parquet"))

# 2. Create the contrastive dataset
dataset = ContrastiveDataset(
    parquet_path=Path("dr_train.parquet"),
    label_column="dr_grade",      # column to define positive/negative pairs
    seed=42,
)

anchor, positive, negative = dataset[0]
# anchor, positive: images with the same dr_grade value
# negative: image with a different dr_grade value
```

`ContrastiveDataset` reads only the `label_column` and `file_path` columns from Parquet for efficient indexing, then loads images on demand. An optional `transform` callable (PIL → PIL or PIL → Tensor) can be passed for augmentation.

---

## 6. Annotation Task Reference

### 6.1 Grading

Pivots disease grades into per-disease columns.

| Output column                   | Type    | Description                                                   |
| ------------------------------- | ------- | ------------------------------------------------------------- |
| `{disease}_grade`             | `int` | Normalised integer grade (when `include_scaled_grade=True`) |
| `{disease}_original_grade`    | `str` | Original text grade (when `include_original_grade=True`)    |
| `{disease}_scale_name`        | `str` | Name of the grading scale used                                |
| `{disease}_annotation_source` | `str` | `"expert"` or `"consensus"`                               |

Example: with `disease_types=["DR"]`, you get `dr_grade`, `dr_original_grade`, `dr_scale_name`, `dr_annotation_source`.

### 6.2 Classification

Pivots each class_name into flat, training-ready columns.

**Binary / multi-class:**

| Output column                | Type                 | Description               |
| ---------------------------- | -------------------- | ------------------------- |
| `{class_name}_label`       | `int` or `float` | Numeric label             |
| `{class_name}_class_label` | `str`              | Human-readable class name |

**Multi-label with keys:**

| Output column          | Type                 | Description                  |
| ---------------------- | -------------------- | ---------------------------- |
| `{class_name}_{key}` | `int` or `float` | One column per sub-label key |

**Multi-label without keys:**

| Output column           | Type    | Description                                         |
| ----------------------- | ------- | --------------------------------------------------- |
| `{class_name}_labels` | `str` | JSON string, e.g.`'{"normal": 0, "diabetes": 1}'` |

### 6.3 Segmentation

| Output column          | Type       | Description                                                                          |
| ---------------------- | ---------- | ------------------------------------------------------------------------------------ |
| `segmentation_masks` | JSON array | Each element:`{annotation_type, lesion_subtype, mask_file_path, confidence_score}` |

### 6.4 Localization

| Output column                | Type       | Description                                                                         |
| ---------------------------- | ---------- | ----------------------------------------------------------------------------------- |
| `localization_annotations` | JSON array | Each element:`{localization_type, target_structure, coordinates, lesion_subtype}` |

### 6.5 Quality

| Output column                    | Type       | Description                                   |
| -------------------------------- | ---------- | --------------------------------------------- |
| `{quality_type}_quality_score` | `float`  | Numeric quality score                         |
| `{quality_type}_quality_label` | `str`    | Categorical quality label                     |
| `quality_annotations`          | JSON array | All quality annotations as structured objects |

Common quality types: `overall`, `gradability`, `clarity`, `blur`, `contrast`, `illumination`.

### 6.6 Keywords

| Output column | Type      | Description                     |
| ------------- | --------- | ------------------------------- |
| `keywords`  | `str[]` | Array of distinct keyword terms |

### 6.7 Description (Clinical)

| Output column                 | Type    | Description                                                 |
| ----------------------------- | ------- | ----------------------------------------------------------- |
| `clinical_description_text` | `str` | Primary clinical description                                |
| `clinical_description_type` | `str` | Type:`diagnosis_text`, `clinical_caption`, or `notes` |
| `clinical_word_count`       | `int` | Word count                                                  |

### Always-present columns

These are included in every export regardless of annotation tasks:

| Column                | Type     | Description                                           |
| --------------------- | -------- | ----------------------------------------------------- |
| `image_id`          | `UUID` | Unique image identifier                               |
| `file_path`         | `str`  | Path to the image file                                |
| `storage_provider`  | `str`  | Storage backend (`local`, `s3`, etc.)             |
| `object_key`        | `str`  | Cloud object key (if applicable)                      |
| `modality`          | `str`  | Image modality (`fundus`, `oct`, `fa`, `uwf`) |
| `eye_laterality`    | `str`  | `left`, `right`, or `unknown`                   |
| `resolution_width`  | `int`  | Image width in pixels                                 |
| `resolution_height` | `int`  | Image height in pixels                                |
| `dataset_name`      | `str`  | Source dataset name                                   |

When split filters are used, `split_name` and `task_type` columns are also added.

---

## 7. Transforms (Spatial & Photometric)

The export pipeline supports a **two-layer transform system** so that images, segmentation masks, and localization coordinates stay aligned during augmentation.

### 7.1 Overview

| Layer        | Parameter   | Applied to                    | When        |
| ------------ | ----------- | ------------------------------ | ----------- |
| **Spatial**  | `spatial`   | Image + all masks + all coords | First       |
| **Photometric** | `transform` | Image only                  | After spatial |

- **Spatial transforms** (e.g. `Resize`, `RandomHorizontalFlip`, `RandomRotation`) use the same random parameters for the image and every mask, and correctly scale or warp bboxes, keypoints, and circles. Masks use nearest-neighbour interpolation for binary masks and bilinear for soft maps (based on `unified_format` in the export).
- **Photometric transforms** (e.g. `ColorJitter`, `CLAHE`, `ToTensor`) change only pixel values and do not touch masks or coordinates.

**Imports:**

```python
from chaksudb.export import ExportSpec, export
from chaksudb.export.transforms import Resize, RandomHorizontalFlip, RandomRotation, CLAHE, GammaCorrection
import torchvision.transforms as T
```

### 7.2 Spatial transforms

Spatial transforms take a `SpatialSample` (image + masks + coords) and return a `SpatialSample`. Use them in the `spatial` list.

**Affine (resize, crop, flip, rotate, affine):**

| Transform | Example | Description |
| --------- | ------- | ----------- |
| `Resize` | `Resize(512)` or `Resize((256, 256))` | Resize to fixed size |
| `RandomResizedCrop` | `RandomResizedCrop(224, scale=(0.08, 1.0))` | Random crop then resize |
| `CenterCrop` | `CenterCrop(224)` | Center crop |
| `RandomCrop` | `RandomCrop(224)` | Random offset crop |
| `Pad` | `Pad(32, fill=0)` | Pad edges |
| `RandomHorizontalFlip` | `RandomHorizontalFlip(p=0.5)` | Random horizontal flip |
| `RandomVerticalFlip` | `RandomVerticalFlip(p=0.5)` | Random vertical flip |
| `RandomRotation` | `RandomRotation(15)` | Random rotation ±15° |
| `RandomAffine` | `RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1))` | Translation + scale jitter |
| `RandomRescale` | `RandomRescale((0.8, 1.2))` | Random uniform scale |

**Non-affine:** `RandomPerspective`, `ElasticTransform`, `PolarTransform` (polar warp; not for use with localization).

**Annotation-aware:** `BoundingBoxCrop(target_structure, padding)` and `ROICrop(target_structure, padding)` crop around a named structure (e.g. `"optic_disc"`, `"fovea"`) using bbox or keypoint/circle data.

**Morphological** (apply to masks or image): `Erosion`, `Dilation`, `Opening`, `MorphologicalClosing`, `ConnectedComponentFiltering` — each accepts `apply_to="image" | "masks" | "both"` and optional `mask_types` filter.

**Example — augmentation for segmentation + localization:**

```python
from chaksudb.export.transforms import (
    Resize,
    RandomHorizontalFlip,
    RandomRotation,
    RandomAffine,
)

spatial = [
    Resize(512),
    RandomHorizontalFlip(p=0.5),
    RandomRotation(degrees=15, expand=False),
    RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
]

dataset = export(
    spec,
    torch="dataset",
    spatial=spatial,
)
image, ann = dataset[0]
# image and ann["_loaded_masks"], ann["_keypoints"], etc. are all consistently transformed
```

### 7.3 Photometric transforms

Photometric transforms are **image-only**. Use them as `transform` (a single callable or a list). They run after the spatial pipeline.

**Standard (torchvision re-exports):** `Normalize`, `ColorJitter`, `RandomAdjustSharpness`, `GaussianBlur`, `RandomAutocontrast`, `RandomEqualize`, `Grayscale`.

**Retinal-specific:** `CLAHE`, `HistogramMatching`, `MultiscaleRetinex`, `MSRCR`, `GammaCorrection`, `ContrastEnhancement`, `IlluminationCorrection`, `GreenChannelExtraction`, `BlueChannelEmphasis`, `BackgroundPolynomialCorrection`.

**Denoising:** `GaussianDenoising`, `MedianFiltering`, `BilateralFiltering`, `Deblurring`, `Deconvolution`.

**Example — spatial + photometric:**

```python
from chaksudb.export.transforms import Resize, RandomHorizontalFlip, CLAHE, GammaCorrection
import torchvision.transforms as T

dataset = export(
    spec,
    torch="dataset",
    spatial=[Resize(512), RandomHorizontalFlip(0.5)],
    transform=T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
)
tensor, ann = dataset[0]  # tensor: (C, H, W) float32
```

Using library photometric transforms (PIL in/out) before ToTensor:

```python
from chaksudb.export.transforms import CLAHE, GammaCorrection

dataset = export(
    spec,
    torch="dataset",
    spatial=[Resize(512)],
    transform=[CLAHE(clip_limit=2.0), GammaCorrection(gamma_range=(0.8, 1.2)), T.ToTensor()],
)
```

### 7.4 Collate strategies

When using `torch="dataloader"`, choose how to batch variable-length annotations:

| Strategy | Use case | Images | Masks / coords |
| -------- | --------- | ------ | ----------------- |
| `"default"` | General | Stacked `(B, C, H, W)` | Lists of lists |
| `"padded"` | Fixed-size batches | Stacked | Masks padded to `(B, max_masks, 1, H, W)` + counts; bboxes/keypoints padded + valid masks |
| `"packed"` | Concatenated batches | Stacked | Masks `(total_masks, 1, H, W)` + sample index; bboxes/keypoints with sample index column |

```python
dataloader = export(
    spec,
    torch="dataloader",
    spatial=[Resize(224)],
    transform=T.ToTensor(),
    collate_fn="padded",
    batch_size=16,
    shuffle=True,
)
```

### 7.5 PyTorch sample format when using spatial

When `spatial` is provided (or when using a dataset that loads masks), each `(image, annotations)` from the dataset includes:

| Key | Type | Description |
| --- | ------ | ------------ |
| `_loaded_masks` | `list[PIL.Image]` | Loaded mask images (same size as `image` after spatial) |
| `_mask_meta` | `list[dict]` | Per-mask metadata: `annotation_type`, `lesion_subtype`, `unified_format`, etc. |
| `_bboxes` | `list[dict]` | Bounding boxes with keys e.g. `xmin`, `ymin`, `xmax`, `ymax` (in output pixel coords) |
| `_keypoints` | `list[dict]` | Keypoints with `x`, `y`, `target_structure` (in output pixel coords) |
| `_circles` | `list[dict]` | Circles with `center_x`, `center_y`, `radius`, `target_structure` |

All coordinate lists are in the same coordinate system as the returned image. Without `spatial`, raw row data still includes `segmentation_masks` (paths) and `localization_annotations` (raw JSON); masks are not loaded automatically unless the spatial pipeline is used.

---

## 8. End-to-End Examples

### 8.1 DR grading training pipeline

```python
from pathlib import Path
from chaksudb.export import ExportSpec, export
from chaksudb.export.transforms import Resize
import torchvision.transforms as T

# 1. Define the spec
spec = ExportSpec(
    dataset_names=["EYEPACS"],
    split_names=["train"],
    annotation_tasks=["grading"],
    disease_types=["DR"],
    annotation_source="prefer_consensus",
    require_annotations_mode="all",
    modalities=["fundus"],
)

# 2. Spatial (resize) + photometric (ToTensor, Normalize)
train_loader = export(
    spec,
    torch="dataloader",
    spatial=[Resize((224, 224))],
    transform=T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)

# 3. Training loop
for images, annotations in train_loader:
    dr_grades = annotations["dr_grade"]  # List of ints
    # images: [B, 3, 224, 224] tensor
    # ... your training code ...
```

### 8.2 Binary glaucoma classification

```python
spec = ExportSpec(
    annotation_tasks=["classification"],
    classification_class_names=["glaucoma"],
    classification_task_types={"glaucoma": "binary"},
    classification_label_type="int",
    require_annotations_mode="all",
)

export(spec, parquet_path=Path("glaucoma.parquet"))

# Inspect
import pandas as pd
df = pd.read_parquet("glaucoma.parquet")
print(df[["image_id", "glaucoma_label", "glaucoma_class_label"]].head())
# image_id                              glaucoma_label  glaucoma_class_label
# 3f2a...                               1               positive
# 8b1c...                               0               negative
```

### 8.3 Multi-label disease classification with flat columns

```python
spec = ExportSpec(
    annotation_tasks=["classification"],
    classification_class_names=["disease_indicators"],
    classification_task_types={"disease_indicators": "multi_label"},
    multi_label_keys={
        "disease_indicators": [
            "normal", "diabetes", "glaucoma", "cataract",
            "amd", "hypertension", "myopia", "other",
        ]
    },
    classification_label_type="int",
    require_annotations_mode="all",
)

export(spec, parquet_path=Path("multilabel.parquet"))
# Columns: disease_indicators_normal, disease_indicators_diabetes, ...
```

### 8.4 Segmentation masks export (with spatial pipeline)

```python
from chaksudb.export.transforms import Resize, RandomHorizontalFlip

spec = ExportSpec(
    annotation_tasks=["segmentation"],
    segmentation_types=["vessel", "lesion"],
    lesion_subtypes=["microaneurysm", "hemorrhage"],
    require_annotations_mode="all",
)

dataset = export(
    spec,
    torch="dataset",
    spatial=[Resize(512), RandomHorizontalFlip(0.5)],
)
image, ann = dataset[0]
# Loaded masks (PIL, same size as image after spatial)
for mask_pil, meta in zip(ann["_loaded_masks"], ann["_mask_meta"]):
    print(meta["annotation_type"], meta.get("lesion_subtype"), mask_pil.size)
# Raw paths still in row-style data if needed: ann.get("segmentation_masks")
```

### 8.5 Cross-dataset export with harmonised grading

```python
spec = ExportSpec(
    dataset_names=["EYEPACS", "MESSIDOR", "IDRID"],
    split_names=["train"],
    annotation_tasks=["grading"],
    disease_types=["DR"],
    grading_scale_name="ICDR",
    annotation_source="prefer_consensus",
    require_annotations_mode="all",
)

export(spec, parquet_path=Path("cross_dataset_dr.parquet"))
```

### 8.6 Combined grading + classification + quality

```python
spec = ExportSpec(
    annotation_tasks=["grading", "classification", "quality"],
    disease_types=["DR"],
    classification_class_names=["glaucoma"],
    classification_task_types={"glaucoma": "binary"},
    require_annotations_mode="any",
)

export(spec, parquet_path=Path("combined.parquet"))
```

### 8.7 Export with OR filters (advanced)

```python
from chaksudb.export.spec import AnnotationOrFilter

spec = ExportSpec(
    annotation_tasks=["segmentation"],
    annotation_or_filters=[
        AnnotationOrFilter(
            task="segmentation",
            conditions=[
                {"segmentation_types": ["lesion"], "lesion_subtypes": ["microaneurysm"]},
                {"segmentation_types": ["vessel"]},
            ]
        )
    ],
    require_annotations_mode="all",
)

export(spec, parquet_path=Path("seg_or_filter.parquet"))
```

### 8.8 All annotations, all images

```python
spec = ExportSpec(
    annotation_tasks=[
        "grading", "segmentation", "localization",
        "classification", "quality", "keyword", "description",
    ],
    classification_class_names=["glaucoma", "disease_category"],
    require_annotations_mode="none",  # Include images even without annotations
)

export(spec, parquet_path=Path("everything.parquet"))
```

### 8.9 VLM captioning end-to-end

Generate training data for a vision-language model using the `fundus_captioning` preset and `CaptionEngine`.

```python
from pathlib import Path
import pyarrow.parquet as pq
from chaksudb.export import presets, export
from chaksudb.export.caption_engine import CaptionEngine

# 1. Export caption columns to Parquet
spec = presets.fundus_captioning(
    datasets=["EYEPACS", "IDRID", "PAPILA"],
    split="train",
)
export(spec, parquet_path=Path("vlm_train.parquet"))

# 2. Load the Parquet file
table = pq.read_table("vlm_train.parquet")
rows = table.to_pylist()

# 3. Instantiate CaptionEngine with your definitions
#    (definitions: dict[str, list[str]], abbreviations: dict[str, str])
definitions = {
    "dr grade 2": ["scattered microaneurysms", "hard exudates", "dot haemorrhages"],
    "moderate non-proliferative diabetic retinopathy": ["microaneurysms", "hard exudates"],
}
abbreviations = {"DR": "diabetic retinopathy", "MA": "microaneurysm"}
engine = CaptionEngine(definitions=definitions, abbreviations=abbreviations)

# 4. Synthesise captions
for row in rows:
    caption = engine.synthesize(row)
    # e.g. "A fundus photograph showing moderate non-proliferative diabetic
    #        retinopathy: microaneurysms, hard exudates."
    print(row["file_path"], caption)
```

`CaptionEngine` exposes per-source methods (`from_grading`, `from_grading_data`, etc.) if you want to compose captions from individual columns rather than using the full `synthesize` method.

---

## 9. Output Schema Reference

### Parquet schema

The Parquet schema is dynamically generated from the spec. You can inspect it:

```python
from chaksudb.export.query_builder import QueryBuilder

spec = ExportSpec(
    annotation_tasks=["grading"],
    disease_types=["DR"],
)

builder = QueryBuilder()
plan = builder.build_query(spec)
print(plan.render_sql())   # The generated SQL
print(plan.params)          # Parameterised values
```

### PyTorch sample format

Each sample returned by the dataset is a tuple:

```python
(image, annotations)
# image:       PIL.Image.Image  (or transformed Tensor if transform returns tensor)
# annotations: dict[str, Any]   with all non-path columns from the row
```

When **spatial** transforms are used, annotations also include loaded spatial data (see [§7.5 PyTorch sample format when using spatial](#75-pytorch-sample-format-when-using-spatial)):

- `_loaded_masks` — list of PIL mask images (same geometry as `image`)
- `_mask_meta` — list of mask metadata dicts
- `_bboxes`, `_keypoints`, `_circles` — coordinate lists in output pixel space

In the DataLoader, the **collate function** (selected by `collate_fn`) stacks images and batches annotations. With `collate_fn="default"`: images are stacked to `(B, C, H, W)` (same size when spatial is used); annotation dicts become `dict[str, list]`. Use `"padded"` or `"packed"` for variable-length masks/coords — see [§7.4 Collate strategies](#74-collate-strategies).

---

## 10. Performance & Best Practices

### Export to Parquet first

For repeated training runs, export once and use `ParquetDataset` on subsequent runs:

```python
# First run — export to Parquet
export(spec, parquet_path=Path("train.parquet"))

# Subsequent runs — load from file (no DB required)
from chaksudb.export.torch_dataset import create_dataloader
loader = create_dataloader(parquet_path=Path("train.parquet"), batch_size=32, shuffle=True)
```

### Batch size tuning

| Dataset size      | Recommended `parquet_batch_size` |
| ----------------- | ---------------------------------- |
| < 100k images     | 5,000 (default)                    |
| 100k – 1M images | 10,000                             |
| > 1M images       | 20,000                             |

### Use specific filters

Narrowing `dataset_names`, `split_names`, `modalities`, and `disease_types` reduces query time and output size.

### DataLoader workers

```python
# Single machine, local storage
num_workers=4

# Shared filesystem / cloud storage — start lower
num_workers=2
```

### Caching

`cache_rows=True` loads all query results into memory. Use only for small datasets (< 50k rows).

### Memory management

The Parquet export uses server-side cursors — memory is bounded by `parquet_batch_size`, not dataset size. For very large datasets, keep batch sizes reasonable and monitor with:

```python
import psutil
print(f"RSS: {psutil.Process().memory_info().rss / 1024**2:.0f} MB")
```

---

## 11. Debugging & Troubleshooting

### Inspect the generated SQL

```python
from chaksudb.export.query_builder import QueryBuilder
from chaksudb.export.spec import ExportSpec

spec = ExportSpec(dataset_names=["EYEPACS"], annotation_tasks=["grading"], disease_types=["DR"])
plan = QueryBuilder().build_query(spec)
print(plan.render_sql())
print(plan.params)
```

### Enable verbose logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Validate a spec before exporting

```python
from pydantic import ValidationError

try:
    spec = ExportSpec(
        disease_types=["DR"],  # Will fail — grading not in annotation_tasks
    )
except ValidationError as e:
    print(e)
```

### Verify a Parquet file

```python
import pyarrow.parquet as pq

table = pq.read_table("output.parquet")
print(f"Rows:    {table.num_rows}")
print(f"Columns: {table.column_names}")
print(f"Schema:\n{table.schema}")
print(f"Sample:\n{table.slice(0, 3).to_pandas()}")
```

### Common errors

| Error                                                    | Cause                                   | Fix                                                                                           |
| -------------------------------------------------------- | --------------------------------------- | --------------------------------------------------------------------------------------------- |
| `disease_types requires 'grading' in annotation_tasks` | Spec validation                         | Add `annotation_tasks=["grading"]`                                                          |
| Empty classification columns                           | No concepts/class names and nothing discovered | Set `classification_concepts=[...]` or `classification_class_names=[...]`, or use `build_dataset_spec` |
| `Image file not found`                                 | File path mismatch                      | Check `STORAGE_DATA_ROOT` / `STORAGE_LOCAL_ROOT` env vars, or use `base_path_for_paths` |
| `Database connection timeout`                          | DB not running or overloaded            | Ensure PostgreSQL is running (see `database_setup.md`); reduce batch size                    |
| `Memory error`                                         | Batch size too large                    | Lower `parquet_batch_size` or `batch_size`                                                |

### Validation rules summary

The spec enforces these constraints at construction time:

- `disease_types`, `grading_scale_name`, `grade_filter` → require `"grading"` in `annotation_tasks`
- `segmentation_types`, `lesion_subtypes` → require `"segmentation"` in `annotation_tasks`
- `localization_types` → require `"localization"` in `annotation_tasks`
- `classification_filter`, `classification_class_names`, `classification_concepts`,
  `classification_positive_for` → require `"classification"` in `annotation_tasks`
- `classification_class_names` is **optional**: when omitted, use `classification_concepts`
  or `build_dataset_spec` discovery to populate columns
- `multi_label_keys` → requires `classification_class_names` to be set
- `annotation_or_filters` tasks must be a subset of `annotation_tasks`

---

## Quick Reference Card

```python
from pathlib import Path
from chaksudb.export import ExportSpec, export
from chaksudb.export.transforms import Resize, RandomHorizontalFlip
import torchvision.transforms as T

# Minimal Parquet export
spec = ExportSpec(dataset_names=["EYEPACS"])
export(spec, parquet_path=Path("out.parquet"))

# DR grading DataLoader (with spatial resize + photometric)
spec = ExportSpec(
    dataset_names=["EYEPACS"],
    annotation_tasks=["grading"],
    disease_types=["DR"],
    require_annotations_mode="all",
)
loader = export(
    spec,
    torch="dataloader",
    spatial=[Resize((224, 224))],
    transform=T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    batch_size=32,
    shuffle=True,
)

# Segmentation with masks loaded and augmented (see §7)
spec = ExportSpec(
    annotation_tasks=["segmentation"],
    segmentation_types=["vessel"],
    require_annotations_mode="all",
)
dataset = export(spec, torch="dataset", spatial=[Resize(512), RandomHorizontalFlip(0.5)])
image, ann = dataset[0]
# ann["_loaded_masks"], ann["_mask_meta"] — loaded masks and metadata

# Parquet → DataLoader (write once, train many)
loader = export(spec, parquet_path=Path("data.parquet"), torch="dataloader", batch_size=64)
```
