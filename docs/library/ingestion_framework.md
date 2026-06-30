# Ingestion Framework Reference

A complete guide to `chaksudb/ingest/framework/` — the reusable utilities that power every dataset ingestion script.

---

## Table of Contents

1. [Overview](#1-overview)
2. [UUID Strategy](#2-uuid-strategy)
3. [Anatomy of an Ingestion Script](#3-anatomy-of-an-ingestion-script)
4. [File Processors Reference](#4-file-processors-reference)
5. [Task Processors Reference](#5-task-processors-reference)
6. [Provenance System](#6-provenance-system)
7. [Mask Converter Reference](#7-mask-converter-reference)
8. [Common Patterns](#8-common-patterns)
9. [Adding a New Dataset: Checklist](#9-adding-a-new-dataset-checklist)

---

## 1. Overview

```
Raw Dataset
    │
    ├── CSV / Excel / JSON / Text files
    │         │
    ├── Image folders / Mask folders
    │         │
    └── ingestion script  (chaksudb/ingest/scripts/ingest_NN_<name>.py)
              │
              ├── gen_uuid.py          ─ deterministic UUIDs
              ├── ingestion_helpers.py ─ process_csv / process_json / process_folder_tree
              ├── task_processors/     ─ process_disease_grade / process_segmentation_* / …
              ├── mask_converter/      ─ binary mask, contour, XML, soft map, …
              └── provenance_context   ─ get_current_provenance()
                        │
                        └── PostgreSQL (images, annotations, provenance tables)
```

Every ingestion script wires the framework utilities to a specific dataset's file layout. The framework handles boilerplate (file registration, provenance chains, error counting); the script provides only dataset-specific logic.

---

## 2. UUID Strategy

All primary keys are UUID v5 (deterministic). The same inputs always produce the same UUID, so ingestion is idempotent — running the same script twice produces no duplicates.

```python
from chaksudb.ingest.framework.gen_uuid import (
    generate_dataset_uuid,
    generate_image_uuid,
    generate_disease_grading_uuid,
    generate_patient_uuid,
    generate_segmentation_uuid,
    generate_localization_uuid,
    generate_classification_uuid,
    generate_quality_uuid,
    generate_dataset_split_uuid,
    generate_image_split_uuid,
)

# Namespace hierarchy
dataset_id = generate_dataset_uuid("EYEPACS")
# → UUID5(NAMESPACE_DNS, "EYEPACS")

image_id = generate_image_uuid(dataset_id, "10015_left")
# → UUID5(dataset_id, "10015_left")

grading_id = generate_disease_grading_uuid(
    image_id=image_id,
    disease_type="DR",
    scale_id=scale_id,
)
# → UUID5(image_id, "DR" + str(scale_id))
```

**Key rule**: if you need to look up a UUID you created earlier, just recompute it — the result is identical. Never query the database to find your own UUIDs.

---

## 3. Anatomy of an Ingestion Script

Every script follows these five phases:

```python
"""
Ingest script for EYEPACS dataset.

Dataset structure:
  01_EYEPACS/
    train.csv        # image,level
    train/           # JPEG images
"""

import asyncio
from pathlib import Path
from chaksudb.config.config import get_data_root
from chaksudb.db.models import Dataset, Image
from chaksudb.db.queries import upsert_dataset, upsert_image
from chaksudb.ingest.framework.gen_uuid import generate_dataset_uuid, generate_image_uuid
from chaksudb.ingest.framework.ingestion_helpers import process_csv
from chaksudb.ingest.framework.provenance_context import get_current_provenance
from chaksudb.ingest.framework.task_processors import process_disease_grade


async def ingest_eyepacs():
    # ─────────────────────────────────────────────────
    # Phase 1: Register dataset
    # ─────────────────────────────────────────────────
    dataset_id = generate_dataset_uuid("EYEPACS")
    await upsert_dataset(Dataset(
        dataset_id=dataset_id,
        dataset_name="EYEPACS",
        source_url="https://www.kaggle.com/c/diabetic-retinopathy-detection",
        license="competition",
        modality_types=["fundus"],
    ))

    # ─────────────────────────────────────────────────
    # Phase 2: Resolve data root
    # ─────────────────────────────────────────────────
    data_root = get_data_root() / "01_EYEPACS"

    # ─────────────────────────────────────────────────
    # Phase 3: Use process_csv to iterate rows
    # ─────────────────────────────────────────────────
    async def handle_row(row: dict, idx: int) -> None:
        # ─────────────────────────────────────────────
        # Phase 4: Create Image + call task processor
        # ─────────────────────────────────────────────
        image_id = generate_image_uuid(dataset_id, row["image"])
        await upsert_image(Image(
            image_id=image_id,
            dataset_id=dataset_id,
            original_image_id=row["image"],
            storage_provider="local",
            file_path=str(data_root / "train" / f"{row['image']}.jpeg"),
            file_format="jpeg",
            modality="fundus",
        ))

        # Provenance IDs from context variable — set automatically by process_csv
        raw_data_id, provenance_chain_id = get_current_provenance()

        await process_disease_grade(
            image_id=image_id,
            grade_value=int(row["level"]),
            disease_type="DR",
            scale_name="ICDR",
            raw_data_id=raw_data_id,
            provenance_chain_id=provenance_chain_id,
        )

    stats, _, _ = await process_csv(
        csv_path=data_root / "train.csv",
        dataset_id=dataset_id,
        unified_annotation_type="grading",
        process_row_fn=handle_row,
    )
    print(f"Ingested {stats.successful_items} images ({stats.failed_items} errors)")

    # ─────────────────────────────────────────────────
    # Phase 5: Register splits
    # ─────────────────────────────────────────────────
    from chaksudb.ingest.framework.gen_uuid import generate_dataset_split_uuid, generate_image_split_uuid
    from chaksudb.db.models import DatasetSplit, ImageSplit
    from chaksudb.db.queries import upsert_dataset_split, upsert_image_split

    split_id = generate_dataset_split_uuid(dataset_id, "train", "grading")
    await upsert_dataset_split(DatasetSplit(
        split_id=split_id,
        dataset_id=dataset_id,
        split_name="train",
        task_type="grading",
    ))
    # … assign images to split …


if __name__ == "__main__":
    asyncio.run(ingest_eyepacs())
```

---

## 4. File Processors Reference

All processors live in `chaksudb/ingest/framework/ingestion_helpers.py`. They handle file registration, provenance context, and error bookkeeping — your callback only does domain logic.

### `process_csv`

```python
async def process_csv(
    csv_path: Path,
    dataset_id: UUID,
    unified_annotation_type: str,   # "grading" | "segmentation" | "classification" | …
    process_row_fn: Callable[[dict, int], Awaitable[None]],
    skip_errors: bool = True,
) -> tuple[OperationStatistics, UUID, UUID]
# Returns (stats, raw_file_id, chain_id)
```

Reads the CSV, registers it as a `raw_annotation_files` row, creates a `provenance_chain`, sets provenance context, then calls `process_row_fn(row_dict, row_index)` for each row. Returns `OperationStatistics` (successful/failed counts + error list).

### `process_excel`

Same signature as `process_csv` plus `sheet_name: str | int = 0`. Use for `.xlsx` / `.xls` annotation files.

### `process_json`

```python
async def process_json(
    json_path: Path,
    dataset_id: UUID,
    unified_annotation_type: str,
    process_entry_fn: Callable[[Any, int], Awaitable[None]],
    skip_errors: bool = True,
) -> tuple[OperationStatistics, UUID, UUID]
```

Handles JSON arrays, JSON objects (iterates over items), and JSONL. The `entry` passed to the callback is `(index, value)` for arrays, `(index, (key, value))` for objects.

### `process_folder_tree`

```python
async def process_folder_tree(
    root_dir: Path,
    dataset_id: UUID,
    unified_annotation_type: str,
    process_file_fn: Callable[[Path, Path, int], Awaitable[None]],
    file_extensions: set[str] | None = None,   # e.g. {".png", ".tif"}
    recursive: bool = True,
    skip_errors: bool = True,
) -> OperationStatistics
```

Walks a directory tree. Each file gets its **own** `raw_annotation_files` row and provenance chain — suitable when annotation files are per-image (segmentation masks, XML annotations). The callback receives `(absolute_path, relative_path, depth)`.

### `process_text_file`

```python
async def process_text_file(
    text_path: Path,
    process_line_fn: Callable[[str, int], Awaitable[None]],
    skip_empty: bool = True,
    skip_comments: bool = True,
    comment_char: str = "#",
    skip_errors: bool = True,
) -> OperationStatistics
```

Iterates over non-empty, non-comment lines. Does **not** register a raw file — use `register_individual_file` manually if provenance is needed. Suitable for simple text manifests (e.g. `train.txt` with one filename per line).

---

## 5. Task Processors Reference

All processors live in `chaksudb/ingest/framework/task_processors/` and are importable from `internal.ingest.framework.task_processors`.

### Grading processor

```python
from chaksudb.ingest.framework.task_processors import process_disease_grade

await process_disease_grade(
    image_id=image_id,
    grade_value=2,              # raw grade (int or str accepted)
    disease_type="DR",          # "DR" | "DME" | "Glaucoma" | "AMD"
    scale_name="ICDR",          # grading scale name
    raw_data_id=raw_data_id,
    provenance_chain_id=provenance_chain_id,
)
```

| Function | Use when |
|---|---|
| `process_disease_grade()` | DR / DME / Glaucoma / AMD severity grades from any scale |
| `get_or_create_scale()` | Look up or register a grading scale by name + disease type |
| `check_scale_mapping_exists()` | Verify that cross-scale mappings are in place |

### Segmentation processor

```python
from chaksudb.ingest.framework.task_processors import (
    process_segmentation_from_binary_mask,
    process_segmentation_from_multiclass_mask,
    process_segmentation_from_contour,
    process_segmentation_from_xml,
    process_segmentation_from_soft_map,
    process_segmentation_from_layer_boundaries,
)

# Binary mask (PNG/GIF already in binary format — validated, no conversion)
await process_segmentation_from_binary_mask(
    mask_path=mask_file,
    annotation_type="vessel",
    image_id=image_id,
    raw_data_id=raw_data_id,
    provenance_chain_id=provenance_chain_id,
)

# Multi-class mask (extract one class by pixel value)
await process_segmentation_from_multiclass_mask(
    mask_path=mask_file,
    annotation_type="optic_disc",
    class_value=1,
    image_id=image_id,
    processed_dir=Path("processed/ORIGA/masks/optic_disc"),
    raw_data_id=raw_data_id,
    provenance_chain_id=provenance_chain_id,
)
```

| Function | Input | Saves to `processed/`? |
|---|---|---|
| `process_segmentation_from_binary_mask()` | Binary PNG/GIF | No — used as-is |
| `process_segmentation_from_multiclass_mask()` | Multi-class mask | Yes — extracted class |
| `process_segmentation_from_contour()` | Contour coordinates | Yes — rasterised PNG |
| `process_segmentation_from_xml()` | XML polygon annotations | Yes — rasterised PNG |
| `process_segmentation_from_soft_map()` | Float32 probability map | No — used as-is |
| `process_segmentation_from_layer_boundaries()` | Layer boundary arrays | No — used as-is |

### Localization processor

```python
from chaksudb.ingest.framework.task_processors import (
    process_localization_from_xml,
    process_localization_from_tsv,
    process_localization_from_json,
    process_localization_from_text_keypoint,
)

# Pascal VOC / ImageRet style XML
localizations = await process_localization_from_xml(
    xml_path=xml_file,
    image_id=image_id,
    raw_data_id=raw_data_id,
    provenance_chain_id=provenance_chain_id,
)

# Tab-separated bounding boxes
await process_localization_from_tsv(
    tsv_path=tsv_file,
    image_id=image_id,
    raw_data_id=raw_data_id,
    provenance_chain_id=provenance_chain_id,
)
```

| Function | Use when |
|---|---|
| `process_localization_from_xml()` | Pascal VOC or ImageRet XML format |
| `process_localization_from_tsv()` | Tab-separated bbox / keypoint files |
| `process_localization_from_json()` | JSON-encoded bbox or keypoint data |
| `process_localization_from_text_keypoint()` | Plain text `x y` keypoint files |

### Classification processor

```python
from chaksudb.ingest.framework.task_processors import process_classification

await process_classification(
    image_id=image_id,
    class_value=1,               # binary: bool/int(0/1)/str; multi_class: index/label; multi_label: dict
    task_type="binary",          # "binary" | "multi_class" | "multi_label"
    class_name="glaucoma",       # the label/class within the task
    task_name="glaucoma",        # the ML task (defaults to class_name); use one task_name per panel
    class_labels={1: "RG", 0: "NRG"},   # keep the *real* labels (recommended for binary)
    concept="glaucoma",          # canonical concept for cross-task filtering (auto-derived if omitted)
    raw_data_id=raw_data_id,
    provenance_chain_id=provenance_chain_id,
)
```

Every row is self-describing: `task_type`, `task_name`, `class_name`, `concept`,
`is_multilabel`, and non-null `class_index` / `class_label`. Multi-disease panels (RFMiD,
ODIR, …) use a single `task_name="disease_panel"` with `class_value` a dict of sub-keys. See
the [classification contract](schema_reference.md#classification_annotations).

### Quality processor

```python
from chaksudb.ingest.framework.task_processors import process_quality_annotation

await process_quality_annotation(
    image_id=image_id,
    quality_type="gradability",  # "overall" | "gradability" | "clarity" | "blur" | …
    quality_score=0.85,          # float, or None
    quality_label="good",        # str, or None
    raw_data_id=raw_data_id,
    provenance_chain_id=provenance_chain_id,
)
```

### Keyword processor

```python
from chaksudb.ingest.framework.task_processors import (
    process_keyword_annotation,
    process_keywords_batch,
    parse_keyword_string,
)

# Single keyword
await process_keyword_annotation(
    image_id=image_id,
    term="diabetic retinopathy",
    raw_data_id=raw_data_id,
    provenance_chain_id=provenance_chain_id,
)

# Batch (list of terms)
await process_keywords_batch(
    image_id=image_id,
    terms=["neovascularisation", "hemorrhage"],
    raw_data_id=raw_data_id,
    provenance_chain_id=provenance_chain_id,
)
```

---

## 6. Provenance System

The framework uses Python **context variables** to pass provenance IDs through the call stack without threading them through every function signature.

```
process_csv / process_excel / process_json / process_folder_tree
    │
    ├── register raw file     → raw_file_id
    ├── create provenance chain → chain_id
    ├── set_provenance_context(raw_file_id, chain_id)
    │
    └── calls your callback
              │
              └── get_current_provenance() → (raw_file_id, chain_id)
                        │
                        └── pass to task processor
```

Inside your callback:

```python
from chaksudb.ingest.framework.provenance_context import get_current_provenance

async def handle_row(row: dict, idx: int) -> None:
    raw_data_id, provenance_chain_id = get_current_provenance()

    await process_disease_grade(
        image_id=image_id,
        grade_value=int(row["level"]),
        disease_type="DR",
        scale_name="ICDR",
        raw_data_id=raw_data_id,
        provenance_chain_id=provenance_chain_id,
    )
```

**You never call `set_provenance_context` directly.** The `ingestion_helpers` functions call it before your callback and reset it after. Context is set per-row (for CSV/JSON) or per-file (for `process_folder_tree`).

---

## 7. Mask Converter Reference

`chaksudb/ingest/framework/mask_converter/` provides utilities for handling the full variety of segmentation mask formats found in ophthalmic datasets.

| Function | Input format | Use when |
|---|---|---|
| `validate_binary_mask(path)` | PNG / GIF binary mask | Confirm mask is valid before storing as-is |
| `extract_class_from_mask(path, class_value)` | Multi-class PNG (pixel values = class IDs) | Extract one class from a colour/multi-class mask; save to `processed/` |
| `extract_classes_from_multiclass_mask(path)` | Multi-class PNG | Extract all classes at once |
| `convert_contour_to_binary_mask(contours, width, height)` | List of `(x, y)` coordinate pairs | Rasterise contour annotations (e.g. Drishti-GS1 boundaries); save to `processed/` |
| `parse_xml_annotations(xml_path)` | Pascal VOC / ImageRet XML | Parse XML polygon annotations |
| `parse_xml_polygon_to_binary_mask(xml_path, width, height)` | XML polygon | Convert XML polygon to binary PNG; save to `processed/` |
| `load_soft_map(path)` | Float32 / NumPy probability map | Load soft attention / saliency maps (stored as-is) |
| `load_layer_boundaries(path)` | MAT / NumPy layer boundary arrays | OCT layer segmentation boundaries (stored as-is) |

**HEI-MED specific:**

| Function | Use when |
|---|---|
| `load_exudate_map_gz(path)` | HEI-MED gzip float32 exudate probability map |
| `parse_gnd_blob_count(path)` | HEI-MED `.GND` file (blob count + type labels) |
| `parse_meta_file(path)` | HEI-MED tilde-delimited patient metadata |

**When to save to `processed/`:**

| Scenario | Saved? |
|---|---|
| Binary mask validation only | No — original path stored in `mask_file_path` |
| Extract class from multi-class | Yes — extracted PNG written to `processed/<dataset>/masks/<type>/` |
| Contour → binary mask | Yes — rasterised PNG |
| XML polygon → binary mask | Yes — rasterised PNG |
| Soft map / layer boundary | No — original path stored |

---

## 8. Common Patterns

### Pattern A: CSV labels + flat image folder

Typical for EYEPACS, APTOS, OIA-DDR, DeepDRiD.

```python
from chaksudb.ingest.framework.ingestion_helpers import process_csv
from chaksudb.ingest.framework.provenance_context import get_current_provenance
from chaksudb.ingest.framework.gen_uuid import generate_dataset_uuid, generate_image_uuid
from chaksudb.ingest.framework.task_processors import process_disease_grade
from chaksudb.db.models import Image
from chaksudb.db.queries import upsert_image

data_root = get_data_root() / "01_EYEPACS"

async def handle_row(row: dict, idx: int) -> None:
    image_id = generate_image_uuid(dataset_id, row["image"])
    await upsert_image(Image(
        image_id=image_id,
        dataset_id=dataset_id,
        original_image_id=row["image"],
        storage_provider="local",
        file_path=str(data_root / "train" / f"{row['image']}.jpeg"),
        file_format="jpeg",
        modality="fundus",
    ))
    raw_data_id, provenance_chain_id = get_current_provenance()
    await process_disease_grade(
        image_id=image_id,
        grade_value=int(row["level"]),
        disease_type="DR",
        scale_name="ICDR",
        raw_data_id=raw_data_id,
        provenance_chain_id=provenance_chain_id,
    )

stats, _, _ = await process_csv(
    csv_path=data_root / "train.csv",
    dataset_id=dataset_id,
    unified_annotation_type="grading",
    process_row_fn=handle_row,
)
```

### Pattern B: Per-image XML annotation files

Typical for Pascal VOC localization, ImageRet lesion polygons, Drishti contours.

```python
from chaksudb.ingest.framework.ingestion_helpers import process_folder_tree
from chaksudb.ingest.framework.provenance_context import get_current_provenance
from chaksudb.ingest.framework.task_processors import process_localization_from_xml

async def handle_xml(file_path: Path, rel_path: Path, depth: int) -> None:
    # Derive image_id from filename stem
    image_id = generate_image_uuid(dataset_id, file_path.stem)

    raw_data_id, provenance_chain_id = get_current_provenance()
    await process_localization_from_xml(
        xml_path=file_path,
        image_id=image_id,
        raw_data_id=raw_data_id,
        provenance_chain_id=provenance_chain_id,
    )

stats = await process_folder_tree(
    root_dir=data_root / "Annotations",
    dataset_id=dataset_id,
    unified_annotation_type="localization",
    process_file_fn=handle_xml,
    file_extensions={".xml"},
)
```

### Pattern C: Hierarchical disease/patient/eye folder structure

Typical for ODIR-5K, FUND-OCT.

```python
from chaksudb.ingest.framework.ingestion_helpers import process_folder_tree
from chaksudb.ingest.framework.gen_uuid import generate_patient_uuid

async def handle_image(file_path: Path, rel_path: Path, depth: int) -> None:
    # rel_path = "Normal/patient001/left/image.jpg"  (depth == 3)
    parts = rel_path.parts
    disease_dir, patient_dir, eye_dir = parts[0], parts[1], parts[2]

    patient_id = generate_patient_uuid(dataset_id, patient_dir)
    image_id = generate_image_uuid(dataset_id, file_path.stem)

    laterality = "left" if "left" in eye_dir.lower() else "right"
    await upsert_image(Image(
        image_id=image_id,
        dataset_id=dataset_id,
        original_image_id=file_path.stem,
        storage_provider="local",
        file_path=str(file_path),
        file_format=file_path.suffix.lstrip("."),
        modality="fundus",
        eye_laterality=laterality,
    ))

stats = await process_folder_tree(
    root_dir=data_root / "Images",
    dataset_id=dataset_id,
    unified_annotation_type="classification",
    process_file_fn=handle_image,
    file_extensions={".jpg", ".jpeg", ".png"},
)
```

### Pattern D: Pre-defined split folders with per-image masks

Typical for IDRID, DRIVE, STARE.

```python
for split_name, image_dir, mask_dir in [
    ("train", data_root / "Training_Images", data_root / "Training_Masks"),
    ("test",  data_root / "Testing_Images",  data_root / "Testing_Masks"),
]:
    # 1. Register images from the split folder
    for img_path in sorted(image_dir.glob("*.jpg")):
        image_id = generate_image_uuid(dataset_id, img_path.stem)
        await upsert_image(Image(
            image_id=image_id,
            dataset_id=dataset_id,
            original_image_id=img_path.stem,
            storage_provider="local",
            file_path=str(img_path),
            file_format="jpg",
            modality="fundus",
        ))

        # 2. Register split assignment
        split_id = generate_dataset_split_uuid(dataset_id, split_name, "segmentation")
        image_split_id = generate_image_split_uuid(image_id, split_id)
        await upsert_image_split(ImageSplit(
            image_split_id=image_split_id,
            image_id=image_id,
            split_id=split_id,
        ))

    # 3. Walk the mask folder with per-file provenance
    async def handle_mask(file_path: Path, rel_path: Path, depth: int) -> None:
        image_id = generate_image_uuid(dataset_id, file_path.stem.replace("_MA", ""))
        raw_data_id, provenance_chain_id = get_current_provenance()
        await process_segmentation_from_binary_mask(
            mask_path=file_path,
            annotation_type="lesion",
            lesion_subtype="microaneurysm",
            image_id=image_id,
            raw_data_id=raw_data_id,
            provenance_chain_id=provenance_chain_id,
        )

    await process_folder_tree(
        root_dir=mask_dir / "Microaneurysms",
        dataset_id=dataset_id,
        unified_annotation_type="segmentation",
        process_file_fn=handle_mask,
        file_extensions={".tif"},
    )
```

---

## 9. Adding a New Dataset: Checklist

1. **Document the dataset** — create `docs/data/NN_<DatasetName>.md` using an existing doc as a template. Include: source URL, license, size, annotation types, folder structure, and any known quirks.

2. **Determine the dataset number** — check existing scripts in `chaksudb/ingest/scripts/` and use the next available two-digit prefix.

3. **Create the script** — `chaksudb/ingest/scripts/ingest_NN_<dataset_name>.py`. Start from the anatomy in [§3](#3-anatomy-of-an-ingestion-script).

4. **Register the dataset** — call `upsert_dataset` with a `generate_dataset_uuid` UUID at the top of the script.

5. **Resolve the data root** — use `get_data_root() / "NN_DatasetName"` (from `internal.config.config`). Never hardcode absolute paths.

6. **Pick the right file processor** — CSV/Excel → `process_csv` / `process_excel`; JSON → `process_json`; per-image files → `process_folder_tree`; text manifests → `process_text_file`.

7. **Pick the right task processors** — use the table in [§5](#5-task-processors-reference). Pass `raw_data_id` and `provenance_chain_id` from `get_current_provenance()` into every processor call.

8. **Handle masks** — if the dataset has segmentation annotations, check the mask format against the [Mask Converter table](#7-mask-converter-reference). Multi-class or contour masks need a `processed_dir` argument to control where extracted PNGs are written.

9. **Register splits** — call `upsert_dataset_split` for each split (`train`, `val`, `test`) and `upsert_image_split` to assign images. Use `generate_dataset_split_uuid` and `generate_image_split_uuid`.

10. **Test idempotency** — run the script twice and verify the second run logs zero new inserts.
