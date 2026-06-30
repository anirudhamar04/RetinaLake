# Examples

Runnable scripts that exercise the ChaksuDB **export** API. Each writes its output under
`examples/export_output/` (gitignored).

## Prerequisites

These examples query the database, so you need:

1. A running PostgreSQL instance with the schema applied and **datasets ingested** — see
   [`docs/library/database_setup.md`](../docs/library/database_setup.md). Examples reference
   specific datasets (IDRID, REFUGE, …); trim the dataset lists to what you actually have.
2. Your connection settings in `.env` (copy from `.env.example`).

Run any example from the repo root:

```bash
uv run python examples/export_example.py
```

The `export()` entry point manages the connection pool for you. The few scripts that call
lower-level APIs (e.g. `export_normal_roi_iqa.py`, `export_classification_concepts.py`) open the
pool explicitly with `await init_pool()` and run under `asyncio`.

The one exception is [`print_export_query.py`](print_export_query.py), which only builds the SQL
string and needs **no database**.

## Where to start

1. [`print_export_query.py`](print_export_query.py) — print the SQL a spec generates (no DB). Best first read.
2. [`export_example.py`](export_example.py) — the guided tour: Parquet, PyTorch Dataset/DataLoader, transforms.
3. [`export_dataset_flat.py`](export_dataset_flat.py) — discover & flatten everything one dataset has (`build_dataset_spec`).

## Index

| Example | Demonstrates |
|---------|--------------|
| [`print_export_query.py`](print_export_query.py) | Print the assembled SQL for a spec (no DB) |
| [`export_example.py`](export_example.py) | End-to-end tour of Parquet + PyTorch Dataset/DataLoader + transforms |
| [`export_with_presets.py`](export_with_presets.py) | One-line presets (`dr_classification`, `glaucoma_detection`, …) |
| [`export_dataset_flat.py`](export_dataset_flat.py) | `build_dataset_spec` — flatten a dataset's full label set |
| [`export_classification_concepts.py`](export_classification_concepts.py) | Concept-centric classification (`classification_concepts`) unified across storage shapes |
| [`export_classification_pivoted.py`](export_classification_pivoted.py) | Exact per-task classification pivots (binary / multi-class / multi-label) |
| [`dr_grade_filtering.py`](dr_grade_filtering.py) | Filter by disease grade (`grade_filter`) |
| [`export_seg_parquets.py`](export_seg_parquets.py) | Segmentation exports (OD/OC, A/V, lesions) per split |
| [`export_coco_detection.py`](export_coco_detection.py) | Object detection in COCO format (`coco_path`) |
| [`export_with_transforms.py`](export_with_transforms.py) | Spatial + photometric transforms on a PyTorch dataset |
| [`export_normal_roi_iqa.py`](export_normal_roi_iqa.py) | `health_status` + IQA quality filter + fundus ROI columns |
| [`export_keyword.py`](export_keyword.py) | Clinical keywords + free-text descriptions |
| [`export_fundus_descriptions.py`](export_fundus_descriptions.py) | Generated captions for VLM training (`caption_mode`) |
| [`ssl_export.py`](ssl_export.py) | Image-only export for self-supervised pretraining (`output_format="ssl"`) |
| [`export_multidisease_zeroshot_baseline.py`](export_multidisease_zeroshot_baseline.py) | Multi-dataset zero-shot evaluation parquet |

For the full export reference see [`docs/library/export_data_guide.md`](../docs/library/export_data_guide.md).
