# RetinaLake Documentation

RetinaLake (the `chaksudb` library) is a unified PostgreSQL data lake and export toolkit for 54
ophthalmic imaging datasets. Start with the project [README](../README.md) for the overview and
Docker quick start, then use the guides below.

## Getting started

- [**Database setup**](library/database_setup.md) — the database, schema, and triggers (Docker or manual).
- [**Architecture**](architecture.md) — how the pieces fit together and a staged route through the codebase.
- [**Datasets**](data/README.md) — the 54-dataset catalogue, sources, and the expected on-disk layout.

## Library guides

| Guide | What it covers |
|-------|----------------|
| [Schema reference](library/schema_reference.md) | Every table, the UUID-v5 design, storage locators, the self-describing classification contract |
| [Storage architecture](library/storage_architecture.md) | `data/` vs `processed/`, and when masks are converted |
| [Ingestion framework](library/ingestion_framework.md) | Writing a dataset adapter: file processors, task processors, provenance, mask conversion |
| [Grading-scale normalization](library/grading_scale_normalization.md) | Cross-scale grade mapping, `scaled_grade`, the DB triggers |
| [Export data guide](library/export_data_guide.md) | `ExportSpec`, output formats, every annotation task, transforms, end-to-end examples |
| [Transforms](library/transforms.md) | Spatial + photometric transform pipeline that keeps masks/boxes/keypoints consistent |
| [IQA & ROI detection](library/iqa_roi_detection.md) | AutoMorph image-quality scoring and fundus ROI circles, and using them at export time |
| [Connection-pool configuration](library/connection_pool_configuration.md) | Tuning `DB_MIN_CONNECTIONS` / `DB_MAX_CONNECTIONS` |
| [Development](library/development.md) | Project structure, design principles, testing, adding a dataset |

## Per-dataset notes

See [`docs/data/`](data/README.md) for the dataset catalogue and per-dataset documentation
(layout, annotation formats, quirks).

## Examples

Runnable scripts live in [`examples/`](../examples/) — see its
[README](../examples/README.md) for the recommended reading order.
