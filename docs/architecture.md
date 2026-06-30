# Architecture & Codebase Tour

This page explains how ChaksuDB fits together and gives a staged route through the code for new
contributors. For the table-by-table contract see [`library/schema_reference.md`](library/schema_reference.md);
for the export API see [`library/export_data_guide.md`](library/export_data_guide.md).

## The mental model

ChaksuDB takes dozens of heterogeneous datasets and lands them in **one PostgreSQL schema**, so
a single query can pull labels across all of them. Everything flows one direction:

```
Raw datasets ──▶ ingest/scripts ──▶ PostgreSQL ──▶ export ──▶ Parquet / PyTorch
                  (uses framework)       ▲
                                  run_roi_iqa.py
                              (IQA scores + ROI circles)
```

Three ideas hold it together:

- **Deterministic UUID v5** — every primary key is derived from content (e.g. dataset name +
  source identifier), so re-running ingestion is idempotent: duplicates are no-ops.
- **Self-describing rows** — a classification row carries both its ML *task* and a canonical
  *concept*, so a disease (e.g. `glaucoma`) is retrievable as one unified binary regardless of
  how a dataset stored it. Concepts live in `chaksudb/ingest/framework/concepts.py`.
- **Composable export** — an `ExportSpec` describes *what you want*; `query_builder` assembles
  SQL from pluggable modules; `api.export()` writes Parquet or builds a PyTorch dataset.

## Major components

| Area | Path | Role |
|------|------|------|
| Schema | `schema/schema.sql` | All tables + the grade-conversion triggers (one `psql -f` installs everything) |
| Ingest framework | `chaksudb/ingest/framework/` | Reusable engine: UUIDs, hashing, file/task processors, mask conversion, provenance |
| Ingest adapters | `chaksudb/ingest/scripts/ingest_NN_*.py` | One per dataset — wires the framework to that dataset's layout |
| IQA / ROI | `chaksudb/ingest/scripts/run_roi_iqa.py` | AutoMorph quality score + fundus ROI circle as pseudo-annotations |
| Export | `chaksudb/export/` | `ExportSpec` → `query_builder` + `modules/` → Parquet / `torch_dataset` |
| Transforms | `chaksudb/export/transforms/` | Spatial + photometric pipeline that keeps masks/boxes/keypoints consistent |
| Storage | `chaksudb/storage/` | Local / S3 / GCS / Azure / HTTP locators |

## A staged route through the code

A focused order that builds understanding without reading every file:

0. **Orientation** — this page + the project README's data-flow.
1. **The schema is the spine** — read `schema/schema.sql` slowly; it defines every contract.
2. **Data models** — `chaksudb/db/models/` (the Python mirror of the schema).
3. **DB access layer** — `chaksudb/db/` (small, mechanical query helpers).
4. **Ingest framework** — `chaksudb/ingest/framework/`, leaves first (gen_uuid, hashing, then
   the task processors).
5. **One ingest script end-to-end** — e.g. `chaksudb/ingest/scripts/ingest_27_brset.py`
   (rich: grading + classification panel + per-type quality + anatomy + patient). Trace it
   line by line, stepping into the framework calls.
6. **Export** — in order: `spec.py` → `query_builder.py` → `modules/` → `api.py` →
   `torch_dataset.py`.
7. **Transforms** — `chaksudb/export/transforms/` (only if you do training-time augmentation).
8. **Storage** — `chaksudb/storage/` (brief).

## The single best debugging trick

Print the SQL a spec generates and read it — it makes every export module concrete:

```python
from chaksudb.export.spec import ExportSpec
from chaksudb.export.query_builder import QueryBuilder

sql = QueryBuilder().build_query(ExportSpec(
    annotation_tasks=["grading"],
    disease_types=["DR"],
)).render_sql()
print(sql)
```

See `examples/print_export_query.py` for a ready-to-run version.

## Verification loop

Don't just read — run. Apply the schema to a scratch DB, ingest one dataset, then export it and
inspect the Parquet. `uv run pytest -v` exercises the framework and export modules against a
test schema.
