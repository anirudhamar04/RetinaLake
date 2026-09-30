# Development

## Project structure

```
chaksudb/
├── data/                        # Dataset files (raw + processed roots, via .env)
├── docs/
│   ├── data/                    # Per-dataset documentation + dataset catalogue
│   └── library/                 # Technical library guides (this folder)
├── chaksudb/
│   ├── common/                  # Shared utilities
│   ├── config/                  # Configuration management
│   ├── db/                      # Database layer (connection.py, models.py, queries/)
│   ├── ingest/
│   │   ├── scripts/             # Dataset-specific ingestion scripts
│   │   └── framework/           # Reusable ingestion utilities + task processors
│   ├── export/                  # Export pipeline (spec, api, query_builder, modules/, …)
│   └── storage/                 # Storage abstractions (local, S3, GCS, Azure, HTTP)
├── schema/
│   ├── schema.sql               # PostgreSQL schema (grade-conversion triggers bundled in)
│   └── triggers/                # Reference copies of the triggers (already in schema.sql)
├── scripts/                     # setup_full_database, export_builder (GUI), find_duplicate_images, …
├── docker-compose.yml           # PostgreSQL + automatic schema init
├── examples/                    # Runnable export examples
├── tests/                       # Test suite
└── pyproject.toml
```

See [Architecture](../architecture.md) for a staged route through the codebase.

## Design principles

1. **Layered ingestion framework** — reusable utilities (framework), dataset-structure adapters
   for common patterns, and dataset-specific code only when necessary.
2. **Idempotent operations** — re-running ingestion doesn't create duplicates; deterministic
   UUIDs from content hashes; upserts with conflict resolution.
3. **Provenance tracking** — every annotation links to its source file; transformations are
   recorded; raw files are hashed and versioned.
4. **Type safety** — Pydantic models for all data structures; PostgreSQL constraints for integrity.

## Testing

```bash
# All tests
uv run pytest -v

# Skip DB-dependent tests (no running PostgreSQL required)
uv run pytest tests/ -m "not requires_db" -v

# Only DB-dependent tests (requires a running PostgreSQL instance)
uv run pytest -m requires_db -v
```

Tests marked `@pytest.mark.requires_db` require a running PostgreSQL instance (e.g. the one from
`docker compose up`). All other tests run without a database.

## Adding a new dataset

1. **Document the dataset** in `docs/data/NN_<DatasetName>.md` and add a row to
   [`docs/data/README.md`](../data/README.md).
2. **Create an ingestion script** in `chaksudb/ingest/scripts/ingest_NN_<name>.py`.
3. **Use `process_csv` / `process_folder_tree`** from `ingestion_helpers` — never loop manually.
4. **Call task processors** for all annotation types; pass `get_current_provenance()` IDs.
5. **Register splits** with `upsert_dataset_split` / `upsert_image_split`.

See [Ingestion Framework](ingestion_framework.md) for the full step-by-step checklist, task
processor API, mask converter guide, and common patterns.
