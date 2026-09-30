# Database Setup Guide

This guide covers creating the ChaksuDB PostgreSQL database, applying the schema, and
verifying that the grade-conversion triggers are installed.

ChaksuDB connects to a PostgreSQL server you provide. Any PostgreSQL **14+** instance works —
a local install, a managed service, or a container you run yourself. Connection settings are
read from environment variables (see `.env.example`):

```bash
DB_HOST=127.0.0.1
DB_PORT=5432
DB_DATABASE=chaksudb
DB_USER=chaksuai      # any role that can create tables/functions
DB_PASSWORD=...       # keep this in .env, never commit it
```

---

## Quick Start (fresh database)

The grade-conversion/label triggers are **bundled at the end of `schema/schema.sql`**, so a
single `psql -f` installs the tables *and* the triggers:

```bash
# 1. Create the database (uses your $DB_USER / $DB_DATABASE)
createdb chaksudb

# 2. Apply the schema + triggers
psql -d chaksudb -f schema/schema.sql

# 3. Bootstrap grading-scale mappings (run once)
uv run python scripts/bootstrap_grading_scales.py

# Done — the database is ready to ingest datasets.
```

If your server needs an explicit host/user, pass them to `createdb`/`psql`:

```bash
createdb -h 127.0.0.1 -U chaksuai chaksudb
psql -h 127.0.0.1 -U chaksuai -d chaksudb -f schema/schema.sql
```

> The full pipeline (`uv run python scripts/setup_full_database.py`) assumes the database
> already exists and the schema has been applied — run the two steps above first.

---

## What `schema/schema.sql` installs

Applying `schema/schema.sql` creates:

1. All tables (core, annotations, vocabularies, provenance, splits).
2. The three triggers folded in at the end of the file:
   - **`set_label_from_original`** — sets `grade_label` from the scale's `value_labels` map.
   - **`trigger_auto_convert_disease_grade`** — populates `disease_grading.scaled_grade`
     (identity for the canonical scale `ICDR_0_4`; otherwise mapped via
     `grading_scale_mappings`) and emits the `grade_conversion` NOTIFY.
   - **`trigger_backfill_grades_on_new_mapping`** — backfills NULL `scaled_grade` rows when a
     new mapping is inserted.

Without these triggers, `scaled_grade` (and therefore the `*_grade` export columns and
`health_status`) stays NULL. The standalone copies under `schema/triggers/*.sql` are kept for
reference only — you do **not** need to apply them separately.

The test suite applies `schema/schema.sql` automatically (`tests/conftest.py`), so DB-backed
grading/health tests get the triggers for free.

---

## Verifying the triggers

```bash
# Functions exist
psql -d chaksudb -c "\df trigger_auto_convert_disease_grade"

# Triggers attached to disease_grading
psql -d chaksudb -c "
SELECT tgname, tgenabled
FROM pg_trigger
WHERE tgrelid = 'disease_grading'::regclass AND NOT tgisinternal;"
```

Expected (the `O` means enabled):

```
              tgname                | tgenabled
------------------------------------+-----------
 set_label_from_original            | O
 trigger_auto_convert_disease_grade | O
```

---

## Resetting the database

`schema/reset_db.sql` drops and recreates the schema. To wipe everything and start over:

```bash
psql -d chaksudb -f schema/reset_db.sql
psql -d chaksudb -f schema/schema.sql
uv run python scripts/bootstrap_grading_scales.py
```

**Warning:** this deletes all data.

---

## Troubleshooting

### `scaled_grade` stays NULL after inserting grades

The triggers are missing or the scale mappings haven't been bootstrapped.

```bash
# Confirm the trigger is attached (see "Verifying the triggers" above).
# If absent, re-apply the schema (idempotent for functions/triggers):
psql -d chaksudb -f schema/schema.sql

# Make sure the scale mappings are loaded:
uv run python scripts/bootstrap_grading_scales.py
```

### `permission denied to create function`

Connect as a role that owns the database (or a superuser). The role in your `.env` must be
able to `CREATE FUNCTION` / `CREATE TRIGGER`.

### Restoring from a `pg_dump` backup

A schema-only or data-only dump may not recreate the triggers. After restoring, re-apply the
schema file (or just the trigger section) and re-run the bootstrap.

---

## Next steps

1. ✅ Database created and schema (+ triggers) applied.
2. ✅ Grading scales bootstrapped.
3. 🚀 Obtain the raw datasets and run ingestion — see the project
   [README](../../README.md) and [`ingestion_framework.md`](ingestion_framework.md).

See also:
- [`grading_scale_normalization.md`](grading_scale_normalization.md) — the grading/normalization system.
- [`schema_reference.md`](schema_reference.md) — full table reference.
