# Grading Scale Normalization System

This document describes the grading scale normalization system that enables automatic conversion between different DR (and other disease) grading scales.

## Overview

Different datasets use different grading scales for disease severity:
- **ICDR (International Clinical DR Severity Scale)**: 0-4 or 0-5
- **AAO (American Academy of Ophthalmology)**: 0-5
- **Scottish DR Protocol**: 0-5
- **EYEPACS**: Custom 0-4 scale
- Many dataset-specific scales

The normalization system provides:
1. **Automatic registration** of unknown scales
2. **Learned mappings** from SUSTech-SYSU dataset
3. **Database triggers** for automatic conversion on insert/update
4. **Backfill triggers** to update historical data when new mappings are learned

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Dataset Ingestion Script                      │
│                                                                   │
│  process_disease_grade(grade_value=2, scale_name="EYEPACS")     │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Grading Processor                             │
│                                                                   │
│  1. Validates disease_type and grade_value                       │
│  2. Gets or creates scale (auto-registers if unknown)           │
│  3. Returns DiseaseGrading model with original_grade             │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Database (upsert)                             │
│                                                                   │
│  INSERT INTO disease_grading (original_grade="2", ...)          │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│              Database Trigger: auto_convert_disease_grade        │
│                                                                   │
│  1. Check if scale has mapping to ICDR_0_4                       │
│  2. If mapping exists, set scaled_grade                          │
│  3. If no mapping, leave scaled_grade = NULL                     │
│  4. Set updated_at timestamp                                     │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Stored in Database                            │
│                                                                   │
│  original_grade = "2"                                            │
│  scaled_grade = 2         (if mapping exists)                    │
│  scaled_grade = NULL      (if no mapping)                        │
└─────────────────────────────────────────────────────────────────┘
```

## Setup

### 1. Bootstrap Scale Mappings

Run this **once** to learn mappings from SUSTech-SYSU dataset:

```bash
uv run scripts/bootstrap_grading_scales.py
```

This script:
- Registers known scales: ICDR_0_4, ICDR_0_5, AAO, Scottish
- Analyzes `drLabels.csv` to learn mappings between scales
- Handles `c5_DR_reclassified.csv` for grade 5 cases
- Stores validated mappings in `grading_scale_mappings` table

**Output:**
```
✅ Bootstrap completed successfully!
   Total mappings: 24
   Exact: 20
   Approximate: 3
   Manual review: 1
```

### 2. Database Triggers

The automatic conversion triggers are **bundled in `schema/schema.sql`** — applying the schema
installs them (no separate step). See [`database_setup.md`](database_setup.md).

The relevant triggers:
- `trigger_auto_convert_disease_grade` — converts grades on INSERT/UPDATE (identity for the
  canonical scale `ICDR_0_4`, otherwise mapped via `grading_scale_mappings`).
- `trigger_backfill_grades_on_new_mapping` — backfills historical rows when new mappings are added.

## Usage in Dataset Scripts

### Basic Usage

```python
from chaksudb.ingest.framework.task_processors import process_disease_grade
from chaksudb.db.queries.grading import upsert_disease_grading

# Process a grade from your dataset
grading = await process_disease_grade(
    grade_value=2,                    # Grade value (int, float, or str)
    disease_type="DR",                # Disease type
    scale_name="EYEPACS_0_4",         # Your dataset's scale name
    image_id=image_id,                # Image UUID
    
    # Optional scale metadata (for auto-registration)
    scale_description="EYEPACS 5-level DR grading",
    min_value=0,
    max_value=4,
    value_labels={
        "0": "No DR",
        "1": "Mild NPDR",
        "2": "Moderate NPDR",
        "3": "Severe NPDR",
        "4": "PDR"
    },
    
    # Optional annotation metadata
    raw_data_id=raw_file_id,
    annotation_method="manual",
    confidence_score=0.95,
)

# Store in database
await upsert_disease_grading(grading)
```

### With Unknown Scale

If your dataset uses an unknown scale, it will be auto-registered:

```python
# First grade from new dataset
grading = await process_disease_grade(
    grade_value="Moderate",           # Can use string labels
    disease_type="DR",
    scale_name="MyDataset_Custom",    # Unknown scale
    image_id=image_id,
    scale_description="Custom 3-level scale",
    value_labels={
        "None": "No DR",
        "Mild": "Mild DR",
        "Moderate": "Moderate or worse"
    },
)

# Warning logged:
# "Auto-registering unknown scale 'MyDataset_Custom' for DR.
#  No conversion mappings exist yet. Add mappings to enable normalization."

# Stored in database with:
# - original_grade = "Moderate"
# - scaled_grade = NULL (no mapping yet)
```

Later, when you add a mapping:

```python
from chaksudb.db.models import GradingScaleMapping
from chaksudb.db.queries.grading import upsert_grading_scale_mapping
from chaksudb.ingest.framework.gen_uuid import generate_grading_scale_mapping_uuid

# Create mapping
source_scale_id = generate_grading_scale_uuid("MyDataset_Custom", "DR")
target_scale_id = generate_grading_scale_uuid("ICDR_0_4", "DR")

mapping_id = generate_grading_scale_mapping_uuid(
    source_scale_id, target_scale_id, "Moderate"
)

mapping = GradingScaleMapping(
    mapping_id=mapping_id,
    source_scale_id=source_scale_id,
    target_scale_id=target_scale_id,
    source_value="Moderate",
    target_value=2,
    mapping_confidence="exact",
)

await upsert_grading_scale_mapping(mapping)

# Backfill trigger automatically updates all historical records
# with scaled_grade = 2
```

## Database Tables

### grading_scales

Stores scale definitions:

```sql
CREATE TABLE grading_scales (
  scale_id UUID PRIMARY KEY,
  scale_name TEXT NOT NULL,
  disease_type TEXT NOT NULL,
  scale_description TEXT,
  min_value INTEGER,
  max_value INTEGER,
  value_labels JSONB
);
```

### grading_scale_mappings

Stores mappings between scales:

```sql
CREATE TABLE grading_scale_mappings (
  mapping_id UUID PRIMARY KEY,
  source_scale_id UUID NOT NULL,
  target_scale_id UUID NOT NULL,
  source_value TEXT NOT NULL,
  target_value INTEGER,
  mapping_confidence TEXT NOT NULL DEFAULT 'exact'
);
```

### disease_grading

Stores actual grades:

```sql
CREATE TABLE disease_grading (
  grading_id UUID PRIMARY KEY,
  image_id UUID NOT NULL,
  disease_type TEXT NOT NULL,
  scale_id UUID NOT NULL,
  original_grade TEXT,        -- Original value from dataset
  scaled_grade INTEGER,       -- Converted to ICDR_0_4 (or NULL)
  grade_label TEXT,
  -- ... other fields
);
```

## Triggers

### auto_convert_disease_grade

**When:** BEFORE INSERT OR UPDATE on `disease_grading`

**What:**
1. Looks up target scale (ICDR_0_4) for the disease_type
2. If incoming scale is already target scale, sets `scaled_grade = original_grade`
3. Otherwise, looks up mapping in `grading_scale_mappings`
4. Sets `scaled_grade` to mapped value (or NULL if no mapping)
5. Updates `updated_at` timestamp

**Example:**
```sql
INSERT INTO disease_grading (
  grading_id, image_id, disease_type, scale_id,
  original_grade, scaled_grade, ...
) VALUES (
  uuid, uuid, 'DR', eyepacs_scale_id,
  '2', NULL, ...  -- scaled_grade is NULL on insert
);

-- Trigger runs:
-- - Finds mapping: EYEPACS:2 -> ICDR_0_4:2
-- - Sets scaled_grade = 2 before insert
```

### backfill_grades_on_new_mapping

**When:** AFTER INSERT on `grading_scale_mappings`

**What:**
1. Finds all `disease_grading` records with:
   - `scale_id = NEW.source_scale_id`
   - `original_grade = NEW.source_value`
   - `scaled_grade IS NULL` (no previous conversion)
2. Updates their `scaled_grade` to `NEW.target_value`
3. Updates `updated_at` timestamp

**Example:**
```sql
-- Insert new mapping
INSERT INTO grading_scale_mappings (
  mapping_id, source_scale_id, target_scale_id,
  source_value, target_value, mapping_confidence
) VALUES (
  uuid, mydataset_scale_id, icdr_scale_id,
  'Moderate', 2, 'exact'
);

-- Trigger runs:
-- - Finds 157 disease_grading records with:
--   - scale_id = mydataset_scale_id
--   - original_grade = 'Moderate'
--   - scaled_grade IS NULL
-- - Updates all 157 records: scaled_grade = 2
-- NOTICE: Backfilled 157 disease_grading records
```

## Querying Normalized Grades

### Get all DR grades in standardized ICDR scale

```sql
SELECT 
  image_id,
  disease_type,
  original_grade,
  scaled_grade,
  s.scale_name as original_scale
FROM disease_grading dg
JOIN grading_scales s ON dg.scale_id = s.scale_id
WHERE disease_type = 'DR'
  AND scaled_grade IS NOT NULL;
```

### Count grades by severity (across all scales)

```sql
SELECT 
  scaled_grade,
  COUNT(*) as image_count
FROM disease_grading
WHERE disease_type = 'DR'
  AND scaled_grade IS NOT NULL
GROUP BY scaled_grade
ORDER BY scaled_grade;
```

### Find ungradable images (no conversion)

```sql
SELECT 
  image_id,
  s.scale_name,
  original_grade
FROM disease_grading dg
JOIN grading_scales s ON dg.scale_id = s.scale_id
WHERE disease_type = 'DR'
  AND scaled_grade IS NULL
  AND original_grade IS NOT NULL;
```

## Handling ICDR 0-4 vs 0-5

Two ICDR scale variants are supported:

- **ICDR_0_4**: Standard clinical scale (0=None, 1=Mild, 2=Moderate, 3=Severe, 4=PDR)
- **ICDR_0_5**: Includes ungradable (5=Ungradable/poor quality/laser scars)

Strategy for grade 5:
1. Store in `disease_grading` with scale=ICDR_0_5, original_grade="5"
2. Optionally also store in `quality_annotations` as ungradable
3. For clinical comparisons, use ICDR_0_4 as target (exclude grade 5)

## Adding New Mappings

You can manually add mappings for new scales:

```python
from chaksudb.ingest.framework.grading_scales import create_mapping

# Example: Map AAO grade 3 to ICDR_0_4 grade 3
mapping_id = await create_mapping(
    source_scale_id=aao_scale_id,
    target_scale_id=icdr_scale_id,
    source_value="3",
    target_value=3,
    mapping_confidence="exact",
)

# Backfill trigger automatically updates historical grades
```

## Troubleshooting

### Warning: Unknown scale registered

**Message:** `Auto-registering unknown scale 'XYZ' for DR. No conversion mappings exist yet.`

**Meaning:** You're using a scale that hasn't been bootstrapped. Data will be stored with `scaled_grade = NULL` until mappings are added.

**Fix:** Add mappings using `create_mapping()` or bootstrap from a reference dataset.

### No mapping found

**Message:** `No mapping found for scale X value Y to target scale ICDR_0_4. scaled_grade set to NULL.`

**Meaning:** A specific grade value doesn't have a mapping defined.

**Fix:** Add the missing mapping:
```python
await create_mapping(
    source_scale_id=your_scale_id,
    target_scale_id=icdr_scale_id,
    source_value="Y",
    target_value=2,  # or appropriate ICDR grade
    mapping_confidence="exact",
)
```

## Best Practices

1. **Run bootstrap first**: Always run `bootstrap_grading_scales.py` before ingesting datasets
2. **Provide scale metadata**: When using auto-registration, provide `value_labels` and `scale_description`
3. **Use consistent scale names**: Use descriptive names like "EYEPACS_0_4" not just "EYEPACS"
4. **Check for existing scales**: Use `get_or_create_scale()` - it's idempotent
5. **Monitor warnings**: Pay attention to "unknown scale" warnings
6. **Document mappings**: When adding manual mappings, document your reasoning
7. **Validate conversions**: After ingestion, check `scaled_grade` distribution
