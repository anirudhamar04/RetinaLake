# ChaksuDB Schema Reference

A complete reference for the PostgreSQL schema (`schema/schema.sql`).

---

## Table of Contents

1. [Design Principles](#1-design-principles)
2. [Core Tables](#2-core-tables)
3. [Annotation Tables](#3-annotation-tables)
4. [Provenance Tables](#4-provenance-tables)
5. [Organisation Tables](#5-organisation-tables)
6. [Storage Fields on `images`](#6-storage-fields-on-images)
7. [Annotation Source Selection](#7-annotation-source-selection)
8. [Triggers & Grading Normalisation](#8-triggers--grading-normalisation)
9. [Key Indexes](#9-key-indexes)

---

## 1. Design Principles

### UUID v5 everywhere

Every primary key is a **UUID v5** (SHA-1-based deterministic UUID). UUIDs are computed from content fields in application code before any INSERT — the database never generates them. Because the same content always produces the same UUID, re-running an ingestion script is a safe no-op (upserts silently skip conflicting rows). There is no need for a `SELECT`-before-`INSERT` pattern.

```
namespace hierarchy:
  dataset_id  = UUID5(NAMESPACE_DNS, dataset_name)
  image_id    = UUID5(dataset_id,    original_image_id)
  grading_id  = UUID5(image_id,      disease_type + scale_id)
  seg_id      = UUID5(image_id,      annotation_type + lesion_subtype + ...)
```

See `chaksudb/ingest/framework/gen_uuid.py` for the full generator set.

### Image-centric, no pixels

Every annotation row has an `image_id` foreign key. No pixel data is stored in PostgreSQL — only file paths and metadata. Images are loaded at read-time by the export pipeline.

### Layered provenance

Every annotation can be traced back to its source through four tables:

```
annotation row
  └── expert_annotation_id  → expert_annotations
        └── raw_data_id      → raw_annotation_files
              └── (chain_id) → provenance_chain
                    └── transformation_ids → transformation_operations
```

---

## 2. Core Tables

### `datasets`

| Column | Type | Description |
|---|---|---|
| `dataset_id` | UUID PK | `UUID5(NAMESPACE_DNS, dataset_name)` |
| `dataset_name` | TEXT NOT NULL | Human-readable name, e.g. `"EYEPACS"` |
| `source_url` | TEXT | Original dataset URL |
| `license` | TEXT | Data license |
| `modality_types` | TEXT[] | e.g. `["fundus", "oct"]` |
| `created_at` | TIMESTAMP | Auto-set |

### `models`

Optional table for ML models used as pseudo-labellers.

| Column | Type | Description |
|---|---|---|
| `model_id` | UUID PK | Deterministic UUID |
| `model_name` | TEXT NOT NULL | e.g. `"ResNet50-DR"` |
| `model_description` | TEXT | Free-text |
| `model_url` | TEXT | Weights or paper URL |

### `experts`

Human annotators or pseudo-experts (models). Every expert belongs to either a dataset or a model — not both.

| Column | Type | Description |
|---|---|---|
| `expert_id` | UUID PK | Deterministic UUID |
| `expert_name` | TEXT | Annotator name |
| `expertise_area` | TEXT | Clinical specialty |
| `dataset_id` | UUID FK | Set for real human experts |
| `model_id` | UUID FK | Set for pseudo-experts (models) |

**Constraint**: `dataset_id IS NOT NULL OR model_id IS NOT NULL`.

### `patients`

| Column | Type | Description |
|---|---|---|
| `patient_id` | UUID PK | `UUID5(dataset_id, original_patient_id)` |
| `dataset_id` | UUID FK NOT NULL | Owning dataset |
| `original_patient_id` | TEXT NOT NULL | Source ID string |
| `age` | INTEGER | Age at annotation |
| `sex` | TEXT | `male`, `female`, or `unknown` |
| `ethnicity` | TEXT | Free-text |
| `nationality` | TEXT | Free-text |
| `comorbidities` | JSONB | Structured comorbidity data |

**Unique constraint**: `(dataset_id, original_patient_id)`.

### `images`

The central table. Every annotation row has an `image_id` FK to this table.

| Column | Type | Description |
|---|---|---|
| `image_id` | UUID PK | `UUID5(dataset_id, original_image_id)` |
| `dataset_id` | UUID FK NOT NULL | Owning dataset |
| `original_image_id` | TEXT | Source filename / ID string |
| `storage_provider` | TEXT | `local`, `s3`, `gcs`, `azure`, `http` |
| `bucket` | TEXT | Cloud bucket (non-local only) |
| `object_key` | TEXT | Cloud object key |
| `file_path` | TEXT | Local file path |
| `file_format` | TEXT | `jpg`, `png`, `tif`, `dicom`, etc. |
| `modality` | TEXT | `fundus`, `oct`, `fa`, `uwf` |
| `group_id` | UUID FK | OCT volume group (optional) |
| `frame_index` | INTEGER | Frame position within OCT volume |
| `resolution_width` | INTEGER | Width in pixels |
| `resolution_height` | INTEGER | Height in pixels |
| `field_of_view` | INTEGER | FOV in degrees |
| `eye_laterality` | TEXT | `left`, `right`, `unknown` |
| `acquisition_date` | DATE | Capture date |
| `file_hash` | TEXT | SHA-256 of the raw bytes (exact-duplicate detection) |
| `content_hash` | TEXT | Hash of decoded pixels — encoding-invariant; the **canonical dedup key** |
| `phash` | TEXT | Perceptual dHash — matches across lossy re-encode/resize (advisory) |

The three hashes are populated centrally in `chaksudb/ingest/framework/image_helpers.py`.
`content_hash` is the canonical dedup key; `phash` is advisory (review before merging
near-duplicates). See `scripts/find_duplicate_images.py`.

See [§6 Storage Fields](#6-storage-fields-on-images) for storage locator rules.

### `image_groups`

Groups OCT volumes and temporal sequences.

| Column | Type | Description |
|---|---|---|
| `group_id` | UUID PK | Deterministic UUID |
| `dataset_id` | UUID FK NOT NULL | Owning dataset |
| `group_type` | TEXT | `oct_volume`, `video`, `sequence` |

### `patient_images`

Many-to-many join table linking patients to images.

| Column | Type | Description |
|---|---|---|
| `relationship_id` | UUID PK | Deterministic UUID |
| `patient_id` | UUID FK NOT NULL | |
| `image_id` | UUID FK NOT NULL | |
| `exam_date` | DATE | Date of examination |

**Unique constraint**: `(patient_id, image_id)`.

---

## 3. Annotation Tables

All annotation tables share these design rules:
- `image_id` FK — links to the image being annotated
- `expert_annotation_id` FK — optional link to `expert_annotations`
- `consensus_id` FK — optional link to `consensus_annotations`
- `raw_data_id` FK — optional link to `raw_annotation_files`
- `provenance_chain_id` FK — optional link to `provenance_chain`
- `annotation_method` — `manual`, `semi_automatic`, `automatic`, or `pseudo`

### `disease_grading`

| Column | Type | Description |
|---|---|---|
| `grading_id` | UUID PK | `UUID5(image_id, disease_type + scale_id)` |
| `image_id` | UUID FK NOT NULL | |
| `disease_type` | TEXT | `DR`, `DME`, `Glaucoma`, `AMD` |
| `scale_id` | UUID FK NOT NULL | Reference to `grading_scales` |
| `original_grade` | TEXT | Raw grade string from source |
| `scaled_grade` | INTEGER | Normalised integer grade |
| `grade_label` | TEXT | Human-readable label |

### `segmentation_annotations`

| Column | Type | Description |
|---|---|---|
| `segmentation_id` | UUID PK | Content-based UUID |
| `image_id` | UUID FK NOT NULL | |
| `annotation_type_id` | UUID FK NOT NULL | References `annotation_type` vocabulary |
| `lesion_subtype` | TEXT | e.g. `microaneurysm`, `hemorrhage` |
| `mask_file_path` | TEXT | Path to the mask to use (original or processed) |
| `original_file_path` | TEXT | Always points to the source in `data/` |
| `unified_format` | TEXT | Storage format: `binary_mask`, `soft_map`, `layer_boundary`, etc. |
| `original_format` | TEXT | Source format before conversion |
| `confidence_score` | FLOAT | Annotation confidence |

### `localization_annotations`

| Column | Type | Description |
|---|---|---|
| `localization_id` | UUID PK | Content-based UUID |
| `image_id` | UUID FK NOT NULL | |
| `localization_type` | TEXT | `bounding_box`, `keypoint`, `center_point` |
| `target_structure` | TEXT | e.g. `optic_disc`, `fovea`, `lesion` |
| `coordinates` | JSONB NOT NULL | Format depends on `localization_type` |
| `lesion_subtype` | TEXT | Optional lesion subtype |

**Coordinate formats by type:**

| Type | JSONB keys |
|---|---|
| `bounding_box` | `xmin`, `ymin`, `xmax`, `ymax` (pixels) |
| `keypoint` | `x`, `y` (pixels) |
| `center_point` | `center_x`, `center_y`, `radius` (pixels) |

### `classification_annotations`

Every row is **self-describing**: it carries both the ML task it belongs to and the canonical
concept it maps to, so the same disease is retrievable across datasets regardless of how it was
stored (binary / multi_class / multi_label).

| Column | Type | Description |
|---|---|---|
| `classification_id` | UUID PK | Content-based UUID |
| `image_id` | UUID FK NOT NULL | |
| `task_type` | TEXT NOT NULL | `binary`, `multi_class`, or `multi_label` (CHECK-constrained) |
| `task_name` | TEXT NOT NULL | The dataset's ML task this row belongs to (e.g. `glaucoma`, `disease_panel`); drives task-pivot export columns |
| `class_name` | TEXT NOT NULL | The label/class **within** the task |
| `concept` | TEXT | Canonical clinical concept for cross-task filtering (e.g. `glaucoma` for both binary AIROGS and multi-label PAPILA); `NULL` = no shared concept |
| `is_multilabel` | BOOLEAN NOT NULL | Explicit shape flag |
| `class_index` | INT NOT NULL | Scalar label index (uniform across all task types) |
| `class_label` | TEXT NOT NULL | The *real* label, e.g. `RG` (not `positive`) |
| `sub_key` | TEXT | Multi-label sub-key (one per disease); `NULL` for binary/multi_class |
| `class_value` | JSONB | Original/raw data, soft labels, or backward-compat payload |
| `annotation_method` | TEXT NOT NULL | `manual`, `adjudicated`, `consensus`, or `pseudo` |
| `confidence_score` | FLOAT | Annotation confidence |

Multi-disease datasets (RFMiD, RFMiD2, MuReD, ODIR, BRSET) are all stored as **one**
`multi_label` row-set with `task_name="disease_panel"` (one `sub_key` per disease). The canonical
concept vocabulary lives in `chaksudb/ingest/framework/concepts.py` and is shared by ingest
(normalization) and export (concept columns / filters).

### `quality_annotations`

| Column | Type | Description |
|---|---|---|
| `quality_id` | UUID PK | Content-based UUID |
| `image_id` | UUID FK NOT NULL | |
| `quality_type` | TEXT FK | FK → `quality_types` registry (`overall`, `gradability`, `clarity`, `blur`, …); new types auto-register via `get_or_create_quality_type` |
| `quality_score` | FLOAT | Numeric score |
| `quality_label` | TEXT | Categorical label (e.g. `good`, `poor`) |

> `quality_type` is a foreign key to the extensible `quality_types` reference table, **not** a
> hard-coded CHECK — pseudo-IQA from `run_roi_iqa.py` and human quality labels share it.

### `keyword_annotations`

| Column | Type | Description |
|---|---|---|
| `keyword_annotation_id` | UUID PK | Content-based UUID |
| `image_id` | UUID FK NOT NULL | |
| `keyword_id` | UUID FK NOT NULL | References `keyword_vocabulary` |

### `keyword_vocabulary`

| Column | Type | Description |
|---|---|---|
| `keyword_id` | UUID PK | `UUID5(NAMESPACE_DNS, term)` |
| `term` | TEXT NOT NULL UNIQUE | The keyword term |
| `definition` | TEXT | Optional definition |

### `clinical_descriptions`

| Column | Type | Description |
|---|---|---|
| `description_id` | UUID PK | Content-based UUID |
| `image_id` | UUID FK NOT NULL | |
| `description_text` | TEXT NOT NULL | Free-text clinical note |
| `description_type` | TEXT | `diagnosis_text`, `clinical_caption`, `notes` |
| `word_count` | INTEGER | Auto-computed word count |

---

## 4. Provenance Tables

### `raw_annotation_files`

Registers every source file that was parsed. Prevents duplicate ingestion via a `(dataset_id, file_hash)` unique index.

| Column | Type | Description |
|---|---|---|
| `raw_file_id` | UUID PK | `UUID5(dataset_id, file_path + file_hash)` |
| `dataset_id` | UUID FK NOT NULL | Owning dataset |
| `file_type` | TEXT | `csv`, `json`, `xml`, `excel`, `txt`, `mat`, etc. |
| `file_name` | TEXT | Base filename |
| `file_hash` | TEXT | SHA-256 hash of file contents |
| `file_size` | BIGINT | File size in bytes |
| `parsed_status` | TEXT | `not_parsed`, `parsed`, `error` |
| `parse_errors` | TEXT | Error log if status is `error` |
| Storage fields | — | Same locator fields as `images` |

### `expert_annotations`

Metadata about a single annotator's contribution for one task on one image.

| Column | Type | Description |
|---|---|---|
| `expert_annotation_id` | UUID PK | Content-based UUID |
| `expert_id` | UUID FK NOT NULL | The annotator |
| `annotation_task` | TEXT | `grading`, `segmentation`, etc. |
| `raw_data_id` | UUID FK | Source file |
| `annotation_value` | JSONB | Snapshot of the raw annotation value |
| `confidence_level` | TEXT | `high`, `medium`, `low` |

### `consensus_annotations`

Aggregated multi-expert annotation for a single image.

| Column | Type | Description |
|---|---|---|
| `consensus_id` | UUID PK | Content-based UUID |
| `image_id` | UUID FK NOT NULL | |
| `annotation_task` | TEXT | Task type |
| `consensus_method` | TEXT | `majority_vote`, `mean`, `median`, `staple`, `adjudicated`, `senior_review` |
| `expert_annotation_ids` | UUID[] | Contributing expert annotation IDs |
| `consensus_value` | JSONB | The agreed-upon value |
| `agreement_score` | FLOAT | Inter-annotator agreement (0–1) |
| `adjudicator_id` | UUID FK | Expert who resolved disagreements |

### `provenance_chain`

Records the lineage of a set of annotations.

| Column | Type | Description |
|---|---|---|
| `chain_id` | UUID PK | Content-based UUID |
| `unified_annotation_type` | TEXT | Task type |
| `source_type` | TEXT | `original`, `transformed`, `pseudo_generated`, `consensus` |
| `root_source_raw_data_id` | UUID FK | The originating raw file |
| `source_annotation_ids` | UUID[] | Input annotation IDs |

### `transformation_operations`

Records transformations applied to annotations (e.g. contour-to-mask conversion, scale normalisation).

| Column | Type | Description |
|---|---|---|
| `transformation_id` | UUID PK | Content-based UUID |
| `operation_type` | TEXT | e.g. `contour_to_mask`, `scale_normalisation` |
| `input_data` | JSONB | Input parameters |
| `output_data` | JSONB | Output parameters |
| `operation_parameters` | JSONB | Transformation configuration |
| `operator` | TEXT | Script or module that performed it |

`provenance_transformations` is a join table linking `provenance_chain` ↔ `transformation_operations`.

---

## 5. Organisation Tables

### `grading_scales`

| Column | Type | Description |
|---|---|---|
| `scale_id` | UUID PK | `UUID5(NAMESPACE_DNS, scale_name + disease_type)` |
| `scale_name` | TEXT NOT NULL | e.g. `ICDR`, `ETDRS` |
| `disease_type` | TEXT | `DR`, `DME`, `Glaucoma`, `AMD` |
| `min_value` | INTEGER | Minimum integer grade |
| `max_value` | INTEGER | Maximum integer grade |
| `value_labels` | JSONB | Map of integer → label, e.g. `{"0": "No DR", "1": "Mild"}` |

### `grading_scale_mappings`

Maps a source grade value to a target grade value across two scales.

| Column | Type | Description |
|---|---|---|
| `mapping_id` | UUID PK | Content-based UUID |
| `source_scale_id` | UUID FK NOT NULL | |
| `target_scale_id` | UUID FK NOT NULL | |
| `source_value` | TEXT NOT NULL | Source grade (text) |
| `target_value` | INTEGER | Mapped target grade |
| `mapping_confidence` | TEXT | `exact`, `approximate`, `manual_review_required` |

### `dataset_splits`

| Column | Type | Description |
|---|---|---|
| `split_id` | UUID PK | `UUID5(dataset_id, split_name + task_type)` |
| `dataset_id` | UUID FK NOT NULL | |
| `split_name` | TEXT NOT NULL | `train`, `val`, `test` |
| `task_type` | TEXT | Task this split was designed for |
| `split_description` | TEXT | Free-text description |

### `image_splits`

Join table assigning images to splits.

| Column | Type | Description |
|---|---|---|
| `image_split_id` | UUID PK | `UUID5(image_id, split_id)` |
| `image_id` | UUID FK NOT NULL | |
| `split_id` | UUID FK NOT NULL | |

**Unique constraint**: `(image_id, split_id)`.

---

## 6. Storage Fields on `images`

The `images` table (and `raw_annotation_files`) uses a flexible storage locator pattern that supports five backends:

| Field | Local | S3 / GCS / Azure | HTTP |
|---|---|---|---|
| `storage_provider` | `"local"` | `"s3"` / `"gcs"` / `"azure"` | `"http"` |
| `file_path` | **Required** | NULL | NULL |
| `bucket` | NULL | Optional | NULL |
| `object_key` | NULL | **Required** | **Required** |
| `version_id` | NULL | Optional (S3) | NULL |

**Enforced by constraint:**

```sql
CONSTRAINT ck_images_locator CHECK (
  (storage_provider = 'local' AND file_path IS NOT NULL)
  OR
  (storage_provider <> 'local' AND object_key IS NOT NULL)
)
```

The export pipeline reads `storage_provider` and the appropriate locator field to resolve image paths at runtime. Set `base_path_for_paths` in `ExportSpec` to prepend a prefix to `file_path` values in the output.

---

## 7. Annotation Source Selection

Each annotation task can have both an expert annotation and a consensus annotation for the same image. The `annotation_source` field in `ExportSpec` controls which one is returned:

| Value | Behaviour |
|---|---|
| `"prefer_consensus"` | Return consensus if available; fall back to expert (default) |
| `"expert_only"` | Return only rows from `expert_annotations` |
| `"consensus_only"` | Return only rows from `consensus_annotations` |
| `"both"` | Return both when both exist (can produce multiple rows per image) |

This is implemented in the query modules via `LEFT JOIN` / `INNER JOIN` selection on the `expert_annotation_id` and `consensus_id` columns in each annotation table.

---

## 8. Triggers & Grading Normalisation

Grade normalisation (mapping `original_grade` to `scaled_grade`) is handled by **database triggers** defined in the schema. When a row is inserted into `disease_grading` with an `original_grade` but no `scaled_grade`, the trigger looks up the `grading_scale_mappings` table and fills in `scaled_grade` automatically.

This means ingestion scripts only need to supply `original_grade` — the trigger handles the rest. See [`docs/library/grading_scale_normalization.md`](grading_scale_normalization.md) for full details on the trigger logic and how to bootstrap the standard scales.

---

## 9. Key Indexes

Performance-critical indexes created by the schema:

| Table | Index | Purpose |
|---|---|---|
| `images` | `idx_images_dataset_id` | Filter images by dataset |
| `images` | `idx_images_original_image_id` | Look up image by source ID |
| `images` | `idx_images_group_frame` | OCT volume frame ordering |
| `images` | `uq_images_dataset_original_image_id` | Unique (partial, WHERE NOT NULL) |
| `disease_grading` | `idx_grading_image` | Join annotation → image |
| `segmentation_annotations` | `idx_seg_image` | Join annotation → image |
| `localization_annotations` | `idx_loc_image` | Join annotation → image |
| `patients` | `idx_patients_dataset_id` | Filter patients by dataset |
| `patients` | `idx_patients_original_id` | Look up by source patient ID |
| `raw_annotation_files` | `idx_raw_files_hash` | Detect duplicate source files |
| `raw_annotation_files` | `uq_raw_files_dataset_hash` | Prevent re-ingestion |
| `experts` | `idx_experts_dataset_id` | Filter experts by dataset |
| `consensus_annotations` | `idx_consensus_image` | Join consensus → image |
