# APTOS

## Overview
The APTOS dataset was released in the context of a Kaggle competition focused on automated detection of diabetic retinopathy severity using retinal fundus images. The dataset was collected by Aravind Eye Hospital in India with the goal of enabling large-scale screening in rural areas where access to ophthalmologists is limited. Images were captured by technicians in rural settings and annotated to support machine learning models that predict disease severity. The dataset supports five discrete diabetic retinopathy severity ratings and was designed for supervised learning and evaluation using the quadratic weighted kappa metric.

## Images
- The dataset consists of retinal fundus images.
- All images are provided in **PNG** format.
- Images have differing resolutions.
- Image filenames correspond to unique image identifiers referenced in the metadata CSV files.
## Metadata
Metadata for the dataset is provided using CSV files that define training labels and test image identifiers.

### Metadata Files

**train.csv**

The `train.csv` file contains image identifiers and corresponding diabetic retinopathy severity labels.
- **id_code** -- Unique identifier corresponding to a fundus image filename.
- **diagnosis** -- Integer label representing diabetic retinopathy severity with possible values `0, 1, 2, 3, 4`.
**test.csv**

The `test.csv` file contains identifiers for test images without labels.
- **id_code** -- Unique identifier corresponding to a fundus image filename.
## Splits
The dataset provides explicit training and test splits defined by separate `train.csv` and `test.csv` files and corresponding image directories. No validation split is provided.

## File Schema
```
15_APTOS/
test_images/
train_images/
test.csv
train.csv
```

## Storage in database

### Tables populated
- **`datasets`**: One record for APTOS with `modality_types=["fundus"]` and `license="CC-BY-4.0"`. No `task_types` set at dataset level.
- **`images`**: One row per image from `train.csv` (looked up in `train_images/`) and `test.csv` (looked up in `test_images/`). `original_image_id` is the `id_code` value from the CSV. Fields extracted via `get_image_metadata_dict`. `modality="fundus"`. Images are expected as `.png`; the script falls back to `.PNG`, `.jpg`, `.JPG`, `.jpeg`, `.JPEG` if the primary path is missing. `acquisition_date` and `image_quality` are `None`.
- **`disease_grading`**: One record per training image. `disease_type="DR"`, `scale_name="ICDR_0_4"`, `grade_value` is the integer `diagnosis` field from `train.csv` (0–4: No DR, Mild NPDR, Moderate NPDR, Severe NPDR, Proliferative DR). Processed via `process_disease_grade`. No grading is stored for test images.
- **`raw_annotation_files`**: One record for `train.csv` and one for `test.csv`, both registered with `unified_annotation_type="grading"` via `process_csv`.
- **`dataset_splits`** / **`image_splits`**: Explicit `train` and `test` splits created via `register_standard_splits`. Images from `train.csv` → train split; images from `test.csv` → test split.

### Image metadata extraction
Each image file is read by `get_image_metadata_dict` to extract resolution, size, and format. No laterality detection and no patient records are created.

### Annotation types
- **Disease grading** (`DR`, scale `ICDR_0_4`): Integer grade 0–4 from `train.csv` → `disease_grading` table. Test images have no grading annotations.

### Splits
Explicit train and test splits. `train.csv` and `test.csv` are processed in parallel; each CSV is registered as a separate raw annotation file.

### Provenance
`train.csv` is registered as a raw annotation file; all `disease_grading` records are linked to it via a provenance chain. `test.csv` is similarly registered. Both chains are logged at ingestion time.

### Special processing
- Both CSVs are processed concurrently via `asyncio.gather`.
- Images are bulk-upserted before gradings to satisfy foreign key constraints.
- Ingestion is idempotent due to deterministic UUID generation and upsert logic.