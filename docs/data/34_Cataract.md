# Cataract

## Overview
The Retina dataset is a retinal image database containing four categories: normal, cataract, glaucoma, and retina disease. Images are organized into category-specific directories with directory-based labeling.

## Images
- All images are stored in **PNG** format.
- Images are organized into four category directories: `1_normal/`, `2_cataract/`, `2_glaucoma/`, and `3_retina_disease/`.
- Normal images are named with the pattern `NL_###`.png (e.g., `NL_001.png`, `NL_300.png`).
- Cataract images are named with the pattern `cataract_###`.png (e.g., `cataract_001.png`, `cataract_100.png`).
- Glaucoma images are named with the pattern `Glaucoma_###`.png (e.g., `Glaucoma_001.png`, `Glaucoma_101.png`).
- Retina disease images are named with the pattern `Retina_###`.png (e.g., `Retina_001.png`, `Retina_100.png`).
## Metadata
The dataset does not provide explicit metadata files. Image labels are inferred from the directory structure, where each directory corresponds to a disease category: normal, cataract, glaucoma, or retina disease.

## Splits
The dataset does not provide explicit train--test splits. All images are organized by category directories, and any data partitioning must be performed externally.

## File Schema
```
34_Cataract/
1_normal/
*.png
2_cataract/
*.png
2_glaucoma/
*.png
3_retina_disease/
*.png
README.md
```

## Storage in database

### Tables populated

- **`datasets`**: One record for Cataract (name=`"Cataract"`, source_url=`https://github.com/sjchoi86/retina_dataset`, license=`Unknown`, `modality_types=['fundus']`).
- **`images`**: One row per `.png` (or `.PNG`) image found across the four category folders. `original_image_id` is the file stem (no extension). `modality='fundus'`. Image metadata (dimensions, format, etc.) is extracted via `get_image_metadata_dict`. Image UUIDs are generated from the dataset UUID and the file stem.
- **`classification_annotations`**: One `ClassificationAnnotation` per image with `class_name="disease_category"`, `task_type="multi_class"`. The class is determined from the parent folder name using this mapping:
  - `"1_normal"` → class index `0` (`"normal"`)
  - `"2_cataract"` → class index `1` (`"cataract"`)
  - `"2_glaucoma"` → class index `2` (`"glaucoma"`)
  - `"3_retina_disease"` → class index `3` (`"retina_disease"`)

  The integer class index is stored as `class_value`. `class_labels={0: "normal", 1: "cataract", 2: "glaucoma", 3: "retina_disease"}` is stored with each annotation. `annotation_method="manual"`. No `expert_annotation_id` or `raw_data_id`.
- **`raw_annotation_files`**: Registered automatically per `process_folder_tree` call (one raw file / provenance chain for the folder tree traversal).
- **`provenance_chains`**: One chain created per `process_folder_tree` call.
- **`dataset_splits`**: A single `"train"` split (`split_type="explicit"`).
- **`image_splits`**: All images assigned to the `"train"` split.

### Annotation types
- **`classification_annotations`**: `class_name="disease_category"`, `task_type="multi_class"`, 4 classes (0=normal, 1=cataract, 2=glaucoma, 3=retina_disease). Label taxonomy: mutually exclusive single-label per image, sourced from folder structure.

### Splits created
- `train` (explicit): all images assigned here. The dataset has no separate test split in its folder structure.

### Provenance / raw annotation files registered
- One folder-tree traversal raw file entry (no named CSV or annotation files — labels come entirely from the folder structure).

### Special processing
- Images in unrecognized folders are skipped with a warning.
- The `process_folder_tree` function uses `recursive=True`, but the images are expected to be directly inside the four category folders (not nested further).
- Only `.png` / `.PNG` files are processed.
- The ingestion process is idempotent. Re-running it will not create duplicate images or annotations due to deterministic UUID generation and upsert logic.