# DRIONS-DB

## Overview
The DRIONS-DB (Digital Retinal Images for Optic Nerve Segmentation Database) dataset is designed for evaluating and benchmarking optic disc (papilla) segmentation algorithms. The dataset consists of 110 colour digital retinal fundus images with expert-annotated optic disc contours. 
The images were selected from an initial pool of 124 eye fundus images obtained from the Ophthalmology Service at Miguel Servet Hospital, Saragossa (Spain). Fourteen images containing severe or moderate cataracts were excluded, resulting in the final set of 110 images. Ground truth contours were independently traced by two medical experts, and the gold standard used in the original work is the average of the two expert contours. The dataset includes images with diverse visual characteristics that may affect segmentation performance.

## Images
- The dataset contains **110 colour retinal fundus images**.
- Images are stored in **JPG** format.
- Images are in **RGB** colour space with **8 bits/pixel**.
- Image resolution is **600 × 400 pixels**.
- Images were acquired using a colour analogical fundus camera and digitised from slides using an HP-PhotoSmart-S20 high-resolution scanner.
- Images are approximately centred on the **ONH (Optic Nerve Head)**.
- Image field of view and camera model beyond the provided information are not specified.
## Metadata
Metadata for the dataset is provided in a single JSON file that maps each image to its corresponding optic disc contour annotation.

### Metadata Files
#### metadata.json
The `metadata.json` file contains an array of 110 objects, one per image.
- **image_file_name** -- Filename of the fundus image (e.g., `18_DRIONS_DB_001.jpg`).
- **dataset_name** -- Dataset identifier, always `18_DRIONS_DB`.
- **image_file_path** -- Relative path to the image file (references `../FundusImages/`, while actual images are located in the `documents/` directory).
- **segmentation** -- Array describing segmentation annotations associated with the image.
- **type** -- Segmentation type, set to `OD` (Optic Disc).
- **contour_path** -- Relative path to the optic disc contour annotation file (e.g., `../Contours/OD/18_DRIONS_DB_001.txt`).
 #### Contour Annotation Files
Contour annotation files are stored in the `Contours/OD/` directory.
- **File format** -- Plain text (`.txt`).
- **Content** -- Each line contains a space-separated `x y` pixel coordinate pair representing a point on the optic disc contour.
- **Coordinate system** -- Pixel coordinates in the corresponding image.
- **Naming convention** -- `18_DRIONS_DB_XXX.txt`, where `XXX` ranges from 001 to 110.
- **Annotation type** -- Optic disc (papilla) boundary.
- **Expert information** -- The provided contour represents the averaged gold standard derived from two independent expert annotations.
## Splits
No explicit training, validation, or test splits are provided. The dataset contains 110 images in total, and any partitioning must be performed externally by the user.

## File Schema
```
18_DRIONS-DB/
metadata.json
documents/
18_DRIONS_DB_001.jpg
18_DRIONS_DB_002.jpg
...
Contours/
OD/
18_DRIONS_DB_001.txt
18_DRIONS_DB_002.txt
...
Misc/
visualise.ipynb
add_to_json.py
```

## Storage in database

### Tables populated
- **`datasets`**: One record for DRIONS-DB with `modality_types=["fundus"]` and `task_types=["segmentation"]`.
- **`images`**: One row per entry in `metadata.json`. `original_image_id` is the filename stem (e.g., `18_DRIONS_DB_001`). Image path is first resolved from the relative `image_file_path` in the JSON (relative to `metadata.json`); if not found, the script falls back to `documents/{image_file_name}`. Fields extracted via `get_image_metadata_dict` and `extract_image_metadata` (the latter also extracts width/height required for contour processing). `modality="fundus"`. `acquisition_date` and `image_quality` are `None`.
- **`segmentation_annotations`**: One record per contour file listed in the `segmentation` array of each JSON entry. `annotation_type` is mapped from the `type` field: `"OD"` → `"optic_disc"`, `"OC"` → `"optic_cup"` (other values are lowercased). Processed via `process_segmentation_from_contour` with `coordinate_format="line_separated"` (each line is `x y`). The contour represents averaged expert annotations (consensus); `expert_annotation_id=None`. `annotation_method="manual"`.
- **`raw_annotation_files`**: One record for `metadata.json` (registered via `process_json` with `unified_annotation_type="segmentation"`). One record per contour `.txt` file (registered with `unified_annotation_type="segmentation"`, `file_type="txt"`). Contour paths are resolved from the JSON's relative `contour_path`; if not found, the script looks in `Contours/OD/` or `Contours/`.

### Image metadata extraction
`extract_image_metadata` is called first to obtain `(width, height)` required for polygon conversion. `get_image_metadata_dict` is called to populate standard image fields. No laterality detection and no patient or expert records are created.

### Annotation types
- **Segmentation** (`optic_disc`): Contour coordinates from text files in `Contours/OD/`. Represents the averaged gold standard (consensus of two expert annotations). No per-expert breakdown is stored.

### Splits
No train/test splits are created. The dataset has no split definitions.

### Provenance
`metadata.json` is registered as the root raw annotation file via `process_json`, which sets the provenance context for all entries. Each contour `.txt` file is registered individually and linked to its `segmentation_annotation` via a separate provenance chain.

### Special processing
- The JSON is read via `process_json`, which sets the provenance context before invoking `process_entry` for each element.
- Image dimensions must be successfully extracted before a segmentation can be created; entries failing dimension extraction are skipped.
- Ingestion is idempotent due to deterministic UUID generation and upsert logic.