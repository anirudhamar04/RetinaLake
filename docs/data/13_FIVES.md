# FIVES

## Overview
The FIVES dataset is a color fundus image vessel segmentation dataset. It consists of 800 high-resolution multi-disease color fundus photographs with pixelwise manual annotations. The annotation process was standardized through crowdsourcing among medical experts, and the quality of each image was evaluated.

## Images
- Images are color fundus photographs intended for vessel segmentation.
- All images are stored in **JPG** format.
- The dataset includes original images and corresponding ground truth segmentation images.
- Images are organized into clean folders separating original images and ground truth annotations.
## Metadata
Metadata is provided in an Excel file containing image quality assessment information.

### Metadata Files
**Quality Assessment.xlsx** 
This file contains quality assessment information for images in the dataset.
- **Disease** -- Disease category identifier (e.g., A). Images are named “n_A/D/G/N.png”, where “n” means the number of images and “A”, “D”, “G”, and “N” stand for “AMD”, “DR”, “Glaucoma” and “Normal”
- **Number** -- Image index number within the disease category.
- **IC** -- Image contrast quality indicator (binary value).
- **Blur** -- Blur quality indicator (binary value).
- **LC** -- Lighting condition quality indicator (binary value).
## Splits
The dataset explicitly provides splits into training and testing sets.

## File Schema
```
13_FIVES/
test/
Ground truth/
Original/
train/
Ground truth/
Original/
Quality Assessment.xlsx
```

## Storage in database

### Tables populated
- **`datasets`**: One record for FIVES with `modality_types=["fundus"]` and `task_types=["segmentation", "classification", "quality"]`.
- **`images`**: One row per image in `train/Original/` and `test/Original/`. Fields extracted via `get_image_metadata_dict`. `modality="fundus"`. `original_image_id` is the filename stem (e.g., `5_A`). Disease code and name are stored in `comorbidities={"disease_code": "A", "disease_name": "AMD"}`. `acquisition_date` and `image_quality` are `None`.
- **`segmentation_annotations`**: One record per image, processed from the corresponding mask in `train/Ground truth/` or `test/Ground truth/` via `process_segmentation_from_binary_mask` with `fill_holes=False`. `annotation_type="vessels"`. No expert attribution.
- **`classification_annotations`**: One record per image. `task_type="multi_class"`, `class_name="disease_type"`. `class_value` is one of: `"AMD"`, `"DR"`, `"Glaucoma"`, `"Normal"` (derived from the single-letter disease code in the filename: `A`→AMD, `D`→DR, `G`→Glaucoma, `N`→Normal). Provenance linked to the `Quality Assessment.xlsx` Excel file (which contains the `Disease` column).
- **`quality_annotations`**: Three records per image (contrast, blur, illumination), sourced from `Quality Assessment.xlsx`. Each has `quality_score` of `0` (poor) or `1` (good) and a corresponding `quality_label` of `"poor"` or `"good"`. Scales: `"FIVES Image Contrast (0=poor, 1=good)"`, `"FIVES Blur Quality (0=poor, 1=good)"`, `"FIVES Lighting Condition (0=poor, 1=good)"` with `scale_min=0`, `scale_max=1`. Missing values default to `1`.
- **`raw_annotation_files`**: One record for the `Quality Assessment.xlsx` file (registered twice — once per sheet: `Train` and `Test`, `unified_annotation_type="quality"`). Ground truth mask files are registered by the segmentation processor.
- **`dataset_splits`** / **`image_splits`**: Explicit `train` and `test` splits created via `register_standard_splits`. All images from `train/` are assigned to train; all images from `test/` are assigned to test.

### Image metadata extraction
Filenames follow the pattern `{number}_{disease_code}.png`. The stem is parsed to extract the disease code and map it to a disease name. `get_image_metadata_dict` reads each file for resolution, size, and format.

### Annotation types
- **Segmentation** (`vessels`): Binary ground truth masks from `Ground truth/` directory; one per image; no expert attribution.
- **Classification** (`disease_type`, multi-class): Disease derived from filename letter code. `class_value` is the full disease name string.
- **Quality** (three types per image): `contrast`, `blur`, `illumination`; binary scale 0/1.

### Splits
Explicit train and test splits. The `Quality Assessment.xlsx` file contains separate `Train` and `Test` sheets; both are processed in parallel.

### Provenance
The `Quality Assessment.xlsx` file is registered as a raw file and linked to both classification and quality annotations via provenance chains. Segmentation mask files are registered by the segmentation processor.

### Special processing
- The Excel file is processed in parallel for both sheets (`Train` and `Test`), each yielding a separate `raw_file_id` and `chain_id`.
- Quality lookup is keyed by `{number}_{disease_code}` matching the `original_image_id`.
- Ingestion is idempotent due to deterministic UUID generation and upsert logic.