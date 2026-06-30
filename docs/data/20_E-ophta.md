# E-ophta

## Overview

E-ophta is composed of lesion-specific subsets intended for the development and evaluation of automatic lesion detection methods. The publicly available subsets include **e-ophtha-MA** for microaneurysm detection and **e-ophtha-EX** for exudate detection. Lesions were manually outlined by ophthalmologists using ADCIS software, and all annotations were subsequently checked by a second ophthalmologist.

## Images
- The dataset contains **color retinal fundus images**.
- Images are stored in **JPEG** format (`.jpg` and `.JPG` extensions).
- Images are compressed using the JPEG standard.
- Images were collected through the OPHDIAT telemedical network between 2008 and 2009.
- Images are organized by **patient ID** folders (e.g., `E0000043`, `E0000225`).
- Each patient folder may contain one or more images.
- Image resolution, field of view, and camera model are not provided.
## Metadata
The dataset does not contain CSV metadata files. Metadata and dataset descriptions are provided through HTML files, and annotation information is encoded in binary mask images and the folder structure.

### Metadata Files

#### e_ophta_MA.html
HTML file describing the e-ophtha-MA subset.
- **Image counts** -- 148 images containing microaneurysms or other small red lesions, and 233 microaneurysm-free images.
- **Annotation description** -- Microaneurysms are annotated as dots or small regions in binary mask images.
#### e_ophta_EX.html
HTML file describing the e-ophtha-EX subset.
- **Image counts** -- 47 images containing exudates, and 35 exudate-free images.
- **Annotation description** -- Exudates are annotated with their position and contours in binary mask images.
#### Annotation Masks
Lesion annotations are provided as binary mask images.
- **File format** -- PNG.
- **e-ophtha-MA annotations** -- Binary masks where each microaneurysm is marked by a dot or small region.
- **e-ophtha-EX annotations** -- Binary masks marking the position and contours of each exudate.
- **Filename correspondence** -- Annotation filenames match the original image filenames, with an optional `_EX` suffix for exudate annotations.
- **Annotation process** -- Created by ophthalmologists using ADCIS software and checked by a second ophthalmologist.
## Splits
No explicit training, validation, or test splits are provided.
- **e-ophtha-MA**: 148 images with microaneurysms and 233 healthy images.
- **e-ophtha-EX**: 47 images with exudates and 35 healthy images.
Any data partitioning must be performed externally by the user.
## File Schema
```
20_E-ophta/
e_optha_MA/
MA/
E0000043/
DS000DGS.JPG
E0000225/
DS000DBO.JPG
DS000DBP.JPG
healthy/
E0000475/
C0001629.jpg
C0001635.jpg
Annotation_MA/
E0000043/
DS000DGS.png
E0000225/
DS000DBO.png
DS000DBP.png
e_optha_MA.html
e_optha_EX/
EX/
E0000404/
C0021833.jpg
C0021834.jpg
healthy/
E0000043/
DS000DGV.JPG
DS000DGY.JPG
Annotation_EX/
E0000404/
C0021833_EX.png
C0021834_EX.png
e_optha_EX.html
count.py
visualise.ipynb
```

## Storage in database

### Tables populated
- **`datasets`**: One record for E-ophta with `modality_types=["fundus"]` and `task_types=["segmentation", "classification"]`.
- **`images`**: One row per image found in the classification subfolders (`MA/`, `EX/`, `healthy/`) within each subset directory. `original_image_id` is `{patient_folder}/{image_filename}` (e.g., `E0000043/DS000DGS.JPG`). Fields extracted via `get_image_metadata_dict`. `modality="fundus"`. `acquisition_date` is `None`. The same image may appear in both `e_optha_MA` and `e_optha_EX` subsets; images are deduplicated by `image_id` before upsert (identical `patient_id + filename` yields the same UUID).
- **`segmentation_annotations`**: One record per image that has a corresponding mask in `Annotation_MA/` or `Annotation_EX/`. `annotation_type="lesions"`. `lesion_subtype` is `"MA"` for microaneurysm masks or `"EX"` for exudate masks. Processed via `process_segmentation_from_binary_mask` with `fill_holes=False` and `merge_nonzero=True`. `expert_annotation_id=None`. Mask filenames: for MA, same stem as image (e.g., `DS000DGS.png`); for EX, stem with `_EX` suffix (e.g., `C0021833_EX.png`). Healthy images have no masks.
- **`classification_annotations`**: One record per image. `task_type="multi_class"`, `class_name="lesion_type"`. `class_value` maps folder name to: `"MA"` folder → `"microaneurysm"`, `"EX"` folder → `"exudate"`, `"healthy"` folder → `"normal"`. `annotation_method="manual"`. `raw_data_id=None` (folder structure is implicit).

### Image metadata extraction
Images are discovered within patient subfolders (one level below the classification folder). `get_image_metadata_dict` extracts resolution, size, and format. Patient ID is taken from the immediate parent directory name (e.g., `E0000043`). No laterality detection and no patient records are created (patient IDs are only implicit in `original_image_id`).

### Annotation types
- **Segmentation** (`lesions`): Binary masks for microaneurysms (`lesion_subtype="MA"`) and exudates (`lesion_subtype="EX"`). Processed with `merge_nonzero=True`.
- **Classification** (`lesion_type`, multi-class): `"microaneurysm"`, `"exudate"`, or `"normal"` derived from the classification folder name.

### Splits
No train/test splits are created. The dataset has no split definitions.

### Provenance
Mask files are registered by the segmentation processor (`process_segmentation_from_binary_mask`). Classification annotations have `raw_data_id=None` as they are derived from the folder structure.

### Special processing
- Both subsets (`e_optha_MA` and `e_optha_EX`) are processed in parallel via `asyncio.gather`.
- Images from both subsets are deduplicated by UUID before bulk upsert; if the same image appears in both subsets (e.g., an image with both MA and EX annotations), its `Image` record is written once but both segmentation and classification annotations are stored.
- Segmentations from both subsets are combined without deduplication, so one image can have both an MA segmentation and an EX segmentation.
- Upsert order: images → segmentations → classifications.
- Ingestion is idempotent due to deterministic UUID generation and upsert logic.