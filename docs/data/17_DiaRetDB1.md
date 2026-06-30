# DiaRetDB1

## Overview
The DiaRetDB1 dataset is a public database designed for evaluating and benchmarking diabetic retinopathy detection algorithms. It contains digital fundus images along with expert-annotated ground truth for several diabetic fundus lesions, including hard exudates, soft exudates, microaneurysms, and hemorrhages. 
The dataset is part of the ImageRet project, and the annotations were created using the ImgAnnoTool. Each image is annotated independently by four experts to support inter-annotator agreement analysis. Both original images and raw ground truth annotations are provided, along with auxiliary Matlab functionality for reading annotations, fusing expert labels, and evaluating detection methods.

## Images
- The dataset contains **digital color fundus images**.
- A total of **89 images** are provided.
- Images are stored in **PNG** format (README also mentions PPM format).
- Each image corresponds to a single eye fundus.
- Image resolution, field of view, camera model, and acquisition parameters are not provided.
- Images are referenced in split files under an `images/` directory, but may physically reside in a `documents/` directory depending on installation.
## Metadata
The dataset does not contain CSV metadata files. Metadata and ground truth annotations are provided in **XML format**, with multiple annotation files per image.

### Metadata Files
For each fundus image, four independent expert annotations are provided as XML files in the `groundtruth/` directory.

#### XML Annotation Files
Each image has four expert annotations, identified by suffixes `_01`, `_02`, `_03`, and `_04`.
- **Regular XML files** (`*_01.xml`, etc.) which may require server connection for full functionality.
- **Plain XML files** (`*_01_plain.xml`, etc.) which are standalone and recommended for use.
Each XML file contains the following information:
- **Header** -- Creator information, software version, affiliation, and copyright.
- **Marking list** -- A collection of lesion annotations for the image.
Each marking includes:
- **Region type** -- Either `circleregion` or `polygonregion`.
- **Region geometry** --
- For `circleregion`: centroid coordinates (x, y) and radius.
- For `polygonregion`: ordered list of (x, y) coordinate pairs defining polygon vertices.
- **Representative point** -- A coordinate (x, y) indicating a representative point inside the marked region.
- **Confidence** -- Annotator confidence level, either `High` or `Medium`.
- **Marking type** -- One of the following lesion classes:
- `Hard_exudates`
- `Soft_exudates`
- `Haemorrhages`
- `Red_small_dots`
 #### Train/Test Split Files
The dataset provides explicit train and test split definition files.
- **ddb1_v02_01_train.txt** -- Training image list with paths to regular XML annotation files.
- **ddb1_v02_01_train_plain.txt** -- Training image list with paths to plain XML annotation files.
- **ddb1_v02_01_test.txt** -- Test image list with paths to regular XML annotation files.
- **ddb1_v02_01_test_plain.txt** -- Test image list with paths to plain XML annotation files.
Each line in these files contains a space-separated list consisting of the image path followed by four XML annotation file paths (one per expert).

## Splits
Explicit dataset splits are provided.
- **Training set**: 28 images.
- **Test set**: 61 images.
- No validation split is provided.
## File Schema
```
17_DiaRetDB1/
README.txt
ddb1_v02_01_train.txt
ddb1_v02_01_train_plain.txt
ddb1_v02_01_test.txt
ddb1_v02_01_test_plain.txt
visualise.ipynb
groundtruth/
diaretdb1_image001_01.xml
diaretdb1_image001_01_plain.xml
diaretdb1_image001_02.xml
diaretdb1_image001_02_plain.xml
diaretdb1_image001_03.xml
diaretdb1_image001_03_plain.xml
diaretdb1_image001_04.xml
diaretdb1_image001_04_plain.xml
documents/
diaretdb1_image001.png
diaretdb1_image002.png
images/
```

## Storage in database

### Tables populated
- **`datasets`**: One record for DiaRetDB1 with `modality_types=["fundus"]` and `license="Restricted academic license"`.
- **`experts`**: Two records — "DiaRetDB01" and "DiaRetDB02", both with `expertise_area="diabetic_retinopathy"`. Experts 03 and 04 from the dataset are intentionally skipped.
- **`images`**: One row per image referenced in `ddb1_v02_01_train_plain.txt` and `ddb1_v02_01_test_plain.txt`. `original_image_id` is the filename stem (e.g., `diaretdb1_image010`). Images are resolved from the `images/` path in the split file; if not found, the script falls back to the `documents/` directory. Fields extracted via `get_image_metadata_dict`. `modality="fundus"`.
- **`expert_annotations`**: One record per (expert, XML file) pair, with `annotation_task="localization"`.
- **`localization_annotations`**: One record per lesion marking in each XML file from experts 01 and 02. The XML parser (`process_localization_from_xml`) automatically detects the ImageRet circle/polygon format and extracts coordinates. Lesion types in the XML include `Hard_exudates`, `Soft_exudates`, `Haemorrhages`, `Red_small_dots`, plus any additional types the XML parser extracts. Each localization is linked to the expert via `expert_annotation_id`.
- **`raw_annotation_files`**: One record per XML annotation file processed (only `_01_plain.xml` and `_02_plain.xml` files). Registered with `unified_annotation_type="localization"`, `file_type="xml"`.
- **`dataset_splits`** / **`image_splits`**: Explicit `train` and `test` splits. Images from `ddb1_v02_01_train_plain.txt` → train; images from `ddb1_v02_01_test_plain.txt` → test.

### Image metadata extraction
Images are processed via `get_image_metadata_dict`. No laterality detection and no patient records are created.

### Annotation types
- **Localization** (lesion): Bounding circles and polygons for lesion regions extracted from XML files. `annotation_method="manual"`. Linked to one of the two registered experts. Lesion geometry types are `circleregion` (centroid x, y, radius) and `polygonregion` (ordered vertex list) as defined in each XML file.

### Splits
Explicit train (28 images) and test (61 images) splits read from the `_plain.txt` split files.

### Provenance
Each `_01_plain.xml` and `_02_plain.xml` file is registered individually in `raw_annotation_files`. Provenance context is set per XML file before calling the localization processor and reset after, ensuring each localization annotation is linked to the correct source file.

### Special processing
- Only experts 01 and 02 are processed. XML files with `_03_plain.xml` or `_04_plain.xml` suffixes are skipped entirely.
- Split file format: each line is `{image_path} {xml_01_path} {xml_02_path} {xml_03_path} {xml_04_path}`. Only the `_01` and `_02` XML paths are acted upon.
- All images and annotations are collected and then bulk-upserted. Images are upserted before localizations due to foreign key constraints.
- Ingestion is idempotent due to deterministic UUID generation and upsert logic.