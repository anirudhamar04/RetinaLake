# ACRIMA

## Overview
The ACRIMA dataset contains 705 fundus images, including 396 images of glaucoma and 309 normal images. These images were obtained from FISABIO Oftalmología Médica in Valencia, Spain, in compliance with the ethical standards of the Helsinki Declaration and with patient consent, and were annotated by glaucoma specialists. The database aims to provide resources for researchers in fundus image analysis, glaucoma diagnosis, and disease research. It supports the development and evaluation of glaucoma diagnostic algorithms, disease classification prediction, and the exploration of fundus image features. Additionally, ACRIMA can be used for medical education, helping professionals and students learn fundus image interpretation and diagnosis.

## Images
- All images are in **PNG** format.
- Images are 2D fundus photographs with resolution ranging from minimum [178,178] to maximum [1420,1420], with median resolution of [524,524].
- Image naming convention: Images begin with the two letters `Im` followed by a three-digit image number (ranging from 001 to 705), and ending with a label (if the image is pathological, the label is `_g`; if the image is normal, the label is `_`). All image names end with the database name `ACRIMA`.
- Glaucoma images: `Im686_g_ACRIMA.png` (example)
- Normal images: `Im001_ACRIMA.png` (example)
- The dataset contains 396 glaucoma images and 309 normal images.
## Metadata
The dataset provides split information through text files: `train.txt` and `val.txt`. These files likely contain lists of image filenames or paths for training and validation splits.

### train.txt
The `train.txt` file contains the list of images assigned to the training split. The exact format and content structure are not explicitly provided in the available documentation.

### val.txt
The `val.txt` file contains the list of images assigned to the validation split. The exact format and content structure are not explicitly provided in the available documentation.

## Splits
The dataset provides explicit train and validation splits through `train.txt` and `val.txt` files. A test split is not mentioned in the provided documentation.

## File Schema
```
36_ACRIMA/
images/
xxxx.png
xxxx.png
...
train.txt
val.txt
```

## Storage in database

### Tables populated

- **`datasets`**: One record for ACRIMA (name=`"ACRIMA"`, source_url=`https://www.kaggle.com/datasets/andrewmvd/glaucoma-detection`, license=`Research/Academic Use`, `modality_types=['fundus']`, with a description string summarizing the filename encoding and image counts).
- **`images`**: One row per image found in `G/` (glaucoma) and `noG/` (normal) folders. `original_image_id` is the file stem (no extension). `modality='fundus'`. Image metadata (dimensions, format, etc.) is extracted via `get_image_metadata_dict`. Image UUIDs are generated from the dataset UUID and the file stem. Supported extensions: `.png`, `.PNG`, `.jpg`, `.JPG`.
- **`classification_annotations`**: One `ClassificationAnnotation` per image with `class_name="glaucoma"`, `task_type="binary"`. `class_value` is `True` (glaucoma) or `False` (normal). `annotation_method="manual"`. No `class_labels` dict is stored. No `expert_annotation_id`.
  - **Classification source**: The label is parsed from the filename by `parse_classification_from_filename` — images with `"_g_"` in the filename are classified as glaucoma (`True`); images with `"_ACRIMA"` in the filename (but without `"_g_"`) are classified as normal (`False`). If neither pattern matches, the image is skipped with an error.
- **`raw_annotation_files`**: The entire data root directory is registered once as a raw annotation file via `register_mask_directory` (representing the folder structure that encodes the annotations). This creates a single provenance chain used for all images and classifications.
- **`provenance_chains`**: One chain created for the entire ingestion, set via `set_provenance_context` before processing begins and reset in a `finally` block afterward. All images and classifications share this provenance.
- **`dataset_splits`**: A single `"train"` split (`split_type="explicit"`).
- **`image_splits`**: All images assigned to the `"train"` split.

### Annotation types
- **`classification_annotations`**: `class_name="glaucoma"`, `task_type="binary"` (True=glaucoma, False=normal). Label extracted from filename pattern (`"_g_"` present = glaucoma), not from a separate annotation file.

### Splits created
- `train` (explicit): all images (from both `G/` and `noG/` folders). The `train.txt` and `val.txt` files present in the dataset are **not** used to define splits; the script assigns everything to a single train split.

### Provenance / raw annotation files registered
- One raw annotation file record representing the data root directory (registered via `register_mask_directory` with `unified_annotation_type="classification"`).

### Special processing
- Images are processed by iterating over two folders separately: `G/` (glaucoma) then `noG/` (normal). Each folder's images are found via `find_images` (non-recursive).
- The `G/` and `noG/` folder placement is informational context for the script; the actual label stored is determined solely by parsing the filename, not the folder.
- Images whose filenames match neither the glaucoma pattern (`_g_`) nor the normal pattern (`_ACRIMA`) are skipped with a filename-parsing error.
- The ingestion process is idempotent. Re-running it will not create duplicate images or annotations due to deterministic UUID generation and upsert logic.