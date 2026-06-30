# HRF (High-Resolution Fundus Image Database)

## Overview

The dataset contains two components: a segmentation dataset and an image quality assessment dataset. The segmentation dataset includes fundus images from healthy subjects, patients with diabetic retinopathy, and patients with glaucoma. The image quality assessment dataset consists of paired images of the same eye captured with different acquisition settings. 
**Important:** Although the original HRF database provides gold standard vessel segmentations and field-of-view (FOV) masks, these annotations are **not included** in the current version of the dataset and only raw fundus images are available.

## Images
- The dataset contains color retinal fundus images.
- The segmentation dataset consists of:
- 15 images of healthy eyes.
- 15 images of glaucomatous eyes.
- 15 images of eyes with diabetic retinopathy.
- Images in the segmentation dataset are stored in mixed-case formats:
- `_dr` images use **.JPG** extension.
- `_g` and `_h` images use **.jpg** extension.
- Class labels for the segmentation dataset are encoded directly in the filenames:
- `XX_h.jpg` -- healthy eyes.
- `XX_g.jpg` -- glaucomatous eyes.
- `XX_dr.JPG` -- diabetic retinopathy eyes.
- The image quality assessment dataset contains 18 image pairs of the same eye from 18 subjects.
- For each pair, one image is of poor quality and the other is of good quality.
- Images in the image quality assessment dataset were acquired using a Canon CR-1 fundus camera with a 45° field of view.
- Poor-quality images primarily exhibit decreased sharpness due to camera defocus.
- Images in the `Noise` folder use the **.JPG** extension.
## Metadata
No explicit CSV files or separate metadata files are provided with this dataset. All labels and categories are encoded directly within the image filenames and folder structure.

## Splits
No explicit training, testing, or validation splits are provided. 
## File Schema
```
23_HRF/
documents/
01_dr.JPG
01_g.jpg
01_h.jpg
\dots
15_dr.JPG
15_g.jpg
15_h.jpg
Noise/
1_bad.JPG
1_good.JPG
\dots
18_bad.JPG
18_good.JPG
```

## Storage in database

### Tables populated

**`datasets`**
One record is inserted for HRF with `modality_types=["fundus"]`. No `task_types` field is set.

**`images`**
One row per image, covering both the `Noise/` and `documents/` folders. Image metadata is extracted from the physical file. Fields stored: `original_image_id`, `modality="fundus"`. No laterality or acquisition date is set.

- Images from `Noise/` get `original_image_id` prefixed with `noise_` (e.g., `noise_5_good`).
- Images from `documents/` use the file stem directly (e.g., `05_h`, `05_dr`).

**`quality_annotations`** (for `Noise/` images)
One quality annotation per image in `Noise/`. The quality label (`"good"` or `"bad"`) is parsed from the filename suffix (`_good` / `_bad`). Stored with:
- `quality_type="overall"`
- `scale_description="HRF Noise folder quality (good vs bad)"`
- No raw file provenance (no CSV or annotation file; labels are derived from filenames).

**`classification_annotations`** (for `documents/` images)
Two binary classification annotations per image in `documents/`, one per disease type:
1. `class_name="DR"`, binary value `True` if the filename suffix is `_dr`, else `False`.
2. `class_name="glaucoma"`, binary value `True` if the filename suffix is `_g`, else `False`.

Both use `task_type="binary"` and `annotation_method="manual"`. No raw file provenance (labels are derived from filenames).

### Annotation types
- `quality` — overall image quality (good / bad) from filename, for `Noise/` images.
- `classification` — binary disease labels (DR and glaucoma) from filename, for `documents/` images. A healthy image (`_h`) yields DR=False and glaucoma=False.

### Splits
All images (from both `Noise/` and `documents/`) are assigned to a single `"train"` split of type `"explicit"`. No test or validation split is created.

### Provenance / raw annotation files
No annotation files are registered in `raw_annotation_files`. All labels are derived from the filename structure and folder organisation; there are no separate CSV or annotation files to register.

### Special processing
The two folders are processed independently:
- `Noise/` folder: filename pattern `{number}_{good|bad}.JPG` (case-insensitive). Both `.JPG` and `.jpg` extensions are accepted.
- `documents/` folder: filename pattern `{number}_{h|g|dr}.{jpg|JPG}`. The label code maps as: `h → healthy` (both DR=False, glaucoma=False), `g → glaucoma` (DR=False, glaucoma=True), `dr → diabetic_retinopathy` (DR=True, glaucoma=False).