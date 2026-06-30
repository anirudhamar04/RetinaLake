# EYEPACS
## Overview
The EyePACS dataset consists of retinal images originally published in the Kaggle competition "Diabetic Retinopathy Detection". 
The dataset contains high resolution fundus images taken under various imaging condition. Each subject has a right and left field present.
## Images
The images in this dataset are of both eyes of a subject from different models and types of cameras. 
- All images are in **jpeg** format.
- Images are labeled with a subject id as well as either left or right (e.g. 1_left.jpeg is the left eye of patient id 1).
- Some images are shown as one would see the retina anatomically (macula on the left, optic nerve on the right for the right eye). Others are shown as one would see through a microscope condensing lens (i.e. inverted, as one sees in a typical live eye exam). Read more on the [Kaggle Page](https://www.kaggle.com/competitions/diabetic-retinopathy-detection/data)
## Metadata
The metadata for each image in the dataset is concisely stored for both train and test data in each of their respective train.csv and test.csv
### train.csv
The train.csv contains 2 coloumn entries namely image, level. 
- **Image** - Contains the Image ID as subject id as well as either left or right (e.g. 1_left is the left eye of patient id 1)
- **Level**- Contains the clinically diagnosed level of DR from 0-4 on the scale given below :
- 0 - No DR
- 1 - Mild
- 2 - Moderate
- 3 - Severe
- 4 - Proliferative DR
 ### test.csv
The test.csv contains 3 coloumn entries namely image, level, Usage 
- **Image** - Same as train.csv
- **Level** - Same as train.csv
- **Usage** - Has 2 values Private or Public. Not sure what they mean
## Splits
Explicit splits of Test and Training
## File Schema
```
01_EYEPACS/
test/
train/
test.csv
train.csv
```

## Storage in database

### Tables populated

- **`datasets`** — One row registered with `dataset_name="EYEPACS"`, `source_url`, `license="CC-BY-4.0"`, and `modality_types=["fundus"]`.
- **`images`** — One row per image. Fields populated via `get_image_metadata_dict` (file size, resolution, format, etc.) plus `modality="fundus"` and `original_image_id` set to the image filename stem (e.g., `1_left`). Eye laterality (`eye_laterality`) is extracted from the filename using the regex `_(left|right)$`; if neither suffix matches, `eye_laterality` is `None`.
- **`disease_grading`** — One row per image. The `level` column from `train.csv` / `test.csv` is stored as the grade value under scale `ICDR_0_4` (0=No DR, 1=Mild, 2=Moderate, 3=Severe, 4=Proliferative DR). `annotation_method` is set to `"manual"`.
- **`dataset_splits`** / **`image_split`** — Two explicit splits are created: `train` (images from `train/` directory and `train.csv`) and `test` (images from `test/` directory and `test.csv`). Each image is assigned to its corresponding split.

### Provenance / raw annotation files

Both `train.csv` and `test.csv` are processed via `process_csv`, which registers each file in `raw_annotation_files` (with file path, hash, and `annotation_type="grading"`) and creates a corresponding `provenance_chain` entry. Every `disease_grading` row references the `raw_file_id` and `provenance_chain_id` from the CSV that sourced it.

### Image metadata extraction

Image metadata (resolution, file size, format) is extracted automatically from each image file using `get_image_metadata_dict`. The image file is located by searching for the stem name (from the CSV `image` column) with `.jpeg`, `.jpg`, or `.png` extensions under the `train/` or `test/` subdirectory. Laterality is parsed from the filename suffix (`_left` or `_right`).

### Special processing

Both CSVs are processed concurrently with `asyncio.gather`. Images are bulk-upserted first (respecting foreign key constraints), then gradings are bulk-upserted in batches of 1000. The script is idempotent: re-running does not create duplicates due to deterministic UUID generation and upsert logic.