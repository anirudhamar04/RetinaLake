# OIA-DDR

## Overview
The images are classified into six classes: five classes according to DR severity (none, mild, moderate, severe, and proliferative DR) and one class for ungradable images with poor quality. All images have been preprocessed to delete the black background. 
The dataset supports three tasks: DR grading (image-level annotations), lesion detection (bounding box annotations), and lesion segmentation (pixel-level annotations). Among the 13,673 images, 757 images with DR (severity levels 1-4) are annotated with four types of DR-related lesions at both pixel-level and bounding-box level.

## Images
- All images are in **JPEG** format.
- Images are retinal fundus photographs captured using 42 types of fundus cameras with a 45-degree field of view (FOV), mainly including Topcon D7000, Topcon TRC NW48, Nikon D5200, and Canon CR 2 cameras.
- Images are captured as single-view images with the centre located at the middle point of the line connecting the optic disc and the fovea.
- All images have been preprocessed to remove black backgrounds.
- Images are labeled with one of two naming conventions:
- Timestamp format: `YYYYMMDDHHMMSS.jpg` (e.g., `20170413102628830.jpg`)
- ID format: `007-XXXX-XXX.jpg` (e.g., `007-2809-100.jpg`)
- Image resolution varies as images were captured using different fundus cameras.
- Eye laterality (left/right) information is not explicitly encoded in the filename or provided metadata.
## Metadata
The metadata for DR grading is stored in text files (`test.txt`, `train.txt`, `valid.txt`) with format: `filename.jpg DR_grade`. 
Lesion detection annotations are provided in XML files following Pascal VOC format, containing bounding box coordinates for detected lesions. 
Lesion segmentation masks are provided as **TIF** files organized by lesion type in subdirectories.

### test.txt, train.txt, valid.txt
The text files contain DR grading labels for each image in the respective split.
- **filename** - Image filename (e.g., `20170413102628830.jpg` or `007-2809-100.jpg`)
- **DR_grade** - Diabetic Retinopathy severity grade (0-5):
- 0 - No DR
- 1 - Mild nonproliferative DR
- 2 - Moderate nonproliferative DR
- 3 - Severe nonproliferative DR
- 4 - Proliferative DR
- 5 - Ungradable (poor quality images)
 Grading follows the International Classification of Diabetic Retinopathy standard. Each image was graded by approximately four trained graders, with final classification determined by majority voting.
### Lesion Detection XML Files
The XML files contain bounding box annotations for diabetic retinopathy lesions in Pascal VOC format. Bounding boxes are automatically generated from pixel-level segmentation annotations.
- **filename** - Corresponding image filename
- **size** - Image dimensions (width, height, depth)
- **object** - Lesion annotations with the following fields:
- **name** - Lesion type: `ma` (microaneurysms), `ex` (hard exudates), `se` (soft exudates/cotton wool spots), or `he` (hemorrhages)
- **bndbox** - Bounding box coordinates (`xmin`, `ymin`, `xmax`, `ymax`)
 ### Lesion Segmentation Masks
Segmentation masks are provided as **TIF** files organized by lesion type in subdirectories. The 757 images with DR (severity levels 1-4) are annotated with four types of lesions:
- **MA/** - Microaneurysm segmentation masks
- **EX/** - Hard exudates segmentation masks
- **SE/** - Soft exudates segmentation masks
- **HE/** - Hemorrhage segmentation masks
Each lesion type is stored in a separate subdirectory, and masks are saved as TIF files.

## Splits
The dataset provides explicit train, validation, and test splits for all three tasks. The splits are randomly selected with approximately 50% training, 20% validation, and 30% testing:
- **DR Grading**: Train (6,835 images), Validation (2,733 images), Test (4,105 images)
- **Lesion Detection**: Train, Test, and Valid splits (757 XML files total)
- **Lesion Segmentation**: Train, Test, and Valid splits with TIF mask files organized by lesion type
## File Schema
```
28_OIA-DDR/
DR_grading/
test/
train/
valid/
test.txt
train.txt
valid.txt
lesion_detection/
test/
train/
valid/
lesion_segmentation/
test/
label/
train/
image/
label/
EX/
HE/
MA/
SE/
valid/
image/
segmentation label/
```

## Storage in database

### Tables populated

**`datasets`**
One record is inserted for OIA-DDR with `modality_types=["fundus"]` and `license="CC-BY-4.0"`.

**`images`**
Images are pooled across all three tasks (DR grading, lesion detection, lesion segmentation) and all three splits (train, test, valid). A global in-memory registry keyed by filename ensures that the same physical image referenced in multiple tasks or splits yields only a single database row. Image metadata is extracted from the physical file. Fields stored:
- `original_image_id`: the image filename (e.g., `20170413102628830.jpg`).
- `modality="fundus"`.
- No laterality or acquisition date is set.

Image lookup order for localization (when image is not already registered from grading): tries `DR_grading/<split>/<filename>` first, then `lesion_segmentation/<split>/image/<filename>`.

**`disease_grading`**
One DR grading annotation per line in each split's `.txt` file (`DR_grading/train.txt`, `DR_grading/test.txt`, `DR_grading/valid.txt`). Format per line: `filename grade`. Fields stored:
- `disease_type="DR"`, `scale_name="ICDR_0_5"` (0–5 integer), `annotation_method` left at default.
- Provenance linked to the respective `.txt` file registered as a raw annotation file.

**`localization_annotations`**
Bounding box localizations extracted from Pascal VOC XML files in `lesion_detection/{train,test,valid}/`. One or more localization records per XML file, one per `<object>` element. Processed via `process_localization_from_xml`. Provenance linked to the per-folder `raw_annotation_files` entry registered by `process_folder_tree`.

**`segmentation_annotations`**
Binary mask segmentations for up to four lesion types per image in `lesion_segmentation/{train,test,valid}/image/`. For each image, the script checks for mask files in `lesion_segmentation/<split>/label/{EX,HE,MA,SE}/<stem>.tif`. Each found mask is processed via `process_segmentation_from_binary_mask` with:
- `annotation_type="lesions"`, `lesion_subtype` set to the lesion folder name (`EX`, `HE`, `MA`, or `SE`).
- `annotation_method="manual"`.
- Each mask TIF file is individually registered in `raw_annotation_files`.

### Annotation types
- `disease_grading` — DR severity on the `ICDR_0_5` scale (0=no DR, 1=mild, 2=moderate, 3=severe, 4=proliferative, 5=ungradable).
- `localization` — bounding boxes for lesions (microaneurysms `ma`, hard exudates `ex`, soft exudates `se`, hemorrhages `he`) from Pascal VOC XML files.
- `segmentation` — binary pixel masks for lesion types EX (hard exudates), HE (hemorrhages), MA (microaneurysms), SE (soft exudates) from TIF files.

### Splits
Three splits are registered with `split_type="explicit"`:
- `"train"`, `"test"`, `"val"` (the dataset folder is named `valid` but the split is registered as `val`).

An image may appear in multiple splits if it is referenced in both the grading and segmentation/detection tasks across different split folders; the script records all applicable split memberships for each image UUID.

### Provenance / raw annotation files
- Each grading `.txt` file (`train.txt`, `test.txt`, `valid.txt`) is individually registered in `raw_annotation_files` with `unified_annotation_type="grading"`, `file_type="txt"`. A provenance context is set/reset around each file's processing.
- Lesion detection XML files are registered via `process_folder_tree` (per-folder registration), `unified_annotation_type="localization"`.
- Each lesion segmentation TIF mask is individually registered with `unified_annotation_type="segmentation"`, `auto_detect_type=False`.

### Special processing
- The three tasks (grading, detection, segmentation) are processed in sequence, not in parallel.
- Image deduplication across tasks is maintained via an in-memory `existing_image_index` dict (filename → UUID). An image processed first by the grading task will not be re-created when the segmentation task encounters the same filename.
- Images are bulk-upserted first; gradings and localizations are bulk-upserted concurrently (via `asyncio.gather`); segmentations are upserted individually (no bulk operation available).