# SUSTech-SYSU

## Overview
The SUSTech-SYSU dataset contains 1219 fundus images from Diabetic Retinopathy (DR) patients and healthy controls with annotations of exudate lesions, and four additional labels for each image: left-versus-right eye label, DR grade (severity scale) from three different grading protocols (DR grade International Clinical DR Severity Scale, DR grade American Academy of Ophthalmology, DR grade Scottish DR grading protocol), the bounding box of the optic disc and fovea location. This version of the dataset only contains the scale 0-4 of DR. Images with laser spots and scars were removed. The original dataset is available at The SUSTech-SYSU dataset.

## Images
- All images are in **JPEG** format.
- Images are named with 4-digit zero-padded numbers ranging from `0001.jpg` to `1219.jpg`.
- Image resolution is 2880×2136 pixels (from XML annotation files).
- Images are stored in the `originalImages` folder.
## Metadata
The metadata for the dataset is stored in two CSV files: **drLabels.csv** and **c5_DR_reclassified.csv**. Additionally, annotations for exudates, optic disc, and fovea locations are provided in XML format following Pascal VOC specifications.

### drLabels.csv
The drLabels.csv contains 1220 entries (including header) with the following columns:
- **Fundus_images** -- Filename of the fundus image (e.g., `0001.jpg`, `1219.jpg`).
- **left_versus_right_eye(left_0_right_1)** -- Eye laterality indicator:
- 0 -- Left eye
- 1 -- Right eye
- **DR_grade(International_Clinical_DR_Severity_Scale)** -- DR grade assessed via the International Clinical DR Severity Scale:
- 0 -- Normal healthy
- 1 -- Mild non-proliferative DR
- 2 -- Moderate non-proliferative DR
- 3 -- Severe non-proliferative DR
- 4 -- Proliferative DR
- 5 -- DR with laser spots or scars (present in original labels but images were removed/reclassified in this version)
- **DR_grade(American_Academy_of_Ophthalmology)** -- DR grade assessed via the American Academy of Ophthalmology protocol (values 0-5).
- **DR_grade(Scottish_DR_grading_protocol)** -- DR grade assessed via the Scottish DR grading protocol (values 0-5).
### c5_DR_reclassified.csv
The c5_DR_reclassified.csv contains 70 entries (including header) and provides reclassified DR labels for images that originally belonged to category 5 (DR with laser spots or scars). The CSV file contains the following columns:
- **Fundus_images** -- Filename of the fundus image (e.g., `0690.jpg`).
- **DR_grade(International_Clinical_DR_Severity_Scale)** -- Reclassified DR grade (values 2, 3, or 4).
- **DR_grade(American_Academy_of_Ophthalmology)** -- Reclassified DR grade (values 2, 3, or 4).
- **DR_grade(Scottish_DR_grading_protocol)** -- Reclassified DR grade (values 2, 3, or 4).
### XML Annotation Files
Annotations are provided in XML format following Pascal VOC specifications:
- **exudatesLabels/** -- Contains 564 XML files with exudate detection labels. Hard exudates are labeled as `ex` and soft exudates are labeled as `se`. Each annotation includes bounding box coordinates (xmin, ymin, xmax, ymax).
- **odFoveaLabels/** -- Contains 919 XML files with optic disc (OD) bounding box coordinates and fovea location coordinates. The fovea location is stored as a small box (F_x, F_y, F_{x+1}, F_{y+1}) for compatibility with LabelImg visualization tool.
## Splits
The dataset does not provide explicit train--test splits. All images are provided in a single directory, and any data partitioning must be performed externally.

## File Schema
```
30_SUSTech-SYSU/
originalImages/
*.jpg
exudatesLabels/
*.xml
odFoveaLabels/
*.xml
drLabels.csv
c5_DR_reclassified.csv
README.txt
LICENSE.txt
```

## Storage in database

### Tables populated

**`datasets`**
One record is inserted for SUSTech-SYSU with `modality_types=["fundus"]` and `license="CC-BY-4.0"`.

**`images`**
Images are pooled across all four annotation sources (two CSVs and two XML directories) via an in-memory dict (`image_filename → image_uuid`). Images are looked up in `originalImages/` by the filename from the CSV/XML; if the exact path does not exist, extension case variants (`.jpg`, `.JPG`, `.jpeg`, `.JPEG`) are tried. A new image row is created only if the filename has not been seen before in the current run. Fields stored:
- `original_image_id`: the image filename as-is (e.g., `0001.jpg`).
- `modality="fundus"`.
- No laterality is set (laterality is present in `drLabels.csv` but is not stored on the image row).

**`disease_grading`**
Up to three grading records per CSV row, one per scale. Both CSVs are processed identically:

From `drLabels.csv` (primary grading file):
- Column `DR_grade(International_Clinical_DR_Severity_Scale)` → `scale_name="ICDR_0_4"`, `disease_type="DR"`, `annotation_method="manual"`.
- Column `DR_grade(American_Academy_of_Ophthalmology)` → `scale_name="AAO"`, `disease_type="DR"`, `annotation_method="manual"`.
- Column `DR_grade(Scottish_DR_grading_protocol)` → `scale_name="Scottish"`, `disease_type="DR"`, `annotation_method="manual"`.

From `c5_DR_reclassified.csv` (reclassified grade-5 images): identical columns and scale names; produces additional grading records for the ~69 reclassified images. Both CSVs produce records linked to their own separate provenance chains.

**`localization_annotations`**
Bounding box (and small-box point) localizations from two XML directories, processed via `process_localization_from_xml` with class filtering:

- `exudatesLabels/*.xml` — filtered for classes `["ex", "exudates", "hard_exudates"]`. Produces localization records for hard exudate bounding boxes.
- `odFoveaLabels/*.xml` — filtered for classes `["OD", "fovea", "optic_disc", "macula"]`. Produces localization records for optic disc bounding boxes and fovea point annotations (stored as small bounding boxes `F_x, F_y, F_{x+1}, F_{y+1}`).

### Annotation types
- `disease_grading` — DR severity on three scales simultaneously: `ICDR_0_4` (0–4), `AAO` (0–4), and `Scottish` (0–4) from both `drLabels.csv` and `c5_DR_reclassified.csv`.
- `localization` — bounding box annotations for hard exudates (from `exudatesLabels/`) and for optic disc and fovea (from `odFoveaLabels/`).

### Splits
No splits are created. The dataset has no official train/test split and the script does not call `register_standard_splits`.

### Provenance / raw annotation files
- `drLabels.csv` is registered in `raw_annotation_files` with `unified_annotation_type="grading"`. All grading annotations derived from it are linked to this provenance chain.
- `c5_DR_reclassified.csv` is registered separately with `unified_annotation_type="grading"`. Grading annotations from this file have their own provenance chain.
- XML files in `exudatesLabels/` are registered via `process_folder_tree` with `unified_annotation_type="localization"`.
- XML files in `odFoveaLabels/` are registered via `process_folder_tree` with `unified_annotation_type="localization"`.

### Special processing
- Image deduplication across all four annotation sources is maintained via the in-memory `image_id_map` dict. An image first encountered in the DR grading CSV will not be re-created when the same filename is referenced from an XML file.
- The four annotation stages are processed sequentially: `drLabels.csv` → `c5_DR_reclassified.csv` → `exudatesLabels/` XMLs → `odFoveaLabels/` XMLs.
- The `left_versus_right_eye` column in `drLabels.csv` is present but is not stored as `eye_laterality` on the image row.
- All gradings, then all localizations, are bulk-upserted at the end in separate calls.