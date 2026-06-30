# JICHI

## Overview
The Jichi dataset is a Japanese public dataset composed of 9,939 images using an AFC-230 Nidek Fundus Camera. This dataset applies a modified Davis DR classification. In the Jichi labels, there are no descriptions of sex, age, quality control, socioeconomic aspects, or ethnicity.

## Images
- All images are in **JPEG** format.
- Images were acquired using an AFC-230 Nidek Fundus Camera.
- Images are named with the pattern `{patient_id`_{image_number}_{L|R}.jpg} (e.g., `1_1_R.jpg`, `1_2_L.jpg`, `2663_2_L.jpg`).
- The filename encodes eye laterality: `L` indicates left eye and `R` indicates right eye.
- Multiple images per patient are numbered sequentially (e.g., `4_1_R.jpg` through `4_7_L.jpg`).
- Images are stored in the `documents` folder.
## Metadata
The metadata for the dataset is stored in a single CSV file named **list.csv**, which contains image filenames and DR grading labels from two different grading perspectives.

### list.csv
The list.csv contains 9940 entries (including header) with the following columns:
- **Image** -- Filename of the fundus image following the pattern `{patient_id`_{image_number}_{L|R}.jpg}.
- **Davis_grading_of_concatenated_figures** -- DR grade assessed based on concatenated figures using the modified Davis DR classification:
- `ndr` -- No Diabetic Retinopathy
- `sdr` -- Simple Diabetic Retinopathy
- `ppdr` -- Pre-Proliferative Diabetic Retinopathy
- `pdr` -- Proliferative Diabetic Retinopathy
- **Davis_grading_of_one_figure** -- DR grade assessed based on a single figure using the modified Davis DR classification (same value options as above: `ndr`, `sdr`, `ppdr`, `pdr`).
Note that the two grading columns may differ for the same image, reflecting different assessment perspectives (concatenated figures versus single figure).

## Splits
The dataset does not provide explicit train--test splits. All images are provided in a single directory, and any data partitioning must be performed externally.
## File Schema
```
31_JICHI/
documents/
*.jpg
list.csv
```

## Storage in database

### Tables populated

- **`datasets`**: One record for the JICHI dataset (name, source_url, license, `modality_types=['fundus']`).
- **`experts`**: Two expert records are created — `"Davis_Concatenated"` (grading from concatenated figures) and `"Davis_OneFigure"` (grading from a single figure) — both with `expertise_area="DR grading"` and no associated model.
- **`grading_scales`**: A custom `"Davis_DR"` scale is registered with `disease_type="DR"`, `min_value=0`, `max_value=4`, and value labels `{0: "ndr (No DR)", 1: "sdr (Mild NPDR)", 3: "ppdr (Severe NPDR)", 4: "pdr (PDR)"}`. The standard `"ICDR_0_4"` scale is also retrieved or created.
- **`grading_scale_mappings`**: Four exact-confidence mappings are created from `Davis_DR` to `ICDR_0_4` — one per label: `0→0`, `1→1`, `3→3`, `4→4`.
- **`images`**: One row per fundus image found in `documents/`, with `original_image_id` set to the filename from `list.csv`, `modality='fundus'`, and image metadata (dimensions, format, etc.) extracted from the file via `get_image_metadata_dict`. No laterality detection is applied; laterality is encoded in the filename but not parsed by the script.
- **`expert_annotations`**: Two `ExpertAnnotation` records per image row (one per grading column), each storing the raw grade label and method (`"concatenated"` or `"one_figure"`) in `annotation_value`, linked to the appropriate expert.
- **`disease_grading`**: Two `DiseaseGrading` records per image row — one for `Davis_grading_of_concatenated_figures` (linked to expert `"Davis_Concatenated"`) and one for `Davis_grading_of_one_figure` (linked to expert `"Davis_OneFigure"`). Both use the `"Davis_DR"` scale and store the mapped integer grade value (0, 1, 3, or 4) plus the original label string. Only rows where the label is one of `{ndr, sdr, ppdr, pdr}` are stored.
- **`raw_annotation_files`**: `list.csv` is registered as a raw annotation file (file path and hash), with `annotation_type="grading"`. This is done via `process_csv`, which registers the file and sets the provenance context for all rows processed from it.
- **`provenance_chains`**: One chain is created per `process_csv` call (one for the entire `list.csv`), linking each grading and expert annotation back to the source CSV.
- **`dataset_splits`**: A single `"train"` split is created (`split_type="explicit"`).
- **`image_splits`**: All images are assigned to the `"train"` split.

### Annotation types
- **`disease_grading`**: DR grading using the custom `Davis_DR` scale (4 possible integer values: 0, 1, 3, 4). Two gradings per image — one per Davis grading method. Scale is mapped to `ICDR_0_4` for cross-dataset comparability. `annotation_method="manual"`.

### Splits created
- `train` (explicit): all images assigned here, as the dataset provides no train/test split.

### Provenance / raw annotation files registered
- `list.csv` (one raw file registration, one provenance chain for all rows).

### Special processing
- Both grading columns are processed independently from the same CSV row, producing separate `ExpertAnnotation` + `DiseaseGrading` pairs for each column.
- If the grade label is missing or not in `{ndr, sdr, ppdr, pdr}`, that column's grading is silently skipped (and a warning is logged).
- Image UUIDs are generated from the dataset UUID and the filename string from the `Image` column.
- Ingestion is idempotent; re-running the script will not create duplicate images, experts, scales, mappings, or annotations due to deterministic UUID generation and upsert logic.