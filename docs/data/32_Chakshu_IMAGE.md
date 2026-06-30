# Chákşu IMAGE

## Overview
Chákşu IMAGE is a new Indian ethnicity retinal fundus image database established for the evaluation of computer-assisted glaucoma prescreening methods. The database consists of 1345 retinal color fundus images acquired using three different devices. Five expert ophthalmologists provided the OD and optic cup (OC) ground-truth for the evaluation of segmentation performance and a binary decision (normal/glaucomatous).

## Images
- All images are stored in **JPEG** or **PNG** format, with 8 bits per color channel.
- Images were acquired using three different devices:
- Remidio non-mydriatic Fundus-on-Phone camera with a resolution of 2448×3264 pixels (1074 images: 810 in training set, 264 in test set).
- Forus 3Nethra Classic non-mydriatic fundus camera with a resolution of 2048×1536 pixels (126 images: 95 in training set, 31 in test set).
- Bosch handheld fundus camera with a resolution of 1920×1440 pixels (145 images: 104 in training set, 41 in test set).
- Most images are OD-centered and particular to the assessment of OD and glaucoma.
- Image filenames follow patterns such as `Image101.jpg-Image101-1.jpg` or `1.jpg-1-1.jpg`.
## Metadata
The metadata for the dataset is stored in CSV files located in the **6.0_Glaucoma_Decision** folder. These files contain quantitative measurements of optic disc and cup, computed clinical parameters, and glaucoma decisions from individual experts and aggregated methods.

### Expert Glaucoma Decision CSV Files
The CSV files for individual experts (e.g., `Expert 1/Bosch.csv`, `Expert 1/Forus.csv`, `Expert 1/Remidio.csv`) contain the following columns:
- **Images** -- Image identifier following the pattern `{filename`-{filename}-1.{ext}} (e.g., `Image101.jpg-Image101-1.jpg`, `1.jpg-1-1.jpg`).
- **Disc Area** -- Optic disc area in pixels.
- **Cup Area** -- Optic cup area in pixels.
- **Rim Area** -- Neuroretinal rim area in pixels.
- **Cup Height** -- Optic cup height in pixels.
- **Cup Width** -- Optic cup width in pixels.
- **Disc Height** -- Optic disc height in pixels.
- **Disc Width** -- Optic disc width in pixels.
- **ACDR** -- Area Cup-to-Disc Ratio (decimal value between 0 and 1).
- **VCDR** -- Vertical Cup-to-Disc Ratio (decimal value between 0 and 1).
- **HCDR** -- Horizontal Cup-to-Disc Ratio (decimal value between 0 and 1).
- **Glaucoma Decision** -- Binary glaucoma classification:
- `NORMAL` -- No glaucoma detected
- `GLAUCOMA SUSPECT` -- Glaucoma suspected
 ### Aggregated Glaucoma Decision CSV Files
The CSV files in `Mean/`, `Median/`, and `Majority/` subdirectories contain the same columns as expert files but with aggregated measurements (mean, median) or without the Glaucoma Decision column (for measurement-only files).

### Glaucoma Decision Comparison CSV Files
The comparison CSV files (e.g., `Glaucoma_Decision_Comparison_Bosch_majority.csv`) contain the following columns:
- **Images** -- Image identifier.
- **Expert.1** through **Expert.5** -- Individual expert glaucoma decisions (`NORMAL` or `GLAUCOMA SUSPECT`).
- **Majority Decision** -- Consensus glaucoma decision based on majority voting (`NORMAL` or `GLAUCOMA SUSPECT`).
## Splits
The entire database of 1345 fundus images is divided into training and test subsets comprising 1009 images and 336 images, respectively. The train and test subsets are approximately in the ratio of 3:1.

## File Schema
```
32_CHAKSU/
Train/
1.0_Original_Fundus_Images/
Bosch/
Forus/
Remidio/
2.0_Doctors_Annotations/
Expert 1/
Expert 2/
Expert 3/
Expert 4/
Expert 5/
3.0_Doctors_Annotations_Binary_OD_OC/
Expert 1/
Expert 2/
Expert 3/
Expert 4/
Expert 5/
4.0_OD_OC_Fusion_Images/
Expert 1/
Expert 2/
Expert 3/
Expert 4/
Expert 5/
Mean/
Median/
Majority/
STAPLE/
5.0_OD_OC_Mean_Median_Majority_STAPLE/
Bosch/
Forus/
Remidio/
6.0_Glaucoma_Decision/
Expert 1/
Expert 2/
Expert 3/
Expert 4/
Expert 5/
Glaucoma_Decision_Comparison/
Mean/
Median/
Majority/
Test/
1.0_Original_Fundus_Images/
2.0_Doctors_Annotations/
3.0_Doctors_Annotations_Binary_OD_OC/
4.0_OD_OC_Fusion_Images/
5.0_OD_OC_Mean_Median_Majority_STAPLE/
6.0_Glaucoma_Decision/
```

## Storage in database

### Tables populated

- **`datasets`**: One record for CHAKSU (name, source_url=`https://doi.org/10.6084/m9.figshare.20123135`, license=`CC-BY-4.0`, `modality_types=['fundus']`).
- **`experts`**: Five expert records, one for each annotator ("Expert 1" through "Expert 5"), all with `expertise_area="Glaucoma diagnosis and OD/OC segmentation"` and no associated model.
- **`images`**: One row per original fundus image found in `{Train|Test}/1.0_Original_Fundus_Images/{Bosch|Forus|Remidio}/`. The `original_image_id` is set to `"{camera}/{filename}"` (e.g., `"Bosch/Image101.jpg"`). `modality='fundus'`. Image metadata (dimensions, format, etc.) is extracted via `get_image_metadata_dict`. No laterality is stored.
- **`expert_annotations`**: One `ExpertAnnotation` record per expert per binary mask processed (segmentation task), and one per expert classification CSV row (classification task). Each stores the image ID, annotation type, mask file path, camera, and split in `annotation_value`.
- **`segmentation_annotations`**: Binary mask segmentations for optic disc (`optic_disc`) and optic cup (`optic_cup`), processed via `process_segmentation_from_binary_mask`. Two sets:
  - **Per-expert** (from `3.0_Doctors_Annotations_Binary_OD_OC/Expert {N}/{camera}/{Cup|Disc}/`): each linked to an `expert_annotations` record for the respective expert.
  - **Consensus** (from `5.0_OD_OC_Mean_Median_Majority_STAPLE/{camera}/{Cup|Disc}/{Majority|Mean|Median|STAPLE}/`): each linked to a `consensus_annotations` record. Consensus method is mapped to schema values: `Majority→majority_vote`, `Mean→mean`, `Median→median`, `STAPLE→staple`. Cup and Disc masks for the same image and method share a single `consensus_id`; `expert_annotation_ids` is stored as an empty list since the exact contributing annotations are not tracked. `merge_nonzero=True` is used when computing the segmentation from the binary mask.
- **`consensus_annotations`**: One record per unique (image, consensus_method) pair for segmentation, and one per image for majority-vote glaucoma classification. The `consensus_value` JSON stores the annotation types (Cup/Disc), method name, and mask file paths. `expert_annotation_ids` is empty (pre-computed masks, no direct mapping to individual annotations).
- **`classification_annotations`**: Binary `"glaucoma"` classifications (`task_type="binary"`, `class_value=True/False`). Two sets:
  - **Per-expert** (from `6.0_Glaucoma_Decision/Expert {N}/{camera}.csv`, column `"Glaucoma Decision"`): values `"GLAUCOMA SUSPECT"` or `"GLAUCOMA"` map to `True`; `"NORMAL"` maps to `False`. Linked to the respective expert via an `expert_annotations` record.
  - **Majority consensus** (from `6.0_Glaucoma_Decision/Glaucoma_Decision_Comparison_{camera}_majority.csv` or `Glaucoma_Decision_Majority_{camera}.csv`, column `"Glaucoma Decision"`): linked to a `consensus_annotations` record with `consensus_method="majority_vote"`. Mean, Median, and STAPLE consensus decisions are not ingested as classifications (only their segmentation masks are).
- **`raw_annotation_files`**: Registered automatically for each expert and consensus classification CSV via `process_csv` (one raw file per CSV), and for each binary mask directory via `process_paired_files` (one raw file per mask file or directory). Each registration creates a `provenance_chain` entry.
- **`provenance_chains`**: Created per `process_csv` call (one per CSV file) and per `process_paired_files` call. Links all segmentation and classification annotations back to their source files.
- **`dataset_splits`**: Two splits — `"train"` (from `Train/`) and `"test"` (from `Test/`) — registered as `split_type="explicit"`.
- **`image_splits`**: Each image is assigned to its corresponding split based on the directory it was found in.

### Annotation types
- **`segmentation_annotations`**: `optic_disc` and `optic_cup`, binary mask type, `annotation_method="manual"`, from 5 experts and 4 consensus methods (Majority, Mean, Median, STAPLE).
- **`classification_annotations`**: `class_name="glaucoma"`, `task_type="binary"`, from 5 experts and majority-vote consensus, `annotation_method="manual"`.

### Splits created
- `train` (explicit): images from `Train/` directory.
- `test` (explicit): images from `Test/` directory.

### Provenance / raw annotation files registered
- Per-expert classification CSVs in `6.0_Glaucoma_Decision/Expert {N}/{camera}.csv` (Train and Test).
- Majority consensus comparison CSVs in `6.0_Glaucoma_Decision/` (Train and Test).
- Binary mask files in `3.0_Doctors_Annotations_Binary_OD_OC/` and `5.0_OD_OC_Mean_Median_Majority_STAPLE/` (registered per file via `process_paired_files`).

### Special processing
- Images and masks are matched by filename stem across directories. If no image ID can be found for a mask, the mask is skipped with an error.
- `2.0_Doctors_Annotations` (raw polygon annotations) is explicitly **not** processed.
- Consensus `consensus_annotations` records are deduplicated by `consensus_id` before upserting (Cup and Disc for the same image/method share one record, with both annotation types listed in `consensus_value["annotation_types"]`).
- Ingestion is idempotent; re-running the script will not create duplicate images, experts, or annotations due to deterministic UUID generation and upsert logic.