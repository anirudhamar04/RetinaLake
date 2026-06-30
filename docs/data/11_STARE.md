# STARE

## Overview
The STARE (STructured Analysis of the Retina) dataset is part of a long-term project conceived and initiated in 1975 by Michael Goldbaum, M.D., at the University of California, San Diego, and funded by the U.S. National Institutes of Health. Images and associated clinical data were provided by the Shiley Eye Center at the University of California, San Diego, and the Veterans Administration Medical Center in San Diego. Over thirty contributors with backgrounds in medicine, science, and engineering participated in the project. The dataset includes raw retinal images, expert annotations, diagnosis information, and multiple forms of pixel-level labeling for retinal analysis tasks.

## Images
- All images in the dataset are stored in **PPM** format.
- The dataset includes retinal images located in the `stare-images` directory.
- Pixel-level labeled images are provided as hand-labeled pixel maps for segmentation by two experts.
- Additional PPM files are present in the `documents` directory.
- Vessel labeling images are provided as `.ah.ppm` files in `labels-ah` and `.vk.ppm` files in `labels-vk`.
- All PPM files represent pixel maps used for segmentation or labeling tasks.
## Metadata
Metadata for the STARE dataset is provided in plain text format through a diagnosis file. No CSV files are provided.

### Metadata Files
The dataset includes a single metadata file:
- **diagnosis.txt**
#### diagnosis.txt
The `diagnosis.txt` file contains diagnosis-related metadata for each image. The file does not include explicit column headers.
- **Image ID** -- The first field contains the image identifier (e.g., `im0001`).
- **Diagnosis Codes** -- One or more numeric codes representing diagnosed conditions for the image.
- **Diagnosis Keywords** -- The final field contains textual keywords describing the diagnosed retinal conditions (e.g., Background Diabetic Retinopathy, Drusen, Choroidal Neovascularization).
## Splits
Train, test, and validation splits are not provided with the dataset.

## File Schema
```
11_STARE/
annotations/
documents/
labels-ah/
labels-vk/
stare-images/
diagnosis.txt
```

## Storage in database

### Tables populated
- **`datasets`**: One record for STARE with `modality_types=["fundus"]` and `task_types=["segmentation"]`.
- **`experts`**: Two records — "Adam Hoover" (Clemson University, vessel segmentation) and "Valentina Kouznetsova" (UC San Diego, vessel segmentation).
- **`images`**: One row per image for all files discovered in both `stare-images/` and `documents/`. Fields extracted automatically via `get_image_metadata_dict` (file path, resolution, file size, etc.). `modality="fundus"`. `acquisition_date` and `image_quality` are `None`.
- **`expert_annotations`**: One record per (expert, mask file) pair — two records per annotated image (one for Adam Hoover, one for Valentina Kouznetsova), with `annotation_task="segmentation"`.
- **`segmentation_annotations`**: One record per vessel mask file. `annotation_type="vessels"`. Processed from binary PPM masks via `process_segmentation_from_binary_mask` with `fill_holes=False`. Linked to the relevant `expert_annotation` record. Only created for images in `stare-images/` (20 images × 2 experts = up to 40 segmentations).
- **`clinical_descriptions`**: One record per image that has a non-empty diagnosis text in `diagnosis.txt`. `description_type="diagnosis_text"`. Sourced from lines formatted as `{image_id}\t{codes}\t{text}`.
- **`keyword_annotations`**: One record per active (present) manifestation per image. Manifestation annotations are read from `.fea.mg.txt` files in `annotations/`. Each 42-character digit string maps position (0-based) to manifestation number (1-based); only states with a defined pathological meaning in `MANIFESTATION_STATES` are stored. Keyword term is formatted as `"{Manifestation Name}: {State Description}"` (e.g., `"Drusen: Fine, few"`). `keyword_source="clinical_description"`, `category="manifestation"`.
- **`raw_annotation_files`**: One record per source annotation file: each `.ppm` vessel mask (registered with `unified_annotation_type="segmentation"`, `file_type=None`), the `diagnosis.txt` file (`unified_annotation_type="description"`, `file_type="txt"`), and each `.fea.mg.txt` manifestation file (`unified_annotation_type="keyword"`, `file_type="txt"`).

### Image metadata extraction
Images are processed by `get_image_metadata_dict`, which reads each file to extract resolution, file size, and format. No laterality detection is performed. No patient records are created.

### Annotation types
- **Segmentation** (`vessels`): Binary masks from PPM files; holes are NOT filled (`fill_holes=False`). Expert attribution via `expert_annotations`.
- **Clinical description** (`diagnosis_text`): Free-text diagnosis strings from `diagnosis.txt`.
- **Keyword** (`manifestation` category): Pathological manifestation states from `.fea.mg.txt` files. 39 manifestation features are defined; only states with value ≥ 2 (present) and a named description in the taxonomy are stored. Non-pathological states (absent, unknown, normal) are skipped.

### Splits
No train/test splits are created. The dataset has no split definitions.

### Provenance
- Each PPM vessel mask is registered individually in `raw_annotation_files` and linked to its `segmentation_annotation` and `expert_annotation` via a provenance chain.
- `diagnosis.txt` is registered once; all `clinical_description` records share the same `raw_data_id`.
- Each `.fea.mg.txt` file is registered individually; all keyword annotations derived from that file share the same `raw_data_id`.

### Special processing
- Images in `stare-images/` (20 images) receive vessel segmentation annotations; images in `documents/` (~400 images) are registered as images only.
- Manifestation file parsing: each character in the 42-character string is treated as a digit. Position `i` (0-based) maps to manifestation number `i+1`. Only manifestation numbers present in `MANIFESTATION_STATES` and whose state value has a defined description are emitted as keyword annotations.
- Ingestion is idempotent due to deterministic UUID generation and upsert logic.