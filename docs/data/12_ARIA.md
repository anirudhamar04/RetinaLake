# ARIA

## Overview
The Automated Retinal Image Analysis (ARIA) dataset is an ophthalmic imaging dataset consisting of retinal fundus images and corresponding segmentation masks. 
The dataset is organized into three distinct subsets and is intended for vessel segmentation tasks, with explicit vessel markup provided.

## Images
- The dataset consists of retinal fundus images and associated segmentation masks.
- All images are stored in **TIFF** (**.tif**) format.
- The dataset is a segmentation dataset with vessel annotations.
- Fundus images are stored separately from their corresponding vessel markup.
- The dataset is a segmentation dataset consisting of retinal fundus images and their corresponding masks.
- Fundus images are stored in directories named `aria_a_markup`, `aria_b_markup`, and `aria_c_markup`.
- Vessel segmentation masks are provided in directories named `aria_a_markup_vessels`, `aria_b_markup_vessels`, and `aria_c_markup_vessels`.
- Masks corresponding to the disc and fovea region are stored in directories named `aria_a_markupdiscfovea`, `aria_b_markupdiscfovea`, and `aria_c_markupdiscfovea`.
 

## Metadata
No structured metadata files such as CSV or JSON files are provided with this dataset.

## Splits
The dataset does not provide explicit training, testing, or validation splits. The images are grouped into three separate subsets, but no further usage protocol is specified.

## File Schema
```
12_ARIA/
aria_a_markup_vessel/
aria_a_markups/
aria_c_markup_vessel/
aria_c_markupdiscfovea/
aria_c_markups/
aria_d_markup_vessel/
aria_d_markupdiscfovea/
aria_d_markups/
```

## Storage in database

### Tables populated
- **`datasets`**: One record for ARIA with `modality_types=["fundus"]` and `task_types=["segmentation"]`.
- **`experts`**: Two records — "Expert BDP" and "Expert BSS", both with `expertise_area="Blood Vessel Segmentation"` and `affiliation="ARIA Project"`.
- **`images`**: One row per image file found in each `aria_{subset}_markups/` directory (subsets: `a`, `c`, `d`). Fields extracted via `get_image_metadata_dict`. `modality="fundus"`. The subset key (`"a"`, `"c"`, or `"d"`) is stored as `comorbidities={"subset": subset_key}`. `acquisition_date` and `image_quality` are `None`. The base image identifier (with prefix like `(0001)` and expert/disc-fovea suffixes stripped) is used as `original_image_id`.
- **`expert_annotations`**: One record per (expert, vessel mask file) pair, with `annotation_task="segmentation"`.
- **`segmentation_annotations`** (vessel): One record per vessel mask file found in `aria_{subset}_markup_vessel/`. Mask filenames contain `_BDP` or `_BSS` suffixes to identify the expert. Processed via `process_segmentation_from_soft_map` (grayscale intensity map). `annotation_type="vessels"`. Linked to the appropriate expert via `expert_annotation_id`. Available for all three subsets.
- **`segmentation_annotations`** (disc/fovea): One record per mask file found in `aria_{subset}_markupdiscfovea/` for subsets `c` and `d` only (subset `a` has no disc/fovea annotations). Processed via `process_segmentation_from_binary_mask` with `fill_holes=False`. `annotation_type="optic_disc_and_fovea"`. `expert_annotation_id=None` (no per-expert attribution for these masks). Suffix `_dfs` or `_dfd` is stripped from filenames when matching to images.
- **`raw_annotation_files`**: One record per TIFF mask file — for vessel masks (registered with `unified_annotation_type="segmentation"`, `file_type=None`) and disc/fovea masks (same).

### Image metadata extraction
Images are processed via `get_image_metadata_dict`. The identifier is extracted by stripping leading numeric prefixes (e.g., `(0001)`) and trailing expert suffixes (`_BDP`, `_BSS`) or disc/fovea suffixes (`_dfs`, `_dfd`). No laterality detection is performed. No patient records are created.

### Annotation types
- **Segmentation** (`vessels`): From grayscale TIFF soft maps in `aria_{subset}_markup_vessel/`. Processed as soft (intensity) maps, one per expert per image. Expert identified from `_BDP` or `_BSS` filename suffix.
- **Segmentation** (`optic_disc_and_fovea`): From binary TIFF masks in `aria_{subset}_markupdiscfovea/`. Combined mask for both optic disc and fovea. Only present in subsets `c` and `d`.

### Splits
No train/test splits are created. The dataset has no split definitions.

### Provenance
Each TIFF vessel or disc/fovea mask file is registered individually in `raw_annotation_files` and linked to its `segmentation_annotation` (and `expert_annotation` for vessel masks) via a provenance chain.

### Special processing
- Subset `a` has vessel annotations only; subsets `c` and `d` have both vessel and disc/fovea annotations.
- Image identifier extraction uses regex to remove the `(NNNN)` prefix and `_BDP`/`_BSS`/`_dfs`/`_dfd` suffixes before matching masks to images.
- Ingestion is idempotent due to deterministic UUID generation and upsert logic.