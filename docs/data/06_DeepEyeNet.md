# DeepEyeNet

## Overview
DeepEyeNet is a large-scale multi-modal retinal imaging dataset constructed for the investigation and benchmarking of machine vision–large language models in clinical ophthalmology, with a primary focus on automatic medical image captioning. The dataset contains a total of 15,710 retinal images, each paired with exactly one expert-written free-form clinical description and an associated set of diagnostic keywords. The dataset is designed to support tasks such as retinal image captioning, multi-label disease classification, feature fusion across modalities, and zero-shot keyword expansion. DeepEyeNet is characterized by a dual annotation schema consisting of unstructured captions and structured diagnostic keywords and has been used as a benchmark in recent retinal captioning research.

## Images
- The dataset includes retinal images from multiple imaging modalities, primarily color fundus photography and fluorescein angiography.
- Color fundus photography images account for approximately 13,898 images (88.55%).
- Fluorescein angiography images account for approximately 1,811 images (11.53%).
- Optical Coherence Tomography (OCT) images and multi-modality image grids are reported as present, but their image counts and proportions are not specified.
- All images are preprocessed to a fixed spatial resolution of **356 × 356 pixels** and in **JPG** format.
- Images are stored with three color channels (RGB).
- Original acquisition device, native resolution, bit-depth, camera model, and field-of-view information are not provided.
- Image filenames correspond to relative paths such as `eyenet0420/train_set/filename.jpg`, as referenced within the metadata JSON files.
## Metadata
Metadata for DeepEyeNet is provided in the form of JSON files corresponding to the training, validation, and test splits. Each JSON file maps image file paths to annotation objects containing diagnostic keywords and a clinical description.

### Metadata Files
The dataset includes three metadata files in JSON format:
- **DeepEyeNet_train.json**
- **DeepEyeNet_valid.json**
- **DeepEyeNet_test.json**
All metadata files follow the same nested JSON structure. At the top level, each file contains entries indexed by the relative file paths of retinal images (e.g., `eyenet0420/train_set/group41-174.jpg`). Each top-level entry corresponds to a single image and stores its associated annotations.

For every image, the annotation object contains the following fields:
- **keywords** -- A string containing a comma-separated list of diagnostic keywords associated with the image. The number of keywords per image typically ranges from 5 to 10, with some images containing up to 15 keywords. During preprocessing, keyword strings may be tokenized using a special `[SEP]` delimiter. All keywords are drawn from a controlled vocabulary of 609 unique terms mapped to 265 retinal diseases.
- **clinical-description** -- A free-form, expert-written clinical caption describing the retinal image. Captions generally contain 5--10 words, with a maximum length of 50 words. The description does not follow a fixed template and may include demographic or contextual information if provided by the annotator.
An example illustrating the metadata format is shown below:
```
{
 "eyenet0420/train_set/group41-174.jpg": {
 "keywords": "macular hole",
 "clinical-description": "43-year-old female, macular hole,"
 },
 "eyenet0420/train_set/group41-177.jpg": {
 "keywords": "macular hole",
 "clinical-description": "43-year-old female, macular hole,"
 }
}
```

## Splits
The dataset is explicitly divided into training, validation, and test splits using a 60/20/20 ratio. This results in approximately 9,426 training images, 3,142 validation images, and 3,142 test images. The splitting procedure is presumed to be random. No information regarding stratification by disease, modality, or patient is provided.

## File Schema
```
06_DEN/
eyenet0420/
test_set/
train_set/
val_set/
DeepEyeNet_test.json
DeepEyeNet_train.json
DeepEyeNet_valid.json
```

## Storage in database

### Tables populated

- **`datasets`** — One row registered with `dataset_name="DeepEyeNet"`, `source_url`, `license="Unknown"`, and `modality_types=["fundus", "fa", "oct"]`.
- **`images`** — One row per image. Fields populated via `get_image_metadata_dict` plus `original_image_id` set to the image filename (e.g., `group41-174.jpg`). The `modality` field is auto-detected per image (not globally defaulted) using the `detect_modality` function, which inspects the combined keywords and clinical-description text for modality indicator terms:
  - Returns `"fa"` if any of: `"fluorescein angiogra"`, `" fa "`, `"angiogram"`, `"angiography"` appear.
  - Returns `"oct"` if any of: `"optical coherence tomography"`, `" oct "`, `"oct scan"`, `"oct image"` appear.
  - Defaults to `"fundus"` otherwise.
- **`keyword_annotations`** — One row per keyword per image. Keywords are parsed from the `"keywords"` field of each JSON entry (comma-delimited), stored with `keyword_source="diagnostic_keywords"` and `annotation_method="manual"`. Images with empty or whitespace-only keyword strings receive no keyword rows. Keywords are individually upserted.
- **`clinical_descriptions`** — One row per image that has a non-empty `"clinical-description"` field. Stored with `description_type="clinical_caption"`, the raw text, and a computed `word_count`. Provenance (`raw_data_id`, `provenance_chain_id`) is taken from the JSON file's context. Descriptions are individually upserted.
- **`dataset_splits`** / **`image_split`** — Three explicit splits: `train` (from `DeepEyeNet_train.json`), `val` (from `DeepEyeNet_valid.json`), `test` (from `DeepEyeNet_test.json`). Each image is assigned to its corresponding split.

### Provenance / raw annotation files

Each of the three JSON files (`DeepEyeNet_train.json`, `DeepEyeNet_valid.json`, `DeepEyeNet_test.json`) is processed via `process_json`, which registers the file in `raw_annotation_files` with `annotation_type="keyword"` and creates a `provenance_chain` entry. All keyword annotations and clinical descriptions from a given JSON file reference that file's `raw_file_id` and `provenance_chain_id`.

### Image file location

Images are located under `06_DEN/eyenet0420/{split_folder}/` where `split_folder` maps as: `train` → `train_set`, `valid` → `val_set`, `test` → `test_set`. Images missing from disk are skipped with a `file_not_found` error.

### JSON entry format

Each JSON file is a list of single-key dicts: `{"eyenet0420/train_set/image.jpg": {"keywords": "...", "clinical-description": "..."}}`. The image filename is extracted from the single key; the metadata dict provides `keywords` and `clinical-description`.

### Special processing

The three JSON files are processed sequentially (not concurrently). Keywords and descriptions are individually upserted after bulk image upsert. The script is idempotent: re-running does not create duplicates due to deterministic UUID generation and upsert logic.