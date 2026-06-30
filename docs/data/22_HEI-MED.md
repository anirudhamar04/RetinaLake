# HEI-MED

## Overview
The Hamilton Eye Institute Macular Edema Dataset (HEI-MED, formerly DMED) is a collection of 169 colour fundus images for training and evaluating algorithms that detect exudates and diabetic macular edema. Images were collected as part of a telemedicine network for diabetic retinopathy diagnosis, developed in collaboration between the Hamilton Eye Institute, the Image Science and Machine Vision Group at ORNL, and the Université de Bourgogne. Collection was completed in 2010.

All exudate areas and other bright lesions (cotton wool spots, drusen, visible fluid) were manually segmented by Dr. Edward Chaum — no distinction was made between hard and soft exudates. Machine-segmented vasculature (Zana-Klein method) is provided as a companion mask. An ELVD image quality score and optic nerve head coordinates are included per image.

## Images
- 169 JPEG images (highest-quality compression).
- Filenames follow the pattern `(NNNNNNN).jpg` (e.g. `(00000003).jpg`).
- All images are in the `DMED/` directory.
- No official train/val/test split is provided.

## Companion Files (per image stem `S`)
| File | Contents |
|------|----------|
| `(S).map.gz` | Gzip float32 exudate probability map — see format note below |
| `(S)_vess.png` | Binary vessel segmentation mask (pre-computed, Zana-Klein) |
| `(S).meta` | Tilde-delimited patient metadata |
| `(S).GND` | Text file: line 1 = exudate blob count, lines 2…N+1 = blob type ("Exudate 1") |

### .map.gz Format
The `.map.gz` files encode pixel-level exudate locations as a gzip-compressed float32 array:
- **Bytes 0–3**: NaN sentinel (ignored)
- **Bytes 4–7**: image height encoded as float32 bit-pattern (uint32 integer)
- **Bytes 8–11**: image width encoded as float32 bit-pattern (uint32 integer)
- **Bytes 12+**: `height × width` float32 values, row-major order; values are subnormal floats — **any non-zero value marks an exudate pixel**

Conversion to binary mask: `mask = (arr != 0).astype(uint8) * 255`

The `.GND` text file contains only blob type labels (all "Exudate 1") and no pixel coordinates; the spatial annotation is entirely in `.map.gz`.

### .meta Format
Tilde-delimited key–value pairs, one per line: `~Key~Value`

| Key | Description |
|-----|-------------|
| `ImageName` | Filename of the image |
| `PatientGender` | `M` or `F` |
| `PatientRace` | Race/ethnicity string |
| `PatientDOB` | Date of birth string |
| `QualityValue` | ELVD quality score (float) |
| `DiabetesType` | Diabetes type string |
| `ONrow` | Optic nerve head row coordinate (pixels) |
| `ONcol` | Optic nerve head column coordinate (pixels) |

## Annotations

### Exudate Segmentation
Derived from `.map.gz`: any non-zero pixel = exudate. Stored as `segmentation_annotations` with `annotation_type = "exudate"`, `original_format = "map.gz"`, `unified_format = "binary_mask"`. Processed PNG masks saved to `<storage_root>/HEI-MED/masks/exudate/`.

### Vessel Segmentation
Ingested directly from `_vess.png`. Stored as `segmentation_annotations` with `annotation_type = "vessel"`. `fill_holes = False` is enforced to preserve natural vessel gaps.

### Quality Annotation
`QualityValue` from `.meta` stored as `quality_annotations` with `quality_type = "image_quality"`.

### Patient Metadata
`PatientGender`, `PatientRace`, `DiabetesType` from `.meta`. Ingested as `patients` / `patient_images` where at least gender or ethnicity is present.

## Splits
No official split. All 169 images are assigned to the `train` split.

## File Schema
```
22_HEI-MED/
└── DMED/
    ├── (00000003).jpg
    ├── (00000003).map.gz
    ├── (00000003)_vess.png
    ├── (00000003).meta
    ├── (00000003).GND
    └── ...  (169 image sets × 5 files = ~848 files)
```

## Storage in Database

### Tables Populated

**`datasets`** — one record, `dataset_name = "HEI-MED"`.

**`images`** — one record per `.jpg` file (169 images).

**`segmentation_annotations`**
  - `exudate` — binary mask converted from `.map.gz`
  - `vessel` — binary mask from `_vess.png`

**`quality_annotations`** — ELVD score from `.meta`, `quality_type = "image_quality"`.

**`patients`** / **`patient_images`** — where gender or race information is available in `.meta`.

**`dataset_splits`** / **`image_split_assignments`** — all images assigned to `train`.

### Mask Converter
The `.map.gz` → binary mask conversion is implemented in `chaksudb/ingest/framework/mask_converter/gnd_maps.py`:
- `load_exudate_map_gz(path)` → `np.ndarray` (uint8, 255 = exudate)
- `parse_gnd_blob_count(path)` → `int` (number of annotated blobs from `.GND`)
- `parse_meta_file(path)` → `dict` (parsed `.meta` key-value pairs)
