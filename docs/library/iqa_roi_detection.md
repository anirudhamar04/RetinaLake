# IQA & Fundus ROI Detection

`chaksudb/ingest/scripts/run_roi_iqa.py` is a standalone pre-compute step that enriches every
image in the database with two **pseudo-annotations** (`annotation_method='pseudo'`, so they're
distinguishable from human/consensus labels). It uses the vendored
[AutoMorph](https://github.com/rmaphoh/AutoMorph) pipeline in `external/automorph/`.

## What it computes

### 1. Image quality (IQA)

AutoMorph's **M1** EyePACS quality ensemble (8× EfficientNet-b4), run on the **M0**-cropped
fundus. The mean softmax over {good, usable, bad} gives:

- `quality_score = p_good` (0–1),
- `quality_label ∈ {good, usable, bad}` following AutoMorph's gradability rule: *good*, or
  *usable* with mean `p_bad < 0.25`, is gradable — otherwise *bad*.

Stored as a `quality_annotations` row with `quality_type='overall'` and
`scale_description='AutoMorph EyePACS QA (ensemble p_good)'`.

### 2. Fundus ROI circle

AutoMorph's **M0** fundus-mask fit (`fundus_prep.get_mask`), preceded by a geometry-preserving
Reinhard **LAB color transfer** to a reference image (ROI-only; helps the circle find the
fundus boundary on dark/badly-illuminated images instead of latching onto the bright optic
disc). The color transfer does not move pixels, so coordinates remain valid on the original.

Stored as a `localization_annotations` row with `localization_type='center_point'`,
`target_structure='fundus_roi'`, and
`coordinates={center_x, center_y, radius, method='automorph'}`.

## Running it

```bash
uv run python chaksudb/ingest/scripts/run_roi_iqa.py                      # all images
uv run python chaksudb/ingest/scripts/run_roi_iqa.py --dataset MESSIDOR   # one dataset
uv run python chaksudb/ingest/scripts/run_roi_iqa.py --datasets DRIVE CHASEDB1
uv run python chaksudb/ingest/scripts/run_roi_iqa.py --no-roi             # IQA only
uv run python chaksudb/ingest/scripts/run_roi_iqa.py --no-iqa            # ROI only
```

Both steps are **idempotent** (stable UUID v5 keys; re-running upserts). `setup_full_database.py`
runs this automatically after ingestion.

**Requirements:** the `efficientnet_pytorch` dependency and the AutoMorph submodule
(`git clone --recurse-submodules`, or `git submodule update --init`).

**ROI reference image:** the LAB color transfer needs a well-exposed reference fundus image.
Set it with `--roi-reference PATH` or the `ROI_REFERENCE_IMAGE` env var; if unset, the ROI step
silently skips normalization.

## Using IQA / ROI at export time

```python
from chaksudb.export.spec import ExportSpec

# Filter to good-quality images
spec = ExportSpec(iqa_min_quality_score=0.7)              # p_good >= 0.7
spec = ExportSpec(iqa_quality_labels=["good", "usable"])  # by label

# Add flat ROI columns for custom DataLoaders
spec = ExportSpec(include_fundus_roi=True)
# → columns: fundus_roi_cx, fundus_roi_cy, fundus_roi_radius, fundus_roi_method
```

Apply ROI masking in the built-in transform pipeline (zeros pixels outside the fundus circle,
and applies to all segmentation masks too):

```python
from chaksudb.export.transforms import FundusROIMask, SpatialCompose, Resize

spatial = SpatialCompose([Resize((512, 512)), FundusROIMask()])
```

For a custom DataLoader reading exported Parquet with `include_fundus_roi=True`:

```python
import numpy as np

cx, cy, r = row["fundus_roi_cx"], row["fundus_roi_cy"], row["fundus_roi_radius"]
h, w = img.shape[-2], img.shape[-1]
Y, X = np.ogrid[:h, :w]
img[:, (X - cx) ** 2 + (Y - cy) ** 2 > r ** 2] = 0
```

See [`transforms.md`](transforms.md) for the full transform catalog and
[`export_data_guide.md` §4.17](export_data_guide.md) for the export-time fields.
