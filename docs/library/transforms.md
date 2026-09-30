# Transform Pipeline

ChaksuDB ships a transform pipeline (`chaksudb/export/transforms/`) designed for retinal data
where an image often comes with **masks, bounding boxes, keypoints, and ROI circles**. The key
idea: spatial transforms update *every* annotation layer consistently, so a flip or crop moves
the image and its masks/boxes/keypoints together.

Everything operates on a `SpatialSample` (image + masks + bboxes + keypoints + circles).

- **Spatial transforms** (`BaseSpatialTransform`) — geometry; update all annotation layers.
- **Photometric transforms** (`BasePhotometricTransform`) — pixel intensity; image only.
- **Composition** — `SpatialCompose([...])` and `PhotometricCompose([...])`.

For end-to-end usage with the export API and the PyTorch dataset, see
[`export_data_guide.md` §7](export_data_guide.md).

## Quick example

```python
from chaksudb.export.transforms import (
    SpatialCompose, PhotometricCompose,
    Resize, RandomHorizontalFlip, FundusROIMask,
    CLAHE, Normalize,
)

spatial = SpatialCompose([
    Resize((512, 512)),
    RandomHorizontalFlip(p=0.5),
    FundusROIMask(),          # zero everything outside the fundus circle (image + masks)
])
photometric = PhotometricCompose([
    CLAHE(clip_limit=2.0),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

## Catalog

### Spatial — geometric
`Resize`, `RandomResizedCrop`, `CenterCrop`, `RandomCrop`, `Pad`, `RandomHorizontalFlip`,
`RandomVerticalFlip`, `RandomRotation`, `RandomAffine`, `RandomRescale`, `RandomPerspective`,
`ElasticTransform`, `PolarTransform`, `FiveCrop`, `TenCrop`, `CornerPatchExtraction`,
`BoundingBoxCrop`, `ROICrop`.

### Spatial — morphological
`Erosion`, `Dilation`, `Opening`, `MorphologicalClosing`, `ConnectedComponentFiltering`.

### Spatial — retinal
`FundusROIMask` — zeros pixels outside the fundus circle carried in current-image space by the
spatial pipeline (no re-scaling); applies to all segmentation masks too. See
[`iqa_roi_detection.md`](iqa_roi_detection.md) for where the circle comes from.

### Photometric — retinal
`CLAHE`, `HistogramMatching`, `MultiscaleRetinex`, `MSRCR`, `GammaCorrection`,
`ContrastEnhancement`, `IlluminationCorrection`, `GreenChannelExtraction`, `BlueChannelEmphasis`,
`BackgroundPolynomialCorrection`.

### Photometric — denoising
`GaussianDenoising`, `MedianFiltering`, `BilateralFiltering`, `Deblurring`, `Deconvolution`.

### Photometric — torchvision wrappers
`Normalize`, `ColorJitter`, `GaussianBlur`, `Grayscale`, `RandomAdjustSharpness`,
`RandomAutocontrast`, `RandomEqualize`.

## Why spatial vs photometric matters

If you randomly rotate the image but not its segmentation mask, your labels are now wrong.
Putting geometry in `SpatialCompose` guarantees the mask, boxes, keypoints, and ROI circle
rotate with the image. Intensity-only operations (CLAHE, normalization, blur) belong in
`PhotometricCompose`, which never touches the annotation layers.
