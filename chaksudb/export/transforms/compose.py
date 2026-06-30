"""
Compose runners for spatial and photometric transform pipelines.
"""

from __future__ import annotations

from PIL import Image as PILImage

from chaksudb.export.transforms.base import BaseSpatialTransform, SpatialSample


class SpatialCompose:
    """Sequentially apply a list of spatial transforms to a SpatialSample."""

    def __init__(self, transforms: list[BaseSpatialTransform]):
        self.transforms = transforms

    def __call__(self, sample: SpatialSample) -> SpatialSample:
        for t in self.transforms:
            result = t(sample)
            if isinstance(result, list):
                raise TypeError(
                    f"{type(t).__name__} returned a list of SpatialSamples. "
                    "Multi-output transforms (FiveCrop, TenCrop, …) cannot be used "
                    "inside SpatialCompose; use MultiCropWrapper at the dataset level."
                )
            sample = result
        return sample

    def __repr__(self) -> str:
        lines = [f"  {t!r}" for t in self.transforms]
        return "SpatialCompose([\n" + ",\n".join(lines) + "\n])"


class PhotometricCompose:
    """Sequentially apply a list of photometric (image-only) transforms."""

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, image: PILImage.Image) -> PILImage.Image:
        for t in self.transforms:
            image = t(image)
        return image

    def __repr__(self) -> str:
        lines = [f"  {t!r}" for t in self.transforms]
        return "PhotometricCompose([\n" + ",\n".join(lines) + "\n])"
