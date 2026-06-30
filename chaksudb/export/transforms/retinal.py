"""
Retinal-specific intensity transforms.

These are the only transforms where we write actual image math, because no
existing library provides them as callable transforms.  Each class stores
parameters in ``__init__`` and processes a PIL Image in ``__call__``.
"""

from __future__ import annotations

import math
import random
from typing import Optional, Sequence

import cv2
import numpy as np
from PIL import Image as PILImage

from chaksudb.export.transforms.base import BasePhotometricTransform, BaseSpatialTransform, SpatialSample


def _pil_to_bgr(image: PILImage.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)


def _bgr_to_pil(arr: np.ndarray) -> PILImage.Image:
    return PILImage.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


def _pil_to_rgb(image: PILImage.Image) -> np.ndarray:
    return np.array(image.convert("RGB"))


def _rgb_to_pil(arr: np.ndarray) -> PILImage.Image:
    return PILImage.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


class CLAHE(BasePhotometricTransform):
    """Contrast Limited Adaptive Histogram Equalisation on the L channel (LAB)."""

    def __init__(self, clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, image: PILImage.Image) -> PILImage.Image:
        bgr = _pil_to_bgr(image)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return _bgr_to_pil(bgr)

    def __repr__(self) -> str:
        return f"CLAHE(clip_limit={self.clip_limit}, tile_grid_size={self.tile_grid_size})"


class HistogramMatching(BasePhotometricTransform):
    """Match image histogram to a reference image using skimage."""

    def __init__(self, reference_image: np.ndarray | PILImage.Image | None = None):
        if reference_image is not None and isinstance(reference_image, PILImage.Image):
            reference_image = _pil_to_rgb(reference_image)
        self.reference_image = reference_image

    def __call__(self, image: PILImage.Image) -> PILImage.Image:
        if self.reference_image is None:
            return image
        from skimage.exposure import match_histograms
        src = _pil_to_rgb(image)
        matched = match_histograms(src, self.reference_image, channel_axis=-1)
        return _rgb_to_pil(matched)

    def __repr__(self) -> str:
        return f"HistogramMatching(reference_image={'set' if self.reference_image is not None else None})"


class MultiscaleRetinex(BasePhotometricTransform):
    """Multi-Scale Retinex (MSR) for illumination normalisation."""

    def __init__(self, scales: Sequence[float] = (15, 80, 250)):
        self.scales = list(scales)
        if not self.scales:
            raise ValueError("scales must be a non-empty sequence of positive numbers")

    def __call__(self, image: PILImage.Image) -> PILImage.Image:
        img = _pil_to_rgb(image).astype(np.float64) + 1.0
        retinex = np.zeros_like(img)
        for sigma in self.scales:
            ksize = max(3, int(6 * sigma + 1)) | 1
            blur = cv2.GaussianBlur(img, (ksize, ksize), sigma)
            retinex += np.log(img) - np.log(blur + 1.0)
        retinex /= len(self.scales)
        retinex = (retinex - retinex.min()) / (retinex.max() - retinex.min() + 1e-8) * 255
        return _rgb_to_pil(retinex)

    def __repr__(self) -> str:
        return f"MultiscaleRetinex(scales={self.scales})"


class MSRCR(BasePhotometricTransform):
    """Multi-Scale Retinex with Colour Restoration."""

    def __init__(
        self,
        scales: Sequence[float] = (15, 80, 250),
        gain: float = 128.0,
        offset: float = 128.0,
    ):
        self.scales = list(scales)
        if not self.scales:
            raise ValueError("scales must be a non-empty sequence of positive numbers")
        self.gain = gain
        self.offset = offset

    def __call__(self, image: PILImage.Image) -> PILImage.Image:
        img = _pil_to_rgb(image).astype(np.float64) + 1.0
        retinex = np.zeros_like(img)
        for sigma in self.scales:
            ksize = max(3, int(6 * sigma + 1)) | 1
            blur = cv2.GaussianBlur(img, (ksize, ksize), sigma)
            retinex += np.log(img) - np.log(blur + 1.0)
        retinex /= len(self.scales)

        intensity = np.sum(img, axis=2, keepdims=True) + 1e-8
        color_restoration = np.log(125.0 * img / intensity)
        result = self.gain * (retinex * color_restoration) + self.offset
        result = np.clip(result, 0, 255)
        return _rgb_to_pil(result)

    def __repr__(self) -> str:
        return f"MSRCR(scales={self.scales}, gain={self.gain}, offset={self.offset})"


class GammaCorrection(BasePhotometricTransform):
    """Random gamma correction within a specified range."""

    def __init__(self, gamma_range: tuple[float, float] = (0.5, 2.0)):
        self.gamma_range = gamma_range

    def __call__(self, image: PILImage.Image) -> PILImage.Image:
        gamma = random.uniform(*self.gamma_range)
        img = _pil_to_rgb(image).astype(np.float64) / 255.0
        corrected = np.power(img, gamma) * 255.0
        return _rgb_to_pil(corrected)

    def __repr__(self) -> str:
        return f"GammaCorrection(gamma_range={self.gamma_range})"


class ContrastEnhancement(BasePhotometricTransform):
    """Dispatch to CLAHE, gamma, or histogram equalisation by method name."""

    def __init__(self, method: str = "clahe", **params):
        self.method = method
        self.params = params
        if method == "clahe":
            self._fn = CLAHE(**params)
        elif method == "gamma":
            self._fn = GammaCorrection(**params)
        elif method == "equalize":
            from torchvision.transforms import RandomEqualize
            self._fn = RandomEqualize(p=1.0)
        else:
            raise ValueError(f"Unknown contrast enhancement method: {method!r}")

    def __call__(self, image: PILImage.Image) -> PILImage.Image:
        return self._fn(image)

    def __repr__(self) -> str:
        return f"ContrastEnhancement(method={self.method!r}, params={self.params})"


class IlluminationCorrection(BasePhotometricTransform):
    """Estimate background illumination with Gaussian blur and subtract."""

    _SUPPORTED_METHODS = {"gaussian"}

    def __init__(self, method: str = "gaussian", degree: int = 51):
        if method not in self._SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported IlluminationCorrection method {method!r}; "
                f"choose from {sorted(self._SUPPORTED_METHODS)}"
            )
        self.method = method
        self.degree = degree

    def __call__(self, image: PILImage.Image) -> PILImage.Image:
        img = _pil_to_rgb(image).astype(np.float64)
        ksize = self.degree | 1
        background = cv2.GaussianBlur(img, (ksize, ksize), 0)
        corrected = img - background + 128.0
        return _rgb_to_pil(corrected)

    def __repr__(self) -> str:
        return f"IlluminationCorrection(degree={self.degree})"


class GreenChannelExtraction(BasePhotometricTransform):
    """Extract the green channel (index 1) as a single-channel image."""

    def __call__(self, image: PILImage.Image) -> PILImage.Image:
        arr = _pil_to_rgb(image)
        green = arr[:, :, 1]
        return PILImage.fromarray(green, "L")

    def __repr__(self) -> str:
        return "GreenChannelExtraction()"


class BlueChannelEmphasis(BasePhotometricTransform):
    """Weighted channel combination emphasising the blue channel."""

    def __init__(self, weight: float = 1.5):
        self.weight = weight

    def __call__(self, image: PILImage.Image) -> PILImage.Image:
        arr = _pil_to_rgb(image).astype(np.float64)
        arr[:, :, 2] *= self.weight
        return _rgb_to_pil(arr)

    def __repr__(self) -> str:
        return f"BlueChannelEmphasis(weight={self.weight})"


class FundusROIMask(BaseSpatialTransform):
    """Zero out pixels outside the fundus circle boundary.

    Looks for a ``target_structure == 'fundus_roi'`` circle in
    ``SpatialSample.circles`` (populated from localization_annotations stored
    by ``run_roi_iqa.py``). Circle coordinates are in the *original* image
    space and are scaled to match the current image size at call-time so this
    transform is safe to use after any resize.

    The same circular mask is applied to every segmentation mask in
    ``sample.masks`` so downstream tasks stay geometrically consistent.

    Args:
        fill_value: Pixel value for masked-out regions (default 0).
        fallback_full_image: If True and no fundus_roi circle is found,
            return the sample unchanged. If False, raise ValueError.
    """

    def __init__(self, fill_value: int = 0, fallback_full_image: bool = True) -> None:
        self.fill_value = fill_value
        self.fallback_full_image = fallback_full_image

    def __call__(self, sample: SpatialSample) -> SpatialSample:
        roi_circle = next(
            (c for c in sample.circles if c.get("target_structure") == "fundus_roi"),
            None,
        )
        if roi_circle is None:
            if self.fallback_full_image:
                return sample
            raise ValueError(
                "FundusROIMask: no 'fundus_roi' circle found in SpatialSample.circles. "
                "Run run_roi_iqa.py to ingest ROI annotations, and include 'localization' "
                "in annotation_tasks when building the ExportSpec."
            )

        cur_w, cur_h = sample.image.size  # PIL: (width, height)

        # The circle coordinates are already in CURRENT image space: every preceding
        # spatial transform runs transform_circle() on sample.circles. Re-scaling by
        # cur/orig here would apply the resize a second time, so use them directly.
        cx = roi_circle["center_x"]
        cy = roi_circle["center_y"]
        r = roi_circle["radius"]

        image_roi = self._make_circle_mask(cur_h, cur_w, cx, cy, r)
        sample.image = self._blank_outside(sample.image, image_roi)

        # Apply to each segmentation mask — rescale the (current-space) circle from the
        # image size to this mask's size when they differ.
        new_masks = []
        for seg_mask in sample.masks:
            mw, mh = seg_mask.size
            if mw == cur_w and mh == cur_h:
                mask_roi = image_roi
            else:
                sw = mw / cur_w if cur_w else 1.0
                sh = mh / cur_h if cur_h else 1.0
                mask_roi = self._make_circle_mask(
                    mh, mw, cx * sw, cy * sh, r * math.sqrt(sw * sh)
                )
            new_masks.append(self._blank_outside(seg_mask, mask_roi))
        sample.masks = new_masks
        return sample

    def _make_circle_mask(self, h: int, w: int, cx: float, cy: float, r: float) -> np.ndarray:
        Y, X = np.ogrid[:h, :w]
        return ((X - cx) ** 2 + (Y - cy) ** 2) <= r ** 2

    def _blank_outside(self, image: PILImage.Image, mask: np.ndarray) -> PILImage.Image:
        arr = np.array(image)
        arr[~mask] = self.fill_value
        return PILImage.fromarray(arr)

    def __repr__(self) -> str:
        return f"FundusROIMask(fill_value={self.fill_value}, fallback_full_image={self.fallback_full_image})"


class BackgroundPolynomialCorrection(BasePhotometricTransform):
    """Estimate and divide out polynomial background per channel."""

    def __init__(self, degree: int = 2):
        self.degree = degree

    def __call__(self, image: PILImage.Image) -> PILImage.Image:
        arr = _pil_to_rgb(image).astype(np.float64)
        h, w, c = arr.shape
        xs = np.arange(w, dtype=np.float64)
        result = np.empty_like(arr)
        for ch in range(c):
            col_means = arr[:, :, ch].mean(axis=0)
            coeffs = np.polyfit(xs, col_means, self.degree)
            bg = np.polyval(coeffs, xs)
            bg = np.tile(bg, (h, 1))
            bg = np.maximum(bg, 1.0)
            result[:, :, ch] = arr[:, :, ch] / bg * np.mean(bg)
        return _rgb_to_pil(result)

    def __repr__(self) -> str:
        return f"BackgroundPolynomialCorrection(degree={self.degree})"
