"""
Morphological transforms (spatial-layer, mask-aware).

These go in ``spatial=[...]`` because they need access to the ``SpatialSample``
to target masks by type.  Each accepts ``apply_to`` and optional ``mask_types``
filter.
"""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np
from PIL import Image as PILImage

from chaksudb.export.transforms.base import BaseMorphologicalTransform, SpatialSample


def _pil_to_uint8(image: PILImage.Image) -> np.ndarray:
    return np.array(image)


def _uint8_to_pil(arr: np.ndarray, mode: str = "L") -> PILImage.Image:
    return PILImage.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode)


_VALID_APPLY_TO = {"image", "masks", "both"}


def _apply_morph_op(
    sample: SpatialSample,
    op_fn,
    kernel: np.ndarray,
    apply_to: str,
    mask_types: Sequence[str] | None,
) -> SpatialSample:
    """Shared dispatcher: apply *op_fn* to image, masks, or both."""
    if apply_to not in _VALID_APPLY_TO:
        raise ValueError(
            f"apply_to must be one of {sorted(_VALID_APPLY_TO)}, got {apply_to!r}"
        )

    if apply_to in ("image", "both"):
        original_mode = sample.image.mode
        arr = _pil_to_uint8(sample.image)
        arr = op_fn(arr, kernel)
        sample.image = PILImage.fromarray(
            np.clip(arr, 0, 255).astype(np.uint8), original_mode
        )

    if apply_to in ("masks", "both"):
        new_masks = []
        for m, meta in zip(sample.masks, sample.mask_meta):
            if mask_types and meta.get("unified_format") not in mask_types:
                new_masks.append(m)
                continue
            m_arr = _pil_to_uint8(m.convert("L"))
            m_arr = op_fn(m_arr, kernel)
            new_masks.append(_uint8_to_pil(m_arr))
        sample.masks = new_masks

    return sample


class Erosion(BaseMorphologicalTransform):
    """cv2.erode on selected targets."""

    def __init__(
        self,
        kernel_size: int = 3,
        apply_to: str = "masks",
        mask_types: Sequence[str] | None = None,
    ):
        self.kernel_size = kernel_size
        self.apply_to = apply_to
        self.mask_types = list(mask_types) if mask_types else None
        self._kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    def __call__(self, sample: SpatialSample) -> SpatialSample:
        return _apply_morph_op(sample, cv2.erode, self._kernel, self.apply_to, self.mask_types)

    def __repr__(self) -> str:
        return f"Erosion(kernel_size={self.kernel_size}, apply_to={self.apply_to!r})"


class Dilation(BaseMorphologicalTransform):
    """cv2.dilate on selected targets."""

    def __init__(
        self,
        kernel_size: int = 3,
        apply_to: str = "masks",
        mask_types: Sequence[str] | None = None,
    ):
        self.kernel_size = kernel_size
        self.apply_to = apply_to
        self.mask_types = list(mask_types) if mask_types else None
        self._kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    def __call__(self, sample: SpatialSample) -> SpatialSample:
        return _apply_morph_op(sample, cv2.dilate, self._kernel, self.apply_to, self.mask_types)

    def __repr__(self) -> str:
        return f"Dilation(kernel_size={self.kernel_size}, apply_to={self.apply_to!r})"


class Opening(BaseMorphologicalTransform):
    """cv2.morphologyEx(MORPH_OPEN) on selected targets."""

    def __init__(
        self,
        kernel_size: int = 3,
        apply_to: str = "masks",
        mask_types: Sequence[str] | None = None,
    ):
        self.kernel_size = kernel_size
        self.apply_to = apply_to
        self.mask_types = list(mask_types) if mask_types else None
        self._kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    def __call__(self, sample: SpatialSample) -> SpatialSample:
        def _open(arr, kern):
            return cv2.morphologyEx(arr, cv2.MORPH_OPEN, kern)
        return _apply_morph_op(sample, _open, self._kernel, self.apply_to, self.mask_types)

    def __repr__(self) -> str:
        return f"Opening(kernel_size={self.kernel_size}, apply_to={self.apply_to!r})"


class MorphologicalClosing(BaseMorphologicalTransform):
    """cv2.morphologyEx(MORPH_CLOSE) on selected targets."""

    def __init__(
        self,
        kernel_size: int = 3,
        apply_to: str = "masks",
        mask_types: Sequence[str] | None = None,
    ):
        self.kernel_size = kernel_size
        self.apply_to = apply_to
        self.mask_types = list(mask_types) if mask_types else None
        self._kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    def __call__(self, sample: SpatialSample) -> SpatialSample:
        def _close(arr, kern):
            return cv2.morphologyEx(arr, cv2.MORPH_CLOSE, kern)
        return _apply_morph_op(sample, _close, self._kernel, self.apply_to, self.mask_types)

    def __repr__(self) -> str:
        return f"MorphologicalClosing(kernel_size={self.kernel_size}, apply_to={self.apply_to!r})"


class ConnectedComponentFiltering(BaseMorphologicalTransform):
    """Remove connected components smaller than *min_size* pixels."""

    def __init__(
        self,
        min_size: int = 100,
        apply_to: str = "masks",
        mask_types: Sequence[str] | None = None,
    ):
        self.min_size = min_size
        self.apply_to = apply_to
        self.mask_types = list(mask_types) if mask_types else None

    def __call__(self, sample: SpatialSample) -> SpatialSample:
        def _filter(arr, _kern):
            if arr.ndim == 3:
                gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            else:
                gray = arr
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
            for label_id in range(1, num_labels):
                area = stats[label_id, cv2.CC_STAT_AREA]
                if area < self.min_size:
                    if arr.ndim == 3:
                        arr[labels == label_id] = 0
                    else:
                        arr[labels == label_id] = 0
            return arr

        return _apply_morph_op(sample, _filter, None, self.apply_to, self.mask_types)

    def __repr__(self) -> str:
        return f"ConnectedComponentFiltering(min_size={self.min_size}, apply_to={self.apply_to!r})"
