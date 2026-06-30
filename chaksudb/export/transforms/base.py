"""
Base types and helpers for the spatial transform pipeline.

SpatialSample bundles image + masks + coordinates so every geometric transform
can operate on them together.  Helper functions handle the coordinate math that
all affine / non-affine transforms share.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image as PILImage


# ---------------------------------------------------------------------------
# SpatialSample
# ---------------------------------------------------------------------------

@dataclass
class SpatialSample:
    """Container that bundles all spatial data for a single sample."""

    image: PILImage.Image
    masks: list[PILImage.Image] = field(default_factory=list)
    mask_meta: list[dict[str, Any]] = field(default_factory=list)
    bboxes: list[dict[str, Any]] = field(default_factory=list)
    keypoints: list[dict[str, Any]] = field(default_factory=list)
    circles: list[dict[str, Any]] = field(default_factory=list)
    original_width: int = 0
    original_height: int = 0
    annotations: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Marker base classes
# ---------------------------------------------------------------------------

class BaseSpatialTransform:
    """Marker base.  Subclasses implement __call__(SpatialSample) -> SpatialSample."""

    def __call__(self, sample: SpatialSample) -> SpatialSample:
        raise NotImplementedError


class BasePhotometricTransform:
    """Marker base.  Subclasses implement __call__(PIL.Image) -> PIL.Image."""

    def __call__(self, image: PILImage.Image) -> PILImage.Image:
        raise NotImplementedError


class BaseMorphologicalTransform(BaseSpatialTransform):
    """Marker for morphological transforms (spatial-layer, mask-aware)."""
    pass


# ---------------------------------------------------------------------------
# Helper: mask interpolation dispatch
# ---------------------------------------------------------------------------

def resize_mask(
    mask: PILImage.Image,
    size: tuple[int, int],
    unified_format: str | None,
) -> PILImage.Image:
    """Resize a mask using interpolation appropriate for its format.

    Args:
        mask: PIL mask image.
        size: Target (width, height).
        unified_format: ``"binary_mask"`` → NEAREST, ``"soft_map"`` → BILINEAR,
            anything else defaults to NEAREST.
    """
    if unified_format == "soft_map":
        return mask.resize(size, PILImage.BILINEAR)
    return mask.resize(size, PILImage.NEAREST)


# ---------------------------------------------------------------------------
# Helper: affine coordinate transforms
# ---------------------------------------------------------------------------

def _apply_affine_point(x: float, y: float, matrix: np.ndarray) -> tuple[float, float]:
    """Apply a 2×3 or 3×3 affine *matrix* to a single (x, y) point."""
    m = np.asarray(matrix, dtype=np.float64)
    pt = np.array([x, y, 1.0])
    out = m[:2] @ pt if m.shape[0] >= 2 else m @ pt
    return float(out[0]), float(out[1])


def transform_bbox(
    bbox: dict[str, Any],
    matrix: np.ndarray,
    image_width: int,
    image_height: int,
) -> dict[str, Any] | None:
    """Transform a bbox dict through an affine matrix, returning a new dict or None if fully OOB."""
    corners = [
        (bbox["xmin"], bbox["ymin"]),
        (bbox["xmax"], bbox["ymin"]),
        (bbox["xmax"], bbox["ymax"]),
        (bbox["xmin"], bbox["ymax"]),
    ]
    tx = [_apply_affine_point(cx, cy, matrix) for cx, cy in corners]
    xs = [p[0] for p in tx]
    ys = [p[1] for p in tx]

    xmin = max(0.0, min(xs))
    ymin = max(0.0, min(ys))
    xmax = min(float(image_width), max(xs))
    ymax = min(float(image_height), max(ys))

    if xmin >= xmax or ymin >= ymax:
        return None

    out = dict(bbox)
    out["xmin"] = xmin
    out["ymin"] = ymin
    out["xmax"] = xmax
    out["ymax"] = ymax
    out["width"] = xmax - xmin
    out["height"] = ymax - ymin
    out["center_x"] = (xmin + xmax) / 2
    out["center_y"] = (ymin + ymax) / 2
    return out


def transform_keypoint(
    kp: dict[str, Any],
    matrix: np.ndarray,
    image_width: int,
    image_height: int,
) -> dict[str, Any] | None:
    """Transform a keypoint dict; returns None if the result is out of bounds."""
    nx, ny = _apply_affine_point(kp["x"], kp["y"], matrix)
    if nx < 0 or nx >= image_width or ny < 0 or ny >= image_height:
        return None
    out = dict(kp)
    out["x"] = nx
    out["y"] = ny
    return out


def transform_circle(
    circle: dict[str, Any],
    matrix: np.ndarray,
    image_width: int,
    image_height: int,
) -> dict[str, Any] | None:
    """Transform a circle dict; scales radius by sqrt(det(M[0:2,0:2]))."""
    m = np.asarray(matrix, dtype=np.float64)
    cx, cy = _apply_affine_point(circle["center_x"], circle["center_y"], m)
    det = abs(m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0])
    scale = math.sqrt(det) if det > 0 else 1.0
    r = circle.get("radius", 0.0) * scale

    if cx + r < 0 or cx - r >= image_width or cy + r < 0 or cy - r >= image_height:
        return None

    out = dict(circle)
    out["center_x"] = cx
    out["center_y"] = cy
    out["radius"] = r
    out["xmin"] = max(0.0, cx - r)
    out["ymin"] = max(0.0, cy - r)
    out["xmax"] = min(float(image_width), cx + r)
    out["ymax"] = min(float(image_height), cy + r)
    return out


def clip_coords(
    sample: SpatialSample,
    width: int,
    height: int,
) -> SpatialSample:
    """Clip all coordinate annotations to (0, 0, width, height), dropping OOB items."""
    identity = np.eye(3)

    sample.bboxes = [
        b for b in (transform_bbox(b, identity, width, height) for b in sample.bboxes) if b is not None
    ]
    sample.keypoints = [
        k for k in (transform_keypoint(k, identity, width, height) for k in sample.keypoints) if k is not None
    ]
    clipped_circles = []
    for c in sample.circles:
        tc = transform_circle(c, identity, width, height)
        if tc is not None and 0 <= tc["center_x"] < width and 0 <= tc["center_y"] < height:
            clipped_circles.append(tc)
    sample.circles = clipped_circles
    return sample
