"""
Geometric / spatial transforms.

Each transform applies the same random parameters to the image, every mask
(with correct interpolation per unified_format), and all coordinate sets.
The actual pixel work is delegated to torchvision.transforms.functional / PIL;
only the coordinate math is done here.
"""

from __future__ import annotations

import math
import random
from typing import Any, Sequence

import numpy as np
from PIL import Image as PILImage

try:
    import torchvision.transforms.functional as F
    from torchvision.transforms.functional import InterpolationMode
except ImportError:  # pragma: no cover
    F = None  # type: ignore[assignment]
    InterpolationMode = None  # type: ignore[assignment,misc]

from chaksudb.export.transforms.base import (
    BaseSpatialTransform,
    SpatialSample,
    clip_coords,
    resize_mask,
    transform_bbox,
    transform_circle,
    transform_keypoint,
)

# ---------------------------------------------------------------------------
# Helpers local to this module
# ---------------------------------------------------------------------------

def _interp_for_mask(meta: dict[str, Any]) -> "InterpolationMode":
    fmt = meta.get("unified_format")
    if fmt == "soft_map":
        return InterpolationMode.BILINEAR
    return InterpolationMode.NEAREST


def _apply_matrix_to_coords(
    sample: SpatialSample,
    matrix: np.ndarray,
    new_w: int,
    new_h: int,
) -> SpatialSample:
    """Apply a 3×3 affine matrix to all coordinate annotations in *sample*."""
    sample.bboxes = [
        b for b in (transform_bbox(b, matrix, new_w, new_h) for b in sample.bboxes)
        if b is not None
    ]
    sample.keypoints = [
        k for k in (transform_keypoint(k, matrix, new_w, new_h) for k in sample.keypoints)
        if k is not None
    ]
    sample.circles = [
        c for c in (transform_circle(c, matrix, new_w, new_h) for c in sample.circles)
        if c is not None
    ]
    return sample


def _size_tuple(size: int | Sequence[int]) -> tuple[int, int]:
    """Normalise *size* to ``(h, w)``."""
    if isinstance(size, int):
        return (size, size)
    return (int(size[0]), int(size[1]))


# ===================================================================
# Affine geometric transforms (10)
# ===================================================================


class Resize(BaseSpatialTransform):
    """Resize image + masks + coords to a fixed size."""

    def __init__(self, size: int | Sequence[int]):
        self.size = _size_tuple(size)  # (h, w)

    def __call__(self, sample: SpatialSample) -> SpatialSample:
        old_w, old_h = sample.image.size
        new_h, new_w = self.size

        sample.image = F.resize(sample.image, [new_h, new_w], InterpolationMode.BILINEAR)
        sample.masks = [
            F.resize(m, [new_h, new_w], _interp_for_mask(meta))
            for m, meta in zip(sample.masks, sample.mask_meta)
        ]

        sx, sy = new_w / old_w, new_h / old_h
        matrix = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=np.float64)
        sample = _apply_matrix_to_coords(sample, matrix, new_w, new_h)
        return sample

    def __repr__(self) -> str:
        return f"Resize(size={self.size})"


class RandomResizedCrop(BaseSpatialTransform):
    """Crop a random portion and resize to *size*."""

    def __init__(
        self,
        size: int | Sequence[int],
        scale: tuple[float, float] = (0.08, 1.0),
        ratio: tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
    ):
        self.size = _size_tuple(size)
        self.scale = scale
        self.ratio = ratio

    def __call__(self, sample: SpatialSample) -> SpatialSample:
        import torchvision.transforms as T

        new_h, new_w = self.size
        top, left, h, w = T.RandomResizedCrop.get_params(sample.image, list(self.scale), list(self.ratio))

        sample.image = F.resized_crop(sample.image, top, left, h, w, [new_h, new_w], InterpolationMode.BILINEAR)
        sample.masks = [
            F.resized_crop(m, top, left, h, w, [new_h, new_w], _interp_for_mask(meta))
            for m, meta in zip(sample.masks, sample.mask_meta)
        ]

        sx, sy = new_w / w, new_h / h
        matrix = np.array([[sx, 0, -left * sx], [0, sy, -top * sy], [0, 0, 1]], dtype=np.float64)
        sample = _apply_matrix_to_coords(sample, matrix, new_w, new_h)
        return sample

    def __repr__(self) -> str:
        return f"RandomResizedCrop(size={self.size}, scale={self.scale}, ratio={self.ratio})"


class CenterCrop(BaseSpatialTransform):
    """Crop the centre of the image."""

    def __init__(self, size: int | Sequence[int]):
        self.size = _size_tuple(size)

    def __call__(self, sample: SpatialSample) -> SpatialSample:
        old_w, old_h = sample.image.size
        new_h, new_w = self.size
        top = (old_h - new_h) // 2
        left = (old_w - new_w) // 2

        sample.image = F.crop(sample.image, top, left, new_h, new_w)
        sample.masks = [F.crop(m, top, left, new_h, new_w) for m in sample.masks]

        matrix = np.array([[1, 0, -left], [0, 1, -top], [0, 0, 1]], dtype=np.float64)
        sample = _apply_matrix_to_coords(sample, matrix, new_w, new_h)
        return sample

    def __repr__(self) -> str:
        return f"CenterCrop(size={self.size})"


class RandomCrop(BaseSpatialTransform):
    """Crop at a random offset."""

    def __init__(self, size: int | Sequence[int]):
        self.size = _size_tuple(size)

    def __call__(self, sample: SpatialSample) -> SpatialSample:
        old_w, old_h = sample.image.size
        new_h, new_w = self.size
        top = random.randint(0, max(0, old_h - new_h))
        left = random.randint(0, max(0, old_w - new_w))

        sample.image = F.crop(sample.image, top, left, new_h, new_w)
        sample.masks = [F.crop(m, top, left, new_h, new_w) for m in sample.masks]

        matrix = np.array([[1, 0, -left], [0, 1, -top], [0, 0, 1]], dtype=np.float64)
        sample = _apply_matrix_to_coords(sample, matrix, new_w, new_h)
        return sample

    def __repr__(self) -> str:
        return f"RandomCrop(size={self.size})"


class Pad(BaseSpatialTransform):
    """Pad image + masks and shift coords accordingly."""

    def __init__(
        self,
        padding: int | Sequence[int],
        fill: int = 0,
        padding_mode: str = "constant",
    ):
        if isinstance(padding, int):
            self.padding = (padding, padding, padding, padding)
        elif len(padding) == 2:
            self.padding = (padding[0], padding[1], padding[0], padding[1])
        else:
            self.padding = tuple(padding)
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, sample: SpatialSample) -> SpatialSample:
        left, top, right, bottom = self.padding

        sample.image = F.pad(sample.image, list(self.padding), self.fill, self.padding_mode)
        sample.masks = [F.pad(m, list(self.padding), 0, self.padding_mode) for m in sample.masks]

        new_w, new_h = sample.image.size
        matrix = np.array([[1, 0, left], [0, 1, top], [0, 0, 1]], dtype=np.float64)
        sample = _apply_matrix_to_coords(sample, matrix, new_w, new_h)
        return sample

    def __repr__(self) -> str:
        return f"Pad(padding={self.padding}, fill={self.fill}, padding_mode={self.padding_mode!r})"


class RandomHorizontalFlip(BaseSpatialTransform):
    """Randomly flip horizontally with probability *p*."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, sample: SpatialSample) -> SpatialSample:
        if random.random() >= self.p:
            return sample

        w, h = sample.image.size
        sample.image = F.hflip(sample.image)
        sample.masks = [F.hflip(m) for m in sample.masks]

        matrix = np.array([[-1, 0, w], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
        sample = _apply_matrix_to_coords(sample, matrix, w, h)
        return sample

    def __repr__(self) -> str:
        return f"RandomHorizontalFlip(p={self.p})"


class RandomVerticalFlip(BaseSpatialTransform):
    """Randomly flip vertically with probability *p*."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, sample: SpatialSample) -> SpatialSample:
        if random.random() >= self.p:
            return sample

        w, h = sample.image.size
        sample.image = F.vflip(sample.image)
        sample.masks = [F.vflip(m) for m in sample.masks]

        matrix = np.array([[1, 0, 0], [0, -1, h], [0, 0, 1]], dtype=np.float64)
        sample = _apply_matrix_to_coords(sample, matrix, w, h)
        return sample

    def __repr__(self) -> str:
        return f"RandomVerticalFlip(p={self.p})"


class RandomRotation(BaseSpatialTransform):
    """Randomly rotate by an angle sampled from ``[-degrees, degrees]``."""

    def __init__(self, degrees: float, expand: bool = False):
        self.degrees = degrees
        self.expand = expand

    def __call__(self, sample: SpatialSample) -> SpatialSample:
        angle = random.uniform(-self.degrees, self.degrees)
        old_w, old_h = sample.image.size

        sample.image = F.rotate(sample.image, angle, InterpolationMode.BILINEAR, expand=self.expand)
        sample.masks = [
            F.rotate(m, angle, _interp_for_mask(meta), expand=self.expand)
            for m, meta in zip(sample.masks, sample.mask_meta)
        ]

        new_w, new_h = sample.image.size
        rad = math.radians(-angle)  # PIL rotates counter-clockwise for positive angle
        cos_a, sin_a = math.cos(rad), math.sin(rad)

        cx_old, cy_old = old_w / 2.0, old_h / 2.0
        cx_new, cy_new = new_w / 2.0, new_h / 2.0

        matrix = np.array([
            [cos_a, -sin_a, -cx_old * cos_a + cy_old * sin_a + cx_new],
            [sin_a,  cos_a, -cx_old * sin_a - cy_old * cos_a + cy_new],
            [0, 0, 1],
        ], dtype=np.float64)

        sample = _apply_matrix_to_coords(sample, matrix, new_w, new_h)
        return sample

    def __repr__(self) -> str:
        return f"RandomRotation(degrees={self.degrees}, expand={self.expand})"


class RandomAffine(BaseSpatialTransform):
    """Random affine: rotation + translate + scale + shear."""

    def __init__(
        self,
        degrees: float,
        translate: tuple[float, float] | None = None,
        scale: tuple[float, float] | None = None,
        shear: float | Sequence[float] | None = None,
    ):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear

    def __call__(self, sample: SpatialSample) -> SpatialSample:
        import torchvision.transforms as T

        w, h = sample.image.size
        angle, translations, scale, shear = T.RandomAffine.get_params(
            degrees=(-self.degrees, self.degrees),
            translate=self.translate,
            scale_ranges=self.scale,
            shears=self._parse_shear(),
            img_size=(h, w),
        )

        sample.image = F.affine(
            sample.image, angle, list(translations), scale, list(shear),
            interpolation=InterpolationMode.BILINEAR,
        )
        sample.masks = [
            F.affine(m, angle, list(translations), scale, list(shear), interpolation=_interp_for_mask(meta))
            for m, meta in zip(sample.masks, sample.mask_meta)
        ]

        rad = math.radians(-angle)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        sx_shear = math.radians(-shear[0]) if shear else 0
        sy_shear = math.radians(-shear[1]) if len(shear) > 1 else 0

        cx, cy = w / 2.0, h / 2.0
        tx, ty = translations

        rot_scale = np.array([
            [scale * cos_a, scale * (-sin_a + math.tan(sx_shear) * cos_a), 0],
            [scale * sin_a, scale * (cos_a + math.tan(sy_shear) * sin_a), 0],
            [0, 0, 1],
        ])
        t_center = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]])
        t_back = np.array([[1, 0, cx + tx], [0, 1, cy + ty], [0, 0, 1]])
        matrix = t_back @ rot_scale @ t_center

        sample = _apply_matrix_to_coords(sample, matrix, w, h)
        return sample

    def _parse_shear(self):
        if self.shear is None:
            return None
        if isinstance(self.shear, (int, float)):
            return [-self.shear, self.shear, 0, 0]
        s = list(self.shear)
        if len(s) == 2:
            return [-s[0], s[0], -s[1], s[1]]
        return s

    def __repr__(self) -> str:
        return (
            f"RandomAffine(degrees={self.degrees}, translate={self.translate}, "
            f"scale={self.scale}, shear={self.shear})"
        )


class RandomRescale(BaseSpatialTransform):
    """Resize to a random scale within *scale_range*, maintaining aspect ratio."""

    def __init__(self, scale_range: tuple[float, float] = (0.5, 2.0)):
        self.scale_range = scale_range

    def __call__(self, sample: SpatialSample) -> SpatialSample:
        s = random.uniform(*self.scale_range)
        old_w, old_h = sample.image.size
        new_w = max(1, int(round(old_w * s)))
        new_h = max(1, int(round(old_h * s)))

        sample.image = F.resize(sample.image, [new_h, new_w], InterpolationMode.BILINEAR)
        sample.masks = [
            F.resize(m, [new_h, new_w], _interp_for_mask(meta))
            for m, meta in zip(sample.masks, sample.mask_meta)
        ]

        sx, sy = new_w / old_w, new_h / old_h
        matrix = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=np.float64)
        sample = _apply_matrix_to_coords(sample, matrix, new_w, new_h)
        return sample

    def __repr__(self) -> str:
        return f"RandomRescale(scale_range={self.scale_range})"


# ===================================================================
# Non-affine geometric transforms (Phase 6)
# ===================================================================


class RandomPerspective(BaseSpatialTransform):
    """Random perspective distortion."""

    def __init__(self, distortion_scale: float = 0.5, p: float = 0.5):
        self.distortion_scale = distortion_scale
        self.p = p

    def __call__(self, sample: SpatialSample) -> SpatialSample:
        if random.random() >= self.p:
            return sample

        w, h = sample.image.size
        half_h, half_w = h // 2, w // 2
        ds = self.distortion_scale

        topleft = [
            random.randint(0, int(ds * half_w)),
            random.randint(0, int(ds * half_h)),
        ]
        topright = [
            random.randint(w - int(ds * half_w) - 1, w - 1),
            random.randint(0, int(ds * half_h)),
        ]
        botright = [
            random.randint(w - int(ds * half_w) - 1, w - 1),
            random.randint(h - int(ds * half_h) - 1, h - 1),
        ]
        botleft = [
            random.randint(0, int(ds * half_w)),
            random.randint(h - int(ds * half_h) - 1, h - 1),
        ]
        startpoints = [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]
        endpoints = [topleft, topright, botright, botleft]

        H = _find_homography(startpoints, endpoints)
        if H is None:
            return sample

        sample.image = F.perspective(sample.image, startpoints, endpoints, InterpolationMode.BILINEAR)
        sample.masks = [
            F.perspective(m, startpoints, endpoints, _interp_for_mask(meta))
            for m, meta in zip(sample.masks, sample.mask_meta)
        ]
        sample = _apply_homography_to_coords(sample, H, w, h)
        return sample

    def __repr__(self) -> str:
        return f"RandomPerspective(distortion_scale={self.distortion_scale}, p={self.p})"


class ElasticTransform(BaseSpatialTransform):
    """Elastic deformation using torchvision or cv2."""

    def __init__(self, alpha: float = 50.0, sigma: float = 5.0):
        self.alpha = alpha
        self.sigma = sigma

    def __call__(self, sample: SpatialSample) -> SpatialSample:
        import torch
        from torchvision.transforms.functional import to_pil_image, to_tensor

        w, h = sample.image.size
        dx = torch.randn(1, 1, h, w) * self.alpha
        dy = torch.randn(1, 1, h, w) * self.alpha

        ksize = max(3, int(6 * self.sigma + 1)) | 1
        from torchvision.transforms.functional import gaussian_blur
        dx = gaussian_blur(dx, [ksize, ksize], [self.sigma, self.sigma])
        dy = gaussian_blur(dy, [ksize, ksize], [self.sigma, self.sigma])

        displacement = torch.cat([dx, dy], dim=1)

        img_t = to_tensor(sample.image).unsqueeze(0)
        img_t = self._apply_displacement(img_t, displacement)
        sample.image = to_pil_image(img_t.squeeze(0))

        new_masks = []
        for m, meta in zip(sample.masks, sample.mask_meta):
            mt = to_tensor(m.convert("L")).unsqueeze(0)
            mode = "bilinear" if meta.get("unified_format") == "soft_map" else "nearest"
            mt = self._apply_displacement(mt, displacement, mode=mode)
            new_masks.append(to_pil_image(mt.squeeze(0)))
        sample.masks = new_masks

        dx_np = displacement[0, 0].numpy()
        dy_np = displacement[0, 1].numpy()
        sample = _apply_displacement_to_coords(sample, dx_np, dy_np, w, h)
        return sample

    @staticmethod
    def _apply_displacement(img: "torch.Tensor", disp: "torch.Tensor", mode: str = "bilinear") -> "torch.Tensor":
        import torch
        import torch.nn.functional as Fnn

        _, _, h, w = img.shape
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij"
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
        grid[..., 0] += disp[0, 0] * 2.0 / w
        grid[..., 1] += disp[0, 1] * 2.0 / h
        return Fnn.grid_sample(img, grid, mode=mode, padding_mode="zeros", align_corners=True)

    def __repr__(self) -> str:
        return f"ElasticTransform(alpha={self.alpha}, sigma={self.sigma})"


class PolarTransform(BaseSpatialTransform):
    """Warp to polar coordinates using cv2.warpPolar.

    Raises ValueError if localization annotations are present (coordinates
    become meaningless in polar space).
    """

    def __init__(self, center: tuple[float, float] | None = None):
        self.center = center

    def __call__(self, sample: SpatialSample) -> SpatialSample:
        if sample.bboxes or sample.keypoints or sample.circles:
            raise ValueError(
                "PolarTransform cannot be used with localization annotations "
                "(bboxes, keypoints, circles) because coordinates become "
                "meaningless in polar space."
            )

        import cv2

        img_arr = np.array(sample.image)
        h, w = img_arr.shape[:2]
        cx, cy = self.center if self.center else (w / 2.0, h / 2.0)
        max_radius = math.sqrt(cx**2 + cy**2)

        img_polar = cv2.warpPolar(img_arr, (w, h), (cx, cy), max_radius, cv2.WARP_FILL_OUTLIERS)
        sample.image = PILImage.fromarray(img_polar)

        new_masks = []
        for m, meta in zip(sample.masks, sample.mask_meta):
            if meta.get("unified_format") == "soft_map":
                m_arr = np.array(m.convert("L")).astype(np.float32)
                flags = cv2.INTER_LINEAR | cv2.WARP_FILL_OUTLIERS
            else:
                m_arr = np.array(m.convert("L"))
                flags = cv2.INTER_NEAREST | cv2.WARP_FILL_OUTLIERS
            m_polar = cv2.warpPolar(m_arr, (w, h), (cx, cy), max_radius, flags)
            new_masks.append(PILImage.fromarray(m_polar.astype(np.uint8)))
        sample.masks = new_masks

        return sample

    def __repr__(self) -> str:
        return f"PolarTransform(center={self.center})"


# ===================================================================
# Multi-output geometric transforms
# ===================================================================


class FiveCrop(BaseSpatialTransform):
    """Produce 5 crops: 4 corners + centre. Returns a list[SpatialSample]."""

    def __init__(self, size: int | Sequence[int]):
        self.size = _size_tuple(size)

    def __call__(self, sample: SpatialSample) -> list[SpatialSample]:
        h, w = self.size
        img_w, img_h = sample.image.size

        offsets = [
            (0, 0),                              # top-left
            (0, img_w - w),                      # top-right
            (img_h - h, 0),                      # bottom-left
            (img_h - h, img_w - w),              # bottom-right
            ((img_h - h) // 2, (img_w - w) // 2),  # centre
        ]

        results = []
        for top, left in offsets:
            s = SpatialSample(
                image=F.crop(sample.image, top, left, h, w),
                masks=[F.crop(m, top, left, h, w) for m in sample.masks],
                mask_meta=list(sample.mask_meta),
                bboxes=list(sample.bboxes),
                keypoints=list(sample.keypoints),
                circles=list(sample.circles),
                original_width=sample.original_width,
                original_height=sample.original_height,
                annotations=dict(sample.annotations),
            )
            matrix = np.array([[1, 0, -left], [0, 1, -top], [0, 0, 1]], dtype=np.float64)
            s = _apply_matrix_to_coords(s, matrix, w, h)
            results.append(s)
        return results

    def __repr__(self) -> str:
        return f"FiveCrop(size={self.size})"


class TenCrop(BaseSpatialTransform):
    """Produce 10 crops: FiveCrop + horizontal flips. Returns a list[SpatialSample]."""

    def __init__(self, size: int | Sequence[int]):
        self.size = _size_tuple(size)

    def __call__(self, sample: SpatialSample) -> list[SpatialSample]:
        five = FiveCrop(self.size)(sample)
        flipped = []
        for s in five:
            w_s, h_s = s.image.size
            fs = SpatialSample(
                image=F.hflip(s.image),
                masks=[F.hflip(m) for m in s.masks],
                mask_meta=list(s.mask_meta),
                bboxes=list(s.bboxes),
                keypoints=list(s.keypoints),
                circles=list(s.circles),
                original_width=s.original_width,
                original_height=s.original_height,
                annotations=dict(s.annotations),
            )
            matrix = np.array([[-1, 0, w_s], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
            fs = _apply_matrix_to_coords(fs, matrix, w_s, h_s)
            flipped.append(fs)
        return five + flipped

    def __repr__(self) -> str:
        return f"TenCrop(size={self.size})"


class CornerPatchExtraction(BaseSpatialTransform):
    """Extract patches from specified corners. Returns a list[SpatialSample]."""

    def __init__(
        self,
        size: int | Sequence[int],
        corners: Sequence[str] = ("top_left", "top_right", "bottom_left", "bottom_right"),
    ):
        self.size = _size_tuple(size)
        self.corners = list(corners)

    def __call__(self, sample: SpatialSample) -> list[SpatialSample]:
        h, w = self.size
        img_w, img_h = sample.image.size

        corner_map = {
            "top_left": (0, 0),
            "top_right": (0, img_w - w),
            "bottom_left": (img_h - h, 0),
            "bottom_right": (img_h - h, img_w - w),
            "center": ((img_h - h) // 2, (img_w - w) // 2),
        }

        results = []
        for name in self.corners:
            top, left = corner_map[name]
            s = SpatialSample(
                image=F.crop(sample.image, top, left, h, w),
                masks=[F.crop(m, top, left, h, w) for m in sample.masks],
                mask_meta=list(sample.mask_meta),
                bboxes=list(sample.bboxes),
                keypoints=list(sample.keypoints),
                circles=list(sample.circles),
                original_width=sample.original_width,
                original_height=sample.original_height,
                annotations=dict(sample.annotations),
            )
            matrix = np.array([[1, 0, -left], [0, 1, -top], [0, 0, 1]], dtype=np.float64)
            s = _apply_matrix_to_coords(s, matrix, w, h)
            results.append(s)
        return results

    def __repr__(self) -> str:
        return f"CornerPatchExtraction(size={self.size}, corners={self.corners})"


# ===================================================================
# Annotation-aware geometric transforms
# ===================================================================


class BoundingBoxCrop(BaseSpatialTransform):
    """Crop to the bounding box of a named target structure, plus padding."""

    def __init__(
        self,
        target_structure: str,
        padding: int = 0,
        fallback: str = "full_image",
    ):
        self.target_structure = target_structure
        self.padding = padding
        self.fallback = fallback

    def __call__(self, sample: SpatialSample) -> SpatialSample:
        w, h = sample.image.size
        bbox = self._find_bbox(sample)

        if bbox is None:
            if self.fallback == "full_image":
                return sample
            raise ValueError(
                f"BoundingBoxCrop: target_structure={self.target_structure!r} not found "
                f"and fallback={self.fallback!r}"
            )

        p = self.padding
        left = max(0, int(bbox["xmin"]) - p)
        top = max(0, int(bbox["ymin"]) - p)
        right = min(w, int(bbox["xmax"]) + p)
        bottom = min(h, int(bbox["ymax"]) + p)
        crop_w, crop_h = right - left, bottom - top

        sample.image = F.crop(sample.image, top, left, crop_h, crop_w)
        sample.masks = [F.crop(m, top, left, crop_h, crop_w) for m in sample.masks]

        matrix = np.array([[1, 0, -left], [0, 1, -top], [0, 0, 1]], dtype=np.float64)
        sample = _apply_matrix_to_coords(sample, matrix, crop_w, crop_h)
        return sample

    def _find_bbox(self, sample: SpatialSample) -> dict[str, Any] | None:
        for b in sample.bboxes:
            if b.get("target_structure") == self.target_structure:
                return b
        for c in sample.circles:
            if c.get("target_structure") == self.target_structure:
                return {
                    "xmin": c.get("xmin", c["center_x"] - c.get("radius", 0)),
                    "ymin": c.get("ymin", c["center_y"] - c.get("radius", 0)),
                    "xmax": c.get("xmax", c["center_x"] + c.get("radius", 0)),
                    "ymax": c.get("ymax", c["center_y"] + c.get("radius", 0)),
                }
        return None

    def __repr__(self) -> str:
        return (
            f"BoundingBoxCrop(target_structure={self.target_structure!r}, "
            f"padding={self.padding}, fallback={self.fallback!r})"
        )


class ROICrop(BaseSpatialTransform):
    """Crop a square region around a keypoint or circle centre."""

    def __init__(
        self,
        target_structure: str,
        padding: int = 64,
        fallback: str = "full_image",
    ):
        self.target_structure = target_structure
        self.padding = padding
        self.fallback = fallback

    def __call__(self, sample: SpatialSample) -> SpatialSample:
        w, h = sample.image.size
        center = self._find_center(sample)

        if center is None:
            if self.fallback == "full_image":
                return sample
            raise ValueError(
                f"ROICrop: target_structure={self.target_structure!r} not found "
                f"and fallback={self.fallback!r}"
            )

        cx, cy = center
        p = self.padding
        left = max(0, int(cx) - p)
        top = max(0, int(cy) - p)
        right = min(w, int(cx) + p)
        bottom = min(h, int(cy) + p)
        crop_w, crop_h = right - left, bottom - top

        sample.image = F.crop(sample.image, top, left, crop_h, crop_w)
        sample.masks = [F.crop(m, top, left, crop_h, crop_w) for m in sample.masks]

        matrix = np.array([[1, 0, -left], [0, 1, -top], [0, 0, 1]], dtype=np.float64)
        sample = _apply_matrix_to_coords(sample, matrix, crop_w, crop_h)
        return sample

    def _find_center(self, sample: SpatialSample) -> tuple[float, float] | None:
        for kp in sample.keypoints:
            if kp.get("target_structure") == self.target_structure:
                return (kp["x"], kp["y"])
        for c in sample.circles:
            if c.get("target_structure") == self.target_structure:
                return (c["center_x"], c["center_y"])
        return None

    def __repr__(self) -> str:
        return (
            f"ROICrop(target_structure={self.target_structure!r}, "
            f"padding={self.padding}, fallback={self.fallback!r})"
        )


# ===================================================================
# Internal helpers for non-affine transforms
# ===================================================================

def _find_homography(
    src: list[list[int]],
    dst: list[list[int]],
) -> np.ndarray | None:
    """Compute 3×3 homography from 4 point correspondences (src → dst)."""
    try:
        import cv2
        src_pts = np.array(src, dtype=np.float32)
        dst_pts = np.array(dst, dtype=np.float32)
        H, _ = cv2.findHomography(src_pts, dst_pts)
        return H
    except Exception:
        return None


def _apply_homography_to_coords(
    sample: SpatialSample,
    H: np.ndarray,
    w: int,
    h: int,
) -> SpatialSample:
    """Apply a 3×3 homography to all coords via perspective division."""
    def _warp_pt(x: float, y: float) -> tuple[float, float]:
        v = H @ np.array([x, y, 1.0])
        return float(v[0] / v[2]), float(v[1] / v[2])

    new_bboxes: list[dict[str, Any]] = []
    for b in sample.bboxes:
        corners = [
            _warp_pt(b["xmin"], b["ymin"]),
            _warp_pt(b["xmax"], b["ymin"]),
            _warp_pt(b["xmax"], b["ymax"]),
            _warp_pt(b["xmin"], b["ymax"]),
        ]
        xs = [p[0] for p in corners]
        ys = [p[1] for p in corners]
        xmin = max(0.0, min(xs))
        ymin = max(0.0, min(ys))
        xmax = min(float(w), max(xs))
        ymax = min(float(h), max(ys))
        if xmin < xmax and ymin < ymax:
            nb = dict(b)
            nb.update(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax,
                      width=xmax - xmin, height=ymax - ymin,
                      center_x=(xmin + xmax) / 2, center_y=(ymin + ymax) / 2)
            new_bboxes.append(nb)
    sample.bboxes = new_bboxes

    new_kps: list[dict[str, Any]] = []
    for kp in sample.keypoints:
        nx, ny = _warp_pt(kp["x"], kp["y"])
        if 0 <= nx < w and 0 <= ny < h:
            nkp = dict(kp)
            nkp["x"] = nx
            nkp["y"] = ny
            new_kps.append(nkp)
    sample.keypoints = new_kps

    new_circles: list[dict[str, Any]] = []
    for c in sample.circles:
        ncx, ncy = _warp_pt(c["center_x"], c["center_y"])
        det = np.linalg.det(H[:2, :2])
        scale = math.sqrt(abs(det)) if det != 0 else 1.0
        nr = c.get("radius", 0.0) * scale
        if ncx + nr >= 0 and ncx - nr < w and ncy + nr >= 0 and ncy - nr < h:
            nc = dict(c)
            nc.update(center_x=ncx, center_y=ncy, radius=nr,
                      xmin=max(0.0, ncx - nr), ymin=max(0.0, ncy - nr),
                      xmax=min(float(w), ncx + nr), ymax=min(float(h), ncy + nr))
            new_circles.append(nc)
    sample.circles = new_circles
    return sample


def _apply_displacement_to_coords(
    sample: SpatialSample,
    dx: np.ndarray,
    dy: np.ndarray,
    w: int,
    h: int,
) -> SpatialSample:
    """Approximate coordinate displacement for elastic transforms."""
    def _lookup(x: float, y: float) -> tuple[float, float]:
        ix, iy = min(int(x), w - 1), min(int(y), h - 1)
        ix, iy = max(0, ix), max(0, iy)
        return x + float(dx[iy, ix]), y + float(dy[iy, ix])

    new_bboxes: list[dict[str, Any]] = []
    for b in sample.bboxes:
        corners = [
            _lookup(b["xmin"], b["ymin"]),
            _lookup(b["xmax"], b["ymin"]),
            _lookup(b["xmax"], b["ymax"]),
            _lookup(b["xmin"], b["ymax"]),
        ]
        xs = [p[0] for p in corners]
        ys = [p[1] for p in corners]
        xmin = max(0.0, min(xs))
        ymin = max(0.0, min(ys))
        xmax = min(float(w), max(xs))
        ymax = min(float(h), max(ys))
        if xmin < xmax and ymin < ymax:
            nb = dict(b)
            nb.update(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax,
                      width=xmax - xmin, height=ymax - ymin,
                      center_x=(xmin + xmax) / 2, center_y=(ymin + ymax) / 2)
            new_bboxes.append(nb)
    sample.bboxes = new_bboxes

    new_kps: list[dict[str, Any]] = []
    for kp in sample.keypoints:
        nx, ny = _lookup(kp["x"], kp["y"])
        if 0 <= nx < w and 0 <= ny < h:
            nkp = dict(kp)
            nkp["x"] = nx
            nkp["y"] = ny
            new_kps.append(nkp)
    sample.keypoints = new_kps

    new_circles: list[dict[str, Any]] = []
    for c in sample.circles:
        ncx, ncy = _lookup(c["center_x"], c["center_y"])
        nr = c.get("radius", 0.0)
        if ncx + nr >= 0 and ncx - nr < w and ncy + nr >= 0 and ncy - nr < h:
            nc = dict(c)
            nc.update(center_x=ncx, center_y=ncy,
                      xmin=max(0.0, ncx - nr), ymin=max(0.0, ncy - nr),
                      xmax=min(float(w), ncx + nr), ymax=min(float(h), ncy + nr))
            new_circles.append(nc)
    sample.circles = new_circles
    return sample
