"""Regression tests for the segmentation collate / mask contract.

Covers the bugs that made trained models receive wrong or missing targets:
  - masks dropped on the output_format="segmentation" path
  - AV color masks collapsed to grayscale
  - deterministic channel ordering
"""

import numpy as np
import torch
from PIL import Image as PILImage

from chaksudb.export.torch_dataset import _fmt_segmentation
from chaksudb.export.transforms.collate import (
    _mask_to_tensor,
    packed_collate,
    padded_collate,
)


def _gray(value: int = 255, size=(8, 8)) -> PILImage.Image:
    return PILImage.fromarray(np.full(size, value, dtype=np.uint8), "L")


def _rgb(size=(8, 8)) -> PILImage.Image:
    arr = np.zeros((*size, 3), dtype=np.uint8)
    arr[..., 0] = 255  # pure "artery" channel
    return PILImage.fromarray(arr, "RGB")


def test_color_mask_keeps_three_channels():
    t = _mask_to_tensor(_rgb(), unified_format="color_mask")
    assert t.shape[0] == 3
    # artery (R) channel preserved, vein (B) channel empty
    assert t[0].max() == 1.0 and t[2].max() == 0.0


def test_binary_mask_is_single_channel():
    assert _mask_to_tensor(_gray()).shape[0] == 1


def test_fmt_segmentation_keeps_collate_contract():
    """output_format='segmentation' must still expose _loaded_masks for collate."""
    img = _rgb()
    row = {
        "_loaded_masks": [_gray(), _gray()],
        "_mask_meta": [
            {"target_structure": "optic_disc", "unified_format": "binary_mask"},
            {"target_structure": "optic_cup", "unified_format": "binary_mask"},
        ],
    }
    _, ann = _fmt_segmentation(img, row)
    assert "_loaded_masks" in ann and len(ann["_loaded_masks"]) == 2
    assert ann["_structure_index"] == {"optic_disc": 0, "optic_cup": 1}
    # batching this produces a real masks tensor (previously absent -> trained on nothing)
    _, batched = padded_collate([(img, ann)])
    assert "masks" in batched
    assert batched["masks"].shape[:2] == (1, 2)


def test_padded_collate_handles_color_masks():
    img = _rgb()
    ann = {
        "_loaded_masks": [_rgb()],
        "_mask_meta": [{"unified_format": "color_mask", "target_structure": "av"}],
    }
    _, batched = padded_collate([(img, ann)])
    # channel dim must accommodate the 3-channel AV mask
    assert batched["masks"].shape[2] == 3


def test_packed_collate_includes_masks():
    img = _gray()
    ann = {
        "_loaded_masks": [_gray()],
        "_mask_meta": [{"unified_format": "binary_mask", "target_structure": "vessels"}],
    }
    _, batched = packed_collate([(img, ann)])
    assert "masks" in batched and batched["masks"].shape[0] == 1
