"""
Collate strategies for batching SpatialSample-based dataset output.

Three strategies:
  - ``"default"``   - images stacked, everything else as lists of lists
  - ``"padded"``    - masks / coords zero-padded to max count with valid-count tensors
  - ``"packed"``    - masks / coords concatenated with sample-index column

All assume images are already the same spatial size (the spatial pipeline resized them).
"""

from __future__ import annotations

from typing import Any, Callable, Union

import torch
from PIL import Image as PILImage


def _pil_to_tensor(img: PILImage.Image) -> torch.Tensor:
    """(C, H, W) float32 in [0, 1]."""
    try:
        import torchvision.transforms as T
        return T.ToTensor()(img)
    except (ImportError, RuntimeError):
        pass
    import numpy as np
    arr = np.array(img, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = arr[None, ...]
    else:
        arr = arr.transpose((2, 0, 1))
    return torch.from_numpy(arr.copy())


def _to_tensor(img: Union[PILImage.Image, torch.Tensor]) -> torch.Tensor:
    if isinstance(img, torch.Tensor):
        if img.dtype == torch.uint8:
            return img.to(torch.float32) / 255.0
        return img.to(torch.float32)
    return _pil_to_tensor(img)


def _mask_to_tensor(
    mask: PILImage.Image, unified_format: str | None = None
) -> torch.Tensor:
    """Mask -> float32 CHW tensor in [0, 1].

    Color (AV) masks keep their 3 RGB channels (R=arteries, G=overlap, B=veins);
    everything else collapses to a single grayscale channel. Forcing "L" here would
    destroy the artery/vein/overlap encoding, so we branch on the mask mode / the
    stored unified_format.
    """
    import numpy as np
    if mask.mode == "RGB" or unified_format == "color_mask":
        arr = np.asarray(mask.convert("RGB"), dtype=np.float32) / 255.0  # (H, W, 3)
        return torch.from_numpy(arr.transpose(2, 0, 1).copy())           # (3, H, W)
    arr = np.array(mask.convert("L"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)                            # (1, H, W)


# ===================================================================
# Strategy: default
# ===================================================================

def default_collate(
    batch: list[tuple[Union[PILImage.Image, torch.Tensor], dict[str, Any]]],
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Stack images; keep everything else as lists.

    Returns ``(images, batched_annotations)`` where ``images`` is
    ``(B, C, H, W)`` and each annotation key maps to a Python list of
    per-sample values.

    Variable-size images are zero-padded to the batch maximum H and W.
    When padding occurs, ``_original_height`` and ``_original_width`` lists
    are added to the batched annotations.
    """
    if not batch:
        raise ValueError("default_collate does not support empty batches")

    tensors = [_to_tensor(item[0]) for item in batch]
    max_h = max(t.shape[-2] for t in tensors)
    max_w = max(t.shape[-1] for t in tensors)
    if any(t.shape[-2] != max_h or t.shape[-1] != max_w for t in tensors):
        padded = torch.zeros(len(tensors), tensors[0].shape[0], max_h, max_w)
        for i, t in enumerate(tensors):
            padded[i, :, : t.shape[-2], : t.shape[-1]] = t
        images = padded
        pad_metadata = {
            "_original_height": [t.shape[-2] for t in tensors],
            "_original_width": [t.shape[-1] for t in tensors],
        }
    else:
        images = torch.stack(tensors)
        pad_metadata = {}

    ann_list = [item[1] for item in batch]

    all_keys: set[str] = set()
    for a in ann_list:
        all_keys.update(a.keys())

    batched: dict[str, Any] = {**pad_metadata}
    for key in sorted(all_keys):
        values = [a.get(key) for a in ann_list]
        if all(isinstance(v, (int, float)) for v in values if v is not None):
            try:
                batched[key] = torch.tensor(values)
                continue
            except Exception:
                pass
        batched[key] = values

    return images, batched


# ===================================================================
# Strategy: padded
# ===================================================================

def padded_collate(
    batch: list[tuple[Union[PILImage.Image, torch.Tensor], dict[str, Any]]],
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Pad masks / bboxes / keypoints to max count, with count tensors."""
    if not batch:
        raise ValueError("padded_collate does not support empty batches")

    B = len(batch)
    images = torch.stack([_to_tensor(item[0]) for item in batch])
    ann_list = [item[1] for item in batch]

    batched: dict[str, Any] = {}

    # --- masks ---
    mask_lists: list[list[PILImage.Image]] = [
        a.get("_loaded_masks", []) for a in ann_list
    ]
    meta_lists: list[list[dict]] = [a.get("_mask_meta", []) for a in ann_list]
    max_masks = max((len(ml) for ml in mask_lists), default=0)
    if max_masks > 0:
        _, _, H, W = images.shape
        # Convert first so we know the channel count (AV color masks are 3-channel).
        tensor_lists = [
            [
                _mask_to_tensor(
                    m,
                    (meta_lists[i][j].get("unified_format") if j < len(meta_lists[i]) else None),
                )
                for j, m in enumerate(ml)
            ]
            for i, ml in enumerate(mask_lists)
        ]
        max_c = max((t.shape[0] for tl in tensor_lists for t in tl), default=1)
        masks_tensor = torch.zeros(B, max_masks, max_c, H, W)
        mask_counts = torch.zeros(B, dtype=torch.long)
        for i, tl in enumerate(tensor_lists):
            mask_counts[i] = len(tl)
            for j, t in enumerate(tl):
                masks_tensor[i, j, : t.shape[0]] = t
        batched["masks"] = masks_tensor
        batched["mask_counts"] = mask_counts

    # --- bboxes ---
    bbox_lists: list[list[dict]] = [a.get("_bboxes", []) for a in ann_list]
    max_boxes = max((len(bl) for bl in bbox_lists), default=0)
    if max_boxes > 0:
        bboxes_tensor = torch.zeros(B, max_boxes, 4)
        bbox_valid = torch.zeros(B, max_boxes, dtype=torch.bool)
        for i, bl in enumerate(bbox_lists):
            for j, b in enumerate(bl):
                bboxes_tensor[i, j] = torch.tensor(
                    [b["xmin"], b["ymin"], b["xmax"], b["ymax"]], dtype=torch.float32
                )
                bbox_valid[i, j] = True
        batched["bboxes"] = bboxes_tensor
        batched["bbox_valid"] = bbox_valid

    # --- keypoints ---
    kp_lists: list[list[dict]] = [a.get("_keypoints", []) for a in ann_list]
    max_kp = max((len(kl) for kl in kp_lists), default=0)
    if max_kp > 0:
        kp_tensor = torch.zeros(B, max_kp, 2)
        kp_counts = torch.zeros(B, dtype=torch.long)
        for i, kl in enumerate(kp_lists):
            kp_counts[i] = len(kl)
            for j, kp in enumerate(kl):
                kp_tensor[i, j] = torch.tensor([kp["x"], kp["y"]], dtype=torch.float32)
        batched["keypoints"] = kp_tensor
        batched["keypoint_counts"] = kp_counts

    # --- scalars / other ---
    skip_keys = {"_loaded_masks", "_mask_meta", "_bboxes", "_keypoints", "_circles"}
    all_keys = set()
    for a in ann_list:
        all_keys.update(a.keys())
    for key in sorted(all_keys - skip_keys):
        values = [a.get(key) for a in ann_list]
        if all(isinstance(v, (int, float)) for v in values if v is not None):
            try:
                batched[key] = torch.tensor(values)
                continue
            except Exception:
                pass
        batched[key] = values

    return images, batched


# ===================================================================
# Strategy: packed
# ===================================================================

def packed_collate(
    batch: list[tuple[Union[PILImage.Image, torch.Tensor], dict[str, Any]]],
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Concatenate masks / coords with a sample-index column."""
    if not batch:
        raise ValueError("packed_collate does not support empty batches")

    images = torch.stack([_to_tensor(item[0]) for item in batch])
    ann_list = [item[1] for item in batch]
    _, _, H, W = images.shape
    batched: dict[str, Any] = {}

    # --- masks ---
    all_masks: list[torch.Tensor] = []
    mask_idx: list[int] = []
    for i, a in enumerate(ann_list):
        meta = a.get("_mask_meta", [])
        for j, m in enumerate(a.get("_loaded_masks", [])):
            uf = meta[j].get("unified_format") if j < len(meta) else None
            all_masks.append(_mask_to_tensor(m, uf))
            mask_idx.append(i)
    if all_masks:
        batched["masks"] = torch.stack(all_masks)  # (total, C, H, W)
        batched["mask_sample_idx"] = torch.tensor(mask_idx, dtype=torch.long)

    # --- bboxes: (total, 5) = [sample_idx, xmin, ymin, xmax, ymax] ---
    all_bboxes: list[list[float]] = []
    for i, a in enumerate(ann_list):
        for b in a.get("_bboxes", []):
            all_bboxes.append([float(i), b["xmin"], b["ymin"], b["xmax"], b["ymax"]])
    if all_bboxes:
        batched["bboxes"] = torch.tensor(all_bboxes, dtype=torch.float32)

    # --- keypoints: (total, 3) = [sample_idx, x, y] ---
    all_kps: list[list[float]] = []
    for i, a in enumerate(ann_list):
        for kp in a.get("_keypoints", []):
            all_kps.append([float(i), kp["x"], kp["y"]])
    if all_kps:
        batched["keypoints"] = torch.tensor(all_kps, dtype=torch.float32)

    # --- scalars / other ---
    skip_keys = {"_loaded_masks", "_mask_meta", "_bboxes", "_keypoints", "_circles"}
    all_keys = set()
    for a in ann_list:
        all_keys.update(a.keys())
    for key in sorted(all_keys - skip_keys):
        values = [a.get(key) for a in ann_list]
        if all(isinstance(v, (int, float)) for v in values if v is not None):
            try:
                batched[key] = torch.tensor(values)
                continue
            except Exception:
                pass
        batched[key] = values

    return images, batched


# ===================================================================
# Resolver: string name → callable
# ===================================================================

COLLATE_STRATEGIES: dict[str, Callable] = {
    "default": default_collate,
    "padded": padded_collate,
    "packed": packed_collate,
}


def get_collate_fn(
    name_or_fn: str | Callable = "default",
) -> Callable:
    """Return a collate callable from a strategy name or pass through a custom callable."""
    if callable(name_or_fn):
        return name_or_fn
    if name_or_fn not in COLLATE_STRATEGIES:
        raise ValueError(
            f"Unknown collate strategy {name_or_fn!r}. "
            f"Choose from {list(COLLATE_STRATEGIES)} or pass a callable."
        )
    return COLLATE_STRATEGIES[name_or_fn]
